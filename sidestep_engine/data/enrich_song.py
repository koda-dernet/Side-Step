"""
Per-song enrichment pipeline for the AI dataset builder.

Orchestrates lyrics fetching, sanitization, and caption generation
for a single audio file.  Called in a loop by the wizard orchestrator.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Optional

from sidestep_engine.data.audio_metadata import resolve_metadata
from sidestep_engine.data.caption_config import parse_structured_response
from sidestep_engine.data.structured_helpers import (
    extract_caption_from_blob,
    looks_like_mapping_blob,
)

logger = logging.getLogger(__name__)


def _normalize_generated_fields(fields: Dict[str, Any]) -> Dict[str, str]:
    normalized: Dict[str, str] = {}
    for key, value in (fields or {}).items():
        if value is None:
            continue
        if key == "caption":
            clean_caption = (
                extract_caption_from_blob(value)
                if looks_like_mapping_blob(value)
                else str(value).strip()
            )
            if clean_caption and not looks_like_mapping_blob(clean_caption):
                normalized[key] = clean_caption
            continue
        if key == "genre" and isinstance(value, (list, tuple, set)):
            joined = ", ".join(str(v).strip() for v in value if str(v).strip())
            if joined:
                normalized[key] = joined
            continue
        normalized[key] = (
            str(value).strip()
            if not isinstance(value, bool)
            else ("true" if value else "false")
        )
    return {k: v for k, v in normalized.items() if str(v).strip()}

def enrich_one(
    audio_path: Path,
    *,
    default_artist: str = "",
    caption_fn: Optional[Any] = None,
    lyrics_fn: Optional[Any] = None,
    metadata_fn: Optional[Any] = None,
    audio_analyze_fn: Optional[Any] = None,
    policy: str = "fill_missing",
) -> Dict[str, str]:
    """Run the enrichment pipeline for a single song.

    Reads existing sidecar, fetches lyrics, generates caption, merges
    results according to the overwrite policy, and writes the sidecar.

    Args:
        audio_path: Path to the audio file.
        default_artist: Folder-level default artist for Genius lookups.
        caption_fn: ``(title, artist, lyrics_excerpt, audio_path) -> str|None``.
        lyrics_fn: ``(artist, title) -> str|None``.
        metadata_fn: ``(audio_path) -> dict|None``. Returns sidecar fields
            such as ``caption``, ``genre``, ``bpm``, ``key``, ``signature``.
        audio_analyze_fn: ``(audio_path) -> dict|None``.  Returns
            ``{"bpm": ..., "key": ..., "signature": ...}``.
        policy: Merge policy (``fill_missing``, ``overwrite_caption``,
            ``overwrite_all``).

    Returns:
        Dict with keys ``status`` (``"written"``, ``"skipped"``,
        ``"failed"``), ``path``, and optional ``error``.
    """
    from sidestep_engine.data.sidecar_io import (
        merge_fields,
        read_sidecar,
        sidecar_path_for,
        write_sidecar,
    )
    from sidestep_engine.data.lyrics_sanitizer import sanitize_headers

    sc_path = sidecar_path_for(audio_path)
    result: Dict[str, Any] = {"path": str(sc_path), "status": "written"}
    warnings: list[str] = []

    try:
        existing = read_sidecar(sc_path)
        existing = _normalize_generated_fields(existing)

        metadata_keys = ("caption", "genre", "bpm", "key", "signature", "language", "is_instrumental")
        analysis_keys = ("bpm", "key", "signature")
        lyrics_keys = ("lyrics",)

        def _has_value(key: str) -> bool:
            return bool(str((existing or {}).get(key, "") or "").strip())

        def _block_complete(*keys: str) -> bool:
            if not existing or not all(_has_value(k) for k in keys):
                return False
            if policy == "fill_missing":
                return True
            if policy == "overwrite_caption":
                return "caption" not in keys
            if policy == "overwrite_lyrics":
                return "lyrics" not in keys
            return False

        def _needs_any(*keys: str) -> bool:
            return not _block_complete(*keys)

        requested_blocks = []
        if lyrics_fn:
            requested_blocks.append(_block_complete(*lyrics_keys))
        if caption_fn:
            requested_blocks.append(_block_complete("caption"))
        if metadata_fn:
            requested_blocks.append(_block_complete(*metadata_keys))
        if audio_analyze_fn:
            requested_blocks.append(_block_complete(*analysis_keys))
        if requested_blocks and all(requested_blocks):
            result["status"] = "skipped"
            return result

        meta = resolve_metadata(audio_path)
        artist, title = meta.artist, meta.title
        if not artist:
            artist = default_artist

        new_fields: Dict[str, str] = {}

        # Local audio analysis (BPM, key, time signature)
        if audio_analyze_fn and _needs_any(*analysis_keys):
            try:
                analysis = audio_analyze_fn(audio_path)
                if analysis:
                    new_fields.update(_normalize_generated_fields(analysis))
            except Exception as exc:
                logger.warning("Audio analysis failed for %s: %s",
                               audio_path.name, exc)
                warnings.append(f"Audio analysis error: {exc}")

        # Fetch lyrics
        if lyrics_fn and _needs_any(*lyrics_keys):
            try:
                lookup_artist = artist or ""
                raw_lyrics = lyrics_fn(lookup_artist, title)
                if raw_lyrics:
                    new_fields["lyrics"] = sanitize_headers(raw_lyrics)
                else:
                    warnings.append("Lyrics not found")
                if not artist:
                    warnings.append("No artist detected — tried title-only lookup")
            except Exception as exc:
                logger.warning("Lyrics fetch failed for %s: %s",
                               audio_path.name, exc)
                warnings.append(f"Lyrics error: {exc}")

        # Generate caption + structured metadata
        if caption_fn and _needs_any("caption"):
            try:
                lyrics_excerpt = (
                    new_fields.get("lyrics")
                    or existing.get("lyrics", "")
                )[:500]
                raw_response = caption_fn(title, artist, lyrics_excerpt, audio_path)
                if raw_response:
                    parsed = parse_structured_response(raw_response)
                    new_fields.update(_normalize_generated_fields(parsed))
                else:
                    warnings.append("Caption returned empty")
            except Exception as exc:
                try:
                    from sidestep_engine.data.caption_provider_local import LocalCaptionOOMError
                except Exception:
                    LocalCaptionOOMError = None  # type: ignore[assignment]

                if LocalCaptionOOMError is not None and isinstance(exc, LocalCaptionOOMError):
                    logger.warning("Local caption OOM for %s: %s", audio_path.name, exc)
                    result["status"] = "failed"
                    result["error"] = str(exc)
                    result["error_code"] = "local_caption_oom"
                    return result
                logger.warning("Caption generation failed for %s: %s",
                               audio_path.name, exc)
                warnings.append(f"Caption error: {exc}")

        if metadata_fn and _needs_any(*metadata_keys):
            try:
                metadata = metadata_fn(audio_path)
                if metadata:
                    new_fields.update(_normalize_generated_fields(metadata))
                else:
                    warnings.append("Metadata returned empty")
            except Exception as exc:
                logger.warning("Metadata generation failed for %s: %s",
                               audio_path.name, exc)
                warnings.append(f"Metadata error: {exc}")

        if not new_fields:
            result["status"] = "skipped"
        else:
            merged = merge_fields(existing, new_fields, policy=policy)
            write_sidecar(sc_path, merged)

    except Exception as exc:
        logger.error("Enrichment failed for %s: %s", audio_path.name, exc)
        result["status"] = "failed"
        result["error"] = str(exc)

    if warnings:
        result["warnings"] = warnings
    return result
