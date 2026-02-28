"""
Per-song enrichment pipeline for the AI dataset builder.

Orchestrates lyrics fetching, sanitization, and caption generation
for a single audio file.  Called in a loop by the wizard orchestrator.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any, Dict, Optional

from sidestep_engine.data.caption_config import parse_structured_response

logger = logging.getLogger(__name__)

# Common filename patterns: "Artist - Title", "Title"
_FILENAME_RE = re.compile(r"^(.+?)\s*[-–—]\s*(.+)$")


def parse_filename(audio_path: Path) -> tuple[str, str]:
    """Extract artist and title from an audio filename.

    Tries ``"Artist - Title"`` patterns first, falls back to using
    the stem as the title with an empty artist.

    Args:
        audio_path: Path to the audio file.

    Returns:
        ``(artist, title)`` tuple.  Artist may be empty.
    """
    stem = audio_path.stem
    match = _FILENAME_RE.match(stem)
    if match:
        return match.group(1).strip(), match.group(2).strip()
    return "", stem.strip()


def enrich_one(
    audio_path: Path,
    *,
    default_artist: str = "",
    caption_fn: Optional[Any] = None,
    lyrics_fn: Optional[Any] = None,
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

        # Early skip: all generated fields already populated → no API calls needed
        if policy == "fill_missing" and existing:
            from sidestep_engine.data.sidecar_io import GENERATED_FIELDS
            if all(existing.get(k, "").strip() for k in GENERATED_FIELDS):
                result["status"] = "skipped"
                return result

        artist, title = parse_filename(audio_path)
        if not artist:
            artist = default_artist

        new_fields: Dict[str, str] = {}

        # Fetch lyrics
        if lyrics_fn:
            try:
                lookup_artist = artist or ""
                raw_lyrics = lyrics_fn(lookup_artist, title)
                if raw_lyrics:
                    new_fields["lyrics"] = sanitize_headers(raw_lyrics)
                else:
                    warnings.append("Not found on Genius")
                if not artist:
                    warnings.append("No artist detected — tried title-only lookup")
            except Exception as exc:
                logger.warning("Lyrics fetch failed for %s: %s",
                               audio_path.name, exc)
                warnings.append(f"Lyrics error: {exc}")

        # Generate caption + structured metadata
        if caption_fn:
            try:
                lyrics_excerpt = (
                    new_fields.get("lyrics")
                    or existing.get("lyrics", "")
                )[:500]
                raw_response = caption_fn(title, artist, lyrics_excerpt, audio_path)
                if raw_response:
                    parsed = parse_structured_response(raw_response)
                    new_fields.update(parsed)
                else:
                    warnings.append("Caption returned empty")
            except Exception as exc:
                logger.warning("Caption generation failed for %s: %s",
                               audio_path.name, exc)
                warnings.append(f"Caption error: {exc}")

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
