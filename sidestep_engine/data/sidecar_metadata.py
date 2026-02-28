"""
Sidecar metadata reader for the preprocessing pipeline.

Reads ``.txt`` sidecar files and normalizes them to the dict format
expected by ``preprocess.py`` (``load_sample_metadata``).  Used as a
fallback when no ``dataset.json`` is available so that captions, lyrics,
BPM, key, genre, etc. are preserved during preprocessing.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List

logger = logging.getLogger(__name__)


def load_sidecars_for_files(audio_files: List[Path]) -> Dict[str, Dict[str, Any]]:
    """Read ``.txt`` sidecar metadata for each audio file.

    Uses the same multi-convention reader as ``dataset_builder`` so that
    per-file captions, lyrics, BPM, key, genre, etc. are preserved when
    no ``dataset.json`` is available.  Also computes audio duration from
    the file when not provided in the sidecar.

    Returns a ``{filename: metadata_dict}`` mapping (only files with
    sidecar data).
    """
    from sidestep_engine.data.dataset_builder import load_sidecar_metadata
    from sidestep_engine.data.audio_duration import get_audio_duration

    meta: Dict[str, Dict[str, Any]] = {}
    for af in audio_files:
        raw = load_sidecar_metadata(af)
        if not raw:
            continue
        normalized = normalize_sidecar(raw, af)
        # Compute real duration from audio file (matches build_dataset behaviour)
        if not normalized.get("duration"):
            normalized["duration"] = get_audio_duration(str(af))
        meta[af.name] = normalized
    if meta:
        logger.info(
            "[Side-Step] Loaded sidecar metadata for %d/%d files",
            len(meta), len(audio_files),
        )
    return meta


def normalize_sidecar(raw: Dict[str, Any], audio_path: Path) -> Dict[str, Any]:
    """Normalize raw sidecar data to the format expected by the preprocess pipeline.

    Handles key remapping (``key`` → ``keyscale``, ``signature`` →
    ``timesignature``), type coercion for BPM, and defaults for missing fields.
    """
    caption = raw.get("caption", "")
    if not caption:
        caption = audio_path.stem.replace("_", " ").replace("-", " ")

    lyrics = raw.get("lyrics", "")
    is_instrumental = (
        raw.get("is_instrumental", "").lower() in ("true", "1", "yes")
        if "is_instrumental" in raw
        else (not lyrics.strip() or "[Instrumental]" in lyrics)
    )
    if not lyrics.strip():
        lyrics = "[Instrumental]"

    bpm_raw = raw.get("bpm", "")
    try:
        bpm = int(float(bpm_raw)) if bpm_raw else None
    except (ValueError, TypeError):
        bpm = None

    return {
        "filename": audio_path.name,
        "caption": caption,
        "lyrics": lyrics,
        "genre": raw.get("genre", ""),
        "bpm": bpm,
        "keyscale": raw.get("key", raw.get("keyscale", "")),
        "timesignature": raw.get("signature", raw.get("timesignature", "")),
        "duration": _parse_duration(raw.get("duration", "")),
        "is_instrumental": is_instrumental,
        "custom_tag": raw.get("custom_tag", raw.get("trigger", "")),
        "prompt_override": raw.get("prompt_override"),
    }


def _parse_duration(raw_val: Any) -> int:
    """Coerce a raw duration value to int seconds, returning 0 on failure."""
    if not raw_val:
        return 0
    try:
        return max(0, int(float(raw_val)))
    except (ValueError, TypeError):
        return 0


def default_sample_meta(af: Path) -> Dict[str, Any]:
    """Return default metadata for an audio file with no sidecar."""
    return {
        "filename": af.name,
        "caption": af.stem.replace("_", " ").replace("-", " "),
        "lyrics": "[Instrumental]",
        "genre": "",
        "bpm": None,
        "keyscale": "",
        "timesignature": "",
        "duration": 0,
        "is_instrumental": True,
        "custom_tag": "",
        "prompt_override": None,
    }
