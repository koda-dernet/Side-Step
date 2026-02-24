"""
Folder-based dataset builder for Side-Step.

Scans a directory for audio files with sidecar metadata and generates
a ``dataset.json`` compatible with the preprocessing pipeline.

Supports three metadata conventions (auto-detected per file):

1. ``song.txt`` with ``key: value`` pairs (Side-Step convention).
2. ``song.caption.txt`` + ``song.lyrics.txt`` (upstream ACE-Step).
3. No sidecar files (defaults derived from the filename).
"""

from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from acestep.training_v2.audio_duration import get_audio_duration
from acestep.training_v2.preprocess_discovery import AUDIO_EXTENSIONS

logger = logging.getLogger(__name__)


def parse_txt_metadata(path: Path) -> Dict[str, str]:
    """Parse a ``key: value`` text file into a dict.

    Handles multi-line values (e.g. lyrics spanning many lines),
    UTF-8 BOM (common on Windows), CRLF line endings, and minor
    leading whitespace on key lines.  Keys are normalised to
    lowercase.  Returns ``{}`` if the file does not exist or is empty.
    """
    if not path.exists():
        return {}
    # utf-8-sig transparently strips a leading BOM if present.
    content = path.read_text(encoding="utf-8-sig", errors="replace").strip()
    if not content:
        return {}

    meta: Dict[str, str] = {}
    current_key: Optional[str] = None
    current_lines: List[str] = []

    for line in content.splitlines():
        # Left-strip for key detection so minor leading whitespace
        # (e.g. from editors that auto-indent) doesn't break parsing.
        lstripped = line.lstrip()
        # New key: contains ':', is not a lyrics section marker like
        # [Verse 1], and is not a tab-indented continuation line.
        if ":" in lstripped and not lstripped.startswith("["):
            if current_key is not None:
                meta[current_key] = "\n".join(current_lines).strip()
            key, value = lstripped.split(":", 1)
            current_key = key.strip().lower()
            current_lines = [value.strip()] if value.strip() else []
        elif current_key is not None:
            current_lines.append(line.rstrip())

    if current_key is not None:
        meta[current_key] = "\n".join(current_lines).strip()

    return meta


def _read_text_file(path: Path) -> str:
    """Read and strip a text file, returning ``""`` if missing.

    Uses ``utf-8-sig`` to transparently handle a Windows BOM.
    """
    if not path.exists():
        return ""
    try:
        return path.read_text(encoding="utf-8-sig", errors="replace").strip()
    except Exception:
        return ""


def load_sidecar_metadata(audio_path: Path) -> Dict[str, Any]:
    """Auto-detect and load metadata for an audio file.

    Tries (in order):
        1. ``<stem>.txt`` with ``key: value`` pairs.
        2. ``<stem>.caption.txt`` + ``<stem>.lyrics.txt`` (ACE-Step).
        3. Defaults derived from filename.
    """
    stem_path = audio_path.with_suffix("")

    # Convention 1: key-value .txt
    txt_path = audio_path.with_suffix(".txt")
    if txt_path.exists():
        meta = parse_txt_metadata(txt_path)
        if meta.get("caption"):
            return meta

    # Convention 2: separate .caption.txt and .lyrics.txt
    caption_path = Path(str(stem_path) + ".caption.txt")
    lyrics_path = Path(str(stem_path) + ".lyrics.txt")
    caption = _read_text_file(caption_path)
    lyrics = _read_text_file(lyrics_path)
    if caption:
        return {"caption": caption, "lyrics": lyrics}

    # Convention 3: fall back to .txt for lyrics only (upstream compat)
    if txt_path.exists() and not caption:
        lyrics_from_txt = _read_text_file(txt_path)
        if lyrics_from_txt:
            return {"lyrics": lyrics_from_txt}

    # Convention 4: no metadata at all
    return {}


def build_dataset(
    input_dir: str,
    tag: str = "",
    tag_position: str = "prepend",
    name: str = "local_dataset",
    output: Optional[str] = None,
    genre_ratio: int = 0,
) -> Tuple[Path, Dict[str, Any]]:
    """Scan *input_dir* for audio + metadata and write ``dataset.json``.

    Args:
        input_dir: Root directory to scan recursively.
        tag: Custom trigger tag applied to all samples.
        tag_position: Tag placement: ``"prepend"``, ``"append"``,
            or ``"replace"``.
        name: Dataset name in the metadata block.
        output: Output JSON path (default: ``<input_dir>/dataset.json``).
        genre_ratio: Percentage (0-100) of samples that use genre
            instead of caption during preprocessing.

    Returns:
        ``(output_path, stats_dict)`` where stats has keys
        ``total``, ``skipped``, ``with_metadata``.
    """
    base = Path(input_dir).resolve()
    if not base.is_dir():
        raise FileNotFoundError(f"Input directory not found: {base}")

    audio_files = sorted(
        f for f in base.rglob("*")
        if f.is_file() and f.suffix.lower() in AUDIO_EXTENSIONS
    )
    if not audio_files:
        raise FileNotFoundError(
            f"No audio files found in {base}. "
            f"Supported: {', '.join(sorted(AUDIO_EXTENSIONS))}"
        )

    samples: List[Dict[str, Any]] = []
    skipped = 0
    with_meta = 0

    for af in audio_files:
        meta = load_sidecar_metadata(af)
        caption = meta.get("caption", "")
        if not caption:
            caption = af.stem.replace("_", " ").replace("-", " ")
        else:
            with_meta += 1

        lyrics = meta.get("lyrics", "")
        is_instrumental = (
            meta.get("is_instrumental", "").lower() in ("true", "1", "yes")
            if "is_instrumental" in meta
            else (not lyrics.strip() or "[Instrumental]" in lyrics)
        )
        if not lyrics.strip():
            lyrics = "[Instrumental]"

        bpm_raw = meta.get("bpm", "")
        try:
            bpm = int(float(bpm_raw)) if bpm_raw else None
        except (ValueError, TypeError):
            bpm = None

        samples.append({
            "audio_path": str(af),
            "filename": af.name,
            "caption": caption,
            "genre": meta.get("genre", ""),
            "lyrics": lyrics,
            "bpm": bpm,
            "keyscale": meta.get("key", meta.get("keyscale", "")),
            "timesignature": meta.get("timesignature", ""),
            "duration": get_audio_duration(str(af)),
            "is_instrumental": is_instrumental,
            "custom_tag": tag,
            "prompt_override": meta.get("prompt_override"),
        })

    output_path = Path(output) if output else base / "dataset.json"
    dataset = {
        "metadata": {
            "name": name,
            "custom_tag": tag,
            "tag_position": tag_position,
            "genre_ratio": genre_ratio,
            "created_at": datetime.now().isoformat(),
            "num_samples": len(samples),
        },
        "samples": samples,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        json.dumps(dataset, indent=2, ensure_ascii=False), encoding="utf-8",
    )

    stats = {"total": len(samples), "skipped": skipped, "with_metadata": with_meta}
    logger.info(
        "[Side-Step] Dataset built: %d samples (%d with metadata) -> %s",
        len(samples), with_meta, output_path,
    )
    return output_path, stats
