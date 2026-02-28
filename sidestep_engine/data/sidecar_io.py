"""
Read, merge, and write Option-A ``.txt`` sidecar files.

Option-A format: ``key: value`` pairs with a multi-line ``lyrics:``
block at the end.  This module handles atomic writes (temp + rename)
and merge policies for the AI dataset builder.
"""

from __future__ import annotations

import logging
import os
import shutil
import tempfile
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

# Keys written to sidecars (order preserved in output)
_FIELD_ORDER = ("caption", "genre", "bpm", "key", "signature")

# Fields that the AI pipeline can generate
GENERATED_FIELDS = {"caption", "lyrics", "genre", "bpm", "key", "signature"}


def read_sidecar(path: Path) -> Dict[str, str]:
    """Parse an Option-A sidecar file into a dict.

    Delegates to the existing ``parse_txt_metadata`` parser from
    ``dataset_builder`` to stay consistent with the rest of the
    pipeline.

    Args:
        path: Path to the ``.txt`` sidecar file.

    Returns:
        Dict of ``{key: value}`` with lowercase keys.  Empty dict
        if the file does not exist or is empty.
    """
    from sidestep_engine.data.dataset_builder import parse_txt_metadata

    return parse_txt_metadata(path)


def merge_fields(
    existing: Dict[str, str],
    new_fields: Dict[str, str],
    policy: str = "fill_missing",
) -> Dict[str, str]:
    """Merge *new_fields* into *existing* according to *policy*.

    Policies:
        ``fill_missing``: only write keys that are absent or empty.
        ``overwrite_caption``: overwrite ``caption`` only; fill rest.
        ``overwrite_all``: overwrite all generated fields.

    Args:
        existing: Current sidecar data.
        new_fields: Newly generated data to merge in.
        policy: One of the three merge policies.

    Returns:
        Merged dict (new copy; inputs are not mutated).
    """
    merged = dict(existing)

    for key, value in new_fields.items():
        if not value:
            continue

        if policy == "fill_missing":
            if not merged.get(key, "").strip():
                merged[key] = value

        elif policy == "overwrite_caption":
            if key == "caption":
                merged[key] = value
            elif not merged.get(key, "").strip():
                merged[key] = value

        elif policy == "overwrite_all":
            if key in GENERATED_FIELDS:
                merged[key] = value
            elif not merged.get(key, "").strip():
                merged[key] = value

    return merged


def _normalize_value(val: Any) -> str:
    """Coerce a sidecar value to a clean string.

    Booleans are lowercased (``True`` -> ``"true"``) so the frontend's
    ``=== 'true'`` check works on roundtrip.
    """
    if isinstance(val, bool):
        return "true" if val else "false"
    s = str(val)
    if s.lower() in ("true", "false"):
        return s.lower()
    return s


def write_sidecar(path: Path, data: Dict[str, str]) -> None:
    """Write an Option-A sidecar file atomically.

    Creates a ``.bak`` backup of the existing file before overwriting.
    Uses a temporary file + rename to avoid partial writes.  The
    ``lyrics`` key is always written last as a multi-line block.

    Args:
        path: Destination ``.txt`` file path.
        data: Dict of sidecar fields to write.
    """
    # Best-effort backup of the previous version
    if path.is_file():
        bak_path = path.with_suffix(".txt.bak")
        try:
            shutil.copy2(str(path), str(bak_path))
        except OSError:
            pass

    lines: list[str] = []
    written_keys: set[str] = set()

    for key in _FIELD_ORDER:
        val = _normalize_value(data.get(key, ""))
        lines.append(f"{key}: {val}")
        written_keys.add(key)

    # Preserve any extra keys not in _FIELD_ORDER (e.g. repeat,
    # is_instrumental, prompt_override) so re-runs don't drop them.
    written_keys.add("lyrics")  # handled separately below
    for key in sorted(data):
        if key not in written_keys and data[key]:
            val = _normalize_value(data[key])
            lines.append(f"{key}: {val}")

    # Lyrics block: written last, multi-line
    lyrics = data.get("lyrics", "")
    if lyrics:
        lines.append(f"lyrics:\n{lyrics}")
    else:
        lines.append("lyrics:")

    content = "\n".join(lines) + "\n"

    # Atomic write: temp file in same directory, then rename
    fd, tmp_path = tempfile.mkstemp(
        dir=str(path.parent), suffix=".tmp", prefix=".sidecar_"
    )
    fd_closed = False
    try:
        os.write(fd, content.encode("utf-8"))
        os.close(fd)
        fd_closed = True
        os.replace(tmp_path, str(path))
        logger.debug("Sidecar written: %s", path)
    except Exception:
        if not fd_closed:
            os.close(fd)
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
        raise


def sidecar_path_for(audio_path: Path) -> Path:
    """Return the ``.txt`` sidecar path for an audio file.

    Args:
        audio_path: Path to an audio file (e.g. ``song.wav``).

    Returns:
        Path with the same stem and ``.txt`` extension.
    """
    return audio_path.with_suffix(".txt")
