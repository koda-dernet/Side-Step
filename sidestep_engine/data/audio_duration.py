"""
Portable audio duration detection for preprocessing and dataset building.

Uses torchcodec (fast, supports all ffmpeg formats) with a soundfile
fallback.  Returns integer seconds to avoid float-precision issues
with ffprobe-style outputs.

Vendored from upstream ``audio_io.get_audio_duration`` so Side-Step
stays fully standalone.
"""

from __future__ import annotations

import logging
import shutil
import subprocess
from pathlib import Path
from typing import List

logger = logging.getLogger(__name__)


def get_audio_duration(audio_path: str) -> int:
    """Return the duration of an audio file in whole seconds.

    Resolution chain:
        1. ``soundfile.info`` (safe, all platforms, wav/flac/ogg).
        2. ``torchcodec.decoders.AudioDecoder`` (all ffmpeg formats).
        3. ``mutagen`` (pure-Python, mp3/m4a/aac/ogg/flac).
        4. ``ffprobe`` subprocess (requires ffmpeg installed).
        5. Returns ``0`` if all fail.

    soundfile is tried first because torchcodec can trigger a Windows
    DLL error dialog on torch version mismatches before Python's
    exception handler runs.

    The result is truncated to ``int`` so callers never deal with
    sub-second float precision.
    """
    # Primary: soundfile (safe, no DLL issues on Windows)
    try:
        import soundfile as sf
        info = sf.info(audio_path)
        return int(info.duration)
    except ImportError:
        logger.debug("soundfile not available, trying torchcodec")
    except Exception as exc:
        logger.debug("soundfile failed for %s: %s", audio_path, exc)

    # Fallback: torchcodec (broader format support via ffmpeg)
    try:
        from torchcodec.decoders import AudioDecoder
        decoder = AudioDecoder(audio_path)
        return int(decoder.metadata.duration_seconds)
    except ImportError:
        logger.debug("torchcodec not available either")
    except Exception as exc:
        logger.debug("torchcodec failed for %s: %s", audio_path, exc)

    # Fallback: mutagen (pure-Python, handles mp3/m4a/aac/ogg/flac)
    try:
        import mutagen
        mf = mutagen.File(audio_path)
        if mf is not None and mf.info is not None:
            return int(mf.info.length)
    except ImportError:
        logger.debug("mutagen not available, trying ffprobe")
    except Exception as exc:
        logger.debug("mutagen failed for %s: %s", audio_path, exc)

    # Fallback: ffprobe subprocess (ffmpeg is required for preprocessing)
    dur = _ffprobe_duration(audio_path)
    if dur > 0:
        return dur

    logger.warning("Failed to get duration for %s", audio_path)
    return 0


def _ffprobe_duration(audio_path: str) -> int:
    """Get duration via ffprobe subprocess. Returns 0 on failure."""
    ffprobe = shutil.which("ffprobe")
    if not ffprobe:
        return 0
    try:
        result = subprocess.run(
            [
                ffprobe, "-v", "error",
                "-show_entries", "format=duration",
                "-of", "default=noprint_wrappers=1:nokey=1",
                str(audio_path),
            ],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode == 0 and result.stdout.strip():
            return int(float(result.stdout.strip()))
    except Exception as exc:
        logger.debug("ffprobe failed for %s: %s", audio_path, exc)
    return 0


def detect_max_duration(audio_files: List[Path]) -> int:
    """Scan *audio_files* and return the longest duration in seconds.

    Returns ``0`` when the list is empty or every probe fails.
    """
    if not audio_files:
        return 0

    longest = 0
    for af in audio_files:
        dur = get_audio_duration(str(af))
        logger.debug("[Side-Step] %s: %ds", af.name, dur)
        if dur > longest:
            longest = dur

    logger.info(
        "[Side-Step] Detected longest clip: %ds (across %d files)",
        longest,
        len(audio_files),
    )
    return longest
