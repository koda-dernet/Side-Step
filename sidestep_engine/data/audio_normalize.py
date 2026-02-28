"""
Audio normalization for training data preprocessing.

Two methods:

- **Peak** (-1.0 dBFS): matches ACE-Step's own ``normalize_audio``.
  Pure torch, no extra dependencies.
- **LUFS** (-14 LUFS, EBU R128): perceptually uniform loudness.
  Requires ``pyloudnorm`` (optional dependency).

The dispatcher ``normalize_audio`` routes by method name and handles
the ``"none"`` passthrough.
"""

from __future__ import annotations

import logging
from typing import Optional

import torch

logger = logging.getLogger(__name__)


def peak_normalize(audio: torch.Tensor, target_db: float = -1.0) -> torch.Tensor:
    """Peak-normalize *audio* so its absolute peak hits *target_db*.

    Matches the behaviour of upstream ``audio_utils.normalize_audio``.
    Silence (peak < 1e-6) is returned unchanged.

    Args:
        audio: Waveform tensor, any shape (e.g. ``[C, S]``).
        target_db: Target peak level in dBFS (default -1.0).

    Returns:
        Normalized copy of *audio* (same shape and dtype).
    """
    peak = torch.max(torch.abs(audio))
    if peak < 1e-6:
        return audio
    target_amp = 10 ** (target_db / 20.0)
    return audio * (target_amp / peak)


def lufs_normalize(
    audio: torch.Tensor,
    sample_rate: int,
    target_lufs: float = -14.0,
) -> torch.Tensor:
    """Loudness-normalize *audio* to *target_lufs* (EBU R128).

    Requires ``pyloudnorm``.  If the library is missing, falls back
    to :func:`peak_normalize` with a warning.

    Args:
        audio: Waveform ``[C, S]`` (channels-first, float).
        sample_rate: Audio sample rate in Hz.
        target_lufs: Target integrated loudness (default -14.0 LUFS).

    Returns:
        Normalized copy of *audio*.
    """
    try:
        import pyloudnorm as pyln
    except ImportError:
        logger.warning(
            "[Side-Step] pyloudnorm not installed -- falling back to peak "
            "normalization.  Install with: pip install pyloudnorm"
        )
        return peak_normalize(audio)

    # pyloudnorm expects numpy [samples, channels]
    np_audio = audio.T.numpy()  # [C, S] -> [S, C]

    meter = pyln.Meter(sample_rate)
    current_lufs = meter.integrated_loudness(np_audio)

    # If the measurement is -inf (silence), skip normalization
    if current_lufs == float("-inf"):
        return audio

    np_normalized = pyln.normalize.loudness(np_audio, current_lufs, target_lufs)
    return torch.from_numpy(np_normalized.T.copy()).to(dtype=audio.dtype)


def normalize_audio(
    audio: torch.Tensor,
    sample_rate: int,
    method: str = "none",
    target_db: float = -1.0,
    target_lufs: float = -14.0,
) -> torch.Tensor:
    """Dispatch audio normalization by *method* name.

    Args:
        audio: Waveform tensor ``[C, S]``.
        sample_rate: Audio sample rate in Hz.
        method: ``"none"`` (passthrough), ``"peak"``, or ``"lufs"``.
        target_db: Target peak dB (peak method only).
        target_lufs: Target loudness (LUFS method only).

    Returns:
        Normalized audio (or unchanged if method is ``"none"``).
    """
    if method == "none":
        return audio
    if method == "peak":
        return peak_normalize(audio, target_db=target_db)
    if method == "lufs":
        return lufs_normalize(audio, sample_rate, target_lufs=target_lufs)
    logger.warning("[Side-Step] Unknown normalize method '%s', skipping", method)
    return audio
