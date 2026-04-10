"""Audio loading utility (vendored from ACE-Step)."""

from __future__ import annotations

import numpy as np
import torch


def load_audio_stereo(audio_path: str, target_sample_rate: int, max_duration: float):
    """Load audio, resample, convert to stereo, and truncate.

    Primary path uses ``soundfile`` + ``librosa.resample`` so we do **not**
    import ``torchaudio`` at module load time.  On Windows, ``import torchaudio``
    initializes ``torio`` and tries to load ``libtorio_ffmpeg*.pyd``; that often
    logs long DEBUG tracebacks when FFmpeg extension DLLs are missing, even
    though a system ``ffmpeg``/``ffprobe`` binary is on ``PATH``.

    ``torchaudio`` is only imported on fallback when ``soundfile`` cannot read
    the file (uncommon formats).
    """
    audio: torch.Tensor
    sr: int

    try:
        import soundfile as sf

        data, sr = sf.read(audio_path, dtype="float32", always_2d=True)
        # (samples, channels) -> (channels, samples)
        audio_np = np.ascontiguousarray(data.T)
        sr = int(sr)
        if sr != target_sample_rate:
            import librosa

            audio_np = librosa.resample(
                audio_np,
                orig_sr=sr,
                target_sr=target_sample_rate,
                axis=1,
            )
            sr = target_sample_rate
        audio = torch.from_numpy(np.ascontiguousarray(audio_np))
    except Exception:
        import torchaudio

        audio, sr = torchaudio.load(audio_path)
        sr = int(sr)
        if sr != target_sample_rate:
            resampler = torchaudio.transforms.Resample(sr, target_sample_rate)
            audio = resampler(audio)
            sr = target_sample_rate

    if audio.shape[0] == 1:
        audio = audio.repeat(2, 1)
    elif audio.shape[0] > 2:
        audio = audio[:2, :]

    max_samples = int(max_duration * target_sample_rate)
    if audio.shape[1] > max_samples:
        audio = audio[:, :max_samples]

    return audio, sr
