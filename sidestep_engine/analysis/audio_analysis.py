"""
Local offline audio analysis for Side-Step.

Extracts BPM, musical key, and time signature from an audio file.
Three quality modes are available:

**faf** ("Fast As F*ck") — ~2-3 s per file
    Skip Demucs.  Single-method detection on the raw mix.

**mid** — ~10-12 s GPU
    Demucs stem separation → 3-method BPM ensemble → 3-profile key
    voting → accent-pattern time-sig.

**sas** ("Smart/Slow As Sh*t") — ~18-30 s GPU
    Everything in *mid*, plus PLP, multi-hop, multi-chroma fusion,
    tonnetz, tuning correction, ending resolution, multi-band onset,
    tempogram harmonics, and energy-gated chunked analysis.

Each detector returns ``(value, confidence)`` where confidence is
``"high"``, ``"medium"``, or ``"low"``.  Confidence is surfaced in the
GUI but **not** written to sidecar files.
"""

from __future__ import annotations

import gc
import logging
import shutil
import tempfile
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

logger = logging.getLogger(__name__)

# ── Analysis modes ─────────────────────────────────────────────────
MODES = ("faf", "mid", "sas")
_DEFAULT_MODE = "mid"
_SAS_NUM_CHUNKS = 5
_SAS_CHUNK_SECONDS = 15  # seconds per analysis window


def _select_chunks(
    y: np.ndarray,
    sr: int,
    n_chunks: int = _SAS_NUM_CHUNKS,
    chunk_sec: float = _SAS_CHUNK_SECONDS,
    min_gap_sec: float = 10.0,
    use_onset: bool = True,
) -> list[np.ndarray]:
    """Select the most informative audio chunks for S-A-S analysis.

    Energy-gated + spread: rank windows by onset density (or RMS),
    discard below-median, then greedily pick *n_chunks* that are
    maximally spread apart (at least *min_gap_sec* between centres).

    Args:
        y: Mono audio signal.
        sr: Sample rate.
        n_chunks: How many chunks to return.
        chunk_sec: Duration of each chunk in seconds.
        min_gap_sec: Minimum gap between chunk centres.
        use_onset: If True, rank by onset density; else by RMS.

    Returns:
        List of audio arrays (each ~chunk_sec long).
    """
    import librosa

    chunk_samples = int(chunk_sec * sr)
    hop_samples = chunk_samples // 2  # 50 % overlap for candidate windows
    if len(y) < chunk_samples:
        return [y]

    # Build candidate windows
    candidates: list[tuple[int, float]] = []  # (start_sample, score)
    for start in range(0, len(y) - chunk_samples + 1, hop_samples):
        window = y[start : start + chunk_samples]
        if use_onset:
            onset_env = librosa.onset.onset_strength(y=window, sr=sr)
            score = float(np.mean(onset_env))
        else:
            score = float(np.sqrt(np.mean(window ** 2)))  # RMS
        candidates.append((start, score))

    if not candidates:
        return [y]

    # Gate: discard below-median energy
    scores = np.array([s for _, s in candidates])
    median_score = float(np.median(scores))
    gated = [(start, score) for start, score in candidates if score >= median_score]
    if not gated:
        gated = candidates

    # Sort by score descending
    gated.sort(key=lambda x: x[1], reverse=True)

    # Greedy spread selection
    min_gap_samples = int(min_gap_sec * sr)
    selected_starts: list[int] = []
    selected_scores: list[float] = []

    for start, score in gated:
        centre = start + chunk_samples // 2
        too_close = any(
            abs(centre - (s + chunk_samples // 2)) < min_gap_samples
            for s in selected_starts
        )
        if not too_close:
            selected_starts.append(start)
            selected_scores.append(score)
            if len(selected_starts) >= n_chunks:
                break

    # If we didn't get enough, relax the gap constraint
    if len(selected_starts) < n_chunks:
        for start, score in gated:
            if start not in selected_starts:
                selected_starts.append(start)
                selected_scores.append(score)
                if len(selected_starts) >= n_chunks:
                    break

    # Return in chronological order
    selected_starts.sort()
    chunks = [y[s : s + chunk_samples] for s in selected_starts]
    return chunks


# ── Key profile families for multi-profile voting ────────────────
#
# Each family provides (major, minor) weight vectors over 12 pitch classes
# starting from C.  Using multiple families and voting across them
# significantly reduces key-detection errors.

_KEY_PROFILES = {
    "krumhansl": {
        "major": np.array([6.35, 2.23, 3.48, 2.33, 4.38, 4.09,
                           2.52, 5.19, 2.39, 3.66, 2.29, 2.88]),
        "minor": np.array([6.33, 2.68, 3.52, 5.38, 2.60, 3.53,
                           2.54, 4.75, 3.98, 2.69, 3.34, 3.17]),
    },
    "temperley": {
        "major": np.array([5.0, 2.0, 3.5, 2.0, 4.5, 4.0,
                           2.0, 4.5, 2.0, 3.5, 1.5, 4.0]),
        "minor": np.array([5.0, 2.0, 3.5, 4.5, 2.0, 3.5,
                           2.0, 4.5, 3.5, 2.0, 1.5, 4.0]),
    },
    "albrecht": {
        "major": np.array([0.238, 0.006, 0.111, 0.006, 0.137, 0.094,
                           0.016, 0.214, 0.009, 0.080, 0.008, 0.081]),
        "minor": np.array([0.220, 0.006, 0.104, 0.123, 0.019, 0.103,
                           0.012, 0.214, 0.062, 0.022, 0.061, 0.052]),
    },
}

_PITCH_CLASSES = [
    "C", "C#", "D", "D#", "E", "F",
    "F#", "G", "G#", "A", "A#", "B",
]


def _flush_vram() -> None:
    """Release GPU memory between pipeline stages."""
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        elif hasattr(torch, 'mps') and torch.mps.is_available():
            torch.mps.empty_cache()
            torch.mps.synchronize()
        elif hasattr(torch, 'xpu') and torch.xpu.is_available():
            torch.xpu.empty_cache()
            torch.xpu.synchronize()
    except ImportError:
        pass


def _resolve_device(device: str = "auto") -> str:
    """Resolve 'auto' to the best available device string."""
    if device != "auto":
        return device
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch, "mps") and torch.mps.is_available():
            return "mps"
        if hasattr(torch, "xpu") and torch.xpu.is_available():
            return "xpu"
    except ImportError:
        pass
    return "cpu"


def _preprocess_audio(y: np.ndarray, sr: int) -> np.ndarray:
    """Trim silence and peak-normalize an audio signal.

    Args:
        y: Audio time series (mono).
        sr: Sample rate.

    Returns:
        Preprocessed audio array.
    """
    import librosa

    # Trim leading/trailing silence
    y_trimmed, _ = librosa.effects.trim(y, top_db=30)
    if len(y_trimmed) < sr:  # less than 1 second after trim — keep original
        y_trimmed = y

    # Peak-normalize to 0 dBFS
    peak = np.max(np.abs(y_trimmed))
    if peak > 0:
        y_trimmed = y_trimmed / peak

    return y_trimmed


# ── Step 1: Demucs stem separation ─────────────────────────────────


def separate_stems(
    audio_path: Path,
    output_dir: Path,
    device: str = "auto",
) -> tuple[Path, Path]:
    """Run Demucs HTDemucs and return paths to drums and harmonics stems.

    Uses ``demucs.pretrained.get_model`` + ``demucs.apply.apply_model``
    (the v4.0.x low-level API).  The harmonics stem is created by
    summing the bass and other stems.  Vocals are discarded.

    Args:
        audio_path: Path to the input audio file.
        output_dir: Directory to write stem WAV files into.
        device: Torch device string (auto, cuda, cpu).

    Returns:
        ``(drums_path, harmonics_path)`` as absolute Paths.
    """
    import torch
    import torchaudio
    from demucs.pretrained import get_model
    from demucs.apply import apply_model

    device = _resolve_device(device)
    torch_device = torch.device(device)

    logger.info("Loading Demucs HTDemucs model on %s", device)
    model = get_model("htdemucs")
    model.to(torch_device)
    model.eval()

    # Load audio — torchaudio returns (channels, samples) and sample rate
    wav, sr = torchaudio.load(str(audio_path))

    # Resample to model's expected rate (44100 Hz) if needed
    if sr != model.samplerate:
        wav = torchaudio.functional.resample(wav, sr, model.samplerate)
        sr = model.samplerate

    # HTDemucs requires stereo (2-channel) input — duplicate mono if needed
    if wav.shape[0] == 1:
        wav = wav.repeat(2, 1)

    # apply_model expects (batch, channels, samples)
    wav = wav.unsqueeze(0).to(torch_device)

    logger.info("Separating stems for %s", audio_path.name)
    with torch.no_grad():
        # apply_model returns (batch, num_sources, channels, samples)
        # apply_model signature: (model, mix, shifts, split, overlap,
        #   transition_power, progress, device, num_workers, segment)
        sources = apply_model(model, wav, device=torch_device)

    # model.sources == ['drums', 'bass', 'other', 'vocals']
    source_map = {name: i for i, name in enumerate(model.sources)}

    drums = sources[0, source_map["drums"]].cpu()
    bass = sources[0, source_map["bass"]].cpu()
    other = sources[0, source_map["other"]].cpu()

    # Merge bass + other → harmonics
    harmonics = bass + other

    # Write to WAV files
    drums_path = output_dir / "drums.wav"
    harmonics_path = output_dir / "harmonics.wav"

    torchaudio.save(str(drums_path), drums, sr)
    torchaudio.save(str(harmonics_path), harmonics, sr)

    # Free the model and tensors
    del model, sources, wav, drums, bass, other, harmonics
    _flush_vram()

    logger.info("Stems written: %s, %s", drums_path, harmonics_path)
    return drums_path, harmonics_path


# ── Step 2: BPM detection (librosa on drums stem) ────────────────


def _octave_correct(bpm: float, lo: float = 70.0, hi: float = 180.0) -> float:
    """Fold a BPM value into the musical sweet-spot range [lo, hi].

    Halves or doubles repeatedly until the value lands in range,
    preferring the direction that requires fewer steps.
    """
    if bpm <= 0:
        return bpm
    candidate = bpm
    while candidate > hi:
        candidate /= 2.0
    while candidate < lo:
        candidate *= 2.0
    # If still out of range after one pass, return original
    if candidate < lo or candidate > hi:
        return bpm
    return candidate


def _bpm_core_ensemble(
    y: np.ndarray, sr: int,
) -> list[float]:
    """Run the 3-method BPM ensemble on a single audio buffer.

    Returns a list of octave-corrected BPM estimates (may be empty).
    """
    import librosa

    estimates: list[float] = []

    # Method A: beat_track
    try:
        tempo_a, _ = librosa.beat.beat_track(y=y, sr=sr)
        val_a = float(np.atleast_1d(tempo_a)[0])
        if val_a > 0:
            estimates.append(_octave_correct(val_a))
    except Exception:
        pass

    # Method B: tempogram peak
    try:
        onset_env = librosa.onset.onset_strength(y=y, sr=sr)
        tempogram = librosa.feature.tempogram(
            onset_envelope=onset_env, sr=sr,
        )
        avg_tempogram = np.mean(tempogram, axis=1)
        bpm_axis = librosa.tempo_frequencies(tempogram.shape[0], sr=sr)
        valid = (bpm_axis >= 30) & (bpm_axis <= 300)
        if np.any(valid):
            masked = avg_tempogram.copy()
            masked[~valid] = 0
            peak_idx = np.argmax(masked)
            val_b = float(bpm_axis[peak_idx])
            if val_b > 0:
                estimates.append(_octave_correct(val_b))
    except Exception:
        pass

    # Method C: onset autocorrelation
    try:
        onset_env = librosa.onset.onset_strength(y=y, sr=sr)
        ac = librosa.autocorrelate(onset_env, max_size=len(onset_env))
        hop = 512
        min_lag = int(60.0 * sr / (300.0 * hop))
        max_lag = int(60.0 * sr / (30.0 * hop))
        max_lag = min(max_lag, len(ac) - 1)
        if min_lag < max_lag and max_lag > 0:
            segment = ac[min_lag:max_lag + 1]
            peak_offset = np.argmax(segment)
            peak_lag = min_lag + peak_offset
            if peak_lag > 0:
                val_c = 60.0 * sr / (peak_lag * hop)
                if val_c > 0:
                    estimates.append(_octave_correct(val_c))
    except Exception:
        pass

    return estimates


def _bpm_consensus(estimates: list[float]) -> tuple[Optional[int], str]:
    """Find consensus BPM from a list of estimates + assign confidence."""
    if not estimates:
        return None, "low"

    estimates_arr = np.array(estimates)
    best_cluster: list[float] = []
    for ref in estimates_arr:
        cluster = [e for e in estimates_arr
                   if abs(e - ref) / max(ref, 1) < 0.08]
        if len(cluster) > len(best_cluster):
            best_cluster = cluster

    consensus = float(np.median(best_cluster)) if best_cluster else estimates[0]
    bpm = int(round(consensus))
    if bpm <= 0:
        return None, "low"

    n_agree = len(best_cluster)
    n_total = len(estimates)
    if n_total >= 6:
        # S-A-S thresholds (many data points)
        if n_agree / n_total >= 0.7:
            confidence = "high"
        elif n_agree / n_total >= 0.4:
            confidence = "medium"
        else:
            confidence = "low"
    else:
        # mid thresholds
        if n_agree >= 3:
            confidence = "high"
        elif n_agree >= 2:
            confidence = "medium"
        else:
            confidence = "low"

    return bpm, confidence


def detect_bpm(
    audio_path: Path,
    *,
    mode: str = _DEFAULT_MODE,
    n_chunks: int = _SAS_NUM_CHUNKS,
) -> tuple[Optional[int], str]:
    """Detect BPM with quality level controlled by *mode*.

    - **faf**: Single ``beat_track`` on the audio + octave correction.
    - **mid**: 3-method ensemble (beat_track + tempogram + onset-AC).
    - **sas**: mid ensemble + PLP + multi-hop + chunked analysis.

    Returns ``(bpm, confidence)``.
    """
    try:
        import librosa

        y, sr = librosa.load(str(audio_path), sr=None, mono=True)
        y = _preprocess_audio(y, sr)

        # ── F-A-F: single method ──
        if mode == "faf":
            try:
                tempo, _ = librosa.beat.beat_track(y=y, sr=sr)
                val = float(np.atleast_1d(tempo)[0])
                if val > 0:
                    bpm = int(round(_octave_correct(val)))
                    logger.info("BPM faf: %d (raw: %.1f)", bpm, val)
                    return bpm, "low"
            except Exception:
                pass
            return None, "low"

        # ── mid: 3-method ensemble ──
        estimates = _bpm_core_ensemble(y, sr)

        # ── sas: additional techniques ──
        if mode == "sas":
            # PLP (Predominant Local Pulse)
            try:
                onset_env = librosa.onset.onset_strength(y=y, sr=sr)
                pulse = librosa.beat.plp(
                    onset_envelope=onset_env, sr=sr,
                )
                # PLP gives per-frame pulse strength; estimate tempo from
                # the autocorrelation of the pulse curve
                plp_ac = librosa.autocorrelate(pulse, max_size=len(pulse))
                hop = 512
                min_lag = int(60.0 * sr / (300.0 * hop))
                max_lag = int(60.0 * sr / (30.0 * hop))
                max_lag = min(max_lag, len(plp_ac) - 1)
                if min_lag < max_lag and max_lag > 0:
                    seg = plp_ac[min_lag:max_lag + 1]
                    peak_lag = min_lag + np.argmax(seg)
                    if peak_lag > 0:
                        plp_bpm = 60.0 * sr / (peak_lag * hop)
                        if plp_bpm > 0:
                            estimates.append(_octave_correct(plp_bpm))
            except Exception:
                pass

            # Multi-hop beat_track (256, 1024 — we already have 512 default)
            for extra_hop in (256, 1024):
                try:
                    tempo_h, _ = librosa.beat.beat_track(
                        y=y, sr=sr, hop_length=extra_hop,
                    )
                    val_h = float(np.atleast_1d(tempo_h)[0])
                    if val_h > 0:
                        estimates.append(_octave_correct(val_h))
                except Exception:
                    pass

            # Chunked ensemble
            chunks = _select_chunks(y, sr, n_chunks=n_chunks, use_onset=True)
            for chunk in chunks:
                chunk_estimates = _bpm_core_ensemble(chunk, sr)
                estimates.extend(chunk_estimates)

            # IBI stability — boost or lower confidence based on beat
            # interval variance (applied after consensus)
            try:
                _, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
                if beat_frames is not None and len(beat_frames) > 4:
                    beat_times = librosa.frames_to_time(beat_frames, sr=sr)
                    ibis = np.diff(beat_times)
                    ibi_cv = float(np.std(ibis) / (np.mean(ibis) + 1e-10))
                    # cv < 0.1 = very stable, > 0.3 = unstable
                    # Stored for confidence adjustment below
                else:
                    ibi_cv = 0.5
            except Exception:
                ibi_cv = 0.5

        bpm, confidence = _bpm_consensus(estimates)

        # sas: IBI stability can upgrade medium→high or downgrade
        if mode == "sas" and bpm is not None:
            if ibi_cv < 0.10 and confidence == "medium":
                confidence = "high"
            elif ibi_cv > 0.30 and confidence == "high":
                confidence = "medium"

        logger.info(
            "BPM [%s]: %s (estimates=%s, conf=%s)",
            mode, bpm,
            [round(e, 1) for e in estimates[:10]],  # cap log length
            confidence,
        )
        return bpm, confidence

    except Exception as exc:
        logger.warning("BPM detection failed: %s", exc)
        return None, "low"


# ── Step 3: Time signature detection (librosa) ────────────────────


def _timesig_core_scores(
    y: np.ndarray, sr: int,
) -> dict[str, float]:
    """Compute the 3-signal time-signature scores on a single buffer.

    Returns a dict mapping signature labels to raw scores.
    """
    import librosa

    scores: dict[str, float] = {}

    tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
    if beat_frames is None or len(beat_frames) < 8:
        return scores

    onset_env = librosa.onset.onset_strength(y=y, sr=sr)
    beat_strengths = onset_env[beat_frames[beat_frames < len(onset_env)]]
    if len(beat_strengths) < 8:
        return scores

    # Signal 1: Accent pattern analysis
    for label, grouping in [("3/4", 3), ("4/4", 4), ("6/8", 6)]:
        if len(beat_strengths) < grouping * 2:
            scores[label] = 0.0
            continue
        usable = len(beat_strengths) - (len(beat_strengths) % grouping)
        grouped = beat_strengths[:usable].reshape(-1, grouping)
        downbeat_mean = float(np.mean(grouped[:, 0]))
        offbeat_mean = float(np.mean(grouped[:, 1:]))
        contrast = downbeat_mean / offbeat_mean if offbeat_mean > 0 else 1.0
        scores[label] = contrast

    # Signal 2: Autocorrelation at meter periods
    hop = 512
    beat_times = librosa.frames_to_time(beat_frames, sr=sr)
    intervals = np.diff(beat_times)
    if len(intervals) > 0:
        median_interval = float(np.median(intervals))
        beat_period = int(round(median_interval * sr / hop))
        if beat_period > 0:
            ac = librosa.autocorrelate(onset_env, max_size=len(onset_env))
            for label, mult in [("3/4", 3), ("4/4", 4), ("6/8", 6)]:
                period = beat_period * mult
                if period < len(ac):
                    lo = max(0, period - 2)
                    hi = min(len(ac), period + 3)
                    ac_score = float(np.mean(ac[lo:hi]))
                    if ac[0] > 0:
                        ac_score /= float(ac[0])
                    scores[label] = scores.get(label, 0.0) + ac_score

    # Signal 3: Beat-strength variance ratio
    for label, grouping in [("3/4", 3), ("4/4", 4)]:
        usable = len(beat_strengths) - (len(beat_strengths) % grouping)
        if usable >= grouping * 2:
            grouped = beat_strengths[:usable].reshape(-1, grouping)
            row_vars = np.var(grouped, axis=1)
            scores[label] = scores.get(label, 0.0) + float(np.mean(row_vars))

    return scores


def detect_time_signature(
    audio_path: Path,
    *,
    mode: str = _DEFAULT_MODE,
    n_chunks: int = _SAS_NUM_CHUNKS,
) -> tuple[Optional[str], str]:
    """Estimate time signature with quality controlled by *mode*.

    - **faf**: Hardcoded ``"4/4"`` (correct ~80%+ of the time).
    - **mid**: Beat-sync accent + AC + variance + 4/4 prior.
    - **sas**: mid signals + PLP periodicity + multi-band onset +
      tempogram harmonic ratios + chunked voting.

    Returns ``(signature, confidence)``.
    """
    if mode == "faf":
        return "4/4", "low"

    try:
        import librosa

        y, sr = librosa.load(str(audio_path), sr=None, mono=True)
        y = _preprocess_audio(y, sr)

        # ── mid: core 3-signal scoring ──
        scores = _timesig_core_scores(y, sr)

        # ── sas: additional techniques ──
        if mode == "sas":
            onset_env = librosa.onset.onset_strength(y=y, sr=sr)

            # PLP periodicity
            try:
                pulse = librosa.beat.plp(onset_envelope=onset_env, sr=sr)
                plp_ac = librosa.autocorrelate(pulse, max_size=len(pulse))
                # Find the beat period from PLP
                tempo_est, _ = librosa.beat.beat_track(y=y, sr=sr)
                tempo_val = float(np.atleast_1d(tempo_est)[0])
                if tempo_val > 0:
                    hop = 512
                    bp = int(round(60.0 / tempo_val * sr / hop))
                    if bp > 0:
                        for label, mult in [("3/4", 3), ("4/4", 4), ("6/8", 6)]:
                            lag = bp * mult
                            if lag < len(plp_ac):
                                lo = max(0, lag - 2)
                                hi = min(len(plp_ac), lag + 3)
                                s = float(np.mean(plp_ac[lo:hi]))
                                if plp_ac[0] > 0:
                                    s /= float(plp_ac[0])
                                scores[label] = scores.get(label, 0.0) + s
            except Exception:
                pass

            # Multi-band onset analysis (low/mid/high)
            try:
                S = np.abs(librosa.stft(y))
                n_bins = S.shape[0]
                third = n_bins // 3
                bands = {
                    "low": S[:third, :],
                    "mid": S[third:2*third, :],
                    "high": S[2*third:, :],
                }
                for band_name, band_S in bands.items():
                    band_onset = librosa.onset.onset_strength(S=band_S, sr=sr)
                    band_ac = librosa.autocorrelate(
                        band_onset, max_size=len(band_onset),
                    )
                    tempo_val2 = float(np.atleast_1d(tempo_est)[0])
                    if tempo_val2 > 0:
                        hop = 512
                        bp2 = int(round(60.0 / tempo_val2 * sr / hop))
                        if bp2 > 0 and band_ac[0] > 0:
                            for label, mult in [("3/4", 3), ("4/4", 4)]:
                                lag = bp2 * mult
                                if lag < len(band_ac):
                                    lo = max(0, lag - 2)
                                    hi = min(len(band_ac), lag + 3)
                                    s = float(np.mean(band_ac[lo:hi]))
                                    s /= float(band_ac[0])
                                    # Weight: low band (kick) counts more
                                    w = 1.5 if band_name == "low" else 1.0
                                    scores[label] = scores.get(label, 0.0) + s * w
            except Exception:
                pass

            # Tempogram harmonic ratios
            try:
                tempogram = librosa.feature.tempogram(
                    onset_envelope=onset_env, sr=sr,
                )
                avg_tg = np.mean(tempogram, axis=1)
                bpm_axis = librosa.tempo_frequencies(tempogram.shape[0], sr=sr)
                if tempo_val > 0:
                    # Find energy at 1×, 2×, 3× tempo
                    for mult_label, t_mult in [("duple", 2.0), ("triple", 3.0)]:
                        target_bpm = tempo_val * t_mult
                        if target_bpm < 300:
                            idx = np.argmin(np.abs(bpm_axis - target_bpm))
                            energy = float(avg_tg[idx])
                            base_idx = np.argmin(np.abs(bpm_axis - tempo_val))
                            base_energy = float(avg_tg[base_idx]) + 1e-10
                            ratio = energy / base_energy
                            if t_mult == 2.0:
                                scores["4/4"] = scores.get("4/4", 0.0) + ratio
                            else:
                                scores["3/4"] = scores.get("3/4", 0.0) + ratio
            except Exception:
                pass

            # Chunked voting
            chunks = _select_chunks(y, sr, n_chunks=n_chunks, use_onset=True)
            chunk_votes: list[str] = []
            for chunk in chunks:
                cs = _timesig_core_scores(chunk, sr)
                if cs:
                    cs["4/4"] = cs.get("4/4", 0.0) * 1.15
                    best_c = max(cs, key=cs.get)
                    chunk_votes.append(best_c)
            # Add chunk votes as score increments
            for vote in chunk_votes:
                scores[vote] = scores.get(vote, 0.0) + 1.0

        # ── Bayesian prior: bias toward 4/4 ──
        _PRIOR_BOOST = 1.15
        scores["4/4"] = scores.get("4/4", 0.0) * _PRIOR_BOOST

        if not scores:
            return "4/4", "low"

        best = max(scores, key=scores.get)

        # Confidence: margin between top 2
        sorted_scores = sorted(scores.values(), reverse=True)
        if len(sorted_scores) >= 2 and sorted_scores[1] > 0:
            margin = sorted_scores[0] / sorted_scores[1]
        else:
            margin = 1.0

        if margin > 1.4:
            confidence = "high"
        elif margin > 1.15:
            confidence = "medium"
        else:
            confidence = "low"

        logger.info(
            "TimeSig [%s]: %s (scores=%s, margin=%.2f, conf=%s)",
            mode, best,
            {k: round(v, 3) for k, v in scores.items()},
            margin, confidence,
        )
        return best, confidence

    except Exception as exc:
        logger.warning("Time signature detection failed: %s", exc)
        return "4/4", "low"


# ── Step 4: Key detection (librosa + Krumhansl-Schmuckler) ────────


def _best_key_for_profile(
    chroma_avg: np.ndarray,
    major_profile: np.ndarray,
    minor_profile: np.ndarray,
) -> tuple[str, float]:
    """Find the best key match for a single profile family.

    Returns ``(key_label, correlation)``.
    """
    major_norm = major_profile / major_profile.sum()
    minor_norm = minor_profile / minor_profile.sum()

    best_corr = -2.0
    best_key = "C major"

    for shift in range(12):
        rotated = np.roll(chroma_avg, -shift)

        corr_maj = float(np.corrcoef(rotated, major_norm)[0, 1])
        if corr_maj > best_corr:
            best_corr = corr_maj
            best_key = f"{_PITCH_CLASSES[shift]} major"

        corr_min = float(np.corrcoef(rotated, minor_norm)[0, 1])
        if corr_min > best_corr:
            best_corr = corr_min
            best_key = f"{_PITCH_CLASSES[shift]} minor"

    return best_key, best_corr


def _key_votes_from_chroma(
    chroma_avg: np.ndarray,
    profiles: dict | None = None,
) -> list[tuple[str, float]]:
    """Vote on key from a single chroma vector using specified profiles.

    Returns list of ``(key_label, correlation)`` — one per profile family.
    If *profiles* is None, uses all ``_KEY_PROFILES``.
    """
    if profiles is None:
        profiles = _KEY_PROFILES

    results: list[tuple[str, float]] = []
    for name, pf in profiles.items():
        key_label, corr = _best_key_for_profile(
            chroma_avg, pf["major"], pf["minor"],
        )
        results.append((key_label, corr))
    return results


def _energy_weighted_chroma(
    chroma: np.ndarray,
    y_harmonic: np.ndarray,
) -> Optional[np.ndarray]:
    """Compute an energy-weighted average chroma vector."""
    import librosa

    rms = librosa.feature.rms(y=y_harmonic, frame_length=2048, hop_length=512)
    rms_vec = rms[0]
    min_len = min(chroma.shape[1], len(rms_vec))
    chroma = chroma[:, :min_len]
    rms_vec = rms_vec[:min_len]

    weights = rms_vec / (rms_vec.sum() + 1e-10)
    chroma_avg = (chroma * weights[np.newaxis, :]).sum(axis=1)

    s = chroma_avg.sum()
    if s == 0:
        return None
    return chroma_avg / s


def detect_key(
    audio_path: Path,
    *,
    mode: str = _DEFAULT_MODE,
    n_chunks: int = _SAS_NUM_CHUNKS,
) -> tuple[Optional[str], str]:
    """Detect musical key with quality controlled by *mode*.

    - **faf**: Single Krumhansl profile on ``chroma_cens`` of
      ``harmonic(mix)`` — no segments.
    - **mid**: 3-profile × energy-weighted ``chroma_cens`` × 8 s
      segment voting.
    - **sas**: mid + multi-chroma fusion (cens/cqt/stft) + tonnetz
      disambiguation + tuning correction + ending resolution +
      energy-gated chunked voting.

    Returns ``(key, confidence)``.
    """
    try:
        import librosa
        from collections import Counter

        y, sr = librosa.load(str(audio_path), sr=None, mono=True)
        y = _preprocess_audio(y, sr)

        # Harmonic enhancement
        margin = 4.0 if mode != "faf" else 2.0
        y_harmonic = librosa.effects.harmonic(y, margin=margin)

        # ── sas: tuning correction ──
        tuning = 0.0
        if mode == "sas":
            try:
                tuning = float(librosa.estimate_tuning(y=y_harmonic, sr=sr))
                logger.debug("Tuning correction: %.3f bins", tuning)
            except Exception:
                tuning = 0.0

        # ── F-A-F: single chroma, single profile ──
        if mode == "faf":
            chroma = librosa.feature.chroma_cens(y=y_harmonic, sr=sr)
            chroma_avg = _energy_weighted_chroma(chroma, y_harmonic)
            if chroma_avg is None:
                return None, "low"
            kr = _KEY_PROFILES["krumhansl"]
            key_label, corr = _best_key_for_profile(
                chroma_avg, kr["major"], kr["minor"],
            )
            logger.info("Key faf: %s (corr=%.3f)", key_label, corr)
            return key_label, "low"

        # ── mid / sas: multi-profile voting ──
        all_votes: list[str] = []
        all_weights: list[float] = []

        # Determine which chroma types to use
        if mode == "sas":
            chroma_types = {
                "cens": lambda: librosa.feature.chroma_cens(
                    y=y_harmonic, sr=sr, tuning=tuning,
                ),
                "cqt": lambda: librosa.feature.chroma_cqt(
                    y=y_harmonic, sr=sr, tuning=tuning,
                ),
                "stft": lambda: librosa.feature.chroma_stft(
                    y=y_harmonic, sr=sr, tuning=tuning,
                ),
            }
        else:
            chroma_types = {
                "cens": lambda: librosa.feature.chroma_cens(
                    y=y_harmonic, sr=sr,
                ),
            }

        for chroma_name, chroma_fn in chroma_types.items():
            try:
                chroma = chroma_fn()
            except Exception:
                continue

            chroma_avg = _energy_weighted_chroma(chroma, y_harmonic)
            if chroma_avg is None:
                continue

            # Global multi-profile vote
            for key_label, corr in _key_votes_from_chroma(chroma_avg):
                all_votes.append(key_label)
                all_weights.append(1.0)

            # Segment-based voting
            rms = librosa.feature.rms(
                y=y_harmonic, frame_length=2048, hop_length=512,
            )
            rms_vec = rms[0]
            min_len = min(chroma.shape[1], len(rms_vec))
            chroma_s = chroma[:, :min_len]
            rms_s = rms_vec[:min_len]

            seg_frames = int(8.0 * sr / 512)
            n_segments = max(1, chroma_s.shape[1] // seg_frames)

            for seg_i in range(n_segments):
                start = seg_i * seg_frames
                end = min(start + seg_frames, chroma_s.shape[1])
                seg_chroma = chroma_s[:, start:end]
                seg_w = rms_s[start:end]

                w_sum = seg_w.sum()
                if w_sum < 1e-10:
                    continue

                seg_w_norm = seg_w / w_sum
                seg_avg = (seg_chroma * seg_w_norm[np.newaxis, :]).sum(axis=1)
                s = seg_avg.sum()
                if s < 1e-10:
                    continue
                seg_avg = seg_avg / s

                for key_label, _ in _key_votes_from_chroma(seg_avg):
                    all_votes.append(key_label)
                    all_weights.append(1.0)

        # ── sas-only extras ──
        if mode == "sas":
            # Tonnetz — weighted vote for major/minor disambiguation
            try:
                tonnetz = librosa.feature.tonnetz(y=y_harmonic, sr=sr)
                tonnetz_avg = np.mean(tonnetz, axis=1)
                # Dimensions 4-5 are major-third projections,
                # dimensions 2-3 are minor-third projections
                major_energy = float(np.sum(tonnetz_avg[4:6] ** 2))
                minor_energy = float(np.sum(tonnetz_avg[2:4] ** 2))
                # Weight factor: how strongly tonnetz leans major vs minor
                tonnetz_ratio = major_energy / (minor_energy + 1e-10)

                # Get the current leading vote to check if tonnetz agrees
                if all_votes:
                    temp_counts = Counter(all_votes)
                    leader = temp_counts.most_common(1)[0][0]
                    leader_is_major = "major" in leader
                    tonnetz_says_major = tonnetz_ratio > 1.0

                    if leader_is_major == tonnetz_says_major:
                        # Agreement — add weighted votes for the leader
                        all_votes.extend([leader] * 3)
                        all_weights.extend([1.5] * 3)
                    else:
                        # Disagreement — add votes for the relative key
                        # with moderate weight
                        alt_mode = "minor" if leader_is_major else "major"
                        # Find the best key of the other mode from chroma
                        chroma_cens = librosa.feature.chroma_cens(
                            y=y_harmonic, sr=sr, tuning=tuning,
                        )
                        ca = _energy_weighted_chroma(chroma_cens, y_harmonic)
                        if ca is not None:
                            for name, pf in _KEY_PROFILES.items():
                                prof = pf[alt_mode]
                                prof_norm = prof / prof.sum()
                                best_corr = -2.0
                                best_k = ""
                                for shift in range(12):
                                    rotated = np.roll(ca, -shift)
                                    c = float(np.corrcoef(rotated, prof_norm)[0, 1])
                                    if c > best_corr:
                                        best_corr = c
                                        best_k = f"{_PITCH_CLASSES[shift]} {alt_mode}"
                                if best_k:
                                    all_votes.append(best_k)
                                    all_weights.append(1.0)
            except Exception:
                pass

            # Ending resolution — last ~5 s weighted extra
            try:
                end_samples = min(int(5.0 * sr), len(y_harmonic))
                y_end = y_harmonic[-end_samples:]
                chroma_end = librosa.feature.chroma_cens(
                    y=y_end, sr=sr, tuning=tuning,
                )
                end_avg = np.mean(chroma_end, axis=1)
                s = end_avg.sum()
                if s > 1e-10:
                    end_avg = end_avg / s
                    for key_label, _ in _key_votes_from_chroma(end_avg):
                        all_votes.append(key_label)
                        all_weights.append(2.0)  # ending gets double weight
            except Exception:
                pass

            # Chunked voting (RMS-gated for harmonic content)
            chunks = _select_chunks(
                y_harmonic, sr, n_chunks=n_chunks, use_onset=False,
            )
            for chunk in chunks:
                try:
                    ch_chroma = librosa.feature.chroma_cens(
                        y=chunk, sr=sr, tuning=tuning,
                    )
                    ch_avg = _energy_weighted_chroma(ch_chroma, chunk)
                    if ch_avg is not None:
                        for key_label, _ in _key_votes_from_chroma(ch_avg):
                            all_votes.append(key_label)
                            all_weights.append(1.0)
                except Exception:
                    pass

        # ── Final vote ──
        if not all_votes:
            return None, "low"

        # Weighted majority vote
        weighted_counts: dict[str, float] = {}
        for vote, w in zip(all_votes, all_weights):
            weighted_counts[vote] = weighted_counts.get(vote, 0.0) + w

        best_key = max(weighted_counts, key=weighted_counts.get)
        total_weight = sum(all_weights)
        best_weight = weighted_counts[best_key]
        share = best_weight / total_weight

        if share >= 0.55:
            confidence = "high"
        elif share >= 0.35:
            confidence = "medium"
        else:
            confidence = "low"

        logger.info(
            "Key [%s]: %s (share=%.0f%%, votes=%d, conf=%s)",
            mode, best_key, share * 100, len(all_votes), confidence,
        )
        return best_key, confidence

    except Exception as exc:
        logger.warning("Key detection failed: %s", exc)
        return None, "low"


# ── Orchestrator ──────────────────────────────────────────────────


def analyze_audio(
    audio_path: Path,
    *,
    device: str = "auto",
    mode: str = _DEFAULT_MODE,
    n_chunks: int = _SAS_NUM_CHUNKS,
) -> Dict[str, Any]:
    """Run the full audio analysis pipeline on a single file.

    **faf** — skip Demucs, analyse raw mix (~2-3 s).
    **mid** — Demucs stems → ensemble (~10-12 s GPU).
    **sas** — Demucs stems → deep multi-technique analysis (~18-30 s GPU).

    Args:
        audio_path: Path to the input audio file.
        device: Torch device (auto, cuda, cpu, mps).
        mode: Quality tier — ``"faf"``, ``"mid"``, or ``"sas"``.
        n_chunks: Number of analysis chunks for S-A-S (override).

    Returns:
        Dict with ``bpm``, ``key``, ``signature`` string values
        (for sidecar writing) and ``confidence`` sub-dict with
        per-field confidence levels (GUI-only, not persisted).
    """
    if mode not in MODES:
        logger.warning("Unknown mode '%s', falling back to '%s'", mode, _DEFAULT_MODE)
        mode = _DEFAULT_MODE

    tmp_dir = Path(tempfile.mkdtemp(prefix="sidestep_analysis_"))
    result: Dict[str, Any] = {}
    confidence: Dict[str, str] = {}

    try:
        logger.info(
            "Starting audio analysis [%s] for %s", mode, audio_path.name,
        )

        # ── Determine audio paths for detectors ──
        if mode == "faf":
            # No Demucs — detectors run on raw mix
            drums_path = audio_path
            harmonics_path = audio_path
        else:
            # Demucs stem separation (mid / sas)
            drums_path, harmonics_path = separate_stems(
                audio_path, tmp_dir, device=device,
            )

        # BPM (runs on drums stem, or raw mix for faf)
        bpm, bpm_conf = detect_bpm(
            drums_path, mode=mode, n_chunks=n_chunks,
        )
        if bpm is not None:
            result["bpm"] = str(bpm)
            confidence["bpm"] = bpm_conf

        # Time signature (runs on drums stem, or raw mix for faf)
        signature, sig_conf = detect_time_signature(
            drums_path, mode=mode, n_chunks=n_chunks,
        )
        if signature:
            result["signature"] = signature
            confidence["signature"] = sig_conf

        # Key (runs on harmonics stem, or raw mix for faf)
        key, key_conf = detect_key(
            harmonics_path, mode=mode, n_chunks=n_chunks,
        )
        if key:
            result["key"] = key
            confidence["key"] = key_conf

        result["confidence"] = confidence
        logger.info("Analysis complete for %s: %s", audio_path.name, result)

    except Exception as exc:
        logger.error("Audio analysis failed for %s: %s", audio_path.name, exc)
        raise

    finally:
        try:
            shutil.rmtree(tmp_dir)
        except OSError as exc:
            logger.debug("Could not clean temp dir %s: %s", tmp_dir, exc)

    return result


def analyze_audio_safe(
    audio_path: Path,
    *,
    device: str = "auto",
) -> Dict[str, Any]:
    """Wrapper around :func:`analyze_audio` that catches exceptions.

    Returns a dict with ``status`` (``"ok"`` or ``"failed"``),
    the analysis ``fields``, and an optional ``error`` message.
    """
    try:
        fields = analyze_audio(audio_path, device=device)
        return {"status": "ok", "fields": fields}
    except Exception as exc:
        return {"status": "failed", "fields": {}, "error": str(exc)}
