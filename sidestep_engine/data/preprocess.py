"""
Two-Pass CLI Preprocessing for ACE-Step Training V2.

Converts raw audio files into ``.pt`` tensor files compatible with
``PreprocessedDataModule``.  Uses upstream sub-functions directly and
loads models **sequentially** to minimise peak VRAM:

    Pass 1 (Light ~3 GB):  VAE + Text Encoder  -> intermediate ``.tmp.pt``
    Pass 2 (Heavy ~6 GB):  DIT encoder          -> final ``.pt``

Input modes:
    * With ``--dataset-json``: rich per-sample metadata (lyrics, genre, BPM, …)
    * Without JSON: scan directory, default to ``[Instrumental]``, filename caption
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import torch

# Split-out helpers
from sidestep_engine.data.preprocess_discovery import (
    discover_audio_files as _discover_audio_files,
    load_dataset_metadata as _load_dataset_metadata,
    load_sample_metadata as _load_sample_metadata,
    safe_output_stem as _safe_output_stem,
    select_genre_indices as _select_genre_indices,
)
from sidestep_engine.data.preprocess_prompt import (
    build_simple_prompt as _build_simple_prompt,
)
from sidestep_engine.data.preprocess_vae import (
    TARGET_SR as _TARGET_SR,
    tiled_vae_encode as _tiled_vae_encode,
)
from sidestep_engine.data.audio_duration import detect_max_duration as _detect_max_duration
from sidestep_engine.data.audio_normalize import normalize_audio as _normalize_audio

logger = logging.getLogger(__name__)


def _compute_pt_checksums(out_path: Path) -> Dict[str, str]:
    """Compute MD5 checksums for all .pt files in the output directory."""
    checksums: Dict[str, str] = {}
    for pt_file in sorted(out_path.glob("*.pt")):
        try:
            h = hashlib.md5()
            with open(pt_file, "rb") as f:
                for chunk in iter(lambda: f.read(1 << 20), b""):
                    h.update(chunk)
            checksums[pt_file.name] = h.hexdigest()
        except OSError:
            pass
    return checksums


def _write_preprocess_meta(
    out_path: Path,
    *,
    audio_dir: Optional[str],
    dataset_json: Optional[str],
    variant: str,
    normalize: str,
    target_db: float,
    target_lufs: float,
    custom_tag: str,
    tag_position: str,
    genre_ratio: int,
    total: int,
    processed: int,
    failed: int,
) -> None:
    """Persist preprocess dataset metadata for GUI lineage/UX display."""
    meta_path = out_path / "preprocess_meta.json"
    checksums = _compute_pt_checksums(out_path)
    payload: Dict[str, Any] = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "audio_dir": audio_dir or "",
        "source_audio_dir": audio_dir or "",
        "dataset_json": dataset_json or "",
        "model_variant": variant,
        "normalize": normalize,
        "target_db": target_db,
        "target_lufs": target_lufs,
        "custom_tag": custom_tag,
        "tag_position": tag_position,
        "genre_ratio": genre_ratio,
        "total": total,
        "processed": processed,
        "failed": failed,
        "checksums": checksums,
    }
    try:
        meta_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    except Exception as exc:
        logger.debug("Could not write preprocess_meta.json: %s", exc)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def preprocess_audio_files(
    audio_dir: Optional[str],
    output_dir: str,
    checkpoint_dir: str,
    variant: str = "turbo",
    max_duration: float = 0,
    dataset_json: Optional[str] = None,
    device: str = "auto",
    precision: str = "auto",
    normalize: str = "none",
    target_db: float = -1.0,
    target_lufs: float = -14.0,
    progress_callback: Optional[Callable] = None,
    cancel_check: Optional[Callable] = None,
    custom_tag: str = "",
    tag_position: str = "",
    genre_ratio: int = 0,
) -> Dict[str, Any]:
    """Preprocess audio files into .pt tensor format (two-pass pipeline).

    Audio files are discovered from one of two sources:

    * **Dataset JSON** (preferred): each entry's ``audio_path`` or
      ``filename`` field locates the audio file directly.
    * **Audio directory** (fallback): scanned **recursively** for
      supported audio formats when no JSON is provided.

    The resulting tensors are adapter-agnostic: they work for both LoRA
    and LoKR training (the adapter type only affects weight injection,
    not the data pipeline).

    Args:
        audio_dir: Directory containing audio files (scanned recursively).
            May be ``None`` when *dataset_json* provides audio paths.
        output_dir: Directory for output .pt files.
        checkpoint_dir: Path to ACE-Step model checkpoints.
        variant: Model variant (turbo, base, sft).
        max_duration: Maximum audio duration in seconds (0 = auto-detect).
        dataset_json: Optional JSON file with per-sample metadata and
            audio paths.
        device: Target device (``"auto"`` to auto-detect).
        precision: Target precision (``"auto"`` to auto-detect).
        normalize: Audio normalization method (``"none"``, ``"peak"``,
            or ``"lufs"``).
        target_db: Peak normalization target in dBFS (peak method only).
        target_lufs: LUFS normalization target (lufs method only).
        progress_callback: ``(current, total, message) -> None``.
        cancel_check: ``() -> bool`` -- return True to cancel.
        custom_tag: Trigger tag applied as fallback to samples without one.
            Overrides the dataset-JSON ``custom_tag`` when non-empty.
        tag_position: Where to place the tag (``"prepend"``, ``"append"``,
            ``"replace"``).  Overrides the dataset-JSON ``tag_position``
            when non-empty.

    Returns:
        Dict with keys: ``processed``, ``failed``, ``total``, ``output_dir``.
    """
    from sidestep_engine.models.gpu_utils import detect_gpu

    gpu = detect_gpu(device, precision)
    dev = gpu.device
    prec = gpu.precision

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    # Clean up orphaned staging files from a previous interrupted run
    for orphan in out_path.glob("*.__writing__"):
        try:
            orphan.unlink()
            logger.info("[Side-Step] Removed orphaned staging file: %s", orphan.name)
        except OSError:
            pass

    # -- Discover audio files -----------------------------------------------
    audio_files = _discover_audio_files(audio_dir, dataset_json)
    if not audio_files:
        logger.warning("[Side-Step] No audio files found")
        return {"processed": 0, "failed": 0, "total": 0, "output_dir": str(out_path)}

    total = len(audio_files)
    logger.info("[Side-Step] Found %d audio files to preprocess", total)

    # -- Auto-detect max duration when requested (0 = auto) ------------------
    if max_duration <= 0:
        detected = _detect_max_duration(audio_files)
        max_duration = float(detected) if detected > 0 else 240.0
        logger.info(
            "[Side-Step] Auto-detected max_duration: %ds", int(max_duration),
        )

    # -- Load metadata -------------------------------------------------------
    sample_meta = _load_sample_metadata(dataset_json, audio_files)
    ds_meta = _load_dataset_metadata(dataset_json)

    # Caller-supplied overrides for JSON-level metadata
    if custom_tag:
        ds_meta["custom_tag"] = custom_tag
    if tag_position:
        ds_meta["tag_position"] = tag_position
    if genre_ratio > 0:
        ds_meta["genre_ratio"] = genre_ratio

    # Apply dataset-level custom_tag as fallback for samples without one
    ds_tag = ds_meta.get("custom_tag", "")
    if ds_tag:
        for sm in sample_meta.values():
            if not sm.get("custom_tag"):
                sm["custom_tag"] = ds_tag

    # -- Pass 1: VAE + Text Encoder -----------------------------------------
    intermediates, pass1_failed = _pass1_light(
        audio_files=audio_files,
        sample_meta=sample_meta,
        ds_meta=ds_meta,
        out_path=out_path,
        checkpoint_dir=checkpoint_dir,
        variant=variant,
        device=dev,
        precision=prec,
        max_duration=max_duration,
        normalize=normalize,
        target_db=target_db,
        target_lufs=target_lufs,
        progress_callback=progress_callback,
        cancel_check=cancel_check,
        audio_dir=audio_dir,
    )

    # -- Pass 2: DIT Encoder ------------------------------------------------
    processed, pass2_failed = _pass2_heavy(
        intermediates=intermediates,
        out_path=out_path,
        checkpoint_dir=checkpoint_dir,
        variant=variant,
        device=dev,
        precision=prec,
        progress_callback=progress_callback,
        cancel_check=cancel_check,
    )

    failed = pass1_failed + pass2_failed
    _write_preprocess_meta(
        out_path,
        audio_dir=audio_dir,
        dataset_json=dataset_json,
        variant=variant,
        normalize=normalize,
        target_db=target_db,
        target_lufs=target_lufs,
        custom_tag=custom_tag,
        tag_position=tag_position,
        genre_ratio=genre_ratio,
        total=total,
        processed=processed,
        failed=failed,
    )
    result = {
        "processed": processed,
        "failed": failed,
        "total": total,
        "output_dir": str(out_path),
    }
    logger.info(
        "[Side-Step] Preprocessing complete: %d/%d processed, %d failed",
        processed, total, failed,
    )
    return result


# ---------------------------------------------------------------------------
# Pass 1 -- Light models (VAE + Text Encoder)
# ---------------------------------------------------------------------------

def _pass1_light(
    audio_files: List[Path],
    sample_meta: Dict[str, Dict[str, Any]],
    ds_meta: Dict[str, Any],
    out_path: Path,
    checkpoint_dir: str,
    variant: str,
    device: str,
    precision: str,
    max_duration: float,
    normalize: str = "none",
    target_db: float = -1.0,
    target_lufs: float = -14.0,
    progress_callback: Optional[Callable] = None,
    cancel_check: Optional[Callable] = None,
    audio_dir: Optional[str] = None,
) -> tuple[List[Path], int]:
    """Load audio, VAE-encode, text-encode, save intermediates.

    Args:
        ds_meta: Dataset-level metadata (``tag_position``, ``genre_ratio``,
            ``custom_tag``) from the JSON's top-level ``metadata`` block.
        normalize: Audio normalization method (``"none"``, ``"peak"``,
            ``"lufs"``).

    Returns ``(list_of_intermediate_paths, fail_count)``.
    """
    from sidestep_engine.models.loader import (
        load_vae,
        load_text_encoder,
        load_silence_latent,
        unload_models,
        _resolve_dtype,
    )
    from sidestep_engine.vendor.preprocess_audio import load_audio_stereo
    from sidestep_engine.vendor.preprocess_text import encode_text
    from sidestep_engine.vendor.preprocess_lyrics import encode_lyrics

    dtype = _resolve_dtype(precision)

    logger.info("[Side-Step] Pass 1/2: Loading VAE + Text Encoder ...")
    vae = load_vae(checkpoint_dir, device, precision)
    tokenizer, text_enc = load_text_encoder(checkpoint_dir, device, precision)
    silence_latent = load_silence_latent(checkpoint_dir, device, precision, variant=variant)

    intermediates: List[Path] = []
    failed = 0
    total = len(audio_files)

    # Dataset-level prompt settings from ACE-Step's metadata block
    tag_position = ds_meta.get("tag_position", "prepend")
    genre_ratio = ds_meta.get("genre_ratio", 0)
    genre_indices = _select_genre_indices(total, genre_ratio)
    if genre_indices:
        logger.info(
            "[Side-Step] genre_ratio=%d%% -- %d/%d samples will use genre as prompt",
            genre_ratio, len(genre_indices), total,
        )
    if tag_position != "prepend":
        logger.info("[Side-Step] tag_position=%s (from dataset metadata)", tag_position)

    try:
        for i, af in enumerate(audio_files):
            if cancel_check and cancel_check():
                logger.info("[Side-Step] Cancelled at %d/%d", i, total)
                break

            # Collision-safe output stem (uses relative path for nested dirs)
            _stem = _safe_output_stem(af, audio_dir)

            # Skip if final .pt already exists (resumable)
            final_pt = out_path / f"{_stem}.pt"
            if final_pt.exists():
                logger.info("[Side-Step] Skipping (final exists): %s", af.name)
                continue

            try:
                # 1. Load audio (stereo, 48 kHz)
                audio, _sr = load_audio_stereo(str(af), _TARGET_SR, max_duration)

                # 1b. Optional normalization (CPU, before GPU transfer)
                if normalize != "none":
                    audio = _normalize_audio(
                        audio, _TARGET_SR, method=normalize,
                        target_db=target_db, target_lufs=target_lufs
                    )

                audio = audio.unsqueeze(0).to(device=device, dtype=vae.dtype)

                # 2. VAE encode (tiled for long audio)
                with torch.no_grad():
                    target_latents = _tiled_vae_encode(vae, audio, dtype)

                # Free raw audio immediately -- no longer needed after VAE encode
                del audio

                # Validate VAE output before saving
                if torch.isnan(target_latents).any() or torch.isinf(target_latents).any():
                    failed += 1
                    logger.warning(
                        "[Side-Step] Pass 1 SKIP (NaN/Inf in VAE latents): %s",
                        af.name,
                    )
                    del target_latents
                    continue

                latent_length = target_latents.shape[1]
                attention_mask = torch.ones(1, latent_length, device=device, dtype=dtype)

                # 3. Text encode (full-path key preferred for disambiguation)
                sm = sample_meta.get(str(af)) or sample_meta.get(af.name, {})
                caption = sm.get("caption", af.stem)
                lyrics = sm.get("lyrics", "[Instrumental]")

                # Build text prompt using dataset-level tag_position and genre_ratio
                use_genre = i in genre_indices
                text_prompt = _build_simple_prompt(sm, tag_position=tag_position, use_genre=use_genre)

                with torch.no_grad():
                    text_hs, text_mask = encode_text(text_enc, tokenizer, text_prompt, device, dtype)
                    lyric_hs, lyric_mask = encode_lyrics(text_enc, tokenizer, lyrics, device, dtype)

                # Validate text encoder outputs
                _bad_tensor = None
                for _tname, _tens in [("text_hs", text_hs), ("lyric_hs", lyric_hs)]:
                    if torch.isnan(_tens).any() or torch.isinf(_tens).any():
                        _bad_tensor = _tname
                        break
                if _bad_tensor is not None:
                    failed += 1
                    logger.warning(
                        "[Side-Step] Pass 1 SKIP (NaN/Inf in %s): %s",
                        _bad_tensor, af.name,
                    )
                    del target_latents, attention_mask, text_hs, text_mask
                    del lyric_hs, lyric_mask
                    continue

                # 4. Save intermediate
                tmp_path = out_path / f"{_stem}.tmp.pt"
                torch.save({
                    "target_latents": target_latents.squeeze(0).cpu(),
                    "attention_mask": attention_mask.squeeze(0).cpu(),
                    "text_hidden_states": text_hs.cpu(),
                    "text_attention_mask": text_mask.cpu(),
                    "lyric_hidden_states": lyric_hs.cpu(),
                    "lyric_attention_mask": lyric_mask.cpu(),
                    "silence_latent": silence_latent.cpu(),
                    "latent_length": latent_length,
                    "metadata": {
                        "audio_path": str(af),
                        "filename": af.name,
                        "caption": caption,
                        "lyrics": lyrics,
                        "duration": sm.get("duration", 0),
                        "bpm": sm.get("bpm"),
                        "keyscale": sm.get("keyscale", ""),
                        "timesignature": sm.get("timesignature", ""),
                        "genre": sm.get("genre", ""),
                        "is_instrumental": sm.get("is_instrumental", True),
                        "custom_tag": sm.get("custom_tag", ""),
                        "prompt_override": sm.get("prompt_override"),
                        "repeat": 1,  # Global dataset_repeats replaces per-sample repeat
                    },
                }, tmp_path)

                # Free GPU tensors from this iteration before the next one
                del target_latents, attention_mask, text_hs, text_mask
                del lyric_hs, lyric_mask
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

                intermediates.append(tmp_path)
                logger.debug("[Side-Step] Pass 1 OK: %s", af.name)
                if progress_callback:
                    progress_callback(i + 1, total, f"[Pass 1] OK: {af.name}")

            except Exception as exc:
                failed += 1
                logger.error("[Side-Step] Pass 1 FAIL %s: %s", af.name, exc)

    finally:
        logger.info("[Side-Step] Unloading VAE + Text Encoder ...")
        unload_models(vae, text_enc, tokenizer, silence_latent)

    if progress_callback:
        progress_callback(total, total, "[Pass 1] Done")

    return intermediates, failed


# ---------------------------------------------------------------------------
# Pass 2 -- Heavy model (DIT encoder)
# ---------------------------------------------------------------------------

def _pass2_heavy(
    intermediates: List[Path],
    out_path: Path,
    checkpoint_dir: str,
    variant: str,
    device: str,
    precision: str,
    progress_callback: Optional[Callable],
    cancel_check: Optional[Callable],
) -> tuple[int, int]:
    """Run DIT encoder on intermediates and write final .pt files.

    Returns ``(processed_count, fail_count)``.
    """
    if not intermediates:
        return 0, 0

    from sidestep_engine.models.loader import (
        load_decoder_for_training,
        unload_models,
        _resolve_dtype,
    )
    from sidestep_engine.vendor.preprocess_encoder import run_encoder
    from sidestep_engine.vendor.preprocess_context import build_context_latents

    dtype = _resolve_dtype(precision)

    logger.info("[Side-Step] Pass 2/2: Loading DIT model (variant=%s) ...", variant)
    model = load_decoder_for_training(checkpoint_dir, variant, device, precision)

    processed = 0
    failed = 0
    total = len(intermediates)

    try:
        for i, tmp_path in enumerate(intermediates):
            if cancel_check and cancel_check():
                logger.info("[Side-Step] Cancelled at %d/%d", i, total)
                break

            try:
                data = torch.load(str(tmp_path), weights_only=False)

                # Move tensors directly to model device/dtype (single .to()
                # avoids creating throwaway intermediate GPU copies).
                model_device = next(model.parameters()).device
                model_dtype = next(model.parameters()).dtype

                text_hs = data["text_hidden_states"].to(model_device, dtype=model_dtype)
                text_mask = data["text_attention_mask"].to(model_device, dtype=model_dtype)
                lyric_hs = data["lyric_hidden_states"].to(model_device, dtype=model_dtype)
                lyric_mask = data["lyric_attention_mask"].to(model_device, dtype=model_dtype)
                silence_latent = data["silence_latent"].to(model_device, dtype=model_dtype)
                latent_length = data["latent_length"]

                # DIT encoder pass (adapter-agnostic: same tensors for
                # LoRA and LoKR -- only the adapter injection differs).
                encoder_hs, encoder_mask = run_encoder(
                    model,
                    text_hidden_states=text_hs,
                    text_attention_mask=text_mask,
                    lyric_hidden_states=lyric_hs,
                    lyric_attention_mask=lyric_mask,
                    device=str(model_device),
                    dtype=model_dtype,
                )

                # Free encoder inputs immediately after use
                del text_hs, text_mask, lyric_hs, lyric_mask

                # Build context latents (silence-based, standard text2music)
                if silence_latent.dim() == 2:
                    silence_latent = silence_latent.unsqueeze(0)

                context_latents = build_context_latents(
                    silence_latent, latent_length, str(model_device), model_dtype,
                )
                del silence_latent

                # Validate DIT encoder / context outputs
                _bad_tensor = None
                for _tname, _tens in [
                    ("encoder_hidden_states", encoder_hs),
                    ("context_latents", context_latents),
                ]:
                    if torch.isnan(_tens).any() or torch.isinf(_tens).any():
                        _bad_tensor = _tname
                        break
                if _bad_tensor is not None:
                    failed += 1
                    logger.warning(
                        "[Side-Step] Pass 2 SKIP (NaN/Inf in %s): %s",
                        _bad_tensor, tmp_path.stem,
                    )
                    del encoder_hs, encoder_mask, context_latents, data
                    continue

                # Atomic write: save to staging path then os.replace to final
                base_name = tmp_path.name.replace(".tmp.pt", ".pt")
                final_path = out_path / base_name
                staging_path = out_path / (base_name + ".__writing__")
                meta = data["metadata"]
                torch.save({
                    "target_latents": data["target_latents"],
                    "attention_mask": data["attention_mask"],
                    "encoder_hidden_states": encoder_hs.squeeze(0).cpu(),
                    "encoder_attention_mask": encoder_mask.squeeze(0).cpu(),
                    "context_latents": context_latents.squeeze(0).cpu(),
                    "metadata": meta,
                }, staging_path)
                os.replace(staging_path, final_path)

                # Free all GPU tensors and the loaded data dict before next iter
                del encoder_hs, encoder_mask, context_latents, data
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

                # Remove intermediate
                tmp_path.unlink(missing_ok=True)

                processed += 1
                logger.debug("[Side-Step] Pass 2 OK: %s", tmp_path.stem)
                if progress_callback:
                    progress_callback(i + 1, total, f"[Pass 2] OK: {tmp_path.stem}")

            except Exception as exc:
                failed += 1
                logger.error("[Side-Step] Pass 2 FAIL %s: %s", tmp_path.stem, exc)

    finally:
        logger.info("[Side-Step] Unloading DIT model ...")
        unload_models(model)

    if progress_callback:
        progress_callback(total, total, "[Pass 2] Done")

    return processed, failed
