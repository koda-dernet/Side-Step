"""Extract ground-truth 5Hz audio-code sequences from a preprocessed dataset.

For each final ``.pt`` in the dataset dir, runs the resident FSQ audio
tokenizer (``model.tokenize``) over the song's real 25Hz latents to get
the exact 5Hz code sequence the LM would need to emit to "plan" that
song, and pairs it with the song's caption/lyrics/metas.  Output is one
JSONL row per song — the SFT dataset for the LM adapter.

"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, Optional

import torch

from sidestep_engine.lm.prompts import CODE_FPS_5, LATENT_FPS_25, apply_trigger_tag

logger = logging.getLogger(__name__)


def _dataset_tag_position(dataset_dir: Path) -> str:
    meta_json = dataset_dir / "preprocess_meta.json"
    if meta_json.is_file():
        try:
            return json.loads(meta_json.read_text(encoding="utf-8")).get("tag_position") or "prepend"
        except Exception:
            pass
    return "prepend"


def _flatten_indices(indices: torch.Tensor) -> list:
    """Normalize ResidualFSQ indices to a flat per-frame list of ints.

    Expected shapes: ``[1, T5]`` or ``[1, T5, 1]`` (single quantizer).
    Multiple quantizers would need an interleaving scheme the engine does
    not use — refuse loudly rather than corrupt silently.
    """
    t = indices
    if t.dim() == 3:
        if t.shape[-1] != 1:
            raise RuntimeError(
                f"ResidualFSQ returned {t.shape[-1]} quantizers per frame; "
                "the 5Hz LM vocabulary encodes exactly one code per frame."
            )
        t = t.squeeze(-1)
    if t.dim() != 2 or t.shape[0] != 1:
        raise RuntimeError(f"Unexpected indices shape {tuple(indices.shape)}")
    return [int(v) for v in t[0].tolist()]


def extract_codes(
    dataset_dir: str | Path,
    checkpoint_dir: str | Path,
    model_variant: str,
    out_path: str | Path,
    device: str = "cuda",
    precision: str = "bf16",
    progress: Optional[Any] = None,
    model: Any = None,
) -> Dict[str, Any]:
    """Run code extraction over every final ``.pt`` in *dataset_dir*.

    Pass a live ``model`` to reuse an already-loaded
    ``AceStepConditionGenerationModel`` (e.g. when chained around DiT
    training); otherwise the full model is loaded and freed here.

    Returns a summary dict: ``{"rows": N, "skipped": M, "out": path}``.
    """
    from sidestep_engine.models.loader import load_silence_latent, _resolve_dtype

    dataset_dir = Path(dataset_dir)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    pt_files = sorted(p for p in dataset_dir.glob("*.pt") if not p.name.endswith(".tmp.pt"))
    if not pt_files:
        raise RuntimeError(f"No .pt tensor files found in {dataset_dir}")

    dtype = _resolve_dtype(precision)
    tag_position = _dataset_tag_position(dataset_dir)

    owns_model = model is None
    if owns_model:
        from sidestep_engine.models.loader import load_decoder_for_training
        logger.info("[LM-Extract] Loading model (%s, %s, %s) ...", model_variant, device, precision)
        model = load_decoder_for_training(checkpoint_dir, model_variant, device, precision)

    silence = load_silence_latent(checkpoint_dir, device, precision, variant=model_variant)

    rows = 0
    skipped = 0
    t0 = time.time()
    try:
        with out_path.open("w", encoding="utf-8") as fh:
            for i, pt in enumerate(pt_files):
                try:
                    data = torch.load(str(pt), map_location="cpu", weights_only=False)
                    meta = dict(data.get("metadata") or {})
                    lat = data["target_latents"]
                    del data
                    if lat.dim() == 2:
                        lat = lat.unsqueeze(0)
                    lat = lat.to(device=device, dtype=dtype)
                    mask = torch.ones(1, lat.shape[1], device=device, dtype=dtype)

                    with torch.no_grad():
                        _q, indices, _m5 = model.tokenize(lat, silence, mask)
                    codes = _flatten_indices(indices)
                    del lat, _q, indices, _m5

                    duration = int(meta.get("duration") or 0)
                    if duration <= 0:
                        duration = round(len(codes) / CODE_FPS_5)

                    caption = str(meta.get("caption") or pt.stem)
                    tagged = apply_trigger_tag(
                        caption, str(meta.get("custom_tag") or ""), tag_position
                    )
                    row = {
                        "file": pt.name,
                        "caption": tagged,
                        "lyrics": str(meta.get("lyrics") or "[Instrumental]"),
                        "bpm": int(meta.get("bpm") or 0),
                        "keyscale": str(meta.get("keyscale") or ""),
                        "timesignature": str(meta.get("timesignature") or ""),
                        "language": str(meta.get("language") or ""),
                        "duration": duration,
                        "codes": codes,
                    }
                    fh.write(json.dumps(row, ensure_ascii=False) + "\n")
                    rows += 1
                    logger.info(
                        "[LM-Extract] %d/%d %s -> %d codes (%.0fs audio)",
                        i + 1, len(pt_files), pt.name, len(codes), len(codes) / CODE_FPS_5,
                    )
                    if progress:
                        progress(i + 1, len(pt_files), pt.name)
                except Exception:
                    skipped += 1
                    logger.exception("[LM-Extract] SKIP %s", pt.name)
                if device.startswith("cuda") and torch.cuda.is_available():
                    torch.cuda.empty_cache()
    finally:
        if owns_model:
            del model
            # torch modules form reference cycles: without an explicit GC pass
            # the whole DiT stays resident and the 4B LM training then spills
            # into shared memory. gc BEFORE empty_cache, or the blocks aren't
            # back in the allocator pool yet.
            import gc
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    logger.info(
        "[LM-Extract] Done: %d rows, %d skipped -> %s (%.1fs)",
        rows, skipped, out_path, time.time() - t0,
    )
    if rows == 0:
        raise RuntimeError("Code extraction produced no rows — see log for per-file errors")
    return {"rows": rows, "skipped": skipped, "out": str(out_path)}
