"""Merge a trained LM LoRA and export it as a loadable model.

Two export modes:

- ``adapter``: deploy the raw PEFT adapter dir for inference engines
  with a runtime LM-LoRA path.
- ``merged``: merge the LoRA into the base LM and save a standard HF
  safetensors checkpoint dir named ``acestep-5Hz-lm-{size}-{name}``
  into the models root, where any engine that scans HF checkpoint dirs
  (Qwen3ForCausalLM ``config.json`` + ``model.safetensors``) picks it
  up directly.  Optionally convert/quantize to GGUF afterwards with
  your engine's usual tooling.
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any, Dict

import torch

from sidestep_engine.lm.train_lm import resolve_lm_dir

logger = logging.getLogger(__name__)


def sanitize_export_name(name: str) -> str:
    name = re.sub(r"[^A-Za-z0-9._-]+", "-", name.strip()).strip("-._")
    return name or "custom"


def export_lm_adapter(
    adapter_dir: str | Path,
    checkpoint_dir: str | Path,
    lm_size: str = "4B",
    export_name: str = "custom",
    deploy_root: str | Path | None = None,
) -> Dict[str, Any]:
    """Deploy the raw PEFT adapter for an engine's runtime LM-LoRA path.

    Copies the adapter dir (adapter_model.safetensors + adapter_config.json)
    into ``<models root>/adapters/lm/<name>-<size>/``.  The engine scans that
    subtree at startup, lists it in the Planner Adapter dropdown, and applies
    it at runtime on top of ANY base LM quant — no merge, no 8 GB copies.
    ``deploy_root`` defaults to ``<checkpoint_dir>/../adapters/lm``.
    """
    import shutil

    adapter_dir = Path(adapter_dir)
    if not (adapter_dir / "adapter_model.safetensors").is_file():
        raise FileNotFoundError(f"No adapter_model.safetensors in {adapter_dir}")
    if not (adapter_dir / "adapter_config.json").is_file():
        raise FileNotFoundError(
            f"No adapter_config.json in {adapter_dir} — the engine needs it "
            "for the lora_alpha/r scaling factor"
        )

    root = Path(deploy_root) if deploy_root else Path(checkpoint_dir).parent / "adapters" / "lm"
    name = f"{sanitize_export_name(export_name)}-{lm_size}"
    dest = root / name
    dest.mkdir(parents=True, exist_ok=True)
    for f in ("adapter_model.safetensors", "adapter_config.json"):
        shutil.copy2(adapter_dir / f, dest / f)

    logger.info(
        "[LM-Export] Adapter deployed -> %s. Restart the inference engine; it "
        "appears in the Planner Adapter dropdown as '%s' and applies to any "
        "base %s LM quant at runtime.", dest, name, lm_size,
    )
    return {"deploy_dir": str(dest), "adapter_name": name}


def merge_and_export(
    adapter_dir: str | Path,
    checkpoint_dir: str | Path,
    lm_size: str = "4B",
    export_name: str = "custom",
    models_root: str | Path | None = None,
    device: str = "cpu",
) -> Dict[str, Any]:
    """Merge the LoRA into the base LM and save a loadable HF checkpoint dir.

    Merging runs on CPU by default (bf16, ~2x model size in RAM) so it
    never competes for VRAM with a running generation or training job.
    ``models_root`` defaults to *checkpoint_dir* (the models root the
    inference engine already reads).
    """
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    adapter_dir = Path(adapter_dir)
    if not (adapter_dir / "adapter_config.json").is_file():
        raise FileNotFoundError(f"No adapter_config.json in {adapter_dir}")

    lm_dir = resolve_lm_dir(checkpoint_dir, lm_size)
    models_root = Path(models_root) if models_root else Path(checkpoint_dir)
    out_name = f"acestep-5Hz-lm-{lm_size}-{sanitize_export_name(export_name)}"
    out_dir = models_root / out_name

    logger.info("[LM-Export] Loading base %s on %s ...", lm_dir.name, device)
    base = AutoModelForCausalLM.from_pretrained(str(lm_dir), torch_dtype=torch.bfloat16)
    base.to(device)
    logger.info("[LM-Export] Applying adapter %s ...", adapter_dir)
    merged = PeftModel.from_pretrained(base, str(adapter_dir))
    merged = merged.merge_and_unload()

    logger.info("[LM-Export] Saving merged model -> %s", out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    merged.save_pretrained(str(out_dir), safe_serialization=True)
    tokenizer = AutoTokenizer.from_pretrained(str(lm_dir))
    tokenizer.save_pretrained(str(out_dir))

    del merged, base
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    logger.info(
        "[LM-Export] Done. Restart the inference engine and select '%s' in the "
        "LM model dropdown. (Optional: convert to GGUF + quantize for less "
        "VRAM — see this module's docstring.)", out_name,
    )
    return {"export_dir": str(out_dir), "export_name": out_name}
