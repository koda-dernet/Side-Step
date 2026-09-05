"""Orchestrator for the LM adapter pipeline: extract -> train -> export.

Used by the ``lm-train`` CLI subcommand (standalone) and by the DiT
training chain hooks (``--lm-train before|after``).

"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

VALID_LM_SIZES = ("0.6B", "1.7B", "4B")


def run_lm_pipeline(
    dataset_dir: str,
    checkpoint_dir: str,
    output_dir: str,
    model_variant: str = "base",
    lm_size: str = "4B",
    rank: int = 16,
    alpha: int = 32,
    dropout: float = 0.05,
    learning_rate: float = 1e-4,
    epochs: int = 4,
    grad_accum: int = 4,
    max_len: int = 8192,
    loss_on_cot: bool = True,
    target_loss: float = 0.0,
    seed: int = 42,
    device: str = "cuda",
    precision: str = "bf16",
    export_name: Optional[str] = None,
    models_root: Optional[str] = None,
    stages: str = "all",  # "all" | "extract" | "train" | "export" (comma-combinable)
    refresh_codes: bool = False,
    export_mode: str = "adapter",  # "adapter" (runtime LoRA) | "merged" | "both"
) -> Dict[str, Any]:
    """Run the requested pipeline stages.  Returns a summary dict.

    The code JSONL is cached at ``{output_dir}/lm_codes.jsonl`` and
    reused unless *refresh_codes* is set (extraction needs the full DiT
    model in memory; training/export do not).
    """
    if lm_size not in VALID_LM_SIZES:
        raise ValueError(f"lm_size must be one of {VALID_LM_SIZES} (got {lm_size!r})")

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    jsonl = out / "lm_codes.jsonl"
    wanted = {s.strip() for s in stages.split(",")} if stages != "all" else {"extract", "train", "export"}
    summary: Dict[str, Any] = {"jsonl": str(jsonl)}

    if "extract" in wanted and (refresh_codes or not jsonl.is_file()):
        from sidestep_engine.lm.extract import extract_codes
        logger.info("[LM] Stage 1/3: extracting audio codes from %s", dataset_dir)
        summary["extract"] = extract_codes(
            dataset_dir, checkpoint_dir, model_variant, jsonl,
            device=device, precision=precision,
        )
    elif jsonl.is_file():
        logger.info("[LM] Stage 1/3: reusing cached codes %s (--lm-refresh-codes to redo)", jsonl)
    elif "train" in wanted:
        raise RuntimeError(f"No code JSONL at {jsonl} and extract stage not requested")

    if "train" in wanted:
        from sidestep_engine.lm.train_lm import train_lm_adapter
        logger.info("[LM] Stage 2/3: training %s LoRA (r=%d) ...", lm_size, rank)
        summary["train"] = train_lm_adapter(
            jsonl, checkpoint_dir, out, lm_size=lm_size, rank=rank, alpha=alpha,
            dropout=dropout, learning_rate=learning_rate, epochs=epochs,
            grad_accum=grad_accum, max_len=max_len, loss_on_cot=loss_on_cot,
            seed=seed, device=device, target_loss=target_loss,
        )

    if "export" in wanted:
        adapter_dir = out / "lm_adapter"
        name = export_name or out.name
        if export_mode not in ("adapter", "merged", "both"):
            raise ValueError(f"export_mode must be adapter|merged|both (got {export_mode!r})")
        if export_mode in ("adapter", "both"):
            from sidestep_engine.lm.export_lm import export_lm_adapter
            logger.info("[LM] Stage 3/3: deploying runtime adapter as %s ...", name)
            summary["export"] = export_lm_adapter(
                adapter_dir, checkpoint_dir, lm_size=lm_size, export_name=name,
                deploy_root=models_root if export_mode == "adapter" and models_root else None,
            )
        if export_mode in ("merged", "both"):
            from sidestep_engine.lm.export_lm import merge_and_export
            logger.info("[LM] Stage 3/3%s: merging + exporting as %s ...",
                        "b" if export_mode == "both" else "", name)
            summary["export_merged"] = merge_and_export(
                adapter_dir, checkpoint_dir, lm_size=lm_size, export_name=name,
                models_root=models_root if export_mode != "both" else None, device="cpu",
            )

    logger.info("[LM] Pipeline complete: %s", {k: v for k, v in summary.items() if k != "jsonl"})
    return summary
