"""
Gradient Sensitivity Estimation -- Reusable Module

Provides ``run_estimation()`` for use from both the CLI and the TUI.
Internally uses the Fisher diagonal engine for noise-robust scoring,
with gradient checkpointing enabled to fix the VRAM leak present in
earlier versions.

The public API (signature + return format) is unchanged so existing
callers (TUI ``EstimateMonitorScreen``, wizard flows) keep working.
"""

from __future__ import annotations

import json
import logging
from typing import Any, Callable, Dict, List, Optional

import torch

logger = logging.getLogger(__name__)


def run_estimation(
    checkpoint_dir: str,
    variant: str,
    dataset_dir: str,
    num_batches: int = 10,
    batch_size: int = 1,
    top_k: int = 16,
    granularity: str = "module",
    progress_callback: Optional[Callable] = None,
    cancel_check: Optional[Callable] = None,
    cfg_ratio: float = 0.0,
) -> List[Dict[str, Any]]:
    """Run gradient sensitivity analysis and return ranked modules.

    Args:
        checkpoint_dir: Path to model checkpoints.
        variant: Model variant (turbo, base, sft).
        dataset_dir: Directory with preprocessed .pt files.
        num_batches: Number of forward/backward passes for estimation.
        batch_size: Samples per estimation batch.
        top_k: Number of top modules to return.
        granularity: ``"module"`` or ``"layer"`` (ignored -- always
            module-level internally, kept for backward compat).
        progress_callback: ``(batch, total, module_name) -> None``.
        cancel_check: ``() -> bool`` -- return True to cancel.
        cfg_ratio: CFG dropout ratio (unused, kept for compat).

    Returns:
        List of dicts ``[{"module": name, "sensitivity": float}, ...]``
        sorted descending by sensitivity.
    """
    from acestep.training_v2.model_loader import (
        load_decoder_for_training,
        read_model_config,
        unload_models,
    )
    from acestep.training_v2.gpu_utils import detect_gpu
    from acestep.training_v2._vendor.data_module import PreprocessedDataModule
    from acestep.training_v2.fisher.modules import (
        find_all_targetable_modules,
        build_param_to_module_map,
    )
    from acestep.training_v2.fisher.engine import single_fisher_run

    gpu = detect_gpu()
    device = torch.device(gpu.device)
    dtype_map = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}
    dtype = dtype_map.get(gpu.precision, torch.bfloat16)

    logger.info("[Side-Step] Loading model for estimation (variant=%s)", variant)
    model = load_decoder_for_training(
        checkpoint_dir=checkpoint_dir,
        variant=variant,
        device=gpu.device,
        precision=gpu.precision,
    )

    # FIX: enable gradient checkpointing (was missing → VRAM leak)
    try:
        from acestep.training_v2.trainer_helpers import configure_memory_features
        configure_memory_features(model.decoder)
    except Exception as exc:
        logger.warning("[Side-Step] Could not enable gradient checkpointing: %s", exc)

    target_modules = find_all_targetable_modules(model)
    logger.info("[Side-Step] Found %d targetable modules", len(target_modules))

    if not target_modules:
        logger.error("[Side-Step] No targetable modules found -- aborting estimation")
        unload_models(model)
        return []

    target_names = [n for n, _ in target_modules]
    param_to_module = build_param_to_module_map(model, target_modules)

    data_module = PreprocessedDataModule(
        tensor_dir=dataset_dir,
        batch_size=batch_size,
        num_workers=0,
        pin_memory=False,
    )
    data_module.setup("fit")

    scores = single_fisher_run(
        model=model,
        loader_factory=data_module.train_dataloader,
        param_to_module=param_to_module,
        target_names=target_names,
        max_batches=num_batches,
        timestep_focus="balanced",
        device=device,
        dtype=dtype,
        patience=num_batches + 1,
        progress_callback=progress_callback,
        cancel_check=cancel_check,
    )

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    results = [
        {"module": name, "sensitivity": score}
        for name, score in ranked[:top_k]
    ]

    logger.info(
        "[Side-Step] Estimation complete (%d batches): top module = %s (%.6f)",
        num_batches,
        results[0]["module"] if results else "none",
        results[0]["sensitivity"] if results else 0.0,
    )

    unload_models(model)
    return results
