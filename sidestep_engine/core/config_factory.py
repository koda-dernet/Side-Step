"""Unified config builder: dict -> (AdapterConfig, TrainingConfigV2).

Accepts a flat dict of training parameters (the common denominator of
CLI argparse, Wizard answers, and GUI JSON) and returns fully-built
config objects.  All three interfaces funnel through this module.
"""

from __future__ import annotations

import json as _json
import logging
from pathlib import Path
from typing import Any, Dict, Optional, Tuple, Union

from sidestep_engine.core.configs import (
    LoRAConfigV2, LoKRConfigV2, LoHAConfigV2, OFTConfigV2, TrainingConfigV2,
)
from sidestep_engine.core.constants import (
    BASE_INFERENCE_STEPS,
    BASE_SHIFT,
    DEFAULT_DATA_PROPORTION,
    DEFAULT_TIMESTEP_MU,
    DEFAULT_TIMESTEP_SIGMA,
    PP_LR_WARN_THRESHOLD,
    TURBO_INFERENCE_STEPS,
    TURBO_SHIFT,
    VARIANT_DIR_MAP,
    is_pp_compatible,
    is_turbo as _is_turbo,
)
from sidestep_engine.training_defaults import (
    DEFAULT_EPOCHS,
    DEFAULT_LEARNING_RATE,
    DEFAULT_OPTIMIZER_TYPE,
    DEFAULT_SAVE_EVERY,
)

logger = logging.getLogger(__name__)

AdapterConfig = Union[LoRAConfigV2, LoKRConfigV2, LoHAConfigV2, OFTConfigV2]


def _get(p: Dict, key: str, default: Any = None) -> Any:
    """Get value from dict, returning *default* when missing or None."""
    v = p.get(key)
    return v if v is not None else default


# ---------------------------------------------------------------------------
# Model config.json reading
# ---------------------------------------------------------------------------

def _resolve_model_config(ckpt_root: Path, variant: str) -> Dict:
    """Read model config.json and return its contents (or empty dict)."""
    mapped = VARIANT_DIR_MAP.get(variant)
    candidates = []
    if mapped:
        candidates.append(ckpt_root / mapped / "config.json")
    candidates.append(ckpt_root / variant / "config.json")

    for config_path in candidates:
        if config_path.is_file():
            try:
                return _json.loads(config_path.read_text(encoding="utf-8"))
            except (_json.JSONDecodeError, OSError) as exc:
                logger.warning(
                    "[Side-Step] Failed to parse %s: %s", config_path, exc,
                )
    return {}


# ---------------------------------------------------------------------------
# Target module resolution
# ---------------------------------------------------------------------------

def _resolve_modules(p: Dict) -> list:
    """Resolve target modules from parameters dict."""
    from sidestep_engine.cli.validation import resolve_target_modules

    target_modules = _get(p, "target_modules", ["q_proj", "k_proj", "v_proj", "o_proj"])
    if isinstance(target_modules, str):
        target_modules = target_modules.split()

    return resolve_target_modules(
        target_modules,
        _get(p, "attention_type", "both"),
        self_target_modules=_get(p, "self_target_modules"),
        cross_target_modules=_get(p, "cross_target_modules"),
        target_mlp=_get(p, "target_mlp", False),
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_training_config(
    params: Dict[str, Any],
) -> Tuple[AdapterConfig, TrainingConfigV2]:
    """Build adapter + training configs from a flat parameter dict.

    This is the single source of truth for config construction.
    All interfaces (CLI, Wizard, GUI) should funnel through here.

    Required keys: ``checkpoint_dir``, ``model_variant``, ``dataset_dir``.
    Everything else has sensible defaults sourced from ``training_defaults.py``
    and ``core/constants.py``.
    """
    p = dict(params)  # shallow copy

    # -- Resolve model config for timestep params ----------------------------
    ckpt_root = Path(p["checkpoint_dir"])
    mcfg = _resolve_model_config(ckpt_root, p["model_variant"])

    timestep_mu = mcfg.get("timestep_mu", DEFAULT_TIMESTEP_MU)
    timestep_sigma = mcfg.get("timestep_sigma", DEFAULT_TIMESTEP_SIGMA)
    data_proportion = mcfg.get("data_proportion", DEFAULT_DATA_PROPORTION)
    num_hidden_layers: Optional[int] = mcfg.get("num_hidden_layers")

    # -- Override from --base-model if config lacked params ------------------
    base_model = _get(p, "base_model")
    if base_model and not mcfg:
        from sidestep_engine.models.discovery import get_base_defaults
        defaults = get_base_defaults(base_model)
        timestep_mu = defaults.get("timestep_mu", timestep_mu)
        timestep_sigma = defaults.get("timestep_sigma", timestep_sigma)

    # -- Explicit user overrides ---------------------------------------------
    if p.get("timestep_mu") is not None:
        timestep_mu = float(p["timestep_mu"])
    if p.get("timestep_sigma") is not None:
        timestep_sigma = float(p["timestep_sigma"])

    # -- GPU info ------------------------------------------------------------
    from sidestep_engine.models.gpu_utils import detect_gpu
    gpu_info = detect_gpu()
    user_device = p.get("device")
    user_precision = p.get("precision")
    if user_device and user_device != "auto":
        gpu_info.device = user_device
    if user_precision and user_precision != "auto":
        gpu_info.precision = user_precision

    # -- Adapter config ------------------------------------------------------
    adapter_type = _get(p, "adapter_type", "lora")
    attention_type = _get(p, "attention_type", "both")
    target_mlp = _get(p, "target_mlp", False)
    resolved_modules = _resolve_modules(p)

    adapter_cfg: AdapterConfig
    if adapter_type == "lokr":
        adapter_cfg = LoKRConfigV2(
            linear_dim=_get(p, "lokr_linear_dim", 64),
            linear_alpha=_get(p, "lokr_linear_alpha", 128),
            factor=_get(p, "lokr_factor", -1),
            decompose_both=_get(p, "lokr_decompose_both", False),
            use_tucker=_get(p, "lokr_use_tucker", False),
            use_scalar=_get(p, "lokr_use_scalar", False),
            weight_decompose=_get(p, "lokr_weight_decompose", False),
            target_modules=resolved_modules,
            attention_type=attention_type,
            target_mlp=target_mlp,
        )
    elif adapter_type == "loha":
        adapter_cfg = LoHAConfigV2(
            linear_dim=_get(p, "loha_linear_dim", 64),
            linear_alpha=_get(p, "loha_linear_alpha", 128),
            factor=_get(p, "loha_factor", -1),
            use_tucker=_get(p, "loha_use_tucker", False),
            use_scalar=_get(p, "loha_use_scalar", False),
            target_modules=resolved_modules,
            attention_type=attention_type,
            target_mlp=target_mlp,
        )
    elif adapter_type == "oft":
        adapter_cfg = OFTConfigV2(
            block_size=_get(p, "oft_block_size", 64),
            coft=_get(p, "oft_coft", False),
            eps=_get(p, "oft_eps", 6e-5),
            target_modules=resolved_modules,
            attention_type=attention_type,
            target_mlp=target_mlp,
        )
    else:
        adapter_cfg = LoRAConfigV2(
            r=_get(p, "rank", 64),
            alpha=_get(p, "alpha", 128),
            dropout=_get(p, "dropout", 0.0),
            target_modules=resolved_modules,
            bias=_get(p, "bias", "none"),
            attention_type=attention_type,
            target_mlp=target_mlp,
            use_dora=(adapter_type == "dora"),
        )

    # -- Fisher map auto-detection -------------------------------------------
    ignore_fisher = _get(p, "ignore_fisher_map", False)
    fisher_map_path = Path(p["dataset_dir"]) / "fisher_map.json"

    if is_pp_compatible(adapter_type) and not ignore_fisher and fisher_map_path.is_file():
        from sidestep_engine.analysis.fisher.io import load_fisher_map
        fisher_data = load_fisher_map(
            fisher_map_path,
            expected_variant=p["model_variant"],
            dataset_dir=p["dataset_dir"],
            expected_num_layers=num_hidden_layers,
        )
        if fisher_data:
            adapter_cfg.target_modules = fisher_data["target_modules"]
            adapter_cfg.rank_pattern = fisher_data["rank_pattern"]
            adapter_cfg.alpha_pattern = fisher_data["alpha_pattern"]
            budget = fisher_data.get("rank_budget", {})
            adapter_cfg.rank_min = budget.get("min", 16)
            adapter_cfg.r = adapter_cfg.rank_min
            adapter_cfg.alpha = adapter_cfg.rank_min * 2
            logger.info(
                "[Side-Step] Fisher map loaded: %d modules, adaptive ranks %d-%d",
                len(fisher_data["rank_pattern"]),
                budget.get("min", 16),
                budget.get("max", 128),
            )

            lr = _get(p, "learning_rate", DEFAULT_LEARNING_RATE)
            if lr > PP_LR_WARN_THRESHOLD:
                logger.warning(
                    "[Side-Step] Preprocessing++ is active with lr=%.1e. "
                    "Adaptive ranks concentrate capacity on fewer modules, "
                    "which overfits faster. Consider lowering to ~5e-5.",
                    lr,
                )

    # -- DataLoader flags ----------------------------------------------------
    num_workers = _get(p, "num_workers", 4)
    prefetch_factor = _get(p, "prefetch_factor", 2)
    persistent_workers = _get(p, "persistent_workers", True)

    if num_workers <= 0:
        if persistent_workers:
            logger.info("[Side-Step] num_workers=0 -- forcing persistent_workers=False")
            persistent_workers = False
        if prefetch_factor and prefetch_factor > 0:
            logger.info("[Side-Step] num_workers=0 -- forcing prefetch_factor=0")
            prefetch_factor = 0

    # -- Turbo auto-detection ------------------------------------------------
    infer_steps = _get(p, "num_inference_steps", TURBO_INFERENCE_STEPS)
    shift = _get(p, "shift", TURBO_SHIFT)
    turbo = _is_turbo(p)

    if not turbo:
        if infer_steps == TURBO_INFERENCE_STEPS:
            infer_steps = BASE_INFERENCE_STEPS
        if shift == TURBO_SHIFT:
            shift = BASE_SHIFT

    # -- Gradient checkpointing ----------------------------------------------
    try:
        gc_ratio = float(_get(p, "gradient_checkpointing_ratio", 1.0))
    except (TypeError, ValueError):
        gc_ratio = 1.0
    gc_enabled = bool(_get(p, "gradient_checkpointing", True))
    if gc_ratio <= 0:
        gc_enabled = False

    # -- Scheduler formula ---------------------------------------------------
    sched_type = _get(p, "scheduler_type", "cosine")
    formula = _get(p, "scheduler_formula", "")
    if sched_type != "custom" and formula:
        logger.warning(
            "[Side-Step] scheduler_formula was provided but scheduler_type "
            "is '%s' (not 'custom') -- the formula will be ignored.",
            sched_type,
        )
        formula = ""

    # -- Training config -----------------------------------------------------
    train_cfg = TrainingConfigV2(
        is_turbo=turbo,
        shift=shift,
        num_inference_steps=infer_steps,
        learning_rate=_get(p, "learning_rate", DEFAULT_LEARNING_RATE),
        batch_size=_get(p, "batch_size", 1),
        gradient_accumulation_steps=_get(p, "gradient_accumulation", 1),
        max_epochs=_get(p, "epochs", DEFAULT_EPOCHS),
        warmup_steps=_get(p, "warmup_steps", 0),
        weight_decay=_get(p, "weight_decay", 0.01),
        max_grad_norm=_get(p, "max_grad_norm", 1.0),
        seed=_get(p, "seed", 42),
        output_dir=_get(p, "output_dir", ""),
        save_every_n_epochs=_get(p, "save_every", DEFAULT_SAVE_EVERY),
        num_workers=num_workers,
        pin_memory=_get(p, "pin_memory", True),
        prefetch_factor=prefetch_factor,
        persistent_workers=persistent_workers,
        adapter_type=adapter_type,
        optimizer_type=_get(p, "optimizer_type", DEFAULT_OPTIMIZER_TYPE),
        scheduler_type=sched_type,
        scheduler_formula=formula,
        gradient_checkpointing=gc_enabled,
        gradient_checkpointing_ratio=gc_ratio,
        offload_encoder=_get(p, "offload_encoder", True),
        save_best=_get(p, "save_best", True),
        save_best_after=_get(p, "save_best_after", 200),
        early_stop_patience=_get(p, "early_stop_patience", 0),
        cfg_ratio=_get(p, "cfg_ratio", 0.15),
        loss_weighting=_get(p, "loss_weighting", "none"),
        snr_gamma=_get(p, "snr_gamma", 5.0),
        timestep_mu=timestep_mu,
        timestep_sigma=timestep_sigma,
        data_proportion=data_proportion,
        model_variant=p["model_variant"],
        checkpoint_dir=p["checkpoint_dir"],
        dataset_dir=p["dataset_dir"],
        device=gpu_info.device,
        precision=gpu_info.precision,
        resume_from=_get(p, "resume_from", ""),
        strict_resume=_get(p, "strict_resume", True),
        run_name=_get(p, "run_name"),
        log_dir=_get(p, "log_dir", "runs"),
        log_every=_get(p, "log_every", 10),
        log_heavy_every=max(0, _get(p, "log_heavy_every", 50)),
        preprocess=_get(p, "preprocess", False),
        audio_dir=_get(p, "audio_dir", ""),
        dataset_json=_get(p, "dataset_json", ""),
        tensor_output=_get(p, "tensor_output", ""),
        max_duration=_get(p, "max_duration", 0),
        normalize=_get(p, "normalize", "none"),
        chunk_duration=_get(p, "chunk_duration") or None,
        chunk_decay_every=_get(p, "chunk_decay_every", 10),
        dataset_repeats=_get(p, "dataset_repeats", 1),
        max_steps=_get(p, "max_steps", 0),
        ema_decay=_get(p, "ema_decay", 0.0),
        val_split=_get(p, "val_split", 0.0),
        adaptive_timestep_ratio=_get(p, "adaptive_timestep_ratio", 0.0),
        warmup_start_factor=_get(p, "warmup_start_factor", 0.1),
        cosine_eta_min_ratio=_get(p, "cosine_eta_min_ratio", 0.01),
        cosine_restarts_count=_get(p, "cosine_restarts_count", 4),
        save_best_every_n_steps=_get(p, "save_best_every_n_steps", 0),
    )

    return adapter_cfg, train_cfg


def namespace_to_params(ns: "argparse.Namespace") -> Dict[str, Any]:
    """Convert an argparse Namespace to a flat params dict for ``build_training_config``."""
    import argparse
    return {k: v for k, v in vars(ns).items() if v is not None}
