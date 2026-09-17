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
    DEFAULT_ALPHA,
    DEFAULT_ATTENTION_TYPE,
    DEFAULT_BATCH_SIZE,
    DEFAULT_BIAS,
    DEFAULT_CFG_RATIO,
    DEFAULT_CHUNK_DECAY_EVERY,
    DEFAULT_COSINE_ETA_MIN_RATIO,
    DEFAULT_COSINE_RESTARTS_COUNT,
    DEFAULT_DATASET_REPEATS,
    DEFAULT_DROPOUT,
    DEFAULT_EARLY_STOP_PATIENCE,
    DEFAULT_EMA_DECAY,
    DEFAULT_EMA_START_STEP,
    DEFAULT_TARGET_LOSS,
    DEFAULT_TARGET_LOSS_FLOOR,
    DEFAULT_TARGET_LOSS_WARMUP,
    DEFAULT_TARGET_LOSS_SMOOTHING,
    DEFAULT_EPOCHS,
    DEFAULT_GRADIENT_ACCUMULATION,
    DEFAULT_GRADIENT_CHECKPOINTING,
    DEFAULT_GRADIENT_CHECKPOINTING_RATIO,
    DEFAULT_LEARNING_RATE,
    DEFAULT_LOG_EVERY,
    DEFAULT_LOG_HEAVY_EVERY,
    DEFAULT_LOHA_FACTOR,
    DEFAULT_LOHA_LINEAR_ALPHA,
    DEFAULT_LOHA_LINEAR_DIM,
    DEFAULT_LOHA_USE_SCALAR,
    DEFAULT_LOHA_USE_TUCKER,
    DEFAULT_LOKR_DECOMPOSE_BOTH,
    DEFAULT_LOKR_FACTOR,
    DEFAULT_LOKR_LINEAR_ALPHA,
    DEFAULT_LOKR_LINEAR_DIM,
    DEFAULT_LOKR_USE_SCALAR,
    DEFAULT_LOKR_USE_TUCKER,
    DEFAULT_LOKR_WEIGHT_DECOMPOSE,
    DEFAULT_LOSS_WEIGHTING,
    DEFAULT_MAX_GRAD_NORM,
    DEFAULT_MAX_STEPS,
    DEFAULT_NUM_WORKERS,
    DEFAULT_OFFLOAD_ENCODER,
    DEFAULT_OFT_BLOCK_SIZE,
    DEFAULT_OFT_COFT,
    DEFAULT_OFT_EPS,
    DEFAULT_OPTIMIZER_TYPE,
    DEFAULT_PERSISTENT_WORKERS,
    DEFAULT_PIN_MEMORY,
    DEFAULT_PREFETCH_FACTOR,
    DEFAULT_RANK,
    DEFAULT_SAVE_BEST,
    DEFAULT_SAVE_BEST_AFTER,
    DEFAULT_SAVE_BEST_EVERY_N_STEPS,
    DEFAULT_SAVE_EVERY,
    DEFAULT_SCHEDULER_TYPE,
    DEFAULT_SEED,
    DEFAULT_SNR_GAMMA,
    DEFAULT_LOSS_FN,
    DEFAULT_HUBER_DELTA,
    DEFAULT_CHANNEL_BALANCE,
    DEFAULT_DYNAMIC_CHANNEL_BALANCE,
    DEFAULT_VAE_CHANNEL_PRIOR,
    DEFAULT_LATENT_NOISE,
    DEFAULT_T_BIAS,
    DEFAULT_LEGACY_LOSS,
    DEFAULT_STRICT_RESUME,
    DEFAULT_TARGET_MLP,
    DEFAULT_TIMESTEP_MODE,
    DEFAULT_ADAPTIVE_TIMESTEP_RATIO,
    DEFAULT_VAL_SPLIT,
    DEFAULT_WARMUP_START_FACTOR,
    DEFAULT_WARMUP_STEPS,
    DEFAULT_WEIGHT_DECAY,
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
        _get(p, "attention_type", DEFAULT_ATTENTION_TYPE),
        self_target_modules=_get(p, "self_target_modules"),
        cross_target_modules=_get(p, "cross_target_modules"),
        target_mlp=_get(p, "target_mlp", DEFAULT_TARGET_MLP),
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
    attention_type = _get(p, "attention_type", DEFAULT_ATTENTION_TYPE)
    target_mlp = _get(p, "target_mlp", DEFAULT_TARGET_MLP)
    resolved_modules = _resolve_modules(p)

    adapter_cfg: AdapterConfig
    if adapter_type == "lokr":
        adapter_cfg = LoKRConfigV2(
            linear_dim=_get(p, "lokr_linear_dim", DEFAULT_LOKR_LINEAR_DIM),
            linear_alpha=_get(p, "lokr_linear_alpha", DEFAULT_LOKR_LINEAR_ALPHA),
            factor=_get(p, "lokr_factor", DEFAULT_LOKR_FACTOR),
            decompose_both=_get(p, "lokr_decompose_both", DEFAULT_LOKR_DECOMPOSE_BOTH),
            use_tucker=_get(p, "lokr_use_tucker", DEFAULT_LOKR_USE_TUCKER),
            use_scalar=_get(p, "lokr_use_scalar", DEFAULT_LOKR_USE_SCALAR),
            weight_decompose=_get(p, "lokr_weight_decompose", DEFAULT_LOKR_WEIGHT_DECOMPOSE),
            target_modules=resolved_modules,
            attention_type=attention_type,
            target_mlp=target_mlp,
        )
    elif adapter_type == "loha":
        adapter_cfg = LoHAConfigV2(
            linear_dim=_get(p, "loha_linear_dim", DEFAULT_LOHA_LINEAR_DIM),
            linear_alpha=_get(p, "loha_linear_alpha", DEFAULT_LOHA_LINEAR_ALPHA),
            factor=_get(p, "loha_factor", DEFAULT_LOHA_FACTOR),
            use_tucker=_get(p, "loha_use_tucker", DEFAULT_LOHA_USE_TUCKER),
            use_scalar=_get(p, "loha_use_scalar", DEFAULT_LOHA_USE_SCALAR),
            target_modules=resolved_modules,
            attention_type=attention_type,
            target_mlp=target_mlp,
        )
    elif adapter_type == "oft":
        adapter_cfg = OFTConfigV2(
            block_size=_get(p, "oft_block_size", DEFAULT_OFT_BLOCK_SIZE),
            coft=_get(p, "oft_coft", DEFAULT_OFT_COFT),
            eps=_get(p, "oft_eps", DEFAULT_OFT_EPS),
            target_modules=resolved_modules,
            attention_type=attention_type,
            target_mlp=target_mlp,
        )
    else:
        adapter_cfg = LoRAConfigV2(
            r=_get(p, "rank", DEFAULT_RANK),
            alpha=_get(p, "alpha", DEFAULT_ALPHA),
            dropout=_get(p, "dropout", DEFAULT_DROPOUT),
            target_modules=resolved_modules,
            bias=_get(p, "bias", DEFAULT_BIAS),
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

            lr = float(_get(p, "learning_rate", DEFAULT_LEARNING_RATE))
            if lr > PP_LR_WARN_THRESHOLD:
                logger.warning(
                    "[Side-Step] Preprocessing++ is active with lr=%.1e. "
                    "Adaptive ranks concentrate capacity on fewer modules, "
                    "which overfits faster. Consider lowering to ~5e-5.",
                    lr,
                )

    # -- DataLoader flags ----------------------------------------------------
    num_workers = _get(p, "num_workers", DEFAULT_NUM_WORKERS)
    prefetch_factor = _get(p, "prefetch_factor", DEFAULT_PREFETCH_FACTOR)
    persistent_workers = _get(p, "persistent_workers", DEFAULT_PERSISTENT_WORKERS)

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
        gc_ratio = float(_get(p, "gradient_checkpointing_ratio", DEFAULT_GRADIENT_CHECKPOINTING_RATIO))
    except (TypeError, ValueError):
        gc_ratio = DEFAULT_GRADIENT_CHECKPOINTING_RATIO
    gc_enabled = bool(_get(p, "gradient_checkpointing", DEFAULT_GRADIENT_CHECKPOINTING))
    if gc_ratio <= 0:
        gc_enabled = False

    # -- Scheduler formula ---------------------------------------------------
    sched_type = _get(p, "scheduler_type", DEFAULT_SCHEDULER_TYPE)
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
        batch_size=_get(p, "batch_size", DEFAULT_BATCH_SIZE),
        gradient_accumulation_steps=_get(p, "gradient_accumulation", DEFAULT_GRADIENT_ACCUMULATION),
        max_epochs=_get(p, "epochs", DEFAULT_EPOCHS),
        warmup_steps=_get(p, "warmup_steps", DEFAULT_WARMUP_STEPS),
        weight_decay=_get(p, "weight_decay", DEFAULT_WEIGHT_DECAY),
        max_grad_norm=_get(p, "max_grad_norm", DEFAULT_MAX_GRAD_NORM),
        seed=_get(p, "seed", DEFAULT_SEED),
        output_dir=_get(p, "output_dir", ""),
        save_every_n_epochs=_get(p, "save_every", DEFAULT_SAVE_EVERY),
        num_workers=num_workers,
        pin_memory=_get(p, "pin_memory", DEFAULT_PIN_MEMORY),
        prefetch_factor=prefetch_factor,
        persistent_workers=persistent_workers,
        adapter_type=adapter_type,
        optimizer_type=_get(p, "optimizer_type", DEFAULT_OPTIMIZER_TYPE),
        scheduler_type=sched_type,
        scheduler_formula=formula,
        gradient_checkpointing=gc_enabled,
        gradient_checkpointing_ratio=gc_ratio,
        offload_encoder=_get(p, "offload_encoder", DEFAULT_OFFLOAD_ENCODER),
        save_best=_get(p, "save_best", DEFAULT_SAVE_BEST),
        save_best_after=_get(p, "save_best_after", DEFAULT_SAVE_BEST_AFTER),
        early_stop_patience=_get(p, "early_stop_patience", DEFAULT_EARLY_STOP_PATIENCE),
        target_loss=_get(p, "target_loss", DEFAULT_TARGET_LOSS),
        target_loss_floor=_get(p, "target_loss_floor", DEFAULT_TARGET_LOSS_FLOOR),
        target_loss_warmup=_get(p, "target_loss_warmup", DEFAULT_TARGET_LOSS_WARMUP),
        target_loss_smoothing=_get(p, "target_loss_smoothing", DEFAULT_TARGET_LOSS_SMOOTHING),
        cfg_ratio=_get(p, "cfg_ratio", DEFAULT_CFG_RATIO),
        loss_weighting=_get(p, "loss_weighting", DEFAULT_LOSS_WEIGHTING),
        snr_gamma=_get(p, "snr_gamma", DEFAULT_SNR_GAMMA),
        loss_fn=_get(p, "loss_fn", DEFAULT_LOSS_FN),
        huber_delta=_get(p, "huber_delta", DEFAULT_HUBER_DELTA),
        channel_balance=_get(p, "channel_balance", DEFAULT_CHANNEL_BALANCE),
        dynamic_channel_balance=_get(p, "dynamic_channel_balance", DEFAULT_DYNAMIC_CHANNEL_BALANCE),
        vae_channel_prior=_get(p, "vae_channel_prior", DEFAULT_VAE_CHANNEL_PRIOR),
        latent_noise=_get(p, "latent_noise", DEFAULT_LATENT_NOISE),
        t_bias=_get(p, "t_bias", DEFAULT_T_BIAS),
        legacy_loss=_get(p, "legacy_loss", DEFAULT_LEGACY_LOSS),
        timestep_mu=timestep_mu,
        timestep_sigma=timestep_sigma,
        data_proportion=data_proportion,
        model_variant=p["model_variant"],
        checkpoint_dir=p["checkpoint_dir"],
        dataset_dir=p["dataset_dir"],
        device=gpu_info.device,
        precision=gpu_info.precision,
        resume_from=_get(p, "resume_from", ""),
        strict_resume=_get(p, "strict_resume", DEFAULT_STRICT_RESUME),
        run_name=_get(p, "run_name"),
        log_dir=_get(p, "log_dir", "runs"),
        log_every=_get(p, "log_every", DEFAULT_LOG_EVERY),
        log_heavy_every=max(0, _get(p, "log_heavy_every", DEFAULT_LOG_HEAVY_EVERY)),
        preprocess=_get(p, "preprocess", False),
        audio_dir=_get(p, "audio_dir", ""),
        dataset_json=_get(p, "dataset_json", ""),
        tensor_output=_get(p, "tensor_output", ""),
        max_duration=_get(p, "max_duration", 0),
        normalize=_get(p, "normalize", "none"),
        chunk_duration=_get(p, "chunk_duration") or None,
        max_latent_length=_get(p, "max_latent_length") or None,
        crop_mode=_get(p, "crop_mode") or None,
        chunk_decay_every=_get(p, "chunk_decay_every", DEFAULT_CHUNK_DECAY_EVERY),
        dataset_repeats=_get(p, "dataset_repeats", DEFAULT_DATASET_REPEATS),
        max_steps=_get(p, "max_steps", DEFAULT_MAX_STEPS),
        ema_decay=_get(p, "ema_decay", DEFAULT_EMA_DECAY),
        ema_start_step=_get(p, "ema_start_step", DEFAULT_EMA_START_STEP),
        val_split=_get(p, "val_split", DEFAULT_VAL_SPLIT),
        timestep_mode=_get(p, "timestep_mode", DEFAULT_TIMESTEP_MODE),
        adaptive_timestep_ratio=_get(p, "adaptive_timestep_ratio", DEFAULT_ADAPTIVE_TIMESTEP_RATIO),
        warmup_start_factor=_get(p, "warmup_start_factor", DEFAULT_WARMUP_START_FACTOR),
        cosine_eta_min_ratio=_get(p, "cosine_eta_min_ratio", DEFAULT_COSINE_ETA_MIN_RATIO),
        cosine_restarts_count=_get(p, "cosine_restarts_count", DEFAULT_COSINE_RESTARTS_COUNT),
        save_best_every_n_steps=_get(p, "save_best_every_n_steps", DEFAULT_SAVE_BEST_EVERY_N_STEPS),
    )

    return adapter_cfg, train_cfg


def namespace_to_params(ns: "argparse.Namespace") -> Dict[str, Any]:
    """Convert an argparse Namespace to a flat params dict for ``build_training_config``."""
    import argparse
    return {k: v for k, v in vars(ns).items() if v is not None}
