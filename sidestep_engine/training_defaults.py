"""
Canonical training defaults shared across entrypoints.

**This module is the single source of truth for every default that
appears in more than one interface (CLI, Wizard, GUI, config_factory).**

When a default changes, update it here and all consumers will pick it up.
The ``TRAINING_DEFAULTS`` dict at the bottom aggregates every constant
for easy consumption by the ``/api/defaults`` endpoint and review table.

GUI field-ID mapping
~~~~~~~~~~~~~~~~~~~~
The GUI uses HTML element IDs (``full-lr``, ``full-batch``, …) that
differ from backend parameter names.  ``GUI_FIELD_MAP`` translates
backend keys → GUI field IDs so the server can emit defaults keyed
the way the frontend expects.  ``GUI_KEY_MAP`` is the reverse
(GUI/JSON config key → backend parameter name) used when *reading*
config dicts produced by the frontend.
"""

from __future__ import annotations

import torch
import os
import sys

# ---------------------------------------------------------------------------
# Training hyper-parameters
# ---------------------------------------------------------------------------

DEFAULT_LEARNING_RATE: float = 3e-4
DEFAULT_BATCH_SIZE: int = 1
DEFAULT_GRADIENT_ACCUMULATION: int = 4
DEFAULT_EPOCHS: int = 1000
DEFAULT_WARMUP_STEPS: int = 100
DEFAULT_WEIGHT_DECAY: float = 0.01
DEFAULT_MAX_GRAD_NORM: float = 1.0
DEFAULT_SEED: int = 42
DEFAULT_MAX_STEPS: int = 0
DEFAULT_DATASET_REPEATS: int = 1

# ---------------------------------------------------------------------------
# Optimizer / scheduler
# ---------------------------------------------------------------------------

DEFAULT_OPTIMIZER_TYPE: str = "adamw8bit" if torch.cuda.is_available() else "adamw"
DEFAULT_SCHEDULER_TYPE: str = "cosine"
DEFAULT_SCHEDULER_FORMULA: str = ""

# ---------------------------------------------------------------------------
# LoRA defaults
# ---------------------------------------------------------------------------

DEFAULT_RANK: int = 64
DEFAULT_ALPHA: int = 128
DEFAULT_DROPOUT: float = 0.1
DEFAULT_TARGET_MODULES: list = ["q_proj", "k_proj", "v_proj", "o_proj"]
DEFAULT_ATTENTION_TYPE: str = "both"
DEFAULT_TARGET_MLP: bool = True
DEFAULT_BIAS: str = "none"

# ---------------------------------------------------------------------------
# LoKR defaults
# ---------------------------------------------------------------------------

DEFAULT_LOKR_LINEAR_DIM: int = 64
DEFAULT_LOKR_LINEAR_ALPHA: int = 128
DEFAULT_LOKR_FACTOR: int = -1
DEFAULT_LOKR_DECOMPOSE_BOTH: bool = False
DEFAULT_LOKR_USE_TUCKER: bool = False
DEFAULT_LOKR_USE_SCALAR: bool = False
DEFAULT_LOKR_WEIGHT_DECOMPOSE: bool = False

# ---------------------------------------------------------------------------
# LoHA defaults
# ---------------------------------------------------------------------------

DEFAULT_LOHA_LINEAR_DIM: int = 64
DEFAULT_LOHA_LINEAR_ALPHA: int = 128
DEFAULT_LOHA_FACTOR: int = -1
DEFAULT_LOHA_USE_TUCKER: bool = False
DEFAULT_LOHA_USE_SCALAR: bool = False

# ---------------------------------------------------------------------------
# OFT defaults
# ---------------------------------------------------------------------------

DEFAULT_OFT_BLOCK_SIZE: int = 64
DEFAULT_OFT_COFT: bool = False
DEFAULT_OFT_EPS: float = 6e-5

# ---------------------------------------------------------------------------
# VRAM / performance
# ---------------------------------------------------------------------------

DEFAULT_GRADIENT_CHECKPOINTING: bool = True
DEFAULT_GRADIENT_CHECKPOINTING_RATIO: float = 1.0
DEFAULT_OFFLOAD_ENCODER: bool = True

# ---------------------------------------------------------------------------
# Checkpointing / output
# ---------------------------------------------------------------------------

DEFAULT_SAVE_EVERY: int = 50
DEFAULT_SAVE_BEST: bool = True
DEFAULT_SAVE_BEST_AFTER: int = 200
DEFAULT_EARLY_STOP_PATIENCE: int = 0
DEFAULT_STRICT_RESUME: bool = True
DEFAULT_TARGET_LOSS: float = 0.0
DEFAULT_TARGET_LOSS_FLOOR: float = 0.01
DEFAULT_TARGET_LOSS_WARMUP: int = 50
DEFAULT_TARGET_LOSS_SMOOTHING: float = 0.98

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

DEFAULT_LOG_EVERY: int = 10
DEFAULT_LOG_HEAVY_EVERY: int = 50

# ---------------------------------------------------------------------------
# CFG / loss
# ---------------------------------------------------------------------------

DEFAULT_CFG_RATIO: float = 0.15
DEFAULT_LOSS_WEIGHTING: str = "none"
DEFAULT_SNR_GAMMA: float = 5.0
DEFAULT_TIMESTEP_MODE: str = "continuous"

# ---------------------------------------------------------------------------
# Chunking / cropping
# ---------------------------------------------------------------------------

DEFAULT_MAX_LATENT_LENGTH: int = 0
DEFAULT_CHUNK_DECAY_EVERY: int = 10

# ---------------------------------------------------------------------------
# DataLoader (platform-dependent)
# ---------------------------------------------------------------------------

DEFAULT_NUM_WORKERS: int = 0 if sys.platform == "win32" else 4
DEFAULT_PREFETCH_FACTOR: int = 0 if sys.platform == "win32" else 2
DEFAULT_PIN_MEMORY: bool = True
DEFAULT_PERSISTENT_WORKERS: bool = sys.platform != "win32"

# ---------------------------------------------------------------------------
# "All the Levers" (experimental enhancements)
# ---------------------------------------------------------------------------

DEFAULT_EMA_DECAY: float = 0.0
DEFAULT_VAL_SPLIT: float = 0.0
DEFAULT_ADAPTIVE_TIMESTEP_RATIO: float = 0.0
DEFAULT_WARMUP_START_FACTOR: float = 0.1
DEFAULT_COSINE_ETA_MIN_RATIO: float = 0.01
DEFAULT_COSINE_RESTARTS_COUNT: int = 4
DEFAULT_SAVE_BEST_EVERY_N_STEPS: int = 0

# ---------------------------------------------------------------------------
# Model / device
# ---------------------------------------------------------------------------

DEFAULT_MODEL_VARIANT: str = "base"
DEFAULT_ADAPTER_TYPE: str = "lora"
DEFAULT_DEVICE: str = "auto"
DEFAULT_PRECISION: str = "auto"

# ---------------------------------------------------------------------------
# Aggregate dict — backend parameter names → default values.
# Used by /api/defaults endpoint and review_summary _DEFAULTS.
# ---------------------------------------------------------------------------

TRAINING_DEFAULTS: dict = {
    # Model
    "model_variant": DEFAULT_MODEL_VARIANT,
    "adapter_type": DEFAULT_ADAPTER_TYPE,
    "device": DEFAULT_DEVICE,
    "precision": DEFAULT_PRECISION,
    # Training
    "learning_rate": DEFAULT_LEARNING_RATE,
    "batch_size": DEFAULT_BATCH_SIZE,
    "gradient_accumulation": DEFAULT_GRADIENT_ACCUMULATION,
    "epochs": DEFAULT_EPOCHS,
    "warmup_steps": DEFAULT_WARMUP_STEPS,
    "weight_decay": DEFAULT_WEIGHT_DECAY,
    "max_grad_norm": DEFAULT_MAX_GRAD_NORM,
    "seed": DEFAULT_SEED,
    "max_steps": DEFAULT_MAX_STEPS,
    "dataset_repeats": DEFAULT_DATASET_REPEATS,
    # Optimizer / scheduler
    "optimizer_type": DEFAULT_OPTIMIZER_TYPE,
    "scheduler_type": DEFAULT_SCHEDULER_TYPE,
    "scheduler_formula": DEFAULT_SCHEDULER_FORMULA,
    # LoRA
    "rank": DEFAULT_RANK,
    "alpha": DEFAULT_ALPHA,
    "dropout": DEFAULT_DROPOUT,
    "attention_type": DEFAULT_ATTENTION_TYPE,
    "target_mlp": DEFAULT_TARGET_MLP,
    "bias": DEFAULT_BIAS,
    # LoKR
    "lokr_linear_dim": DEFAULT_LOKR_LINEAR_DIM,
    "lokr_linear_alpha": DEFAULT_LOKR_LINEAR_ALPHA,
    "lokr_factor": DEFAULT_LOKR_FACTOR,
    "lokr_decompose_both": DEFAULT_LOKR_DECOMPOSE_BOTH,
    "lokr_use_tucker": DEFAULT_LOKR_USE_TUCKER,
    "lokr_use_scalar": DEFAULT_LOKR_USE_SCALAR,
    "lokr_weight_decompose": DEFAULT_LOKR_WEIGHT_DECOMPOSE,
    # LoHA
    "loha_linear_dim": DEFAULT_LOHA_LINEAR_DIM,
    "loha_linear_alpha": DEFAULT_LOHA_LINEAR_ALPHA,
    "loha_factor": DEFAULT_LOHA_FACTOR,
    "loha_use_tucker": DEFAULT_LOHA_USE_TUCKER,
    "loha_use_scalar": DEFAULT_LOHA_USE_SCALAR,
    # OFT
    "oft_block_size": DEFAULT_OFT_BLOCK_SIZE,
    "oft_coft": DEFAULT_OFT_COFT,
    "oft_eps": DEFAULT_OFT_EPS,
    # VRAM
    "gradient_checkpointing": DEFAULT_GRADIENT_CHECKPOINTING,
    "gradient_checkpointing_ratio": DEFAULT_GRADIENT_CHECKPOINTING_RATIO,
    "offload_encoder": DEFAULT_OFFLOAD_ENCODER,
    # Checkpointing
    "save_every": DEFAULT_SAVE_EVERY,
    "save_best": DEFAULT_SAVE_BEST,
    "save_best_after": DEFAULT_SAVE_BEST_AFTER,
    "early_stop_patience": DEFAULT_EARLY_STOP_PATIENCE,
    "strict_resume": DEFAULT_STRICT_RESUME,
    "target_loss": DEFAULT_TARGET_LOSS,
    "target_loss_floor": DEFAULT_TARGET_LOSS_FLOOR,
    "target_loss_warmup": DEFAULT_TARGET_LOSS_WARMUP,
    "target_loss_smoothing": DEFAULT_TARGET_LOSS_SMOOTHING,
    # Logging
    "log_every": DEFAULT_LOG_EVERY,
    "log_heavy_every": DEFAULT_LOG_HEAVY_EVERY,
    # CFG / loss
    "cfg_ratio": DEFAULT_CFG_RATIO,
    "loss_weighting": DEFAULT_LOSS_WEIGHTING,
    "snr_gamma": DEFAULT_SNR_GAMMA,
    "timestep_mode": DEFAULT_TIMESTEP_MODE,
    # Chunking
    "max_latent_length": DEFAULT_MAX_LATENT_LENGTH,
    "chunk_decay_every": DEFAULT_CHUNK_DECAY_EVERY,
    # DataLoader
    "num_workers": DEFAULT_NUM_WORKERS,
    "prefetch_factor": DEFAULT_PREFETCH_FACTOR,
    "pin_memory": DEFAULT_PIN_MEMORY,
    "persistent_workers": DEFAULT_PERSISTENT_WORKERS,
    # All the Levers
    "ema_decay": DEFAULT_EMA_DECAY,
    "val_split": DEFAULT_VAL_SPLIT,
    "adaptive_timestep_ratio": DEFAULT_ADAPTIVE_TIMESTEP_RATIO,
    "warmup_start_factor": DEFAULT_WARMUP_START_FACTOR,
    "cosine_eta_min_ratio": DEFAULT_COSINE_ETA_MIN_RATIO,
    "cosine_restarts_count": DEFAULT_COSINE_RESTARTS_COUNT,
    "save_best_every_n_steps": DEFAULT_SAVE_BEST_EVERY_N_STEPS,
}

# ---------------------------------------------------------------------------
# GUI key mapping — frontend config key → backend parameter name.
# Keys not listed here are assumed to match the backend name exactly.
# ---------------------------------------------------------------------------

GUI_KEY_MAP: dict = {
    "lr": "learning_rate",
    "learning-rate": "learning_rate",
    "batch-size": "batch_size",
    "gradient-accumulation": "gradient_accumulation",
    "save-every": "save_every",
    "grad_accum": "gradient_accumulation",
    "scheduler": "scheduler_type",
    "early_stop": "early_stop_patience",
    "target-loss": "target_loss",
    "target-loss-floor": "target_loss_floor",
    "target-loss-warmup": "target_loss_warmup",
    "target-loss-smoothing": "target_loss_smoothing",
    "projections": "target_modules",
    "self_projections": "self_target_modules",
    "cross_projections": "cross_target_modules",
}

# ---------------------------------------------------------------------------
# Backend parameter name → GUI field ID.
# Used by /api/defaults to emit defaults keyed the way the frontend expects.
# ---------------------------------------------------------------------------

GUI_FIELD_MAP: dict = {
    "model_variant": "full-model-variant",
    "adapter_type": "full-adapter-type",
    "rank": "full-rank",
    "alpha": "full-alpha",
    "dropout": "full-dropout",
    "lokr_linear_dim": "full-lokr-dim",
    "lokr_linear_alpha": "full-lokr-alpha",
    "lokr_factor": "full-lokr-factor",
    "loha_linear_dim": "full-loha-dim",
    "loha_linear_alpha": "full-loha-alpha",
    "loha_factor": "full-loha-factor",
    "oft_block_size": "full-oft-block-size",
    "oft_eps": "full-oft-eps",
    "attention_type": "full-attention-type",
    "target_mlp": "full-target-mlp",
    "bias": "full-bias",
    "learning_rate": "full-lr",
    "batch_size": "full-batch",
    "gradient_accumulation": "full-grad-accum",
    "epochs": "full-epochs",
    "warmup_steps": "full-warmup",
    "max_steps": "full-max-steps",
    "cfg_ratio": "full-cfg-dropout",
    "loss_weighting": "full-loss-weighting",
    "snr_gamma": "full-snr-gamma",
    "offload_encoder": "full-offload-encoder",
    "gradient_checkpointing_ratio": "full-grad-ckpt-ratio",
    "chunk_decay_every": "full-chunk-decay-every",
    "optimizer_type": "full-optimizer",
    "scheduler_type": "full-scheduler",
    "scheduler_formula": "full-scheduler-formula",
    "max_latent_length": "full-max-latent-length",
    "device": "full-device",
    "precision": "full-precision",
    "save_every": "full-save-every",
    "log_every": "full-log-every",
    "log_heavy_every": "full-log-heavy-every",
    "save_best": "full-save-best",
    "save_best_after": "full-save-best-after",
    "early_stop_patience": "full-early-stop",
    "target_loss": "full-target-loss",
    "target_loss_floor": "full-target-loss-floor",
    "target_loss_warmup": "full-target-loss-warmup",
    "target_loss_smoothing": "full-target-loss-smoothing",
    "strict_resume": "full-strict-resume",
    "weight_decay": "full-weight-decay",
    "max_grad_norm": "full-max-grad-norm",
    "seed": "full-seed",
    "dataset_repeats": "full-dataset-repeats",
    "warmup_start_factor": "full-warmup-start-factor",
    "cosine_eta_min_ratio": "full-cosine-eta-min",
    "cosine_restarts_count": "full-cosine-restarts",
    "ema_decay": "full-ema-decay",
    "val_split": "full-val-split",
    "timestep_mode": "full-timestep-mode",
    "adaptive_timestep_ratio": "full-adaptive-timestep",
    "save_best_every_n_steps": "full-save-best-every-n-steps",
    "num_workers": "full-num-workers",
    "prefetch_factor": "full-prefetch-factor",
    "pin_memory": "full-pin-memory",
    "persistent_workers": "full-persistent-workers",
}


# ---------------------------------------------------------------------------
# Float formatting hints — values that defaults.json expresses in
# scientific notation so the GUI shows e.g. "3e-4" instead of "0.0003".
# ---------------------------------------------------------------------------

_SCI_NOTATION_FIELDS: set = {
    "full-lr",      # 3e-4
    "full-oft-eps", # 6e-5
}

# Fields where we always want a trailing decimal (e.g. "1.0" not "1")
_FORCE_DECIMAL_FIELDS: set = {
    "full-snr-gamma",
    "full-grad-ckpt-ratio",
    "full-max-grad-norm",
}


def _fmt_float(field_id: str, value: float) -> str:
    """Format a float to match the style defaults.json uses for *field_id*."""
    if field_id in _SCI_NOTATION_FIELDS and 0 < abs(value) < 0.01:
        # Compact scientific: "3e-4", "6e-5"
        s = f"{value:.0e}"           # e.g. "3e-04"
        # Strip leading zeros in exponent: "3e-04" → "3e-4"
        base, exp = s.split("e")
        return f"{base}e{int(exp)}"
    if field_id in _FORCE_DECIMAL_FIELDS:
        # Always show one decimal place: "1.0", "5.0"
        return f"{value:.1f}" if value == int(value) else f"{value:g}"
    return f"{value:g}"


def get_gui_defaults() -> dict:
    """Return ``TRAINING_DEFAULTS`` keyed by GUI field IDs.

    Values are converted to strings (matching HTML form value semantics)
    except booleans which stay as ``bool`` for checkbox binding.

    Also includes GUI-only keys (projections, settings paths, model-
    variant-dependent values) so the response is a complete superset
    of ``defaults.json``.
    """
    out: dict = {}
    for backend_key, value in TRAINING_DEFAULTS.items():
        field_id = GUI_FIELD_MAP.get(backend_key)
        if not field_id:
            continue
        if isinstance(value, bool):
            out[field_id] = value
        elif isinstance(value, float):
            out[field_id] = _fmt_float(field_id, value)
        else:
            out[field_id] = str(value)

    # -- Model-variant-dependent defaults (base is the default variant) -----
    out["full-shift"] = "1.0"
    out["full-inference-steps"] = "50"

    # -- UI presentation defaults (no backend equivalent) ------------------
    _projs = "q_proj k_proj v_proj o_proj"
    out["full-projections"] = _projs
    out["full-self-projections"] = _projs
    out["full-cross-projections"] = _projs
    out["full-chunk-duration"] = "0"
    out["full-max-latent-length"] = "0"
    out["full-crop-mode"] = "full"

    # -- Timestep defaults (from model config, not training params) --------
    out["full-timestep-mu"] = "-0.4"
    out["full-timestep-sigma"] = "1.0"
    out["full-timestep-mode"] = DEFAULT_TIMESTEP_MODE

    # -- Empty-string defaults ---------------------------------------------
    out["full-resume-from"] = ""
    out["full-log-dir"] = ""

    # -- Settings path defaults (OS-native separators) ---------------------
    _sep = "\\" if os.name == "nt" else "/"
    out["settings-checkpoint-dir"] = f".{_sep}checkpoints"
    out["settings-adapters-dir"] = f".{_sep}trained_adapters"
    out["settings-tensors-dir"] = f".{_sep}preprocessed_tensors"
    out["settings-audio-dir"] = f".{_sep}my_audio"
    out["settings-exported-loras-dir"] = f".{_sep}exported_loras"

    return out

