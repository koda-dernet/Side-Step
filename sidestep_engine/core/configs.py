"""
Extended Training Configuration for ACE-Step Training V2

Uses vendored base configs from ``_vendor/configs.py`` so Side-Step can run
standalone.  Extends them with corrected-training-specific fields (CFG dropout,
continuous timestep sampling parameters, estimation, TensorBoard, sample
generation, etc.).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import List, Optional

# Vendored base configs -- no base ACE-Step installation required
from sidestep_engine.vendor.configs import (  # noqa: F401
    LoRAConfig,
    LoKRConfig,
    LoHAConfig,
    OFTConfig,
    TrainingConfig,
)


# ---------------------------------------------------------------------------
# Extended LoRA config (unchanged for now, but available for future extension)
# ---------------------------------------------------------------------------

@dataclass
class LoRAConfigV2(LoRAConfig):
    """Extended LoRA configuration.

    Inherits all fields from the original LoRAConfig and adds:
    - attention_type: Which attention layers to target (self, cross, or both)
    """

    attention_type: str = "both"
    """Which attention layers to target: 'self', 'cross', or 'both'."""

    target_mlp: bool = True
    """Also target MLP/FFN layers (gate_proj, up_proj, down_proj)."""

    use_dora: bool = False
    """Enable DoRA (Weight-Decomposed Low-Rank Adaptation).
    Decomposes weight updates into magnitude and direction components."""

    def __post_init__(self) -> None:
        if self.r < 1:
            raise ValueError(f"LoRA rank (r) must be >= 1 (got {self.r})")
        if self.r > 1024:
            raise ValueError(f"LoRA rank (r) > 1024 is almost certainly wrong (got {self.r})")
        if self.alpha < 1:
            raise ValueError(f"LoRA alpha must be >= 1 (got {self.alpha})")

    def to_dict(self) -> dict:
        base = super().to_dict()
        base["attention_type"] = self.attention_type
        base["target_mlp"] = self.target_mlp
        base["use_dora"] = self.use_dora
        return base

    def save_json(self, path: Path) -> None:
        """Persist the adapter config to a JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    @classmethod
    def from_json(cls, path: Path) -> "LoRAConfigV2":
        """Load adapter config from a JSON file."""
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        field_names = set(cls.__dataclass_fields__) | set(LoRAConfig.__dataclass_fields__)
        filtered = {}
        for k, v in data.items():
            mapped_key = {"lora_alpha": "alpha", "lora_dropout": "dropout"}.get(k, k)
            if mapped_key in field_names:
                filtered[mapped_key] = v
        return cls(**filtered)

    # --- Data loading (declared here for compatibility with base packages
    #     that may not include these fields in TrainingConfig) -----------------
    num_workers: int = 4
    """Number of DataLoader worker processes."""

    pin_memory: bool = True
    """Pin memory in DataLoader for faster host-to-device transfer."""

    prefetch_factor: int = 2
    """Number of batches to prefetch per DataLoader worker."""

    persistent_workers: bool = True
    """Keep DataLoader workers alive between epochs."""

    pin_memory_device: str = ""
    """Device for pinned memory ("" = default CUDA device)."""


# ---------------------------------------------------------------------------
# Extended LoKR config
# ---------------------------------------------------------------------------

@dataclass
class LoKRConfigV2(LoKRConfig):
    """Extended LoKR configuration.

    Inherits all fields from the original LoKRConfig and adds:
    - attention_type: Which attention layers to target (self, cross, or both)
    """

    attention_type: str = "both"
    """Which attention layers to target: 'self', 'cross', or 'both'."""

    target_mlp: bool = True
    """Also target MLP/FFN layers (gate_proj, up_proj, down_proj)."""

    def to_dict(self) -> dict:
        base = super().to_dict()
        base["attention_type"] = self.attention_type
        base["target_mlp"] = self.target_mlp
        return base

    def save_json(self, path: Path) -> None:
        """Persist the adapter config to a JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    @classmethod
    def from_json(cls, path: Path) -> "LoKRConfigV2":
        """Load adapter config from a JSON file."""
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        field_names = set(cls.__dataclass_fields__) | set(LoKRConfig.__dataclass_fields__)
        return cls(**{k: v for k, v in data.items() if k in field_names})


# ---------------------------------------------------------------------------
# Extended LoHA config
# ---------------------------------------------------------------------------

@dataclass
class LoHAConfigV2(LoHAConfig):
    """Extended LoHA configuration.

    Inherits all fields from the original LoHAConfig and adds:
    - attention_type: Which attention layers to target (self, cross, or both)
    - target_mlp: Also target MLP/FFN layers
    """

    attention_type: str = "both"
    """Which attention layers to target: 'self', 'cross', or 'both'."""

    target_mlp: bool = True
    """Also target MLP/FFN layers (gate_proj, up_proj, down_proj)."""

    def to_dict(self) -> dict:
        """Serialize to dictionary."""
        base = super().to_dict()
        base["attention_type"] = self.attention_type
        base["target_mlp"] = self.target_mlp
        return base

    def save_json(self, path: Path) -> None:
        """Persist the adapter config to a JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    @classmethod
    def from_json(cls, path: Path) -> "LoHAConfigV2":
        """Load adapter config from a JSON file."""
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        field_names = set(cls.__dataclass_fields__) | set(LoHAConfig.__dataclass_fields__)
        return cls(**{k: v for k, v in data.items() if k in field_names})


# ---------------------------------------------------------------------------
# Extended OFT config
# ---------------------------------------------------------------------------

@dataclass
class OFTConfigV2(OFTConfig):
    """Extended OFT configuration.

    Inherits all fields from the original OFTConfig and adds:
    - attention_type: Which attention layers to target (self, cross, or both)
    - target_mlp: Also target MLP/FFN layers

    OFT is experimental for audio DiT models.
    """

    attention_type: str = "both"
    """Which attention layers to target: 'self', 'cross', or 'both'."""

    target_mlp: bool = True
    """Also target MLP/FFN layers (gate_proj, up_proj, down_proj)."""

    def to_dict(self) -> dict:
        """Serialize to dictionary."""
        base = super().to_dict()
        base["attention_type"] = self.attention_type
        base["target_mlp"] = self.target_mlp
        return base

    def save_json(self, path: Path) -> None:
        """Persist the adapter config to a JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    @classmethod
    def from_json(cls, path: Path) -> "OFTConfigV2":
        """Load adapter config from a JSON file."""
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        field_names = set(cls.__dataclass_fields__) | set(OFTConfig.__dataclass_fields__)
        return cls(**{k: v for k, v in data.items() if k in field_names})


# ---------------------------------------------------------------------------
# Extended Training config
# ---------------------------------------------------------------------------

@dataclass
class TrainingConfigV2(TrainingConfig):
    """Extended training configuration with corrected-training fields.

    New fields compared to the original TrainingConfig:
    - CFG dropout (cfg_ratio)
    - Continuous timestep sampling parameters (timestep_mu, timestep_sigma,
      data_proportion)
    - Model variant selection
    - Device / precision auto-detection
    - Estimation parameters
    - Extended TensorBoard logging
    - Sample generation during training
    - Checkpoint resume
    - Preprocessing flags
    """

    # --- Data loading (declared here for compatibility with base packages
    #     that may not include these fields in TrainingConfig) -----------------
    num_workers: int = 4
    """Number of DataLoader worker processes."""

    pin_memory: bool = True
    """Pin memory in DataLoader for faster host-to-device transfer."""

    prefetch_factor: int = 2
    """Number of batches to prefetch per DataLoader worker."""

    persistent_workers: bool = True
    """Keep DataLoader workers alive between epochs."""

    pin_memory_device: str = ""
    """Device for pinned memory ("" = default CUDA device)."""

    # --- Optimizer / Scheduler ------------------------------------------------
    optimizer_type: str = "adamw"
    """Optimizer: 'adamw', 'adamw8bit', 'adafactor', 'prodigy'."""

    scheduler_type: str = "cosine"
    """LR scheduler: 'cosine', 'cosine_restarts', 'linear', 'constant',
    'constant_with_warmup', or 'custom' (uses scheduler_formula)."""

    scheduler_formula: str = ""
    """Custom LR formula (Python math expression).  Only used when
    ``scheduler_type='custom'``.  The formula controls the post-warmup
    curve; warmup is auto-prepended.  See ``formula_scheduler.py``."""

    # --- VRAM management ------------------------------------------------------
    gradient_checkpointing: bool = True
    """Trade compute for memory by recomputing activations during backward.
    Enabled by default to match ACE-Step's behaviour and save ~40-60%
    activation VRAM.  Adds ~10-30% training time overhead."""

    gradient_checkpointing_ratio: float = 1.0
    """Fraction of decoder layers to checkpoint (0.0=none, 0.5=half, 1.0=all).
    Only applies when ``gradient_checkpointing=True``.  Lower values trade
    VRAM savings for speed.  0.5 is a good middle ground."""

    offload_encoder: bool = True
    """Move encoder/VAE to CPU after setup to free ~2-4 GB VRAM.
    Enabled by default to prevent OOM on consumer GPUs (8-12 GB)."""

    vram_profile: str = "auto"
    """VRAM preset: 'auto', 'comfortable', 'standard', 'tight', 'minimal'."""

    # --- Corrected training params ------------------------------------------
    cfg_ratio: float = 0.15
    """Classifier-free guidance dropout probability."""

    timestep_mu: float = -0.4
    """Mean for logit-normal timestep sampling (from model config)."""

    timestep_sigma: float = 1.0
    """Std for logit-normal timestep sampling (from model config)."""

    data_proportion: float = 0.5
    """Data proportion for sample_t_r (from model config)."""

    loss_weighting: str = "none"
    """Loss weighting strategy: 'none' (flat MSE) or 'min_snr'.
    min-SNR can yield better results on SFT and base models."""

    snr_gamma: float = 5.0
    """Gamma clamp for min-SNR weighting. Only used when loss_weighting='min_snr'."""

    # --- Adapter selection ----------------------------------------------------
    adapter_type: str = "lora"
    """Adapter type: 'lora', 'dora', 'lokr', 'loha', or 'oft'."""

    # --- Model variant detection ---------------------------------------------
    is_turbo: bool = False
    """Auto-detected: ``True`` when the model is turbo or a turbo-based
    fine-tune (``num_inference_steps == 8``).  Controls whether training
    uses discrete 8-step sampling (turbo) or continuous logit-normal
    sampling + CFG dropout (base/sft).  Not user-facing."""

    # --- Model / paths ------------------------------------------------------
    model_variant: str = "turbo"
    """Model variant: 'turbo', 'base', or 'sft'."""

    checkpoint_dir: str = "./checkpoints"
    """Path to checkpoints root directory."""

    dataset_dir: str = ""
    """Directory containing preprocessed .pt tensor files."""

    # --- Device / precision -------------------------------------------------
    device: str = "auto"
    """Device selection: 'auto', 'cuda', 'cuda:0', 'mps', 'xpu', 'cpu'."""

    precision: str = "auto"
    """Precision: 'auto', 'bf16', 'fp16', 'fp32'."""

    # --- Checkpointing ------------------------------------------------------
    resume_from: Optional[str] = None
    """Path to checkpoint directory to resume training from."""

    strict_resume: bool = True
    """When True, abort if critical checkpoint state (optimizer/scheduler)
    cannot be restored or if config has changed since the checkpoint.
    When False, warn and continue with partial restore."""

    run_name: Optional[str] = None
    """User-chosen name for this training run.  Used for output directory
    naming, TensorBoard log versioning, and session artifacts."""

    save_best: bool = True
    """Auto-save the adapter with the lowest smoothed loss (MA5)."""

    save_best_after: int = 200
    """Epoch at which best-model tracking activates.  Earlier epochs are
    ignored because loss is typically noisy during warmup."""

    early_stop_patience: int = 0
    """Stop training if smoothed loss doesn't improve for this many epochs
    after best-model tracking is active.  0 = disabled."""

    # --- Extended TensorBoard logging ---------------------------------------
    log_dir: Optional[str] = None
    """TensorBoard log directory.  Defaults to {output_dir}/runs."""

    log_every: int = 10
    """Log basic metrics (loss, LR) every N optimiser steps."""

    log_heavy_every: int = 50
    """Log per-layer gradient norms every N optimiser steps."""

    # --- Fisher analysis params -----------------------------------------------
    fisher_runs: Optional[int] = None
    """Number of Fisher estimation runs (None = auto-scale with dataset)."""

    fisher_batches_per_run: Optional[int] = None
    """Batches per Fisher run (None = auto-scale with dataset)."""

    convergence_patience: int = 5
    """Stop a Fisher run early when top-K ranking is stable for N batches."""

    timestep_focus: str = "balanced"
    """Timestep focus for Fisher: 'texture', 'structure', 'balanced', or 'low,high'."""

    rank_min: int = 16
    """Minimum adaptive LoRA rank."""

    rank_max: int = 128
    """Maximum adaptive LoRA rank."""

    ignore_fisher_map: bool = False
    """Bypass auto-detection of fisher_map.json in dataset_dir."""

    # --- Preprocessing flags ------------------------------------------------
    preprocess: bool = False
    """Run preprocessing before training."""

    audio_dir: Optional[str] = None
    """Source audio directory for preprocessing."""

    dataset_json: Optional[str] = None
    """Labeled dataset JSON for preprocessing."""

    tensor_output: Optional[str] = None
    """Output directory for preprocessed .pt tensor files."""

    max_duration: float = 0
    """Maximum audio duration in seconds (0 = auto-detect, preprocessing)."""

    normalize: str = "none"
    """Audio normalization: 'none', 'peak' (-1.0 dBFS), 'lufs' (-14 LUFS)."""

    chunk_duration: Optional[int] = None
    """Random latent chunking: extract a random window of this many seconds
    from each sample during training.  ``None`` = disabled, ``60`` = recommended.
    Values below 60 (e.g. 30) may reduce training quality for full-length
    inference."""

    chunk_decay_every: int = 10
    """Epoch interval for halving the chunk coverage histogram.
    Controls how quickly previously-trained regions become eligible again.
    ``0`` disables decay entirely.  Default ``10``."""

    dataset_repeats: int = 1
    """Global dataset repetition multiplier.  Each sample appears this many
    times per epoch.  1 = no repetition (default).  Higher values effectively
    multiply dataset size without modifying ``.pt`` files."""

    max_steps: int = 0
    """Maximum optimizer steps.  0 = disabled (use max_epochs only).
    When > 0, training stops at this step count regardless of epoch."""

    save_best_every_n_steps: int = 0
    """Check for step-level best model every N optimizer steps.
    0 = epoch-only best tracking (default).  When > 0, a rolling smoothed
    loss is checked every N steps and the best model is saved mid-epoch."""

    # --- "All the Levers" enhancements ------------------------------------

    ema_decay: float = 0.0
    """EMA decay rate for adapter weights.  0.0 = disabled (default).
    Typical value: 0.9999.  Maintains a smoothed shadow copy of trainable
    params on CPU; the EMA weights are used for best-model saves."""

    val_split: float = 0.0
    """Fraction of dataset held out for validation.  0.0 = disabled.
    When > 0, a no-grad validation pass runs each epoch and val loss
    replaces train MA5 for best-model selection."""

    adaptive_timestep_ratio: float = 0.0
    """Fraction of timesteps from loss-weighted adaptive distribution.
    0.0 = pure logit-normal (default).  0.3 = 30% adaptive + 70% base.
    Only applies to base/SFT continuous sampling; turbo is unaffected."""

    warmup_start_factor: float = 0.1
    """LR warmup ramps from ``base_lr * warmup_start_factor`` to ``base_lr``.
    Default 0.1 matches the current hardcoded value in ``optim.py``."""

    cosine_eta_min_ratio: float = 0.01
    """Cosine scheduler decays LR to ``base_lr * cosine_eta_min_ratio``.
    Default 0.01 matches the current hardcoded value in ``optim.py``."""

    cosine_restarts_count: int = 4
    """Number of cosine restart cycles (``cosine_restarts`` scheduler only).
    Default 4 matches the current hardcoded value in ``optim.py``."""

    # -----------------------------------------------------------------------
    # Validation
    # -----------------------------------------------------------------------

    def __post_init__(self) -> None:
        super().__post_init__()
        errors: list[str] = []
        if self.learning_rate <= 0:
            errors.append(f"learning_rate must be > 0 (got {self.learning_rate})")
        if self.learning_rate > 1.0:
            errors.append(
                f"learning_rate={self.learning_rate} is dangerously high -- "
                "typical range is 1e-5..1e-3"
            )
        if self.batch_size < 1:
            errors.append(f"batch_size must be >= 1 (got {self.batch_size})")
        if self.max_grad_norm <= 0:
            errors.append(f"max_grad_norm must be > 0 (got {self.max_grad_norm})")
        if self.max_epochs < 1:
            errors.append(f"max_epochs must be >= 1 (got {self.max_epochs})")
        if self.gradient_accumulation_steps < 1:
            errors.append(
                f"gradient_accumulation_steps must be >= 1 "
                f"(got {self.gradient_accumulation_steps})"
            )
        if self.save_every_n_epochs < 1:
            errors.append(
                f"save_every_n_epochs must be >= 1 (got {self.save_every_n_epochs})"
            )
        if not (0.0 <= self.ema_decay < 1.0):
            errors.append(f"ema_decay must be >= 0 and < 1 (got {self.ema_decay})")
        if not (0.0 <= self.val_split <= 0.5):
            errors.append(f"val_split must be >= 0 and <= 0.5 (got {self.val_split})")
        if not (0.0 <= self.adaptive_timestep_ratio <= 1.0):
            errors.append(
                f"adaptive_timestep_ratio must be >= 0 and <= 1 "
                f"(got {self.adaptive_timestep_ratio})"
            )
        if not (0.0 < self.warmup_start_factor <= 1.0):
            errors.append(
                f"warmup_start_factor must be > 0 and <= 1 "
                f"(got {self.warmup_start_factor})"
            )
        if not (0.0 <= self.cosine_eta_min_ratio <= 1.0):
            errors.append(
                f"cosine_eta_min_ratio must be >= 0 and <= 1 "
                f"(got {self.cosine_eta_min_ratio})"
            )
        if self.cosine_restarts_count < 1:
            errors.append(
                f"cosine_restarts_count must be >= 1 "
                f"(got {self.cosine_restarts_count})"
            )
        if errors:
            raise ValueError(
                "Invalid training configuration:\n  - " + "\n  - ".join(errors)
            )

    # -----------------------------------------------------------------------
    # Helpers
    # -----------------------------------------------------------------------

    @property
    def effective_log_dir(self) -> Path:
        """Return the resolved TensorBoard log directory.

        When ``run_name`` is set and ``resume_from`` is not set, uses
        versioned subdirectory naming (``{run_name}_v0``, ``_v1``, ...).
        When resuming, falls back to the non-versioned directory so
        TensorBoard curves stay continuous.

        The result is cached after first access to avoid creating
        multiple versioned directories on repeated calls.
        """
        cached = getattr(self, "_effective_log_dir_cache", None)
        if cached is not None:
            return cached
        has_custom_log_root = self.log_dir is not None and str(self.log_dir).strip() != ""
        log_root = Path(self.log_dir) if has_custom_log_root else Path(self.output_dir) / "runs"
        if self.run_name and not self.resume_from:
            if has_custom_log_root:
                from sidestep_engine.logging.tensorboard_utils import resolve_versioned_log_dir

                result = resolve_versioned_log_dir(log_root, self.run_name)
            else:
                # Default GUI path: each timestamped run_name gets its own stable
                # run-local TensorBoard directory under output_dir/runs/<run_name>.
                result = log_root / self.run_name
                result.mkdir(parents=True, exist_ok=True)
        else:
            # Resume continuity: if custom log_dir was explicitly restored from
            # the original run config, use it; otherwise continue default
            # run-local path under output_dir/runs/<run_name>.
            if self.resume_from and self.run_name:
                if has_custom_log_root:
                    from sidestep_engine.logging.tensorboard_utils import resolve_latest_versioned_log_dir

                    latest = resolve_latest_versioned_log_dir(log_root, self.run_name)
                    result = latest if latest is not None else log_root
                else:
                    result = log_root / self.run_name
            else:
                result = log_root
        object.__setattr__(self, "_effective_log_dir_cache", result)
        return result

    def to_dict(self) -> dict:
        """Serialize every field, including new ones."""
        base = super().to_dict()
        base.update(
            {
                "num_workers": self.num_workers,
                "pin_memory": self.pin_memory,
                "prefetch_factor": self.prefetch_factor,
                "persistent_workers": self.persistent_workers,
                "pin_memory_device": self.pin_memory_device,
                "optimizer_type": self.optimizer_type,
                "scheduler_type": self.scheduler_type,
                "scheduler_formula": self.scheduler_formula,
                "gradient_checkpointing": self.gradient_checkpointing,
                "gradient_checkpointing_ratio": self.gradient_checkpointing_ratio,
                "offload_encoder": self.offload_encoder,
                "vram_profile": self.vram_profile,
                "adapter_type": self.adapter_type,
                "cfg_ratio": self.cfg_ratio,
                "timestep_mu": self.timestep_mu,
                "timestep_sigma": self.timestep_sigma,
                "data_proportion": self.data_proportion,
                "loss_weighting": self.loss_weighting,
                "snr_gamma": self.snr_gamma,
                "is_turbo": self.is_turbo,
                "model_variant": self.model_variant,
                "checkpoint_dir": self.checkpoint_dir,
                "dataset_dir": self.dataset_dir,
                "device": self.device,
                "precision": self.precision,
                "resume_from": self.resume_from,
                "strict_resume": self.strict_resume,
                "run_name": self.run_name,
                "save_best": self.save_best,
                "save_best_after": self.save_best_after,
                "early_stop_patience": self.early_stop_patience,
                "log_dir": self.log_dir,
                "log_every": self.log_every,
                "log_heavy_every": self.log_heavy_every,
                "preprocess": self.preprocess,
                "audio_dir": self.audio_dir,
                "dataset_json": self.dataset_json,
                "tensor_output": self.tensor_output,
                "max_duration": self.max_duration,
                "normalize": self.normalize,
                "chunk_duration": self.chunk_duration,
                "dataset_repeats": self.dataset_repeats,
                "fisher_runs": self.fisher_runs,
                "fisher_batches_per_run": self.fisher_batches_per_run,
                "convergence_patience": self.convergence_patience,
                "timestep_focus": self.timestep_focus,
                "rank_min": self.rank_min,
                "rank_max": self.rank_max,
                "ignore_fisher_map": self.ignore_fisher_map,
                "ema_decay": self.ema_decay,
                "val_split": self.val_split,
                "adaptive_timestep_ratio": self.adaptive_timestep_ratio,
                "warmup_start_factor": self.warmup_start_factor,
                "cosine_eta_min_ratio": self.cosine_eta_min_ratio,
                "cosine_restarts_count": self.cosine_restarts_count,
            }
        )
        return base

    def save_json(self, path: Path) -> None:
        """Persist the full config to a JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    @classmethod
    def from_json(cls, path: Path) -> "TrainingConfigV2":
        """Load config from a JSON file."""
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})
