"""
FixedLoRATrainer -- Orchestration for ACE-Step V2 adapter fine-tuning.

The actual per-step training logic lives in ``fixed_lora_module.py``
(``FixedLoRAModule``).  Checkpoint, memory, and verification helpers
live in ``trainer_helpers.py``.

Supports both adapter types:
    - **LoRA** via PEFT (``inject_lora_into_dit``)
    - **LoKR** via LyCORIS (``inject_lokr_into_dit``)

Uses vendored copies of ACE-Step utilities from ``_vendor/``.
"""

from __future__ import annotations

import logging
import math
import random
import sys
import time
from pathlib import Path
from typing import Any, Dict, Generator, Optional, Tuple

import torch
import torch.nn as nn
from sidestep_engine.core.optim import (
    build_layer_param_groups,
    build_optimizer,
    build_scheduler,
    classify_trainable_params,
)
from sidestep_engine.vendor.data_module import PreprocessedDataModule

_MAX_CONSECUTIVE_NAN = 10  # halt training after this many NaN/Inf losses in a row

# V2 modules
from sidestep_engine.core.configs import TrainingConfigV2
from sidestep_engine.logging.tensorboard_utils import TrainingLogger
from sidestep_engine.core.types import TrainingUpdate

# Split-out modules
from sidestep_engine.core.lora_module import (
    AdapterConfig,
    FixedLoRAModule,
    _normalize_device_type,
    _select_compute_dtype,
)
from sidestep_engine.core.trainer_helpers import (
    capture_rng_state,
    configure_memory_features,
    force_disable_decoder_cache,
    offload_non_decoder,
    restore_rng_state,
    resume_checkpoint,
    save_adapter_flat,
    save_checkpoint,
    save_final,
    save_on_early_exit,
    verify_saved_adapter,
)
logger = logging.getLogger(__name__)


def _collect_trainable_params(
    model: nn.Module, lycoris_net: Optional[nn.Module] = None,
) -> list[torch.nn.Parameter]:
    """Collect trainable params with LyCORIS fallback.

    LyCORIS-injected params may not appear in ``model.parameters()``
    depending on the wrapper chain.  Fall back to the LyCORIS network
    directly so the optimizer always receives the correct parameter list.
    """
    params = [p for p in model.parameters() if p.requires_grad]
    if params:
        return list({id(p): p for p in params}.values())

    if lycoris_net is None:
        return []

    fallback: list[torch.nn.Parameter] = []
    for m in getattr(lycoris_net, "loras", []) or []:
        for p in m.parameters():
            if p.requires_grad:
                fallback.append(p)
    if not fallback:
        for p in lycoris_net.parameters():
            if p.requires_grad:
                fallback.append(p)

    if fallback:
        logger.info(
            "[Side-Step] model.parameters() missed adapter params; "
            "recovered %d from LyCORIS network",
            len(fallback),
        )
    return list({id(p): p for p in fallback}.values())


# ===========================================================================
# FixedLoRATrainer -- orchestration
# ===========================================================================

class FixedLoRATrainer:
    """High-level trainer for corrected ACE-Step adapter fine-tuning.

    Supports both LoRA (PEFT) and LoKR (LyCORIS) adapters.
    Uses pure PyTorch with bf16-true precision.
    """

    def __init__(
        self,
        model: nn.Module,
        adapter_config: AdapterConfig,
        training_config: TrainingConfigV2,
    ) -> None:
        self.model = model
        self.adapter_config = adapter_config
        self.training_config = training_config
        self.adapter_type = training_config.adapter_type

        # Backward-compat alias
        self.lora_config = adapter_config

        self.module: Optional[FixedLoRAModule] = None
        self.is_training = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def train(
        self,
        training_state: Optional[Dict[str, Any]] = None,
    ) -> Generator[Tuple[int, float, str], None, None]:
        """Run the full training loop.

        Yields ``(global_step, loss, status_message)`` tuples.
        """
        self.is_training = True
        cfg = self.training_config

        try:
            # -- Validate ---------------------------------------------------
            ds_dir = Path(cfg.dataset_dir)
            if not ds_dir.is_dir():
                yield TrainingUpdate(0, 0.0, f"[FAIL] Dataset directory not found: {ds_dir}", kind="fail")
                return

            # -- Seed (may be overridden by checkpoint RNG restore) ----------
            self._rng_seeded_fresh = True
            torch.manual_seed(cfg.seed)
            random.seed(cfg.seed)
            if torch.cuda.is_available():
                torch.cuda.manual_seed_all(cfg.seed)

            # -- Build module -----------------------------------------------
            device = torch.device(cfg.device)
            dtype = _select_compute_dtype(
                _normalize_device_type(device),
                getattr(cfg, "precision", "auto"),
            )

            self.module = FixedLoRAModule(
                model=self.model,
                adapter_config=self.adapter_config,
                training_config=cfg,
                device=device,
                dtype=dtype,
            )

            # -- Channel stats for fidelity balancing ----------------------
            if not cfg.legacy_loss:
                from sidestep_engine.vendor.data_module import PreprocessedTensorDataset
                ch_stats = PreprocessedTensorDataset.load_channel_stats(cfg.dataset_dir)
                if ch_stats is not None:
                    ch_std = torch.tensor(ch_stats["channel_std"], dtype=torch.float32)
                    # Build channel weights: inverse-std, optionally scaled by VAE importance
                    inv_std = 1.0 / ch_std.clamp(min=1e-6)
                    if cfg.vae_channel_prior and "vae_channel_norms" in ch_stats:
                        vae_imp = torch.tensor(ch_stats["vae_channel_norms"], dtype=torch.float32)
                        vae_imp = vae_imp / vae_imp.mean().clamp(min=1e-6)
                        channel_weights = inv_std * vae_imp
                    else:
                        channel_weights = inv_std
                    channel_weights = channel_weights / channel_weights.mean()  # normalize to mean=1
                    self.module._channel_weights = channel_weights
                    self.module._channel_std = ch_std
                    logger.info(
                        "[Side-Step] Channel balancing active: weight range %.3f-%.3f (mean=1)",
                        float(channel_weights.min()), float(channel_weights.max()),
                    )
                else:
                    logger.info("[Side-Step] No channel_stats found -- channel balancing disabled")

            # -- Data -------------------------------------------------------
            num_workers = cfg.num_workers
            if sys.platform == "win32" and num_workers > 0:
                # Windows uses 'spawn' for multiprocessing -- works but has
                # higher per-worker overhead.  Cap at a sensible ceiling.
                _win_cap = min(num_workers, 4)
                if _win_cap != num_workers:
                    logger.info("[Side-Step] Windows: capping num_workers=%d -> %d", num_workers, _win_cap)
                    num_workers = _win_cap

            data_module = PreprocessedDataModule(
                tensor_dir=cfg.dataset_dir,
                batch_size=cfg.batch_size,
                num_workers=num_workers,
                pin_memory=cfg.pin_memory,
                prefetch_factor=cfg.prefetch_factor if num_workers > 0 else None,
                persistent_workers=cfg.persistent_workers if num_workers > 0 else False,
                pin_memory_device=cfg.pin_memory_device,
                val_split=getattr(cfg, "val_split", 0.0),
                chunk_duration=getattr(cfg, "chunk_duration", None),
                max_latent_length=getattr(cfg, "max_latent_length", None),
                chunk_decay_every=getattr(cfg, "chunk_decay_every", 10),
                dataset_repeats=getattr(cfg, "dataset_repeats", 1),
                genre_ratio=getattr(cfg, "genre_ratio", -1),
            )
            data_module.setup("fit")

            if len(data_module.train_dataset) == 0:
                fail_msg = "[FAIL] No valid samples found in dataset directory"
                ds = data_module.train_dataset
                manifest_path = getattr(ds, "manifest_path", None)
                manifest_error = getattr(ds, "manifest_error", None)
                fallback_used = bool(getattr(ds, "manifest_fallback_used", False))

                if manifest_error:
                    fail_msg += f"\n       {manifest_error}"
                elif manifest_path and Path(manifest_path).is_file() and fallback_used:
                    fail_msg += (
                        "\n       manifest.json could not be used; fallback scan found no valid .pt files"
                    )
                elif manifest_path and not Path(manifest_path).is_file():
                    fail_msg += "\n       manifest.json not found and directory contains no valid .pt files"

                yield TrainingUpdate(0, 0.0, fail_msg, kind="fail")
                return

            yield TrainingUpdate(0, 0.0, f"[OK] Loaded {len(data_module.train_dataset)} preprocessed samples", kind="info")

            # -- Training loop ----------------------------------------------
            yield from self._train_loop(data_module, training_state)

        except Exception as exc:
            logger.exception("Training failed")
            yield TrainingUpdate(0, 0.0, f"[FAIL] Training failed: {exc}", kind="fail")
        finally:
            self.is_training = False

    def stop(self) -> None:
        self.is_training = False

    # ------------------------------------------------------------------
    # Delegate helpers (thin wrappers around trainer_helpers functions)
    # ------------------------------------------------------------------

    @staticmethod
    def _iter_module_wrappers(module: nn.Module) -> list:
        from sidestep_engine.core.trainer_helpers import iter_module_wrappers
        return iter_module_wrappers(module)

    @classmethod
    def _configure_memory_features(cls, decoder: nn.Module) -> tuple:
        return configure_memory_features(decoder)

    @staticmethod
    def _offload_non_decoder(model: nn.Module) -> int:
        return offload_non_decoder(model)

    def _save_adapter_flat(self, output_dir: str) -> None:
        save_adapter_flat(self, output_dir)

    def _save_checkpoint(
        self, optimizer: Any, scheduler: Any, epoch: int, global_step: int, ckpt_dir: str,
        runtime_state: Optional[Dict[str, Any]] = None,
    ) -> None:
        save_checkpoint(self, optimizer, scheduler, epoch, global_step, ckpt_dir, runtime_state)

    def _save_final(self, output_dir: str) -> None:
        save_final(self, output_dir)

    @staticmethod
    def _verify_saved_adapter(output_dir: str) -> None:
        verify_saved_adapter(output_dir)

    def _resume_checkpoint(
        self, resume_path: str, optimizer: Any, scheduler: Any,
    ) -> Generator[TrainingUpdate, None, Optional[Tuple[int, int]]]:
        return (yield from resume_checkpoint(self, resume_path, optimizer, scheduler))

    # ------------------------------------------------------------------
    # Training loop
    # ------------------------------------------------------------------

    def _train_loop(
        self,
        data_module: PreprocessedDataModule,
        training_state: Optional[Dict[str, Any]],
    ) -> Generator[TrainingUpdate, None, None]:
        cfg = self.training_config
        assert self.module is not None

        output_dir = Path(cfg.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        from sidestep_engine.core.progress_writer import ProgressWriter
        _pw = ProgressWriter(output_dir)

        device_type = self.module.device_type

        # -- Device setup ---------------------------------------------------
        if device_type == "cuda":
            torch.cuda.set_device(self.module.device)

        yield TrainingUpdate(0, 0.0, f"[INFO] Starting training (device: {device_type}, dtype: {self.module.dtype})", kind="info")

        # -- TensorBoard logger ---------------------------------------------
        tb = TrainingLogger(cfg.effective_log_dir)

        # -- Dataloader -----------------------------------------------------
        train_loader = data_module.train_dataloader()

        # -- Trainable params / optimizer -----------------------------------
        trainable_params = _collect_trainable_params(
            self.module.model,
            getattr(self.module, "lycoris_net", None),
        )
        if not trainable_params:
            yield TrainingUpdate(0, 0.0, "[FAIL] No trainable parameters found", kind="fail")
            tb.close()
            return

        yield TrainingUpdate(0, 0.0, f"[INFO] Training {sum(p.numel() for p in trainable_params):,} parameters", kind="info")

        # Classify params by layer type and build per-group LR scaling
        classified = classify_trainable_params(
            self.module.model,
            getattr(self.module, "lycoris_net", None),
        )
        optimizer_params = build_layer_param_groups(
            classified,
            base_lr=cfg.learning_rate,
            lr_scale_self_attn=getattr(cfg, "lr_scale_self_attn", 1.0),
            lr_scale_cross_attn=getattr(cfg, "lr_scale_cross_attn", 1.0),
            lr_scale_mlp=getattr(cfg, "lr_scale_mlp", 1.0),
        )

        optimizer_type = getattr(cfg, "optimizer_type", "adamw")
        optimizer = build_optimizer(
            optimizer_params,
            optimizer_type=optimizer_type,
            lr=cfg.learning_rate,
            weight_decay=cfg.weight_decay,
            device_type=self.module.device.type,
        )
        yield TrainingUpdate(0, 0.0, f"[INFO] Optimizer: {optimizer_type}", kind="info")

        # -- Scheduler -------------------------------------------------------
        steps_per_epoch = max(1, math.ceil(len(train_loader) / cfg.gradient_accumulation_steps))
        total_steps = steps_per_epoch * cfg.max_epochs

        scheduler_type = getattr(cfg, "scheduler_type", "cosine")
        scheduler = build_scheduler(
            optimizer,
            scheduler_type=scheduler_type,
            total_steps=total_steps,
            warmup_steps=cfg.warmup_steps,
            lr=cfg.learning_rate,
            optimizer_type=optimizer_type,
            n_restarts=getattr(cfg, "cosine_restarts_count", 4),
            formula=getattr(cfg, "scheduler_formula", ""),
            steps_per_epoch=steps_per_epoch,
            total_epochs=cfg.max_epochs,
            warmup_start_factor=getattr(cfg, "warmup_start_factor", 0.1),
            cosine_eta_min_ratio=getattr(cfg, "cosine_eta_min_ratio", 0.01),
        )
        yield TrainingUpdate(0, 0.0, f"[INFO] Scheduler: {scheduler_type}", kind="info")

        # -- "All the Levers" optional enhancements --------------------------
        _ema = None
        _ema_decay = getattr(cfg, "ema_decay", 0.0)
        if _ema_decay > 0:
            from sidestep_engine.core.ema import AdapterEMA
            _ema_start = getattr(cfg, "ema_start_step", 2000)
            _ema = AdapterEMA(trainable_params, decay=_ema_decay, start_step=_ema_start)
            yield TrainingUpdate(
                0, 0.0,
                f"[INFO] EMA enabled (decay={_ema_decay}, starts at step {_ema_start})",
                kind="info",
            )

        _val_loader = None
        _val_split = getattr(cfg, "val_split", 0.0)
        if _val_split > 0:
            _val_loader = data_module.val_dataloader()
            if _val_loader is not None:
                n_val = len(data_module.val_dataset) if data_module.val_dataset else 0
                yield TrainingUpdate(
                    0, 0.0,
                    f"[INFO] Validation enabled ({n_val} samples, {_val_split:.0%} split)",
                    kind="info",
                )
            else:
                yield TrainingUpdate(
                    0, 0.0,
                    "[WARN] val_split > 0 but validation loader is empty -- validation disabled",
                    kind="warn",
                )

        _adaptive_sampler = None
        _adaptive_ratio = getattr(cfg, "adaptive_timestep_ratio", 0.0)
        if _adaptive_ratio > 0 and not getattr(cfg, "is_turbo", False):
            from sidestep_engine.core.adaptive_timestep import AdaptiveTimestepSampler
            _adaptive_sampler = AdaptiveTimestepSampler(ratio=_adaptive_ratio)
            self.module._adaptive_sampler = _adaptive_sampler
            yield TrainingUpdate(
                0, 0.0,
                f"[INFO] Adaptive timestep sampling enabled (ratio={_adaptive_ratio})",
                kind="info",
            )

        # -- Training memory features ----------------------------------------
        cache_forced_off = force_disable_decoder_cache(self.module.model.decoder)
        if cache_forced_off:
            yield TrainingUpdate(
                0, 0.0,
                "[INFO] Disabled decoder KV cache for training VRAM stability",
                kind="info",
            )

        if getattr(cfg, "gradient_checkpointing", True):
            ratio = getattr(cfg, "gradient_checkpointing_ratio", 1.0)
            if ratio < 1.0 and ratio > 0.0:
                # Selective: checkpoint only a fraction of layers
                from sidestep_engine.core.selective_checkpointing import (
                    apply_selective_checkpointing,
                )
                n_ckpt = apply_selective_checkpointing(
                    self.module.model.decoder, ratio,
                )
                if n_ckpt > 0:
                    self.module.force_input_grads_for_checkpointing = True
                    yield TrainingUpdate(
                        0, 0.0,
                        f"[INFO] Selective gradient checkpointing: "
                        f"{n_ckpt} layers (ratio={ratio:.2f})",
                        kind="info",
                    )
                else:
                    # Fallback to full checkpointing
                    ckpt_ok, cache_off, grads_ok = configure_memory_features(
                        self.module.model.decoder
                    )
                    self.module.force_input_grads_for_checkpointing = ckpt_ok
                    yield TrainingUpdate(
                        0, 0.0,
                        "[WARN] Selective checkpointing failed — "
                        "fell back to full checkpointing",
                        kind="warn",
                    )
            else:
                # Full checkpointing (ratio=1.0) or effective full
                ckpt_ok, cache_off, grads_ok = configure_memory_features(
                    self.module.model.decoder
                )
                self.module.force_input_grads_for_checkpointing = ckpt_ok
                if ckpt_ok:
                    yield TrainingUpdate(
                        0, 0.0,
                        f"[INFO] Gradient checkpointing enabled "
                        f"(use_cache={not cache_off}, input_grads={grads_ok})",
                        kind="info",
                    )
                else:
                    yield TrainingUpdate(
                        0, 0.0, "[WARN] Gradient checkpointing not supported by this model",
                        kind="warn",
                    )
        else:
            yield TrainingUpdate(
                0, 0.0,
                "[INFO] Gradient checkpointing OFF (faster but uses more VRAM)",
                kind="info",
            )

        # -- Encoder/VAE offloading ------------------------------------------
        if getattr(cfg, "offload_encoder", False):
            offloaded = offload_non_decoder(self.module.model)
            if offloaded:
                yield TrainingUpdate(0, 0.0, f"[INFO] Offloaded {offloaded} model components to CPU (saves VRAM)", kind="info")
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        # -- dtype setup ----------------------------------------------------
        self.module.model = self.module.model.to(self.module.dtype)

        # -- Resume ---------------------------------------------------------
        start_epoch = 0
        global_step = 0
        _resumed_runtime: Optional[Dict[str, Any]] = None

        if cfg.resume_from and Path(cfg.resume_from).exists():
            try:
                yield TrainingUpdate(0, 0.0, f"[INFO] Loading checkpoint from {cfg.resume_from}", kind="info")
                resumed = yield from self._resume_checkpoint(
                    cfg.resume_from, optimizer, scheduler,
                )
                if resumed is not None:
                    start_epoch, global_step = resumed[0], resumed[1]
                    _resumed_runtime = resumed[2] if len(resumed) > 2 else None
            except Exception as exc:
                logger.exception("Failed to load checkpoint")
                yield TrainingUpdate(0, 0.0, f"[WARN] Checkpoint load failed: {exc} -- starting fresh", kind="warn")
                start_epoch = 0
                global_step = 0

        # Restore RNG state from checkpoint (overrides initial seed)
        if _resumed_runtime and _resumed_runtime.get("rng_state"):
            restored_components = restore_rng_state(
                _resumed_runtime["rng_state"],
                current_device=self.module.device,
            )
            if restored_components:
                self._rng_seeded_fresh = False
                yield TrainingUpdate(
                    0, 0.0,
                    f"[OK] RNG state restored: {', '.join(restored_components)}",
                    kind="info",
                )

        # Stash total_steps on cfg for checkpoint metadata
        cfg._checkpoint_total_steps = total_steps

        # -- Training loop --------------------------------------------------
        accumulation_step = 0
        accumulated_loss = 0.0
        optimizer.zero_grad(set_to_none=True)
        self.module.model.decoder.train()

        # NaN/Inf guard -- consecutive bad losses trigger a halt
        consecutive_nan = 0

        # Best-model tracking (MA5 smoothed loss)
        best_loss = float('inf')
        best_epoch = 0
        patience_counter = 0
        best_tracking_active = False
        min_delta = 0.001
        loss_window_size = 5
        recent_losses: list = []

        # Step-level best tracking (smoothed over recent steps)
        _step_best_every = getattr(cfg, "save_best_every_n_steps", 0)
        _step_best_loss = float('inf')
        _step_loss_window: list = []
        _step_loss_window_size = 20

        # Target loss cruise control
        _target_loss = getattr(cfg, "target_loss", 0.0)
        _target_loss_floor = getattr(cfg, "target_loss_floor", 0.01)
        _cruise_active = False  # one-time message flag
        _cruise_ema: Optional[float] = None
        _cruise_beta = getattr(cfg, "target_loss_smoothing", 0.98)
        _scheduler_lr: Optional[float] = None  # pure scheduler LR (before damping)
        _CRUISE_MIN_STEPS = getattr(cfg, "target_loss_warmup", 50)

        # -- Cruise control conflict guards --
        if _target_loss > 0:
            # Prodigy manages LR internally — cruise control would fight it
            if optimizer_type == "prodigy":
                _target_loss = 0.0
                yield TrainingUpdate(
                    0, 0.0,
                    "[WARN] Cruise control disabled — Prodigy optimizer manages LR internally. "
                    "Use a different optimizer (e.g. adamw8bit) to enable target_loss.",
                    kind="warn",
                )

            # Early stopping triggers on plateau — cruise intentionally plateaus loss
            if _target_loss > 0 and cfg.early_stop_patience > 0:
                yield TrainingUpdate(
                    0, 0.0,
                    f"[WARN] Early stopping (patience={cfg.early_stop_patience}) auto-disabled — "
                    f"cruise control intentionally holds loss steady, which would false-trigger early stop.",
                    kind="warn",
                )
                cfg.early_stop_patience = 0

            # Never engage cruise before scheduler warmup finishes
            if _target_loss > 0 and _CRUISE_MIN_STEPS < cfg.warmup_steps:
                _CRUISE_MIN_STEPS = cfg.warmup_steps
                yield TrainingUpdate(
                    0, 0.0,
                    f"[INFO] Cruise warmup clamped to {cfg.warmup_steps} (scheduler warmup steps) "
                    f"to avoid fighting the LR ramp.",
                    kind="info",
                )

            # Cosine restarts cause periodic LR resets that fight smooth cruising
            if _target_loss > 0 and scheduler_type == "cosine_restarts":
                yield TrainingUpdate(
                    0, 0.0,
                    "[WARN] Cosine restarts + cruise control: LR resets will cause oscillating "
                    "damping. Consider using 'cosine' scheduler for smoother cruising.",
                    kind="warn",
                )

            # Combined minimum effective LR warning
            if _target_loss > 0 and scheduler_type in ("cosine", "cosine_restarts"):
                _eta_min_ratio = getattr(cfg, "cosine_eta_min_ratio", 0.01)
                _combined_min = cfg.learning_rate * _eta_min_ratio * _target_loss_floor
                if _combined_min < cfg.learning_rate * 1e-4:
                    yield TrainingUpdate(
                        0, 0.0,
                        f"[WARN] Combined min LR (eta_min × floor) = {_combined_min:.2e} — "
                        f"this may be too low. Consider raising cosine_eta_min_ratio or target_loss_floor.",
                        kind="warn",
                    )

        # Rehydrate tracker state from checkpoint if available
        if _resumed_runtime and _resumed_runtime.get("tracker_state"):
            ts = _resumed_runtime["tracker_state"]
            best_loss = ts.get("best_loss", best_loss)
            best_epoch = ts.get("best_epoch", best_epoch)
            best_tracking_active = ts.get("best_tracking_active", best_tracking_active)
            recent_losses = list(ts.get("recent_losses", []))
            saved_patience = ts.get("patience_counter", 0)
            patience_counter = min(saved_patience, cfg.early_stop_patience) if cfg.early_stop_patience > 0 else 0
            _saved_cruise_ema = ts.get("cruise_ema")
            if _saved_cruise_ema is not None and _target_loss > 0:
                _cruise_ema = _saved_cruise_ema
                _CRUISE_MIN_STEPS = 0  # already warmed — engage immediately
            yield TrainingUpdate(
                0, 0.0,
                f"[OK] Tracker state restored (best_loss={best_loss:.4f}, "
                f"best_epoch={best_epoch}, patience={patience_counter})",
                kind="info",
            )

        # Restore EMA / adaptive sampler state from checkpoint
        if _resumed_runtime:
            if _ema is not None and _resumed_runtime.get("ema_state"):
                try:
                    _ema.load_state_dict(_resumed_runtime["ema_state"])
                    yield TrainingUpdate(0, 0.0, "[OK] EMA state restored from checkpoint", kind="info")
                except Exception as exc:
                    yield TrainingUpdate(0, 0.0, f"[WARN] EMA state restore failed: {exc}", kind="warn")
            if _adaptive_sampler is not None and _resumed_runtime.get("adaptive_sampler_state"):
                try:
                    _adaptive_sampler.load_state_dict(_resumed_runtime["adaptive_sampler_state"])
                    yield TrainingUpdate(0, 0.0, "[OK] Adaptive sampler state restored from checkpoint", kind="info")
                except Exception as exc:
                    yield TrainingUpdate(0, 0.0, f"[WARN] Adaptive sampler state restore failed: {exc}", kind="warn")

        # Chunk sampler tracking (parity with basic loop)
        _chunk_sampler = getattr(data_module.train_dataset, "_chunk_sampler", None)
        if _chunk_sampler is not None and _resumed_runtime and _resumed_runtime.get("chunk_coverage_state"):
            _chunk_sampler.load_state_dict(_resumed_runtime["chunk_coverage_state"])
            yield TrainingUpdate(0, 0.0, "[OK] Chunk coverage state restored", kind="info")

        for epoch in range(start_epoch, cfg.max_epochs):
            # Notify chunk sampler of epoch boundary for histogram decay
            if _chunk_sampler is not None:
                _chunk_sampler.notify_epoch(epoch)

            epoch_loss = 0.0
            num_updates = 0
            epoch_start = time.time()
            _accum_cond_info: list = []

            for _batch_idx, batch in enumerate(train_loader):
                # Stop signal
                if training_state and training_state.get("should_stop", False):
                    _stop_loss = (
                        accumulated_loss * cfg.gradient_accumulation_steps
                        / max(accumulation_step, 1)
                    )
                    yield from save_on_early_exit(
                        self, str(output_dir), global_step, "user_stop", ema=_ema,
                    )
                    yield TrainingUpdate(global_step, _stop_loss, "[INFO] Training stopped by user", kind="complete")
                    tb.close()
                    return

                # Extract non-tensor keys before training_step
                _batch_cond = batch.pop("conditioning_info", None)
                batch.pop("metadata", None)
                if _batch_cond:
                    _accum_cond_info.extend(_batch_cond)

                loss = self.module.training_step(batch)

                # Guard: skip backward on NaN/Inf to protect optimizer state
                if torch.isnan(loss) or torch.isinf(loss):
                    consecutive_nan += 1
                    del loss
                    if consecutive_nan >= _MAX_CONSECUTIVE_NAN:
                        yield from save_on_early_exit(
                            self, str(output_dir), global_step, "nan_halt", ema=_ema,
                        )
                        yield TrainingUpdate(
                            global_step, 0.0,
                            f"[FAIL] {consecutive_nan} consecutive NaN/Inf losses -- halting training",
                            kind="fail",
                        )
                        tb.close()
                        return
                    # Discard any partially-accumulated gradients so the
                    # next valid batch starts a fresh accumulation cycle.
                    if accumulation_step > 0:
                        optimizer.zero_grad(set_to_none=True)
                        accumulated_loss = 0.0
                        accumulation_step = 0
                    continue
                consecutive_nan = 0

                loss = loss / cfg.gradient_accumulation_steps
                loss.backward()
                accumulated_loss += loss.item()
                del loss
                accumulation_step += 1

                if accumulation_step >= cfg.gradient_accumulation_steps:
                    torch.nn.utils.clip_grad_norm_(
                        trainable_params, max_norm=cfg.max_grad_norm,
                    )
                    optimizer.step()
                    # Restore un-damped LR so scheduler.step() sees the pure value,
                    # NOT the cruise-damped value (prevents compounding spiral).
                    if _scheduler_lr is not None:
                        for pg in optimizer.param_groups:
                            pg["lr"] = _scheduler_lr
                    scheduler.step()
                    if _ema is not None:
                        _ema.update()
                    global_step += 1

                    avg_loss = accumulated_loss * cfg.gradient_accumulation_steps / accumulation_step

                    # Snapshot the scheduler's pure LR before any damping
                    _scheduler_lr = scheduler.get_last_lr()[0]

                    # -- Target loss cruise control (LR damping) v3 -----
                    if _target_loss > 0 and global_step >= _CRUISE_MIN_STEPS:
                        # EMA loss smoothing
                        if _cruise_ema is None:
                            _cruise_ema = avg_loss
                        else:
                            _cruise_ema = _cruise_beta * _cruise_ema + (1.0 - _cruise_beta) * avg_loss

                        if _cruise_ema > _target_loss:
                            # Above target: smoothstep over tight 30% margin
                            _margin = 0.3
                            _headroom = (_cruise_ema - _target_loss) / max(_target_loss * _margin, 1e-8)
                            _t = max(0.0, min(1.0, _headroom))
                            _smooth_t = _t * _t * (3.0 - 2.0 * _t)
                            _scale = _target_loss_floor + _smooth_t * (1.0 - _target_loss_floor)
                        else:
                            # Below target: exponentially decay toward zero LR
                            _overshoot = (_target_loss - _cruise_ema) / max(_target_loss, 1e-8)
                            _scale = _target_loss_floor * math.exp(-10.0 * _overshoot)

                        # Fast-path: raw loss already below target → clamp now,
                        # don't wait for the lagging EMA to catch up
                        if avg_loss < _target_loss:
                            _scale = min(_scale, _target_loss_floor)

                        for pg in optimizer.param_groups:
                            pg["lr"] = _scheduler_lr * _scale
                        tb.log_scalar("target_loss_scale", _scale, global_step)
                        tb.log_scalar("target_loss_ema", _cruise_ema, global_step)
                        if not _cruise_active and _scale < 0.99:
                            _cruise_active = True
                            yield TrainingUpdate(
                                step=global_step, loss=avg_loss,
                                msg=f"[INFO] Target loss cruise control activated "
                                    f"(target={_target_loss}, ema={_cruise_ema:.4f}, scale={_scale:.3f})",
                                kind="info", epoch=epoch + 1, max_epochs=cfg.max_epochs,
                            )

                    _lr = optimizer.param_groups[0]["lr"]
                    _pw_extra = {}
                    if _target_loss > 0 and global_step >= _CRUISE_MIN_STEPS:
                        _pw_extra["target_loss_scale"] = _scale
                        _pw_extra["target_loss_ema"] = _cruise_ema
                    # Pipe fidelity metrics for real-time GUI mini-charts
                    _latest_sm = self.module._step_metrics
                    if _latest_sm:
                        _pw_extra["fidelity"] = dict(_latest_sm)
                    if _ema is not None:
                        _pw_extra["ema_active"] = _ema._active
                    if _accum_cond_info:
                        _pw_extra["conditioning_info"] = _accum_cond_info
                    _pw.maybe_write(step=global_step, epoch=epoch + 1,
                                    max_epochs=cfg.max_epochs, loss=avg_loss, lr=_lr,
                                    best_loss=best_loss, best_epoch=best_epoch,
                                    steps_per_epoch=steps_per_epoch, **_pw_extra)
                    _accum_cond_info = []
                    if global_step % cfg.log_every == 0:
                        tb.log_loss(avg_loss, global_step)
                        tb.log_lr(_lr, global_step)

                        # Fidelity metrics from training_step
                        _sm = self.module.drain_step_metrics()
                        for _tag, _val in _sm.items():
                            tb.log_scalar(_tag, _val, global_step)

                        # EMA status
                        if _ema is not None:
                            tb.log_scalar("ema/active", 1.0 if _ema._active else 0.0, global_step)
                            tb.log_scalar("ema/step_count", float(_ema._step_count), global_step)
                            tb.log_scalar("ema/effective_decay", _ema.effective_decay, global_step)

                        yield TrainingUpdate(
                            step=global_step, loss=avg_loss,
                            msg=f"Epoch {epoch + 1}/{cfg.max_epochs}, Step {global_step}, Loss: {avg_loss:.4f}",
                            kind="step", epoch=epoch + 1, max_epochs=cfg.max_epochs, lr=_lr,
                            steps_per_epoch=steps_per_epoch,
                        )

                    # max_steps stop condition
                    _max_steps = getattr(cfg, "max_steps", 0)
                    if _max_steps > 0 and global_step >= _max_steps:
                        optimizer.zero_grad(set_to_none=True)
                        epoch_loss += avg_loss
                        num_updates += 1
                        accumulated_loss = 0.0
                        accumulation_step = 0
                        yield from save_on_early_exit(
                            self, str(output_dir), global_step, "max_steps", ema=_ema,
                        )
                        yield TrainingUpdate(
                            global_step, avg_loss,
                            f"[INFO] Reached max_steps={_max_steps} -- stopping training",
                            kind="complete",
                        )
                        tb.close()
                        return

                    if cfg.log_heavy_every > 0 and global_step % cfg.log_heavy_every == 0:
                        tb.log_per_layer_grad_norms(self.module.model, global_step)
                        tb.flush()

                    timestep_every = max(0, int(getattr(cfg, "log_timestep_every", cfg.log_every)))
                    _has_weighting = getattr(cfg, "loss_weighting", "none") in ("min_snr", "flow_snr")
                    if (
                        _has_weighting
                        and timestep_every > 0
                        and global_step % timestep_every == 0
                    ):
                        ts_buf = self.module.drain_timestep_buffer()
                        if ts_buf is not None:
                            tb.log_timestep_histogram(ts_buf, global_step)
                        tb.flush()

                    optimizer.zero_grad(set_to_none=True)
                    epoch_loss += avg_loss
                    num_updates += 1
                    accumulated_loss = 0.0
                    accumulation_step = 0

                    # Step-level best-model tracking
                    if _step_best_every > 0 and cfg.save_best and best_tracking_active:
                        _step_loss_window.append(avg_loss)
                        if len(_step_loss_window) > _step_loss_window_size:
                            _step_loss_window.pop(0)
                        if (global_step % _step_best_every == 0
                                and len(_step_loss_window) >= _step_loss_window_size // 2):
                            _smoothed = sum(_step_loss_window) / len(_step_loss_window)
                            if _smoothed < _step_best_loss - min_delta:
                                _step_best_loss = _smoothed
                                best_path = str(output_dir / "best")
                                self.module.model.decoder.eval()
                                if _ema is not None:
                                    _ema.apply()
                                try:
                                    self._save_adapter_flat(best_path)
                                finally:
                                    if _ema is not None:
                                        _ema.restore()
                                self.module.model.decoder.train()
                                yield TrainingUpdate(
                                    step=global_step, loss=avg_loss,
                                    msg=f"[OK] Step-level best saved (step {global_step}, smoothed: {_smoothed:.4f})",
                                    kind="checkpoint", epoch=epoch + 1, max_epochs=cfg.max_epochs,
                                    checkpoint_path=best_path,
                                )

                    # Periodic CUDA cache cleanup to prevent intra-epoch
                    # memory fragmentation on consumer GPUs.
                    if torch.cuda.is_available() and global_step % cfg.log_every == 0:
                        torch.cuda.empty_cache()

            # Flush remainder
            if accumulation_step > 0:
                torch.nn.utils.clip_grad_norm_(
                    trainable_params, max_norm=cfg.max_grad_norm,
                )
                optimizer.step()
                # Restore un-damped LR so scheduler.step() sees the pure value
                # (same guard as the main accumulation path).
                if _scheduler_lr is not None:
                    for pg in optimizer.param_groups:
                        pg["lr"] = _scheduler_lr
                scheduler.step()
                if _ema is not None:
                    _ema.update()
                global_step += 1

                avg_loss = accumulated_loss * cfg.gradient_accumulation_steps / accumulation_step
                # Keep _scheduler_lr in sync so the next epoch's restore is correct.
                _scheduler_lr = scheduler.get_last_lr()[0]
                _lr = _scheduler_lr
                if global_step % cfg.log_every == 0:
                    tb.log_loss(avg_loss, global_step)
                    tb.log_lr(_lr, global_step)
                    yield TrainingUpdate(
                        step=global_step, loss=avg_loss,
                        msg=f"Epoch {epoch + 1}/{cfg.max_epochs}, Step {global_step}, Loss: {avg_loss:.4f}",
                        kind="step", epoch=epoch + 1, max_epochs=cfg.max_epochs, lr=_lr,
                        steps_per_epoch=steps_per_epoch,
                    )

                optimizer.zero_grad(set_to_none=True)
                epoch_loss += avg_loss
                num_updates += 1
                accumulated_loss = 0.0
                accumulation_step = 0

            # End of epoch
            epoch_time = time.time() - epoch_start
            avg_epoch_loss = epoch_loss / max(num_updates, 1)
            tb.log_epoch_loss(avg_epoch_loss, epoch + 1)
            tb.flush()

            # -- Validation loss (if enabled) --------------------------------
            _val_loss = None
            if _val_loader is not None:
                from sidestep_engine.core.validation import run_validation_epoch
                if _ema is not None:
                    _ema.apply()
                try:
                    _val_loss = run_validation_epoch(self.module, _val_loader, self.module.device)
                finally:
                    if _ema is not None:
                        _ema.restore()
                self.module.model.decoder.train()
                tb.log_scalar("val_loss", _val_loss, epoch + 1)
                yield TrainingUpdate(
                    step=global_step, loss=avg_epoch_loss,
                    msg=f"  Val loss: {_val_loss:.4f}",
                    kind="info", epoch=epoch + 1, max_epochs=cfg.max_epochs,
                )

            # Loss used for best-model tracking: val loss when available,
            # otherwise train loss.  This keeps behavior identical when
            # val_split=0 (default).
            _tracking_loss = _val_loss if _val_loss is not None else avg_epoch_loss

            # -- Best-model tracking (MA5) ----------------------------------
            # Activate tracking once we pass the warmup threshold
            if (cfg.save_best and cfg.save_best_after > 0
                    and (epoch + 1) >= cfg.save_best_after
                    and not best_tracking_active):
                best_tracking_active = True
                best_loss = float('inf')
                patience_counter = 0
                recent_losses.clear()
                yield TrainingUpdate(
                    step=global_step, loss=avg_epoch_loss,
                    msg=f"[INFO] Best-model tracking activated from epoch {epoch + 1}",
                    kind="info", epoch=epoch + 1, max_epochs=cfg.max_epochs,
                )

            # Update rolling window
            recent_losses.append(_tracking_loss)
            if len(recent_losses) > loss_window_size:
                recent_losses.pop(0)
            smoothed_loss = sum(recent_losses) / len(recent_losses)

            # Check for new best
            is_new_best = best_tracking_active and smoothed_loss < best_loss - min_delta
            if is_new_best:
                best_loss = smoothed_loss
                patience_counter = 0
                best_epoch = epoch + 1
            elif best_tracking_active:
                patience_counter += 1

            # Build epoch message with MA5 info
            ma5_str = f", MA5: {smoothed_loss:.4f}" if len(recent_losses) >= 2 else ""
            best_str = f" (best: {best_loss:.4f} @ ep{best_epoch})" if best_tracking_active else ""

            yield TrainingUpdate(
                step=global_step, loss=avg_epoch_loss,
                msg=f"[OK] Epoch {epoch + 1}/{cfg.max_epochs} in {epoch_time:.1f}s, Loss: {avg_epoch_loss:.4f}{ma5_str}{best_str}",
                kind="epoch", epoch=epoch + 1, max_epochs=cfg.max_epochs, epoch_time=epoch_time,
            )
            _pw.write_event(kind="epoch", step=global_step, epoch=epoch + 1,
                            max_epochs=cfg.max_epochs, loss=avg_epoch_loss,
                            epoch_time=epoch_time, best_loss=best_loss,
                            best_epoch=best_epoch)

            # Auto-save best model (eval mode for consistent saved weights)
            if is_new_best:
                best_path = str(output_dir / "best")
                self.module.model.decoder.eval()
                if _ema is not None:
                    _ema.apply()
                try:
                    self._save_adapter_flat(best_path)
                finally:
                    if _ema is not None:
                        _ema.restore()
                self.module.model.decoder.train()
                yield TrainingUpdate(
                    step=global_step, loss=avg_epoch_loss,
                    msg=f"[OK] Best model saved (epoch {epoch + 1}, MA5: {best_loss:.4f})",
                    kind="checkpoint", epoch=epoch + 1, max_epochs=cfg.max_epochs,
                    checkpoint_path=best_path,
                )

            # Early stopping
            if (cfg.early_stop_patience > 0 and best_tracking_active
                    and patience_counter >= cfg.early_stop_patience):
                yield TrainingUpdate(
                    step=global_step, loss=avg_epoch_loss,
                    msg=(
                        f"[INFO] Early stopping at epoch {epoch + 1} "
                        f"(no improvement for {cfg.early_stop_patience} epochs, "
                        f"best MA5={best_loss:.4f} at epoch {best_epoch})"
                    ),
                    kind="info", epoch=epoch + 1, max_epochs=cfg.max_epochs,
                )
                break

            # Periodic checkpoint (eval mode for consistent saved weights)
            if (epoch + 1) % cfg.save_every_n_epochs == 0:
                ckpt_dir = str(output_dir / "checkpoints" / f"epoch_{epoch + 1}")
                self.module.model.decoder.eval()
                _rt_state = {
                    "rng_state": capture_rng_state(self.module.device),
                    "tracker_state": {
                        "best_loss": best_loss,
                        "best_epoch": best_epoch,
                        "patience_counter": patience_counter,
                        "best_tracking_active": best_tracking_active,
                        "recent_losses": list(recent_losses),
                        "cruise_ema": _cruise_ema,
                    },
                }
                if _chunk_sampler is not None:
                    _rt_state["chunk_coverage_state"] = _chunk_sampler.state_dict()
                if _ema is not None:
                    _rt_state["ema_state"] = _ema.state_dict()
                if _adaptive_sampler is not None:
                    _rt_state["adaptive_sampler_state"] = _adaptive_sampler.state_dict()
                if _ema is not None:
                    _ema.apply()
                try:
                    self._save_checkpoint(optimizer, scheduler, epoch + 1, global_step, ckpt_dir, _rt_state)
                finally:
                    if _ema is not None:
                        _ema.restore()
                self.module.model.decoder.train()
                yield TrainingUpdate(
                    step=global_step, loss=avg_epoch_loss,
                    msg=f"[OK] Checkpoint saved at epoch {epoch + 1}",
                    kind="checkpoint", epoch=epoch + 1, max_epochs=cfg.max_epochs,
                    checkpoint_path=ckpt_dir,
                )

            # Clear CUDA cache AFTER checkpoint save so serialization
            # temporaries are also freed.
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        # -- Sanity check: did we actually train? ----------------------------
        if global_step == 0:
            tb.close()
            _pw.write_event(kind="fail", step=0, msg="0 steps completed")
            _pw.close()
            yield TrainingUpdate(
                step=0, loss=0.0,
                msg=(
                    "[FAIL] Training completed 0 steps -- no batches were processed.\n"
                    "       Possible causes:\n"
                    "         - Dataset directory is empty or contains no valid .pt files\n"
                    "         - DataLoader failed to yield batches (device/platform issue)\n"
                    "       Check the dataset path and try again."
                ),
                kind="fail",
            )
            return

        # -- Final save -----------------------------------------------------
        final_path = str(output_dir / "final")
        best_path = str(output_dir / "best")
        adapter_label = "LoKR" if self.adapter_type == "lokr" else "LoRA"
        final_loss = self.module.training_losses[-1] if self.module.training_losses else 0.0

        if best_tracking_active and best_epoch > 0 and Path(best_path).exists():
            import shutil
            if Path(final_path).exists():
                shutil.rmtree(final_path)
            shutil.copytree(best_path, final_path)
            tb.flush()
            tb.close()
            _pw.write_event(kind="complete", step=global_step, loss=final_loss,
                            best_loss=best_loss, best_epoch=best_epoch)
            _pw.close()
            yield TrainingUpdate(
                step=global_step, loss=final_loss,
                msg=(
                    f"[OK] Training complete! {adapter_label} final = best MA5 "
                    f"(epoch {best_epoch}, MA5: {best_loss:.4f}) saved to {final_path}\n"
                    f"     For inference, set your LoRA path to: {final_path}"
                ),
                kind="complete",
            )
        else:
            self.module.model.decoder.eval()
            if _ema is not None:
                _ema.apply()
            try:
                self._save_final(final_path)
            finally:
                if _ema is not None:
                    _ema.restore()
            tb.flush()
            tb.close()
            _pw.write_event(kind="complete", step=global_step, loss=final_loss)
            _pw.close()
            yield TrainingUpdate(
                step=global_step, loss=final_loss,
                msg=(
                    f"[OK] Training complete! {adapter_label} saved to {final_path}\n"
                    f"     For inference, set your LoRA path to: {final_path}"
                ),
                kind="complete",
            )
