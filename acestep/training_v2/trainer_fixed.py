"""
FixedLoRATrainer -- Orchestration for ACE-Step V2 adapter fine-tuning.

The actual per-step training logic lives in ``fixed_lora_module.py``
(``FixedLoRAModule``).  The non-Fabric fallback loop lives in
``trainer_basic_loop.py``.  Checkpoint, memory, and verification helpers
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
from acestep.training_v2.optim import build_optimizer, build_scheduler
from acestep.training_v2.path_utils import normalize_path
from acestep.training_v2._vendor.data_module import PreprocessedDataModule

_MAX_CONSECUTIVE_NAN = 10  # halt training after this many NaN/Inf losses in a row

# V2 modules
from acestep.training_v2.configs import TrainingConfigV2
from acestep.training_v2.tensorboard_utils import TrainingLogger
from acestep.training_v2.ui import TrainingUpdate

# Split-out modules
from acestep.training_v2.fixed_lora_module import (
    AdapterConfig,
    FixedLoRAModule,
    _normalize_device_type,
    _select_compute_dtype,
    _select_fabric_precision,
)
from acestep.training_v2.trainer_helpers import (
    capture_rng_state,
    configure_memory_features,
    force_disable_decoder_cache,
    offload_non_decoder,
    restore_rng_state,
    resume_checkpoint,
    save_adapter_flat,
    save_checkpoint,
    save_final,
    verify_saved_adapter,
)
from acestep.training_v2.trainer_basic_loop import run_basic_training_loop

logger = logging.getLogger(__name__)

# Try to import Lightning Fabric
try:
    from lightning.fabric import Fabric

    _FABRIC_AVAILABLE = True
except ImportError:
    _FABRIC_AVAILABLE = False
    logger.warning("[WARN] Lightning Fabric not installed. Training will use basic loop.")


# ===========================================================================
# FixedLoRATrainer -- orchestration
# ===========================================================================

class FixedLoRATrainer:
    """High-level trainer for corrected ACE-Step adapter fine-tuning.

    Supports both LoRA (PEFT) and LoKR (LyCORIS) adapters.
    Uses Lightning Fabric for mixed precision and gradient scaling.
    Falls back to a basic PyTorch loop when Fabric is not installed.
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
        self.fabric: Optional[Any] = None
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
            dataset_dir_raw = cfg.dataset_dir
            if not dataset_dir_raw or not str(dataset_dir_raw).strip():
                yield TrainingUpdate(0, 0.0, "[FAIL] Dataset directory is empty or not set", kind="fail")
                return
            ds_dir = Path(normalize_path(dataset_dir_raw) or str(dataset_dir_raw).strip())
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
            dtype = _select_compute_dtype(_normalize_device_type(device))

            self.module = FixedLoRAModule(
                model=self.model,
                adapter_config=self.adapter_config,
                training_config=cfg,
                device=device,
                dtype=dtype,
            )

            # -- Data -------------------------------------------------------
            # Windows uses spawn for multiprocessing; default to 0 workers there
            num_workers = cfg.num_workers
            if sys.platform == "win32" and num_workers > 0:
                logger.info("[Side-Step] Windows detected -- setting num_workers=0 (spawn incompatible)")
                num_workers = 0

            data_module = PreprocessedDataModule(
                tensor_dir=str(ds_dir),
                batch_size=cfg.batch_size,
                num_workers=num_workers,
                pin_memory=cfg.pin_memory,
                prefetch_factor=cfg.prefetch_factor if num_workers > 0 else None,
                persistent_workers=cfg.persistent_workers if num_workers > 0 else False,
                pin_memory_device=cfg.pin_memory_device,
                chunk_duration=getattr(cfg, "chunk_duration", None),
            )
            data_module.setup("fit")

            if len(data_module.train_dataset) == 0:
                fail_msg = "[FAIL] No valid samples found in dataset directory"
                # Unwrap Subset from random_split to access PreprocessedTensorDataset attrs
                ds = data_module.train_dataset
                while hasattr(ds, "dataset"):
                    ds = ds.dataset
                manifest_path = getattr(ds, "manifest_path", None)
                manifest_error = getattr(ds, "manifest_error", None)
                fallback_used = bool(getattr(ds, "manifest_fallback_used", False))
                manifest_loaded = bool(getattr(ds, "manifest_loaded", False))
                sample_paths = getattr(ds, "sample_paths", [])

                # Always include resolved dataset path for verification
                resolved = str(ds_dir.resolve())
                fail_msg += f"\n       Dataset dir: {resolved}"

                if manifest_error:
                    fail_msg += f"\n       {manifest_error}"
                elif manifest_loaded and sample_paths and not fallback_used:
                    # Manifest loaded but all paths invalid (e.g. folder renamed)
                    missing_preview = [p for p in sample_paths if not Path(p).exists()][:3]
                    fail_msg += (
                        "\n       manifest.json paths don't exist (e.g. folder renamed?)."
                        "\n       Try deleting manifest.json to fall back to directory scan, or re-run preprocessing."
                    )
                    if missing_preview:
                        fail_msg += "\n       Missing path preview: " + ", ".join(
                            Path(p).name for p in missing_preview
                        )
                elif manifest_path and Path(manifest_path).is_file() and fallback_used:
                    fail_msg += (
                        "\n       manifest.json could not be used; fallback scan found no valid .pt files."
                        "\n       Point to the preprocessed tensor folder (--tensor-output from preprocessing), not the raw audio folder."
                    )
                elif manifest_path and not Path(manifest_path).is_file():
                    fail_msg += (
                        "\n       manifest.json not found and directory contains no valid .pt files."
                        "\n       Point to the preprocessed tensor folder (--tensor-output from preprocessing), not the raw audio folder."
                    )

                yield TrainingUpdate(0, 0.0, fail_msg, kind="fail")
                return

            yield TrainingUpdate(0, 0.0, f"[OK] Loaded {len(data_module.train_dataset)} preprocessed samples", kind="info")

            # -- Dispatch to Fabric or basic loop ---------------------------
            if _FABRIC_AVAILABLE:
                yield from self._train_fabric(data_module, training_state)
            else:
                yield from run_basic_training_loop(self, data_module, training_state)

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
        from acestep.training_v2.trainer_helpers import iter_module_wrappers
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
    # Fabric training loop
    # ------------------------------------------------------------------

    def _train_fabric(
        self,
        data_module: PreprocessedDataModule,
        training_state: Optional[Dict[str, Any]],
    ) -> Generator[TrainingUpdate, None, None]:
        cfg = self.training_config
        assert self.module is not None

        output_dir = Path(cfg.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        device_type = self.module.device_type
        precision = _select_fabric_precision(device_type)
        accelerator = device_type if device_type in ("cuda", "xpu", "mps", "cpu") else "auto"

        # -- Fabric init ----------------------------------------------------
        # Always use devices=1 (integer).  Passing devices=[index] (a list)
        # causes Fabric on Windows to create a DistributedSampler wrapper
        # that yields 0 batches, silently breaking the training loop.
        # Instead, we set the default CUDA device so Fabric's single-device
        # mode picks up the correct GPU.
        if device_type == "cuda":
            device_idx = self.module.device.index or 0
            torch.cuda.set_device(device_idx)

        self.fabric = Fabric(
            accelerator=accelerator,
            devices=1,
            precision=precision,
        )
        self.fabric.launch()

        yield TrainingUpdate(0, 0.0, f"[INFO] Starting training (device: {device_type}, precision: {precision})", kind="info")

        # -- TensorBoard logger ---------------------------------------------
        tb = TrainingLogger(cfg.effective_log_dir)

        # -- Dataloader -----------------------------------------------------
        train_loader = data_module.train_dataloader()

        # -- Trainable params / optimizer -----------------------------------
        trainable_params = [p for p in self.module.model.parameters() if p.requires_grad]
        if not trainable_params:
            yield TrainingUpdate(0, 0.0, "[FAIL] No trainable parameters found", kind="fail")
            tb.close()
            return

        yield TrainingUpdate(0, 0.0, f"[INFO] Training {sum(p.numel() for p in trainable_params):,} parameters", kind="info")

        optimizer_type = getattr(cfg, "optimizer_type", "adamw")
        optimizer = build_optimizer(
            trainable_params,
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
        )
        yield TrainingUpdate(0, 0.0, f"[INFO] Scheduler: {scheduler_type}", kind="info")

        # -- Training memory features ----------------------------------------
        cache_forced_off = force_disable_decoder_cache(self.module.model.decoder)
        if cache_forced_off:
            yield TrainingUpdate(
                0, 0.0,
                "[INFO] Disabled decoder KV cache for training VRAM stability",
                kind="info",
            )

        if getattr(cfg, "gradient_checkpointing", True):
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

        # -- dtype / Fabric setup -------------------------------------------
        self.module.model = self.module.model.to(self.module.dtype)
        self.module.model.decoder, optimizer = self.fabric.setup(self.module.model.decoder, optimizer)

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

        # Rehydrate tracker state from checkpoint if available
        if _resumed_runtime and _resumed_runtime.get("tracker_state"):
            ts = _resumed_runtime["tracker_state"]
            best_loss = ts.get("best_loss", best_loss)
            best_epoch = ts.get("best_epoch", best_epoch)
            best_tracking_active = ts.get("best_tracking_active", best_tracking_active)
            recent_losses = list(ts.get("recent_losses", []))
            saved_patience = ts.get("patience_counter", 0)
            patience_counter = min(saved_patience, cfg.early_stop_patience) if cfg.early_stop_patience > 0 else 0
            yield TrainingUpdate(
                0, 0.0,
                f"[OK] Tracker state restored (best_loss={best_loss:.4f}, "
                f"best_epoch={best_epoch}, patience={patience_counter})",
                kind="info",
            )

        for epoch in range(start_epoch, cfg.max_epochs):
            epoch_loss = 0.0
            num_updates = 0
            epoch_start = time.time()

            for _batch_idx, batch in enumerate(train_loader):
                # Stop signal
                if training_state and training_state.get("should_stop", False):
                    _stop_loss = (
                        accumulated_loss * cfg.gradient_accumulation_steps
                        / max(accumulation_step, 1)
                    )
                    yield TrainingUpdate(global_step, _stop_loss, "[INFO] Training stopped by user", kind="complete")
                    tb.close()
                    return

                loss = self.module.training_step(batch)

                # Guard: skip backward on NaN/Inf to protect optimizer state
                if torch.isnan(loss) or torch.isinf(loss):
                    consecutive_nan += 1
                    del loss
                    if consecutive_nan >= _MAX_CONSECUTIVE_NAN:
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
                self.fabric.backward(loss)
                accumulated_loss += loss.item()
                del loss
                accumulation_step += 1

                if accumulation_step >= cfg.gradient_accumulation_steps:
                    self.fabric.clip_gradients(
                        self.module.model.decoder, optimizer, max_norm=cfg.max_grad_norm,
                    )
                    optimizer.step()
                    scheduler.step()
                    global_step += 1

                    avg_loss = accumulated_loss * cfg.gradient_accumulation_steps / accumulation_step
                    _lr = scheduler.get_last_lr()[0]
                    if global_step % cfg.log_every == 0:
                        tb.log_loss(avg_loss, global_step)
                        tb.log_lr(_lr, global_step)
                        yield TrainingUpdate(
                            step=global_step, loss=avg_loss,
                            msg=f"Epoch {epoch + 1}/{cfg.max_epochs}, Step {global_step}, Loss: {avg_loss:.4f}",
                            kind="step", epoch=epoch + 1, max_epochs=cfg.max_epochs, lr=_lr,
                            steps_per_epoch=steps_per_epoch,
                        )

                    if cfg.log_heavy_every > 0 and global_step % cfg.log_heavy_every == 0:
                        tb.log_per_layer_grad_norms(self.module.model, global_step)
                        tb.flush()

                    timestep_every = max(0, int(getattr(cfg, "log_timestep_every", cfg.log_every)))
                    if (
                        getattr(cfg, "loss_weighting", "none") == "min_snr"
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

                    # Periodic CUDA cache cleanup to prevent intra-epoch
                    # memory fragmentation on consumer GPUs.
                    if torch.cuda.is_available() and global_step % cfg.log_every == 0:
                        torch.cuda.empty_cache()

            # Flush remainder
            if accumulation_step > 0:
                self.fabric.clip_gradients(
                    self.module.model.decoder, optimizer, max_norm=cfg.max_grad_norm,
                )
                optimizer.step()
                scheduler.step()
                global_step += 1

                avg_loss = accumulated_loss * cfg.gradient_accumulation_steps / accumulation_step
                _lr = scheduler.get_last_lr()[0]
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
            recent_losses.append(avg_epoch_loss)
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

            # Auto-save best model (eval mode for consistent saved weights)
            if is_new_best:
                best_path = str(output_dir / "best")
                self.module.model.decoder.eval()
                self._save_adapter_flat(best_path)
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
                    },
                }
                self._save_checkpoint(optimizer, scheduler, epoch + 1, global_step, ckpt_dir, _rt_state)
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
            self._save_final(final_path)
            tb.flush()
            tb.close()
            yield TrainingUpdate(
                step=global_step, loss=final_loss,
                msg=(
                    f"[OK] Training complete! {adapter_label} saved to {final_path}\n"
                    f"     For inference, set your LoRA path to: {final_path}"
                ),
                kind="complete",
            )
