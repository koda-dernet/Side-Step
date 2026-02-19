"""
Basic (non-Fabric) training loop for FixedLoRATrainer.

Extracted from ``FixedLoRATrainer._train_basic`` to keep
``trainer_fixed.py`` under the LOC limit.  This module provides a single
generator function that yields ``TrainingUpdate`` objects exactly like
the Fabric loop, but uses manual ``loss.backward()`` and
``torch.nn.utils.clip_grad_norm_`` instead of Fabric wrappers.
"""

from __future__ import annotations

import logging
import math
import shutil
import time
from pathlib import Path
from typing import Any, Dict, Generator, List, Optional, Tuple

import torch

from acestep.training_v2.optim import build_optimizer, build_scheduler

_MAX_CONSECUTIVE_NAN = 10  # halt training after this many NaN/Inf losses in a row
from acestep.training_v2.tensorboard_utils import TrainingLogger
from acestep.training_v2.trainer_helpers import (
    capture_rng_state, configure_memory_features, force_disable_decoder_cache,
    restore_rng_state, save_adapter_flat, save_checkpoint, save_final,
)
from acestep.training_v2.ui import TrainingUpdate

logger = logging.getLogger(__name__)


def _flush_accumulated(
    trainable_params: list,
    optimizer,
    scheduler,
    accumulated_loss: float,
    accumulation_step: int,
    cfg,
    tb,
    module,
    epoch: int,
    global_step: int,
    steps_per_epoch: int,
) -> Tuple[int, float, List[TrainingUpdate]]:
    """Clip gradients, step optimizer/scheduler, log, then zero grads.

    Gradient zeroing is deferred until *after* heavy logging so that
    ``log_per_layer_grad_norms`` can read the gradients (``set_to_none``
    sets ``.grad`` to ``None``, making them invisible to the logger).

    Returns:
        ``(global_step, avg_loss, updates)`` where *updates* is a list
        of ``TrainingUpdate`` objects the caller should yield.
    """
    torch.nn.utils.clip_grad_norm_(trainable_params, cfg.max_grad_norm)
    optimizer.step()
    scheduler.step()
    global_step += 1

    avg_loss = accumulated_loss * cfg.gradient_accumulation_steps / accumulation_step
    _lr = scheduler.get_last_lr()[0]
    updates: List[TrainingUpdate] = []

    if global_step % cfg.log_every == 0:
        tb.log_loss(avg_loss, global_step)
        tb.log_lr(_lr, global_step)
        updates.append(TrainingUpdate(
            step=global_step, loss=avg_loss,
            msg=f"Epoch {epoch + 1}, Step {global_step}, Loss: {avg_loss:.4f}",
            kind="step", epoch=epoch + 1, max_epochs=cfg.max_epochs, lr=_lr,
            steps_per_epoch=steps_per_epoch,
        ))

    if cfg.log_heavy_every > 0 and global_step % cfg.log_heavy_every == 0:
        tb.log_per_layer_grad_norms(module.model, global_step)
        tb.flush()

    timestep_every = max(0, int(getattr(cfg, "log_timestep_every", cfg.log_every)))
    if (
        getattr(cfg, "loss_weighting", "none") == "min_snr"
        and timestep_every > 0
        and global_step % timestep_every == 0
    ):
        ts_buf = module.drain_timestep_buffer()
        if ts_buf is not None:
            tb.log_timestep_histogram(ts_buf, global_step)
        tb.flush()

    # Zero gradients only after logging has had a chance to read them.
    optimizer.zero_grad(set_to_none=True)

    return global_step, avg_loss, updates


def run_basic_training_loop(
    trainer: Any,
    data_module: Any,
    training_state: Optional[Dict[str, Any]],
) -> Generator[TrainingUpdate, None, None]:
    """Execute the basic (non-Fabric) training loop.

    This is a standalone generator extracted from
    ``FixedLoRATrainer._train_basic``.  It accesses trainer state via
    the *trainer* parameter (e.g. ``trainer.module``,
    ``trainer.training_config``, ``trainer._save_checkpoint``).

    Args:
        trainer: The ``FixedLoRATrainer`` instance.
        data_module: ``PreprocessedDataModule`` with training data.
        training_state: Optional dict with ``should_stop`` flag.

    Yields:
        ``TrainingUpdate`` tuples for each step/epoch/event.
    """
    cfg = trainer.training_config
    module = trainer.module
    assert module is not None

    output_dir = Path(cfg.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    yield TrainingUpdate(0, 0.0, "[INFO] Starting basic training loop (no Fabric)", kind="info")

    tb = TrainingLogger(cfg.effective_log_dir)
    train_loader = data_module.train_dataloader()

    trainable_params = [p for p in module.model.parameters() if p.requires_grad]
    if not trainable_params:
        yield TrainingUpdate(0, 0.0, "[FAIL] No trainable parameters found", kind="fail")
        tb.close()
        return

    device_type = module.device_type if hasattr(module, "device_type") else str(module.device).split(":")[0]
    optimizer_type = getattr(cfg, "optimizer_type", "adamw")
    optimizer = build_optimizer(
        trainable_params,
        optimizer_type=optimizer_type,
        lr=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
        device_type=device_type,
    )

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

    # -- Training memory features (same as Fabric path) ----------------
    force_disable_decoder_cache(module.model.decoder)
    if getattr(cfg, "gradient_checkpointing", True):
        ckpt_ok, cache_off, grads_ok = configure_memory_features(
            module.model.decoder
        )
        module.force_input_grads_for_checkpointing = ckpt_ok
        if ckpt_ok:
            yield TrainingUpdate(
                0, 0.0,
                f"[INFO] Gradient checkpointing enabled "
                f"(use_cache={not cache_off}, input_grads={grads_ok})",
                kind="info",
            )

    # -- Resume ---------------------------------------------------------
    start_epoch = 0
    global_step = 0
    _resumed_runtime: Optional[Dict[str, Any]] = None

    if cfg.resume_from and Path(cfg.resume_from).exists():
        try:
            yield TrainingUpdate(0, 0.0, f"[INFO] Loading checkpoint from {cfg.resume_from}", kind="info")
            from acestep.training_v2.trainer_helpers import resume_checkpoint
            resumed = yield from resume_checkpoint(trainer,
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

    # Restore RNG state from checkpoint
    if _resumed_runtime and _resumed_runtime.get("rng_state"):
        restored_components = restore_rng_state(
            _resumed_runtime["rng_state"],
            current_device=module.device,
        )
        if restored_components:
            yield TrainingUpdate(
                0, 0.0,
                f"[OK] RNG state restored: {', '.join(restored_components)}",
                kind="info",
            )

    # Stash total_steps on cfg for checkpoint metadata
    cfg._checkpoint_total_steps = total_steps

    accumulation_step = 0
    accumulated_loss = 0.0
    optimizer.zero_grad(set_to_none=True)

    module.model.decoder.train()

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

        for batch in train_loader:
            if training_state and training_state.get("should_stop", False):
                _stop_loss = (
                    accumulated_loss * cfg.gradient_accumulation_steps
                    / max(accumulation_step, 1)
                )
                yield TrainingUpdate(global_step, _stop_loss, "[INFO] Training stopped", kind="complete")
                tb.close()
                return

            loss = module.training_step(batch)

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
                continue
            consecutive_nan = 0

            loss = loss / cfg.gradient_accumulation_steps
            loss.backward()
            accumulated_loss += loss.item()
            del loss
            accumulation_step += 1

            if accumulation_step >= cfg.gradient_accumulation_steps:
                global_step, avg_loss, updates = _flush_accumulated(
                    trainable_params, optimizer, scheduler,
                    accumulated_loss, accumulation_step, cfg, tb, module,
                    epoch, global_step, steps_per_epoch,
                )
                yield from updates
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
            global_step, avg_loss, updates = _flush_accumulated(
                trainable_params, optimizer, scheduler,
                accumulated_loss, accumulation_step, cfg, tb, module,
                epoch, global_step, steps_per_epoch,
            )
            yield from updates
            epoch_loss += avg_loss
            num_updates += 1
            accumulated_loss = 0.0
            accumulation_step = 0

        epoch_time = time.time() - epoch_start
        avg_epoch_loss = epoch_loss / max(num_updates, 1)
        tb.log_epoch_loss(avg_epoch_loss, epoch + 1)

        # -- Best-model tracking (MA5) ----------------------------------
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

        recent_losses.append(avg_epoch_loss)
        if len(recent_losses) > loss_window_size:
            recent_losses.pop(0)
        smoothed_loss = sum(recent_losses) / len(recent_losses)

        is_new_best = best_tracking_active and smoothed_loss < best_loss - min_delta
        if is_new_best:
            best_loss = smoothed_loss
            patience_counter = 0
            best_epoch = epoch + 1
        elif best_tracking_active:
            patience_counter += 1

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
            module.model.decoder.eval()
            save_adapter_flat(trainer, best_path)
            module.model.decoder.train()
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
            module.model.decoder.eval()
            _rt_state = {
                "rng_state": capture_rng_state(module.device),
                "tracker_state": {
                    "best_loss": best_loss,
                    "best_epoch": best_epoch,
                    "patience_counter": patience_counter,
                    "best_tracking_active": best_tracking_active,
                    "recent_losses": list(recent_losses),
                },
            }
            save_checkpoint(trainer, optimizer, scheduler, epoch + 1, global_step, ckpt_dir, _rt_state)
            module.model.decoder.train()
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

    final_path = str(output_dir / "final")
    best_path = str(output_dir / "best")
    adapter_label = "LoKR" if trainer.adapter_type == "lokr" else "LoRA"
    final_loss = module.training_losses[-1] if module.training_losses else 0.0

    if best_tracking_active and best_epoch > 0 and Path(best_path).exists():
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
        module.model.decoder.eval()
        save_final(trainer, final_path)
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
