"""Loss-milestone adapter snapshots + audio previews.

Saves an inference-ready adapter snapshot (and generates an audio
preview) each time the smoothed epoch loss first crosses a multiple of
``loss_milestone_interval`` (e.g. 0.1 -> snapshots at loss 0.9, 0.8, ...
down toward 0).  The point is to map *where in the loss curve the
adapter starts to sound like the artist*, so future runs can use a
higher target_loss and finish sooner.


Design notes:

- The tracker keeps its **own** MA5 window over the per-epoch tracking
  loss (val loss when enabled, else train loss).  It deliberately does
  NOT reuse the loop's ``recent_losses`` window because that window is
  cleared when best-model tracking activates (``save_best_after``),
  which would briefly un-smooth the signal and could false-trigger a
  milestone on one noisy epoch.
- The milestone ladder is initialized strictly *below* the first
  observed smoothed loss.  A fresh run starting at loss ~1.05 gets the
  full ladder (1.0, 0.9, ...); a resumed run starting at 0.55 only gets
  0.5 and below — milestones already passed are never re-saved.
- When several milestones are crossed in a single epoch (fast early
  descent), the weights are identical for all of them, so only ONE
  snapshot is saved — named for the lowest milestone crossed — and the
  skipped values are recorded in its ``milestone.json`` and the log.
- Snapshot failures are caught and reported as warnings; a broken save
  must never kill a training run (mirrors the EpochSampler contract).
"""

from __future__ import annotations

import json
import logging
import math
from pathlib import Path
from typing import Any, Callable, Generator, List, Optional

from sidestep_engine.core.types import TrainingUpdate

logger = logging.getLogger(__name__)

_WINDOW_SIZE = 5  # MA5, matching the loop's best-tracking smoothing
_EPS = 1e-9


class LossMilestoneTracker:
    """Detects first crossings of evenly spaced loss milestones.

    Feed :meth:`observe` once per epoch with the tracking loss; it
    returns the list of newly crossed milestone values (highest first,
    empty for most epochs).
    """

    def __init__(self, cfg: Any) -> None:
        self.interval: float = float(
            getattr(cfg, "loss_milestone_interval", 0.0) or 0.0
        )
        self.enabled: bool = self.interval > 0
        self._window: List[float] = []
        self._next: Optional[float] = None  # ladder position; None until seeded
        self.smoothed: float = 0.0  # last smoothed value (for logging)

    def observe(self, epoch_loss: float) -> List[float]:
        """Record one epoch's tracking loss; return newly crossed milestones."""
        if not self.enabled:
            return []
        if not math.isfinite(epoch_loss):
            return []

        self._window.append(epoch_loss)
        if len(self._window) > _WINDOW_SIZE:
            self._window.pop(0)
        smoothed = sum(self._window) / len(self._window)
        self.smoothed = smoothed

        if self._next is None:
            # Seed the ladder strictly below the first observation so a
            # resumed (or already-converged) run never re-fires passed
            # milestones.  First epoch never triggers a save.
            k = math.floor((smoothed - _EPS) / self.interval)
            self._next = round(k * self.interval, 6)
            while self._next >= smoothed - _EPS:
                self._next = round(self._next - self.interval, 6)
            return []

        crossed: List[float] = []
        while self._next > _EPS and smoothed <= self._next + _EPS:
            crossed.append(self._next)
            self._next = round(self._next - self.interval, 6)
        return crossed

    @property
    def next_milestone(self) -> Optional[float]:
        return self._next if (self.enabled and self._next and self._next > 0) else None


def milestone_epoch_hook(
    tracker: LossMilestoneTracker,
    sampler: Any,
    module: Any,
    ema: Any,
    save_fn: Callable[[str], None],
    *,
    tracking_loss: float,
    epoch: int,
    max_epochs: int,
    global_step: int,
    output_dir: Path,
    pw: Any,
) -> Generator[TrainingUpdate, None, None]:
    """Run the per-epoch milestone check; save + sample on a crossing.

    Args:
        tracker: The run's :class:`LossMilestoneTracker`.
        sampler: The run's ``EpochSampler`` (shared conditioning cache);
            may have periodic sampling disabled — milestone previews
            force-generate through it regardless.
        module: Live ``FixedLoRAModule``.
        ema: Optional ``AdapterEMA`` (applied around save + sample).
        save_fn: Callable that writes the inference-ready adapter to a
            directory path (``save_adapter_flat`` bound to the trainer).
        tracking_loss: This epoch's tracking loss (val if enabled, else
            train) — the same value the loop feeds best-model tracking.
        pw: The run's ``ProgressWriter``.
    """
    crossed = tracker.observe(tracking_loss)
    if not crossed:
        return

    m = crossed[-1]  # lowest milestone crossed this epoch
    label = f"loss_{m:.2f}"
    mdir = Path(output_dir) / "milestones" / label

    decoder = module.model.decoder
    decoder.eval()
    if ema is not None:
        ema.apply()
    try:
        save_fn(str(mdir))
        save_ok = True
        save_err = ""
    except Exception as exc:  # noqa: BLE001 — snapshots must never kill training
        save_ok = False
        save_err = str(exc)
        logger.exception("[Milestone] Snapshot save failed for %s", label)
    finally:
        if ema is not None:
            ema.restore()
    decoder.train()

    if not save_ok:
        yield TrainingUpdate(
            step=global_step, loss=tracking_loss,
            msg=f"[WARN] Loss milestone {m:.2f} reached but snapshot save failed: {save_err}",
            kind="warn", epoch=epoch, max_epochs=max_epochs,
        )
        return

    also = [f"{x:.2f}" for x in crossed[:-1]]
    try:
        (mdir / "milestone.json").write_text(
            json.dumps(
                {
                    "milestone": m,
                    "also_crossed_this_epoch": [float(a) for a in also],
                    "smoothed_loss": tracker.smoothed,
                    "epoch_loss": tracking_loss,
                    "epoch": epoch,
                    "global_step": global_step,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
    except Exception:
        logger.exception("[Milestone] Failed to write milestone.json for %s", label)

    note = f" (also crossed {', '.join(also)} this epoch — same weights, saved once)" if also else ""
    yield TrainingUpdate(
        step=global_step, loss=tracking_loss,
        msg=(
            f"[OK] Loss milestone {m:.2f} reached (MA5 {tracker.smoothed:.4f}, "
            f"epoch {epoch}) — adapter saved to milestones/{label}{note}"
        ),
        kind="checkpoint", epoch=epoch, max_epochs=max_epochs,
        checkpoint_path=str(mdir),
    )
    pw.write_event(
        kind="loss_milestone", step=global_step, epoch=epoch,
        milestone=m, smoothed_loss=tracker.smoothed, path=str(mdir),
    )

    if sampler is not None:
        _sample_msg, _sample_ok = sampler.generate(
            module, epoch, Path(output_dir), ema=ema, tag=label, force=True,
            adapter_path=str(mdir),
        )
        if _sample_msg:
            yield TrainingUpdate(
                step=global_step, loss=tracking_loss, msg=_sample_msg,
                kind="info" if _sample_ok else "warn",
                epoch=epoch, max_epochs=max_epochs,
            )
            if _sample_ok:
                pw.write_event(
                    kind="sample", step=global_step, epoch=epoch,
                    path=sampler.last_sample_path or "",
                )
