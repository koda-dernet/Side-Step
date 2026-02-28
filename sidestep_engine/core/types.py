"""Shared types used across the Side-Step engine.

Types defined here live in the ``core`` layer so that lower layers
(core, data, models) can import them without depending on the UI layer.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, Tuple


@dataclass
class TrainingUpdate:
    """Structured object yielded by the trainer, backward-compatible with
    ``(step, loss, msg)`` tuple unpacking.

    Extra fields give the UI enough context to render a live dashboard
    without parsing message strings.
    """

    step: int
    loss: float
    msg: str
    kind: str = "info"
    """One of: info, step, epoch, checkpoint, complete, warn, fail."""
    epoch: int = 0
    max_epochs: int = 0
    lr: float = 0.0
    epoch_time: float = 0.0
    samples_per_sec: float = 0.0
    steps_per_epoch: int = 0
    """Total optimizer steps per epoch (for step-level progress bar)."""
    resume_start_epoch: int = -1
    """Checkpoint resume epoch used as ETA baseline (when available)."""
    checkpoint_path: str = ""
    """Filesystem path emitted with kind='checkpoint'."""

    # -- backward compat: ``for step, loss, msg in trainer.train():`` --------
    def __iter__(self) -> Iterator[Tuple[int, float, str]]:  # type: ignore[override]
        return iter((self.step, self.loss, self.msg))
