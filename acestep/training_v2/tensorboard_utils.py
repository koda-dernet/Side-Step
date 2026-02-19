"""
Extended TensorBoard Logging for ACE-Step Training V2

Provides helpers for:
    - Per-layer gradient norms
    - Learning rate tracking
    - Loss curves
    - Estimation score logging
    - Versioned log directory management
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import Any, Dict, Optional

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Versioned log directory helpers
# ---------------------------------------------------------------------------

def resolve_versioned_log_dir(log_root: str | Path, run_name: str) -> Path:
    """Find the next available versioned log directory for *run_name*.

    Scans *log_root* for existing ``{run_name}_v{N}`` directories and
    returns the next version.  First run gets ``_v0``.

    Examples::

        logs/jazz_v0/   # first run named "jazz"
        logs/jazz_v1/   # second run (re-run, not resume)
    """
    log_root = Path(log_root)
    log_root.mkdir(parents=True, exist_ok=True)
    pattern = re.compile(re.escape(run_name) + r"_v(\d+)$")
    max_ver = -1
    for child in log_root.iterdir():
        if child.is_dir():
            m = pattern.match(child.name)
            if m:
                max_ver = max(max_ver, int(m.group(1)))
    next_ver = max_ver + 1
    result = log_root / f"{run_name}_v{next_ver}"
    try:
        result.mkdir(exist_ok=False)
    except FileExistsError:
        next_ver += 1
        result = log_root / f"{run_name}_v{next_ver}"
        result.mkdir(parents=True, exist_ok=True)
    return result


def resolve_latest_versioned_log_dir(log_root: str | Path, run_name: str) -> Optional[Path]:
    """Return latest existing ``{run_name}_vN`` directory, or ``None``.

    Used by resume flow so TensorBoard continues in the previous run dir.
    """
    log_root = Path(log_root)
    if not log_root.is_dir():
        return None
    pattern = re.compile(re.escape(run_name) + r"_v(\d+)$")
    best: tuple[int, Path] | None = None
    for child in log_root.iterdir():
        if not child.is_dir():
            continue
        m = pattern.match(child.name)
        if not m:
            continue
        ver = int(m.group(1))
        if best is None or ver > best[0]:
            best = (ver, child)
    return best[1] if best is not None else None

try:
    from torch.utils.tensorboard import SummaryWriter

    _TB_AVAILABLE = True
except ImportError:
    _TB_AVAILABLE = False
    SummaryWriter = None  # type: ignore[misc,assignment]


class TrainingLogger:
    """Wrapper around ``SummaryWriter`` with training-specific helpers.

    If TensorBoard is not installed, all methods are silent no-ops.
    """

    def __init__(self, log_dir: str | Path, enabled: bool = True) -> None:
        self._writer: Optional[Any] = None
        self._enabled = enabled and _TB_AVAILABLE
        if self._enabled:
            log_dir = Path(log_dir)
            log_dir.mkdir(parents=True, exist_ok=True)
            self._writer = SummaryWriter(log_dir=str(log_dir))
            logger.info("[OK] TensorBoard logger initialised at %s", log_dir)
        else:
            if enabled and not _TB_AVAILABLE:
                logger.warning(
                    "[WARN] tensorboard not installed -- logging disabled. "
                    "Install with: pip install tensorboard"
                )

    # ------------------------------------------------------------------
    # Basic scalars
    # ------------------------------------------------------------------

    def log_scalar(self, tag: str, value: float, step: int) -> None:
        """Log a single scalar value."""
        if self._writer is not None:
            self._writer.add_scalar(tag, value, global_step=step)

    def log_loss(self, loss: float, step: int) -> None:
        self.log_scalar("train/loss", loss, step)

    def log_lr(self, lr: float, step: int) -> None:
        self.log_scalar("train/lr", lr, step)

    def log_epoch_loss(self, loss: float, epoch: int) -> None:
        self.log_scalar("train/epoch_loss", loss, epoch)

    def log_grad_norm(self, norm: float, step: int) -> None:
        self.log_scalar("train/grad_norm", norm, step)

    # ------------------------------------------------------------------
    # Per-layer gradient norms (heavy)
    # ------------------------------------------------------------------

    def log_per_layer_grad_norms(
        self,
        model: nn.Module,
        step: int,
        prefix: str = "grad_norm",
    ) -> Dict[str, float]:
        """Compute and log the L2 gradient norm of every named parameter.

        Only parameters with ``requires_grad=True`` and a non-None
        ``.grad`` are considered (i.e. only LoRA parameters).

        Returns:
            Dict mapping ``{prefix}/{param_name}`` -> norm value.
        """
        if self._writer is None:
            return {}
        norms: Dict[str, float] = {}
        with torch.no_grad():
            for name, param in model.named_parameters():
                if param.requires_grad and param.grad is not None:
                    # Compute norm in native dtype (avoids creating a
                    # temporary fp32 GPU copy of each gradient tensor).
                    norm_val = param.grad.data.norm(2).item()
                    tag = f"{prefix}/{name}"
                    norms[tag] = norm_val
                    if self._writer is not None:
                        self._writer.add_scalar(tag, norm_val, global_step=step)
        return norms

    # ------------------------------------------------------------------
    # Estimation scores
    # ------------------------------------------------------------------

    def log_estimation_scores(
        self,
        scores: Dict[str, float],
        step: int = 0,
        prefix: str = "estimation",
    ) -> None:
        """Log estimation gradient sensitivity scores."""
        for module_name, score in scores.items():
            tag = f"{prefix}/{module_name}"
            self.log_scalar(tag, score, step)

    # ------------------------------------------------------------------
    # Histogram helpers
    # ------------------------------------------------------------------

    def log_param_histogram(
        self,
        model: nn.Module,
        step: int,
        prefix: str = "params",
    ) -> None:
        """Log weight histograms for trainable parameters."""
        if self._writer is None:
            return
        with torch.no_grad():
            for name, param in model.named_parameters():
                if param.requires_grad:
                    # Move to CPU in native dtype first, then upcast
                    # to float32 on CPU (avoids GPU fp32 intermediate).
                    cpu_data = param.data.cpu().float()
                    self._writer.add_histogram(
                        f"{prefix}/{name}", cpu_data, global_step=step,
                    )
                    del cpu_data

    # ------------------------------------------------------------------
    # Timestep distribution
    # ------------------------------------------------------------------

    def log_timestep_histogram(
        self, timesteps: torch.Tensor, step: int,
    ) -> None:
        """Log sampled timestep distribution as a histogram."""
        if self._writer is not None:
            self._writer.add_histogram(
                "train/timestep_distribution",
                timesteps.cpu().float(),
                global_step=step,
            )

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def flush(self) -> None:
        if self._writer is not None:
            self._writer.flush()

    def close(self) -> None:
        if self._writer is not None:
            self._writer.close()
            self._writer = None

    def __enter__(self) -> "TrainingLogger":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()
