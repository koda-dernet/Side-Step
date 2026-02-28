"""
Exponential Moving Average (EMA) of adapter weights.

Maintains a shadow copy of trainable parameters on CPU, updated each
optimizer step.  The EMA weights are swapped in for evaluation and saving,
then restored so training continues with the raw weights.

CPU storage keeps GPU VRAM overhead at zero — at LoRA rank 8 the shadow
copy is ~5 MB.
"""

from __future__ import annotations

import copy
import logging
from typing import Dict, Iterable, List

import torch

logger = logging.getLogger(__name__)


class AdapterEMA:
    """EMA tracker for adapter (or any trainable) parameters.

    Args:
        params: Iterable of trainable ``nn.Parameter`` objects.
        decay: EMA decay rate.  Higher = smoother.
            Typical: 0.9999 (slow), 0.999 (faster tracking).
    """

    def __init__(self, params: Iterable[torch.nn.Parameter], decay: float) -> None:
        if not (0.0 <= decay < 1.0):
            raise ValueError(f"EMA decay must be >= 0 and < 1 (got {decay})")
        self.decay = decay
        # Deep-clone to CPU in float32 so we don't consume GPU VRAM
        # and accumulate in full precision regardless of mixed-precision dtype.
        self._params: List[torch.nn.Parameter] = list(params)
        self._shadow: List[torch.Tensor] = [
            p.data.detach().clone().float().cpu() for p in self._params
        ]
        # Backup slot for restore-after-apply.
        self._backup: List[torch.Tensor] = []
        self._step_count: int = 0

    # ------------------------------------------------------------------
    # Core operations
    # ------------------------------------------------------------------

    @torch.no_grad()
    def update(self) -> None:
        """Update shadow params: ``shadow = decay * shadow + (1-decay) * param``."""
        d = self.decay
        for shadow, param in zip(self._shadow, self._params):
            # Cast to float32 to match shadow dtype (params may be bf16/fp16).
            shadow.lerp_(param.data.detach().float().cpu(), 1.0 - d)
        self._step_count += 1

    @torch.no_grad()
    def apply(self) -> None:
        """Copy EMA shadow weights into the model (for eval / save).

        Call :meth:`restore` afterwards to resume training with raw weights.
        """
        self._backup = [p.data.detach().clone() for p in self._params]
        for shadow, param in zip(self._shadow, self._params):
            param.data.copy_(shadow.to(dtype=param.dtype, device=param.device))

    @torch.no_grad()
    def restore(self) -> None:
        """Restore raw training weights after an :meth:`apply` call."""
        if not self._backup:
            logger.warning("[EMA] restore() called without a prior apply()")
            return
        for backup, param in zip(self._backup, self._params):
            param.data.copy_(backup)
        self._backup = []

    # ------------------------------------------------------------------
    # Checkpoint support
    # ------------------------------------------------------------------

    def state_dict(self) -> Dict[str, object]:
        """Serialize EMA state for checkpoint saving."""
        return {
            "decay": self.decay,
            "step_count": self._step_count,
            "shadow": [s.clone() for s in self._shadow],
        }

    def load_state_dict(self, state: Dict[str, object]) -> None:
        """Restore EMA state from a checkpoint.

        Raises:
            ValueError: If shadow tensor count doesn't match params.
        """
        saved_shadow = state["shadow"]
        if len(saved_shadow) != len(self._shadow):
            raise ValueError(
                f"EMA checkpoint has {len(saved_shadow)} shadow tensors "
                f"but model has {len(self._shadow)} trainable params"
            )
        for i, s in enumerate(saved_shadow):
            self._shadow[i].copy_(s)
        self._step_count = int(state.get("step_count", 0))
        saved_decay = state.get("decay")
        if saved_decay is not None and saved_decay != self.decay:
            logger.info(
                "[EMA] Checkpoint decay=%.6f differs from current=%.6f; "
                "using current value",
                saved_decay,
                self.decay,
            )
