"""
Loss-weighted adaptive timestep sampler for flow matching training.

Maintains a per-bin EMA of training loss across the [0, 1] timestep
range.  A configurable fraction of each batch's timesteps are sampled
from the loss-weighted distribution (higher loss → more samples),
while the remainder uses the standard logit-normal base sampler.

Only applies to base/SFT continuous sampling; turbo uses fixed 8 steps.
"""

from __future__ import annotations

import logging
from typing import Callable, Dict, Optional, Tuple

import torch

logger = logging.getLogger(__name__)


class AdaptiveTimestepSampler:
    """Adaptive timestep sampler with loss-weighted bin selection.

    Args:
        n_bins: Number of uniform bins spanning [0, 1].
        ema_decay: Smoothing factor for per-bin loss tracking.
        ratio: Fraction of batch sampled adaptively (rest from base).
    """

    def __init__(
        self,
        n_bins: int = 10,
        ema_decay: float = 0.99,
        ratio: float = 0.3,
    ) -> None:
        if not (0.0 <= ratio <= 1.0):
            raise ValueError(f"ratio must be in [0, 1] (got {ratio})")
        if n_bins < 2:
            raise ValueError(f"n_bins must be >= 2 (got {n_bins})")

        self.n_bins = n_bins
        self.ema_decay = ema_decay
        self.ratio = ratio
        # Per-bin loss EMA, initialised to 1.0 (uniform prior).
        self._bin_loss = torch.ones(n_bins, dtype=torch.float32)
        self._bin_counts = torch.zeros(n_bins, dtype=torch.int64)
        self._total_updates = 0

    # ------------------------------------------------------------------
    # Update
    # ------------------------------------------------------------------

    @torch.no_grad()
    def update(self, timesteps: torch.Tensor, losses: torch.Tensor) -> None:
        """Update per-bin loss EMA from a training batch.

        Args:
            timesteps: Sampled timesteps ``[B]`` in [0, 1].
            losses: Per-sample loss ``[B]`` (positive floats).
        """
        t_cpu = timesteps.detach().float().cpu()
        l_cpu = losses.detach().float().cpu()

        bins = (t_cpu * self.n_bins).long().clamp(0, self.n_bins - 1)
        d = self.ema_decay

        for i in range(self.n_bins):
            mask = bins == i
            if mask.any():
                bin_mean = l_cpu[mask].mean()
                self._bin_loss[i] = d * self._bin_loss[i] + (1.0 - d) * bin_mean
                self._bin_counts[i] += mask.sum()

        self._total_updates += 1

    # ------------------------------------------------------------------
    # Sample
    # ------------------------------------------------------------------

    def sample(
        self,
        batch_size: int,
        base_sampler: Callable[..., Tuple[torch.Tensor, torch.Tensor]],
        device: torch.device,
        dtype: torch.dtype,
        **base_kwargs,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample timesteps mixing adaptive + base distributions.

        Args:
            batch_size: Total samples to produce.
            base_sampler: The standard ``sample_timesteps`` function.
            device: Target device for output tensors.
            dtype: Target dtype for output tensors.
            **base_kwargs: Forwarded to ``base_sampler``.

        Returns:
            ``(t, r)`` tensors of shape ``[batch_size]``.
        """
        n_adaptive = int(batch_size * self.ratio)
        n_base = batch_size - n_adaptive

        parts_t = []
        parts_r = []

        # Base portion
        if n_base > 0:
            t_base, r_base = base_sampler(
                batch_size=n_base, device=device, dtype=dtype, **base_kwargs,
            )
            parts_t.append(t_base)
            parts_r.append(r_base)

        # Adaptive portion — sample bins proportional to loss EMA
        if n_adaptive > 0:
            weights = self._bin_loss.clamp(min=1e-8)
            probs = weights / weights.sum()
            bin_indices = torch.multinomial(
                probs, n_adaptive, replacement=True,
            )
            # Uniform sample within each selected bin
            bin_lo = bin_indices.float() / self.n_bins
            bin_hi = (bin_indices.float() + 1.0) / self.n_bins
            t_adaptive = (
                bin_lo + (bin_hi - bin_lo) * torch.rand(n_adaptive)
            ).to(device=device, dtype=dtype)
            # r = t (same as use_meanflow=False)
            parts_t.append(t_adaptive)
            parts_r.append(t_adaptive.clone())

        t = torch.cat(parts_t, dim=0)
        r = torch.cat(parts_r, dim=0)

        # Shuffle so adaptive/base samples are interleaved
        perm = torch.randperm(batch_size, device=device)
        return t[perm], r[perm]

    # ------------------------------------------------------------------
    # Histogram for TensorBoard
    # ------------------------------------------------------------------

    def get_histogram(self) -> torch.Tensor:
        """Return per-bin loss EMA as a 1-D tensor for logging."""
        return self._bin_loss.clone()

    # ------------------------------------------------------------------
    # Checkpoint support
    # ------------------------------------------------------------------

    def state_dict(self) -> Dict[str, object]:
        """Serialize sampler state."""
        return {
            "n_bins": self.n_bins,
            "ema_decay": self.ema_decay,
            "ratio": self.ratio,
            "bin_loss": self._bin_loss.clone(),
            "bin_counts": self._bin_counts.clone(),
            "total_updates": self._total_updates,
        }

    def load_state_dict(self, state: Dict[str, object]) -> None:
        """Restore sampler state from a checkpoint.

        Raises:
            ValueError: If bin count doesn't match.
        """
        saved_bins = state["bin_loss"]
        if saved_bins.shape[0] != self.n_bins:
            raise ValueError(
                f"Checkpoint has {saved_bins.shape[0]} bins "
                f"but sampler has {self.n_bins}"
            )
        self._bin_loss = saved_bins.clone()
        self._bin_counts = state["bin_counts"].clone()
        self._total_updates = int(state.get("total_updates", 0))
