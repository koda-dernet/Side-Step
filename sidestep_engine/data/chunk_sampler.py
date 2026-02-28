"""
Coverage-weighted chunk sampler for latent chunking.

Replaces uniform random offset selection with inverse-frequency
histogram sampling so that under-trained regions of each audio sample
get higher selection probability.  Tracks per-sample coverage across
epochs and supports persistence for training resume.

State is backward-compatible: old checkpoints without coverage data
simply start with a fresh histogram.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional

import torch


class CoverageChunkSampler:
    """Selects chunk offsets biased toward less-trained regions.

    Each sample's time axis is divided into *n_bins* bins.  On every
    call to :meth:`sample_offset`, the bin with fewer past hits is more
    likely to be chosen.  A periodic decay halves all counts every
    *decay_every* epochs to prevent permanent lock-out.

    Args:
        n_bins: Number of histogram bins per sample (default 32).
        decay_every: Apply 0.5× decay to all counts every N epochs.
            ``0`` disables decay entirely.
    """

    def __init__(self, n_bins: int = 32, decay_every: int = 10) -> None:
        self.n_bins = max(1, n_bins)
        self.decay_every = max(0, decay_every)
        self._histograms: Dict[str, torch.Tensor] = {}
        self._last_decay_epoch: int = 0

    def sample_offset(
        self,
        sample_key: str,
        total_frames: int,
        chunk_frames: int,
    ) -> int:
        """Return a coverage-weighted random start offset.

        Args:
            sample_key: Unique identifier for the sample (e.g. file path).
            total_frames: Total time-axis length of the sample.
            chunk_frames: Length of the chunk window in frames.

        Returns:
            Integer start offset in ``[0, total_frames - chunk_frames)``.
        """
        max_offset = total_frames - chunk_frames
        if max_offset <= 0:
            return 0

        hist = self._get_or_create(sample_key)

        # Compute inverse-frequency weights: w[i] = 1 / (hits[i] + 1)
        weights = 1.0 / (hist.float() + 1.0)

        # Sample a bin proportionally
        bin_idx = torch.multinomial(weights, 1).item()

        # Pick a uniform offset within that bin's range
        bin_size = max_offset / self.n_bins
        bin_start = int(bin_idx * bin_size)
        bin_end = min(int((bin_idx + 1) * bin_size), max_offset)
        if bin_end <= bin_start:
            offset = bin_start
        else:
            offset = bin_start + torch.randint(0, bin_end - bin_start, (1,)).item()

        # Record the hit
        hist[bin_idx] += 1
        return offset

    def notify_epoch(self, epoch: int) -> None:
        """Call at the start of each epoch to apply periodic decay.

        Args:
            epoch: The current (0-indexed) epoch number.
        """
        if self.decay_every <= 0:
            return
        if epoch > 0 and epoch != self._last_decay_epoch and epoch % self.decay_every == 0:
            self._apply_decay()
            self._last_decay_epoch = epoch

    def state_dict(self) -> Dict[str, Any]:
        """Serialize coverage state for checkpoint persistence.

        Returns:
            A plain dict safe for ``torch.save``.
        """
        return {
            "n_bins": self.n_bins,
            "decay_every": self.decay_every,
            "last_decay_epoch": self._last_decay_epoch,
            "histograms": {k: v.tolist() for k, v in self._histograms.items()},
        }

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        """Restore coverage state from a checkpoint.

        Backward-compatible: missing or malformed state is silently
        ignored, starting with a fresh histogram.

        Args:
            state: Dict previously returned by :meth:`state_dict`.
        """
        if not isinstance(state, dict):
            return
        self.n_bins = state.get("n_bins", self.n_bins)
        self.decay_every = state.get("decay_every", self.decay_every)
        self._last_decay_epoch = state.get("last_decay_epoch", 0)
        raw = state.get("histograms", {})
        self._histograms = {}
        for k, v in raw.items():
            if isinstance(v, list) and len(v) == self.n_bins:
                self._histograms[k] = torch.tensor(v, dtype=torch.long)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_or_create(self, key: str) -> torch.Tensor:
        """Return the histogram for *key*, creating one if needed."""
        if key not in self._histograms:
            self._histograms[key] = torch.zeros(self.n_bins, dtype=torch.long)
        return self._histograms[key]

    def _apply_decay(self) -> None:
        """Halve all histogram counts (floor division)."""
        for hist in self._histograms.values():
            hist.div_(2, rounding_mode="floor")
