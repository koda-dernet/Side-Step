"""
Validation epoch runner for adapter training.

Runs a single no-grad pass over a held-out validation DataLoader,
returning the mean loss.  Stateless — the trainer orchestrates when
to call this and what to do with the result.
"""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from torch.utils.data import DataLoader

logger = logging.getLogger(__name__)


def run_validation_epoch(
    module: object,
    val_loader: "DataLoader",
    device: torch.device,
) -> float:
    """Run a no-grad validation pass and return mean loss.

    Args:
        module: A ``FixedLoRAModule`` (or any object with a
            ``training_step(batch) -> Tensor`` method).
        val_loader: Validation DataLoader.  If empty, returns
            ``float('inf')``.
        device: Target device (unused directly — module handles
            device transfers internally).

    Returns:
        Mean validation loss as a Python float, or ``float('inf')``
        if the loader yielded no valid batches.
    """
    total_loss = 0.0
    n_batches = 0

    was_training = getattr(module, "training", False)
    if hasattr(module, "model") and hasattr(module.model, "decoder"):
        module.model.decoder.eval()

    try:
        with torch.no_grad():
            for batch in val_loader:
                try:
                    loss = module.training_step(batch)
                except Exception:
                    logger.debug("[Validation] Skipping batch due to error", exc_info=True)
                    continue

                if torch.isnan(loss) or torch.isinf(loss):
                    continue

                total_loss += loss.item()
                n_batches += 1
    finally:
        if hasattr(module, "model") and hasattr(module.model, "decoder"):
            if was_training:
                module.model.decoder.train()

    if n_batches == 0:
        logger.warning("[Validation] No valid batches — returning inf")
        return float("inf")

    mean_loss = total_loss / n_batches
    return mean_loss
