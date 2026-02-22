"""Spectral analysis for LoRA rank sizing.

Computes the SVD effective rank of each targetable weight matrix.
The effective rank measures how many singular-value dimensions explain
95% of the matrix's energy, indicating structural complexity.
"""

from __future__ import annotations

import gc
import logging
from typing import Dict, List, Tuple

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

_ENERGY_THRESHOLD = 0.95


def compute_spectral_complexity(
    target_modules: List[Tuple[str, nn.Module]],
    device: torch.device,
    threshold: float = _ENERGY_THRESHOLD,
) -> Dict[str, int]:
    """Compute SVD effective rank for each targetable module.

    Each weight matrix is cast to fp32 for numerical accuracy, the
    singular values are computed, and the effective rank is the number
    of singular values that explain *threshold* of the total energy.

    Modules are processed one at a time to keep peak VRAM low (~100 MB).

    Args:
        target_modules: ``[(name, nn.Module)]`` from module discovery.
        device: Torch device (GPU recommended for speed).
        threshold: Cumulative energy fraction (default 0.95).

    Returns:
        ``{module_name: effective_rank}``.
    """
    results: Dict[str, int] = {}
    total = len(target_modules)

    for idx, (name, mod) in enumerate(target_modules):
        try:
            eff_rank = _effective_rank_of(mod.weight.data, device, threshold)
            results[name] = eff_rank
        except Exception as exc:
            logger.warning("SVD failed for %s: %s -- skipping spectral data", name, exc)
            results[name] = -1  # sentinel for "unknown"
        finally:
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            elif hasattr(torch, 'mps') and torch.mps.is_available():
                torch.mps.empty_cache()

        if (idx + 1) % 50 == 0 or idx == total - 1:
            logger.debug("Spectral analysis: %d/%d modules done", idx + 1, total)

    return results


def _effective_rank_of(
    weight: torch.Tensor,
    device: torch.device,
    threshold: float,
) -> int:
    """Compute effective rank for a single weight tensor.

    Args:
        weight: 2-D parameter tensor (any dtype).
        device: Device for computation.
        threshold: Cumulative energy threshold.

    Returns:
        Number of singular values explaining *threshold* of total energy.
    """
    W = weight.to(device=device, dtype=torch.float32)
    svdvals = torch.linalg.svdvals(W)
    energy = svdvals.pow(2)
    total_energy = energy.sum()
    if total_energy < 1e-12:
        return 1
    cumulative = torch.cumsum(energy, dim=0) / total_energy
    eff_rank = int((cumulative < threshold).sum().item()) + 1
    del W, svdvals, energy, cumulative
    return eff_rank
