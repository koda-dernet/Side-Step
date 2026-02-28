"""Selective gradient checkpointing for the ACE-Step decoder.

Provides partial checkpointing (checkpoint a fraction of decoder layers)
as a middle ground between full checkpointing (slow, low VRAM) and none
(fast, high VRAM).

VRAM estimation and suggestion helpers live in
``sidestep_engine.core.vram_estimation`` and are re-exported here for
backward compatibility.
"""

from __future__ import annotations

import logging
from typing import List, Optional

import torch
import torch.nn as nn

# Re-export estimation API so existing callers keep working.
from sidestep_engine.core.vram_estimation import (  # noqa: F401
    _BACKWARD_MULTIPLIER,
    _MB_PER_LAYER_60S,
    _MODEL_OVERHEAD_NO_OFFLOAD_MB,
    _MODEL_OVERHEAD_OFFLOAD_MB,
    build_checkpointing_options,
    detect_attn_backend,
    estimate_activation_mb,
    estimate_optimizer_state_mb,
    estimate_peak_vram_mb,
    suggest_checkpointing,
)

logger = logging.getLogger(__name__)

_ORIG_FWD_ATTR = "_selective_ckpt_orig_forward"


# ---------------------------------------------------------------------------
# Apply / remove selective checkpointing
# ---------------------------------------------------------------------------

def _find_decoder_layers(decoder: nn.Module) -> Optional[nn.ModuleList]:
    """Locate the main transformer layer list inside the decoder."""
    for attr in ("layers", "blocks", "transformer_blocks"):
        candidate = getattr(decoder, attr, None)
        if isinstance(candidate, nn.ModuleList) and len(candidate) > 0:
            return candidate
    # Walk one level of wrappers
    for child in decoder.children():
        for attr in ("layers", "blocks", "transformer_blocks"):
            candidate = getattr(child, attr, None)
            if isinstance(candidate, nn.ModuleList) and len(candidate) > 0:
                return candidate
    return None


def _make_checkpointed_forward(original_forward):
    """Wrap a layer forward with torch.utils.checkpoint."""
    def checkpointed_forward(*args, **kwargs):
        return torch.utils.checkpoint.checkpoint(
            original_forward, *args, use_reentrant=False, **kwargs,
        )
    return checkpointed_forward


def _select_layer_indices(num_layers: int, ratio: float) -> List[int]:
    """Return evenly-spaced layer indices to checkpoint.

    Args:
        num_layers: Total decoder layers.
        ratio: Fraction to checkpoint (0.0–1.0).

    Returns:
        Sorted list of layer indices that should be checkpointed.
    """
    count = max(0, min(num_layers, round(num_layers * ratio)))
    if count == 0:
        return []
    if count >= num_layers:
        return list(range(num_layers))
    if count == 1:
        return [0]
    # Even spacing: pick `count` indices spread across the range
    return sorted(set(
        round(i * (num_layers - 1) / (count - 1)) for i in range(count)
    ))


def apply_selective_checkpointing(
    decoder: nn.Module, ratio: float, num_layers: Optional[int] = None,
) -> int:
    """Wrap a fraction of decoder layers with torch.utils.checkpoint.

    Args:
        decoder: The ACE-Step decoder module.
        ratio: Fraction of layers to checkpoint (0.0=none, 1.0=all).
        num_layers: Override layer count (auto-detected if None).

    Returns:
        Number of layers actually checkpointed.
    """
    if ratio <= 0.0:
        return 0

    layers = _find_decoder_layers(decoder)
    if layers is None:
        logger.warning(
            "[WARN] Could not find decoder layers for selective checkpointing"
        )
        return 0

    n = num_layers if num_layers is not None else len(layers)
    indices = _select_layer_indices(n, ratio)

    checkpointed = 0
    for idx in indices:
        if idx >= len(layers):
            continue
        layer = layers[idx]
        if hasattr(layer, _ORIG_FWD_ATTR):
            continue  # already wrapped
        setattr(layer, _ORIG_FWD_ATTR, layer.forward)
        layer.forward = _make_checkpointed_forward(layer.forward)
        checkpointed += 1

    logger.info(
        "[INFO] Selective gradient checkpointing: %d/%d layers (ratio=%.2f)",
        checkpointed, len(layers), ratio,
    )
    return checkpointed


def remove_selective_checkpointing(decoder: nn.Module) -> int:
    """Restore original forward methods on all selectively checkpointed layers.

    Returns:
        Number of layers restored.
    """
    layers = _find_decoder_layers(decoder)
    if layers is None:
        return 0

    restored = 0
    for layer in layers:
        orig = getattr(layer, _ORIG_FWD_ATTR, None)
        if orig is not None:
            layer.forward = orig
            delattr(layer, _ORIG_FWD_ATTR)
            restored += 1
    return restored
