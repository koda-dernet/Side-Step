"""Canonical constants shared across all Side-Step interfaces.

Every magic number, variant mapping, and well-known default that more
than one module needs lives here so there is exactly one source of truth.
"""

from __future__ import annotations

import sys
from typing import Dict, FrozenSet

# ---------------------------------------------------------------------------
# Model variant → checkpoint subdirectory mapping
# ---------------------------------------------------------------------------

VARIANT_DIR_MAP: Dict[str, str] = {
    "turbo": "acestep-v15-turbo",
    "base": "acestep-v15-base",
    "sft": "acestep-v15-sft",
}

# ---------------------------------------------------------------------------
# Per-variant default parameters (timestep, shift, inference steps)
# ---------------------------------------------------------------------------

BASE_MODEL_DEFAULTS: Dict[str, Dict] = {
    "turbo": {
        "is_turbo": True,
        "timestep_mu": -0.4,
        "timestep_sigma": 1.0,
        "shift": 3.0,
        "num_inference_steps": 8,
    },
    "base": {
        "is_turbo": False,
        "timestep_mu": -0.4,
        "timestep_sigma": 1.0,
        "shift": 1.0,
        "num_inference_steps": 50,
    },
    "sft": {
        "is_turbo": False,
        "timestep_mu": -0.4,
        "timestep_sigma": 1.0,
        "shift": 1.0,
        "num_inference_steps": 50,
    },
}

# ---------------------------------------------------------------------------
# Timestep / inference defaults (fallbacks when config.json is absent)
# ---------------------------------------------------------------------------

DEFAULT_TIMESTEP_MU: float = -0.4
DEFAULT_TIMESTEP_SIGMA: float = 1.0
DEFAULT_DATA_PROPORTION: float = 0.5

TURBO_SHIFT: float = 3.0
BASE_SHIFT: float = 1.0

TURBO_INFERENCE_STEPS: int = 8
BASE_INFERENCE_STEPS: int = 50

# Threshold: learning rates above this when using PP++ Fisher maps
# usually indicate the user forgot to lower their LR.
PP_LR_WARN_THRESHOLD: float = 1e-4

# ---------------------------------------------------------------------------
# Audio / data constants
# ---------------------------------------------------------------------------

AUDIO_EXTENSIONS: FrozenSet[str] = frozenset(
    {".wav", ".mp3", ".flac", ".ogg", ".opus", ".m4a", ".aac"}
)
"""Superset of all supported audio formats across preprocessing and GUI."""

LATENT_FPS: float = 25.0
"""Latent frames per second produced by the ACE-Step VAE."""

ADAPTER_TYPES: FrozenSet[str] = frozenset(
    {"lora", "dora", "lokr", "loha", "oft"}
)

# ---------------------------------------------------------------------------
# DataLoader workers (platform-dependent default)
# ---------------------------------------------------------------------------

DEFAULT_NUM_WORKERS: int = 0 if sys.platform == "win32" else 4


# ---------------------------------------------------------------------------
# Adapter compatibility helpers
# ---------------------------------------------------------------------------

PP_COMPATIBLE_ADAPTERS: FrozenSet[str] = frozenset({"lora", "dora"})
"""Adapter types that support Preprocessing++ (Fisher-guided per-module ranks)."""


def is_pp_compatible(adapter_type: str) -> bool:
    """Return True if *adapter_type* supports Preprocessing++ rank maps."""
    return adapter_type in PP_COMPATIBLE_ADAPTERS


def is_turbo(params: dict) -> bool:
    """Return ``True`` if parameters indicate a turbo-based model.

    Works with both wizard ``answers`` dicts and flat config dicts.
    Detection prefers ``base_model`` / ``model_variant`` name.  For
    unknown custom names, falls back to ``num_inference_steps``
    (8 = turbo-style schedule).
    """
    base = params.get("base_model", params.get("model_variant", "turbo"))
    label = base.lower() if isinstance(base, str) else ""
    if "turbo" in label:
        return True
    if "base" in label or "sft" in label:
        return False
    infer_steps = params.get("num_inference_steps", TURBO_INFERENCE_STEPS)
    try:
        return int(infer_steps) == TURBO_INFERENCE_STEPS
    except (TypeError, ValueError):
        return True
