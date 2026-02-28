"""
Model discovery and selection for Side-Step.

Scans a checkpoint directory for model subdirectories (identified by the
presence of a ``config.json``), classifies them as official or custom
(fine-tune), and provides an interactive fuzzy-search picker.
"""

from __future__ import annotations

import difflib
import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# Official ACE-Step model directory name patterns.
_OFFICIAL_PREFIXES = ("acestep-v15-", "acestep-v1-")

from sidestep_engine.core.constants import BASE_MODEL_DEFAULTS as _BASE_DEFAULTS


@dataclass
class ModelInfo:
    """Metadata about a discovered model directory."""

    name: str
    path: Path
    is_official: bool
    config: Dict = field(default_factory=dict)
    base_model: str = "unknown"


# ---------------------------------------------------------------------------
# Scanning
# ---------------------------------------------------------------------------

def scan_models(checkpoint_dir: str | Path) -> List[ModelInfo]:
    """Scan *checkpoint_dir* for model subdirectories.

    A valid model directory contains a ``config.json`` with either an
    ``auto_map`` key (HuggingFace custom model) or a ``model_type`` key.
    Non-model directories (``vae/``, ``Qwen3-*``, ``*.lm-*``) are
    excluded automatically.

    Returns a sorted list of :class:`ModelInfo` (officials first, then
    alphabetical).
    """
    ckpt = Path(checkpoint_dir)
    if not ckpt.is_dir():
        return []

    skip_names = {"vae", ".git", "__pycache__"}
    skip_prefixes = ("Qwen", "acestep-5Hz")

    results: List[ModelInfo] = []
    for child in sorted(ckpt.iterdir()):
        if not child.is_dir():
            continue
        if child.name in skip_names:
            continue
        if any(child.name.startswith(p) for p in skip_prefixes):
            continue

        cfg_path = child / "config.json"
        if not cfg_path.is_file():
            continue

        try:
            cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            continue

        # Must look like a model config (not just any JSON)
        if "auto_map" not in cfg and "model_type" not in cfg:
            continue

        is_official = any(child.name.startswith(p) for p in _OFFICIAL_PREFIXES)
        base = detect_base_model(cfg, child.name)

        results.append(ModelInfo(
            name=child.name,
            path=child,
            is_official=is_official,
            config=cfg,
            base_model=base,
        ))

    # Sort: officials first, then alphabetical
    results.sort(key=lambda m: (not m.is_official, m.name))
    return results


# ---------------------------------------------------------------------------
# Base-model detection
# ---------------------------------------------------------------------------

def detect_base_model(config: Dict, dir_name: str = "") -> str:
    """Infer which base variant a model descends from.

    Uses ``is_turbo`` flag and directory name heuristics.  Returns one
    of ``"turbo"``, ``"base"``, ``"sft"``, or ``"unknown"``.
    """
    # Explicit is_turbo flag
    if config.get("is_turbo", False):
        return "turbo"

    # Match by directory name for official models
    name_lower = dir_name.lower()
    for variant in ("turbo", "base", "sft"):
        if variant in name_lower:
            return variant

    return "unknown"


def get_base_defaults(base_model: str) -> Dict:
    """Return default timestep params for a known base variant."""
    return dict(_BASE_DEFAULTS.get(base_model, _BASE_DEFAULTS["base"]))


# ---------------------------------------------------------------------------
# Fuzzy search
# ---------------------------------------------------------------------------

def fuzzy_search(query: str, models: List[ModelInfo]) -> List[ModelInfo]:
    """Filter models by fuzzy name match.

    Tries substring match first, then ``difflib.get_close_matches``.
    Returns matching models in relevance order.
    """
    if not query:
        return list(models)

    q = query.lower()

    # 1. Substring matches (most intuitive)
    substring_hits = [m for m in models if q in m.name.lower()]
    if substring_hits:
        return substring_hits

    # 2. Fuzzy matches via difflib
    names = [m.name for m in models]
    close = difflib.get_close_matches(query, names, n=5, cutoff=0.4)
    name_set = set(close)
    return [m for m in models if m.name in name_set]


# ---------------------------------------------------------------------------
# Weight file check (best-effort)
# ---------------------------------------------------------------------------

def _has_weight_files(model_path: Path) -> bool:
    """Return True if model dir appears to contain loadable weights.

    Checks for common patterns: model.safetensors, pytorch_model.bin, *.safetensors.
    Used to warn about incomplete downloads; does not exclude from menu.
    """
    if not model_path.is_dir():
        return False
    for name in ("model.safetensors", "pytorch_model.bin"):
        if (model_path / name).is_file():
            return True
    return any(model_path.glob("*.safetensors"))


def warn_if_no_weights(model_path: Path, model_name: str) -> None:
    """Log a warning if model directory has no obvious weight files.

    Does NOT import UI modules -- callers in the UI layer should
    present the warning however they prefer.
    """
    if _has_weight_files(model_path):
        return
    logger.warning(
        "'%s' has no model.safetensors, pytorch_model.bin, or *.safetensors. "
        "This may be an incomplete download.",
        model_name,
    )


# Backward-compat aliases (will be removed in a future release).
# Interactive functions now live in ui.flows.wizard_shared_steps.
def pick_model(checkpoint_dir: str | Path) -> Optional[Tuple[str, "ModelInfo"]]:
    """Deprecated -- use ``sidestep_engine.ui.flows.wizard_shared_steps.pick_model``."""
    from sidestep_engine.ui.flows.wizard_shared_steps import pick_model as _pick
    return _pick(checkpoint_dir)


def prompt_base_model(model_name: str) -> str:
    """Deprecated -- use ``sidestep_engine.ui.flows.wizard_shared_steps.prompt_base_model``."""
    from sidestep_engine.ui.flows.wizard_shared_steps import prompt_base_model as _prompt
    return _prompt(model_name)
