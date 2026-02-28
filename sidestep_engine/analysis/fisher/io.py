"""Fisher map JSON persistence.

Handles saving analysis results to ``fisher_map.json`` and loading
them back with schema validation and staleness checks.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

_SCHEMA_VERSION = 2


def save_fisher_map(data: Dict[str, Any], path: str | Path) -> Path:
    """Write a Fisher map to disk atomically.

    Uses a temporary file + ``os.replace`` so a crash during write
    never leaves a truncated ``fisher_map.json``.

    Args:
        data: Fisher map dictionary (must contain ``"version"``).
        path: Target file path.

    Returns:
        The resolved path that was written.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data.setdefault("version", _SCHEMA_VERSION)
    staging = path.with_suffix(".json.__writing__")
    staging.write_text(json.dumps(data, indent=2), encoding="utf-8")
    os.replace(staging, path)
    logger.info("Fisher map saved to %s", path)
    return path


def load_fisher_map(
    path: str | Path,
    expected_variant: Optional[str] = None,
    dataset_dir: Optional[str | Path] = None,
    expected_num_layers: Optional[int] = None,
) -> Optional[Dict[str, Any]]:
    """Load a Fisher map from disk with validation.

    Args:
        path: Path to ``fisher_map.json``.
        expected_variant: If set, warn when the map's variant differs.
        dataset_dir: If set, warn when the dataset hash doesn't match.
        expected_num_layers: If set, warn when the map's architecture
            (``num_hidden_layers``) doesn't match the current model.

    Returns:
        The parsed dict, or ``None`` on failure.
    """
    path = Path(path)
    if not path.is_file():
        return None

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("Failed to read fisher map %s: %s", path, exc)
        return None

    if not isinstance(data, dict):
        logger.warning("Fisher map %s is not a JSON object", path)
        return None

    # Schema version check
    version = data.get("version", 1)
    if version < _SCHEMA_VERSION:
        logger.warning(
            "Fisher map %s is version %d (expected %d) -- consider re-running fisher",
            path, version, _SCHEMA_VERSION,
        )

    # Required fields
    for key in ("target_modules", "rank_pattern", "alpha_pattern"):
        if key not in data:
            logger.warning("Fisher map %s missing required key '%s'", path, key)
            return None

    # Variant staleness check
    if expected_variant and data.get("model_variant"):
        if data["model_variant"] != expected_variant:
            logger.warning(
                "Fisher map was computed for '%s' but training uses '%s'. "
                "Consider re-running fisher or use --ignore-fisher-map.",
                data["model_variant"], expected_variant,
            )

    # Architecture compatibility check
    if expected_num_layers is not None and data.get("num_hidden_layers") is not None:
        map_layers = data["num_hidden_layers"]
        if map_layers != expected_num_layers:
            logger.warning(
                "Fisher map was computed for a %d-layer model but current model has "
                "%d layers. The rank pattern will not map correctly. "
                "Re-run Preprocessing++ for this model.",
                map_layers, expected_num_layers,
            )

    # Dataset staleness check
    if dataset_dir and data.get("dataset_hash"):
        current = compute_dataset_hash(dataset_dir)
        if current != data["dataset_hash"]:
            logger.warning(
                "Dataset has changed since Fisher analysis. "
                "Consider re-running fisher."
            )

    return data


def compute_dataset_hash(dataset_dir: str | Path) -> str:
    """Hash the sorted list of .pt filenames and sizes in *dataset_dir*.

    A lightweight staleness detector -- catches added/removed files
    and re-preprocessed files (size changes from different normalization
    or chunking settings).
    """
    d = Path(dataset_dir)
    entries = sorted(
        (f.name, f.stat().st_size)
        for f in d.glob("*.pt") if f.is_file()
    )
    payload = "|".join(f"{name}:{size}" for name, size in entries)
    return hashlib.md5(payload.encode()).hexdigest()[:12]
