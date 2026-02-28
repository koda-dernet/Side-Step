"""Unified dataset validation shared by CLI, Wizard, and GUI.

Provides a single ``validate_dataset`` function that returns a
:class:`DatasetStatus` describing what was found and any issues.
No UI imports -- callers handle presentation.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

from sidestep_engine.core.constants import AUDIO_EXTENSIONS

logger = logging.getLogger(__name__)


@dataclass
class DatasetStatus:
    """Result of validating a dataset directory."""

    kind: str = "empty"
    """One of: ``preprocessed``, ``raw_audio``, ``mixed``, ``empty``, ``invalid``."""

    tensor_count: int = 0
    audio_count: int = 0
    total_duration_s: int = 0

    issues: List[str] = field(default_factory=list)
    """Human-readable warnings (e.g. stale tensors, missing manifest)."""

    model_hash: Optional[str] = None
    """Model variant from ``preprocess_meta.json`` (used for staleness checks)."""

    is_stale: bool = False
    """True when tensors were preprocessed with a different model variant."""

    is_trainable: bool = False
    """True when the dataset can be used for training as-is (no preprocessing needed)."""

    has_manifest: bool = False
    """True when a ``manifest.json`` file is present."""


def validate_dataset(
    dataset_dir: str,
    *,
    expected_model_variant: Optional[str] = None,
) -> DatasetStatus:
    """Validate a dataset directory and return its status.

    Checks for:
    - Existence of the directory.
    - Presence of ``.pt`` tensor files.
    - Presence of audio files with supported extensions.
    - Validity of ``manifest.json`` (if present).
    - Model variant match via ``preprocess_meta.json``.

    Args:
        dataset_dir: Path to the dataset folder.
        expected_model_variant: When given, compare against the variant
            stored in ``preprocess_meta.json`` to detect stale tensors.
    """
    status = DatasetStatus()

    d = Path(dataset_dir)
    if not d.is_dir():
        status.kind = "invalid"
        status.issues.append(f"Dataset directory not found: {d}")
        return status

    pt_files = list(d.glob("*.pt"))
    audio_files = [
        f for f in d.iterdir()
        if f.is_file() and f.suffix.lower() in AUDIO_EXTENSIONS
    ]

    status.tensor_count = len(pt_files)
    status.audio_count = len(audio_files)

    # -- Check manifest.json --------------------------------------------------
    manifest_path = d / "manifest.json"
    if manifest_path.is_file():
        status.has_manifest = True
        try:
            raw = json.loads(manifest_path.read_text(encoding="utf-8"))
            if not isinstance(raw, dict):
                status.issues.append("manifest.json must be a JSON object with a 'samples' list")
            else:
                samples = raw.get("samples", [])
                if not isinstance(samples, list):
                    status.issues.append("manifest.json field 'samples' must be a list")
                elif not samples:
                    status.issues.append("manifest.json has no samples")
                elif not any(isinstance(s, str) and s.strip() for s in samples):
                    status.issues.append("manifest.json samples are empty or not string paths")
        except json.JSONDecodeError as exc:
            status.issues.append(
                f"manifest.json is invalid JSON: {exc}. "
                "Escape backslashes (\\\\) or use forward slashes (/)."
            )
        except Exception as exc:
            status.issues.append(f"Failed to read manifest.json: {exc}")

    # -- Check preprocess_meta.json for model variant -------------------------
    meta_path = d / "preprocess_meta.json"
    if meta_path.is_file():
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            status.model_hash = str(meta.get("model_variant") or "")
            dur = meta.get("total_duration") or meta.get("total_duration_s")
            if dur:
                status.total_duration_s = int(float(dur))
        except Exception:
            pass

    if expected_model_variant and status.model_hash:
        if status.model_hash != expected_model_variant:
            status.is_stale = True
            status.issues.append(
                f"Tensors were preprocessed with '{status.model_hash}' but you "
                f"selected '{expected_model_variant}'. Re-preprocess to avoid "
                "silent quality loss."
            )

    # -- Classify the dataset -------------------------------------------------
    if pt_files and audio_files:
        status.kind = "mixed"
        status.is_trainable = True
    elif pt_files:
        status.kind = "preprocessed"
        status.is_trainable = len(status.issues) == 0
    elif audio_files:
        status.kind = "raw_audio"
        status.is_trainable = False
    elif status.has_manifest:
        status.kind = "preprocessed"
        status.is_trainable = len(status.issues) == 0
    else:
        status.kind = "empty"
        status.issues.append(
            "No .pt tensors, audio files, or manifest.json found in this folder"
        )

    return status
