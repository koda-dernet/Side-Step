"""Shared dataset scanning utilities used by GUI and Wizard.

Provides tensor-folder scanning, duration computation, and a
standardized ``DatasetInfo`` dict structure.  No UI or GUI imports.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from sidestep_engine.core.constants import AUDIO_EXTENSIONS, LATENT_FPS

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Duration helpers
# ---------------------------------------------------------------------------

def fmt_duration(seconds: int) -> str:
    """Format *seconds* as a human-readable ``Xh Xm Xs`` / ``Xm Xs`` string."""
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h > 0:
        return f"{h}h {m}m {s}s"
    return f"{m}m {s}s"


def pt_total_duration(folder: Path) -> int:
    """Sum approximate duration from ``.pt`` file metadata (seconds).

    Prefers ``preprocess_meta.json`` (written by preprocess.py) for O(1) lookup.
    Falls back to scanning individual ``.pt`` files if meta JSON is missing.
    """
    meta_json = folder / "preprocess_meta.json"
    if meta_json.is_file():
        try:
            meta = json.loads(meta_json.read_text(encoding="utf-8"))
            dur = meta.get("total_duration") or meta.get("total_duration_s")
            if dur and float(dur) > 0:
                return int(float(dur))
        except Exception:
            pass

    total = 0
    for f in folder.glob("*.pt"):
        try:
            import torch
            data = torch.load(f, map_location="cpu", weights_only=True)
            if not isinstance(data, dict):
                continue
            dur = 0
            meta = data.get("metadata")
            if isinstance(meta, dict):
                dur = meta.get("duration", 0)
            if not dur and "target_latents" in data:
                T = data["target_latents"].shape[0]
                dur = T / LATENT_FPS
            total += int(dur)
        except Exception:
            pass
    return total


# ---------------------------------------------------------------------------
# Tensor-folder scanning
# ---------------------------------------------------------------------------

def scan_tensor_folder(folder: Path) -> Optional[Dict[str, Any]]:
    """Return metadata dict for a single preprocessed tensor folder, or ``None``."""
    pt_files = list(folder.glob("*.pt"))
    if not pt_files:
        return None

    dur = pt_total_duration(folder)
    audio_linked = ""
    created_at = ""
    model_variant = ""
    normalize = ""
    meta_json = folder / "preprocess_meta.json"
    if meta_json.is_file():
        try:
            meta = json.loads(meta_json.read_text(encoding="utf-8"))
            audio_linked = meta.get("audio_dir") or meta.get("source_audio_dir", "")
            created_at = str(meta.get("created_at") or "")
            model_variant = str(meta.get("model_variant") or "")
            normalize = str(meta.get("normalize") or "")
        except Exception:
            pass

    pp_map = (folder / "fisher_map.json").is_file()
    return {
        "name": folder.name,
        "type": "tensors",
        "path": str(folder),
        "count": len(pt_files),
        "files_label": f"{len(pt_files)} .pt",
        "duration": dur,
        "duration_label": fmt_duration(dur),
        "audio_linked": audio_linked,
        "created_at": created_at,
        "model_variant": model_variant,
        "normalize": normalize,
        "pp_map": pp_map,
    }


def scan_tensors_dir(tensors_dir: Path) -> List[Dict[str, Any]]:
    """Scan *tensors_dir* for preprocessed tensor sub-folders."""
    if not tensors_dir.is_dir():
        return []
    datasets = []
    for child in sorted(tensors_dir.iterdir()):
        if not child.is_dir():
            continue
        info = scan_tensor_folder(child)
        if info is not None:
            datasets.append(info)
    return datasets


def scan_audio_folder(folder: Path) -> Optional[Dict[str, Any]]:
    """Return a metadata dict for an audio folder, or ``None`` if no audio found."""
    audio_files = [
        f for f in folder.rglob("*")
        if f.is_file() and f.suffix.lower() in AUDIO_EXTENSIONS
    ]
    if not audio_files:
        return None

    total_dur = 0
    try:
        from sidestep_engine.data.audio_duration import get_audio_duration
        for f in audio_files:
            try:
                total_dur += int(get_audio_duration(str(f)))
            except Exception:
                pass
    except ImportError:
        pass

    return {
        "name": folder.name,
        "type": "audio",
        "path": str(folder),
        "count": len(audio_files),
        "files_label": f"{len(audio_files)} files",
        "duration": total_dur,
        "duration_label": fmt_duration(total_dur),
        "audio_linked": "",
    }


def has_raw_audio(folder: Path) -> bool:
    """Return True if *folder* contains audio files (any depth)."""
    return any(
        f.suffix.lower() in AUDIO_EXTENSIONS
        for f in folder.rglob("*") if f.is_file()
    )


def has_preprocessed_tensors(folder: Path) -> bool:
    """Return True if *folder* contains ``.pt`` files."""
    return any(folder.glob("*.pt"))
