"""
Required wizard steps: configuration mode and core path collection.

Extracted from ``train_steps.py`` to meet the module LOC policy.
"""

from __future__ import annotations

import logging
from datetime import datetime

logger = logging.getLogger(__name__)

from sidestep_engine.ui.prompt_helpers import (
    ask,
    ask_output_path,
    menu,
    print_message,
    section,
)


# ---- Helpers ----------------------------------------------------------------

def _has_fisher_map(a: dict) -> bool:
    """Return True if a valid Preprocessing++ map exists in dataset dir.

    Caches the result in ``a["_fisher_map_cached"]`` so repeated calls
    within the same wizard run don't re-read the file.  The cache is
    invalidated when ``dataset_dir`` changes.
    """
    from pathlib import Path
    ds = a.get("dataset_dir")
    if not ds:
        return False

    # Invalidate cache when dataset_dir changes
    cached_dir = a.get("_fisher_map_cached_dir")
    if cached_dir != ds:
        a.pop("_fisher_map_cached", None)
        a["_fisher_map_cached_dir"] = ds

    cached = a.get("_fisher_map_cached")
    if cached is not None:
        return cached
    p = Path(ds) / "fisher_map.json"
    if not p.is_file():
        a["_fisher_map_cached"] = False
        return False
    try:
        import json
        data = json.loads(p.read_text(encoding="utf-8"))
        result = bool(data.get("rank_pattern"))
    except Exception as exc:
        logger.debug("Could not read fisher_map.json in %s: %s", ds, exc)
        result = False
    a["_fisher_map_cached"] = result
    return result


# ---- Steps ------------------------------------------------------------------

def step_config_mode(a: dict) -> None:
    """Choose basic vs advanced configuration depth."""
    a["config_mode"] = menu(
        "How much do you want to configure?",
        [
            ("basic", "Basic (recommended defaults, fewer questions)"),
            ("advanced", "Advanced (all settings exposed)"),
        ],
        default=1,
        allow_back=True,
    )


def _generate_run_base(a: dict) -> str:
    """Build the user-editable portion of the run name (no timestamp)."""
    adapter = a.get("adapter_type", "lora")
    variant = a.get("model_variant", "turbo")
    return f"{adapter}_{variant}"


def _stamp_run_name(base: str) -> str:
    """Append a non-editable timestamp suffix to a run-name base."""
    ts = datetime.now().strftime("%Y%m%d_%H%M")
    return f"{base}_{ts}"


def step_required(a: dict) -> None:
    """Collect required paths, model selection, run name, and output directory.

    The dataset folder prompt accepts **either** a folder of preprocessed
    ``.pt`` tensors (ready to train) **or** a folder of raw audio files
    (with optional ``.txt`` sidecar metadata).  When raw audio is detected
    the wizard will automatically preprocess before training — the user
    does not need to run a separate preprocess step.

    The run name is asked early so the output directory can be auto-derived
    as ``{trained_adapters_dir}/{adapter_type}/{run_name}/``.
    """
    from pathlib import Path
    from sidestep_engine.settings import get_trained_adapters_dir
    from sidestep_engine.ui.flows.common import _AUDIO_EXTENSIONS
    from sidestep_engine.ui.flows.wizard_shared_steps import (
        ask_model_and_checkpoint,
        ask_dataset_folder,
        show_whats_changed_notice,
    )

    section("Required Settings")
    ask_model_and_checkpoint(a, default_variant="turbo", prompt_base_model=True)
    show_whats_changed_notice()
    ask_dataset_folder(a, allow_audio=True)

    # Show extra detail for raw audio (training-specific UX)
    if a.get("_auto_preprocess_audio_dir"):
        ds = Path(a["dataset_dir"])
        audio_files = [
            f for f in ds.rglob("*")
            if f.is_file() and f.suffix.lower() in _AUDIO_EXTENSIONS
        ]
        txt_files = [f for f in ds.rglob("*.txt") if f.is_file()]
        has_dj = (ds / "dataset.json").is_file()

        print_message(
            f"\nFound {len(audio_files)} audio file(s)"
            f" and {len(txt_files)} .txt sidecar(s) in this folder.",
            kind="ok",
        )
        if has_dj:
            print_message("dataset.json detected — metadata will be used.", kind="ok")
        print_message("Preprocessing will run automatically before training.\n", kind="dim")

    # Run name → output dir
    # The timestamp suffix is auto-appended and non-editable (safety feature).
    default_base = a.get("_run_name_base") or _generate_run_base(a)
    a["_run_name_base"] = ask(
        "Run name base (timestamp will be appended automatically)",
        default=default_base,
        required=True,
    )
    a["run_name"] = _stamp_run_name(a["_run_name_base"])

    adapter_type = a.get("adapter_type", "lora")
    adapters_dir = get_trained_adapters_dir()
    auto_output = str(Path(adapters_dir) / adapter_type / a["run_name"])
    print_message(f"Output directory: {auto_output}", kind="dim")

    if a.get("config_mode") == "advanced":
        a["output_dir"] = ask_output_path(
            "Output directory for adapter weights",
            default=a.get("output_dir") or auto_output,
            required=True,
        )
    else:
        a["output_dir"] = auto_output
