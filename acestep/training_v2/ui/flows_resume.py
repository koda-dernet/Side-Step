"""
Wizard flow for resuming a previous training run.

Auto-discovers checkpoints, reloads original configs, and offers
guarded editing of select parameters before dispatching training.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Optional

from acestep.training_v2.ui import console, is_rich_active
from acestep.training_v2.ui.prompt_helpers import (
    GoBack,
    _esc,
    ask,
    ask_bool,
    ask_path,
    menu,
    print_message,
    section,
)
from acestep.training_v2.ui.flows_common import build_train_namespace

# ---------------------------------------------------------------------------
# Field editing tiers for resume
# ---------------------------------------------------------------------------

_SAFE_FIELDS: dict[str, str] = {
    "epochs": "Max epochs",
    "learning_rate": "Learning rate",
    "save_every": "Save checkpoint every N epochs",
    "log_every": "Log metrics every N steps",
    "log_heavy_every": "Log gradient norms every N steps (0=disabled)",
    "save_best": "Auto-save best model",
    "save_best_after": "Start best-model tracking after epoch",
    "early_stop_patience": "Early stop patience (0=disabled)",
    "run_name": "Run name",
}

_DANGEROUS_FIELDS: dict[str, str] = {
    "batch_size": "Batch size",
    "gradient_accumulation": "Gradient accumulation steps",
    "weight_decay": "Weight decay",
    "max_grad_norm": "Max gradient norm",
    "warmup_steps": "Warmup steps",
    "scheduler_type": "LR scheduler",
    "optimizer_type": "Optimizer",
    "chunk_duration": "Latent chunk duration",
}

_LOCKED_FIELDS: set[str] = {
    "adapter_type", "model_variant", "checkpoint_dir", "dataset_dir",
    "rank", "alpha", "dropout", "target_modules", "attention_type",
    "target_mlp", "bias",
    "lokr_linear_dim", "lokr_linear_alpha", "lokr_factor",
    "lokr_decompose_both", "lokr_use_tucker", "lokr_use_scalar",
    "lokr_weight_decompose",
    "shift", "num_inference_steps", "seed",
}


def _answers_from_saved_training_config(saved_config: dict[str, Any]) -> dict[str, Any]:
    """Convert persisted TrainingConfigV2 JSON into wizard answer keys.

    The training wizard namespace builder uses wizard-style keys
    (``epochs``, ``gradient_accumulation``, ``save_every``), while saved
    training config uses dataclass field names
    (``max_epochs``, ``gradient_accumulation_steps``, ``save_every_n_epochs``).
    This mapper normalizes those aliases so resume keeps original values.
    """
    a: dict[str, Any] = dict(saved_config)

    # Key aliases: persisted config -> wizard answer shape
    if "max_epochs" in a and "epochs" not in a:
        a["epochs"] = a["max_epochs"]
    if "gradient_accumulation_steps" in a and "gradient_accumulation" not in a:
        a["gradient_accumulation"] = a["gradient_accumulation_steps"]
    if "save_every_n_epochs" in a and "save_every" not in a:
        a["save_every"] = a["save_every_n_epochs"]

    # Resume should continue in the same TB directory when possible.
    if not a.get("log_dir") and a.get("run_name") and a.get("output_dir"):
        from acestep.training_v2.tensorboard_utils import resolve_latest_versioned_log_dir

        latest = resolve_latest_versioned_log_dir(
            Path(a["output_dir"]) / "runs",
            str(a["run_name"]),
        )
        if latest is not None:
            a["log_dir"] = str(latest)

    return a


# ---------------------------------------------------------------------------
# Checkpoint discovery
# ---------------------------------------------------------------------------

def _find_resumable_runs(search_root: str | Path) -> list[dict[str, Any]]:
    """Scan *search_root* for directories containing training artifacts.

    Returns a list of dicts sorted newest-first, each with keys:
        - path: str -- absolute path to the checkpoint dir
        - epoch: int | None
        - global_step: int | None
        - has_config: bool
        - label: str -- human-friendly description
    """
    root = Path(search_root).expanduser().resolve()
    if not root.is_dir():
        return []

    results: list[dict[str, Any]] = []

    def _probe(d: Path) -> Optional[dict[str, Any]]:
        has_ts = (d / "training_state.pt").exists()
        has_adapter = (d / "adapter_model.safetensors").exists() or (d / "lokr_weights.safetensors").exists()
        if not (has_ts or has_adapter):
            return None

        epoch = None
        step = None
        has_config = (d / "training_config.json").exists()

        if has_ts:
            try:
                import torch
                state = torch.load(str(d / "training_state.pt"), map_location="cpu", weights_only=False)
                epoch = state.get("epoch")
                step = state.get("global_step")
            except Exception:
                pass

        parts: list[str] = [d.name]
        if epoch is not None:
            parts.append(f"epoch {epoch}")
        if step is not None:
            parts.append(f"step {step}")
        if has_config:
            parts.append("config available")

        return {
            "path": str(d),
            "epoch": epoch,
            "global_step": step,
            "has_config": has_config,
            "label": " | ".join(parts),
            "_mtime": (d / "training_state.pt").stat().st_mtime if has_ts else 0,
        }

    for child in sorted(root.iterdir(), key=lambda p: p.name):
        if not child.is_dir():
            continue
        # Direct checkpoint dir (e.g. output_dir/checkpoints/epoch_50)
        hit = _probe(child)
        if hit:
            results.append(hit)
        else:
            # Nested checkpoints dir (e.g. output_dir/checkpoints/epoch_*)
            for grandchild in sorted(child.iterdir(), key=lambda p: p.name):
                if grandchild.is_dir():
                    hit = _probe(grandchild)
                    if hit:
                        results.append(hit)

    results.sort(key=lambda r: r.get("_mtime", 0), reverse=True)
    return results


# ---------------------------------------------------------------------------
# Config reloading
# ---------------------------------------------------------------------------

def _load_saved_config(ckpt_path: str | Path) -> Optional[dict[str, Any]]:
    """Try to load training_config.json from the checkpoint."""
    cfg_path = Path(ckpt_path) / "training_config.json"
    if not cfg_path.is_file():
        return None
    try:
        return json.loads(cfg_path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _load_saved_adapter_config(ckpt_path: str | Path) -> Optional[dict[str, Any]]:
    """Try to load sidestep_adapter_config.json from the checkpoint."""
    cfg_path = Path(ckpt_path) / "sidestep_adapter_config.json"
    if not cfg_path.is_file():
        return None
    try:
        return json.loads(cfg_path.read_text(encoding="utf-8"))
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Guarded field editing
# ---------------------------------------------------------------------------

def _edit_safe_fields(answers: dict) -> None:
    """Let the user edit freely-changeable fields."""
    section("Safe settings (freely editable)")
    for key, label in _SAFE_FIELDS.items():
        current = answers.get(key)
        if key == "save_best":
            answers[key] = ask_bool(label, default=bool(current) if current is not None else True, allow_back=True)
        elif key == "run_name":
            raw = ask(label, default=current, allow_back=True)
            answers[key] = raw if raw not in (None, "None", "") else None
        elif isinstance(current, float):
            answers[key] = ask(label, default=current, type_fn=float, allow_back=True)
        elif isinstance(current, int):
            answers[key] = ask(label, default=current, type_fn=int, allow_back=True)
        else:
            answers[key] = ask(label, default=current, allow_back=True)


def _edit_dangerous_fields(answers: dict) -> None:
    """Let the user edit risky fields with warnings."""
    section("Dangerous settings (may affect training stability)")
    print_message(
        "WARNING: Changing these settings mid-training can cause instability, "
        "loss spikes, or divergence. Only modify if you know what you're doing.",
        style="yellow",
    )
    for key, label in _DANGEROUS_FIELDS.items():
        current = answers.get(key)
        display = f"{label} (current: {current})"
        if not ask_bool(f"Change {label}?", default=False, allow_back=True):
            continue
        if isinstance(current, float):
            answers[key] = ask(display, default=current, type_fn=float, allow_back=True)
        elif isinstance(current, int):
            answers[key] = ask(display, default=current, type_fn=int, allow_back=True)
        else:
            answers[key] = ask(display, default=current, allow_back=True)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def wizard_resume(
    prefill: dict | None = None,
) -> argparse.Namespace:
    """Interactive wizard for resuming a previous training run.

    1. Asks for an output directory to scan.
    2. Discovers available checkpoints.
    3. Reloads the original training configuration.
    4. Optionally lets the user edit select settings.
    5. Returns a ready-to-dispatch Namespace.

    Raises:
        GoBack: If the user backs out.
    """
    section("Resume Training")
    defaults = dict(prefill) if prefill else {}

    # Step 1: find a checkpoint
    scan_dir = ask_path(
        "Output directory to scan for checkpoints",
        default=defaults.get("output_dir") or defaults.get("resume_from"),
        must_exist=True,
        allow_back=True,
    )

    runs = _find_resumable_runs(scan_dir)
    if not runs:
        print_message(
            f"No resumable checkpoints found in: {scan_dir}",
            style="yellow",
        )
        print_message(
            "Make sure the path contains checkpoint directories with "
            "training_state.pt or adapter weights.",
            style="dim",
        )
        raise GoBack()

    # Step 2: pick a checkpoint
    options: list[tuple[str, str]] = []
    for i, run in enumerate(runs):
        options.append((str(i), run["label"]))
    options.append(("manual", "Enter checkpoint path manually"))

    choice = menu("Select checkpoint to resume from:", options, default=1, allow_back=True)

    if choice == "manual":
        ckpt_path = ask_path(
            "Checkpoint directory path",
            must_exist=True,
            allow_back=True,
        )
    else:
        ckpt_path = runs[int(choice)]["path"]

    if is_rich_active() and console is not None:
        console.print(f"\n  [bold]Selected:[/] {_esc(ckpt_path)}\n")
    else:
        print(f"\n  Selected: {ckpt_path}\n")

    # Step 3: reload original config
    saved_config = _load_saved_config(ckpt_path)
    saved_adapter = _load_saved_adapter_config(ckpt_path)

    if saved_config:
        print_message("Original training configuration loaded from checkpoint.", style="green")
        answers: dict[str, Any] = _answers_from_saved_training_config(saved_config)
    else:
        print_message(
            "No saved training_config.json found in this checkpoint. "
            "This checkpoint was created before config persistence was added.\n"
            "You will need to re-enter your settings through the regular training wizard.",
            style="yellow",
        )
        raise GoBack()

    if saved_adapter:
        adapter_type = saved_adapter.get("adapter_type")
        if adapter_type is None:
            adapter_type = "lokr" if "linear_dim" in saved_adapter else "lora"
        answers["adapter_type"] = adapter_type

        if adapter_type == "lora":
            answers["rank"] = saved_adapter.get("r", saved_adapter.get("rank", 64))
            answers["alpha"] = saved_adapter.get("lora_alpha", saved_adapter.get("alpha", 128))
            answers["dropout"] = saved_adapter.get("lora_dropout", saved_adapter.get("dropout", 0.1))
            answers["target_modules"] = saved_adapter.get("target_modules", ["q_proj", "k_proj", "v_proj", "o_proj"])
            answers["bias"] = saved_adapter.get("bias", "none")
            answers["attention_type"] = saved_adapter.get("attention_type", "both")
            answers["target_mlp"] = saved_adapter.get("target_mlp", False)
        else:
            for k in ("linear_dim", "linear_alpha", "factor", "decompose_both",
                       "use_tucker", "use_scalar", "weight_decompose"):
                if k in saved_adapter:
                    answers[f"lokr_{k}"] = saved_adapter[k]
            answers["attention_type"] = saved_adapter.get("attention_type", "both")
            answers["target_mlp"] = saved_adapter.get("target_mlp", False)

    # Wire resume path and skip warmup
    answers["resume_from"] = ckpt_path
    answers["strict_resume"] = True
    answers["warmup_steps"] = 0

    # Step 4: optionally edit settings
    edit_choice = menu(
        "Would you like to modify any settings before resuming?",
        [
            ("no", "Resume with original settings (recommended)"),
            ("safe", "Edit safe settings only (epochs, LR, logging)"),
            ("all", "Edit safe + dangerous settings (use with caution)"),
        ],
        default=1,
        allow_back=True,
    )

    if edit_choice in ("safe", "all"):
        print_message(
            "Note: Locked settings (adapter type, model, rank, dataset) "
            "cannot be changed during resume.",
            style="dim",
        )
        try:
            _edit_safe_fields(answers)
        except GoBack:
            pass

    if edit_choice == "all":
        try:
            _edit_dangerous_fields(answers)
        except GoBack:
            pass

    # Step 5: review
    adapter_type = answers.get("adapter_type", "lora")
    adapter_label = "LoKR" if adapter_type == "lokr" else "LoRA"

    if is_rich_active() and console is not None:
        console.print("\n  [bold cyan]Resume Configuration[/]")
        console.print(f"  [dim]Checkpoint:[/] {_esc(ckpt_path)}")
        console.print(f"  [dim]Adapter:[/] {adapter_label}")
        console.print(f"  [dim]Model:[/] {_esc(answers.get('model_variant', '?'))}")
        console.print(
            f"  [dim]LR:[/] {answers.get('learning_rate', '?')}  "
            f"[dim]Epochs:[/] {answers.get('epochs', '?')}  "
            f"[dim]Warmup:[/] {answers.get('warmup_steps', 0)} (skipped)"
        )
        console.print("")
    else:
        print("\n  Resume Configuration")
        print(f"  Checkpoint: {ckpt_path}")
        print(f"  Adapter: {adapter_label}")
        print(f"  Model: {answers.get('model_variant', '?')}")
        print(
            f"  LR: {answers.get('learning_rate', '?')}  "
            f"Epochs: {answers.get('epochs', '?')}  "
            f"Warmup: {answers.get('warmup_steps', 0)} (skipped)"
        )
        print()

    if not ask_bool("Proceed with resume?", default=True, allow_back=True):
        raise GoBack()

    # Build namespace
    ns = build_train_namespace(answers)

    from acestep.training_v2.ui.dependency_check import (
        ensure_optional_dependencies,
        required_training_optionals,
    )
    ensure_optional_dependencies(
        required_training_optionals(ns),
        interactive=True,
        allow_install_prompt=True,
    )
    return ns
