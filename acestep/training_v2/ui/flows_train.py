"""
Wizard flow for training.

Training mode is auto-detected from the model variant (turbo vs base/sft).
Uses a step-list pattern for go-back navigation.  Step functions are
defined in ``flows_train_steps`` to keep this module under the LOC cap.

Supports both LoRA (PEFT) and LoKR (LyCORIS) adapters.
"""

from __future__ import annotations

import argparse
from typing import Callable

from acestep.training_v2.ui import console, is_rich_active
from acestep.training_v2.ui.prompt_helpers import GoBack, _esc, menu, step_indicator
from acestep.training_v2.ui.flows_common import build_train_namespace
from acestep.training_v2.ui.flows_train_steps import (
    step_config_mode,
    step_required,
    step_lora,
    step_lokr,
    step_training,
    step_cfg,
    step_logging,
    step_run_name,
    step_chunk_duration,
    step_advanced_device,
    step_advanced_optimizer,
    step_advanced_vram,
    step_advanced_training,
    step_advanced_dataloader,
    step_advanced_logging,
)


# ---- Step list builder ------------------------------------------------------

def _is_turbo_variant(answers: dict) -> bool:
    """Return ``True`` if the selected model is turbo or turbo-based.

    Detection prefers model/base name.  For unknown custom names, falls
    back to ``num_inference_steps`` (8 => turbo-style schedule).
    """
    base = answers.get("base_model", answers.get("model_variant", "turbo"))
    label_lower = base.lower() if isinstance(base, str) else ""
    if "turbo" in label_lower:
        return True
    if "base" in label_lower or "sft" in label_lower:
        return False
    infer_steps = answers.get("num_inference_steps", 8)
    try:
        return int(infer_steps) == 8
    except (TypeError, ValueError):
        # Unknown/no metadata: preserve historical turbo default.
        return True


def _build_steps(answers: dict, config_mode: str, adapter_type: str = "lora") -> list[tuple[str, callable]]:
    """Return the ordered list of ``(label, step_fn)`` for this wizard run."""
    adapter_step = step_lokr if adapter_type == "lokr" else step_lora
    adapter_label = "LoKR Settings" if adapter_type == "lokr" else "LoRA Settings"

    steps = [
        ("Required Settings", step_required),
        (adapter_label, adapter_step),
        ("Training Settings", step_training),
    ]

    # CFG dropout settings only apply to base/sft (turbo doesn't use CFG)
    if not _is_turbo_variant(answers):
        steps.append(("CFG Dropout Settings", step_cfg))

    steps.append(("Logging & Checkpoints", step_logging))
    steps.append(("Run Name", step_run_name))
    steps.append(("Latent Chunking", step_chunk_duration))

    if config_mode == "advanced":
        steps.extend([
            ("Device & Precision", step_advanced_device),
            ("Optimizer & Scheduler", step_advanced_optimizer),
            ("VRAM Savings", step_advanced_vram),
            ("Advanced Training", step_advanced_training),
            ("Data Loading", step_advanced_dataloader),
            ("Advanced Logging", step_advanced_logging),
        ])

    return steps


# ---- Preset helpers ---------------------------------------------------------

def _offer_load_preset(answers: dict) -> None:
    """Ask user if they want to load a preset; merge values into answers."""
    from acestep.training_v2.ui.presets import list_presets, load_preset

    presets = list_presets()
    if not presets:
        return

    options: list[tuple[str, str]] = [("fresh", "Start fresh (defaults)")]
    for p in presets:
        tag = " (built-in)" if p["builtin"] else ""
        desc = f" -- {p['description']}" if p["description"] else ""
        options.append((p["name"], f"{p['name']}{tag}{desc}"))

    choice = menu("Load a preset?", options, default=1, allow_back=True)

    if choice != "fresh":
        data = load_preset(choice)
        if data:
            answers.update(data)
            if is_rich_active() and console is not None:
                console.print(f"  [green]Loaded preset '{choice}'.[/]\n")
            else:
                print(f"  Loaded preset '{choice}'.\n")


def _offer_save_preset(answers: dict) -> None:
    """After wizard completes, offer to save settings as a preset.

    Errors from file I/O or name validation are caught and displayed
    so the user gets feedback rather than a silent failure.
    """
    from acestep.training_v2.ui.presets import save_preset
    from acestep.training_v2.ui.prompt_helpers import ask_bool, ask as _ask, section

    try:
        section("Save Preset")
        if not ask_bool("Save these settings as a reusable preset?", default=True):
            return
        name = _ask("Preset name", required=True)
        desc = _ask("Short description", default="")
        path = save_preset(name, desc, answers)

        # Verify the file was actually written
        if path.is_file():
            size = path.stat().st_size
            if is_rich_active() and console is not None:
                console.print(
                    f"  [green]Preset '{_esc(name)}' saved ({size} bytes)[/]\n"
                    f"  [dim]Location: {_esc(path)}[/]\n"
                )
            else:
                print(f"  Preset '{name}' saved ({size} bytes)")
                print(f"  Location: {path}\n")
        else:
            if is_rich_active() and console is not None:
                console.print(f"  [red]Warning: preset file not found after save: {_esc(path)}[/]\n")
            else:
                print(f"  Warning: preset file not found after save: {path}\n")
    except (KeyboardInterrupt, EOFError):
        pass
    except Exception as exc:
        # Catch ValueError (bad name), OSError/PermissionError, etc.
        if is_rich_active() and console is not None:
            console.print(f"  [red]Failed to save preset: {_esc(exc)}[/]\n")
        else:
            print(f"  Failed to save preset: {exc}\n")


# ---- Public entry point -----------------------------------------------------

def _print_training_strategy(answers: dict) -> None:
    """Show the auto-detected training strategy after model selection."""
    if _is_turbo_variant(answers):
        msg = "Turbo detected -- using discrete 8-step sampling (no CFG)"
    else:
        msg = "Base/SFT detected -- using continuous sampling + CFG dropout"

    if is_rich_active() and console is not None:
        console.print(f"\n  [bold cyan]{msg}[/]\n")
    else:
        print(f"\n  {msg}\n")


def _check_fisher_map(answers: dict, adapter_type: str) -> None:
    """Inform the user about Preprocessing++ map status in dataset directory.

    When a fisher_map.json exists, prints a notice.  When absent and the
    adapter is LoRA, prints a non-blocking recommendation to run
    Preprocessing++ before training.
    """
    from pathlib import Path

    answers["_fisher_map_detected"] = False
    answers["_pp_recommended"] = False
    answers["_pp_sample_count"] = 0

    dataset_dir = answers.get("dataset_dir")
    if not dataset_dir:
        return

    fisher_path = Path(dataset_dir) / "fisher_map.json"
    if fisher_path.is_file():
        try:
            import json
            data = json.loads(fisher_path.read_text(encoding="utf-8"))
            n = len(data.get("rank_pattern", {}))
            budget = data.get("rank_budget", {})
            msg = (
                f"Preprocessing++ map detected: {n} modules, "
                f"adaptive ranks {budget.get('min', '?')}-{budget.get('max', '?')}.\n"
                "  Training will use Preprocessing++ targeting."
            )
        except Exception:
            msg = "Preprocessing++ map detected (could not read details)."

        if is_rich_active() and console is not None:
            console.print(f"\n  [bold green]{msg}[/]\n")
        else:
            print(f"\n  {msg}\n")
        answers["_fisher_map_detected"] = True
        return

    if adapter_type != "lora":
        return

    # Non-blocking recommendation (no inline long-running analysis).
    pt_count = len(list(Path(dataset_dir).glob("*.pt")))
    if pt_count == 0:
        return
    if is_rich_active() and console is not None:
        console.print(
            f"\n  [yellow]No Preprocessing++ map found for this dataset ({pt_count} samples).[/]\n"
            "  [dim]Training will use flat-rank settings. For stronger dataset-specific\n"
            "  targeting, run Preprocessing++ from the main menu before training.[/]\n"
        )
    else:
        print(
            f"\n  No Preprocessing++ map found for this dataset ({pt_count} samples).\n"
            "  Training will use flat-rank settings. For stronger dataset-specific\n"
            "  targeting, run Preprocessing++ from the main menu before training.\n"
        )
    answers["_pp_recommended"] = True
    answers["_pp_sample_count"] = pt_count


def _index_for_step(
    steps: list[tuple[str, Callable[..., None]]],
    target_fn: Callable[..., None],
    fallback: int = 0,
) -> int:
    """Return the index for a step function in the current step list."""
    for idx, (_, fn) in enumerate(steps):
        if fn is target_fn:
            return idx
    return fallback


def _review_and_confirm(
    answers: dict, config_mode: str, steps: list[tuple[str, Callable[..., None]]]
) -> int | None:
    """Show a concise review and return the section index to edit, if any."""
    adapter_type = answers.get("adapter_type", "lora")
    adapter_label = "LoKR" if adapter_type == "lokr" else "LoRA"
    projection_label = (
        answers.get("target_modules_str")
        or f"self={answers.get('self_target_modules_str', '')} | cross={answers.get('cross_target_modules_str', '')}"
    )

    if is_rich_active() and console is not None:
        console.print("\n  [bold cyan]Review Training Configuration[/]")
        console.print(
            f"  [dim]Adapter:[/] {adapter_label}    "
            f"[dim]Mode:[/] {config_mode}    "
            f"[dim]Model:[/] {_esc(answers.get('model_variant', 'turbo'))}"
        )
        console.print(
            f"  [dim]Dataset:[/] {_esc(answers.get('dataset_dir', ''))}\n"
            f"  [dim]Output:[/] {_esc(answers.get('output_dir', ''))}"
        )
        console.print(
            f"  [dim]Train:[/] lr={answers.get('learning_rate', 1e-4)}  "
            f"batch={answers.get('batch_size', 1)}  "
            f"accum={answers.get('gradient_accumulation', 4)}  "
            f"epochs={answers.get('epochs', 100)}"
        )
        if adapter_type == "lokr":
            console.print(
                f"  [dim]LoKR:[/] dim={answers.get('lokr_linear_dim', 64)}  "
                f"alpha={answers.get('lokr_linear_alpha', 128)}"
            )
        elif answers.get("_fisher_map_detected"):
            console.print(
                "  [dim]LoRA:[/] adaptive Preprocessing++ map detected "
                "(rank/targets locked by map)"
            )
        else:
            console.print(
                f"  [dim]LoRA:[/] rank={answers.get('rank', 64)}  "
                f"alpha={answers.get('alpha', 128)}  "
                f"dropout={answers.get('dropout', 0.1)}"
            )
        if answers.get("_pp_recommended"):
            console.print(
                "  [yellow]Note:[/] No Preprocessing++ map found; this run uses flat-rank targeting."
            )
        if projection_label:
            console.print(f"  [dim]Targets:[/] {_esc(projection_label)}")
    else:
        print("\n  Review Training Configuration")
        print(
            f"  Adapter: {adapter_label}    "
            f"Mode: {config_mode}    "
            f"Model: {answers.get('model_variant', 'turbo')}"
        )
        print(
            f"  Dataset: {answers.get('dataset_dir', '')}\n"
            f"  Output: {answers.get('output_dir', '')}"
        )
        print(
            f"  Train: lr={answers.get('learning_rate', 1e-4)}  "
            f"batch={answers.get('batch_size', 1)}  "
            f"accum={answers.get('gradient_accumulation', 4)}  "
            f"epochs={answers.get('epochs', 100)}"
        )
        if adapter_type == "lokr":
            print(
                f"  LoKR: dim={answers.get('lokr_linear_dim', 64)}  "
                f"alpha={answers.get('lokr_linear_alpha', 128)}"
            )
        elif answers.get("_fisher_map_detected"):
            print("  LoRA: adaptive Preprocessing++ map detected (rank/targets locked by map)")
        else:
            print(
                f"  LoRA: rank={answers.get('rank', 64)}  "
                f"alpha={answers.get('alpha', 128)}  "
                f"dropout={answers.get('dropout', 0.1)}"
            )
        if answers.get("_pp_recommended"):
            print("  Note: No Preprocessing++ map found; this run uses flat-rank targeting.")
        if projection_label:
            print(f"  Targets: {projection_label}")

    options: list[tuple[str, str]] = [
        ("start", "Start training"),
        ("edit_required", "Edit required settings"),
        ("edit_adapter", f"Edit {adapter_label} settings"),
        ("edit_training", "Edit training settings"),
        ("edit_logging", "Edit logging & checkpoints"),
        ("edit_chunking", "Edit latent chunking"),
        ("cancel", "Cancel and return to main menu"),
    ]
    if config_mode == "advanced":
        options.insert(6, ("edit_advanced", "Edit advanced settings"))

    choice = menu("Review complete. What would you like to do?", options, default=1)
    if choice == "start":
        return None
    if choice == "cancel":
        raise GoBack()
    if choice == "edit_required":
        return _index_for_step(steps, step_required, fallback=0)
    if choice == "edit_adapter":
        return _index_for_step(steps, step_lokr if adapter_type == "lokr" else step_lora, fallback=1)
    if choice == "edit_training":
        return _index_for_step(steps, step_training, fallback=2)
    if choice == "edit_logging":
        return _index_for_step(steps, step_logging, fallback=3)
    if choice == "edit_chunking":
        return _index_for_step(steps, step_chunk_duration, fallback=max(0, len(steps) - 1))
    if choice == "edit_advanced":
        return _index_for_step(steps, step_advanced_device, fallback=max(0, len(steps) - 1))
    return 0


def wizard_train(
    mode: str = "fixed",
    adapter_type: str = "lora",
    preset: dict | None = None,
) -> argparse.Namespace:
    """Interactive wizard for training.

    Training mode is always ``'fixed'``; turbo vs base/sft is
    auto-detected from the selected model variant.

    Args:
        mode: Training subcommand (always ``'fixed'``).
        adapter_type: Adapter type ('lora' or 'lokr').
        preset: Optional dict of pre-filled answer values (e.g. dataset_dir
            from the chain flow after preprocessing).  These values are
            used as defaults but do NOT suppress preset loading.

    Returns:
        A populated ``argparse.Namespace`` ready for dispatch.

    Raises:
        GoBack: If the user backs out of the very first step.
    """
    # Pre-fill any values the caller provided (e.g. dataset_dir from
    # the preprocess chain flow).  These are saved and restored after
    # preset loading so they always take priority.
    prefill: dict = dict(preset) if preset else {}
    answers: dict = dict(prefill)
    answers["adapter_type"] = adapter_type

    # Always offer to load a preset.  Pre-filled values (like
    # dataset_dir) are not in PRESET_FIELDS, so they survive the
    # update.  adapter_type is guarded below regardless.
    try:
        _offer_load_preset(answers)
    except GoBack:
        raise

    # Guard: adapter_type always comes from the menu selection, never
    # from a preset.  Pre-filled values also take priority over preset
    # values in case of overlap.
    answers["adapter_type"] = adapter_type
    answers.update(prefill)

    # Step 0: config depth
    try:
        step_config_mode(answers)
    except GoBack:
        raise

    config_mode = answers["config_mode"]
    steps = _build_steps(answers, config_mode, adapter_type)
    total = len(steps)
    i = 0

    while True:
        while i < total:
            label, step_fn = steps[i]
            try:
                step_indicator(i + 1, total, label)
                step_fn(answers)
                i += 1

                # After model selection (step_required), show auto-detected
                # strategy and rebuild steps in case turbo/base changed.
                if step_fn is step_required:
                    _print_training_strategy(answers)
                    _check_fisher_map(answers, adapter_type)
                    steps = _build_steps(answers, config_mode, adapter_type)
                    total = len(steps)
            except GoBack:
                if i == 0:
                    try:
                        step_config_mode(answers)
                    except GoBack:
                        raise
                    config_mode = answers["config_mode"]
                    steps = _build_steps(answers, config_mode, adapter_type)
                    total = len(steps)
                    i = 0
                else:
                    i -= 1

        jump_to = _review_and_confirm(answers, config_mode, steps)
        if jump_to is None:
            break
        i = jump_to

    _offer_save_preset(answers)

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
