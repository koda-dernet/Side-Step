"""
Wizard flow for training.

Training mode is auto-detected from the model variant (turbo vs base/sft).
Uses a step-list pattern for go-back navigation.  Step functions are
defined in ``flows_train_steps`` to keep this module under the LOC cap.

Supports both LoRA (PEFT) and LoKR (LyCORIS) adapters.
"""

from __future__ import annotations

import argparse
import logging
from typing import Callable

logger = logging.getLogger(__name__)

from sidestep_engine.ui.prompt_helpers import (
    GoBack,
    _esc,
    menu,
    print_message,
    print_rich,
    step_indicator,
)
from sidestep_engine.ui.flows.common import build_train_namespace, is_turbo
from sidestep_engine.ui.flows.train_steps import (
    step_config_mode,
    step_required,
    step_lora,
    step_dora,
    step_lokr,
    step_loha,
    step_oft,
    ADAPTER_STEP_MAP,
    ADAPTER_LABEL_MAP,
    step_training,
    step_cfg,
    step_logging,
    step_chunk_duration,
    step_advanced_device,
    step_advanced_optimizer,
    step_advanced_vram,
    step_advanced_training,
    step_advanced_dataloader,
    step_advanced_logging,
    step_all_the_levers,
)
from sidestep_engine.ui.flows.common import is_pp_compatible




def _build_steps(answers: dict, config_mode: str, adapter_type: str = "lora") -> list[tuple[str, callable]]:
    """Return the ordered list of ``(label, step_fn)`` for this wizard run."""
    adapter_step = ADAPTER_STEP_MAP.get(adapter_type, step_lora)
    adapter_label = ADAPTER_LABEL_MAP.get(adapter_type, "LoRA Settings")

    steps = [
        ("Required Settings", step_required),
        (adapter_label, adapter_step),
        ("Training Settings", step_training),
    ]

    # CFG dropout settings only apply to base/sft (turbo doesn't use CFG)
    if not is_turbo(answers):
        steps.append(("CFG Dropout Settings", step_cfg))

    steps.append(("Logging & Checkpoints", step_logging))
    steps.append(("Latent Chunking", step_chunk_duration))

    if config_mode == "advanced":
        steps.extend([
            ("Device & Precision", step_advanced_device),
            ("Optimizer & Scheduler", step_advanced_optimizer),
            ("VRAM Savings", step_advanced_vram),
            ("Advanced Training", step_advanced_training),
            ("Data Loading", step_advanced_dataloader),
            ("Advanced Logging", step_advanced_logging),
            ("All the Levers", step_all_the_levers),
        ])

    return steps


# ---- Preset helpers ---------------------------------------------------------

def _offer_load_preset(answers: dict) -> None:
    """Ask user if they want to load a preset; merge values into answers."""
    from sidestep_engine.ui.presets import list_presets, load_preset

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
            print_message(f"Loaded preset '{choice}'.", kind="ok")


def _offer_save_preset(answers: dict) -> None:
    """After wizard completes, offer to save settings as a preset.

    Errors from file I/O or name validation are caught and displayed
    so the user gets feedback rather than a silent failure.
    """
    from sidestep_engine.ui.presets import save_preset
    from sidestep_engine.ui.prompt_helpers import ask_bool, ask as _ask, section

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
            print_message(f"Preset '{name}' saved ({size} bytes)", kind="ok")
            print_message(f"Location: {path}", kind="dim")
        else:
            print_message(f"Warning: preset file not found after save: {path}", kind="error")
    except (KeyboardInterrupt, EOFError):
        pass
    except Exception as exc:
        # Catch ValueError (bad name), OSError/PermissionError, etc.
        print_message(f"Failed to save preset: {exc}", kind="error")


# ---- Public entry point -----------------------------------------------------

def _print_training_strategy(answers: dict) -> None:
    """Show the auto-detected training strategy after model selection."""
    if is_turbo(answers):
        msg = "Turbo detected -- using discrete 8-step sampling (no CFG)"
    else:
        msg = "Base/SFT detected -- using continuous sampling + CFG dropout"

    print_message(f"\n{msg}", kind="heading")


def _run_pp_inline(answers: dict) -> None:
    """Run Preprocessing++ inline with default rank settings.

    Updates ``answers`` with fisher map status on success.
    Catches and reports errors without aborting the wizard.
    """
    try:
        from sidestep_engine.analysis.fisher import run_fisher_analysis

        def _progress(batch: int, total: int, msg: str) -> None:
            print_message(f"{msg} ({batch}/{total})", kind="dim")

        print_message("\nRunning Preprocessing++ ...", kind="info")
        result = run_fisher_analysis(
            checkpoint_dir=answers["checkpoint_dir"],
            variant=answers["model_variant"],
            dataset_dir=answers["dataset_dir"],
            base_rank=64,
            rank_min=16,
            rank_max=128,
            timestep_focus="balanced",
            progress_callback=_progress,
            auto_confirm=True,
        )
        if result is not None:
            print_message("Preprocessing++ complete — adaptive ranks saved.", kind="ok")
        else:
            print_message("Preprocessing++ did not produce a map.", kind="warn")
    except Exception as exc:
        print_message(f"Preprocessing++ failed: {exc}", kind="error")
        print_message("Continuing with flat-rank settings.", kind="dim")


def _check_fisher_map(answers: dict, adapter_type: str) -> None:
    """Inform the user about Preprocessing++ map status in dataset directory.

    When a fisher_map.json exists and the adapter is PP++-compatible,
    prints a notice.  When the map exists but the adapter is incompatible,
    warns that the map will be ignored.  When absent and a PP++-compatible
    adapter is selected, offers to run PP++ inline.
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
        # Map exists — but only PP++-compatible adapters can use it
        if not is_pp_compatible(adapter_type):
            print_message(
                f"\nPreprocessing++ map found but {adapter_type.upper()} does not support "
                "per-module rank assignment. The map will be ignored.",
                kind="warn",
            )
            print_message(
                "PP++ works by assigning adaptive ranks to each module via PEFT's "
                "rank_pattern. Only LoRA and DoRA support this mechanism.",
                kind="dim",
            )
            return

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
        except Exception as exc:
            logger.debug("Could not parse fisher_map.json: %s", exc)
            msg = "Preprocessing++ map detected (could not read details)."

        print_message(f"\n{msg}", kind="ok")
        answers["_fisher_map_detected"] = True
        return

    # No map — only offer PP++ for compatible adapters
    if not is_pp_compatible(adapter_type):
        return

    pt_count = len(list(Path(dataset_dir).glob("*.pt")))
    if pt_count == 0:
        return
    print_message(
        f"\nNo Preprocessing++ map found for this dataset ({pt_count} samples).",
        kind="warn",
    )

    # Offer to run Preprocessing++ inline
    from sidestep_engine.ui.prompt_helpers import ask_bool
    if ask_bool("Run Preprocessing++ now? (adaptive ranks, ~1-2 min)", default=False):
        _run_pp_inline(answers)
        # Re-check: did it create the map?
        if (Path(dataset_dir) / "fisher_map.json").is_file():
            answers["_fisher_map_detected"] = True
            answers["_pp_recommended"] = False
            return

    print_message(
        "Training will use flat-rank settings.",
        kind="dim",
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
    """Show a full grouped review table and return the section index to edit, if any."""
    from sidestep_engine.ui.flows.review_summary import show_review_table

    show_review_table(answers)

    adapter_type = answers.get("adapter_type", "lora")
    adapter_label = ADAPTER_LABEL_MAP.get(adapter_type, "LoRA Settings").replace(" Settings", "")

    if answers.get("_fisher_map_detected"):
        print_rich(
            "[bold green]Preprocessing++ map detected[/] "
            "(rank/targets locked by map)"
        )
    elif answers.get("_pp_recommended"):
        print_rich(
            "[yellow]Note:[/] No Preprocessing++ map found; "
            "this run uses flat-rank targeting."
        )

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
        target_step = ADAPTER_STEP_MAP.get(adapter_type, step_lora)
        return _index_for_step(steps, target_step, fallback=1)
    if choice == "edit_training":
        return _index_for_step(steps, step_training, fallback=2)
    if choice == "edit_logging":
        return _index_for_step(steps, step_logging, fallback=3)
    if choice == "edit_chunking":
        return _index_for_step(steps, step_chunk_duration, fallback=max(0, len(steps) - 1))
    if choice == "edit_advanced":
        return _index_for_step(steps, step_advanced_device, fallback=max(0, len(steps) - 1))
    return 0


def _run_inline_preprocess(answers: dict) -> None:
    """Run preprocessing inline when the training wizard detects raw audio.

    Delegates to the shared ``run_inline_preprocess`` runner.
    On failure, offers recovery via ``_handle_preprocess_failure``.
    """
    from sidestep_engine.ui.flows.inline_preprocess import run_inline_preprocess

    try:
        run_inline_preprocess(answers, label="Auto-Preprocessing")
    except RuntimeError as exc:
        _handle_preprocess_failure(answers, exc)


def _handle_preprocess_failure(answers: dict, exc: Exception) -> None:
    """Let the user recover from a failed preprocessing run.

    Shows the error and prompts for an alternative preprocessed directory.
    """
    from sidestep_engine.ui.prompt_helpers import ask_path
    from sidestep_engine.ui.flows.common import (
        describe_preprocessed_dataset_issue,
        show_dataset_issue,
    )

    print_message(f"\nPreprocessing failed: {exc}", kind="error")
    print_message("You can still point to a different preprocessed directory.", kind="warn")

    while True:
        answers["dataset_dir"] = ask_path(
            "Dataset directory (preprocessed .pt files)",
            default=answers.get("dataset_dir"),
            must_exist=True,
            allow_back=False,
        )
        issue = describe_preprocessed_dataset_issue(answers["dataset_dir"])
        if issue is None:
            break
        show_dataset_issue(issue)


def wizard_train(
    mode: str = "train",
    adapter_type: str = "lora",
    preset: dict | None = None,
) -> argparse.Namespace:
    """Interactive wizard for training.

    Training mode is always ``'train'``; turbo vs base/sft is
    auto-detected from the selected model variant.

    Args:
        mode: Training subcommand (always ``'train'``).
        adapter_type: Adapter type ('lora', 'dora', 'lokr', 'loha', or 'oft').
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
                    # Auto-preprocess if user accepted the offer
                    if answers.pop("_auto_preprocess_audio_dir", None):
                        _run_inline_preprocess(answers)

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

    # Save reproducible CLI command to output dir
    from sidestep_engine.ui.flows.review_summary import save_cli_command
    save_cli_command(answers)

    ns = build_train_namespace(answers)
    from sidestep_engine.ui.dependency_check import (
        ensure_optional_dependencies,
        required_training_optionals,
    )
    ensure_optional_dependencies(
        required_training_optionals(ns),
        interactive=True,
        allow_install_prompt=True,
    )
    return ns
