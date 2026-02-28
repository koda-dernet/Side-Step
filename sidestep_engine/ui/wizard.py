"""
Interactive wizard for Side-Step.

Launched when ``sidestep`` is run with no subcommand.  Provides a
session loop so the user can preprocess, train, manage presets, and access
experimental features without restarting.

Submenus are in ``wizard_menus.py``; flow builders are in ``flows*.py``.
"""

from __future__ import annotations

import argparse
from typing import Any, Generator, Optional

from sidestep_engine.ui.prompt_helpers import GoBack, _esc, menu, print_message, print_rich
from sidestep_engine.ui.flows import (
    wizard_train,
    wizard_preprocess,
    wizard_preprocessing_pp,
)
from sidestep_engine.ui.flows.build_dataset import wizard_build_dataset
from sidestep_engine.ui.flows.resume import wizard_resume
from sidestep_engine.ui.wizard_menus import manage_presets_menu


# ---- First-run check -------------------------------------------------------

def _ensure_first_run_done() -> None:
    """Run the first-time setup wizard if settings don't exist yet."""
    from sidestep_engine.settings import is_first_run, save_settings
    from sidestep_engine.ui.flows.setup import run_first_setup

    if not is_first_run():
        return

    try:
        data = run_first_setup()
        save_settings(data)
    except (KeyboardInterrupt, EOFError):
        print_message("\nSetup skipped. You can run it later from Settings.", kind="dim")


# ---- Session loop -----------------------------------------------------------

_SESSION_KEYS = (
    "checkpoint_dir",
    "model_variant",
    "base_model",
    "dataset_dir",
    "output_dir",
    "resume_from",
    "tensor_output",
    "audio_dir",
    "dataset_json",
    "normalize",
    "rank",
    "rank_min",
    "rank_max",
    "timestep_focus",
)


def _remember(session_defaults: dict[str, Any], key: str, value: Any) -> None:
    """Store value in session defaults when it is meaningful."""
    if value in (None, ""):
        return
    session_defaults[key] = value


_RESUME_ONLY_KEYS = ("resume_from",)


def _update_session_defaults(
    session_defaults: dict[str, Any], ns: argparse.Namespace
) -> None:
    """Capture reusable fields from the latest wizard action."""
    for key in _SESSION_KEYS:
        _remember(session_defaults, key, getattr(ns, key, None))

    # Clear resume_from when the action is NOT a resume — stale resume paths
    # from a previous run should not leak into a fresh training session.
    subcommand = getattr(ns, "subcommand", None)
    if subcommand != "train" or not getattr(ns, "resume_from", None):
        for k in _RESUME_ONLY_KEYS:
            session_defaults.pop(k, None)

    tensor_output = getattr(ns, "tensor_output", None)
    if tensor_output:
        # Preprocess writes tensors to tensor_output, which is the dataset
        # path users most often want to reuse in subsequent training.
        _remember(session_defaults, "dataset_dir", tensor_output)


def _print_chain_context(tensor_output: str, session_defaults: dict[str, Any]) -> None:
    """Explain what carries over when chaining preprocess -> train."""
    ckpt = session_defaults.get("checkpoint_dir")
    model = session_defaults.get("model_variant")
    print_message("\nPreprocessing complete.", kind="heading")
    print_rich(f"  [dim]Carry over:[/] dataset = [bold]{_esc(tensor_output)}[/]")
    if ckpt:
        print_rich(f"  [dim]Carry over:[/] checkpoint = [bold]{_esc(ckpt)}[/]")
    if model:
        print_rich(f"  [dim]Carry over:[/] model = [bold]{_esc(model)}[/]")
    print_message("You can still change all training settings before start.", kind="dim")


def run_wizard_session() -> Generator[argparse.Namespace, None, None]:
    """Launch the interactive wizard as a session loop.

    Yields one ``argparse.Namespace`` per action the user selects.
    The caller (``train.py:main()``) dispatches each, cleans up GPU,
    and the loop shows the menu again.

    After preprocessing, offers to chain directly into training.
    """
    from sidestep_engine.ui.banner import show_banner
    from sidestep_engine.ui.prompt_helpers import ask_bool

    show_banner(subcommand="interactive")

    # First-run setup (skippable)
    _ensure_first_run_done()

    session_defaults: dict[str, Any] = {}

    while True:
        try:
            ns = _main_menu(session_defaults=session_defaults)
        except (KeyboardInterrupt, EOFError):
            _print_abort()
            return

        if ns is None:
            return  # user chose Exit

        is_preprocess = getattr(ns, "preprocess", False)
        tensor_output = getattr(ns, "tensor_output", None)

        yield ns
        _update_session_defaults(session_defaults, ns)

        # PP++ chain: offer training AFTER fisher analysis has completed
        is_pp = getattr(ns, "subcommand", None) == "analyze"
        if is_pp:
            chain_ns = _offer_pp_train_chain(ns, session_defaults)
            if chain_ns is not None:
                yield chain_ns
                _update_session_defaults(session_defaults, chain_ns)
                continue

        # Flow chaining: after preprocess, offer to train on the output
        if is_preprocess and tensor_output:
            try:
                _print_chain_context(tensor_output, session_defaults)
                if ask_bool("Start training now with these tensors?", default=True):
                    try:
                        adapter = _pick_adapter_type()
                        chain_ns = wizard_train(
                            mode="train",
                            adapter_type=adapter,
                            preset={
                                **session_defaults,
                                "dataset_dir": tensor_output,
                            },
                        )
                        yield chain_ns
                        _update_session_defaults(session_defaults, chain_ns)
                    except GoBack:
                        pass
            except (KeyboardInterrupt, EOFError):
                pass


# ---- Main menu --------------------------------------------------------------

def _main_menu(session_defaults: dict[str, Any] | None = None) -> Optional[argparse.Namespace]:
    """Show the main menu and return a Namespace, or None to exit.

    Uses a loop instead of recursion to avoid hitting the stack limit
    when the user navigates back and forth many times.
    """
    prefill = dict(session_defaults or {})

    while True:
        action = menu(
            "What would you like to do?",
            [
                ("train_lora", "Train a LoRA (PEFT) — PP++ compatible"),
                ("train_dora", "Train a DoRA (PEFT) — PP++ compatible"),
                ("train_lokr", "Train a LoKR (LyCORIS)"),
                ("train_loha", "Train a LoHA (LyCORIS)"),
                ("train_oft", "Train an OFT [Experimental] (PEFT)"),
                ("resume", "Resume a previous training run"),
                ("build_dataset", "Build dataset (AI captions / export JSON)"),
                ("preprocess", "Preprocess audio → tensors"),
                ("preprocessing_pp", "Preprocessing++ (adaptive ranks & auto-targeting)"),
                ("presets", "Manage presets"),
                ("history", "View run history"),
                ("settings", "Settings"),
                ("exit", "Exit"),
            ],
            default=1,
        )

        if action == "exit":
            return None

        if action == "presets":
            manage_presets_menu()
            continue  # loop back to main menu

        if action == "settings":
            _run_settings_editor()
            continue  # loop back to main menu

        if action == "history":
            _show_run_history()
            continue  # loop back to main menu

        if action == "build_dataset":
            try:
                result = wizard_build_dataset()
                if result:
                    audio_dir = result.get("audio_dir") or result.get("input_dir") or ""
                    if audio_dir:
                        chain_ns = _offer_build_train_chain(audio_dir, prefill)
                        if chain_ns is not None:
                            return chain_ns
            except GoBack:
                pass
            continue  # loop back to main menu after build

        try:
            if action == "preprocess":
                ns = wizard_preprocess(preset=prefill)
                return ns

            if action == "preprocessing_pp":
                return wizard_preprocessing_pp(preset=prefill)

            if action == "resume":
                return wizard_resume(prefill=prefill)

            if action.startswith("train_"):
                adapter = action.removeprefix("train_")
                return wizard_train(mode="train", adapter_type=adapter, preset=prefill)
        except GoBack:
            continue  # loop back to main menu


# ---- Helpers ----------------------------------------------------------------

def _pick_adapter_type() -> str:
    """Show adapter picker with PP++ labels and return the chosen type."""
    return menu(
        "Which adapter type?",
        [
            ("lora", "LoRA (PEFT) — PP++ compatible"),
            ("dora", "DoRA (PEFT) — PP++ compatible"),
            ("lokr", "LoKR (LyCORIS)"),
            ("loha", "LoHA (LyCORIS)"),
            ("oft", "OFT [Experimental] (PEFT)"),
        ],
        default=1,
    )


def _offer_pp_train_chain(
    ns: argparse.Namespace, session_defaults: dict,
) -> Optional[argparse.Namespace]:
    """After PP++ analysis completes, offer to chain into training.

    Called from the session loop **after** the fisher namespace has been
    dispatched and the analysis has finished, so ``fisher_map.json``
    exists on disk when the training wizard checks for it.

    Returns a training namespace if the user accepts, or ``None``.
    """
    from pathlib import Path
    from sidestep_engine.ui.prompt_helpers import ask_bool

    dataset_dir = getattr(ns, "dataset_dir", None)
    if not dataset_dir:
        return None

    fisher_path = Path(dataset_dir) / "fisher_map.json"
    if not fisher_path.is_file():
        # Analysis failed or was cancelled — nothing to chain from
        return None

    try:
        print_message("\nPreprocessing++ complete.", kind="heading")
        print_rich(f"  [dim]Fisher map saved in:[/] [bold]{_esc(dataset_dir)}[/]")

        if ask_bool("Start training now with PP++ targeting?", default=True):
            adapter = _pick_adapter_type()
            return wizard_train(
                mode="train",
                adapter_type=adapter,
                preset={**session_defaults, "dataset_dir": dataset_dir},
            )
    except (KeyboardInterrupt, EOFError, GoBack):
        pass

    return None


def _offer_build_train_chain(
    audio_dir: str, session_defaults: dict,
) -> Optional[argparse.Namespace]:
    """After dataset build completes, offer to preprocess & train.

    Returns the training namespace if the user accepts chaining,
    or ``None`` if they decline or cancel.
    """
    from sidestep_engine.ui.prompt_helpers import ask_bool, print_message

    try:
        print_message(f"\nDataset built in: {audio_dir}", kind="ok")
        if not ask_bool("Preprocess & train now?", default=True):
            return None

        preset = {**session_defaults, "dataset_dir": audio_dir}
        adapter = _pick_adapter_type()
        return wizard_train(mode="train", adapter_type=adapter, preset=preset)
    except (KeyboardInterrupt, EOFError, GoBack):
        return None


def _run_settings_editor() -> None:
    """Open the settings editor and save any changes."""
    from sidestep_engine.settings import save_settings
    from sidestep_engine.ui.flows.setup import run_settings_editor

    data = run_settings_editor()
    if data is not None:
        save_settings(data)


def _show_run_history() -> None:
    """Display past training runs in a compact table."""
    from sidestep_engine.gui.file_ops import build_history

    runs = build_history()
    if not runs:
        print_message("No training runs found.", kind="dim")
        print_message(
            "Runs are discovered from trained_adapters_dir and "
            "history_output_roots in settings.",
            kind="dim",
        )
        return

    print_message(f"\n  Found {len(runs)} run(s):\n", kind="heading")

    for r in runs[:30]:
        name = r.get("run_name", "?")
        adapter = r.get("adapter", "?")
        epochs = r.get("epochs", 0)
        best = r.get("best_loss")
        status = r.get("status", "?")
        best_str = f"{best:.6f}" if best and isinstance(best, (int, float)) else "--"

        print_rich(
            f"  [bold]{_esc(name)}[/]  "
            f"[dim]{adapter}[/]  "
            f"ep={epochs}  "
            f"best={best_str}  "
            f"[{'green' if status == 'completed' else 'yellow'}]{status}[/]"
        )

    if len(runs) > 30:
        print_message(f"  ... and {len(runs) - 30} more", kind="dim")


def _print_abort() -> None:
    print_message("\nAborted.", kind="dim")
