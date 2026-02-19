"""
Interactive wizard for ACE-Step Training V2.

Launched when ``python train.py`` is run with no subcommand.  Provides a
session loop so the user can preprocess, train, manage presets, and access
experimental features without restarting.

Submenus are in ``wizard_menus.py``; flow builders are in ``flows*.py``.
"""

from __future__ import annotations

import argparse
from typing import Any, Generator, Optional

from acestep.training_v2.ui import console, is_rich_active
from acestep.training_v2.ui.prompt_helpers import GoBack, _esc, menu
from acestep.training_v2.ui.flows import wizard_train, wizard_preprocess
from acestep.training_v2.ui.flows_build_dataset import wizard_build_dataset
from acestep.training_v2.ui.wizard_menus import manage_presets_menu
from acestep.training_v2.ui.flows_fisher import wizard_preprocessing_pp


# ---- First-run check -------------------------------------------------------

def _ensure_first_run_done() -> None:
    """Run the first-time setup wizard if settings don't exist yet."""
    from acestep.training_v2.settings import is_first_run, save_settings
    from acestep.training_v2.ui.flows_setup import run_first_setup

    if not is_first_run():
        return

    try:
        data = run_first_setup()
        save_settings(data)
    except (KeyboardInterrupt, EOFError):
        if is_rich_active() and console is not None:
            console.print("\n  [dim]Setup skipped. You can run it later from Settings.[/]")
        else:
            print("\n  Setup skipped. You can run it later from Settings.")


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


def _update_session_defaults(
    session_defaults: dict[str, Any], ns: argparse.Namespace
) -> None:
    """Capture reusable fields from the latest wizard action."""
    for key in _SESSION_KEYS:
        _remember(session_defaults, key, getattr(ns, key, None))

    tensor_output = getattr(ns, "tensor_output", None)
    if tensor_output:
        # Preprocess writes tensors to tensor_output, which is the dataset
        # path users most often want to reuse in subsequent training.
        _remember(session_defaults, "dataset_dir", tensor_output)


def _print_chain_context(tensor_output: str, session_defaults: dict[str, Any]) -> None:
    """Explain what carries over when chaining preprocess -> train."""
    ckpt = session_defaults.get("checkpoint_dir")
    model = session_defaults.get("model_variant")
    if is_rich_active() and console is not None:
        console.print("\n  [bold cyan]Preprocessing complete.[/]")
        console.print(
            f"  [dim]Carry over:[/] dataset = [bold]{_esc(tensor_output)}[/]"
        )
        if ckpt:
            console.print(f"  [dim]Carry over:[/] checkpoint = [bold]{_esc(ckpt)}[/]")
        if model:
            console.print(f"  [dim]Carry over:[/] model = [bold]{_esc(model)}[/]")
        console.print(
            "  [dim]You can still change all training settings before start.[/]"
        )
    else:
        print("\n  Preprocessing complete.")
        print(f"  Carry over: dataset = {tensor_output}")
        if ckpt:
            print(f"  Carry over: checkpoint = {ckpt}")
        if model:
            print(f"  Carry over: model = {model}")
        print("  You can still change all training settings before start.")


def run_wizard_session() -> Generator[argparse.Namespace, None, None]:
    """Launch the interactive wizard as a session loop.

    Yields one ``argparse.Namespace`` per action the user selects.
    The caller (``train.py:main()``) dispatches each, cleans up GPU,
    and the loop shows the menu again.

    After preprocessing, offers to chain directly into training.
    """
    from acestep.training_v2.ui.banner import show_banner
    from acestep.training_v2.ui.prompt_helpers import ask_bool

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

        # Flow chaining: after preprocess, offer to train on the output
        if is_preprocess and tensor_output:
            try:
                _print_chain_context(tensor_output, session_defaults)
                if ask_bool("Start training now with these tensors?", default=True):
                    try:
                        adapter = menu(
                            "Which adapter type?",
                            [("lora", "LoRA (PEFT)"), ("lokr", "LoKR (LyCORIS)")],
                            default=1,
                        )
                        chain_ns = wizard_train(
                            mode="fixed",
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
                ("train_lora", "Train a LoRA (PEFT)"),
                ("train_lokr", "Train a LoKR (LyCORIS)"),
                ("build_dataset", "Build dataset from folder"),
                ("preprocess", "Preprocess audio into tensors"),
                ("preprocessing_pp", "Preprocessing++ (auto-targeting + adaptive ranks)"),
                ("presets", "Manage presets"),
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

        if action == "build_dataset":
            try:
                wizard_build_dataset()
            except GoBack:
                pass
            continue  # loop back to main menu after build

        try:
            if action == "preprocess":
                return wizard_preprocess(preset=prefill)

            if action == "preprocessing_pp":
                return wizard_preprocessing_pp(preset=prefill)

            if action in ("train_lora", "train_lokr"):
                adapter = "lokr" if action == "train_lokr" else "lora"
                return wizard_train(mode="fixed", adapter_type=adapter, preset=prefill)
        except GoBack:
            continue  # loop back to main menu


# ---- Helpers ----------------------------------------------------------------

def _run_settings_editor() -> None:
    """Open the settings editor and save any changes."""
    from acestep.training_v2.settings import save_settings
    from acestep.training_v2.ui.flows_setup import run_settings_editor

    data = run_settings_editor()
    if data is not None:
        save_settings(data)


def _print_abort() -> None:
    if is_rich_active() and console is not None:
        console.print("\n  [dim]Aborted.[/]")
    else:
        print("\n  Aborted.")
