"""First-run setup wizard and settings editor for Side-Step.

Uses a step-list pattern so pressing 'b' navigates to the previous step.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Callable

from sidestep_engine.ui.prompt_helpers import (
    GoBack,
    _esc,
    ask_bool,
    ask_output_path,
    ask_path,
    native_path,
    print_message,
    print_rich,
    section,
    step_indicator,
)
from sidestep_engine.ui.flows.setup_api_keys import collect_api_keys as _collect_api_keys

logger = logging.getLogger(__name__)


def _smart_checkpoint_default() -> str:
    """Pick a sensible default checkpoint path based on context."""
    for rel in ("./checkpoints", "../ACE-Step-1.5/checkpoints"):
        if Path(rel).is_dir():
            return native_path(rel)
    return native_path("./checkpoints")


# ---- Step functions --------------------------------------------------------

def _step_welcome(data: dict, *, is_edit: bool = False) -> None:
    """Welcome screen and disclaimer (skipped on settings edit)."""
    if is_edit:
        return
    section("Welcome to Side-Step")
    print_rich(
        "  [bold]Important notes:[/]\n"
        "  [yellow]1.[/] Download model weights yourself (e.g. via [bold]acestep-download[/]).\n"
        "  [yellow]2.[/] Fine-tunes [bold]MUST[/] have the original base model too.\n"
        "  [yellow]3.[/] [bold]Never rename checkpoint folders.[/]\n"
        "\n  Side-Step: point to audio → pick model → train. Preprocessing is automatic.\n"
        "  [dim]Already have .pt tensors? Point to those and skip straight to training.[/]\n"
    )


def _step_checkpoint(data: dict) -> None:
    """Ask for checkpoint directory and scan for models."""
    section("Model Checkpoints")
    print_rich("  Where are your model checkpoint folders?")
    print_rich("  [dim](Each variant lives in its own subfolder, e.g. checkpoints/acestep-v15-turbo/)[/]\n")

    default_ckpt = data.get("checkpoint_dir") or _smart_checkpoint_default()
    while True:
        ckpt_dir = ask_path("Checkpoint directory", default=default_ckpt)
        ckpt_path = Path(ckpt_dir)
        if not ckpt_path.is_dir():
            print_rich(f"  [red]Directory not found: {_esc(ckpt_dir)}[/]")
            if not ask_bool("Try a different path?", default=True, allow_back=False):
                raise KeyboardInterrupt
            continue

        from sidestep_engine.models.discovery import scan_models
        models = scan_models(ckpt_dir)
        if models:
            print_rich(f"\n  [green]Found {len(models)} model(s):[/]")
            for m in models:
                tag = "[green](official)[/]" if m.is_official else "[yellow](custom)[/]"
                print_rich(f"    - {m.name}  {tag}")
            print_rich("")
            break
        else:
            print_rich("  [yellow]No model directories found in that location.[/]")
            print_rich("  [dim]Examples: acestep-v15-turbo, acestep-v15-base, acestep-v15-sft[/]")
            if not ask_bool("Try a different path?", default=True, allow_back=False):
                print_rich("  [red]Setup requires at least one valid model directory.[/]")
                raise KeyboardInterrupt

    data["checkpoint_dir"] = ckpt_dir
    data["first_run_complete"] = True


def _step_output_dirs(data: dict) -> None:
    """Ask where to store adapter weights and preprocessed tensors."""
    section("Output Directories")
    print_message(
        "Adapter weights and preprocessed tensors will be organized in these folders.",
        kind="dim",
    )

    data["trained_adapters_dir"] = ask_output_path(
        "Trained adapters directory",
        default=data.get("trained_adapters_dir") or native_path("./trained_adapters"),
    )
    data["preprocessed_tensors_dir"] = ask_output_path(
        "Preprocessed tensors directory",
        default=data.get("preprocessed_tensors_dir") or native_path("./preprocessed_tensors"),
    )


def _step_summary(data: dict) -> None:
    """Show a summary of the configured settings."""
    section("Setup Complete")
    print_rich(f"  Checkpoint dir  : [bold]{_esc(data['checkpoint_dir'])}[/]")
    print_rich(f"  Adapters dir    : [bold]{_esc(data.get('trained_adapters_dir', './trained_adapters'))}[/]")
    print_rich(f"  Tensors dir     : [bold]{_esc(data.get('preprocessed_tensors_dir', './preprocessed_tensors'))}[/]")
    print_rich(f"  Gemini key      : {'[green]set[/]' if data.get('gemini_api_key') else '[dim]not set[/]'}")
    if data.get('gemini_model'):
        print_rich(f"  Gemini model    : [bold]{_esc(data['gemini_model'])}[/]")
    print_rich(f"  OpenAI key      : {'[green]set[/]' if data.get('openai_api_key') else '[dim]not set[/]'}")
    if data.get('openai_model'):
        print_rich(f"  OpenAI model    : [bold]{_esc(data['openai_model'])}[/]")
    if data.get('openai_base_url'):
        print_rich(f"  OpenAI base URL : [bold]{_esc(data['openai_base_url'])}[/]")
    print_rich(f"  Genius token    : {'[green]set[/]' if data.get('genius_api_token') else '[dim]not set[/]'}")
    print_rich("\n  [dim]Change any time from Settings.[/]\n")


# ---- Step list runner ------------------------------------------------------

def run_first_setup(seed: dict | None = None, *, is_edit: bool = False) -> dict:
    """Walk the user through first-time setup using a step-list.

    Pressing 'b' at any prompt navigates to the previous step.
    GoBack on the first step raises ``KeyboardInterrupt``.

    Args:
        seed: Optional existing settings to use as defaults.
        is_edit: Skip welcome/disclaimer when editing existing settings.

    Returns the settings dict ready for ``save_settings()``.
    """
    from sidestep_engine.settings import _default_settings

    data = dict(seed) if seed else _default_settings()

    steps: list[tuple[str, Callable[..., Any]]] = [
        ("Welcome", lambda d: _step_welcome(d, is_edit=is_edit)),
        ("Checkpoints", _step_checkpoint),
        ("Output Directories", _step_output_dirs),
        ("API Keys", _collect_api_keys),
        ("Summary", _step_summary),
    ]
    total = len(steps)
    i = 0

    while i < total:
        label, step_fn = steps[i]
        try:
            step_indicator(i + 1, total, label)
            step_fn(data)
            i += 1
        except GoBack:
            if i == 0:
                raise KeyboardInterrupt
            i -= 1

    return data



def run_settings_editor() -> dict | None:
    """Re-run setup seeded with existing settings so values are preserved.

    Returns the updated settings dict, or ``None`` if the user cancels.
    """
    from sidestep_engine.settings import load_settings, _default_settings

    print_rich("\n  [bold]Re-running Side-Step setup...[/]\n")
    try:
        existing = load_settings() or _default_settings()
        result = run_first_setup(seed=existing, is_edit=True)
        return result
    except (KeyboardInterrupt, EOFError):
        print_rich("  [dim]Cancelled.[/]")
        return None
