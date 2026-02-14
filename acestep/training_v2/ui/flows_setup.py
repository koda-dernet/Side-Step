"""
First-run setup wizard and settings editor for Side-Step.

Runs automatically on the first launch.  Collects environment paths
(checkpoint directory, ACE-Step install location) and vanilla-mode intent.
Re-accessible from the main menu under "Settings".
"""

from __future__ import annotations

import logging
from pathlib import Path

from acestep.training_v2.ui import console, is_rich_active
from acestep.training_v2.ui.prompt_helpers import (
    ask_bool,
    ask_path,
    native_path,
    section,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _print(msg: str) -> None:
    """Print via Rich console if available, else plain print."""
    if is_rich_active() and console is not None:
        console.print(msg)
    else:
        # Strip Rich markup for plain output
        import re
        print(re.sub(r"\[/?[^\]]*\]", "", msg))


def _smart_checkpoint_default(ace_step_dir: str | None) -> str:
    """Pick a sensible default checkpoint path based on context."""
    if ace_step_dir:
        candidate = Path(ace_step_dir) / "checkpoints"
        if candidate.is_dir():
            return str(candidate)
    # Fall back to common relative paths
    for rel in ("./checkpoints", "../ACE-Step-1.5/checkpoints"):
        if Path(rel).is_dir():
            return native_path(rel)
    return native_path("./checkpoints")


def _validate_ace_step_dir(path_str: str) -> bool:
    """Check that the given path looks like an ACE-Step installation."""
    p = Path(path_str)
    if not p.is_dir():
        return False
    # Must have the acestep package with training/trainer.py for vanilla
    trainer = p / "acestep" / "training" / "trainer.py"
    return trainer.is_file()


# ---------------------------------------------------------------------------
# First-run wizard
# ---------------------------------------------------------------------------

def run_first_setup() -> dict:
    """Walk the user through first-time setup.

    Returns the settings dict ready for ``save_settings()``.
    """
    from acestep.training_v2.settings import _default_settings

    data = _default_settings()

    # -- Welcome + disclaimer -----------------------------------------------
    section("Welcome to Side-Step")
    _print("  [bold]Before we begin, a few important notes:[/]\n")
    _print("  [yellow]1.[/] You are responsible for downloading the model weights")
    _print("     you want to train on (e.g. via [bold]acestep-download[/] or manually).")
    _print("  [yellow]2.[/] If you are training on a fine-tune, you [bold]MUST[/] also have")
    _print("     the original base model that fine-tune was built from.")
    _print("  [yellow]3.[/] [bold]Never rename checkpoint folders.[/] The model loader uses")
    _print("     folder names and config.json files to identify models.\n")

    # -- Vanilla intent -----------------------------------------------------
    section("Vanilla Training Mode")
    _print("  Side-Step's [bold green]corrected (fixed)[/] training is fully standalone.")
    _print("  [bold yellow]Vanilla[/] mode reproduces the original ACE-Step training")
    _print("  behavior and requires a base ACE-Step installation.\n")

    data["vanilla_enabled"] = ask_bool(
        "Do you plan to use Vanilla training mode?",
        default=False,
    )

    ace_step_dir: str | None = None
    if data["vanilla_enabled"]:
        _print("\n  Point me to your ACE-Step 1.5 installation directory.")
        _print("  [dim](The folder that contains the acestep/ package.)[/]\n")
        while True:
            ace_step_dir = ask_path(
                "ACE-Step install directory",
                default=native_path("../ACE-Step-1.5"),
            )
            if _validate_ace_step_dir(ace_step_dir):
                _print(f"  [green]Validated: {ace_step_dir}[/]")
                break
            _print("  [red]That directory does not look like an ACE-Step install.[/]")
            _print("  [dim]Expected to find: acestep/training/trainer.py[/]")
            if not ask_bool("Try a different path?", default=True):
                _print("  [yellow]Vanilla mode will not be available until you configure this.[/]")
                data["vanilla_enabled"] = False
                break

    data["ace_step_dir"] = ace_step_dir

    # -- Checkpoint directory -----------------------------------------------
    section("Model Checkpoints")
    _print("  Where are your model checkpoint folders?")
    _print("  [dim](Each model variant lives in its own subfolder, e.g.\n   checkpoints/acestep-v15-turbo/, checkpoints/acestep-v15-base/, etc.)[/]\n")

    default_ckpt = _smart_checkpoint_default(ace_step_dir)
    while True:
        ckpt_dir = ask_path("Checkpoint directory", default=default_ckpt)
        ckpt_path = Path(ckpt_dir)
        if not ckpt_path.is_dir():
            _print(f"  [red]Directory not found: {ckpt_dir}[/]")
            if not ask_bool("Try a different path?", default=True):
                break
            continue

        # Scan for model subdirectories
        from acestep.training_v2.model_discovery import scan_models
        models = scan_models(ckpt_dir)
        if models:
            _print(f"\n  [green]Found {len(models)} model(s):[/]")
            for m in models:
                tag = "[green](official)[/]" if m.is_official else "[yellow](custom)[/]"
                _print(f"    - {m.name}  {tag}")
            _print("")
            break
        else:
            _print("  [yellow]No model directories found in that location.[/]")
            _print("  [dim](Looking for subfolders with a config.json file.)[/]")
            if not ask_bool("Try a different path?", default=True):
                break

    data["checkpoint_dir"] = ckpt_dir
    data["first_run_complete"] = True

    # -- Summary ------------------------------------------------------------
    section("Setup Complete")
    _print(f"  Checkpoint dir : [bold]{data['checkpoint_dir']}[/]")
    if data["vanilla_enabled"] and data["ace_step_dir"]:
        _print(f"  ACE-Step dir   : [bold]{data['ace_step_dir']}[/]")
        _print(f"  Vanilla mode   : [bold green]enabled[/]")
    else:
        _print(f"  Vanilla mode   : [bold yellow]disabled[/] (corrected mode is standalone)")
    _print("")
    _print("  [dim]You can change these any time from the main menu → Settings.[/]\n")

    return data


# ---------------------------------------------------------------------------
# Settings editor (re-run setup from the menu)
# ---------------------------------------------------------------------------

def run_settings_editor() -> dict | None:
    """Re-run the setup flow, pre-filling current settings.

    Returns the updated settings dict, or ``None`` if the user cancels.
    """
    from acestep.training_v2.settings import load_settings

    _print("\n  [bold]Re-running Side-Step setup...[/]\n")
    try:
        return run_first_setup()
    except (KeyboardInterrupt, EOFError):
        _print("  [dim]Cancelled.[/]")
        return None
