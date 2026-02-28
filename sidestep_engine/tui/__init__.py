"""
Side-Step TUI - Full Interactive Terminal User Interface

.. deprecated:: 1.0.0-beta
    **The TUI (sidestep_engine.tui) is DEPRECATED.**
    It will be removed in a future release.
    Use ``python train.py`` (the CLI wizard) instead.
    Do NOT add new features to this package.
    Do NOT import from this package in new code.

A Textual-based TUI for ACE-Step LoRA training with live monitoring,
interactive configuration, and dataset management.
"""

from __future__ import annotations

import warnings as _warnings

_warnings.warn(
    "sidestep_engine.tui is DEPRECATED and will be removed. "
    "Use 'python train.py' (CLI wizard) instead.",
    DeprecationWarning,
    stacklevel=2,
)

__all__ = ["SideStepApp", "run_tui"]


def run_tui() -> None:
    """Launch the Side-Step TUI application.

    .. deprecated:: 1.0.0-beta
        Use ``python train.py`` instead.
    """
    from sidestep_engine.tui.app import SideStepApp
    
    app = SideStepApp()
    app.run()
