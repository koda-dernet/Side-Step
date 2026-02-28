"""
Side-Step -- Rich UI Layer

Provides a shared ``Console`` instance and a ``RICH_AVAILABLE`` flag so that
every UI module can degrade gracefully when Rich is not installed.

Exports
-------
console : Console | None
    Shared Rich console (``None`` when Rich is missing).
RICH_AVAILABLE : bool
    ``True`` when ``rich>=13`` is importable.
plain_mode : bool
    Module-level flag toggled by ``--plain``.  When ``True`` *or* stdout is
    not a TTY, all UI helpers fall back to plain ``print()`` output.
"""

from __future__ import annotations

import sys

# ---- Rich availability check ------------------------------------------------

RICH_AVAILABLE: bool = False
console = None  # type: ignore[assignment]

try:
    from rich.console import Console as _Console
    from rich.theme import Theme as _Theme

    # Override prompt.default so Rich Prompt.ask() defaults render magenta,
    # matching the manually-styled defaults in the allow_back=True path.
    _sidestep_theme = _Theme({"prompt.default": "magenta"})
    console = _Console(stderr=True, theme=_sidestep_theme)
    RICH_AVAILABLE = True
except ImportError:
    pass

# ---- Plain-mode flag (set via --plain CLI arg) ------------------------------

plain_mode: bool = False


def set_plain_mode(value: bool) -> None:
    """Toggle plain-text output globally."""
    global plain_mode
    plain_mode = value


def is_rich_active() -> bool:
    """Return True when Rich output should be used."""
    if plain_mode or not RICH_AVAILABLE:
        return False
    if console is not None and not console.is_terminal:
        return False
    return True


# ---- TrainingUpdate (backward-compat re-export from core.types) --------------

from sidestep_engine.core.types import TrainingUpdate  # noqa: E402, F401
