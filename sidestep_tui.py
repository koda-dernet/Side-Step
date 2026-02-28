#!/usr/bin/env python3
"""
Side-Step TUI -- Interactive Terminal Interface for ACE-Step LoRA Training
by dernet

.. deprecated:: 1.0.0-beta
    **The TUI is DEPRECATED and will be removed in a future release.**
    Use the CLI wizard instead: ``python train.py``
    The CLI wizard provides the same functionality with better compatibility
    and is the only actively maintained interface going forward.

Usage:
    python sidestep_tui.py

The CLI wizard (train.py) is the recommended interface.
This TUI is kept for backward compatibility but receives no new features.

Dependencies:
    pip install -r requirements-sidestep.txt
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

SIDESTEP_VERSION = "1.0.0-beta"

_BANNER = r"""
  ███████ ██ ██████  ███████       ███████ ████████ ███████ ██████
  ██      ██ ██   ██ ██            ██         ██    ██      ██   ██
  ███████ ██ ██   ██ █████   █████ ███████    ██    █████   ██████
       ██ ██ ██   ██ ██                 ██    ██    ██      ██
  ███████ ██ ██████  ███████       ███████    ██    ███████ ██
"""


from sidestep_engine._compat import install_torchao_warning_filter


def check_dependencies() -> list[str]:
    """Return list of missing required dependencies."""
    missing = []
    try:
        import textual  # noqa: F401
    except ImportError:
        missing.append("textual>=0.47.0")
    try:
        import rich  # noqa: F401
    except ImportError:
        missing.append("rich>=13.0.0")
    return missing


def main() -> int:
    """Launch the Side-Step TUI.

    .. deprecated:: 1.0.0-beta
        Use ``python train.py`` instead.
    """
    import warnings
    warnings.warn(
        "The Side-Step TUI is DEPRECATED and will be removed in a future release. "
        "Use 'python train.py' for the CLI wizard instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    print()
    print("  ╔══════════════════════════════════════════════════════════╗")
    print("  ║  ⚠  DEPRECATED — The TUI is no longer maintained.      ║")
    print("  ║     Use 'python train.py' for the CLI wizard instead.   ║")
    print("  ╚══════════════════════════════════════════════════════════╝")
    print()

    install_torchao_warning_filter()

    missing = check_dependencies()
    if missing:
        print(_BANNER.strip())
        print(f"  Side-Step v{SIDESTEP_VERSION} by dernet")
        print()
        print("  [!] Missing dependencies:")
        for dep in missing:
            print(f"      - {dep}")
        print()
        print("  Install them with:")
        print("      pip install -r requirements-sidestep.txt")
        print()
        return 1

    # Ensure the project root is in the path
    project_root = Path(__file__).parent
    if str(project_root) not in sys.path:
        sys.path.insert(0, str(project_root))

    try:
        from sidestep_engine.tui import run_tui
        run_tui()
        return 0
    except KeyboardInterrupt:
        print("\n[Side-Step] Interrupted.")
        return 130
    except Exception as e:
        print(f"[Side-Step] Error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
