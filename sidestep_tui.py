#!/usr/bin/env python3
"""
Side-Step TUI -- Interactive Terminal Interface for ACE-Step LoRA Training
by dernet

Usage:
    python sidestep_tui.py

The original CLI (train.py) remains unchanged and fully functional.
This TUI is an alternative interface that reuses the same core training modules.

Dependencies:
    pip install -r requirements-sidestep.txt
"""

from __future__ import annotations

import sys
from pathlib import Path

SIDESTEP_VERSION = "0.9.0-beta"

_BANNER = r"""
  ███████ ██ ██████  ███████       ███████ ████████ ███████ ██████
  ██      ██ ██   ██ ██            ██         ██    ██      ██   ██
  ███████ ██ ██   ██ █████   █████ ███████    ██    █████   ██████
       ██ ██ ██   ██ ██                 ██    ██    ██      ██
  ███████ ██ ██████  ███████       ███████    ██    ███████ ██
"""


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
    """Launch the Side-Step TUI."""
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
        from acestep.training_v2.tui import run_tui
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
