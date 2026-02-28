"""
Rich-enhanced argparse help formatter.

Replaces the default ``argparse.HelpFormatter`` with a version that uses
Rich markup for colored, readable ``--help`` output.

This is a lightweight custom implementation that avoids the need for the
``rich-argparse`` third-party package.
"""

from __future__ import annotations

import argparse
import io
import sys
from typing import Optional


class RichHelpFormatter(argparse.HelpFormatter):
    """argparse help formatter that uses Rich for styled terminal output.

    Falls back to the default formatter if Rich is not available.
    """

    def __init__(
        self,
        prog: str,
        indent_increment: int = 2,
        max_help_position: int = 30,
        width: Optional[int] = None,
    ) -> None:
        # Try to get terminal width from Rich
        if width is None:
            try:
                from rich.console import Console
                width = Console().width
            except ImportError:
                width = 80
        super().__init__(prog, indent_increment, max_help_position, width)

    def format_help(self) -> str:
        """Override to apply Rich styling to the help output."""
        raw_help = super().format_help()

        try:
            from rich.console import Console
            from rich.text import Text
        except ImportError:
            return raw_help

        # Build styled output
        console = Console(file=io.StringIO(), force_terminal=True, width=self._width)

        for line in raw_help.split("\n"):
            stripped = line.strip()

            # Section headers (e.g. "positional arguments:", "options:")
            if stripped.endswith(":") and not stripped.startswith("-"):
                console.print(f"[bold cyan]{line}[/]")

            # Argument lines (start with -)
            elif stripped.startswith("-"):
                # Split into arg name and help text
                parts = line.split("  ", 1)
                if len(parts) == 2:
                    arg_part = parts[0]
                    help_part = parts[1]
                    console.print(f"[bold green]{arg_part}[/]  {help_part}")
                else:
                    console.print(f"[bold green]{line}[/]")

            # Subcommand names (indented words before help text)
            elif stripped and not stripped.startswith("{") and "  " in line and line.startswith("  "):
                parts = line.split("  ", 1)
                # Filter out empty parts
                parts = [p for p in line.split("  ") if p.strip()]
                if len(parts) >= 2:
                    cmd = parts[0]
                    desc = "  ".join(parts[1:])
                    indent = line[:len(line) - len(line.lstrip())]
                    console.print(f"{indent}[bold yellow]{cmd.strip()}[/]  {desc.strip()}")
                else:
                    console.print(line)

            # Usage line
            elif stripped.startswith("usage:"):
                console.print(f"[bold]{line}[/]")

            # Everything else
            else:
                console.print(line)

        output = console.file.getvalue()
        return output


