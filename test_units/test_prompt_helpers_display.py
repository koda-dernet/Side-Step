"""Tests for print_message, print_rich, and blank_line display helpers.

Covers the new helpers added during wizard text color standardization:
leading-newline stripping, print_rich markup pass-through / plain strip,
blank_line output, and KIND_STYLES completeness.
"""

from __future__ import annotations

import io
import re
import unittest
from unittest.mock import patch

from sidestep_engine.ui.prompt_helpers import (
    _KIND_STYLES,
    blank_line,
    print_message,
    print_rich,
)


def _capture_plain(fn, *args, **kwargs) -> str:
    """Run *fn* with Rich disabled and capture stdout."""
    with patch("sidestep_engine.ui.prompt_helpers.is_rich_active", return_value=False):
        buf = io.StringIO()
        with patch("builtins.print", side_effect=lambda *a, **kw: buf.write(" ".join(str(x) for x in a) + "\n")):
            fn(*args, **kwargs)
    return buf.getvalue()


class TestPrintMessageLeadingNewlines(unittest.TestCase):
    """print_message strips leading \\n and emits bare blank lines."""

    def test_no_leading_newline(self):
        out = _capture_plain(print_message, "hello")
        self.assertEqual(out.strip(), "hello")

    def test_single_leading_newline(self):
        out = _capture_plain(print_message, "\nhello")
        lines = out.split("\n")
        # First line should be empty (bare blank line), second has content
        self.assertEqual(lines[0].strip(), "")
        self.assertIn("hello", lines[1])

    def test_double_leading_newline(self):
        out = _capture_plain(print_message, "\n\nhello")
        lines = out.split("\n")
        self.assertEqual(lines[0].strip(), "")
        self.assertEqual(lines[1].strip(), "")
        self.assertIn("hello", lines[2])

    def test_no_trailing_spaces_on_blank_lines(self):
        """Regression: the old bug was '  \\n' (2-space indent on blank line)."""
        out = _capture_plain(print_message, "\nhello")
        first_line = out.split("\n")[0]
        self.assertEqual(first_line, "", "Blank line should have no trailing spaces")


class TestPrintMessageKinds(unittest.TestCase):
    """print_message kind= parameter maps through _KIND_STYLES."""

    def test_all_kinds_have_nonempty_styles(self):
        for kind, style in _KIND_STYLES.items():
            self.assertTrue(style, f"kind={kind!r} maps to empty style")

    def test_heading_kind_exists(self):
        self.assertIn("heading", _KIND_STYLES)
        self.assertEqual(_KIND_STYLES["heading"], "bold cyan")

    def test_banner_kind_exists(self):
        self.assertIn("banner", _KIND_STYLES)
        self.assertEqual(_KIND_STYLES["banner"], "bold")

    def test_recalled_kind_exists(self):
        self.assertIn("recalled", _KIND_STYLES)
        self.assertEqual(_KIND_STYLES["recalled"], "magenta")

    def test_kind_warn_plain_output(self):
        out = _capture_plain(print_message, "caution", kind="warn")
        self.assertIn("caution", out)


class TestPrintRich(unittest.TestCase):
    """print_rich passes markup through in Rich mode, strips in plain."""

    def test_plain_strips_tags(self):
        out = _capture_plain(print_rich, "[bold]hello[/] world")
        self.assertIn("hello world", out)
        self.assertNotIn("[bold]", out)

    def test_plain_strips_nested_tags(self):
        out = _capture_plain(print_rich, "[bold red]err[/] [dim]hint[/]")
        self.assertIn("err", out)
        self.assertIn("hint", out)
        self.assertNotIn("[bold red]", out)

    def test_leading_newline_stripped(self):
        out = _capture_plain(print_rich, "\n[bold]title[/]")
        lines = out.split("\n")
        self.assertEqual(lines[0].strip(), "")
        self.assertIn("title", lines[1])


class TestBlankLine(unittest.TestCase):
    """blank_line emits exactly one empty line."""

    def test_output_is_single_newline(self):
        out = _capture_plain(blank_line)
        self.assertEqual(out, "\n")


if __name__ == "__main__":
    unittest.main()
