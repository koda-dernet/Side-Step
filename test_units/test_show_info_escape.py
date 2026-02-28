"""Tests for Rich markup escaping in errors.show_info().

Regression test: paths containing square brackets (e.g.
``/media/user/[volume]/song.pt``) must be displayed verbatim, not
interpreted as Rich markup tags.
"""

from __future__ import annotations

import unittest
from unittest.mock import MagicMock, patch

from sidestep_engine.ui.errors import show_info


class TestShowInfoEscape(unittest.TestCase):
    """Verify show_info escapes Rich markup in user-provided messages."""

    @patch("sidestep_engine.ui.prompt_helpers.is_rich_active", return_value=True)
    @patch("sidestep_engine.ui.prompt_helpers.console")
    def test_brackets_in_path_are_escaped(
        self, mock_console: MagicMock, _mock_rich: MagicMock,
    ) -> None:
        """A path like /media/[volume]/song.pt must not be parsed as markup."""
        show_info("Loading /media/[volume]/song.pt")
        mock_console.print.assert_called_once()
        printed = mock_console.print.call_args[0][0]
        # The bracket must be escaped as \[ so Rich doesn't parse [volume] as a tag
        self.assertIn("\\[volume]", printed)
        self.assertNotIn("[volume]", printed.replace("\\[volume]", ""))

    @patch("sidestep_engine.ui.prompt_helpers.is_rich_active", return_value=True)
    @patch("sidestep_engine.ui.prompt_helpers.console")
    def test_plain_message_unchanged(
        self, mock_console: MagicMock, _mock_rich: MagicMock,
    ) -> None:
        """A message without brackets should pass through normally."""
        show_info("Training started successfully")
        mock_console.print.assert_called_once()
        printed = mock_console.print.call_args[0][0]
        self.assertIn("Training started successfully", printed)

    @patch("sidestep_engine.ui.prompt_helpers.is_rich_active", return_value=False)
    def test_plain_mode_no_escaping(self, _mock_rich: MagicMock) -> None:
        """In plain mode, brackets are printed as-is (no Rich to interpret them)."""
        with patch("builtins.print") as mock_print:
            show_info("Path is /data/[test]/file.pt")
            mock_print.assert_called_once()
            printed = mock_print.call_args[0][0]
            self.assertIn("[test]", printed)


if __name__ == "__main__":
    unittest.main()
