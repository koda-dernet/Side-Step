"""Tests for _warn_non_base_variant in wizard_shared_steps."""

from __future__ import annotations

import unittest
from unittest.mock import patch

from sidestep_engine.ui.prompt_helpers import GoBack


# Path to the function under test
_MOD = "sidestep_engine.ui.flows.wizard_shared_steps"


class TestWarnNonBaseVariant(unittest.TestCase):
    """Verify the non-base variant warning + confirmation gate."""

    def _call(self, answers: dict) -> None:
        from sidestep_engine.ui.flows.wizard_shared_steps import _warn_non_base_variant
        _warn_non_base_variant(answers)

    # ---- Base variants should pass silently ----

    def test_base_no_warning(self):
        """Selecting 'base' should not trigger any warning."""
        with patch(f"{_MOD}.print_message") as mock_msg:
            self._call({"base_model": "base"})
        mock_msg.assert_not_called()

    def test_base_in_name_no_warning(self):
        """A variant name containing 'base' should not trigger a warning."""
        with patch(f"{_MOD}.print_message") as mock_msg:
            self._call({"base_model": "acestep-v15-base"})
        mock_msg.assert_not_called()

    # ---- Non-base variants should warn ----

    @patch(f"{_MOD}.ask_bool", return_value=True)
    @patch(f"{_MOD}.print_message")
    def test_turbo_warns_and_continues(self, mock_msg, mock_bool):
        """Turbo should warn; answering yes continues without GoBack."""
        self._call({"base_model": "turbo"})
        self.assertTrue(mock_msg.called)
        mock_bool.assert_called_once()
        # Check the warning mentions turbo-specific reason
        warn_text = mock_msg.call_args_list[0][0][0]
        self.assertIn("Turbo", warn_text)

    @patch(f"{_MOD}.ask_bool", return_value=False)
    @patch(f"{_MOD}.print_message")
    def test_turbo_declines_raises_goback(self, mock_msg, mock_bool):
        """Turbo warning declined should raise GoBack."""
        with self.assertRaises(GoBack):
            self._call({"base_model": "turbo"})

    @patch(f"{_MOD}.ask_bool", return_value=True)
    @patch(f"{_MOD}.print_message")
    def test_sft_warns_and_continues(self, mock_msg, mock_bool):
        """SFT should warn; answering yes continues without GoBack."""
        self._call({"base_model": "sft"})
        self.assertTrue(mock_msg.called)
        warn_text = mock_msg.call_args_list[0][0][0]
        self.assertIn("SFT", warn_text)

    @patch(f"{_MOD}.ask_bool", return_value=False)
    @patch(f"{_MOD}.print_message")
    def test_sft_declines_raises_goback(self, mock_msg, mock_bool):
        """SFT warning declined should raise GoBack."""
        with self.assertRaises(GoBack):
            self._call({"base_model": "sft"})

    @patch(f"{_MOD}.ask_bool", return_value=True)
    @patch(f"{_MOD}.print_message")
    def test_unknown_variant_warns(self, mock_msg, mock_bool):
        """An unknown variant should still warn with generic message."""
        self._call({"base_model": "unknown"})
        self.assertTrue(mock_msg.called)
        warn_text = mock_msg.call_args_list[0][0][0]
        self.assertIn("strongly recommended", warn_text)

    # ---- Falls back to model_variant when base_model missing ----

    @patch(f"{_MOD}.ask_bool", return_value=True)
    @patch(f"{_MOD}.print_message")
    def test_falls_back_to_model_variant(self, mock_msg, mock_bool):
        """When base_model is absent, model_variant should be checked."""
        self._call({"model_variant": "sft"})
        self.assertTrue(mock_msg.called)

    def test_base_model_variant_fallback_no_warning(self):
        """When base_model is absent but model_variant contains 'base', no warning."""
        with patch(f"{_MOD}.print_message") as mock_msg:
            self._call({"model_variant": "acestep-v15-base"})
        mock_msg.assert_not_called()

    @patch(f"{_MOD}.ask_bool", return_value=True)
    @patch(f"{_MOD}.print_message")
    def test_ask_bool_default_is_no(self, mock_msg, mock_bool):
        """The confirmation default should be False (no)."""
        self._call({"base_model": "turbo"})
        _args, kwargs = mock_bool.call_args
        self.assertFalse(kwargs.get("default", _args[1] if len(_args) > 1 else True))


if __name__ == "__main__":
    unittest.main()
