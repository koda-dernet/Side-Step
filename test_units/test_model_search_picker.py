"""Tests for model search-by-name picker validation.

Ensures single fuzzy match shows a picker instead of auto-selecting,
and that back navigation works correctly.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from sidestep_engine.models.discovery import ModelInfo
from sidestep_engine.ui.flows.wizard_shared_steps import _search_loop


def _make_model(name: str, base: str = "turbo", official: bool = True) -> ModelInfo:
    return ModelInfo(
        name=name,
        path=Path(f"/fake/{name}"),
        is_official=official,
        base_model=base,
    )


class TestSearchPicker(unittest.TestCase):
    """Test _search_loop shows picker for single match and handles back."""

    _PATCH_ASK = "sidestep_engine.ui.flows.wizard_shared_steps.ask"
    _PATCH_MENU = "sidestep_engine.ui.flows.wizard_shared_steps.menu"

    def test_single_fuzzy_match_shows_picker(self) -> None:
        models = [
            _make_model("acestep-v15-base", "base"),
            _make_model("acestep-v15-sft", "sft"),
            _make_model("acestep-v15-turbo", "turbo"),
        ]
        menu_calls = []

        def _capture_menu(title, options, default=1, allow_back=True):
            menu_calls.append((title, options))
            return "acestep-v15-base"

        with patch(self._PATCH_ASK, side_effect=["sft+base"]), \
             patch(self._PATCH_MENU, side_effect=_capture_menu):
            result = _search_loop(models)

        self.assertIsNotNone(result)
        self.assertEqual(result[0], "acestep-v15-base")
        self.assertEqual(len(menu_calls), 1)
        title, options = menu_calls[0]
        self.assertIn("Matched 1 model", title)
        self.assertEqual(len(options), 1)
        self.assertEqual(options[0][0], "acestep-v15-base")

    def test_single_match_picker_returns_model_when_selected(self) -> None:
        models = [_make_model("my-custom-model", "turbo", official=False)]

        with patch(self._PATCH_ASK, return_value="custom"), \
             patch(self._PATCH_MENU, return_value="my-custom-model"):
            result = _search_loop(models)

        self.assertIsNotNone(result)
        self.assertEqual(result[0], "my-custom-model")
        self.assertEqual(result[1].name, "my-custom-model")

    def test_single_match_picker_back_returns_to_search(self) -> None:
        from sidestep_engine.ui.prompt_helpers import GoBack

        models = [_make_model("acestep-v15-base", "base")]

        with patch(self._PATCH_ASK, side_effect=["sft+base", "base"]), \
             patch(self._PATCH_MENU, side_effect=[GoBack(), "acestep-v15-base"]):
            result = _search_loop(models)

        self.assertIsNotNone(result)
        self.assertEqual(result[0], "acestep-v15-base")

    def test_multiple_matches_unchanged(self) -> None:
        models = [
            _make_model("acestep-v15-base", "base"),
            _make_model("acestep-v15-sft", "sft"),
        ]
        menu_calls = []

        def _capture_menu(title, options, default=1, allow_back=True):
            menu_calls.append((title, options))
            return "acestep-v15-sft"

        with patch(self._PATCH_ASK, return_value="v15"), \
             patch(self._PATCH_MENU, side_effect=_capture_menu):
            result = _search_loop(models)

        self.assertIsNotNone(result)
        self.assertEqual(result[0], "acestep-v15-sft")
        self.assertEqual(len(menu_calls), 1)
        title, options = menu_calls[0]
        self.assertIn("Multiple matches", title)
        self.assertEqual(len(options), 2)

    def test_no_matches_loops(self) -> None:
        models = [
            _make_model("acestep-v15-turbo", "turbo"),
        ]

        with patch(self._PATCH_ASK, side_effect=["nonexistent-xyz-123", "turbo"]), \
             patch(self._PATCH_MENU, return_value="acestep-v15-turbo"):
            result = _search_loop(models)

        self.assertIsNotNone(result)
        self.assertEqual(result[0], "acestep-v15-turbo")


if __name__ == "__main__":
    unittest.main()
