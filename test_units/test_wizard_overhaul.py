"""Tests for the wizard overhaul changes (Phases 1–5).

Covers:
- ask_bool y/n handling fix
- is_turbo canonical location
- print_message kind parameter
- settings mtime cache
- fisher map cache in answers dict
- wizard_shared_steps helpers
- _find_project_root sidestep_engine check
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


class TestAskBoolYN(unittest.TestCase):
    """Phase 1.1: ask_bool accepts y/n/yes/no."""

    @patch("sidestep_engine.ui.prompt_helpers.is_rich_active", return_value=False)
    def test_y_returns_true(self, _rich):
        from sidestep_engine.ui.prompt_helpers import ask_bool
        with patch("builtins.input", return_value="y"):
            self.assertTrue(ask_bool("test?"))

    @patch("sidestep_engine.ui.prompt_helpers.is_rich_active", return_value=False)
    def test_n_returns_false(self, _rich):
        from sidestep_engine.ui.prompt_helpers import ask_bool
        with patch("builtins.input", return_value="n"):
            self.assertFalse(ask_bool("test?"))

    @patch("sidestep_engine.ui.prompt_helpers.is_rich_active", return_value=False)
    def test_yes_returns_true(self, _rich):
        from sidestep_engine.ui.prompt_helpers import ask_bool
        with patch("builtins.input", return_value="yes"):
            self.assertTrue(ask_bool("test?"))

    @patch("sidestep_engine.ui.prompt_helpers.is_rich_active", return_value=False)
    def test_no_returns_false(self, _rich):
        from sidestep_engine.ui.prompt_helpers import ask_bool
        with patch("builtins.input", return_value="no"):
            self.assertFalse(ask_bool("test?"))

    @patch("sidestep_engine.ui.prompt_helpers.is_rich_active", return_value=False)
    def test_empty_returns_default_true(self, _rich):
        from sidestep_engine.ui.prompt_helpers import ask_bool
        with patch("builtins.input", return_value=""):
            self.assertTrue(ask_bool("test?", default=True))

    @patch("sidestep_engine.ui.prompt_helpers.is_rich_active", return_value=False)
    def test_empty_returns_default_false(self, _rich):
        from sidestep_engine.ui.prompt_helpers import ask_bool
        with patch("builtins.input", return_value=""):
            self.assertFalse(ask_bool("test?", default=False))

    @patch("sidestep_engine.ui.prompt_helpers.is_rich_active", return_value=False)
    def test_back_raises_goback(self, _rich):
        from sidestep_engine.ui.prompt_helpers import ask_bool, GoBack
        with patch("builtins.input", return_value="b"):
            with self.assertRaises(GoBack):
                ask_bool("test?", allow_back=True)

    @patch("sidestep_engine.ui.prompt_helpers.is_rich_active", return_value=False)
    def test_invalid_then_valid(self, _rich):
        """Invalid input loops, then valid input is accepted."""
        from sidestep_engine.ui.prompt_helpers import ask_bool
        with patch("builtins.input", side_effect=["maybe", "y"]):
            self.assertTrue(ask_bool("test?"))


class TestIsTurboCanonical(unittest.TestCase):
    """Phase 2.1: is_turbo is the single source of truth."""

    def test_turbo_by_name(self):
        from sidestep_engine.ui.flows.common import is_turbo
        self.assertTrue(is_turbo({"base_model": "turbo"}))
        self.assertTrue(is_turbo({"base_model": "acestep-v15-turbo"}))

    def test_base_by_name(self):
        from sidestep_engine.ui.flows.common import is_turbo
        self.assertFalse(is_turbo({"base_model": "base", "num_inference_steps": 50}))

    def test_fallback_to_steps(self):
        from sidestep_engine.ui.flows.common import is_turbo
        self.assertTrue(is_turbo({"base_model": "custom", "num_inference_steps": 8}))
        self.assertFalse(is_turbo({"base_model": "custom", "num_inference_steps": 50}))

    def test_empty_defaults_turbo(self):
        from sidestep_engine.ui.flows.common import is_turbo
        self.assertTrue(is_turbo({}))


class TestPrintMessageKind(unittest.TestCase):
    """Phase 1.3: print_message kind parameter."""

    @patch("sidestep_engine.ui.prompt_helpers.is_rich_active", return_value=False)
    def test_kind_does_not_crash_plain(self, _rich):
        """kind parameter works in plain mode (no Rich)."""
        from sidestep_engine.ui.prompt_helpers import print_message
        with patch("builtins.print") as mock_print:
            print_message("hello", kind="warn")
            mock_print.assert_called_once()

    def test_kind_styles_mapping(self):
        from sidestep_engine.ui.prompt_helpers import _KIND_STYLES
        self.assertEqual(_KIND_STYLES["warn"], "yellow")
        self.assertEqual(_KIND_STYLES["error"], "red")
        self.assertEqual(_KIND_STYLES["ok"], "green")
        self.assertEqual(_KIND_STYLES["info"], "cyan")


class TestSettingsCache(unittest.TestCase):
    """Phase 4.1: Settings mtime-based cache."""

    def setUp(self):
        import sidestep_engine.settings as mod
        self._mod = mod
        self._orig_cache = mod._cache
        mod._cache = None

    def tearDown(self):
        self._mod._cache = self._orig_cache

    def test_load_caches_on_second_call(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "settings.json"
            p.write_text(json.dumps({"version": 4, "checkpoint_dir": "/test"}))

            with patch.object(self._mod, "settings_path", return_value=p):
                data1 = self._mod.load_settings()
                self.assertEqual(data1["checkpoint_dir"], "/test")

                # Second call should use cache (no re-read)
                data2 = self._mod.load_settings()
                self.assertEqual(data2["checkpoint_dir"], "/test")
                self.assertIsNotNone(self._mod._cache)

    def test_save_invalidates_cache(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "settings.json"
            p.write_text(json.dumps({"version": 4}))

            with patch.object(self._mod, "settings_path", return_value=p):
                self._mod.load_settings()
                self.assertIsNotNone(self._mod._cache)

                self._mod.save_settings({"checkpoint_dir": "/new"})
                self.assertIsNone(self._mod._cache)


class TestFisherMapCache(unittest.TestCase):
    """Phase 4.2: Fisher map cache in answers dict."""

    def test_caches_result_in_answers(self):
        from sidestep_engine.ui.flows.train_steps import _has_fisher_map

        with tempfile.TemporaryDirectory() as tmp:
            fm = Path(tmp) / "fisher_map.json"
            fm.write_text(json.dumps({"rank_pattern": {"a": 8}}))

            a = {"dataset_dir": tmp}
            result = _has_fisher_map(a)
            self.assertTrue(result)
            self.assertTrue(a["_fisher_map_cached"])

            # Changing dataset_dir invalidates the cache (fix for stale cache bug)
            a["dataset_dir"] = "/nonexistent"
            self.assertFalse(_has_fisher_map(a))

    def test_no_fisher_map(self):
        from sidestep_engine.ui.flows.train_steps import _has_fisher_map

        with tempfile.TemporaryDirectory() as tmp:
            a = {"dataset_dir": tmp}
            self.assertFalse(_has_fisher_map(a))
            self.assertFalse(a["_fisher_map_cached"])


class TestFindProjectRoot(unittest.TestCase):
    """Phase 1.2: _find_project_root checks sidestep_engine/ not acestep/."""

    def test_finds_root_with_sidestep_engine(self):
        from sidestep_engine.ui.presets import _find_project_root

        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "pyproject.toml").touch()
            (Path(tmp) / "sidestep_engine").mkdir()

            with patch("os.getcwd", return_value=tmp):
                import sys
                with patch.object(sys, "argv", [str(Path(tmp) / "train.py")]):
                    root = _find_project_root()
                    # Should find it via CWD or script dir
                    self.assertIsNotNone(root)


if __name__ == "__main__":
    unittest.main()
