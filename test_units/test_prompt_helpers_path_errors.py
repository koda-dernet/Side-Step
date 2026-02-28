from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from sidestep_engine.ui.prompt_helpers import ask_output_path, ask_path


class TestAskPathErrors(unittest.TestCase):
    def test_permission_denied_then_success(self) -> None:
        with (
            patch(
                "sidestep_engine.ui.prompt_helpers.ask",
                side_effect=["/tmp/denied", "/tmp/ok"],
            ),
            patch(
                "pathlib.Path.exists",
                side_effect=[PermissionError("denied"), True],
            ),
            patch("pathlib.Path.stat", return_value=None),
            patch("pathlib.Path.resolve", side_effect=lambda *args, **kwargs: "/tmp/ok"),
        ):
            out = ask_path("Dataset dir", must_exist=True)
        self.assertIn("/tmp/ok", str(out))

    def test_not_found_then_success(self) -> None:
        with (
            patch(
                "sidestep_engine.ui.prompt_helpers.ask",
                side_effect=["/tmp/missing", "/tmp/ok2"],
            ),
            patch("pathlib.Path.exists", side_effect=[False, True]),
            patch("pathlib.Path.stat", return_value=None),
            patch("pathlib.Path.resolve", side_effect=lambda *args, **kwargs: "/tmp/ok2"),
        ):
            out = ask_path("Checkpoint dir", must_exist=True)
        self.assertIn("/tmp/ok2", str(out))


class TestAskOutputPath(unittest.TestCase):
    """Tests for ask_output_path writability validation."""

    def test_writable_dir_accepts(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            out_path = Path(td) / "output"
            with patch(
                "sidestep_engine.ui.prompt_helpers.ask",
                return_value=str(out_path),
            ):
                result = ask_output_path("Output dir", required=True)
            self.assertIn(str(td), result)
            self.assertIn("output", result)

    def test_writable_file_path_accepts(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            out_file = Path(td) / "results.json"
            with patch(
                "sidestep_engine.ui.prompt_helpers.ask",
                return_value=str(out_file),
            ):
                result = ask_output_path("Output file", required=True, for_file=True)
            self.assertIn("results.json", result)

    def test_required_false_empty_returns_empty_string(self) -> None:
        with patch(
            "sidestep_engine.ui.prompt_helpers.ask",
            return_value="",
        ):
            result = ask_output_path("Output", default="", required=False)
        self.assertEqual(result, "")

    def test_unwritable_then_writable_accepts_second(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            bad_path = Path(td) / "readonly_subdir"
            bad_path.mkdir()
            good_path = Path(td) / "output"
            with patch(
                "sidestep_engine.ui.prompt_helpers.ask",
                side_effect=[str(bad_path), str(good_path)],
            ), patch(
                "sidestep_engine.ui.prompt_helpers._check_path_writable",
                side_effect=["Permission denied", None],
            ):
                result = ask_output_path("Output dir", required=True)
            self.assertIn(str(td), result)
            self.assertIn("output", result)


if __name__ == "__main__":
    unittest.main()

