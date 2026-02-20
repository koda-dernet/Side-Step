"""Tests for model weight file detection in model_discovery."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from acestep.training_v2.model_discovery import _has_weight_files


class TestHasWeightFiles(unittest.TestCase):
    """Test _has_weight_files detects loadable weight files."""

    def test_model_safetensors(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            (Path(td) / "model.safetensors").write_bytes(b"x")
            self.assertTrue(_has_weight_files(Path(td)))

    def test_pytorch_model_bin(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            (Path(td) / "pytorch_model.bin").write_bytes(b"x")
            self.assertTrue(_has_weight_files(Path(td)))

    def test_glob_safetensors(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            (Path(td) / "model-00001-of-00002.safetensors").write_bytes(b"x")
            self.assertTrue(_has_weight_files(Path(td)))

    def test_empty_dir_returns_false(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            self.assertFalse(_has_weight_files(Path(td)))

    def test_config_only_returns_false(self) -> None:
        with tempfile.TemporaryDirectory() as td:
            (Path(td) / "config.json").write_text("{}")
            self.assertFalse(_has_weight_files(Path(td)))

    def test_nonexistent_dir_returns_false(self) -> None:
        self.assertFalse(_has_weight_files(Path("/nonexistent/path/xyz")))


if __name__ == "__main__":
    unittest.main()
