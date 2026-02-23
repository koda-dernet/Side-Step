"""Regression tests for manifest parsing and Windows-style paths."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path


class TestManifestWindowsPaths(unittest.TestCase):
    """Ensure manifest loading is resilient to common Windows pitfalls."""

    def test_invalid_json_manifest_falls_back_to_directory_scan(self) -> None:
        from acestep.training_v2._vendor.data_module import PreprocessedTensorDataset

        with tempfile.TemporaryDirectory() as td:
            tensor_dir = Path(td)
            (tensor_dir / "a.pt").write_bytes(b"dummy")

            # Intentionally invalid JSON due to unescaped backslashes.
            bad_path = ".C:" + "\\temp\\broken\\a.pt"
            (tensor_dir / "manifest.json").write_text(
                '{\n  "samples": ["' + bad_path + '"]\n}\n',
                encoding="utf-8",
            )

            ds = PreprocessedTensorDataset(str(tensor_dir))
            self.assertEqual(len(ds), 1)
            self.assertTrue(ds.manifest_fallback_used)
            self.assertIsNotNone(ds.manifest_error)

    def test_non_list_samples_falls_back_to_directory_scan(self) -> None:
        from acestep.training_v2._vendor.data_module import PreprocessedTensorDataset

        with tempfile.TemporaryDirectory() as td:
            tensor_dir = Path(td)
            (tensor_dir / "b.pt").write_bytes(b"dummy")
            (tensor_dir / "manifest.json").write_text(
                '{"samples": {"bad": "shape"}}',
                encoding="utf-8",
            )

            ds = PreprocessedTensorDataset(str(tensor_dir))
            self.assertEqual(len(ds), 1)
            self.assertTrue(ds.manifest_fallback_used)
            self.assertIn("samples", ds.manifest_error or "")

    def test_dot_prefixed_windows_absolute_path_is_normalized(self) -> None:
        from acestep.training_v2._vendor.data_module import PreprocessedTensorDataset

        normalized = PreprocessedTensorDataset._normalize_manifest_path(
            ".C:\\Users\\kenna\\ACE-Step-1.5\\datasets\\sample.pt",
            tensor_dir=".",
        )
        assert normalized is not None
        self.assertTrue(normalized.startswith("C:"))

    def test_manifest_paths_invalid_falls_back_to_directory_scan(self) -> None:
        """When manifest loads but all paths don't exist (e.g. folder renamed), fall back to scan."""
        from acestep.training_v2._vendor.data_module import PreprocessedTensorDataset

        with tempfile.TemporaryDirectory() as td:
            tensor_dir = Path(td)
            (tensor_dir / "c.pt").write_bytes(b"dummy")
            # Valid JSON but paths point to non-existent location (e.g. ACE-Step-1.5 vs ACE-Step-1.5_)
            (tensor_dir / "manifest.json").write_text(
                '{"samples": ["/nonexistent/old_folder/c.pt"]}',
                encoding="utf-8",
            )

            ds = PreprocessedTensorDataset(str(tensor_dir))
            self.assertEqual(len(ds), 1)
            self.assertTrue(ds.manifest_fallback_used)
            self.assertFalse(ds.manifest_error)  # No parse error, just path fallback

    def test_tensor_dir_with_whitespace(self) -> None:
        """Path with leading/trailing whitespace is normalized and samples are found."""
        from acestep.training_v2._vendor.data_module import PreprocessedTensorDataset

        with tempfile.TemporaryDirectory() as td:
            tensor_dir = Path(td)
            (tensor_dir / "d.pt").write_bytes(b"dummy")

            ds = PreprocessedTensorDataset("  " + str(tensor_dir) + "  ")
            self.assertEqual(len(ds), 1)
            self.assertFalse(ds.tensor_dir.startswith(" ") or ds.tensor_dir.endswith(" "))


if __name__ == "__main__":
    unittest.main()
