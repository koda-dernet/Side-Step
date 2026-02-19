"""Tests for wizard dataset preflight checks."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path


class TestWizardDatasetPreflight(unittest.TestCase):
    """Validate early detection of malformed preprocessed dataset dirs."""

    def test_accepts_directory_with_pt_files(self) -> None:
        from acestep.training_v2.ui.flows_common import describe_preprocessed_dataset_issue

        with tempfile.TemporaryDirectory() as td:
            d = Path(td)
            (d / "x.pt").write_bytes(b"dummy")
            self.assertIsNone(describe_preprocessed_dataset_issue(str(d)))

    def test_flags_invalid_manifest_json(self) -> None:
        from acestep.training_v2.ui.flows_common import describe_preprocessed_dataset_issue

        with tempfile.TemporaryDirectory() as td:
            d = Path(td)
            (d / "manifest.json").write_text(
                '{\n  "samples": [".C:\\Users\\bad\\x.pt"]\n}\n',
                encoding="utf-8",
            )
            issue = describe_preprocessed_dataset_issue(str(d))
            assert issue is not None
            self.assertIn("invalid JSON", issue)

    def test_flags_missing_pt_and_manifest(self) -> None:
        from acestep.training_v2.ui.flows_common import describe_preprocessed_dataset_issue

        with tempfile.TemporaryDirectory() as td:
            issue = describe_preprocessed_dataset_issue(td)
            assert issue is not None
            self.assertIn("manifest.json is missing", issue)


if __name__ == "__main__":
    unittest.main()
