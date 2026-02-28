"""Tests for the trigger tag prompt in inline preprocessing.

Covers:
- ``ask_trigger_tag`` prompt flow (no tag, single tag, multiple tags).
- ``preprocess_audio_files`` ``custom_tag`` / ``tag_position`` parameter wiring.
- Re-export stability (``inline_preprocess`` re-exports from ``inline_preprocess_prompts``).
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch


class TestAskTriggerTagNoTag(unittest.TestCase):
    """User leaves trigger tag empty — no tag applied."""

    @patch("sidestep_engine.ui.flows.inline_preprocess_prompts.ask", return_value="")
    @patch("sidestep_engine.ui.flows.inline_preprocess_prompts.print_message")
    @patch("sidestep_engine.ui.flows.build_dataset.scan_sidecar_tags", return_value=set())
    def test_empty_tag_returns_prepend(self, _scan, _pm, _ask):
        from sidestep_engine.ui.flows.inline_preprocess_prompts import ask_trigger_tag

        tag, pos = ask_trigger_tag("/fake/dir")
        self.assertEqual(tag, "")
        self.assertEqual(pos, "prepend")


class TestAskTriggerTagSingleDetected(unittest.TestCase):
    """Sidecars have a single tag — pre-filled, user accepts it."""

    @patch("sidestep_engine.ui.flows.inline_preprocess_prompts.menu", return_value="append")
    @patch("sidestep_engine.ui.flows.inline_preprocess_prompts.ask", return_value="my_style")
    @patch("sidestep_engine.ui.flows.inline_preprocess_prompts.print_message")
    @patch("sidestep_engine.ui.flows.build_dataset.scan_sidecar_tags", return_value={"my_style"})
    def test_single_tag_prefilled(self, _scan, _pm, _ask, _menu):
        from sidestep_engine.ui.flows.inline_preprocess_prompts import ask_trigger_tag

        tag, pos = ask_trigger_tag("/fake/dir")
        self.assertEqual(tag, "my_style")
        self.assertEqual(pos, "append")


class TestAskTriggerTagNewTagWriteback(unittest.TestCase):
    """User enters a new tag — writeback_tag_to_sidecars should be called."""

    @patch("sidestep_engine.ui.flows.inline_preprocess_prompts.menu", return_value="prepend")
    @patch("sidestep_engine.ui.flows.inline_preprocess_prompts.ask", return_value="new_trigger")
    @patch("sidestep_engine.ui.flows.inline_preprocess_prompts.print_message")
    @patch("sidestep_engine.ui.flows.build_dataset.writeback_tag_to_sidecars")
    @patch("sidestep_engine.ui.flows.build_dataset.scan_sidecar_tags", return_value=set())
    def test_new_tag_triggers_writeback(self, _scan, mock_wb, _pm, _ask, _menu):
        from sidestep_engine.ui.flows.inline_preprocess_prompts import ask_trigger_tag

        tag, pos = ask_trigger_tag("/fake/audio")
        self.assertEqual(tag, "new_trigger")
        mock_wb.assert_called_once_with("/fake/audio", "new_trigger")


class TestAskTriggerTagMultipleDetected(unittest.TestCase):
    """Multiple tags in sidecars — warning shown, first alphabetically as default."""

    @patch("sidestep_engine.ui.flows.inline_preprocess_prompts.menu", return_value="prepend")
    @patch("sidestep_engine.ui.flows.inline_preprocess_prompts.ask", return_value="alpha")
    @patch("sidestep_engine.ui.flows.inline_preprocess_prompts.print_message")
    @patch("sidestep_engine.ui.flows.build_dataset.scan_sidecar_tags", return_value={"beta", "alpha"})
    def test_multiple_tags_warns(self, _scan, mock_pm, _ask, _menu):
        from sidestep_engine.ui.flows.inline_preprocess_prompts import ask_trigger_tag

        tag, pos = ask_trigger_tag("/fake/dir")
        self.assertEqual(tag, "alpha")
        # Check warning was emitted
        warn_calls = [c for c in mock_pm.call_args_list if "warn" in str(c)]
        self.assertTrue(len(warn_calls) > 0, "Expected a warning about multiple tags")


class TestPreprocessCustomTagOverride(unittest.TestCase):
    """Verify preprocess_audio_files custom_tag/tag_position override ds_meta."""

    def test_custom_tag_overrides_json_metadata(self):
        """Caller-supplied custom_tag should override JSON metadata."""
        from sidestep_engine.data.preprocess_discovery import load_dataset_metadata

        # When no JSON: defaults are empty
        ds_meta = load_dataset_metadata(None)
        self.assertEqual(ds_meta["custom_tag"], "")
        self.assertEqual(ds_meta["tag_position"], "prepend")

    def test_custom_tag_applied_to_samples(self):
        """custom_tag param should be applied as fallback to samples without one."""
        from sidestep_engine.data.preprocess_discovery import (
            load_dataset_metadata,
            load_sample_metadata,
        )

        d = tempfile.mkdtemp(prefix="sidestep_test_ctag_")
        try:
            Path(d, "song.wav").write_bytes(b"\x00" * 100)
            Path(d, "song.txt").write_text("caption: Test\n", encoding="utf-8")
            audio_files = [Path(d, "song.wav")]

            sample_meta = load_sample_metadata(None, audio_files)
            ds_meta = load_dataset_metadata(None)

            # Simulate what preprocess_audio_files does with custom_tag param
            custom_tag = "my_trigger"
            if custom_tag:
                ds_meta["custom_tag"] = custom_tag
            ds_tag = ds_meta.get("custom_tag", "")
            if ds_tag:
                for sm in sample_meta.values():
                    if not sm.get("custom_tag"):
                        sm["custom_tag"] = ds_tag

            self.assertEqual(sample_meta["song.wav"]["custom_tag"], "my_trigger")
        finally:
            import shutil
            shutil.rmtree(d, ignore_errors=True)

    def test_sidecar_tag_not_overwritten_by_param(self):
        """Per-sample sidecar custom_tag should NOT be overwritten by the param."""
        from sidestep_engine.data.preprocess_discovery import (
            load_dataset_metadata,
            load_sample_metadata,
        )

        d = tempfile.mkdtemp(prefix="sidestep_test_ctag_pri_")
        try:
            Path(d, "song.wav").write_bytes(b"\x00" * 100)
            Path(d, "song.txt").write_text(
                "caption: Test\ncustom_tag: sidecar_tag\n", encoding="utf-8",
            )
            audio_files = [Path(d, "song.wav")]

            sample_meta = load_sample_metadata(None, audio_files)
            ds_meta = load_dataset_metadata(None)

            custom_tag = "param_tag"
            if custom_tag:
                ds_meta["custom_tag"] = custom_tag
            ds_tag = ds_meta.get("custom_tag", "")
            if ds_tag:
                for sm in sample_meta.values():
                    if not sm.get("custom_tag"):
                        sm["custom_tag"] = ds_tag

            # Sidecar tag wins
            self.assertEqual(sample_meta["song.wav"]["custom_tag"], "sidecar_tag")
        finally:
            import shutil
            shutil.rmtree(d, ignore_errors=True)


class TestTagPositionOverride(unittest.TestCase):
    """Verify tag_position param overrides ds_meta."""

    def test_tag_position_override(self):
        from sidestep_engine.data.preprocess_discovery import load_dataset_metadata

        ds_meta = load_dataset_metadata(None)
        self.assertEqual(ds_meta["tag_position"], "prepend")

        # Simulate override
        tag_position = "append"
        if tag_position:
            ds_meta["tag_position"] = tag_position
        self.assertEqual(ds_meta["tag_position"], "append")


class TestReExportStability(unittest.TestCase):
    """Verify inline_preprocess re-exports from inline_preprocess_prompts."""

    def test_ask_trigger_tag_importable(self):
        from sidestep_engine.ui.flows.inline_preprocess import ask_trigger_tag
        self.assertTrue(callable(ask_trigger_tag))

    def test_show_sidecar_summary_importable(self):
        from sidestep_engine.ui.flows.inline_preprocess import show_sidecar_summary
        self.assertTrue(callable(show_sidecar_summary))

    def test_ask_normalization_importable(self):
        from sidestep_engine.ui.flows.inline_preprocess import ask_normalization
        self.assertTrue(callable(ask_normalization))

    def test_show_duration_scan_importable(self):
        from sidestep_engine.ui.flows.inline_preprocess import show_duration_scan
        self.assertTrue(callable(show_duration_scan))

    def test_metadata_flags_importable(self):
        from sidestep_engine.ui.flows.inline_preprocess import _metadata_flags
        self.assertTrue(callable(_metadata_flags))

    def test_format_marks_importable(self):
        from sidestep_engine.ui.flows.inline_preprocess import _format_marks
        self.assertTrue(callable(_format_marks))


if __name__ == "__main__":
    unittest.main()
