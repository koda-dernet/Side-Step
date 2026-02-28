"""Tests for Rich live progress log normalization/truncation."""

from __future__ import annotations

from types import SimpleNamespace
import unittest
from unittest.mock import patch

from rich.console import Console

from sidestep_engine.ui.progress import (
    _LOG_HISTORY,
    _append_recent_message,
    _build_display,
    _normalize_live_log_message,
    _truncate_for_log_panel,
    TrainingStats,
)


class TestProgressLiveLogSafety(unittest.TestCase):
    def test_normalize_collapses_multiline_messages(self) -> None:
        msg = "[WARN] first line\n\n second   line\twith   spaces "
        normalized = _normalize_live_log_message(msg)
        self.assertEqual(normalized, "[WARN] first line | second line with spaces")

    def test_append_dedupes_adjacent_messages(self) -> None:
        recent: list[str] = []
        _append_recent_message(recent, "[INFO] hello")
        _append_recent_message(recent, "[INFO] hello")
        self.assertEqual(recent, ["[INFO] hello"])

    def test_append_caps_history(self) -> None:
        recent: list[str] = []
        for i in range(_LOG_HISTORY + 5):
            _append_recent_message(recent, f"[INFO] msg {i}")
        self.assertEqual(len(recent), _LOG_HISTORY)
        self.assertEqual(recent[0], "[INFO] msg 5")

    def test_truncate_adds_ellipsis(self) -> None:
        text = "abcdefghijklmnopqrstuvwxyz"
        self.assertEqual(_truncate_for_log_panel(text, 10), "abcdefg...")
        self.assertEqual(_truncate_for_log_panel(text, 30), text)

    def test_progress_bar_renders_not_repr_string(self) -> None:
        class _FakeGPU:
            available = True

            @staticmethod
            def snapshot():
                return SimpleNamespace(percent=29.0, used_gb=4.5, total_gb=15.6)

        stats = TrainingStats(start_time=0.0, max_epochs=700, current_epoch=0, current_step=0)
        test_console = Console(record=True, width=120)
        with patch("sidestep_engine.ui.progress.console", test_console):
            test_console.print(_build_display(stats, _FakeGPU(), ["[INFO] Scheduler: cosine"]))
        rendered = test_console.export_text()
        self.assertNotIn("<Bar of", rendered)


if __name__ == "__main__":
    unittest.main()
