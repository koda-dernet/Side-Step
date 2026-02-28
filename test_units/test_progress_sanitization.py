"""Tests for Infinity/NaN sanitization across the progress pipeline.

Covers:
1. ProgressWriter — must never emit non-standard JSON (Infinity, NaN)
2. _tail_progress sanitizer — defense in depth for existing files
3. TensorBoard launcher — skip_prompt vs interactive priority
"""

import json
import math
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest


# ---------------------------------------------------------------------------
# 1. ProgressWriter sanitization
# ---------------------------------------------------------------------------

class TestProgressWriterSanitization:
    """ProgressWriter must produce strictly valid JSON (RFC 7159)."""

    def test_infinity_replaced_with_null(self, tmp_path):
        from sidestep_engine.core.progress_writer import ProgressWriter

        pw = ProgressWriter(tmp_path, interval=0)
        pw.maybe_write(step=1, loss=0.5, best_loss=float("inf"))
        pw.close()

        line = (tmp_path / ".progress.jsonl").read_text().strip()
        data = json.loads(line)  # must not raise
        assert data["best_loss"] is None
        assert data["loss"] == 0.5

    def test_negative_infinity_replaced(self, tmp_path):
        from sidestep_engine.core.progress_writer import ProgressWriter

        pw = ProgressWriter(tmp_path, interval=0)
        pw.maybe_write(step=1, loss=float("-inf"))
        pw.close()

        line = (tmp_path / ".progress.jsonl").read_text().strip()
        data = json.loads(line)
        assert data["loss"] is None

    def test_nan_replaced_with_null(self, tmp_path):
        from sidestep_engine.core.progress_writer import ProgressWriter

        pw = ProgressWriter(tmp_path, interval=0)
        pw.maybe_write(step=1, loss=float("nan"), lr=1e-4)
        pw.close()

        line = (tmp_path / ".progress.jsonl").read_text().strip()
        data = json.loads(line)
        assert data["loss"] is None
        assert data["lr"] == 1e-4

    def test_normal_floats_preserved(self, tmp_path):
        from sidestep_engine.core.progress_writer import ProgressWriter

        pw = ProgressWriter(tmp_path, interval=0)
        pw.maybe_write(step=42, loss=0.1234, lr=3e-4, best_loss=0.0999)
        pw.close()

        line = (tmp_path / ".progress.jsonl").read_text().strip()
        data = json.loads(line)
        assert data["loss"] == pytest.approx(0.1234)
        assert data["lr"] == pytest.approx(3e-4)
        assert data["best_loss"] == pytest.approx(0.0999)
        assert data["step"] == 42

    def test_write_event_sanitizes(self, tmp_path):
        from sidestep_engine.core.progress_writer import ProgressWriter

        pw = ProgressWriter(tmp_path, interval=0)
        pw.write_event(kind="epoch", loss=0.5, best_loss=float("inf"))
        pw.close()

        line = (tmp_path / ".progress.jsonl").read_text().strip()
        data = json.loads(line)
        assert data["kind"] == "epoch"
        assert data["best_loss"] is None

    def test_output_is_valid_json_parse(self, tmp_path):
        """Every line must be parseable by strict JSON (no bare Infinity/NaN)."""
        from sidestep_engine.core.progress_writer import ProgressWriter

        pw = ProgressWriter(tmp_path, interval=0)
        # Write various problematic values
        pw.maybe_write(step=1, loss=float("inf"), lr=float("nan"), best_loss=float("-inf"))
        pw.write_event(kind="complete", loss=float("nan"))
        pw.close()

        for line in (tmp_path / ".progress.jsonl").read_text().strip().splitlines():
            # json.loads with parse_constant that rejects non-finite
            data = json.loads(line)
            # Verify no non-finite floats survived
            _assert_no_nonfinite(data)

    def test_allow_nan_false_enforced(self, tmp_path):
        """The file must be parseable by json.loads without Python's Infinity extension."""
        from sidestep_engine.core.progress_writer import ProgressWriter

        pw = ProgressWriter(tmp_path, interval=0)
        pw.maybe_write(step=1, best_loss=float("inf"))
        pw.close()

        raw = (tmp_path / ".progress.jsonl").read_text().strip()
        # This would fail if Infinity was written as a bare word
        assert "Infinity" not in raw
        assert "NaN" not in raw


# ---------------------------------------------------------------------------
# 2. _tail_progress sanitizer (defense in depth)
# ---------------------------------------------------------------------------

class TestTailProgressSanitizer:
    """_sanitize_floats in task_manager must catch Infinity/NaN from old files."""

    def test_sanitize_infinity(self):
        from sidestep_engine.gui.task_manager import _sanitize_floats

        data = {"loss": 0.5, "best_loss": float("inf"), "lr": 1e-4}
        clean = _sanitize_floats(data)
        assert clean["best_loss"] is None
        assert clean["loss"] == 0.5
        assert clean["lr"] == 1e-4

    def test_sanitize_nan(self):
        from sidestep_engine.gui.task_manager import _sanitize_floats

        data = {"loss": float("nan")}
        clean = _sanitize_floats(data)
        assert clean["loss"] is None

    def test_sanitize_nested(self):
        from sidestep_engine.gui.task_manager import _sanitize_floats

        data = {"outer": {"inner": float("inf")}, "list": [1.0, float("nan"), 3.0]}
        clean = _sanitize_floats(data)
        assert clean["outer"]["inner"] is None
        assert clean["list"] == [1.0, None, 3.0]

    def test_sanitize_preserves_non_float(self):
        from sidestep_engine.gui.task_manager import _sanitize_floats

        data = {"step": 42, "kind": "epoch", "loss": 0.5, "nested": {"a": "b"}}
        clean = _sanitize_floats(data)
        assert clean == data


# ---------------------------------------------------------------------------
# 3. TensorBoard launcher logic
# ---------------------------------------------------------------------------

class TestTensorBoardLaunchLogic:
    """should_launch_tensorboard must respect skip_prompt before interactive."""

    def test_skip_prompt_true_non_interactive_returns_default(self):
        from sidestep_engine.ui.tensorboard_launcher import should_launch_tensorboard

        # This is the GUI subprocess case: skip_prompt=True, stdin not a TTY
        result = should_launch_tensorboard(
            "/tmp/fake_log_dir",
            default=True,
            skip_prompt=True,
            interactive=False,
        )
        assert result is True

    def test_skip_prompt_true_default_false(self):
        from sidestep_engine.ui.tensorboard_launcher import should_launch_tensorboard

        result = should_launch_tensorboard(
            "/tmp/fake_log_dir",
            default=False,
            skip_prompt=True,
            interactive=False,
        )
        assert result is False

    def test_non_interactive_no_skip_returns_false(self):
        from sidestep_engine.ui.tensorboard_launcher import should_launch_tensorboard

        # CI/automation case: no skip, no TTY
        result = should_launch_tensorboard(
            "/tmp/fake_log_dir",
            default=True,
            skip_prompt=False,
            interactive=False,
        )
        assert result is False

    def test_skip_prompt_takes_priority_over_interactive_none(self):
        from sidestep_engine.ui.tensorboard_launcher import should_launch_tensorboard

        # interactive=None would normally check isatty()
        result = should_launch_tensorboard(
            "/tmp/fake_log_dir",
            default=True,
            skip_prompt=True,
            interactive=None,
        )
        assert result is True


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _assert_no_nonfinite(obj):
    """Recursively assert no non-finite floats in a parsed JSON structure."""
    if isinstance(obj, float):
        assert math.isfinite(obj), f"Non-finite float found: {obj}"
    elif isinstance(obj, dict):
        for v in obj.values():
            _assert_no_nonfinite(v)
    elif isinstance(obj, list):
        for v in obj:
            _assert_no_nonfinite(v)
