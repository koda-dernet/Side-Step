"""Tests for the progress file writer used by the GUI telemetry pipeline."""

from __future__ import annotations

import json
import time
from pathlib import Path

import pytest

from sidestep_engine.core.progress_writer import ProgressWriter


class TestProgressWriter:

    def test_creates_file(self, tmp_path):
        pw = ProgressWriter(tmp_path, interval=0)
        pw.maybe_write(step=1, loss=0.5)
        pw.close()
        assert (tmp_path / ".progress.jsonl").exists()

    def test_writes_jsonl_lines(self, tmp_path):
        pw = ProgressWriter(tmp_path, interval=0)
        pw.maybe_write(step=1, loss=0.5)
        pw.maybe_write(step=2, loss=0.4)
        pw.close()
        lines = (tmp_path / ".progress.jsonl").read_text().strip().splitlines()
        assert len(lines) == 2
        data = json.loads(lines[0])
        assert data["step"] == 1
        assert data["loss"] == 0.5
        assert data["kind"] == "step"
        assert "ts" in data

    def test_time_gating(self, tmp_path):
        """With interval=10s, only the first write should go through."""
        pw = ProgressWriter(tmp_path, interval=10)
        pw.maybe_write(step=1, loss=0.5)
        pw.maybe_write(step=2, loss=0.4)
        pw.maybe_write(step=3, loss=0.3)
        pw.close()
        lines = (tmp_path / ".progress.jsonl").read_text().strip().splitlines()
        assert len(lines) == 1  # only first write passes the gate

    def test_write_event_bypasses_throttle(self, tmp_path):
        """write_event should always write regardless of interval."""
        pw = ProgressWriter(tmp_path, interval=10)
        pw.maybe_write(step=1, loss=0.5)  # consumes the gate
        pw.write_event(kind="epoch", step=10, epoch=1)
        pw.write_event(kind="complete", step=100)
        pw.close()
        lines = (tmp_path / ".progress.jsonl").read_text().strip().splitlines()
        assert len(lines) == 3

    def test_close_idempotent(self, tmp_path):
        pw = ProgressWriter(tmp_path, interval=0)
        pw.maybe_write(step=1, loss=0.1)
        pw.close()
        pw.close()  # should not raise

    def test_event_kinds(self, tmp_path):
        pw = ProgressWriter(tmp_path, interval=0)
        pw.write_event(kind="epoch", step=50, epoch=5, max_epochs=100)
        pw.write_event(kind="complete", step=100, loss=0.01)
        pw.write_event(kind="fail", step=0, msg="0 steps completed")
        pw.close()
        lines = (tmp_path / ".progress.jsonl").read_text().strip().splitlines()
        kinds = [json.loads(l)["kind"] for l in lines]
        assert kinds == ["epoch", "complete", "fail"]

    def test_creates_parent_dirs(self, tmp_path):
        nested = tmp_path / "a" / "b" / "c"
        pw = ProgressWriter(nested, interval=0)
        pw.maybe_write(step=1, loss=0.5)
        pw.close()
        assert (nested / ".progress.jsonl").exists()
