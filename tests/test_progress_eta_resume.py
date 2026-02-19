"""Tests for resume-safe ETA calculation in Rich progress tracking."""

from __future__ import annotations

import unittest
from unittest.mock import patch

from acestep.training_v2.ui import TrainingUpdate
from acestep.training_v2.ui.progress import (
    TrainingStats,
    _process_structured,
    _process_tuple,
)


class TestProgressEtaResume(unittest.TestCase):
    def test_resume_eta_uses_session_epoch_baseline(self) -> None:
        stats = TrainingStats(start_time=100.0, max_epochs=1000)
        update = TrainingUpdate(
            step=805,
            loss=0.1234,
            msg="Epoch 115/1000, Step 805, Loss: 0.1234",
            kind="step",
            epoch=115,
            max_epochs=1000,
        )

        with patch("acestep.training_v2.ui.progress.time.time", return_value=370.0):
            _process_structured(update, stats)
            eta = stats.eta_seconds

        self.assertEqual(stats.session_start_epoch, 114)
        expected = ((370.0 - 100.0) / 1.0) * (1000 - 115)
        self.assertAlmostEqual(eta, expected, places=6)

    def test_resume_metadata_overrides_inferred_baseline(self) -> None:
        stats = TrainingStats(start_time=100.0, max_epochs=1000)

        resume_info = TrainingUpdate(
            step=0,
            loss=0.0,
            msg="[OK] Resumed from epoch 100, step 700",
            kind="info",
            resume_start_epoch=100,
        )
        _process_structured(resume_info, stats)

        step_update = TrainingUpdate(
            step=805,
            loss=0.1234,
            msg="Epoch 115/1000, Step 805, Loss: 0.1234",
            kind="step",
            epoch=115,
            max_epochs=1000,
        )

        with patch("acestep.training_v2.ui.progress.time.time", return_value=370.0):
            _process_structured(step_update, stats)
            eta = stats.eta_seconds

        self.assertEqual(stats.session_start_epoch, 100)
        expected = ((370.0 - 100.0) / 15.0) * (1000 - 115)
        self.assertAlmostEqual(eta, expected, places=6)

    def test_eta_prefers_last_epoch_time_when_available(self) -> None:
        stats = TrainingStats(start_time=100.0, max_epochs=1000)
        update = TrainingUpdate(
            step=900,
            loss=0.1111,
            msg="[OK] Epoch 116/1000 in 210.0s, Loss: 0.1111",
            kind="epoch",
            epoch=116,
            max_epochs=1000,
            epoch_time=210.0,
        )

        with patch("acestep.training_v2.ui.progress.time.time", return_value=500.0):
            _process_structured(update, stats)
            eta = stats.eta_seconds

        self.assertEqual(eta, 210.0 * (1000 - 116))

    def test_tuple_updates_capture_session_baseline(self) -> None:
        stats = TrainingStats(start_time=10.0)
        msg = "Epoch 116/1000, Step 12, Loss: 0.2000"

        with patch("acestep.training_v2.ui.progress.time.time", return_value=310.0):
            _process_tuple(step=12, loss=0.2, msg=msg, stats=stats)
            eta = stats.eta_seconds

        self.assertEqual(stats.session_start_epoch, 115)
        expected = ((310.0 - 10.0) / 1.0) * (1000 - 116)
        self.assertAlmostEqual(eta, expected, places=6)

    def test_eta_zero_after_last_epoch(self) -> None:
        stats = TrainingStats(
            start_time=100.0,
            max_epochs=10,
            current_epoch=10,
            last_epoch_time=20.0,
        )
        self.assertEqual(stats.eta_seconds, 0.0)


if __name__ == "__main__":
    unittest.main()
