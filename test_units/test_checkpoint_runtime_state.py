"""Tests for checkpoint runtime state persistence.

Verifies that save_checkpoint correctly persists all runtime state fields
(including ema_state and adaptive_sampler_state) and that
_extract_runtime_state correctly extracts them on load.
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import torch

from sidestep_engine.core.trainer_helpers import (
    _extract_runtime_state,
    save_checkpoint,
)


class TestExtractRuntimeState(unittest.TestCase):
    """_extract_runtime_state should extract all known runtime keys."""

    def test_extracts_ema_state(self) -> None:
        state = {"ema_state": {"decay": 0.999, "shadow": []}}
        rt = _extract_runtime_state(state)
        self.assertIn("ema_state", rt)
        self.assertEqual(rt["ema_state"]["decay"], 0.999)

    def test_extracts_adaptive_sampler_state(self) -> None:
        state = {"adaptive_sampler_state": {"bin_loss": torch.ones(10)}}
        rt = _extract_runtime_state(state)
        self.assertIn("adaptive_sampler_state", rt)

    def test_extracts_all_known_keys(self) -> None:
        state = {
            "rng_state": {"python": None},
            "tracker_state": {"best_loss": 0.1},
            "chunk_coverage_state": {},
            "config_meta": {"optimizer_type": "adamw"},
            "ema_state": {"decay": 0.99},
            "adaptive_sampler_state": {"bin_loss": torch.ones(5)},
        }
        rt = _extract_runtime_state(state)
        for key in state:
            self.assertIn(key, rt)

    def test_ignores_unknown_keys(self) -> None:
        state = {"rng_state": {}, "some_future_key": "value"}
        rt = _extract_runtime_state(state)
        self.assertIn("rng_state", rt)
        self.assertNotIn("some_future_key", rt)

    def test_empty_state_logs_info(self) -> None:
        with self.assertLogs("sidestep_engine.core.trainer_helpers", level="INFO"):
            rt = _extract_runtime_state({})
        self.assertEqual(rt, {})


class TestSaveCheckpointRuntimeState(unittest.TestCase):
    """save_checkpoint should persist ema_state and adaptive_sampler_state."""

    def _make_mock_trainer(self, tmp_dir: str) -> MagicMock:
        trainer = MagicMock()
        trainer.adapter_type = "lora"
        trainer.module = MagicMock()
        trainer.training_config = MagicMock()
        trainer.training_config.optimizer_type = "adamw"
        trainer.training_config.scheduler_type = "cosine"
        trainer.training_config.scheduler_formula = ""
        trainer.training_config.warmup_steps = 0
        trainer.training_config.learning_rate = 1e-4
        trainer.training_config._checkpoint_total_steps = 100
        return trainer

    @patch("sidestep_engine.core.trainer_helpers.save_adapter_flat")
    def test_ema_state_persisted(self, mock_save_flat: MagicMock) -> None:
        """Regression: ema_state was silently dropped from checkpoints."""
        with tempfile.TemporaryDirectory() as td:
            trainer = self._make_mock_trainer(td)
            optimizer = MagicMock()
            optimizer.state_dict.return_value = {}
            scheduler = MagicMock()
            scheduler.state_dict.return_value = {}

            ema_state = {"decay": 0.999, "step_count": 42, "shadow": [torch.ones(4)]}
            runtime = {
                "rng_state": {},
                "ema_state": ema_state,
            }

            save_checkpoint(trainer, optimizer, scheduler, 1, 100, td, runtime)

            state_path = Path(td) / "training_state.pt"
            self.assertTrue(state_path.exists())
            loaded = torch.load(str(state_path), weights_only=False)
            self.assertIn("ema_state", loaded)
            self.assertEqual(loaded["ema_state"]["decay"], 0.999)
            self.assertEqual(loaded["ema_state"]["step_count"], 42)

    @patch("sidestep_engine.core.trainer_helpers.save_adapter_flat")
    def test_adaptive_sampler_state_persisted(self, mock_save_flat: MagicMock) -> None:
        """Regression: adaptive_sampler_state was silently dropped."""
        with tempfile.TemporaryDirectory() as td:
            trainer = self._make_mock_trainer(td)
            optimizer = MagicMock()
            optimizer.state_dict.return_value = {}
            scheduler = MagicMock()
            scheduler.state_dict.return_value = {}

            sampler_state = {"bin_loss": torch.ones(10), "total_updates": 5}
            runtime = {
                "adaptive_sampler_state": sampler_state,
            }

            save_checkpoint(trainer, optimizer, scheduler, 1, 50, td, runtime)

            loaded = torch.load(
                str(Path(td) / "training_state.pt"), weights_only=False,
            )
            self.assertIn("adaptive_sampler_state", loaded)
            self.assertEqual(loaded["adaptive_sampler_state"]["total_updates"], 5)


if __name__ == "__main__":
    unittest.main()
