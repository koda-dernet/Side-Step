"""Tests for sidestep_engine.core.validation — run_validation_epoch."""

import unittest
from unittest.mock import MagicMock

import torch

from sidestep_engine.core.validation import run_validation_epoch


def _make_batch():
    """Return a minimal dummy batch dict."""
    return {"target_latents": torch.zeros(1)}


class TestRunValidationEpoch(unittest.TestCase):
    """Validation pass returns correct mean loss."""

    def test_mean_of_two_batches(self):
        module = MagicMock()
        module.training_step = MagicMock(side_effect=[
            torch.tensor(0.5), torch.tensor(1.5),
        ])
        loader = [_make_batch(), _make_batch()]
        result = run_validation_epoch(module, loader, torch.device("cpu"))
        self.assertAlmostEqual(result, 1.0, places=5)

    def test_empty_loader_returns_inf(self):
        module = MagicMock()
        result = run_validation_epoch(module, [], torch.device("cpu"))
        self.assertEqual(result, float("inf"))

    def test_nan_batches_skipped(self):
        module = MagicMock()
        module.training_step = MagicMock(side_effect=[
            torch.tensor(float("nan")),
            torch.tensor(2.0),
        ])
        loader = [_make_batch(), _make_batch()]
        result = run_validation_epoch(module, loader, torch.device("cpu"))
        self.assertAlmostEqual(result, 2.0, places=5)

    def test_all_nan_returns_inf(self):
        module = MagicMock()
        module.training_step = MagicMock(return_value=torch.tensor(float("nan")))
        loader = [_make_batch()]
        result = run_validation_epoch(module, loader, torch.device("cpu"))
        self.assertEqual(result, float("inf"))

    def test_exception_in_batch_skipped(self):
        module = MagicMock()
        call_count = [0]
        def _side_effect(batch):
            call_count[0] += 1
            if call_count[0] == 1:
                raise RuntimeError("boom")
            return torch.tensor(3.0)
        module.training_step = _side_effect
        loader = [_make_batch(), _make_batch()]
        result = run_validation_epoch(module, loader, torch.device("cpu"))
        self.assertAlmostEqual(result, 3.0, places=5)

    def test_decoder_eval_mode_restored(self):
        module = MagicMock()
        module.training = True
        decoder = MagicMock()
        module.model.decoder = decoder
        module.training_step = MagicMock(return_value=torch.tensor(1.0))
        loader = [_make_batch()]
        run_validation_epoch(module, loader, torch.device("cpu"))
        decoder.eval.assert_called_once()
        decoder.train.assert_called_once()


if __name__ == "__main__":
    unittest.main()
