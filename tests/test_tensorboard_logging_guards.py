"""Tests for TensorBoard heavy logging guards."""

from __future__ import annotations

import unittest

import torch
import torch.nn as nn

from acestep.training_v2.tensorboard_utils import TrainingLogger


class _Tiny(nn.Module):
    def __init__(self):
        super().__init__()
        self.w = nn.Parameter(torch.ones(4, requires_grad=True))


class TestTensorboardHeavyLoggingGuards(unittest.TestCase):
    """Ensure heavy logging is a no-op when writer is unavailable."""

    def test_log_per_layer_returns_empty_without_writer(self):
        model = _Tiny()
        model.w.grad = torch.ones_like(model.w)
        logger = TrainingLogger(log_dir="/tmp/unused", enabled=False)
        values = logger.log_per_layer_grad_norms(model, step=1)
        self.assertEqual(values, {})


if __name__ == "__main__":
    unittest.main()

