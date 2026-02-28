"""Tests for TensorBoard logging fixes.

Covers:
- Timestep buffer accumulation and drain in FixedLoRAModule.
- Gradient norm logging ordering in the basic training loop.
"""

from __future__ import annotations

import unittest
from collections import deque
from unittest.mock import MagicMock, patch

import torch


class TestTimestepBuffer(unittest.TestCase):
    """Verify the rolling timestep buffer in FixedLoRAModule."""

    def _make_module(self, is_turbo=False):
        """Create a FixedLoRAModule with mocked adapter injection."""
        from sidestep_engine.core.lora_module import FixedLoRAModule
        from sidestep_engine.core.configs import LoRAConfigV2, TrainingConfigV2

        model = MagicMock()
        model.config = MagicMock()
        model.null_condition_emb = torch.randn(1, 1, 128)
        model.decoder = MagicMock()
        model.decoder.return_value = (torch.randn(2, 10, 64),)

        lora_cfg = LoRAConfigV2(r=8, alpha=16)
        train_cfg = TrainingConfigV2(
            is_turbo=is_turbo,
            output_dir="/tmp/test",
        )

        with patch.object(FixedLoRAModule, "_inject_lora"):
            module = FixedLoRAModule(model, lora_cfg, train_cfg, "cpu", torch.float32)
            module.model = model

        return module

    def _make_batch(self, bsz=2):
        return {
            "target_latents": torch.randn(bsz, 10, 64),
            "attention_mask": torch.ones(bsz, 10),
            "encoder_hidden_states": torch.randn(bsz, 5, 128),
            "encoder_attention_mask": torch.ones(bsz, 5),
            "context_latents": torch.randn(bsz, 10, 128),
        }

    def test_drain_empty_returns_none(self):
        """drain_timestep_buffer() returns None on a fresh module."""
        module = self._make_module()
        self.assertIsNone(module.drain_timestep_buffer())

    @patch("sidestep_engine.core.lora_module.sample_timesteps")
    def test_single_step_accumulates(self, mock_sample):
        """One training step should add one entry to the buffer."""
        mock_sample.return_value = (
            torch.tensor([0.3, 0.7]),
            torch.tensor([0.3, 0.7]),
        )
        module = self._make_module(is_turbo=False)
        module.training_step(self._make_batch())

        buf = module.drain_timestep_buffer()
        self.assertIsNotNone(buf)
        self.assertEqual(buf.shape, (2,))
        self.assertTrue(buf.device == torch.device("cpu"))

    @patch("sidestep_engine.core.lora_module.sample_timesteps")
    def test_multiple_steps_accumulate(self, mock_sample):
        """Multiple training steps should concatenate in the buffer."""
        mock_sample.return_value = (
            torch.tensor([0.4, 0.6]),
            torch.tensor([0.4, 0.6]),
        )
        module = self._make_module(is_turbo=False)

        for _ in range(5):
            module.training_step(self._make_batch())

        buf = module.drain_timestep_buffer()
        self.assertIsNotNone(buf)
        self.assertEqual(buf.shape, (10,))  # 5 steps * batch_size 2

    @patch("sidestep_engine.core.lora_module.sample_timesteps")
    def test_drain_clears_buffer(self, mock_sample):
        """drain_timestep_buffer() must clear the buffer after draining."""
        mock_sample.return_value = (
            torch.tensor([0.5]),
            torch.tensor([0.5]),
        )
        module = self._make_module(is_turbo=False)
        module.training_step(self._make_batch(bsz=1))

        first = module.drain_timestep_buffer()
        self.assertIsNotNone(first)

        second = module.drain_timestep_buffer()
        self.assertIsNone(second)

    @patch("sidestep_engine.core.lora_module.sample_timesteps")
    def test_buffer_respects_maxlen(self, mock_sample):
        """Buffer is capped at 100 entries to bound memory."""
        mock_sample.return_value = (
            torch.tensor([0.5]),
            torch.tensor([0.5]),
        )
        module = self._make_module(is_turbo=False)

        for _ in range(150):
            module.training_step(self._make_batch(bsz=1))

        self.assertEqual(len(module._timestep_buffer), 100)
        buf = module.drain_timestep_buffer()
        self.assertEqual(buf.shape, (100,))

    @patch("sidestep_engine.core.lora_module.sample_timesteps")
    def test_buffer_stores_cpu_tensors(self, mock_sample):
        """All buffered tensors must be on CPU to avoid GPU memory leak."""
        mock_sample.return_value = (
            torch.tensor([0.3, 0.7]),
            torch.tensor([0.3, 0.7]),
        )
        module = self._make_module(is_turbo=False)
        module.training_step(self._make_batch())

        for t in module._timestep_buffer:
            self.assertEqual(t.device, torch.device("cpu"))


class TestBasicLoopGradNormOrdering(unittest.TestCase):
    """Verify _flush_accumulated logs grad norms before zeroing."""

    def test_grad_norms_logged_before_zero_grad(self):
        """Grad norms must be non-empty when log_heavy_every fires."""
        from sidestep_engine.core.trainer_loop import _flush_accumulated

        param = torch.nn.Parameter(torch.randn(4, 4))
        param.grad = torch.randn(4, 4)
        trainable_params = [param]

        optimizer = MagicMock()
        scheduler = MagicMock()
        scheduler.get_last_lr.return_value = [1e-4]

        cfg = MagicMock()
        cfg.max_grad_norm = 1.0
        cfg.gradient_accumulation_steps = 1
        cfg.log_every = 10
        cfg.log_heavy_every = 1  # fires every step
        cfg.max_epochs = 1

        tb = MagicMock()
        logged_norms = {}

        def capture_grad_norms(model, step):
            for name, p in model.named_parameters():
                if p.requires_grad and p.grad is not None:
                    logged_norms[name] = p.grad.data.norm(2).item()

        tb.log_per_layer_grad_norms.side_effect = capture_grad_norms

        module = MagicMock()
        module.model = torch.nn.Linear(4, 4)
        module.model.weight = param
        module.drain_timestep_buffer.return_value = None

        global_step, _, _ = _flush_accumulated(
            trainable_params, optimizer, scheduler,
            accumulated_loss=0.5, accumulation_step=1,
            cfg=cfg, tb=tb, module=module,
            epoch=0, global_step=0, steps_per_epoch=10,
        )

        tb.log_per_layer_grad_norms.assert_called_once()
        tb.flush.assert_called_once()

    def test_zero_grad_called_after_logging(self):
        """optimizer.zero_grad must be called after heavy logging."""
        from sidestep_engine.core.trainer_loop import _flush_accumulated

        call_order = []

        optimizer = MagicMock()
        optimizer.zero_grad.side_effect = lambda **kw: call_order.append("zero_grad")

        scheduler = MagicMock()
        scheduler.get_last_lr.return_value = [1e-4]

        cfg = MagicMock()
        cfg.max_grad_norm = 1.0
        cfg.gradient_accumulation_steps = 1
        cfg.log_every = 10
        cfg.log_heavy_every = 1
        cfg.max_epochs = 1

        tb = MagicMock()
        tb.log_per_layer_grad_norms.side_effect = lambda *a: call_order.append("log_grad_norms")
        tb.flush.side_effect = lambda: call_order.append("flush")

        module = MagicMock()
        module.drain_timestep_buffer.return_value = None

        _flush_accumulated(
            [torch.nn.Parameter(torch.randn(2, 2))],
            optimizer, scheduler,
            accumulated_loss=0.5, accumulation_step=1,
            cfg=cfg, tb=tb, module=module,
            epoch=0, global_step=0, steps_per_epoch=10,
        )

        self.assertIn("log_grad_norms", call_order)
        self.assertIn("flush", call_order)
        self.assertIn("zero_grad", call_order)

        grad_idx = call_order.index("log_grad_norms")
        flush_idx = call_order.index("flush")
        zero_idx = call_order.index("zero_grad")
        self.assertLess(grad_idx, zero_idx, "grad norms must be logged before zero_grad")
        self.assertLess(flush_idx, zero_idx, "flush must happen before zero_grad")


if __name__ == "__main__":
    unittest.main()
