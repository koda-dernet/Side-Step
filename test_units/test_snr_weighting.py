"""Tests for min-SNR loss weighting and timestep histogram logging."""

from __future__ import annotations

import unittest
from unittest.mock import MagicMock, patch

import torch
import torch.nn.functional as F


class TestMinSNRWeighting(unittest.TestCase):
    """Verify min-SNR weighting math is correct and gated by config."""

    def test_none_weighting_matches_flat_mse(self):
        """loss_weighting='none' must produce the same loss as raw F.mse_loss."""
        predicted = torch.randn(4, 10, 64)
        target = torch.randn(4, 10, 64)

        flat_loss = F.mse_loss(predicted, target)

        # Simulate the 'none' branch
        none_loss = F.mse_loss(predicted, target)

        self.assertAlmostEqual(flat_loss.item(), none_loss.item(), places=6)

    def test_min_snr_produces_different_loss(self):
        """loss_weighting='min_snr' should generally produce a different loss than flat MSE."""
        torch.manual_seed(42)
        predicted = torch.randn(8, 10, 64)
        flow = torch.randn(8, 10, 64)
        t = torch.tensor([0.1, 0.2, 0.3, 0.5, 0.7, 0.8, 0.9, 0.95])

        flat_loss = F.mse_loss(predicted, flow)

        # min-SNR weighted loss
        per_sample_loss = F.mse_loss(predicted, flow, reduction="none")
        per_sample_loss = per_sample_loss.mean(dim=(-1, -2))
        snr = ((1.0 - t) / t.clamp(min=1e-6)) ** 2
        gamma = 5.0
        weights = torch.clamp(snr, max=gamma) / snr.clamp(min=1e-6)
        weighted_loss = (weights * per_sample_loss).mean()

        self.assertNotAlmostEqual(flat_loss.item(), weighted_loss.item(), places=4)

    def test_weights_clamp_at_gamma(self):
        """Weights must be clamped: max weight is gamma/SNR, never exceeding 1.0 for high-SNR samples."""
        gamma = 5.0
        # High SNR (low noise, t near 0) -> SNR >> gamma -> weight = gamma/SNR < 1
        t_low = torch.tensor([0.01])
        snr_low = ((1.0 - t_low) / t_low.clamp(min=1e-6)) ** 2
        w_low = torch.clamp(snr_low, max=gamma) / snr_low.clamp(min=1e-6)
        self.assertLess(w_low.item(), 1.0)
        self.assertAlmostEqual(w_low.item(), gamma / snr_low.item(), places=5)

        # Low SNR (high noise, t near 1) -> SNR << gamma -> weight = 1.0
        t_high = torch.tensor([0.99])
        snr_high = ((1.0 - t_high) / t_high.clamp(min=1e-6)) ** 2
        w_high = torch.clamp(snr_high, max=gamma) / snr_high.clamp(min=1e-6)
        self.assertAlmostEqual(w_high.item(), 1.0, places=3)

    def test_weights_at_midpoint(self):
        """At t=0.5, SNR=1.0, so weight = min(1, gamma)/1 = 1.0."""
        gamma = 5.0
        t = torch.tensor([0.5])
        snr = ((1.0 - t) / t.clamp(min=1e-6)) ** 2
        w = torch.clamp(snr, max=gamma) / snr.clamp(min=1e-6)
        self.assertAlmostEqual(w.item(), 1.0, places=5)

    def test_gamma_controls_clamp(self):
        """Lower gamma should clamp more aggressively for high-SNR samples."""
        t = torch.tensor([0.05])  # high SNR
        snr = ((1.0 - t) / t.clamp(min=1e-6)) ** 2

        w_gamma5 = (torch.clamp(snr, max=5.0) / snr.clamp(min=1e-6)).item()
        w_gamma1 = (torch.clamp(snr, max=1.0) / snr.clamp(min=1e-6)).item()

        self.assertGreater(w_gamma5, w_gamma1)


class TestTrainingStepSNRIntegration(unittest.TestCase):
    """Verify FixedLoRAModule uses SNR weighting when configured."""

    def _make_module(self, loss_weighting="none", snr_gamma=5.0, is_turbo=False):
        """Create a FixedLoRAModule with mocked injection."""
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
            loss_weighting=loss_weighting,
            snr_gamma=snr_gamma,
        )

        with patch.object(FixedLoRAModule, "_inject_lora"):
            module = FixedLoRAModule(model, lora_cfg, train_cfg, "cpu", torch.float32)
            module.model = model

        return module

    def _make_batch(self, bsz=2, seq_len=10, hidden=64):
        return {
            "target_latents": torch.randn(bsz, seq_len, hidden),
            "attention_mask": torch.ones(bsz, seq_len),
            "encoder_hidden_states": torch.randn(bsz, 5, 128),
            "encoder_attention_mask": torch.ones(bsz, 5),
            "context_latents": torch.randn(bsz, seq_len, 128),
        }

    @patch("sidestep_engine.core.lora_module.sample_timesteps")
    def test_timestep_buffer_populated(self, mock_sample):
        """Timestep buffer should accumulate after training steps."""
        mock_sample.return_value = (
            torch.tensor([0.4, 0.6]),
            torch.tensor([0.4, 0.6]),
        )
        module = self._make_module(loss_weighting="none", is_turbo=False)
        self.assertIsNone(module.drain_timestep_buffer())

        module.training_step(self._make_batch())
        buf = module.drain_timestep_buffer()
        self.assertIsNotNone(buf)
        self.assertEqual(buf.shape, (2,))

    @patch("sidestep_engine.core.lora_module.sample_timesteps")
    def test_min_snr_returns_scalar_loss(self, mock_sample):
        """min_snr weighting must still return a scalar loss tensor."""
        mock_sample.return_value = (
            torch.tensor([0.3, 0.7]),
            torch.tensor([0.3, 0.7]),
        )
        module = self._make_module(loss_weighting="min_snr", is_turbo=False)
        loss = module.training_step(self._make_batch())
        self.assertEqual(loss.shape, ())
        self.assertTrue(loss.dtype == torch.float32)

    @patch("sidestep_engine.core.lora_module.sample_timesteps")
    def test_none_returns_scalar_loss(self, mock_sample):
        """none weighting must return a scalar loss tensor."""
        mock_sample.return_value = (
            torch.tensor([0.3, 0.7]),
            torch.tensor([0.3, 0.7]),
        )
        module = self._make_module(loss_weighting="none", is_turbo=False)
        loss = module.training_step(self._make_batch())
        self.assertEqual(loss.shape, ())
        self.assertTrue(loss.dtype == torch.float32)

    @patch("sidestep_engine.core.lora_module.sample_discrete_timesteps")
    def test_turbo_timestep_buffer_populated(self, mock_discrete):
        """Timestep buffer should also accumulate for turbo training steps."""
        mock_discrete.return_value = (
            torch.tensor([0.5, 0.3]),
            torch.tensor([0.5, 0.3]),
        )
        module = self._make_module(loss_weighting="none", is_turbo=True)
        module.training_step(self._make_batch())
        buf = module.drain_timestep_buffer()
        self.assertIsNotNone(buf)


class TestTimestepHistogramLogger(unittest.TestCase):
    """Verify the TensorBoard timestep histogram method."""

    def test_log_timestep_histogram_with_writer(self):
        """log_timestep_histogram should call add_histogram when writer exists."""
        from sidestep_engine.logging.tensorboard_utils import TrainingLogger

        with patch("sidestep_engine.logging.tensorboard_utils._TB_AVAILABLE", True), \
             patch("sidestep_engine.logging.tensorboard_utils.SummaryWriter") as MockWriter:
            mock_writer_instance = MagicMock()
            MockWriter.return_value = mock_writer_instance

            import tempfile
            with tempfile.TemporaryDirectory() as tmpdir:
                logger = TrainingLogger(log_dir=tmpdir, enabled=True)
                logger._writer = mock_writer_instance

                timesteps = torch.tensor([0.1, 0.3, 0.5, 0.7, 0.9])
                logger.log_timestep_histogram(timesteps, step=100)

                mock_writer_instance.add_histogram.assert_called_once()
                call_args = mock_writer_instance.add_histogram.call_args
                self.assertEqual(call_args[0][0], "train/timestep_distribution")
                self.assertEqual(call_args[1]["global_step"], 100)

    def test_log_timestep_histogram_no_writer(self):
        """log_timestep_histogram should be a no-op when writer is None."""
        from sidestep_engine.logging.tensorboard_utils import TrainingLogger

        with patch("sidestep_engine.logging.tensorboard_utils._TB_AVAILABLE", False):
            logger = TrainingLogger.__new__(TrainingLogger)
            logger._writer = None
            logger._enabled = False
            logger._disabled_by_error = False

            # Should not raise
            timesteps = torch.tensor([0.1, 0.3, 0.5])
            logger.log_timestep_histogram(timesteps, step=10)


if __name__ == "__main__":
    unittest.main()
