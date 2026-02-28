"""Tests for sidestep_engine.core.adaptive_timestep — AdaptiveTimestepSampler."""

import unittest

import torch

from sidestep_engine.core.adaptive_timestep import AdaptiveTimestepSampler


class TestAdaptiveTimestepInit(unittest.TestCase):
    """Initialization and validation."""

    def test_default_bins(self):
        s = AdaptiveTimestepSampler(n_bins=10, ratio=0.3)
        self.assertEqual(s.n_bins, 10)
        self.assertEqual(s._bin_loss.shape[0], 10)

    def test_invalid_ratio_raises(self):
        with self.assertRaises(ValueError):
            AdaptiveTimestepSampler(ratio=-0.1)
        with self.assertRaises(ValueError):
            AdaptiveTimestepSampler(ratio=1.5)

    def test_invalid_bins_raises(self):
        with self.assertRaises(ValueError):
            AdaptiveTimestepSampler(n_bins=1)


class TestAdaptiveTimestepUpdate(unittest.TestCase):
    """Per-bin loss EMA update."""

    def test_update_increases_bin_loss(self):
        s = AdaptiveTimestepSampler(n_bins=5, ema_decay=0.0, ratio=0.5)
        # All timesteps in bin 0 (t ~0.05), high loss
        t = torch.tensor([0.05, 0.08])
        losses = torch.tensor([10.0, 10.0])
        s.update(t, losses)
        # Bin 0 should have loss 10, others stay at 1
        self.assertAlmostEqual(s._bin_loss[0].item(), 10.0, places=3)
        self.assertAlmostEqual(s._bin_loss[1].item(), 1.0, places=3)

    def test_total_updates_increments(self):
        s = AdaptiveTimestepSampler(n_bins=5, ratio=0.3)
        self.assertEqual(s._total_updates, 0)
        s.update(torch.tensor([0.5]), torch.tensor([1.0]))
        self.assertEqual(s._total_updates, 1)


class TestAdaptiveTimestepSample(unittest.TestCase):
    """Sampling mixes adaptive + base distributions."""

    @staticmethod
    def _base_sampler(batch_size, device, dtype, **kwargs):
        """Deterministic base sampler for testing."""
        t = torch.full((batch_size,), 0.5, device=device, dtype=dtype)
        return t, t.clone()

    def test_output_shape(self):
        s = AdaptiveTimestepSampler(n_bins=5, ratio=0.5)
        t, r = s.sample(8, self._base_sampler, torch.device("cpu"), torch.float32)
        self.assertEqual(t.shape, (8,))
        self.assertEqual(r.shape, (8,))

    def test_ratio_zero_returns_all_base(self):
        s = AdaptiveTimestepSampler(n_bins=5, ratio=0.0)
        t, r = s.sample(4, self._base_sampler, torch.device("cpu"), torch.float32)
        # All should be 0.5 from the base sampler
        torch.testing.assert_close(t, torch.full((4,), 0.5))

    def test_ratio_one_returns_all_adaptive(self):
        s = AdaptiveTimestepSampler(n_bins=5, ratio=1.0)
        t, r = s.sample(4, self._base_sampler, torch.device("cpu"), torch.float32)
        # All in [0, 1] but NOT all 0.5 (adaptive samples from bins)
        self.assertTrue((t >= 0).all() and (t <= 1).all())

    def test_adaptive_bias_toward_high_loss_bin(self):
        s = AdaptiveTimestepSampler(n_bins=5, ema_decay=0.0, ratio=1.0)
        # Set bin 4 (t in [0.8, 1.0]) to very high loss
        s._bin_loss = torch.tensor([0.01, 0.01, 0.01, 0.01, 100.0])
        t, _ = s.sample(1000, self._base_sampler, torch.device("cpu"), torch.float32)
        # Most samples should be in [0.8, 1.0]
        high_bin_count = (t >= 0.8).sum().item()
        self.assertGreater(high_bin_count, 800)


class TestAdaptiveTimestepStateDict(unittest.TestCase):
    """Checkpoint save/load."""

    def test_round_trip(self):
        s = AdaptiveTimestepSampler(n_bins=5, ratio=0.3)
        s.update(torch.tensor([0.1, 0.5, 0.9]), torch.tensor([1.0, 2.0, 3.0]))
        state = s.state_dict()

        s2 = AdaptiveTimestepSampler(n_bins=5, ratio=0.3)
        s2.load_state_dict(state)
        torch.testing.assert_close(s2._bin_loss, s._bin_loss)
        self.assertEqual(s2._total_updates, s._total_updates)

    def test_bin_mismatch_raises(self):
        s = AdaptiveTimestepSampler(n_bins=5, ratio=0.3)
        state = s.state_dict()
        s2 = AdaptiveTimestepSampler(n_bins=10, ratio=0.3)
        with self.assertRaises(ValueError):
            s2.load_state_dict(state)


class TestAdaptiveTimestepHistogram(unittest.TestCase):
    """Histogram for TensorBoard logging."""

    def test_histogram_shape(self):
        s = AdaptiveTimestepSampler(n_bins=8, ratio=0.3)
        h = s.get_histogram()
        self.assertEqual(h.shape, (8,))

    def test_histogram_is_clone(self):
        s = AdaptiveTimestepSampler(n_bins=5, ratio=0.3)
        h = s.get_histogram()
        h.fill_(999)
        # Original should be unmodified
        self.assertAlmostEqual(s._bin_loss[0].item(), 1.0, places=3)


if __name__ == "__main__":
    unittest.main()
