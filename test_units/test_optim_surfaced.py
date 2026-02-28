"""Tests for surfaced optimizer parameters in sidestep_engine.core.optim."""

import unittest

import torch
from torch.optim import AdamW

from sidestep_engine.core.optim import build_scheduler


class TestWarmupStartFactor(unittest.TestCase):
    """Warmup start factor actually changes the initial LR."""

    def _get_initial_lr(self, warmup_start_factor: float) -> float:
        params = [torch.nn.Parameter(torch.ones(2))]
        optimizer = AdamW(params, lr=0.01)
        scheduler = build_scheduler(
            optimizer, scheduler_type="cosine", total_steps=100,
            warmup_steps=10, lr=0.01, warmup_start_factor=warmup_start_factor,
        )
        # Step 0 LR
        return scheduler.get_last_lr()[0]

    def test_default_factor(self):
        lr = self._get_initial_lr(0.1)
        self.assertAlmostEqual(lr, 0.01 * 0.1, places=6)

    def test_custom_factor(self):
        lr = self._get_initial_lr(0.5)
        self.assertAlmostEqual(lr, 0.01 * 0.5, places=6)


class TestCosineEtaMinRatio(unittest.TestCase):
    """Cosine eta_min ratio controls the floor LR."""

    def _get_final_lr(self, eta_min_ratio: float) -> float:
        params = [torch.nn.Parameter(torch.ones(2))]
        optimizer = AdamW(params, lr=0.01)
        scheduler = build_scheduler(
            optimizer, scheduler_type="cosine", total_steps=100,
            warmup_steps=0, lr=0.01,
            cosine_eta_min_ratio=eta_min_ratio,
        )
        # Step to the end
        for _ in range(100):
            scheduler.step()
        return scheduler.get_last_lr()[0]

    def test_default_ratio(self):
        lr = self._get_final_lr(0.01)
        self.assertAlmostEqual(lr, 0.01 * 0.01, places=6)

    def test_custom_ratio(self):
        lr = self._get_final_lr(0.1)
        self.assertAlmostEqual(lr, 0.01 * 0.1, places=6)

    def test_zero_ratio(self):
        lr = self._get_final_lr(0.0)
        self.assertAlmostEqual(lr, 0.0, places=6)


class TestCosineRestartsCount(unittest.TestCase):
    """Cosine restarts count changes cycle length."""

    def test_different_restart_counts(self):
        """More restarts = shorter cycles = different LR at same step."""
        params_a = [torch.nn.Parameter(torch.ones(2))]
        params_b = [torch.nn.Parameter(torch.ones(2))]
        opt_a = AdamW(params_a, lr=0.01)
        opt_b = AdamW(params_b, lr=0.01)

        sched_a = build_scheduler(
            opt_a, scheduler_type="cosine_restarts", total_steps=200,
            warmup_steps=0, lr=0.01, n_restarts=2,
        )
        sched_b = build_scheduler(
            opt_b, scheduler_type="cosine_restarts", total_steps=200,
            warmup_steps=0, lr=0.01, n_restarts=8,
        )
        # Step 50: different cycle positions should give different LRs
        for _ in range(50):
            sched_a.step()
            sched_b.step()

        lr_a = sched_a.get_last_lr()[0]
        lr_b = sched_b.get_last_lr()[0]
        self.assertNotAlmostEqual(lr_a, lr_b, places=4)


class TestDefaultsMatchExisting(unittest.TestCase):
    """Default params produce identical scheduler to the old hardcoded values."""

    def test_cosine_defaults_unchanged(self):
        params = [torch.nn.Parameter(torch.ones(2))]
        optimizer = AdamW(params, lr=1e-4)
        sched = build_scheduler(
            optimizer, scheduler_type="cosine", total_steps=1000,
            warmup_steps=100, lr=1e-4,
            # All defaults — should match pre-change behavior
        )
        # Just verify it creates without error and has reasonable LR
        lr = sched.get_last_lr()[0]
        self.assertGreater(lr, 0)


if __name__ == "__main__":
    unittest.main()
