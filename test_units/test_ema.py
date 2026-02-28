"""Tests for sidestep_engine.core.ema — AdapterEMA."""

import unittest

import torch

from sidestep_engine.core.ema import AdapterEMA


class TestAdapterEMAInit(unittest.TestCase):
    """Initialisation and basic invariants."""

    def test_shadow_matches_initial_params(self):
        p = torch.nn.Parameter(torch.tensor([1.0, 2.0, 3.0]))
        ema = AdapterEMA([p], decay=0.99)
        torch.testing.assert_close(ema._shadow[0], p.data.cpu())

    def test_shadow_on_cpu(self):
        p = torch.nn.Parameter(torch.ones(4))
        ema = AdapterEMA([p], decay=0.99)
        self.assertEqual(ema._shadow[0].device, torch.device("cpu"))

    def test_invalid_decay_raises(self):
        p = torch.nn.Parameter(torch.ones(2))
        with self.assertRaises(ValueError):
            AdapterEMA([p], decay=1.0)
        with self.assertRaises(ValueError):
            AdapterEMA([p], decay=-0.1)


class TestAdapterEMAUpdate(unittest.TestCase):
    """EMA update formula: shadow = decay * shadow + (1-decay) * param."""

    def test_single_update(self):
        p = torch.nn.Parameter(torch.tensor([1.0]))
        ema = AdapterEMA([p], decay=0.9)
        p.data.fill_(2.0)
        ema.update()
        # shadow = 0.9 * 1.0 + 0.1 * 2.0 = 1.1
        self.assertAlmostEqual(ema._shadow[0].item(), 1.1, places=5)

    def test_multiple_updates(self):
        p = torch.nn.Parameter(torch.tensor([0.0]))
        ema = AdapterEMA([p], decay=0.5)
        p.data.fill_(1.0)
        ema.update()  # 0.5*0 + 0.5*1 = 0.5
        ema.update()  # 0.5*0.5 + 0.5*1 = 0.75
        self.assertAlmostEqual(ema._shadow[0].item(), 0.75, places=5)

    def test_decay_zero_instant_tracking(self):
        p = torch.nn.Parameter(torch.tensor([5.0]))
        ema = AdapterEMA([p], decay=0.0)
        p.data.fill_(99.0)
        ema.update()
        self.assertAlmostEqual(ema._shadow[0].item(), 99.0, places=5)

    def test_step_count_increments(self):
        p = torch.nn.Parameter(torch.ones(2))
        ema = AdapterEMA([p], decay=0.99)
        self.assertEqual(ema._step_count, 0)
        ema.update()
        ema.update()
        self.assertEqual(ema._step_count, 2)


class TestAdapterEMAApplyRestore(unittest.TestCase):
    """Apply/restore round-trip."""

    def test_apply_copies_shadow_to_params(self):
        p = torch.nn.Parameter(torch.tensor([1.0]))
        ema = AdapterEMA([p], decay=0.9)
        p.data.fill_(10.0)
        ema.update()
        expected_shadow = ema._shadow[0].item()
        ema.apply()
        self.assertAlmostEqual(p.data.item(), expected_shadow, places=5)

    def test_restore_recovers_training_weights(self):
        p = torch.nn.Parameter(torch.tensor([1.0]))
        ema = AdapterEMA([p], decay=0.9)
        p.data.fill_(10.0)
        ema.update()
        ema.apply()
        ema.restore()
        self.assertAlmostEqual(p.data.item(), 10.0, places=5)

    def test_restore_without_apply_warns(self):
        p = torch.nn.Parameter(torch.ones(2))
        ema = AdapterEMA([p], decay=0.99)
        with self.assertLogs("sidestep_engine.core.ema", level="WARNING"):
            ema.restore()


class TestAdapterEMAStateDict(unittest.TestCase):
    """Checkpoint save/load."""

    def test_round_trip(self):
        p = torch.nn.Parameter(torch.tensor([1.0, 2.0]))
        ema = AdapterEMA([p], decay=0.99)
        p.data.fill_(5.0)
        ema.update()
        ema.update()

        state = ema.state_dict()

        # New EMA with fresh params
        p2 = torch.nn.Parameter(torch.zeros(2))
        ema2 = AdapterEMA([p2], decay=0.99)
        ema2.load_state_dict(state)

        torch.testing.assert_close(ema2._shadow[0], ema._shadow[0])
        self.assertEqual(ema2._step_count, 2)

    def test_mismatched_param_count_raises(self):
        p1 = torch.nn.Parameter(torch.ones(2))
        ema = AdapterEMA([p1], decay=0.99)
        state = ema.state_dict()

        p2a = torch.nn.Parameter(torch.ones(2))
        p2b = torch.nn.Parameter(torch.ones(3))
        ema2 = AdapterEMA([p2a, p2b], decay=0.99)
        with self.assertRaises(ValueError):
            ema2.load_state_dict(state)

    def test_decay_mismatch_logs_info(self):
        p = torch.nn.Parameter(torch.ones(2))
        ema = AdapterEMA([p], decay=0.99)
        state = ema.state_dict()
        ema2 = AdapterEMA([torch.nn.Parameter(torch.ones(2))], decay=0.999)
        with self.assertLogs("sidestep_engine.core.ema", level="INFO"):
            ema2.load_state_dict(state)


class TestAdapterEMAMixedPrecision(unittest.TestCase):
    """Regression: EMA must work when params are bf16/fp16 (mixed precision)."""

    def test_update_with_bfloat16_params(self):
        """F1: lerp_ crashed with bf16 params and float32 shadow."""
        p = torch.nn.Parameter(torch.tensor([1.0], dtype=torch.bfloat16))
        ema = AdapterEMA([p], decay=0.9)
        # Shadow must be float32 regardless of param dtype
        self.assertEqual(ema._shadow[0].dtype, torch.float32)
        p.data.fill_(2.0)
        ema.update()  # must not raise
        self.assertAlmostEqual(ema._shadow[0].item(), 1.1, places=2)

    def test_update_with_float16_params(self):
        p = torch.nn.Parameter(torch.tensor([1.0], dtype=torch.float16))
        ema = AdapterEMA([p], decay=0.9)
        self.assertEqual(ema._shadow[0].dtype, torch.float32)
        p.data.fill_(2.0)
        ema.update()  # must not raise
        self.assertAlmostEqual(ema._shadow[0].item(), 1.1, places=2)

    def test_apply_casts_back_to_param_dtype(self):
        """apply() must cast float32 shadow to the param's dtype."""
        p = torch.nn.Parameter(torch.tensor([1.0], dtype=torch.bfloat16))
        ema = AdapterEMA([p], decay=0.9)
        p.data.fill_(2.0)
        ema.update()
        ema.apply()
        self.assertEqual(p.dtype, torch.bfloat16)

    def test_restore_preserves_param_dtype(self):
        p = torch.nn.Parameter(torch.tensor([1.0], dtype=torch.bfloat16))
        ema = AdapterEMA([p], decay=0.9)
        p.data.fill_(2.0)
        ema.update()
        ema.apply()
        ema.restore()
        self.assertEqual(p.dtype, torch.bfloat16)
        self.assertAlmostEqual(p.data.float().item(), 2.0, places=2)


if __name__ == "__main__":
    unittest.main()
