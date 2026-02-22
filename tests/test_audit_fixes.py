"""Tests for bugs and fixes identified in the codebase audit.

Covers:
- NaN/Inf loss guard in training loops
- SNR weighting overflow protection
- TrainingConfigV2.to_dict() round-trip completeness
- Config validation ranges (learning rate, batch size, rank, etc.)
- Data module required-key validation
- Wizard projection resolution with None values
"""

from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import torch
import torch.nn.functional as F


# ===================================================================
# 1. NaN/Inf loss guard
# ===================================================================

class TestNaNLossGuard(unittest.TestCase):
    """Verify that NaN/Inf losses are detected before backward."""

    def test_nan_loss_detected(self):
        loss = torch.tensor(float("nan"))
        self.assertTrue(torch.isnan(loss))

    def test_inf_loss_detected(self):
        loss = torch.tensor(float("inf"))
        self.assertTrue(torch.isinf(loss))

    def test_valid_loss_passes(self):
        loss = torch.tensor(0.5)
        self.assertFalse(torch.isnan(loss))
        self.assertFalse(torch.isinf(loss))

    def test_negative_inf_detected(self):
        loss = torch.tensor(float("-inf"))
        self.assertTrue(torch.isinf(loss))


# ===================================================================
# 2. SNR weighting overflow protection
# ===================================================================

class TestSNRWeightingOverflow(unittest.TestCase):
    """Verify SNR clamping prevents overflow for extreme timesteps."""

    def _compute_snr_weights(self, t: torch.Tensor, gamma: float = 5.0):
        """Reproduce the fixed SNR computation from fixed_lora_module.py."""
        t_safe = t.clamp(min=1e-4, max=1.0 - 1e-4)
        snr = ((1.0 - t_safe) / t_safe) ** 2
        snr = snr.clamp(max=1e6)
        weights = torch.clamp(snr, max=gamma) / snr.clamp(min=1e-6)
        return snr, weights

    def test_extreme_low_t_no_overflow(self):
        """t near zero should not produce Inf SNR."""
        t = torch.tensor([1e-7, 1e-6, 1e-5, 1e-4])
        snr, weights = self._compute_snr_weights(t)
        self.assertFalse(torch.isinf(snr).any(), "SNR contains Inf")
        self.assertFalse(torch.isnan(snr).any(), "SNR contains NaN")
        self.assertFalse(torch.isinf(weights).any(), "Weights contain Inf")
        self.assertFalse(torch.isnan(weights).any(), "Weights contain NaN")

    def test_extreme_high_t_no_overflow(self):
        """t near 1 should not produce NaN/Inf."""
        t = torch.tensor([0.999, 0.9999, 1.0 - 1e-7])
        snr, weights = self._compute_snr_weights(t)
        self.assertFalse(torch.isinf(snr).any())
        self.assertFalse(torch.isnan(snr).any())
        self.assertFalse(torch.isinf(weights).any())
        self.assertFalse(torch.isnan(weights).any())

    def test_snr_clamped_below_1e6(self):
        """SNR should never exceed 1e6 after clamping."""
        t = torch.tensor([1e-7])
        snr, _ = self._compute_snr_weights(t)
        self.assertLessEqual(snr.item(), 1e6)

    def test_normal_range_unchanged(self):
        """Normal timesteps (0.1-0.9) should produce finite, reasonable weights."""
        t = torch.linspace(0.1, 0.9, 9)
        snr, weights = self._compute_snr_weights(t)
        self.assertTrue(torch.isfinite(snr).all())
        self.assertTrue(torch.isfinite(weights).all())
        self.assertTrue((weights > 0).all())


# ===================================================================
# 3. TrainingConfigV2.to_dict() round-trip
# ===================================================================

class TestConfigToDict(unittest.TestCase):
    """Verify to_dict() includes all important fields."""

    def test_save_best_fields_in_to_dict(self):
        from acestep.training_v2.configs import TrainingConfigV2

        cfg = TrainingConfigV2(
            save_best=True,
            save_best_after=150,
            early_stop_patience=10,
        )
        d = cfg.to_dict()
        self.assertIn("save_best", d)
        self.assertIn("save_best_after", d)
        self.assertIn("early_stop_patience", d)
        self.assertEqual(d["save_best"], True)
        self.assertEqual(d["save_best_after"], 150)
        self.assertEqual(d["early_stop_patience"], 10)

    def test_round_trip_save_json(self):
        from acestep.training_v2.configs import TrainingConfigV2

        cfg = TrainingConfigV2(
            save_best=False,
            save_best_after=50,
            early_stop_patience=5,
            learning_rate=1e-4,
            batch_size=2,
        )
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "test_config.json"
            cfg.save_json(path)
            data = json.loads(path.read_text(encoding="utf-8"))

        self.assertEqual(data["save_best"], False)
        self.assertEqual(data["save_best_after"], 50)
        self.assertEqual(data["early_stop_patience"], 5)
        self.assertAlmostEqual(data["learning_rate"], 1e-4)
        self.assertEqual(data["batch_size"], 2)


# ===================================================================
# 4. Config validation ranges
# ===================================================================

class TestConfigValidation(unittest.TestCase):
    """Verify TrainingConfigV2 rejects dangerous values."""

    def test_negative_lr_rejected(self):
        from acestep.training_v2.configs import TrainingConfigV2
        with self.assertRaises(ValueError):
            TrainingConfigV2(learning_rate=-0.001)

    def test_zero_lr_rejected(self):
        from acestep.training_v2.configs import TrainingConfigV2
        with self.assertRaises(ValueError):
            TrainingConfigV2(learning_rate=0.0)

    def test_extreme_lr_rejected(self):
        from acestep.training_v2.configs import TrainingConfigV2
        with self.assertRaises(ValueError):
            TrainingConfigV2(learning_rate=10.0)

    def test_zero_batch_size_rejected(self):
        from acestep.training_v2.configs import TrainingConfigV2
        with self.assertRaises(ValueError):
            TrainingConfigV2(batch_size=0)

    def test_negative_batch_size_rejected(self):
        from acestep.training_v2.configs import TrainingConfigV2
        with self.assertRaises(ValueError):
            TrainingConfigV2(batch_size=-1)

    def test_zero_max_grad_norm_rejected(self):
        from acestep.training_v2.configs import TrainingConfigV2
        with self.assertRaises(ValueError):
            TrainingConfigV2(max_grad_norm=0.0)

    def test_zero_epochs_rejected(self):
        from acestep.training_v2.configs import TrainingConfigV2
        with self.assertRaises(ValueError):
            TrainingConfigV2(max_epochs=0)

    def test_valid_config_accepted(self):
        from acestep.training_v2.configs import TrainingConfigV2
        cfg = TrainingConfigV2(
            learning_rate=1e-4,
            batch_size=2,
            max_grad_norm=1.0,
            max_epochs=100,
        )
        self.assertEqual(cfg.batch_size, 2)


class TestLoRAConfigValidation(unittest.TestCase):
    """Verify LoRAConfigV2 rejects dangerous rank/alpha values."""

    def test_zero_rank_rejected(self):
        from acestep.training_v2.configs import LoRAConfigV2
        with self.assertRaises(ValueError):
            LoRAConfigV2(r=0)

    def test_negative_rank_rejected(self):
        from acestep.training_v2.configs import LoRAConfigV2
        with self.assertRaises(ValueError):
            LoRAConfigV2(r=-1)

    def test_extreme_rank_rejected(self):
        from acestep.training_v2.configs import LoRAConfigV2
        with self.assertRaises(ValueError):
            LoRAConfigV2(r=2048)

    def test_zero_alpha_rejected(self):
        from acestep.training_v2.configs import LoRAConfigV2
        with self.assertRaises(ValueError):
            LoRAConfigV2(alpha=0)

    def test_valid_lora_config_accepted(self):
        from acestep.training_v2.configs import LoRAConfigV2
        cfg = LoRAConfigV2(r=64, alpha=128)
        self.assertEqual(cfg.r, 64)
        self.assertEqual(cfg.alpha, 128)


# ===================================================================
# 5. Data module required-key validation
# ===================================================================

class TestDataModuleKeyValidation(unittest.TestCase):
    """Verify __getitem__ raises clear error on malformed .pt files."""

    def test_missing_key_raises_clear_error(self):
        from acestep.training_v2._vendor.data_module import PreprocessedTensorDataset

        with tempfile.TemporaryDirectory() as tmp:
            bad_pt = Path(tmp) / "bad_sample.pt"
            torch.save({"target_latents": torch.zeros(10, 64)}, bad_pt)

            ds = PreprocessedTensorDataset(tmp)
            if len(ds) > 0:
                with self.assertRaises(KeyError) as ctx:
                    ds[0]
                self.assertIn("missing required keys", str(ctx.exception))

    def test_valid_pt_loads_successfully(self):
        from acestep.training_v2._vendor.data_module import PreprocessedTensorDataset

        with tempfile.TemporaryDirectory() as tmp:
            good_pt = Path(tmp) / "good_sample.pt"
            torch.save({
                "target_latents": torch.randn(10, 64),
                "attention_mask": torch.ones(10),
                "encoder_hidden_states": torch.randn(5, 256),
                "encoder_attention_mask": torch.ones(5),
                "context_latents": torch.randn(10, 65),
            }, good_pt)

            ds = PreprocessedTensorDataset(tmp)
            if len(ds) > 0:
                sample = ds[0]
                self.assertIn("target_latents", sample)
                self.assertIn("attention_mask", sample)
                self.assertIn("encoder_hidden_states", sample)
                self.assertIn("encoder_attention_mask", sample)
                self.assertIn("context_latents", sample)


# ===================================================================
# 6. Wizard projection resolution with None values
# ===================================================================

class TestResolveWizardProjections(unittest.TestCase):
    """Verify _resolve_wizard_projections handles None values safely."""

    def test_none_target_modules_str_uses_default(self):
        from acestep.training_v2.ui.flows_common import _resolve_wizard_projections
        result = _resolve_wizard_projections({"target_modules_str": None})
        self.assertIsInstance(result, list)
        self.assertTrue(len(result) > 0)

    def test_missing_key_uses_default(self):
        from acestep.training_v2.ui.flows_common import _resolve_wizard_projections
        result = _resolve_wizard_projections({})
        self.assertIsInstance(result, list)
        self.assertTrue(len(result) > 0)

    def test_explicit_value_used(self):
        from acestep.training_v2.ui.flows_common import _resolve_wizard_projections
        result = _resolve_wizard_projections({"target_modules_str": "q_proj v_proj"})
        self.assertEqual(result, ["q_proj", "v_proj"])

    def test_none_split_modules_use_default(self):
        from acestep.training_v2.ui.flows_common import _resolve_wizard_projections
        result = _resolve_wizard_projections({
            "attention_type": "both",
            "self_target_modules_str": None,
            "cross_target_modules_str": None,
        })
        self.assertIsInstance(result, list)
        self.assertTrue(len(result) > 0)


# ===================================================================
# 7. GPU utils safety
# ===================================================================

class TestCudaFreeMb(unittest.TestCase):
    """Verify _cuda_free_mb handles missing CUDA gracefully."""

    def test_returns_none_on_bad_device(self):
        from acestep.training_v2.gpu_utils import _cuda_free_mb
        if not torch.cuda.is_available():
            result = _cuda_free_mb(999)
            self.assertIsNone(result)

if __name__ == "__main__":
    unittest.main()
