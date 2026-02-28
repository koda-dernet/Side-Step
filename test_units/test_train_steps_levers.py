"""Tests for sidestep_engine.ui.flows.train_steps_levers — wizard step."""

import unittest
from unittest.mock import patch, MagicMock

from sidestep_engine.core.configs import TrainingConfigV2


class TestConfigDefaults(unittest.TestCase):
    """New config fields have correct defaults and pass validation."""

    def test_defaults_zero_behavior_change(self):
        cfg = TrainingConfigV2()
        self.assertEqual(cfg.ema_decay, 0.0)
        self.assertEqual(cfg.val_split, 0.0)
        self.assertEqual(cfg.adaptive_timestep_ratio, 0.0)
        self.assertEqual(cfg.warmup_start_factor, 0.1)
        self.assertEqual(cfg.cosine_eta_min_ratio, 0.01)
        self.assertEqual(cfg.cosine_restarts_count, 4)

    def test_to_dict_includes_new_fields(self):
        cfg = TrainingConfigV2()
        d = cfg.to_dict()
        self.assertIn("ema_decay", d)
        self.assertIn("val_split", d)
        self.assertIn("adaptive_timestep_ratio", d)
        self.assertIn("warmup_start_factor", d)
        self.assertIn("cosine_eta_min_ratio", d)
        self.assertIn("cosine_restarts_count", d)

    def test_ema_decay_boundary(self):
        # 0.0 should be fine
        TrainingConfigV2(ema_decay=0.0)
        # 0.9999 should be fine
        TrainingConfigV2(ema_decay=0.9999)
        # 1.0 should fail
        with self.assertRaises(ValueError):
            TrainingConfigV2(ema_decay=1.0)

    def test_val_split_boundary(self):
        TrainingConfigV2(val_split=0.0)
        TrainingConfigV2(val_split=0.5)
        with self.assertRaises(ValueError):
            TrainingConfigV2(val_split=0.6)

    def test_adaptive_ratio_boundary(self):
        TrainingConfigV2(adaptive_timestep_ratio=0.0)
        TrainingConfigV2(adaptive_timestep_ratio=1.0)
        with self.assertRaises(ValueError):
            TrainingConfigV2(adaptive_timestep_ratio=1.1)

    def test_warmup_start_factor_boundary(self):
        TrainingConfigV2(warmup_start_factor=0.01)
        TrainingConfigV2(warmup_start_factor=1.0)
        with self.assertRaises(ValueError):
            TrainingConfigV2(warmup_start_factor=0.0)

    def test_cosine_eta_min_ratio_boundary(self):
        TrainingConfigV2(cosine_eta_min_ratio=0.0)
        TrainingConfigV2(cosine_eta_min_ratio=1.0)
        with self.assertRaises(ValueError):
            TrainingConfigV2(cosine_eta_min_ratio=-0.1)

    def test_cosine_restarts_count_boundary(self):
        TrainingConfigV2(cosine_restarts_count=1)
        with self.assertRaises(ValueError):
            TrainingConfigV2(cosine_restarts_count=0)


class TestPresetsIncludeNewFields(unittest.TestCase):
    """New fields are in the saveable preset set."""

    def test_preset_fields_contain_levers(self):
        from sidestep_engine.ui.presets import PRESET_FIELDS
        for key in ("ema_decay", "val_split", "adaptive_timestep_ratio",
                     "warmup_start_factor", "cosine_eta_min_ratio",
                     "cosine_restarts_count", "save_best_every_n_steps"):
            self.assertIn(key, PRESET_FIELDS, f"Missing from PRESET_FIELDS: {key}")


class TestReviewSummaryDefaults(unittest.TestCase):
    """Review summary knows about the new defaults."""

    def test_defaults_present(self):
        from sidestep_engine.ui.flows.review_summary import _DEFAULTS
        for key in ("ema_decay", "val_split", "adaptive_timestep_ratio",
                     "warmup_start_factor", "cosine_eta_min_ratio",
                     "cosine_restarts_count", "save_best_every_n_steps"):
            self.assertIn(key, _DEFAULTS, f"Missing from _DEFAULTS: {key}")


class TestBuildTrainNamespace(unittest.TestCase):
    """build_train_namespace includes the new fields."""

    def test_namespace_has_levers(self):
        from sidestep_engine.ui.flows.common import build_train_namespace
        a = {
            "checkpoint_dir": "/tmp/ckpt",
            "model_variant": "turbo",
            "dataset_dir": "/tmp/data",
            "output_dir": "/tmp/out",
        }
        ns = build_train_namespace(a)
        self.assertEqual(ns.ema_decay, 0.0)
        self.assertEqual(ns.val_split, 0.0)
        self.assertEqual(ns.adaptive_timestep_ratio, 0.0)
        self.assertEqual(ns.warmup_start_factor, 0.1)
        self.assertEqual(ns.cosine_eta_min_ratio, 0.01)
        self.assertEqual(ns.cosine_restarts_count, 4)
        self.assertEqual(ns.save_best_every_n_steps, 0)


if __name__ == "__main__":
    unittest.main()
