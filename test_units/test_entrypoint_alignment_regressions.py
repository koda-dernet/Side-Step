"""Regression tests for GUI/CLI/Wizard alignment."""

from __future__ import annotations

import unittest
from pathlib import Path

from sidestep_engine.training_defaults import (
    DEFAULT_EPOCHS,
    DEFAULT_LEARNING_RATE,
    DEFAULT_OPTIMIZER_TYPE,
    DEFAULT_SAVE_EVERY,
)


class TestCanonicalDefaultsParity(unittest.TestCase):
    """Ensure all entrypoints use the canonical default values."""

    def test_cli_defaults_match_canonical(self) -> None:
        from sidestep_engine.cli.args import build_root_parser

        args = build_root_parser().parse_args([
            "train",
            "--checkpoint-dir", "/tmp/ckpt",
            "--dataset-dir", "/tmp/data",
            "--output-dir", "/tmp/out",
        ])
        self.assertEqual(args.learning_rate, DEFAULT_LEARNING_RATE)
        self.assertEqual(args.epochs, DEFAULT_EPOCHS)
        self.assertEqual(args.optimizer_type, DEFAULT_OPTIMIZER_TYPE)
        self.assertEqual(args.save_every, DEFAULT_SAVE_EVERY)

    def test_wizard_namespace_defaults_match_canonical(self) -> None:
        from sidestep_engine.ui.flows.common import build_train_namespace

        ns = build_train_namespace({
            "checkpoint_dir": "/tmp/ckpt",
            "model_variant": "turbo",
            "base_model": "turbo",
            "dataset_dir": "/tmp/data",
            "output_dir": "/tmp/out",
        })
        self.assertEqual(ns.learning_rate, DEFAULT_LEARNING_RATE)
        self.assertEqual(ns.epochs, DEFAULT_EPOCHS)
        self.assertEqual(ns.optimizer_type, DEFAULT_OPTIMIZER_TYPE)
        self.assertEqual(ns.save_every, DEFAULT_SAVE_EVERY)

    def test_review_defaults_match_canonical(self) -> None:
        from sidestep_engine.ui.flows.review_summary import _DEFAULTS

        self.assertEqual(_DEFAULTS["learning_rate"], DEFAULT_LEARNING_RATE)
        self.assertEqual(_DEFAULTS["epochs"], DEFAULT_EPOCHS)
        self.assertEqual(_DEFAULTS["optimizer_type"], DEFAULT_OPTIMIZER_TYPE)
        self.assertEqual(_DEFAULTS["save_every"], DEFAULT_SAVE_EVERY)


class TestGuiCommandExportRegression(unittest.TestCase):
    """Guard CLI export behavior from GUI config."""

    def test_target_module_flags_use_list_serializer(self) -> None:
        src = Path("frontend/js/api-cli.js").read_text(encoding="utf-8")
        self.assertIn("addList('--target-modules'", src)
        self.assertIn("addList('--self-target-modules'", src)
        self.assertIn("addList('--cross-target-modules'", src)
        self.assertNotIn("add('--target-modules', config.projections)", src)


class TestUnknownVariantInferenceRegression(unittest.TestCase):
    """Unknown/custom variants should infer strategy by inference steps."""

    def test_wizard_common_infers_turbo_from_8_steps(self) -> None:
        from sidestep_engine.ui.flows.common import is_turbo

        self.assertTrue(is_turbo({"base_model": "custom-model", "num_inference_steps": 8}))
        self.assertFalse(is_turbo({"base_model": "custom-model", "num_inference_steps": 50}))

    def test_gui_model_behavior_mentions_step_inference(self) -> None:
        src = Path("frontend/js/workspace-behaviors.js").read_text(encoding="utf-8")
        self.assertIn("inferred turbo from 8 steps", src)
        self.assertIn("inferred base/SFT from inference steps", src)


if __name__ == "__main__":
    unittest.main()

