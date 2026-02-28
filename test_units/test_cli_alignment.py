"""Tests for CLI ↔ wizard alignment.

Covers:
- Short flags parse correctly for the ``fixed`` and ``fisher`` subcommands.
- New args (``--max-steps``, ``--chunk-decay-every``) parse and wire through.
- Default alignment (``offload_encoder=True``, ``target_mlp=True``).
- Branding string in help output.
- Enriched review table groups, non-default highlighting, and _resolve helper.
"""

from __future__ import annotations

import unittest


# ---------------------------------------------------------------------------
# Minimal required args for each subcommand (avoids "required arg" errors)
# ---------------------------------------------------------------------------

_FIXED_REQUIRED = [
    "train",
    "--checkpoint-dir", "/ckpt",
    "--dataset-dir", "/data",
    "--output-dir", "/out",
]

_FISHER_REQUIRED = [
    "analyze",
    "--checkpoint-dir", "/ckpt",
    "--dataset-dir", "/data",
]


def _parse_fixed(extra: list[str] | None = None):
    """Parse a ``fixed`` subcommand with required args + extras."""
    from sidestep_engine.cli.args import build_root_parser
    return build_root_parser().parse_args(_FIXED_REQUIRED + (extra or []))


def _parse_fisher(extra: list[str] | None = None):
    """Parse a ``fisher`` subcommand with required args + extras."""
    from sidestep_engine.cli.args import build_root_parser
    return build_root_parser().parse_args(_FISHER_REQUIRED + (extra or []))


# ===========================================================================
# Short flags
# ===========================================================================

class TestShortFlags(unittest.TestCase):
    """Verify all new short flags parse to the correct dest."""

    def test_checkpoint_dir_short(self):
        args = _parse_fixed(["-c", "/alt_ckpt"])
        # -c should override the earlier --checkpoint-dir
        self.assertEqual(args.checkpoint_dir, "/alt_ckpt")

    def test_dataset_dir_short(self):
        args = _parse_fixed(["-d", "/alt_data"])
        self.assertEqual(args.dataset_dir, "/alt_data")

    def test_output_dir_short(self):
        args = _parse_fixed(["-o", "/alt_out"])
        self.assertEqual(args.output_dir, "/alt_out")

    def test_learning_rate_short(self):
        args = _parse_fixed(["-l", "5e-5"])
        self.assertAlmostEqual(args.learning_rate, 5e-5)

    def test_batch_size_short(self):
        args = _parse_fixed(["-b", "4"])
        self.assertEqual(args.batch_size, 4)

    def test_gradient_accumulation_short(self):
        args = _parse_fixed(["-g", "8"])
        self.assertEqual(args.gradient_accumulation, 8)

    def test_epochs_short(self):
        args = _parse_fixed(["-e", "50"])
        self.assertEqual(args.epochs, 50)

    def test_max_steps_short(self):
        args = _parse_fixed(["-m", "1000"])
        self.assertEqual(args.max_steps, 1000)

    def test_dataset_repeats_short(self):
        args = _parse_fixed(["-R", "3"])
        self.assertEqual(args.dataset_repeats, 3)

    def test_seed_short(self):
        args = _parse_fixed(["-s", "123"])
        self.assertEqual(args.seed, 123)

    def test_run_name_short(self):
        args = _parse_fixed(["-n", "my_run"])
        self.assertEqual(args.run_name, "my_run")

    def test_fisher_checkpoint_dir_short(self):
        args = _parse_fisher(["-c", "/alt_ckpt"])
        self.assertEqual(args.checkpoint_dir, "/alt_ckpt")

    def test_fisher_dataset_dir_short(self):
        args = _parse_fisher(["-d", "/alt_data"])
        self.assertEqual(args.dataset_dir, "/alt_data")


# ===========================================================================
# New CLI args
# ===========================================================================

class TestNewArgs(unittest.TestCase):
    """Verify --max-steps and --chunk-decay-every parse correctly."""

    def test_max_steps_default(self):
        args = _parse_fixed()
        self.assertEqual(args.max_steps, 0)

    def test_max_steps_override(self):
        args = _parse_fixed(["--max-steps", "500"])
        self.assertEqual(args.max_steps, 500)

    def test_chunk_decay_every_default(self):
        args = _parse_fixed()
        self.assertEqual(args.chunk_decay_every, 10)

    def test_chunk_decay_every_override(self):
        args = _parse_fixed(["--chunk-decay-every", "5"])
        self.assertEqual(args.chunk_decay_every, 5)


# ===========================================================================
# Default alignment
# ===========================================================================

class TestDefaultAlignment(unittest.TestCase):
    """Verify CLI defaults match wizard and config defaults."""

    def test_offload_encoder_default_true(self):
        args = _parse_fixed()
        self.assertTrue(args.offload_encoder)

    def test_offload_encoder_disable(self):
        args = _parse_fixed(["--no-offload-encoder"])
        self.assertFalse(args.offload_encoder)

    def test_target_mlp_default_true(self):
        args = _parse_fixed()
        self.assertTrue(args.target_mlp)

    def test_target_mlp_disable(self):
        args = _parse_fixed(["--no-target-mlp"])
        self.assertFalse(args.target_mlp)


# ===========================================================================
# Branding
# ===========================================================================

class TestBranding(unittest.TestCase):
    """Verify parser description uses Side-Step branding."""

    def test_description(self):
        from sidestep_engine.cli.args import build_root_parser
        parser = build_root_parser()
        self.assertIn("Side-Step", parser.description)
        self.assertNotIn("ACE-Step Training V2", parser.description)


# ===========================================================================
# Review table enrichment
# ===========================================================================

class TestReviewTableEnrichment(unittest.TestCase):
    """Verify the enriched wizard review table has all expected groups."""

    def test_groups_include_corrected_training(self):
        from sidestep_engine.ui.flows.review_summary import _build_groups
        groups = _build_groups({"adapter_type": "lora"})
        names = [n for n, _ in groups]
        self.assertIn("Corrected Training", names)

    def test_training_group_has_effective_batch(self):
        from sidestep_engine.ui.flows.review_summary import _build_groups
        groups = _build_groups({})
        training_keys = []
        for name, keys in groups:
            if name == "Training":
                training_keys = [k for k, _ in keys]
        self.assertIn("_effective_batch", training_keys)
        self.assertIn("shift", training_keys)
        self.assertIn("num_inference_steps", training_keys)
        self.assertIn("dataset_repeats", training_keys)

    def test_checkpointing_has_run_name(self):
        from sidestep_engine.ui.flows.review_summary import _build_groups
        groups = _build_groups({})
        ckpt_keys = []
        for name, keys in groups:
            if name == "Checkpointing":
                ckpt_keys = [k for k, _ in keys]
        self.assertIn("run_name", ckpt_keys)

    def test_logging_has_log_dir(self):
        from sidestep_engine.ui.flows.review_summary import _build_groups
        groups = _build_groups({})
        log_keys = []
        for name, keys in groups:
            if name == "Logging":
                log_keys = [k for k, _ in keys]
        self.assertIn("log_dir", log_keys)

    def test_lora_has_bias(self):
        from sidestep_engine.ui.flows.review_summary import _build_groups
        groups = _build_groups({"adapter_type": "lora"})
        lora_keys = []
        for name, keys in groups:
            if name == "LoRA":
                lora_keys = [k for k, _ in keys]
        self.assertIn("bias", lora_keys)


class TestReviewNonDefaultHighlight(unittest.TestCase):
    """Verify _is_default correctly identifies non-default values."""

    def test_default_lr(self):
        from sidestep_engine.ui.flows.review_summary import _is_default
        self.assertTrue(_is_default("learning_rate", 1e-4))

    def test_nondefault_lr(self):
        from sidestep_engine.ui.flows.review_summary import _is_default
        self.assertFalse(_is_default("learning_rate", 5e-5))

    def test_unknown_key_is_default(self):
        from sidestep_engine.ui.flows.review_summary import _is_default
        self.assertTrue(_is_default("_unknown_key", "anything"))

    def test_default_seed(self):
        from sidestep_engine.ui.flows.review_summary import _is_default
        self.assertTrue(_is_default("seed", 42))
        self.assertFalse(_is_default("seed", 123))


class TestReviewResolve(unittest.TestCase):
    """Verify _resolve handles computed keys."""

    def test_effective_batch(self):
        from sidestep_engine.ui.flows.review_summary import _resolve
        answers = {"batch_size": 2, "gradient_accumulation": 4}
        self.assertEqual(_resolve("_effective_batch", answers), 8)

    def test_effective_batch_defaults(self):
        from sidestep_engine.ui.flows.review_summary import _resolve
        self.assertEqual(_resolve("_effective_batch", {}), 4)

    def test_regular_key(self):
        from sidestep_engine.ui.flows.review_summary import _resolve
        answers = {"epochs": 50}
        self.assertEqual(_resolve("epochs", answers), 50)

    def test_missing_key(self):
        from sidestep_engine.ui.flows.review_summary import _resolve
        self.assertIsNone(_resolve("epochs", {}))


if __name__ == "__main__":
    unittest.main()
