"""Tests for wizard UX refinements and validation cleanup.

Covers:
- Gradient estimation removal (no lingering imports)
- validate_fn in ask()
- Settings schema v5 (new output dir fields)
- Per-sample repeat removal + global dataset_repeats
- Run name generation
- Training step helpers (estimate_total_steps, warmup ratio)
- allow_back defaults
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch, MagicMock


class TestEstimateNuked(unittest.TestCase):
    """Verify gradient estimation code is fully removed."""

    def test_no_estimate_subcommand_in_args(self) -> None:
        """The 'estimate' argparse subcommand should be gone."""
        from sidestep_engine.cli.args import build_root_parser
        parser = build_root_parser()
        # Check no 'estimate' in subparser choices
        for action in parser._subparsers._actions:
            if hasattr(action, "choices") and action.choices:
                self.assertNotIn("estimate", action.choices)

    def test_no_estimate_fields_in_config(self) -> None:
        """TrainingConfigV2 should not have estimate_batches, etc."""
        from sidestep_engine.core.configs import TrainingConfigV2
        cfg = TrainingConfigV2(learning_rate=1e-4, batch_size=1)
        self.assertFalse(hasattr(cfg, "estimate_batches"))
        self.assertFalse(hasattr(cfg, "auto_estimate"))
        self.assertFalse(hasattr(cfg, "estimate_output"))

    def test_no_wizard_estimate_in_flows(self) -> None:
        """wizard_estimate should not be importable from flows."""
        from sidestep_engine.ui.flows import __all__ as flows_all
        self.assertNotIn("wizard_estimate", flows_all)


class TestValidateFn(unittest.TestCase):
    """Test the new validate_fn parameter on ask()."""

    @patch("sidestep_engine.ui.prompt_helpers.is_rich_active", return_value=False)
    def test_validate_fn_rejects_bad_input(self, _mock_rich) -> None:
        """validate_fn returning a string should reject and re-prompt."""
        from sidestep_engine.ui.prompt_helpers import ask

        calls = iter(["0.0", "0.001"])
        with patch("builtins.input", side_effect=calls):
            result = ask(
                "LR", default=None, required=True, type_fn=float,
                allow_back=False,
                validate_fn=lambda v: "Must be > 0" if v <= 0 else None,
            )
        self.assertAlmostEqual(result, 0.001)

    @patch("sidestep_engine.ui.prompt_helpers.is_rich_active", return_value=False)
    def test_validate_fn_accepts_good_input(self, _mock_rich) -> None:
        """validate_fn returning None should accept the value."""
        from sidestep_engine.ui.prompt_helpers import ask

        with patch("builtins.input", return_value="0.001"):
            result = ask(
                "LR", type_fn=float, allow_back=False,
                validate_fn=lambda v: None,
            )
        self.assertAlmostEqual(result, 0.001)


class TestSettingsSchemaV5(unittest.TestCase):
    """Settings schema includes directory fields and history root memory."""

    def test_default_settings_have_new_fields(self) -> None:
        from sidestep_engine.settings import _default_settings
        d = _default_settings()
        self.assertIn("trained_adapters_dir", d)
        self.assertIn("preprocessed_tensors_dir", d)
        self.assertIn("history_output_roots", d)
        self.assertIsNone(d["trained_adapters_dir"])
        self.assertIsNone(d["preprocessed_tensors_dir"])
        self.assertEqual(d["history_output_roots"], [])

    def test_schema_version_is_7(self) -> None:
        from sidestep_engine.settings import _SCHEMA_VERSION
        self.assertEqual(_SCHEMA_VERSION, 7)

    def test_get_trained_adapters_dir_default(self) -> None:
        from sidestep_engine.settings import get_trained_adapters_dir
        with patch("sidestep_engine.settings.load_settings", return_value=None):
            self.assertEqual(get_trained_adapters_dir(), "./trained_adapters")

    def test_get_preprocessed_tensors_dir_default(self) -> None:
        from sidestep_engine.settings import get_preprocessed_tensors_dir
        with patch("sidestep_engine.settings.load_settings", return_value=None):
            self.assertEqual(get_preprocessed_tensors_dir(), "./preprocessed_tensors")

    def test_get_trained_adapters_dir_from_settings(self) -> None:
        from sidestep_engine.settings import get_trained_adapters_dir
        with patch("sidestep_engine.settings.load_settings",
                   return_value={"trained_adapters_dir": "/custom/adapters"}):
            self.assertEqual(get_trained_adapters_dir(), "/custom/adapters")

    def test_get_history_output_roots_default(self) -> None:
        from sidestep_engine.settings import get_history_output_roots
        with patch("sidestep_engine.settings.load_settings", return_value=None):
            self.assertEqual(get_history_output_roots(), [])

    def test_remember_history_output_root_dedupes(self) -> None:
        from sidestep_engine import settings as settings_mod

        defaults = settings_mod._default_settings()
        with patch("sidestep_engine.settings.load_settings", return_value=defaults.copy()), \
             patch("sidestep_engine.settings.save_settings") as mock_save:
            settings_mod.remember_history_output_root("/tmp/override")

        mock_save.assert_called_once()
        saved = mock_save.call_args.args[0]
        self.assertEqual(saved["history_output_roots"], [str(Path("/tmp/override").resolve(strict=False))])


class TestEffectiveLogDirDefaults(unittest.TestCase):
    """TensorBoard log dirs should remain run-local when GUI sends empty log_dir."""

    def test_empty_log_dir_uses_output_runs_run_name(self) -> None:
        from sidestep_engine.core.configs import TrainingConfigV2

        with tempfile.TemporaryDirectory() as td:
            cfg = TrainingConfigV2(
                learning_rate=1e-4,
                batch_size=1,
                output_dir=td,
                run_name="lora_turbo_20260226_2100",
                log_dir="",
            )
            log_dir = cfg.effective_log_dir

        self.assertEqual(log_dir, Path(td) / "runs" / "lora_turbo_20260226_2100")

    def test_custom_log_dir_remains_versioned(self) -> None:
        from sidestep_engine.core.configs import TrainingConfigV2

        with tempfile.TemporaryDirectory() as td:
            cfg = TrainingConfigV2(
                learning_rate=1e-4,
                batch_size=1,
                output_dir=td,
                run_name="lora_turbo_20260226_2100",
                log_dir=str(Path(td) / "tb_custom"),
            )
            log_dir = cfg.effective_log_dir

        self.assertTrue(log_dir.name.startswith("lora_turbo_20260226_2100_v"))


class TestDatasetRepeats(unittest.TestCase):
    """Global dataset_repeats replaces per-sample metadata.repeat."""

    def test_config_has_dataset_repeats(self) -> None:
        from sidestep_engine.core.configs import TrainingConfigV2
        cfg = TrainingConfigV2(learning_rate=1e-4, batch_size=1)
        self.assertEqual(cfg.dataset_repeats, 1)

    def test_config_to_dict_includes_dataset_repeats(self) -> None:
        from sidestep_engine.core.configs import TrainingConfigV2
        cfg = TrainingConfigV2(learning_rate=1e-4, batch_size=1, dataset_repeats=3)
        d = cfg.to_dict()
        self.assertEqual(d["dataset_repeats"], 3)

    def test_dataset_repeats_multiplies_samples(self) -> None:
        """PreprocessedTensorDataset with dataset_repeats=3 should triple valid_paths."""
        import torch
        with tempfile.TemporaryDirectory() as td:
            # Create 2 dummy .pt files
            for i in range(2):
                p = Path(td) / f"sample_{i}.pt"
                torch.save({
                    "target_latents": torch.zeros(10, 64),
                    "attention_mask": torch.ones(10),
                    "text_hidden_states": torch.zeros(1, 768),
                    "text_attention_mask": torch.ones(1),
                    "lyric_hidden_states": torch.zeros(1, 768),
                    "lyric_attention_mask": torch.ones(1),
                    "metadata": {"duration": 30, "repeat": 5},  # repeat=5 should be IGNORED
                }, p)

            from sidestep_engine.vendor.data_module import PreprocessedTensorDataset
            ds = PreprocessedTensorDataset(td, dataset_repeats=3)
            # Should have 2*3=6 entries (ignoring per-sample repeat=5)
            self.assertEqual(len(ds.valid_paths), 6)
            self.assertEqual(ds._unique_count, 2)

    def test_dataset_repeats_default_no_multiply(self) -> None:
        """dataset_repeats=1 should not duplicate."""
        import torch
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "sample.pt"
            torch.save({
                "target_latents": torch.zeros(10, 64),
                "attention_mask": torch.ones(10),
                "text_hidden_states": torch.zeros(1, 768),
                "text_attention_mask": torch.ones(1),
                "lyric_hidden_states": torch.zeros(1, 768),
                "lyric_attention_mask": torch.ones(1),
                "metadata": {"duration": 30, "repeat": 3},
            }, p)

            from sidestep_engine.vendor.data_module import PreprocessedTensorDataset
            ds = PreprocessedTensorDataset(td, dataset_repeats=1)
            self.assertEqual(len(ds.valid_paths), 1)


class TestRunNameGeneration(unittest.TestCase):
    """Run name auto-generation includes adapter type, variant, and timestamp."""

    def test_generate_run_base_format(self) -> None:
        from sidestep_engine.ui.flows.train_steps_required import _generate_run_base, _stamp_run_name
        a = {"adapter_type": "lora", "model_variant": "turbo"}
        base = _generate_run_base(a)
        self.assertEqual(base, "lora_turbo")
        name = _stamp_run_name(base)
        self.assertTrue(name.startswith("lora_turbo_"))
        # Should have timestamp like 20260223_1320
        parts = name.split("_")
        self.assertEqual(len(parts), 4)  # lora, turbo, date, time

    def test_generate_run_base_lokr(self) -> None:
        from sidestep_engine.ui.flows.train_steps_required import _generate_run_base, _stamp_run_name
        a = {"adapter_type": "lokr", "model_variant": "base"}
        base = _generate_run_base(a)
        self.assertEqual(base, "lokr_base")
        name = _stamp_run_name(base)
        self.assertTrue(name.startswith("lokr_base_"))


class TestTrainingStepHelpers(unittest.TestCase):
    """Test extracted training step estimation helpers."""

    def test_estimate_total_steps_with_repeats(self) -> None:
        from sidestep_engine.ui.flows.train_steps_helpers import estimate_total_steps
        with tempfile.TemporaryDirectory() as td:
            for i in range(10):
                (Path(td) / f"s{i}.pt").touch()
            a = {
                "dataset_dir": td, "batch_size": 1,
                "gradient_accumulation": 1, "epochs": 10,
                "dataset_repeats": 2, "max_steps": 0,
            }
            result = estimate_total_steps(a)
            # 10 samples * 2 repeats / (1*1) = 20 steps/epoch * 10 epochs = 200
            self.assertEqual(result, 200)

    def test_estimate_total_steps_max_steps_override(self) -> None:
        from sidestep_engine.ui.flows.train_steps_helpers import estimate_total_steps
        with tempfile.TemporaryDirectory() as td:
            for i in range(10):
                (Path(td) / f"s{i}.pt").touch()
            a = {
                "dataset_dir": td, "batch_size": 1,
                "gradient_accumulation": 1, "epochs": 100,
                "dataset_repeats": 1, "max_steps": 50,
            }
            result = estimate_total_steps(a)
            self.assertEqual(result, 50)

    def test_smart_save_best_default(self) -> None:
        from sidestep_engine.ui.flows.train_steps_helpers import smart_save_best_default
        a = {"warmup_steps": 100}
        result = smart_save_best_default(a)
        self.assertGreaterEqual(result, 110)

    def test_warn_warmup_ratio_no_crash(self) -> None:
        """warn_warmup_ratio should not crash even with empty dataset."""
        from sidestep_engine.ui.flows.train_steps_helpers import warn_warmup_ratio
        a = {"warmup_steps": 100}
        # Should not raise
        warn_warmup_ratio(a)


class TestAllowBackDefaults(unittest.TestCase):
    """All prompt helpers now default to allow_back=True."""

    def test_ask_default_allow_back(self) -> None:
        import inspect
        from sidestep_engine.ui.prompt_helpers import ask
        sig = inspect.signature(ask)
        self.assertTrue(sig.parameters["allow_back"].default)

    def test_menu_default_allow_back(self) -> None:
        import inspect
        from sidestep_engine.ui.prompt_helpers import menu
        sig = inspect.signature(menu)
        self.assertTrue(sig.parameters["allow_back"].default)

    def test_ask_path_default_allow_back(self) -> None:
        import inspect
        from sidestep_engine.ui.prompt_helpers import ask_path
        sig = inspect.signature(ask_path)
        self.assertTrue(sig.parameters["allow_back"].default)

    def test_ask_output_path_default_allow_back(self) -> None:
        import inspect
        from sidestep_engine.ui.prompt_helpers import ask_output_path
        sig = inspect.signature(ask_output_path)
        self.assertTrue(sig.parameters["allow_back"].default)

    def test_ask_bool_default_allow_back(self) -> None:
        import inspect
        from sidestep_engine.ui.prompt_helpers import ask_bool
        sig = inspect.signature(ask_bool)
        self.assertTrue(sig.parameters["allow_back"].default)


if __name__ == "__main__":
    unittest.main()
