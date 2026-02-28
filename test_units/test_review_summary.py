"""Tests for review_summary module.

Covers:
- CLI command builder (build_cli_command).
- CLI command file save (save_cli_command).
- Review table group builder (_build_groups).
- Value formatter (_fmt).
"""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path


class TestBuildCliCommand(unittest.TestCase):
    """Verify build_cli_command produces correct CLI strings."""

    def test_basic_command(self):
        from sidestep_engine.ui.flows.review_summary import build_cli_command

        answers = {
            "checkpoint_dir": "/ckpt",
            "dataset_dir": "/data",
            "output_dir": "/out",
            "learning_rate": 1e-4,
            "epochs": 100,
        }
        cmd = build_cli_command(answers)
        self.assertIn("python train.py fixed", cmd)
        self.assertIn("--checkpoint-dir /ckpt", cmd)
        self.assertIn("--dataset-dir /data", cmd)
        self.assertIn("--output-dir /out", cmd)
        self.assertIn("--lr 0.0001", cmd)
        self.assertIn("--epochs 100", cmd)

    def test_skips_internal_keys(self):
        from sidestep_engine.ui.flows.review_summary import build_cli_command

        answers = {
            "_fisher_map_detected": True,
            "_pp_recommended": False,
            "config_mode": "basic",
            "checkpoint_dir": "/ckpt",
        }
        cmd = build_cli_command(answers)
        self.assertNotIn("_fisher", cmd)
        self.assertNotIn("config-mode", cmd)
        self.assertIn("--checkpoint-dir", cmd)

    def test_bool_handling(self):
        from sidestep_engine.ui.flows.review_summary import build_cli_command

        answers = {"save_best": True, "target_mlp": False}
        cmd = build_cli_command(answers)
        self.assertIn("--save-best", cmd)
        self.assertIn("--no-target-mlp", cmd)

    def test_none_values_skipped(self):
        from sidestep_engine.ui.flows.review_summary import build_cli_command

        answers = {"resume_from": None, "epochs": 100}
        cmd = build_cli_command(answers)
        self.assertNotIn("resume", cmd)
        self.assertIn("--epochs", cmd)

    def test_list_handling(self):
        from sidestep_engine.ui.flows.review_summary import build_cli_command

        answers = {"target_modules": ["q_proj", "v_proj"]}
        cmd = build_cli_command(answers)
        self.assertIn("--target-modules q_proj v_proj", cmd)


class TestSaveCliCommand(unittest.TestCase):
    """Verify save_cli_command writes to the output directory."""

    def test_saves_file(self):
        from sidestep_engine.ui.flows.review_summary import save_cli_command

        d = tempfile.mkdtemp(prefix="sidestep_test_cli_")
        try:
            answers = {"output_dir": d, "epochs": 50, "learning_rate": 1e-4}
            path = save_cli_command(answers)
            self.assertIsNotNone(path)
            self.assertTrue(path.exists())
            content = path.read_text(encoding="utf-8")
            self.assertIn("python train.py fixed", content)
            self.assertIn("--epochs 50", content)
        finally:
            import shutil
            shutil.rmtree(d, ignore_errors=True)

    def test_no_output_dir(self):
        from sidestep_engine.ui.flows.review_summary import save_cli_command

        result = save_cli_command({})
        self.assertIsNone(result)


class TestBuildGroups(unittest.TestCase):
    """Verify _build_groups returns the right adapter section."""

    def test_lora_groups(self):
        from sidestep_engine.ui.flows.review_summary import _build_groups

        groups = _build_groups({"adapter_type": "lora"})
        group_names = [name for name, _ in groups]
        self.assertIn("LoRA", group_names)
        self.assertNotIn("LoKR", group_names)

    def test_lokr_groups(self):
        from sidestep_engine.ui.flows.review_summary import _build_groups

        groups = _build_groups({"adapter_type": "lokr"})
        group_names = [name for name, _ in groups]
        self.assertIn("LoKR", group_names)
        self.assertNotIn("LoRA", group_names)

    def test_default_is_lora(self):
        from sidestep_engine.ui.flows.review_summary import _build_groups

        groups = _build_groups({})
        group_names = [name for name, _ in groups]
        self.assertIn("LoRA", group_names)


class TestFmt(unittest.TestCase):
    """Verify _fmt value formatting."""

    def test_none(self):
        from sidestep_engine.ui.flows.review_summary import _fmt
        self.assertEqual(_fmt(None), "(auto)")

    def test_bool(self):
        from sidestep_engine.ui.flows.review_summary import _fmt
        self.assertEqual(_fmt(True), "yes")
        self.assertEqual(_fmt(False), "no")

    def test_small_float(self):
        from sidestep_engine.ui.flows.review_summary import _fmt
        self.assertEqual(_fmt(1e-4), "1.0e-04")

    def test_normal_float(self):
        from sidestep_engine.ui.flows.review_summary import _fmt
        self.assertEqual(_fmt(0.1), "0.1")

    def test_list(self):
        from sidestep_engine.ui.flows.review_summary import _fmt
        self.assertEqual(_fmt(["a", "b"]), "a, b")

    def test_int(self):
        from sidestep_engine.ui.flows.review_summary import _fmt
        self.assertEqual(_fmt(42), "42")


class TestSmartSaveBestDefault(unittest.TestCase):
    """Verify the updated _smart_save_best_default accounts for warmup."""

    def test_warmup_buffer(self):
        from sidestep_engine.ui.flows.train_steps import _smart_save_best_default

        answers = {
            "dataset_dir": "/nonexistent",
            "warmup_steps": 50,
            "batch_size": 1,
            "gradient_accumulation": 4,
            "epochs": 100,
        }
        result = _smart_save_best_default(answers)
        # Should be at least warmup + 10
        self.assertGreaterEqual(result, 60)

    def test_unknown_dataset_uses_warmup(self):
        from sidestep_engine.ui.flows.train_steps import _smart_save_best_default

        answers = {"warmup_steps": 100}
        result = _smart_save_best_default(answers)
        self.assertGreaterEqual(result, 110)


if __name__ == "__main__":
    unittest.main()
