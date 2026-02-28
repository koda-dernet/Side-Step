"""Tests for CLI path validation and target module resolution."""

from __future__ import annotations

import tempfile
import unittest
from argparse import Namespace
from pathlib import Path

from acestep.training_v2.cli.validation import (
    resolve_target_modules,
    validate_paths,
)


class TestResolveTargetModules(unittest.TestCase):
    """Test target module resolution logic."""

    def test_attention_type_both_no_per_type_lists(self):
        """When attention_type is 'both' and no per-type lists provided, return unchanged."""
        result = resolve_target_modules(
            target_modules=["q_proj", "v_proj"],
            attention_type="both",
            self_target_modules=None,
            cross_target_modules=None,
            target_mlp=False,
        )
        self.assertEqual(result, ["q_proj", "v_proj"])

    def test_attention_type_both_with_per_type_lists(self):
        """When attention_type is 'both' with per-type lists, prefix each independently."""
        result = resolve_target_modules(
            target_modules=["q_proj"],
            attention_type="both",
            self_target_modules=["q_proj", "v_proj"],
            cross_target_modules=["q_proj"],
            target_mlp=False,
        )
        expected = ["self_attn.q_proj", "self_attn.v_proj", "cross_attn.q_proj"]
        self.assertEqual(result, expected)

    def test_attention_type_self(self):
        """When attention_type is 'self', prefix with self_attn."""
        result = resolve_target_modules(
            target_modules=["q_proj", "v_proj"],
            attention_type="self",
            self_target_modules=None,
            cross_target_modules=None,
            target_mlp=False,
        )
        self.assertEqual(result, ["self_attn.q_proj", "self_attn.v_proj"])

    def test_attention_type_cross(self):
        """When attention_type is 'cross', prefix with cross_attn."""
        result = resolve_target_modules(
            target_modules=["q_proj", "v_proj"],
            attention_type="cross",
            self_target_modules=None,
            cross_target_modules=None,
            target_mlp=False,
        )
        self.assertEqual(result, ["cross_attn.q_proj", "cross_attn.v_proj"])

    def test_target_mlp_adds_mlp_modules(self):
        """When target_mlp is True, MLP modules should be appended."""
        result = resolve_target_modules(
            target_modules=["q_proj"],
            attention_type="self",
            self_target_modules=None,
            cross_target_modules=None,
            target_mlp=True,
        )
        expected = ["self_attn.q_proj", "gate_proj", "up_proj", "down_proj"]
        self.assertEqual(result, expected)

    def test_target_mlp_does_not_duplicate(self):
        """MLP modules should not be duplicated if already present."""
        result = resolve_target_modules(
            target_modules=["q_proj", "gate_proj"],
            attention_type="self",
            self_target_modules=None,
            cross_target_modules=None,
            target_mlp=True,
        )
        expected = ["self_attn.q_proj", "self_attn.gate_proj", "gate_proj", "up_proj", "down_proj"]
        self.assertEqual(result, expected)

    def test_prefix_modules_skips_already_qualified(self):
        """Modules with dots should not be re-prefixed."""
        result = resolve_target_modules(
            target_modules=["self_attn.q_proj", "v_proj"],
            attention_type="self",
            self_target_modules=None,
            cross_target_modules=None,
            target_mlp=False,
        )
        # Already qualified names should be left alone
        self.assertEqual(result, ["self_attn.q_proj", "self_attn.v_proj"])

    def test_unknown_attention_type_returns_unchanged(self):
        """Unknown attention_type should return target_modules unchanged."""
        result = resolve_target_modules(
            target_modules=["q_proj", "v_proj"],
            attention_type="unknown",
            self_target_modules=None,
            cross_target_modules=None,
            target_mlp=False,
        )
        self.assertEqual(result, ["q_proj", "v_proj"])

    def test_empty_target_modules(self):
        """Empty target_modules list should return empty (or just MLP if enabled)."""
        result = resolve_target_modules(
            target_modules=[],
            attention_type="self",
            self_target_modules=None,
            cross_target_modules=None,
            target_mlp=False,
        )
        self.assertEqual(result, [])

    def test_empty_target_modules_with_mlp(self):
        """Empty target_modules with target_mlp should only return MLP modules."""
        result = resolve_target_modules(
            target_modules=[],
            attention_type="self",
            self_target_modules=None,
            cross_target_modules=None,
            target_mlp=True,
        )
        self.assertEqual(result, ["gate_proj", "up_proj", "down_proj"])


class TestValidatePaths(unittest.TestCase):
    """Test path validation logic."""

    def test_validate_paths_missing_checkpoint_dir(self):
        """Should fail when checkpoint directory does not exist."""
        with tempfile.TemporaryDirectory() as tmpdir:
            args = Namespace(
                subcommand="train",
                checkpoint_dir=str(Path(tmpdir) / "nonexistent"),
                model_variant="turbo",
                dataset_dir=tmpdir,
                resume_from=None,
            )
            result = validate_paths(args)
            self.assertFalse(result)

    def test_validate_paths_missing_model_variant(self):
        """Should fail when model variant directory does not exist."""
        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt_dir = Path(tmpdir) / "ckpts"
            ckpt_dir.mkdir()
            args = Namespace(
                subcommand="train",
                checkpoint_dir=str(ckpt_dir),
                model_variant="nonexistent_variant",
                dataset_dir=tmpdir,
                resume_from=None,
            )
            result = validate_paths(args)
            self.assertFalse(result)

    def test_validate_paths_success_with_variant(self):
        """Should succeed when all paths are valid."""
        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt_dir = Path(tmpdir) / "ckpts"
            variant_dir = ckpt_dir / "acestep-v15-turbo"
            variant_dir.mkdir(parents=True)
            ds_dir = Path(tmpdir) / "dataset"
            ds_dir.mkdir()

            args = Namespace(
                subcommand="train",
                checkpoint_dir=str(ckpt_dir),
                model_variant="turbo",
                dataset_dir=str(ds_dir),
                resume_from=None,
            )
            result = validate_paths(args)
            self.assertTrue(result)
            self.assertEqual(args.model_dir, variant_dir)

    def test_validate_paths_missing_dataset_dir(self):
        """Should fail when dataset directory does not exist."""
        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt_dir = Path(tmpdir) / "ckpts"
            variant_dir = ckpt_dir / "acestep-v15-turbo"
            variant_dir.mkdir(parents=True)

            args = Namespace(
                subcommand="train",
                checkpoint_dir=str(ckpt_dir),
                model_variant="turbo",
                dataset_dir=str(Path(tmpdir) / "nonexistent"),
                resume_from=None,
            )
            result = validate_paths(args)
            self.assertFalse(result)

    def test_validate_paths_missing_resume_checkpoint(self):
        """Should fail when resume_from path does not exist."""
        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt_dir = Path(tmpdir) / "ckpts"
            variant_dir = ckpt_dir / "acestep-v15-turbo"
            variant_dir.mkdir(parents=True)
            ds_dir = Path(tmpdir) / "dataset"
            ds_dir.mkdir()

            args = Namespace(
                subcommand="train",
                checkpoint_dir=str(ckpt_dir),
                model_variant="turbo",
                dataset_dir=str(ds_dir),
                resume_from=str(Path(tmpdir) / "nonexistent.ckpt"),
            )
            result = validate_paths(args)
            self.assertFalse(result)

    def test_validate_paths_empty_resume_from_ignored(self):
        """Empty or whitespace-only resume_from should be ignored."""
        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt_dir = Path(tmpdir) / "ckpts"
            variant_dir = ckpt_dir / "acestep-v15-turbo"
            variant_dir.mkdir(parents=True)
            ds_dir = Path(tmpdir) / "dataset"
            ds_dir.mkdir()

            args = Namespace(
                subcommand="train",
                checkpoint_dir=str(ckpt_dir),
                model_variant="turbo",
                dataset_dir=str(ds_dir),
                resume_from="  ",
            )
            result = validate_paths(args)
            self.assertTrue(result)

    def test_validate_paths_compare_configs_subcommand(self):
        """compare-configs subcommand should only validate config files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg1 = Path(tmpdir) / "config1.json"
            cfg2 = Path(tmpdir) / "config2.json"
            cfg1.write_text('{"test": 1}')
            cfg2.write_text('{"test": 2}')

            args = Namespace(
                subcommand="compare-configs",
                configs=[str(cfg1), str(cfg2)],
            )
            result = validate_paths(args)
            self.assertTrue(result)

    def test_validate_paths_compare_configs_missing_file(self):
        """compare-configs should fail if any config file is missing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg1 = Path(tmpdir) / "config1.json"
            cfg1.write_text('{"test": 1}')

            args = Namespace(
                subcommand="compare-configs",
                configs=[str(cfg1), str(Path(tmpdir) / "nonexistent.json")],
            )
            result = validate_paths(args)
            self.assertFalse(result)

    def test_validate_paths_custom_variant_directory_name(self):
        """Should support custom variant folder names (not in VARIANT_DIR_MAP)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt_dir = Path(tmpdir) / "ckpts"
            custom_dir = ckpt_dir / "my_custom_model"
            custom_dir.mkdir(parents=True)
            ds_dir = Path(tmpdir) / "dataset"
            ds_dir.mkdir()

            args = Namespace(
                subcommand="train",
                checkpoint_dir=str(ckpt_dir),
                model_variant="my_custom_model",
                dataset_dir=str(ds_dir),
                resume_from=None,
            )
            result = validate_paths(args)
            self.assertTrue(result)
            self.assertEqual(args.model_dir, custom_dir)


if __name__ == "__main__":
    unittest.main()