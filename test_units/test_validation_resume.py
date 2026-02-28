"""Tests for CLI path validation, especially resume path."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import argparse


class TestValidatePathsResume(unittest.TestCase):
    """Validate that missing resume path fails (does not warn and proceed)."""

    def test_resume_path_missing_fails(self) -> None:
        from sidestep_engine.cli.validation import validate_paths

        with tempfile.TemporaryDirectory() as td:
            ckpt = Path(td) / "checkpoints"
            model_dir = ckpt / "acestep-v15-turbo"
            ds_dir = Path(td) / "dataset"
            model_dir.mkdir(parents=True)
            ds_dir.mkdir()
            (model_dir / "config.json").write_text('{"model_type": "test"}')
            (ds_dir / "x.pt").write_bytes(b"dummy")

            args = argparse.Namespace(
                subcommand="train",
                checkpoint_dir=str(ckpt),
                model_variant="acestep-v15-turbo",
                dataset_dir=str(ds_dir),
                resume_from=str(Path(td) / "nonexistent_resume"),
            )
            self.assertFalse(validate_paths(args))

    def test_resume_path_empty_ok(self) -> None:
        from sidestep_engine.cli.validation import validate_paths

        with tempfile.TemporaryDirectory() as td:
            ckpt = Path(td) / "checkpoints"
            model_dir = ckpt / "acestep-v15-turbo"
            ds_dir = Path(td) / "dataset"
            model_dir.mkdir(parents=True)
            ds_dir.mkdir()
            (model_dir / "config.json").write_text('{"model_type": "test"}')
            (ds_dir / "x.pt").write_bytes(b"dummy")

            args = argparse.Namespace(
                subcommand="train",
                checkpoint_dir=str(ckpt),
                model_variant="acestep-v15-turbo",
                dataset_dir=str(ds_dir),
                resume_from=None,
            )
            self.assertTrue(validate_paths(args))

    def test_resume_path_whitespace_only_does_not_fail(self) -> None:
        from sidestep_engine.cli.validation import validate_paths

        with tempfile.TemporaryDirectory() as td:
            ckpt = Path(td) / "checkpoints"
            model_dir = ckpt / "acestep-v15-turbo"
            ds_dir = Path(td) / "dataset"
            model_dir.mkdir(parents=True)
            ds_dir.mkdir()
            (model_dir / "config.json").write_text('{"model_type": "test"}')
            (ds_dir / "x.pt").write_bytes(b"dummy")

            args = argparse.Namespace(
                subcommand="train",
                checkpoint_dir=str(ckpt),
                model_variant="acestep-v15-turbo",
                dataset_dir=str(ds_dir),
                resume_from="   ",
            )
            self.assertTrue(validate_paths(args))

    def test_resume_path_exists_ok(self) -> None:
        from sidestep_engine.cli.validation import validate_paths

        with tempfile.TemporaryDirectory() as td:
            ckpt = Path(td) / "checkpoints"
            model_dir = ckpt / "acestep-v15-turbo"
            ds_dir = Path(td) / "dataset"
            resume_dir = Path(td) / "resume_ckpt"
            model_dir.mkdir(parents=True)
            ds_dir.mkdir()
            resume_dir.mkdir()
            (model_dir / "config.json").write_text('{"model_type": "test"}')
            (ds_dir / "x.pt").write_bytes(b"dummy")

            args = argparse.Namespace(
                subcommand="train",
                checkpoint_dir=str(ckpt),
                model_variant="acestep-v15-turbo",
                dataset_dir=str(ds_dir),
                resume_from=str(resume_dir),
            )
            self.assertTrue(validate_paths(args))


if __name__ == "__main__":
    unittest.main()
