"""
Tests for CLI cleanup: --config, --preprocess chain, --base-model deprecation,
relaxed required args, --gui flag.
"""

from __future__ import annotations

import argparse
import json
import warnings
from pathlib import Path
from unittest import mock

import pytest

from sidestep_engine.cli.args import build_root_parser
from sidestep_engine.cli.config_builder import (
    _apply_config_file,
    _populate_defaults_cache,
    _warn_deprecated_base_model,
    _DEFAULTS_CACHE,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_fixed(*extra_args: str) -> argparse.Namespace:
    """Parse args for the 'train' subcommand with sensible test defaults."""
    parser = build_root_parser()
    base = [
        "train",
        "--checkpoint-dir", "/tmp/ckpts",
        "--dataset-dir", "/tmp/tensors",
        "--output-dir", "/tmp/output",
    ]
    return parser.parse_args(list(base) + list(extra_args))


# ---------------------------------------------------------------------------
# 1. --base-model deprecation
# ---------------------------------------------------------------------------

class TestBaseModelDeprecation:

    def test_base_model_flag_removed(self):
        """--base-model should no longer exist in the train subparser."""
        parser = build_root_parser()
        for action in parser._subparsers._actions:
            if hasattr(action, "_parser_class"):
                for name, sub in action.choices.items():
                    if name == "train":
                        for a in sub._actions:
                            if "--base-model" in getattr(a, "option_strings", []):
                                pytest.fail("--base-model should have been removed")

    def test_deprecated_warn_is_noop(self):
        """_warn_deprecated_base_model is now a no-op stub."""
        args = argparse.Namespace(base_model="base")
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            _warn_deprecated_base_model(args)
            dep_warnings = [x for x in w if issubclass(x.category, DeprecationWarning)]
            assert len(dep_warnings) == 0


# ---------------------------------------------------------------------------
# 2. Relaxed required args
# ---------------------------------------------------------------------------

class TestRelaxedRequiredArgs:

    def test_checkpoint_dir_not_required(self):
        """--checkpoint-dir should parse as None when omitted."""
        parser = build_root_parser()
        args = parser.parse_args([
            "train", "--dataset-dir", "/tmp/t", "--output-dir", "/tmp/o",
        ])
        assert args.checkpoint_dir is None

    def test_dataset_dir_not_required(self):
        """--dataset-dir should parse as None when omitted."""
        parser = build_root_parser()
        args = parser.parse_args([
            "train", "--checkpoint-dir", "/tmp/c", "--output-dir", "/tmp/o",
        ])
        assert args.dataset_dir is None

    def test_output_dir_not_required(self):
        """--output-dir should parse as None when omitted."""
        parser = build_root_parser()
        args = parser.parse_args([
            "train", "--checkpoint-dir", "/tmp/c", "--dataset-dir", "/tmp/t",
        ])
        assert args.output_dir is None


# ---------------------------------------------------------------------------
# 3. --config file loading
# ---------------------------------------------------------------------------

class TestConfigFile:

    def test_config_arg_exists(self):
        """--config should be a valid arg on the fixed subparser."""
        args = _parse_fixed()
        assert hasattr(args, "config")
        assert args.config is None

    def test_config_file_applied(self, tmp_path):
        """JSON config values should fill unset args."""
        config = {
            "checkpoint_dir": "/from/json/ckpts",
            "learning_rate": 5e-5,
            "epochs": 200,
            "rank": 32,
        }
        config_file = tmp_path / "config.json"
        config_file.write_text(json.dumps(config))

        parser = build_root_parser()
        # Only provide subcommand — everything else from JSON
        args = parser.parse_args(["train"])
        args.config = str(config_file)

        _populate_defaults_cache()
        _apply_config_file(args)

        assert args.checkpoint_dir == "/from/json/ckpts"
        assert args.learning_rate == 5e-5
        assert args.epochs == 200
        assert args.rank == 32

    def test_cli_overrides_config(self, tmp_path):
        """CLI args should take priority over JSON config values."""
        config = {
            "learning_rate": 5e-5,
            "epochs": 200,
        }
        config_file = tmp_path / "config.json"
        config_file.write_text(json.dumps(config))

        # CLI provides --epochs 50 explicitly
        args = _parse_fixed("--epochs", "50")
        args.config = str(config_file)

        _populate_defaults_cache()
        _apply_config_file(args)

        # epochs was explicitly set to 50 via CLI — should NOT be overridden
        assert args.epochs == 50
        # learning_rate was not set via CLI — should come from JSON
        assert args.learning_rate == 5e-5

    def test_missing_config_file_raises(self):
        """Non-existent config file should raise FileNotFoundError."""
        args = argparse.Namespace(config="/nonexistent/config.json")
        with pytest.raises(FileNotFoundError):
            _apply_config_file(args)

    def test_no_config_is_noop(self):
        """When config is None, _apply_config_file should be a no-op."""
        args = argparse.Namespace(config=None, learning_rate=1e-4)
        _apply_config_file(args)
        assert args.learning_rate == 1e-4


# ---------------------------------------------------------------------------
# 4. --preprocess chain
# ---------------------------------------------------------------------------

class TestPreprocessChain:

    def test_preprocess_flag_exists(self):
        """--preprocess should be parseable."""
        args = _parse_fixed("--preprocess", "--audio-dir", "/tmp/a", "--tensor-output", "/tmp/t")
        assert args.preprocess is True

    def test_preprocess_only_flag_exists(self):
        """--preprocess-only should be parseable."""
        args = _parse_fixed("--preprocess-only", "--audio-dir", "/tmp/a", "--tensor-output", "/tmp/t")
        assert args.preprocess_only is True

    def test_preprocess_help_text_updated(self):
        """--preprocess help should mention continuing to training."""
        parser = build_root_parser()
        for action in parser._subparsers._actions:
            if hasattr(action, "_parser_class"):
                for name, sub in action.choices.items():
                    if name == "train":
                        for a in sub._actions:
                            if "--preprocess" in a.option_strings:
                                assert "then continue" in a.help
                                return
        pytest.fail("--preprocess not found")


# ---------------------------------------------------------------------------
# 5. --gui and --port flags
# ---------------------------------------------------------------------------

class TestGuiFlags:

    def test_gui_flag_on_root_parser(self):
        """--gui should be a root-level flag."""
        parser = build_root_parser()
        # Parse with no subcommand (--gui doesn't need one)
        args = parser.parse_args(["--gui"])
        assert args.gui is True
        assert args.subcommand is None

    def test_port_default(self):
        """--port should default to 8770."""
        parser = build_root_parser()
        args = parser.parse_args(["--gui"])
        assert args.port == 8770

    def test_port_custom(self):
        """--port should accept a custom value."""
        parser = build_root_parser()
        args = parser.parse_args(["--gui", "--port", "9000"])
        assert args.port == 9000

    def test_subcommand_not_required(self):
        """Subcommand should not be required (for --gui and wizard mode)."""
        parser = build_root_parser()
        # Should not raise SystemExit
        args = parser.parse_args(["--gui"])
        assert args.subcommand is None


# ---------------------------------------------------------------------------
# 6. JSON key mapping
# ---------------------------------------------------------------------------

class TestJsonKeyMapping:

    def test_hyphenated_keys_mapped(self, tmp_path):
        """JSON keys with hyphens should map to argparse underscored dests."""
        config = {
            "batch-size": 4,
            "gradient-accumulation": 8,
            "learning-rate": 2e-4,
        }
        config_file = tmp_path / "config.json"
        config_file.write_text(json.dumps(config))

        parser = build_root_parser()
        args = parser.parse_args(["train"])
        args.config = str(config_file)

        _populate_defaults_cache()
        _apply_config_file(args)

        assert args.batch_size == 4
        assert args.gradient_accumulation == 8
        assert args.learning_rate == 2e-4

    def test_unknown_keys_ignored(self, tmp_path):
        """Unknown JSON keys should be silently skipped."""
        config = {
            "some_unknown_key": "value",
            "epochs": 50,
        }
        config_file = tmp_path / "config.json"
        config_file.write_text(json.dumps(config))

        parser = build_root_parser()
        args = parser.parse_args(["train"])
        args.config = str(config_file)

        _populate_defaults_cache()
        _apply_config_file(args)

        assert args.epochs == 50
        assert not hasattr(args, "some_unknown_key")
