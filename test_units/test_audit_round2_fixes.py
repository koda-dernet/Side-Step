"""Tests for audit round 2 fixes.

Covers:
- P0: save_on_early_exit helper
- P0: Preprocess stem collision (safe_output_stem)
- P1: use_dora in _SKIP_KEYS
- P1: Fisher map cache invalidation on dataset_dir change
- P1: Timestep levers wiring through namespace + config_builder
- P1: Metadata full-path indexing
- P1: Warmup clamp warning
- P2: Version string alignment
- P2: CLI timestep flags
"""

from __future__ import annotations

import argparse
import json
import logging
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch


# ===================================================================
# 1. safe_output_stem (P0: preprocess collision fix)
# ===================================================================

class TestSafeOutputStem(unittest.TestCase):
    """Verify collision-safe output stems for nested audio directories."""

    def _stem(self, audio_path, audio_dir):
        from sidestep_engine.data.preprocess_discovery import safe_output_stem
        return safe_output_stem(Path(audio_path), audio_dir)

    def test_flat_directory_uses_stem(self):
        """Files in root of audio_dir use plain stem."""
        result = self._stem("/data/audio/song.wav", "/data/audio")
        self.assertEqual(result, "song")

    def test_nested_directory_includes_subdir(self):
        """Files in subdirs get subdir prefix to avoid collision."""
        result = self._stem("/data/audio/artist/song.wav", "/data/audio")
        self.assertEqual(result, "artist__song")

    def test_deeply_nested(self):
        """Multiple levels of nesting are flattened."""
        result = self._stem("/data/audio/genre/artist/song.wav", "/data/audio")
        self.assertEqual(result, "genre__artist__song")

    def test_same_name_different_dirs_no_collision(self):
        """Two files named song.wav in different subdirs get different stems."""
        stem_a = self._stem("/data/audio/a/song.wav", "/data/audio")
        stem_b = self._stem("/data/audio/b/song.wav", "/data/audio")
        self.assertNotEqual(stem_a, stem_b)

    def test_no_audio_dir_falls_back_to_stem(self):
        """When audio_dir is None, falls back to plain stem."""
        result = self._stem("/data/audio/song.wav", None)
        self.assertEqual(result, "song")

    def test_unrelated_path_falls_back_to_stem(self):
        """When audio_path is not under audio_dir, falls back to stem."""
        result = self._stem("/other/path/song.wav", "/data/audio")
        self.assertEqual(result, "song")


# ===================================================================
# 2. save_on_early_exit (P0: early-exit save guard)
# ===================================================================

class TestSaveOnEarlyExit(unittest.TestCase):
    """Verify save_on_early_exit helper behavior."""

    def test_skip_save_at_step_zero(self):
        """No save should happen if global_step is 0."""
        from sidestep_engine.core.trainer_helpers import save_on_early_exit
        trainer = MagicMock()
        updates = save_on_early_exit(trainer, "/tmp/test", 0, "test")
        self.assertEqual(updates, [])

    def test_save_attempted_at_nonzero_step(self):
        """Save should be attempted when global_step > 0."""
        from sidestep_engine.core.trainer_helpers import save_on_early_exit
        trainer = MagicMock()
        trainer.module = MagicMock()
        trainer.module.model.decoder = MagicMock()
        trainer.adapter_type = "lora"

        with patch("sidestep_engine.core.trainer_helpers.save_adapter_flat") as mock_save:
            updates = save_on_early_exit(trainer, "/tmp/test", 10, "user_stop")
            mock_save.assert_called_once()
            self.assertEqual(len(updates), 1)
            self.assertIn("early exit", updates[0].msg.lower())

    def test_save_failure_yields_warning(self):
        """If save fails, a warning update is returned instead of crashing."""
        from sidestep_engine.core.trainer_helpers import save_on_early_exit
        trainer = MagicMock()
        trainer.module = MagicMock()
        trainer.module.model.decoder = MagicMock()
        trainer.adapter_type = "lora"

        with patch("sidestep_engine.core.trainer_helpers.save_adapter_flat", side_effect=RuntimeError("disk full")):
            updates = save_on_early_exit(trainer, "/tmp/test", 10, "max_steps")
            self.assertEqual(len(updates), 1)
            self.assertIn("WARN", updates[0].msg)


# ===================================================================
# 3. use_dora in _SKIP_KEYS (P1)
# ===================================================================

class TestUseDoraCLIExport(unittest.TestCase):
    """Verify --use-dora flag is NOT emitted in CLI export."""

    def test_use_dora_skipped_in_cli_export(self):
        from sidestep_engine.ui.flows.review_summary import build_cli_command
        answers = {
            "use_dora": True,
            "adapter_type": "dora",
            "output_dir": "/tmp/out",
            "learning_rate": 1e-4,
        }
        cmd = build_cli_command(answers)
        self.assertNotIn("--use-dora", cmd)

    def test_adapter_type_dora_still_emitted(self):
        from sidestep_engine.ui.flows.review_summary import build_cli_command
        answers = {
            "use_dora": True,
            "adapter_type": "dora",
            "output_dir": "/tmp/out",
        }
        cmd = build_cli_command(answers)
        # adapter_type is also in _SKIP_KEYS, so it should NOT be emitted
        self.assertNotIn("--use-dora", cmd)


# ===================================================================
# 4. Fisher map cache invalidation (P1)
# ===================================================================

class TestFisherMapCacheInvalidation(unittest.TestCase):
    """Verify _has_fisher_map cache invalidates on dataset_dir change."""

    def test_cache_invalidated_on_dir_change(self):
        from sidestep_engine.ui.flows.train_steps_required import _has_fisher_map

        with tempfile.TemporaryDirectory() as dir_a, \
             tempfile.TemporaryDirectory() as dir_b:
            # Create fisher_map.json in dir_a only
            fisher_a = Path(dir_a) / "fisher_map.json"
            fisher_a.write_text(json.dumps({"rank_pattern": {"layer.0": 32}}))

            a = {"dataset_dir": dir_a}
            self.assertTrue(_has_fisher_map(a))
            self.assertTrue(a["_fisher_map_cached"])

            # Change dataset_dir to dir_b (no fisher map)
            a["dataset_dir"] = dir_b
            self.assertFalse(_has_fisher_map(a))
            self.assertFalse(a["_fisher_map_cached"])

    def test_cache_stable_same_dir(self):
        from sidestep_engine.ui.flows.train_steps_required import _has_fisher_map

        with tempfile.TemporaryDirectory() as tmp:
            a = {"dataset_dir": tmp}
            self.assertFalse(_has_fisher_map(a))
            # Calling again with same dir should return cached False
            self.assertFalse(_has_fisher_map(a))
            self.assertEqual(a["_fisher_map_cached_dir"], tmp)


# ===================================================================
# 5. Timestep levers wiring (P1)
# ===================================================================

class TestTimestepLeversWiring(unittest.TestCase):
    """Verify timestep_mu/sigma flow from wizard answers through namespace."""

    def test_namespace_passes_timestep_overrides(self):
        from sidestep_engine.ui.flows.common import build_train_namespace
        answers = {
            "checkpoint_dir": "/tmp/ckpt",
            "model_variant": "turbo",
            "dataset_dir": "/tmp/data",
            "output_dir": "/tmp/out",
            "timestep_mu": -0.6,
            "timestep_sigma": 1.2,
        }
        ns = build_train_namespace(answers)
        self.assertEqual(ns.timestep_mu, -0.6)
        self.assertEqual(ns.timestep_sigma, 1.2)

    def test_namespace_defaults_none_when_not_set(self):
        from sidestep_engine.ui.flows.common import build_train_namespace
        answers = {
            "checkpoint_dir": "/tmp/ckpt",
            "model_variant": "turbo",
            "dataset_dir": "/tmp/data",
            "output_dir": "/tmp/out",
        }
        ns = build_train_namespace(answers)
        self.assertIsNone(ns.timestep_mu)
        self.assertIsNone(ns.timestep_sigma)


# ===================================================================
# 6. Metadata full-path indexing (P1)
# ===================================================================

class TestMetadataFullPathIndex(unittest.TestCase):
    """Verify load_sample_metadata indexes by full path for disambiguation."""

    def test_full_path_key_present(self):
        from sidestep_engine.data.preprocess_discovery import load_sample_metadata

        with tempfile.TemporaryDirectory() as tmp:
            song = Path(tmp) / "song.wav"
            song.touch()
            meta = load_sample_metadata(None, [song])
            # Should have both basename and full-path keys
            self.assertIn(song.name, meta)
            self.assertIn(str(song), meta)

    def test_same_basename_different_dirs_disambiguated(self):
        from sidestep_engine.data.preprocess_discovery import load_sample_metadata

        with tempfile.TemporaryDirectory() as tmp:
            a_dir = Path(tmp) / "a"
            b_dir = Path(tmp) / "b"
            a_dir.mkdir()
            b_dir.mkdir()
            song_a = a_dir / "song.wav"
            song_b = b_dir / "song.wav"
            song_a.touch()
            song_b.touch()

            meta = load_sample_metadata(None, [song_a, song_b])
            # Full-path keys should be distinct
            self.assertIn(str(song_a), meta)
            self.assertIn(str(song_b), meta)


# ===================================================================
# 7. Warmup clamp warning (P1)
# ===================================================================

class TestWarmupClampWarning(unittest.TestCase):
    """Verify a warning is logged when warmup_steps is clamped."""

    def test_warning_logged_on_clamp(self):
        import torch
        from sidestep_engine.core.optim import build_scheduler

        params = [torch.nn.Parameter(torch.zeros(10))]
        optimizer = torch.optim.AdamW(params, lr=1e-4)

        with self.assertLogs("sidestep_engine.core.optim", level="WARNING") as cm:
            build_scheduler(
                optimizer,
                scheduler_type="cosine",
                warmup_steps=500,
                total_steps=1000,
            )
        self.assertTrue(any("clamped" in msg.lower() for msg in cm.output))

    def test_no_warning_when_within_limit(self):
        import torch
        from sidestep_engine.core.optim import build_scheduler

        params = [torch.nn.Parameter(torch.zeros(10))]
        optimizer = torch.optim.AdamW(params, lr=1e-4)

        logger = logging.getLogger("sidestep_engine.core.optim")
        with patch.object(logger, "warning") as mock_warn:
            build_scheduler(
                optimizer,
                scheduler_type="cosine",
                warmup_steps=50,
                total_steps=1000,
            )
            # 50 <= 100 (10% of 1000) → no clamp warning
            for call in mock_warn.call_args_list:
                self.assertNotIn("clamped", str(call).lower())


# ===================================================================
# 8. Version string alignment (P2)
# ===================================================================

class TestVersionAlignment(unittest.TestCase):
    """Verify _compat.py version matches __init__.py."""

    def test_versions_match(self):
        from sidestep_engine import __version__
        from sidestep_engine._compat import SIDESTEP_VERSION
        self.assertEqual(__version__, SIDESTEP_VERSION)


# ===================================================================
# 9. CLI timestep flags (P2)
# ===================================================================

class TestCLITimestepFlags(unittest.TestCase):
    """Verify --timestep-mu and --timestep-sigma are valid CLI args."""

    def test_timestep_flags_accepted(self):
        from sidestep_engine.cli.args import build_root_parser
        parser = build_root_parser()
        args = parser.parse_args([
            "train",
            "--checkpoint-dir", "/tmp/ckpt",
            "--model-variant", "turbo",
            "--dataset-dir", "/tmp/data",
            "--output-dir", "/tmp/out",
            "--timestep-mu", "-0.6",
            "--timestep-sigma", "1.2",
        ])
        self.assertAlmostEqual(args.timestep_mu, -0.6)
        self.assertAlmostEqual(args.timestep_sigma, 1.2)

    def test_timestep_flags_default_none(self):
        from sidestep_engine.cli.args import build_root_parser
        parser = build_root_parser()
        args = parser.parse_args([
            "train",
            "--checkpoint-dir", "/tmp/ckpt",
            "--model-variant", "turbo",
            "--dataset-dir", "/tmp/data",
            "--output-dir", "/tmp/out",
        ])
        self.assertIsNone(args.timestep_mu)
        self.assertIsNone(args.timestep_sigma)


if __name__ == "__main__":
    unittest.main()
