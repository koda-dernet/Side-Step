"""Tests for Fisher analysis helper functions."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import torch

from sidestep_engine.analysis.fisher.engine import (
    sample_focused_timestep,
)


class TestSampleFocusedTimestep(unittest.TestCase):
    """Test timestep sampling for Fisher analysis."""

    def test_texture_focus_range(self):
        """Texture focus should sample in [0.0, 0.4] range."""
        batch_size = 100
        device = torch.device("cpu")
        dtype = torch.float32

        timesteps = sample_focused_timestep(
            batch_size=batch_size,
            focus="texture",
            device=device,
            dtype=dtype,
        )

        self.assertEqual(timesteps.shape, (batch_size,))
        self.assertTrue((timesteps >= 0.0).all())
        self.assertTrue((timesteps <= 0.4).all())

    def test_structure_focus_range(self):
        """Structure focus should sample in [0.6, 1.0] range."""
        batch_size = 100
        device = torch.device("cpu")
        dtype = torch.float32

        timesteps = sample_focused_timestep(
            batch_size=batch_size,
            focus="structure",
            device=device,
            dtype=dtype,
        )

        self.assertEqual(timesteps.shape, (batch_size,))
        self.assertTrue((timesteps >= 0.6).all())
        self.assertTrue((timesteps <= 1.0).all())

    def test_balanced_focus_range(self):
        """Balanced focus should use training distribution."""
        batch_size = 100
        device = torch.device("cpu")
        dtype = torch.float32

        timesteps = sample_focused_timestep(
            batch_size=batch_size,
            focus="balanced",
            device=device,
            dtype=dtype,
            timestep_mu=-0.4,
            timestep_sigma=1.0,
            data_proportion=0.5,
        )

        self.assertEqual(timesteps.shape, (batch_size,))
        # Balanced should span wider range
        self.assertTrue((timesteps >= 0.0).all())
        self.assertTrue((timesteps <= 1.0).all())

    def test_custom_focus_range(self):
        """Custom focus range should be respected."""
        batch_size = 100
        device = torch.device("cpu")
        dtype = torch.float32

        timesteps = sample_focused_timestep(
            batch_size=batch_size,
            focus="0.2,0.7",
            device=device,
            dtype=dtype,
        )

        self.assertEqual(timesteps.shape, (batch_size,))
        self.assertTrue((timesteps >= 0.2).all())
        self.assertTrue((timesteps <= 0.7).all())

    def test_device_placement(self):
        """Timesteps should be on requested device."""
        batch_size = 10
        device = torch.device("cpu")
        dtype = torch.float32

        timesteps = sample_focused_timestep(
            batch_size=batch_size,
            focus="texture",
            device=device,
            dtype=dtype,
        )

        self.assertEqual(timesteps.device.type, "cpu")

    def test_dtype_correct(self):
        """Timesteps should have requested dtype."""
        batch_size = 10
        device = torch.device("cpu")
        dtype = torch.float16

        timesteps = sample_focused_timestep(
            batch_size=batch_size,
            focus="texture",
            device=device,
            dtype=dtype,
        )

        self.assertEqual(timesteps.dtype, torch.float16)


class TestFisherAnalysisHelpers(unittest.TestCase):
    """Test miscellaneous Fisher analysis helper functions."""

    def test_focus_to_range_texture(self):
        """_focus_to_range should convert texture to [0.0, 0.4]."""
        from sidestep_engine.analysis.fisher.analysis import _focus_to_range

        result = _focus_to_range("texture")
        self.assertEqual(result, [0.0, 0.4])

    def test_focus_to_range_structure(self):
        """_focus_to_range should convert structure to [0.6, 1.0]."""
        from sidestep_engine.analysis.fisher.analysis import _focus_to_range

        result = _focus_to_range("structure")
        self.assertEqual(result, [0.6, 1.0])

    def test_focus_to_range_balanced(self):
        """_focus_to_range should convert balanced to [0.0, 1.0]."""
        from sidestep_engine.analysis.fisher.analysis import _focus_to_range

        result = _focus_to_range("balanced")
        self.assertEqual(result, [0.0, 1.0])

    def test_focus_to_range_custom(self):
        """_focus_to_range should parse custom ranges."""
        from sidestep_engine.analysis.fisher.analysis import _focus_to_range

        result = _focus_to_range("0.3,0.8")
        self.assertEqual(result, [0.3, 0.8])

    def test_aggregate_runs_single_run(self):
        """_aggregate_runs should compute mean for single run."""
        from sidestep_engine.analysis.fisher.analysis import _aggregate_runs

        all_scores = [{"module1": 0.5, "module2": 0.8}]
        names = ["module1", "module2"]

        means, stds = _aggregate_runs(all_scores, names)

        self.assertEqual(means["module1"], 0.5)
        self.assertEqual(means["module2"], 0.8)
        self.assertEqual(stds["module1"], 0.0)  # Single run has zero std
        self.assertEqual(stds["module2"], 0.0)

    def test_aggregate_runs_multiple_runs(self):
        """_aggregate_runs should compute mean and std for multiple runs."""
        from sidestep_engine.analysis.fisher.analysis import _aggregate_runs

        all_scores = [
            {"module1": 0.4, "module2": 0.7},
            {"module1": 0.6, "module2": 0.9},
        ]
        names = ["module1", "module2"]

        means, stds = _aggregate_runs(all_scores, names)

        self.assertEqual(means["module1"], 0.5)  # (0.4 + 0.6) / 2
        self.assertEqual(means["module2"], 0.8)  # (0.7 + 0.9) / 2
        self.assertGreater(stds["module1"], 0.0)
        self.assertGreater(stds["module2"], 0.0)

    def test_aggregate_runs_missing_module(self):
        """_aggregate_runs should handle missing modules with zero."""
        from sidestep_engine.analysis.fisher.analysis import _aggregate_runs

        all_scores = [
            {"module1": 0.5},
            {"module2": 0.8},  # Different modules in different runs
        ]
        names = ["module1", "module2"]

        means, stds = _aggregate_runs(all_scores, names)

        # Should fill missing values with 0
        self.assertEqual(means["module1"], 0.25)  # (0.5 + 0.0) / 2
        self.assertEqual(means["module2"], 0.4)   # (0.0 + 0.8) / 2

    def test_build_run_subset_deterministic(self):
        """_build_run_subset should be deterministic with same seed."""
        from sidestep_engine.analysis.fisher.analysis import _build_run_subset

        subset1 = _build_run_subset(
            dataset_size=100, max_batches=10, run_idx=0, base_seed=1729
        )
        subset2 = _build_run_subset(
            dataset_size=100, max_batches=10, run_idx=0, base_seed=1729
        )

        self.assertEqual(subset1, subset2)

    def test_build_run_subset_different_per_run(self):
        """_build_run_subset should differ across run indices."""
        from sidestep_engine.analysis.fisher.analysis import _build_run_subset

        subset1 = _build_run_subset(
            dataset_size=100, max_batches=10, run_idx=0, base_seed=1729
        )
        subset2 = _build_run_subset(
            dataset_size=100, max_batches=10, run_idx=1, base_seed=1729
        )

        self.assertNotEqual(subset1, subset2)

    def test_build_run_subset_respects_max_batches(self):
        """_build_run_subset should limit to max_batches."""
        from sidestep_engine.analysis.fisher.analysis import _build_run_subset

        subset = _build_run_subset(
            dataset_size=100, max_batches=10, run_idx=0, base_seed=1729
        )

        self.assertEqual(len(subset), 10)

    def test_build_run_subset_handles_small_dataset(self):
        """_build_run_subset should handle dataset smaller than max_batches."""
        from sidestep_engine.analysis.fisher.analysis import _build_run_subset

        subset = _build_run_subset(
            dataset_size=5, max_batches=10, run_idx=0, base_seed=1729
        )

        self.assertEqual(len(subset), 5)

    def test_preflight_tensor_check_valid(self):
        """_preflight_tensor_check should pass for valid tensors."""
        from sidestep_engine.analysis.fisher.analysis import _preflight_tensor_check

        with tempfile.TemporaryDirectory() as tmpdir:
            pt_file = Path(tmpdir) / "test.pt"
            data = {
                "target_latents": torch.randn(100, 64),
                "attention_mask": torch.ones(100),
                "encoder_hidden_states": torch.randn(50, 768),
                "encoder_attention_mask": torch.ones(50),
            }
            torch.save(data, pt_file)

            error = _preflight_tensor_check([pt_file])
            self.assertIsNone(error)

    def test_preflight_tensor_check_missing_keys(self):
        """_preflight_tensor_check should detect missing keys."""
        from sidestep_engine.analysis.fisher.analysis import _preflight_tensor_check

        with tempfile.TemporaryDirectory() as tmpdir:
            pt_file = Path(tmpdir) / "test.pt"
            data = {
                "target_latents": torch.randn(100, 64),
                # Missing other required keys
            }
            torch.save(data, pt_file)

            error = _preflight_tensor_check([pt_file])
            self.assertIsNotNone(error)
            self.assertIn("missing required keys", error)

    def test_preflight_tensor_check_wrong_type(self):
        """_preflight_tensor_check should detect non-tensor values."""
        from sidestep_engine.analysis.fisher.analysis import _preflight_tensor_check

        with tempfile.TemporaryDirectory() as tmpdir:
            pt_file = Path(tmpdir) / "test.pt"
            data = {
                "target_latents": "not a tensor",
                "attention_mask": torch.ones(100),
                "encoder_hidden_states": torch.randn(50, 768),
                "encoder_attention_mask": torch.ones(50),
            }
            torch.save(data, pt_file)

            error = _preflight_tensor_check([pt_file])
            self.assertIsNotNone(error)

    def test_preflight_tensor_check_empty_list(self):
        """_preflight_tensor_check should handle empty file list."""
        from sidestep_engine.analysis.fisher.analysis import _preflight_tensor_check

        error = _preflight_tensor_check([])
        self.assertIsNotNone(error)
        self.assertIn("No .pt files", error)


if __name__ == "__main__":
    unittest.main()