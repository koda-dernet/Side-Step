"""Tests for timestep focus sampling during Fisher analysis."""

from __future__ import annotations

import unittest
from unittest.mock import patch

import torch

from sidestep_engine.analysis.fisher.engine import sample_focused_timestep


class TestSampleFocusedTimestep(unittest.TestCase):
    """Verify timestep samples land in the correct ranges."""

    def _sample(self, focus: str, n: int = 1000):
        return sample_focused_timestep(n, focus, torch.device("cpu"), torch.float32)

    def test_texture_range(self):
        """'texture' should produce t in [0, 0.4]."""
        t = self._sample("texture")
        self.assertTrue((t >= 0.0).all())
        self.assertTrue((t <= 0.4).all())

    def test_structure_range(self):
        """'structure' should produce t in [0.6, 1.0]."""
        t = self._sample("structure")
        self.assertTrue((t >= 0.6).all())
        self.assertTrue((t <= 1.0).all())

    def test_balanced_range(self):
        """'balanced' should produce t in [0, 1]."""
        t = self._sample("balanced")
        self.assertTrue((t >= 0.0).all())
        self.assertTrue((t <= 1.0).all())

    @patch("sidestep_engine.analysis.fisher.engine.sample_timesteps")
    def test_balanced_passes_model_timestep_params(self, mock_sample_timesteps):
        """Balanced mode should forward mu/sigma/data_proportion to sampler."""
        mock_sample_timesteps.return_value = (
            torch.full((8,), 0.5, dtype=torch.float32),
            torch.full((8,), 0.5, dtype=torch.float32),
        )
        out = sample_focused_timestep(
            8,
            "balanced",
            torch.device("cpu"),
            torch.float32,
            timestep_mu=-0.25,
            timestep_sigma=0.75,
            data_proportion=0.35,
        )
        self.assertEqual(out.shape, (8,))
        mock_sample_timesteps.assert_called_once()
        _args, kwargs = mock_sample_timesteps.call_args
        self.assertEqual(kwargs["timestep_mu"], -0.25)
        self.assertEqual(kwargs["timestep_sigma"], 0.75)
        self.assertEqual(kwargs["data_proportion"], 0.35)

    def test_custom_range(self):
        """Custom 'low,high' string should bound samples correctly."""
        t = self._sample("0.1,0.5")
        self.assertTrue((t >= 0.1).all())
        self.assertTrue((t <= 0.5).all())

    def test_correct_shape(self):
        """Output shape should match requested batch size."""
        t = self._sample("texture", n=42)
        self.assertEqual(t.shape, (42,))

    def test_texture_not_degenerate(self):
        """Should produce varied samples, not a constant."""
        t = self._sample("texture", n=100)
        self.assertGreater(t.std().item(), 0.01)


if __name__ == "__main__":
    unittest.main()
