"""Tests for Fisher map JSON persistence."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from sidestep_engine.analysis.fisher.io import (
    compute_dataset_hash,
    load_fisher_map,
    save_fisher_map,
)


class TestSaveLoadRoundTrip(unittest.TestCase):
    """Verify save_fisher_map -> load_fisher_map round-trips correctly."""

    def setUp(self):
        self.tmpdir = tempfile.mkdtemp()
        self.path = Path(self.tmpdir) / "fisher_map.json"
        self.sample_data = {
            "version": 2,
            "model_variant": "turbo",
            "target_modules": ["self_attn.q_proj"],
            "rank_pattern": {"layers.0.self_attn.q_proj": 64},
            "alpha_pattern": {"layers.0.self_attn.q_proj": 128},
            "rank_budget": {"min": 16, "max": 128},
            "timestep_params": {"mu": -0.4, "sigma": 1.0, "data_proportion": 0.5},
            "sample_coverage": [
                {
                    "run": 1,
                    "selected_count": 8,
                    "total_count": 12,
                    "coverage_ratio": 0.666667,
                    "selected_files": ["a.pt", "b.pt"],
                }
            ],
        }

    def test_round_trip(self):
        """Saved data loads back identically."""
        save_fisher_map(self.sample_data, self.path)
        loaded = load_fisher_map(self.path)
        self.assertIsNotNone(loaded)
        self.assertEqual(loaded["target_modules"], self.sample_data["target_modules"])
        self.assertEqual(loaded["rank_pattern"], self.sample_data["rank_pattern"])
        self.assertIn("timestep_params", loaded)
        self.assertIn("sample_coverage", loaded)

    def test_missing_file_returns_none(self):
        """Non-existent path should return None."""
        result = load_fisher_map(Path(self.tmpdir) / "nope.json")
        self.assertIsNone(result)

    def test_corrupted_json_returns_none(self):
        """Corrupted JSON should return None gracefully."""
        self.path.write_text("{invalid json!!!}")
        result = load_fisher_map(self.path)
        self.assertIsNone(result)

    def test_missing_required_key_returns_none(self):
        """Missing required fields should return None."""
        bad = {"version": 2, "target_modules": ["q_proj"]}
        self.path.write_text(json.dumps(bad))
        result = load_fisher_map(self.path)
        self.assertIsNone(result)

    def test_variant_mismatch_still_loads(self):
        """Variant mismatch warns but still loads the data."""
        save_fisher_map(self.sample_data, self.path)
        loaded = load_fisher_map(self.path, expected_variant="base")
        self.assertIsNotNone(loaded)

    def test_version_defaults(self):
        """Missing version should still load (warns)."""
        data = dict(self.sample_data)
        del data["version"]
        self.path.write_text(json.dumps(data))
        loaded = load_fisher_map(self.path)
        self.assertIsNotNone(loaded)


class TestDatasetHash(unittest.TestCase):
    """Tests for dataset hash computation."""

    def test_consistent_hash(self):
        """Same file set should produce the same hash."""
        tmpdir = tempfile.mkdtemp()
        for i in range(3):
            (Path(tmpdir) / f"sample_{i}.pt").touch()
        h1 = compute_dataset_hash(tmpdir)
        h2 = compute_dataset_hash(tmpdir)
        self.assertEqual(h1, h2)

    def test_different_files_different_hash(self):
        """Adding a file should change the hash."""
        tmpdir = tempfile.mkdtemp()
        (Path(tmpdir) / "a.pt").touch()
        h1 = compute_dataset_hash(tmpdir)
        (Path(tmpdir) / "b.pt").touch()
        h2 = compute_dataset_hash(tmpdir)
        self.assertNotEqual(h1, h2)


if __name__ == "__main__":
    unittest.main()
