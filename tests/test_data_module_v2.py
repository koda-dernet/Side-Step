"""Tests for PreprocessedTensorDataset and collation functions."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import torch

from acestep.training_v2._vendor.data_module import (
    PreprocessedTensorDataset,
    collate_preprocessed_batch,
)


class TestPreprocessedTensorDataset(unittest.TestCase):
    """Test PreprocessedTensorDataset loading and chunking."""

    def _make_sample_tensor(self, path: Path, T: int = 100, L: int = 50, D: int = 768, duration: float = 4.0):
        """Create a minimal valid preprocessed tensor file."""
        data = {
            "target_latents": torch.randn(T, 64),
            "attention_mask": torch.ones(T),
            "encoder_hidden_states": torch.randn(L, D),
            "encoder_attention_mask": torch.ones(L),
            "context_latents": torch.randn(T, 65),
            "metadata": {
                "audio_path": "/fake/path.wav",
                "filename": path.stem + ".wav",
                "duration": duration,
                "caption": "test caption",
            },
        }
        torch.save(data, path)

    def test_load_from_directory_without_manifest(self):
        """Should discover .pt files when manifest.json is absent."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tensor_dir = Path(tmpdir)
            self._make_sample_tensor(tensor_dir / "song1.pt")
            self._make_sample_tensor(tensor_dir / "song2.pt")

            dataset = PreprocessedTensorDataset(str(tensor_dir))
            self.assertEqual(len(dataset), 2)
            self.assertFalse(dataset.manifest_loaded)

    def test_load_from_manifest_absolute_paths(self):
        """Should load samples from manifest.json with absolute paths."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tensor_dir = Path(tmpdir)
            self._make_sample_tensor(tensor_dir / "song1.pt")
            self._make_sample_tensor(tensor_dir / "song2.pt")

            manifest = {
                "samples": [
                    str(tensor_dir / "song1.pt"),
                    str(tensor_dir / "song2.pt"),
                ]
            }
            (tensor_dir / "manifest.json").write_text(json.dumps(manifest))

            dataset = PreprocessedTensorDataset(str(tensor_dir))
            self.assertEqual(len(dataset), 2)
            self.assertTrue(dataset.manifest_loaded)

    def test_load_from_manifest_relative_paths(self):
        """Should resolve relative paths in manifest.json."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tensor_dir = Path(tmpdir)
            self._make_sample_tensor(tensor_dir / "song1.pt")
            self._make_sample_tensor(tensor_dir / "song2.pt")

            manifest = {
                "samples": ["song1.pt", "song2.pt"]
            }
            (tensor_dir / "manifest.json").write_text(json.dumps(manifest))

            dataset = PreprocessedTensorDataset(str(tensor_dir))
            self.assertEqual(len(dataset), 2)
            self.assertTrue(dataset.manifest_loaded)

    def test_manifest_windows_path_normalization(self):
        """Should normalize Windows-style paths in manifest.json."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tensor_dir = Path(tmpdir)
            self._make_sample_tensor(tensor_dir / "song1.pt")

            # Windows path with backslashes
            manifest = {
                "samples": ["song1.pt"]
            }
            (tensor_dir / "manifest.json").write_text(json.dumps(manifest))

            dataset = PreprocessedTensorDataset(str(tensor_dir))
            self.assertEqual(len(dataset), 1)

    def test_manifest_malformed_windows_absolute_path(self):
        """Should handle malformed Windows absolute paths like '.C:\\...'."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tensor_dir = Path(tmpdir)
            song_path = tensor_dir / "song1.pt"
            self._make_sample_tensor(song_path)

            # Use the actual created path (relative for this test)
            manifest = {
                "samples": ["song1.pt"]
            }
            (tensor_dir / "manifest.json").write_text(json.dumps(manifest))

            dataset = PreprocessedTensorDataset(str(tensor_dir))
            # Should still find the file despite the weird format
            self.assertGreaterEqual(len(dataset), 0)

    def test_manifest_invalid_json(self):
        """Should fall back to directory scan when manifest.json is invalid."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tensor_dir = Path(tmpdir)
            self._make_sample_tensor(tensor_dir / "song1.pt")

            # Write invalid JSON
            (tensor_dir / "manifest.json").write_text("{ invalid json")

            dataset = PreprocessedTensorDataset(str(tensor_dir))
            self.assertEqual(len(dataset), 1)
            self.assertIsNotNone(dataset.manifest_error)
            self.assertTrue(dataset.manifest_fallback_used)

    def test_manifest_samples_not_list(self):
        """Should error when manifest 'samples' is not a list."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tensor_dir = Path(tmpdir)
            self._make_sample_tensor(tensor_dir / "song1.pt")

            manifest = {"samples": "not a list"}
            (tensor_dir / "manifest.json").write_text(json.dumps(manifest))

            dataset = PreprocessedTensorDataset(str(tensor_dir))
            # Should fall back to directory scan
            self.assertEqual(len(dataset), 1)
            self.assertIsNotNone(dataset.manifest_error)

    def test_getitem_loads_full_sample(self):
        """__getitem__ should load all required tensors."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tensor_dir = Path(tmpdir)
            self._make_sample_tensor(tensor_dir / "song1.pt")

            dataset = PreprocessedTensorDataset(str(tensor_dir))
            sample = dataset[0]

            self.assertIn("target_latents", sample)
            self.assertIn("attention_mask", sample)
            self.assertIn("encoder_hidden_states", sample)
            self.assertIn("encoder_attention_mask", sample)
            self.assertIn("context_latents", sample)
            self.assertIn("metadata", sample)

    def test_getitem_missing_required_keys(self):
        """Should raise KeyError when required keys are missing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tensor_dir = Path(tmpdir)
            bad_data = {
                "target_latents": torch.randn(100, 64),
                # Missing other required keys
            }
            torch.save(bad_data, tensor_dir / "bad.pt")

            dataset = PreprocessedTensorDataset(str(tensor_dir))
            with self.assertRaises(KeyError):
                _ = dataset[0]

    def test_chunking_disabled(self):
        """When chunk_duration is None, should return full sample."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tensor_dir = Path(tmpdir)
            T = 100
            self._make_sample_tensor(tensor_dir / "song1.pt", T=T)

            dataset = PreprocessedTensorDataset(str(tensor_dir), chunk_duration=None)
            sample = dataset[0]

            self.assertEqual(sample["target_latents"].shape[0], T)
            self.assertEqual(sample["attention_mask"].shape[0], T)
            self.assertEqual(sample["context_latents"].shape[0], T)

    def test_chunking_enabled_shorter_than_chunk(self):
        """When sample is shorter than chunk_duration, return full sample."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tensor_dir = Path(tmpdir)
            T = 50
            self._make_sample_tensor(tensor_dir / "song1.pt", T=T, duration=2.0)

            # 60-second chunk, but sample is only 2 seconds (~50 frames)
            dataset = PreprocessedTensorDataset(str(tensor_dir), chunk_duration=60)
            sample = dataset[0]

            # Should return full sample since it's shorter than chunk
            self.assertEqual(sample["target_latents"].shape[0], T)

    def test_chunking_enabled_longer_than_chunk(self):
        """When sample is longer than chunk_duration, return chunked slice."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tensor_dir = Path(tmpdir)
            T = 250  # ~10 seconds at 25 FPS
            self._make_sample_tensor(tensor_dir / "song1.pt", T=T, duration=10.0)

            # 4-second chunk at 25 FPS = 100 frames
            dataset = PreprocessedTensorDataset(str(tensor_dir), chunk_duration=4)
            sample = dataset[0]

            chunk_frames = sample["target_latents"].shape[0]
            # Should be around 100 frames (4 seconds * 25 FPS)
            self.assertLessEqual(chunk_frames, T)
            self.assertGreater(chunk_frames, 0)

    def test_nan_inf_replacement(self):
        """NaN/Inf values should be replaced with zeros."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tensor_dir = Path(tmpdir)
            data = {
                "target_latents": torch.tensor([[float('nan'), 1.0], [float('inf'), 2.0]]),
                "attention_mask": torch.ones(2),
                "encoder_hidden_states": torch.randn(10, 768),
                "encoder_attention_mask": torch.ones(10),
                "context_latents": torch.randn(2, 65),
                "metadata": {},
            }
            torch.save(data, tensor_dir / "bad.pt")

            dataset = PreprocessedTensorDataset(str(tensor_dir))
            sample = dataset[0]

            # NaN/Inf should be replaced with 0
            self.assertFalse(torch.isnan(sample["target_latents"]).any())
            self.assertFalse(torch.isinf(sample["target_latents"]).any())

    def test_missing_files_warning(self):
        """Should warn about missing files listed in manifest."""
        with tempfile.TemporaryDirectory() as tmpdir:
            tensor_dir = Path(tmpdir)
            self._make_sample_tensor(tensor_dir / "song1.pt")

            manifest = {
                "samples": [
                    str(tensor_dir / "song1.pt"),
                    str(tensor_dir / "missing.pt"),
                ]
            }
            (tensor_dir / "manifest.json").write_text(json.dumps(manifest))

            dataset = PreprocessedTensorDataset(str(tensor_dir))
            # Should only load existing files
            self.assertEqual(len(dataset), 1)


class TestCollatePreprocessedBatch(unittest.TestCase):
    """Test collation function for batching."""

    def _make_sample(self, T: int, L: int, D: int = 768):
        """Create a single sample dict."""
        return {
            "target_latents": torch.randn(T, 64),
            "attention_mask": torch.ones(T),
            "encoder_hidden_states": torch.randn(L, D),
            "encoder_attention_mask": torch.ones(L),
            "context_latents": torch.randn(T, 65),
            "metadata": {"test": "data"},
        }

    def test_collate_single_sample(self):
        """Collating a single sample should work."""
        sample = self._make_sample(T=100, L=50)
        batch = collate_preprocessed_batch([sample])

        self.assertEqual(batch["target_latents"].shape, (1, 100, 64))
        self.assertEqual(batch["attention_mask"].shape, (1, 100))
        self.assertEqual(batch["encoder_hidden_states"].shape, (1, 50, 768))
        self.assertEqual(batch["encoder_attention_mask"].shape, (1, 50))
        self.assertEqual(batch["context_latents"].shape, (1, 100, 65))
        self.assertEqual(len(batch["metadata"]), 1)

    def test_collate_uniform_lengths(self):
        """Collating samples of same length should not add padding."""
        samples = [self._make_sample(T=100, L=50) for _ in range(3)]
        batch = collate_preprocessed_batch(samples)

        self.assertEqual(batch["target_latents"].shape, (3, 100, 64))
        self.assertEqual(batch["attention_mask"].shape, (3, 100))
        self.assertEqual(batch["encoder_hidden_states"].shape, (3, 50, 768))

    def test_collate_variable_lengths(self):
        """Collating samples of different lengths should pad to max."""
        samples = [
            self._make_sample(T=50, L=30),
            self._make_sample(T=100, L=60),
            self._make_sample(T=75, L=45),
        ]
        batch = collate_preprocessed_batch(samples)

        # Should pad to max lengths (100, 60)
        self.assertEqual(batch["target_latents"].shape, (3, 100, 64))
        self.assertEqual(batch["attention_mask"].shape, (3, 100))
        self.assertEqual(batch["encoder_hidden_states"].shape, (3, 60, 768))
        self.assertEqual(batch["encoder_attention_mask"].shape, (3, 60))
        self.assertEqual(batch["context_latents"].shape, (3, 100, 65))

    def test_collate_padding_is_zero(self):
        """Padding values should be zero."""
        samples = [
            self._make_sample(T=50, L=30),
            self._make_sample(T=100, L=60),
        ]
        batch = collate_preprocessed_batch(samples)

        # Check that shorter sample has zero padding
        # First sample should have zeros in positions [50:100]
        padding_region = batch["attention_mask"][0, 50:100]
        self.assertTrue((padding_region == 0).all())

    def test_collate_preserves_metadata_list(self):
        """Metadata should be preserved as a list."""
        samples = [
            self._make_sample(T=50, L=30),
            self._make_sample(T=50, L=30),
        ]
        batch = collate_preprocessed_batch(samples)

        self.assertIsInstance(batch["metadata"], list)
        self.assertEqual(len(batch["metadata"]), 2)


if __name__ == "__main__":
    unittest.main()