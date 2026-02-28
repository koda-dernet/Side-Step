"""Tests for random latent chunking in PreprocessedTensorDataset."""

from __future__ import annotations

import os
import tempfile
import unittest

import torch


def _make_sample_pt(path: str, T: int = 3000, L: int = 256, D: int = 768, duration: int = 120):
    """Create a minimal .pt file mimicking the preprocessing output."""
    torch.save({
        "target_latents": torch.randn(T, 64),
        "attention_mask": torch.ones(T),
        "encoder_hidden_states": torch.randn(L, D),
        "encoder_attention_mask": torch.ones(L),
        "context_latents": torch.randn(T, 65),
        "metadata": {"duration": duration, "filename": os.path.basename(path)},
    }, path)


class TestChunkingDisabled(unittest.TestCase):
    """When chunk_duration is None, samples pass through unchanged."""

    def test_full_sample_returned(self):
        """Without chunking, target_latents has full T dimension."""
        from sidestep_engine.vendor.data_module import PreprocessedTensorDataset

        with tempfile.TemporaryDirectory() as td:
            _make_sample_pt(os.path.join(td, "song.pt"), T=3000, duration=120)

            ds = PreprocessedTensorDataset(td, chunk_duration=None)
            sample = ds[0]

            self.assertEqual(sample["target_latents"].shape[0], 3000)
            self.assertEqual(sample["attention_mask"].shape[0], 3000)
            self.assertEqual(sample["context_latents"].shape[0], 3000)


class TestChunkingEnabled(unittest.TestCase):
    """When chunk_duration is set, samples are randomly windowed."""

    def test_chunk_slices_t_dimension(self):
        """Chunked sample should have shorter T than the full sample."""
        from sidestep_engine.vendor.data_module import PreprocessedTensorDataset

        with tempfile.TemporaryDirectory() as td:
            # 3000 frames, 120s => 25 fps; chunk_duration=60 => 1500 frames
            _make_sample_pt(os.path.join(td, "song.pt"), T=3000, duration=120)

            ds = PreprocessedTensorDataset(td, chunk_duration=60)
            sample = ds[0]

            expected_frames = 60 * 25  # 1500
            self.assertEqual(sample["target_latents"].shape[0], expected_frames)
            self.assertEqual(sample["attention_mask"].shape[0], expected_frames)
            self.assertEqual(sample["context_latents"].shape[0], expected_frames)

    def test_non_t_tensors_unchanged(self):
        """encoder_hidden_states and encoder_attention_mask must not be sliced."""
        from sidestep_engine.vendor.data_module import PreprocessedTensorDataset

        with tempfile.TemporaryDirectory() as td:
            _make_sample_pt(os.path.join(td, "song.pt"), T=3000, L=256, D=768, duration=120)

            ds = PreprocessedTensorDataset(td, chunk_duration=60)
            sample = ds[0]

            self.assertEqual(sample["encoder_hidden_states"].shape, torch.Size([256, 768]))
            self.assertEqual(sample["encoder_attention_mask"].shape, torch.Size([256]))

    def test_short_sample_passthrough(self):
        """Sample shorter than chunk_duration should pass through unchanged."""
        from sidestep_engine.vendor.data_module import PreprocessedTensorDataset

        with tempfile.TemporaryDirectory() as td:
            # 500 frames, 20s => 25 fps; chunk_duration=60 => 1500 frames (> 500)
            _make_sample_pt(os.path.join(td, "short.pt"), T=500, duration=20)

            ds = PreprocessedTensorDataset(td, chunk_duration=60)
            sample = ds[0]

            self.assertEqual(sample["target_latents"].shape[0], 500)

    def test_random_offset_varies(self):
        """Two calls to __getitem__ should (usually) produce different slices."""
        from sidestep_engine.vendor.data_module import PreprocessedTensorDataset

        with tempfile.TemporaryDirectory() as td:
            _make_sample_pt(os.path.join(td, "song.pt"), T=5000, duration=200)

            ds = PreprocessedTensorDataset(td, chunk_duration=60)

            # Get two samples -- they are random, so occasionally they match.
            # Run several times to be confident they differ at least once.
            differs = False
            for _ in range(10):
                s1 = ds[0]["target_latents"]
                s2 = ds[0]["target_latents"]
                if not torch.equal(s1, s2):
                    differs = True
                    break

            self.assertTrue(differs, "Random chunking should produce different slices")

    def test_chunk_bounds_valid(self):
        """Chunked tensor should never exceed original T."""
        from sidestep_engine.vendor.data_module import PreprocessedTensorDataset

        with tempfile.TemporaryDirectory() as td:
            _make_sample_pt(os.path.join(td, "song.pt"), T=3000, duration=120)

            ds = PreprocessedTensorDataset(td, chunk_duration=60)

            for _ in range(50):
                sample = ds[0]
                T_out = sample["target_latents"].shape[0]
                self.assertLessEqual(T_out, 3000)
                self.assertGreater(T_out, 0)

    def test_latent_fps_auto_detected(self):
        """Latent FPS should be computed from first sample metadata."""
        from sidestep_engine.vendor.data_module import PreprocessedTensorDataset

        with tempfile.TemporaryDirectory() as td:
            # 2500 frames / 100s = 25 fps
            _make_sample_pt(os.path.join(td, "song.pt"), T=2500, duration=100)

            ds = PreprocessedTensorDataset(td, chunk_duration=60)
            self.assertAlmostEqual(ds._latent_fps, 25.0, places=1)

    def test_fallback_fps_when_no_duration(self):
        """When duration=0, fallback FPS (25) should be used."""
        from sidestep_engine.vendor.data_module import PreprocessedTensorDataset

        with tempfile.TemporaryDirectory() as td:
            _make_sample_pt(os.path.join(td, "song.pt"), T=3000, duration=0)

            ds = PreprocessedTensorDataset(td, chunk_duration=60)
            self.assertEqual(ds._latent_fps, 25.0)


if __name__ == "__main__":
    unittest.main()
