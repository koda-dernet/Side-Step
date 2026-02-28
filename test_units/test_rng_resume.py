"""Tests for RNG state capture/restore used when resuming training from checkpoint."""

from __future__ import annotations

import unittest

import torch

from sidestep_engine.core.trainer_helpers import capture_rng_state, restore_rng_state


class TestRestoreRngState(unittest.TestCase):
    """Ensure RNG state restores correctly, including when loaded from CUDA (map_location)."""

    def test_restore_rng_state_cpu_tensor_succeeds(self) -> None:
        """Baseline: valid RNG state from capture_rng_state restores without error."""
        rng = capture_rng_state(device=None)
        self.assertIn("torch_cpu", rng)

        restored = restore_rng_state(rng, current_device=None)
        self.assertIn("torch_cpu_rng", restored)

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA required")
    def test_restore_rng_state_torch_cpu_on_cuda_succeeds(self) -> None:
        """Regression: torch_cpu on CUDA (simulating map_location load) restores without TypeError."""
        rng = capture_rng_state(device=torch.device("cuda", 0))
        # Simulate what happens when training_state.pt is loaded with map_location=cuda
        rng["torch_cpu"] = rng["torch_cpu"].cuda()

        restored = restore_rng_state(rng, current_device=torch.device("cuda", 0))
        self.assertIn("torch_cpu_rng", restored)

    def test_capture_restore_roundtrip(self) -> None:
        """Capture state, restore it, generate randoms; verify RNG sequence continues identically."""
        torch.manual_seed(42)
        _ = torch.randn(10)  # advance RNG

        captured = capture_rng_state(device=None)
        # Generate sequence A from restored state
        restore_rng_state(captured)
        seq_a = [torch.randn(1).item() for _ in range(5)]

        # Restore same state, generate sequence B
        restore_rng_state(captured)
        seq_b = [torch.randn(1).item() for _ in range(5)]

        self.assertEqual(seq_a, seq_b)


if __name__ == "__main__":
    unittest.main()
