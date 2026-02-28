"""Tests for spectral analysis (SVD effective rank)."""

from __future__ import annotations

import unittest

import torch
import torch.nn as nn

from sidestep_engine.analysis.fisher.spectral import (
    _effective_rank_of,
    compute_spectral_complexity,
)


class TestEffectiveRank(unittest.TestCase):
    """Unit tests for the SVD effective rank calculation."""

    def test_identity_matrix(self):
        """Identity matrix should have effective rank 1 at 0.95 threshold.

        All singular values are 1.0 so energy is uniformly distributed.
        For an NxN identity, cumsum reaches 0.95 * N at position ceil(0.95*N).
        """
        W = torch.eye(64)
        eff = _effective_rank_of(W, torch.device("cpu"), 0.95)
        # 95% of 64 singular values = 61
        self.assertGreater(eff, 50)
        self.assertLessEqual(eff, 64)

    def test_rank_one_matrix(self):
        """A rank-1 matrix should have effective rank 1."""
        a = torch.randn(64, 1)
        W = a @ a.T
        eff = _effective_rank_of(W, torch.device("cpu"), 0.95)
        self.assertEqual(eff, 1)

    def test_full_rank_random(self):
        """Full-rank random matrix should have high effective rank."""
        torch.manual_seed(42)
        W = torch.randn(64, 64)
        eff = _effective_rank_of(W, torch.device("cpu"), 0.95)
        self.assertGreater(eff, 30)

    def test_rank_deficient(self):
        """Matrix with known rank should have correct effective rank."""
        torch.manual_seed(42)
        k = 10
        A = torch.randn(64, k)
        B = torch.randn(k, 64)
        W = A @ B
        eff = _effective_rank_of(W, torch.device("cpu"), 0.95)
        self.assertLessEqual(eff, k + 2)
        self.assertGreater(eff, 0)

    def test_small_matrix(self):
        """Very small matrix (8x8) should not error."""
        W = torch.randn(8, 8)
        eff = _effective_rank_of(W, torch.device("cpu"), 0.95)
        self.assertGreater(eff, 0)
        self.assertLessEqual(eff, 8)

    def test_zero_matrix(self):
        """All-zero matrix should have effective rank 1 (degenerate)."""
        W = torch.zeros(32, 32)
        eff = _effective_rank_of(W, torch.device("cpu"), 0.95)
        self.assertEqual(eff, 1)


class TestComputeSpectralComplexity(unittest.TestCase):
    """Integration test for compute_spectral_complexity."""

    def test_processes_module_list(self):
        """Should return a dict with one entry per module."""
        modules = []
        for i in range(3):
            m = nn.Linear(32, 32, bias=False)
            modules.append((f"decoder.layers.{i}.self_attn.q_proj", m))

        result = compute_spectral_complexity(modules, torch.device("cpu"))
        self.assertEqual(len(result), 3)
        for name, eff in result.items():
            self.assertGreater(eff, 0)


if __name__ == "__main__":
    unittest.main()
