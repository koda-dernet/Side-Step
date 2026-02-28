"""Tests for Fisher module discovery."""

from __future__ import annotations

import unittest

import torch.nn as nn

from sidestep_engine.analysis.fisher.modules import (
    find_all_targetable_modules,
    group_modules_for_chunking,
)
from sidestep_engine.analysis.fisher.analysis import _build_run_subset


def _build_mock_decoder(n_layers: int = 2):
    """Build a minimal mock model with self_attn, cross_attn, MLP layers."""

    class AttnBlock(nn.Module):
        def __init__(self):
            super().__init__()
            self.q_proj = nn.Linear(64, 64, bias=False)
            self.k_proj = nn.Linear(64, 64, bias=False)
            self.v_proj = nn.Linear(64, 64, bias=False)
            self.o_proj = nn.Linear(64, 64, bias=False)

    class MLPBlock(nn.Module):
        def __init__(self):
            super().__init__()
            self.gate_proj = nn.Linear(64, 128, bias=False)
            self.up_proj = nn.Linear(64, 128, bias=False)
            self.down_proj = nn.Linear(128, 64, bias=False)

    class Layer(nn.Module):
        def __init__(self):
            super().__init__()
            self.self_attn = AttnBlock()
            self.cross_attn = AttnBlock()
            self.mlp = MLPBlock()

    class Decoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.layers = nn.ModuleList([Layer() for _ in range(n_layers)])

    class Model(nn.Module):
        def __init__(self):
            super().__init__()
            self.decoder = Decoder()

    return Model()


class TestFindAllTargetableModules(unittest.TestCase):
    """Test module discovery in mock decoder."""

    def test_correct_count(self):
        """2-layer model: 4 self + 4 cross + 3 MLP = 11 per layer, 22 total."""
        model = _build_mock_decoder(n_layers=2)
        modules = find_all_targetable_modules(model)
        self.assertEqual(len(modules), 22)

    def test_names_prefixed_with_decoder(self):
        """All names should start with 'decoder.'."""
        model = _build_mock_decoder(n_layers=1)
        modules = find_all_targetable_modules(model)
        for name, _ in modules:
            self.assertTrue(name.startswith("decoder."), f"Bad prefix: {name}")

    def test_empty_model(self):
        """Model with no matching modules returns empty list."""
        model = nn.Sequential(nn.Linear(10, 10))
        modules = find_all_targetable_modules(model)
        self.assertEqual(len(modules), 0)

    def test_no_non_decoder_modules(self):
        """Modules outside decoder should not be included."""
        model = _build_mock_decoder(n_layers=1)
        model.encoder = nn.Linear(64, 64)  # type: ignore[attr-defined]
        modules = find_all_targetable_modules(model)
        for name, _ in modules:
            self.assertNotIn("encoder", name)


class TestGroupModulesForChunking(unittest.TestCase):
    """Test chunking into self_attn/cross_attn/mlp groups."""

    def test_three_groups(self):
        """Mock model should produce 3 chunk groups."""
        model = _build_mock_decoder(n_layers=2)
        modules = find_all_targetable_modules(model)
        groups = group_modules_for_chunking(modules)
        names = [g[0] for g in groups]
        self.assertIn("self_attn", names)
        self.assertIn("cross_attn", names)
        self.assertIn("mlp", names)

    def test_membership_correct(self):
        """Each module should be in exactly one group."""
        model = _build_mock_decoder(n_layers=2)
        modules = find_all_targetable_modules(model)
        groups = group_modules_for_chunking(modules)
        all_names = set()
        for _, member_names in groups:
            for n in member_names:
                self.assertNotIn(n, all_names, f"{n} in multiple groups")
                all_names.add(n)
        self.assertEqual(len(all_names), len(modules))


class TestFisherRunSubset(unittest.TestCase):
    """Test deterministic per-run subset selection."""

    def test_deterministic_for_same_run(self):
        a = _build_run_subset(dataset_size=20, max_batches=8, run_idx=1)
        b = _build_run_subset(dataset_size=20, max_batches=8, run_idx=1)
        self.assertEqual(a, b)

    def test_different_runs_produce_different_order(self):
        a = _build_run_subset(dataset_size=20, max_batches=8, run_idx=1)
        b = _build_run_subset(dataset_size=20, max_batches=8, run_idx=2)
        self.assertNotEqual(a, b)

    def test_subset_size_capped_by_dataset(self):
        a = _build_run_subset(dataset_size=5, max_batches=50, run_idx=0)
        self.assertEqual(len(a), 5)


if __name__ == "__main__":
    unittest.main()
