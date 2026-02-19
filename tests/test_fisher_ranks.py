"""Tests for Fisher rank assignment algorithm."""

from __future__ import annotations

import unittest

from acestep.training_v2.fisher.ranks import (
    _median,
    _round_to_multiple,
    _select_modules,
    _to_peft_key,
    assign_ranks,
)


class TestAssignRanks(unittest.TestCase):
    """Success-path and edge-case tests for assign_ranks()."""

    def _make_scores(self, n: int, values=None):
        """Helper to build fisher_scores and spectral_ranks dicts."""
        names = [f"decoder.layers.{i}.self_attn.q_proj" for i in range(n)]
        if values is None:
            values = [0.01 * (n - i) for i in range(n)]
        fisher = dict(zip(names, values))
        spectral = {name: 50 for name in names}
        return fisher, spectral, names

    def test_basic_assignment(self):
        """Known Fisher + spectral scores produce valid rank assignments."""
        fisher, spectral, names = self._make_scores(10)
        targets, rank_pat, alpha_pat = assign_ranks(
            fisher, spectral, base_rank=64, rank_min=16, rank_max=128,
            inclusion_percentile=0.40,
        )
        self.assertTrue(len(rank_pat) > 0)
        self.assertEqual(len(rank_pat), len(alpha_pat))
        for key, rank in rank_pat.items():
            self.assertGreaterEqual(rank, 16)
            self.assertLessEqual(rank, 128)
            self.assertEqual(alpha_pat[key], rank * 2)

    def test_all_equal_fisher_gives_base_rank(self):
        """When all Fisher scores are equal, every module gets base_rank."""
        fisher, spectral, _ = self._make_scores(10, values=[0.01] * 10)
        _, rank_pat, _ = assign_ranks(
            fisher, spectral, base_rank=64, rank_min=16, rank_max=128,
        )
        for rank in rank_pat.values():
            self.assertEqual(rank, 64)

    def test_single_module(self):
        """A single module should get base_rank."""
        fisher, spectral, _ = self._make_scores(1, values=[0.05])
        _, rank_pat, _ = assign_ranks(
            fisher, spectral, base_rank=64, rank_min=16, rank_max=128,
            inclusion_percentile=1.0,
        )
        self.assertEqual(len(rank_pat), 1)
        self.assertEqual(list(rank_pat.values())[0], 64)

    def test_rank_clipping_min(self):
        """Very low Fisher factor should be clipped to rank_min."""
        fisher = {
            "decoder.layers.0.self_attn.q_proj": 0.001,
            "decoder.layers.1.self_attn.q_proj": 0.1,
        }
        spectral = {k: 10 for k in fisher}
        _, rank_pat, _ = assign_ranks(
            fisher, spectral, base_rank=64, rank_min=16, rank_max=128,
            inclusion_percentile=1.0,
        )
        for rank in rank_pat.values():
            self.assertGreaterEqual(rank, 16)

    def test_rank_clipping_max(self):
        """Very high Fisher factor should be clipped to rank_max."""
        fisher = {
            "decoder.layers.0.self_attn.q_proj": 10.0,
            "decoder.layers.1.self_attn.q_proj": 0.001,
        }
        spectral = {k: 200 for k in fisher}
        _, rank_pat, _ = assign_ranks(
            fisher, spectral, base_rank=64, rank_min=16, rank_max=128,
            inclusion_percentile=1.0,
        )
        for rank in rank_pat.values():
            self.assertLessEqual(rank, 128)

    def test_round_to_8(self):
        """Ranks should be multiples of 8."""
        fisher, spectral, _ = self._make_scores(20)
        _, rank_pat, _ = assign_ranks(fisher, spectral, base_rank=64)
        for rank in rank_pat.values():
            self.assertEqual(rank % 8, 0, f"rank {rank} not a multiple of 8")

    def test_alpha_is_twice_rank(self):
        """alpha_pattern values must be 2 * rank_pattern values."""
        fisher, spectral, _ = self._make_scores(10)
        _, rank_pat, alpha_pat = assign_ranks(fisher, spectral, base_rank=64)
        for key in rank_pat:
            self.assertEqual(alpha_pat[key], rank_pat[key] * 2)

    def test_excluded_modules_get_no_rank(self):
        """Modules below the inclusion threshold should not appear."""
        fisher, spectral, names = self._make_scores(10)
        _, rank_pat, _ = assign_ranks(
            fisher, spectral, base_rank=64, inclusion_percentile=0.30,
        )
        self.assertTrue(len(rank_pat) < 10)

    def test_empty_input(self):
        """Empty fisher_scores returns empty results."""
        targets, rank_pat, alpha_pat = assign_ranks({}, {}, base_rank=64)
        self.assertEqual(targets, [])
        self.assertEqual(rank_pat, {})
        self.assertEqual(alpha_pat, {})


class TestHelpers(unittest.TestCase):
    """Unit tests for internal helpers."""

    def test_round_to_multiple(self):
        self.assertEqual(_round_to_multiple(33, 8), 32)
        self.assertEqual(_round_to_multiple(37, 8), 40)
        self.assertEqual(_round_to_multiple(64, 8), 64)
        self.assertEqual(_round_to_multiple(36, 8), 32)  # banker's rounding: 4.5 -> 4

    def test_median_odd(self):
        self.assertAlmostEqual(_median([1, 3, 5]), 3.0)

    def test_median_even(self):
        self.assertAlmostEqual(_median([1, 3, 5, 7]), 4.0)

    def test_median_empty(self):
        self.assertAlmostEqual(_median([]), 0.0)

    def test_to_peft_key_strips_decoder(self):
        self.assertEqual(
            _to_peft_key("decoder.layers.0.self_attn.q_proj"),
            "layers.0.self_attn.q_proj",
        )

    def test_to_peft_key_no_decoder(self):
        self.assertEqual(
            _to_peft_key("layers.0.self_attn.q_proj"),
            "layers.0.self_attn.q_proj",
        )

    def test_select_modules_top_percentile(self):
        scores = {"a": 0.9, "b": 0.5, "c": 0.1}
        result = _select_modules(scores, 0.34)
        self.assertEqual(result, ["a"])


if __name__ == "__main__":
    unittest.main()
