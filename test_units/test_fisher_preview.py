"""Tests for Fisher preview output and confirmation."""

from __future__ import annotations

import io
import sys
import unittest

from sidestep_engine.analysis.fisher.preview import _low_confidence_modules, print_preview


class TestLowConfidenceModules(unittest.TestCase):
    """Test low-confidence detection logic."""

    def test_high_variance_flagged(self):
        """Modules where std > 50% of mean should be flagged."""
        scores = {"a": 0.01, "b": 0.01}
        stds = {"a": 0.002, "b": 0.008}  # b has 80% std/mean
        result = _low_confidence_modules(scores, stds)
        names = [r[0] for r in result]
        self.assertIn("b", names)

    def test_low_variance_not_flagged(self):
        """Stable modules should not be flagged."""
        scores = {"a": 0.01, "b": 0.01}
        stds = {"a": 0.001, "b": 0.001}
        result = _low_confidence_modules(scores, stds)
        self.assertEqual(len(result), 0)

    def test_zero_mean_not_flagged(self):
        """Zero-mean modules should not cause division errors."""
        scores = {"a": 0.0}
        stds = {"a": 0.0}
        result = _low_confidence_modules(scores, stds)
        self.assertEqual(len(result), 0)


class TestPrintPreview(unittest.TestCase):
    """Verify print_preview runs without error and produces output."""

    def test_produces_output(self):
        """print_preview should write to stdout."""
        captured = io.StringIO()
        old_stdout = sys.stdout
        sys.stdout = captured

        try:
            print_preview(
                fisher_scores={"decoder.layers.0.self_attn.q_proj": 0.01},
                fisher_stds={"decoder.layers.0.self_attn.q_proj": 0.001},
                spectral_ranks={"decoder.layers.0.self_attn.q_proj": 50},
                rank_pattern={"layers.0.self_attn.q_proj": 64},
                alpha_pattern={"layers.0.self_attn.q_proj": 128},
                target_modules=["self_attn.q_proj"],
                excluded=["decoder.layers.1.cross_attn.k_proj"],
                base_rank=64, rank_min=16, rank_max=128,
                total_batches=10, num_runs=2,
                variant="turbo", timestep_focus="texture",
                num_analyzed=264,
            )
        finally:
            sys.stdout = old_stdout

        output = captured.getvalue()
        self.assertIn("Fisher", output)
        self.assertIn("turbo", output)
        self.assertIn("texture", output)


if __name__ == "__main__":
    unittest.main()
