"""
Tests for selective gradient checkpointing.

Covers:
    - Layer selection algorithm (_select_layer_indices)
    - Application and removal of checkpointing
    - VRAM activation estimation (re-exported from vram_estimation)
    - Suggestion function
    - Menu option builder
"""

from __future__ import annotations

import unittest

import torch.nn as nn

from sidestep_engine.core.selective_checkpointing import (
    _ORIG_FWD_ATTR,
    _select_layer_indices,
    apply_selective_checkpointing,
    build_checkpointing_options,
    estimate_activation_mb,
    remove_selective_checkpointing,
    suggest_checkpointing,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_mock_decoder(num_layers: int = 24) -> nn.Module:
    """Create a mock decoder with `.layers` ModuleList."""
    decoder = nn.Module()
    decoder.layers = nn.ModuleList([nn.Linear(4, 4) for _ in range(num_layers)])
    return decoder


# ---------------------------------------------------------------------------
# _select_layer_indices
# ---------------------------------------------------------------------------

class TestSelectLayerIndices(unittest.TestCase):
    """Tests for the even-spacing algorithm."""

    def test_ratio_zero(self):
        self.assertEqual(_select_layer_indices(24, 0.0), [])

    def test_ratio_one(self):
        self.assertEqual(_select_layer_indices(24, 1.0), list(range(24)))

    def test_ratio_half(self):
        indices = _select_layer_indices(24, 0.5)
        self.assertEqual(len(indices), 12)

    def test_single_layer(self):
        indices = _select_layer_indices(24, 0.05)
        self.assertEqual(len(indices), 1)

    def test_no_layers(self):
        self.assertEqual(_select_layer_indices(0, 1.0), [])

    def test_sorted(self):
        indices = _select_layer_indices(24, 0.3)
        self.assertEqual(indices, sorted(indices))


# ---------------------------------------------------------------------------
# apply / remove
# ---------------------------------------------------------------------------

class TestApplyRemove(unittest.TestCase):

    def test_apply_half_wraps_correct_number(self):
        decoder = _make_mock_decoder(24)
        n = apply_selective_checkpointing(decoder, 0.5)
        self.assertEqual(n, 12)
        wrapped = sum(1 for l in decoder.layers if hasattr(l, _ORIG_FWD_ATTR))
        self.assertEqual(wrapped, 12)

    def test_apply_full_wraps_all(self):
        decoder = _make_mock_decoder(8)
        n = apply_selective_checkpointing(decoder, 1.0)
        self.assertEqual(n, 8)

    def test_apply_zero_wraps_none(self):
        decoder = _make_mock_decoder(8)
        n = apply_selective_checkpointing(decoder, 0.0)
        self.assertEqual(n, 0)

    def test_remove_restores_all(self):
        decoder = _make_mock_decoder(8)
        apply_selective_checkpointing(decoder, 1.0)
        restored = remove_selective_checkpointing(decoder)
        self.assertEqual(restored, 8)
        for layer in decoder.layers:
            self.assertFalse(hasattr(layer, _ORIG_FWD_ATTR))

    def test_double_apply_is_idempotent(self):
        decoder = _make_mock_decoder(8)
        apply_selective_checkpointing(decoder, 1.0)
        n2 = apply_selective_checkpointing(decoder, 1.0)
        self.assertEqual(n2, 0)

    def test_no_layers_attribute_returns_zero(self):
        decoder = nn.Module()
        n = apply_selective_checkpointing(decoder, 0.5)
        self.assertEqual(n, 0)


# ---------------------------------------------------------------------------
# VRAM estimation (re-exported)
# ---------------------------------------------------------------------------

class TestEstimateActivationMb(unittest.TestCase):

    def test_zero_layers_is_near_zero(self):
        """Zero uncheckpointed layers still has a small backward peak."""
        est = estimate_activation_mb(0, 1, 60)
        self.assertLess(est, 20)

    def test_scales_with_batch_size(self):
        single = estimate_activation_mb(24, 1, 60)
        double = estimate_activation_mb(24, 2, 60)
        self.assertAlmostEqual(double / single, 2.0, places=0)

    def test_scales_with_layers(self):
        half = estimate_activation_mb(12, 1, 60)
        full = estimate_activation_mb(24, 1, 60)
        self.assertAlmostEqual(full / half, 2.0, places=0)

    def test_sdpa_scales_linearly(self):
        short = estimate_activation_mb(24, 1, 60, attn_backend="sdpa")
        long = estimate_activation_mb(24, 1, 120, attn_backend="sdpa")
        self.assertAlmostEqual(long / short, 2.0, places=0)

    def test_none_chunk_uses_default(self):
        est = estimate_activation_mb(24, 1, None)
        self.assertGreater(est, 0)


# ---------------------------------------------------------------------------
# suggest_checkpointing
# ---------------------------------------------------------------------------

class TestSuggestCheckpointing(unittest.TestCase):

    def test_tiny_gpu_suggests_heavy_checkpointing(self):
        ratio, _ = suggest_checkpointing(8 * 1024)
        self.assertGreaterEqual(ratio, 0.5)

    def test_large_gpu_suggests_less(self):
        ratio, _ = suggest_checkpointing(80 * 1024, chunk_duration_s=60)
        self.assertLessEqual(ratio, 0.5)

    def test_returns_string_reason(self):
        _, reason = suggest_checkpointing(16 * 1024)
        self.assertIsInstance(reason, str)
        self.assertGreater(len(reason), 0)

    def test_offload_affects_suggestion(self):
        r_offload, _ = suggest_checkpointing(12 * 1024, offload_encoder=True)
        r_no_offload, _ = suggest_checkpointing(12 * 1024, offload_encoder=False)
        self.assertGreaterEqual(r_no_offload, r_offload)

    def test_free_vram_tightens_budget(self):
        r_total_only, _ = suggest_checkpointing(16 * 1024)
        r_with_free, _ = suggest_checkpointing(16 * 1024, vram_free_mb=7 * 1024)
        self.assertGreaterEqual(r_with_free, r_total_only)

    def test_very_low_free_vram_suggests_full(self):
        ratio, _ = suggest_checkpointing(16 * 1024, vram_free_mb=4 * 1024)
        self.assertEqual(ratio, 1.0)


# ---------------------------------------------------------------------------
# build_checkpointing_options
# ---------------------------------------------------------------------------

class TestBuildOptions(unittest.TestCase):

    def test_returns_five_options(self):
        opts = build_checkpointing_options(16 * 1024)
        self.assertEqual(len(opts), 5)

    def test_option_ratios(self):
        opts = build_checkpointing_options(16 * 1024)
        ratios = [r for r, _, _ in opts]
        self.assertEqual(ratios, [1.0, 0.75, 0.5, 0.25, 0.0])

    def test_estimates_are_positive(self):
        opts = build_checkpointing_options(16 * 1024, batch_size=1, chunk_duration_s=60)
        for _, _, est in opts:
            self.assertGreaterEqual(est, 0)


if __name__ == "__main__":
    unittest.main()
