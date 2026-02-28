"""
Tests for VRAM estimation module.

Covers:
    - Per-layer activation calculation (first-principles)
    - Activation estimation (scaling, flash discount)
    - Optimizer state estimation (adapter types, optimizer types)
    - Trainable param counting (GQA-aware)
    - Peak VRAM estimation (combined breakdown)
    - VRAM verdict (soft gate)
    - Suggestion function (budget-based)
    - Menu option builder
    - Attention backend detection (mocked)
"""

from __future__ import annotations

import unittest
from unittest.mock import patch

from sidestep_engine.core.vram_estimation import (
    _FLASH_ATTN_DISCOUNT,
    _activation_bytes_per_layer,
    _estimate_trainable_params,
    _seq_len_from_chunk,
    build_checkpointing_options,
    detect_attn_backend,
    estimate_activation_mb,
    estimate_optimizer_state_mb,
    estimate_peak_vram_mb,
    suggest_checkpointing,
    system_vram_used_mb,
    vram_verdict,
)


# ---------------------------------------------------------------------------
# Per-layer activation bytes
# ---------------------------------------------------------------------------

class TestActivationBytesPerLayer(unittest.TestCase):
    """Tests for the first-principles per-layer activation calculation."""

    def test_scales_linearly_with_batch(self):
        b1 = _activation_bytes_per_layer(1, 750)
        b2 = _activation_bytes_per_layer(2, 750)
        self.assertAlmostEqual(b2 / b1, 2.0, places=1)

    def test_scales_linearly_with_seq_len(self):
        s1 = _activation_bytes_per_layer(1, 750)
        s2 = _activation_bytes_per_layer(1, 1500)
        self.assertAlmostEqual(s2 / s1, 2.0, places=1)

    def test_mlp_dominates(self):
        """MLP intermediates should be >40% of total per-layer activation."""
        total = _activation_bytes_per_layer(1, 750)
        B, S, I, bpe = 1, 750, 6144, 2
        mlp_intermediates = 4 * B * S * I * bpe  # gate, up, silu, intermediate
        self.assertGreater(mlp_intermediates / total, 0.40)

    def test_flash_attn_reduces(self):
        sdpa = _activation_bytes_per_layer(1, 750, attn_backend="sdpa")
        flash = _activation_bytes_per_layer(1, 750, attn_backend="flash_attention_2")
        self.assertLess(flash, sdpa)

    def test_positive_for_typical_config(self):
        result = _activation_bytes_per_layer(1, 750)
        self.assertGreater(result, 0)


# ---------------------------------------------------------------------------
# Sequence length helper
# ---------------------------------------------------------------------------

class TestSeqLenFromChunk(unittest.TestCase):

    def test_60s_gives_750(self):
        self.assertEqual(_seq_len_from_chunk(60), 750)

    def test_120s_gives_1500(self):
        self.assertEqual(_seq_len_from_chunk(120), 1500)

    def test_none_uses_default_60s(self):
        self.assertEqual(_seq_len_from_chunk(None), 750)

    def test_zero_uses_max_audio(self):
        """0 means chunking disabled -> worst-case 240s."""
        self.assertEqual(_seq_len_from_chunk(0), 3000)


# ---------------------------------------------------------------------------
# estimate_activation_mb
# ---------------------------------------------------------------------------

class TestEstimateActivationMb(unittest.TestCase):

    def test_zero_layers_is_near_zero(self):
        """Zero uncheckpointed layers still has a small backward peak."""
        est = estimate_activation_mb(0, 1, 60)
        self.assertLess(est, 20)  # backward peak only, very small

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

    def test_flash_attn_cheaper_than_sdpa(self):
        sdpa = estimate_activation_mb(24, 1, 60, attn_backend="sdpa")
        flash = estimate_activation_mb(24, 1, 60, attn_backend="flash_attention_2")
        self.assertLess(flash, sdpa)

    def test_none_chunk_uses_default_60s(self):
        default = estimate_activation_mb(24, 1, None)
        explicit = estimate_activation_mb(24, 1, 60)
        self.assertAlmostEqual(default, explicit, places=1)

    def test_chunk_duration_increases_vram(self):
        short_chunk = estimate_activation_mb(24, 1, 60)
        long_chunk = estimate_activation_mb(24, 1, 240)
        self.assertGreater(long_chunk, short_chunk)

    def test_realistic_range_single_layer_60s(self):
        """One layer, batch=1, 60s should be ~80-200 MB."""
        est = estimate_activation_mb(1, 1, 60)
        self.assertGreater(est, 50)
        self.assertLess(est, 200)

    def test_no_checkpointing_batch4_240s_realistic(self):
        """batch=4, 240s, no ckpt should be ~30-45 GB (matches 48GB card observation)."""
        est_mb = estimate_activation_mb(24, 4, 240)
        est_gb = est_mb / 1024
        self.assertGreater(est_gb, 25)
        self.assertLess(est_gb, 50)


# ---------------------------------------------------------------------------
# Trainable param estimation (GQA-aware)
# ---------------------------------------------------------------------------

class TestEstimateTrainableParams(unittest.TestCase):

    def test_lora_gqa_aware(self):
        """LoRA params should reflect GQA: k/v projections are 1024, not 2048."""
        params = _estimate_trainable_params("lora", rank=64, num_layers=24)
        # Per attention block: rank * (4096 + 3072 + 3072 + 4096) = rank * 14336
        # Both (self + cross): 2 * 24 * 64 * 14336 = 44,040,192
        expected = 2 * 24 * 64 * (4096 + 3072 + 3072 + 4096)
        self.assertEqual(params, expected)

    def test_mlp_targeting_increases_params(self):
        no_mlp = _estimate_trainable_params("lora", rank=64, target_mlp=False)
        with_mlp = _estimate_trainable_params("lora", rank=64, target_mlp=True)
        self.assertGreater(with_mlp, no_mlp)

    def test_oft_uses_rank_squared(self):
        params = _estimate_trainable_params("oft", rank=64, num_layers=24)
        lora_params = _estimate_trainable_params("lora", rank=64, num_layers=24)
        self.assertLess(params, lora_params)

    def test_self_only_half_of_both(self):
        both = _estimate_trainable_params("lora", rank=64, attention_type="both")
        self_only = _estimate_trainable_params("lora", rank=64, attention_type="self")
        self.assertAlmostEqual(self_only / both, 0.5, places=2)

    def test_higher_rank_more_params(self):
        low = _estimate_trainable_params("lora", rank=8)
        high = _estimate_trainable_params("lora", rank=64)
        self.assertGreater(high, low)


# ---------------------------------------------------------------------------
# Optimizer state estimation
# ---------------------------------------------------------------------------

class TestEstimateOptimizerState(unittest.TestCase):

    def test_adamw_more_than_adamw8bit(self):
        full = estimate_optimizer_state_mb(optimizer_type="adamw")
        eight_bit = estimate_optimizer_state_mb(optimizer_type="adamw8bit")
        self.assertGreater(full, eight_bit)

    def test_adafactor_less_than_adamw(self):
        ada = estimate_optimizer_state_mb(optimizer_type="adafactor")
        adamw = estimate_optimizer_state_mb(optimizer_type="adamw")
        self.assertLess(ada, adamw)

    def test_target_mlp_increases_states(self):
        no_mlp = estimate_optimizer_state_mb(target_mlp=False)
        with_mlp = estimate_optimizer_state_mb(target_mlp=True)
        self.assertGreater(with_mlp, no_mlp)

    def test_higher_rank_more_states(self):
        low = estimate_optimizer_state_mb(rank=8)
        high = estimate_optimizer_state_mb(rank=64)
        self.assertGreater(high, low)

    def test_prodigy_more_than_adamw(self):
        """Prodigy has more state per param than AdamW."""
        prodigy = estimate_optimizer_state_mb(optimizer_type="prodigy", rank=64)
        adamw = estimate_optimizer_state_mb(optimizer_type="adamw", rank=64)
        self.assertGreater(prodigy, adamw)


# ---------------------------------------------------------------------------
# Peak VRAM estimation
# ---------------------------------------------------------------------------

class TestEstimatePeakVram(unittest.TestCase):

    def test_returns_breakdown_dict(self):
        peak, bd = estimate_peak_vram_mb(1.0, batch_size=1, chunk_duration_s=60)
        self.assertIn("model_mb", bd)
        self.assertIn("activation_mb", bd)
        self.assertIn("optimizer_mb", bd)
        self.assertIn("gradient_mb", bd)
        self.assertIn("adapter_mb", bd)
        self.assertIn("cuda_overhead_mb", bd)
        self.assertIn("fragmentation_mb", bd)
        self.assertIn("peak_mb", bd)
        self.assertAlmostEqual(peak, bd["peak_mb"])

    def test_full_checkpointing_cheapest(self):
        full, _ = estimate_peak_vram_mb(1.0, chunk_duration_s=60)
        none, _ = estimate_peak_vram_mb(0.0, chunk_duration_s=60)
        self.assertLess(full, none)

    def test_checkpointing_dramatic_difference(self):
        """Full ckpt vs none should differ by >2x for batch=2, 240s."""
        full, _ = estimate_peak_vram_mb(1.0, batch_size=2, chunk_duration_s=240)
        none, _ = estimate_peak_vram_mb(0.0, batch_size=2, chunk_duration_s=240)
        self.assertGreater(none / full, 2.0)

    def test_flash_reduces_peak(self):
        sdpa, _ = estimate_peak_vram_mb(0.5, attn_backend="sdpa", chunk_duration_s=120)
        flash, _ = estimate_peak_vram_mb(0.5, attn_backend="flash_attention_2", chunk_duration_s=120)
        self.assertLess(flash, sdpa)

    def test_optimizer_type_affects_peak(self):
        adamw, _ = estimate_peak_vram_mb(1.0, optimizer_type="adamw", rank=64)
        eight_bit, _ = estimate_peak_vram_mb(1.0, optimizer_type="adamw8bit", rank=64)
        self.assertGreater(adamw, eight_bit)

    def test_batch4_240s_no_ckpt_near_48gb(self):
        """batch=4, 240s, no ckpt should estimate ~35-50 GB."""
        peak, _ = estimate_peak_vram_mb(
            0.0, batch_size=4, chunk_duration_s=240,
            optimizer_type="adamw8bit", rank=64,
        )
        peak_gb = peak / 1024
        self.assertGreater(peak_gb, 30)
        self.assertLess(peak_gb, 55)

    def test_basic_8gb_config(self):
        """rank=16, batch=1, 60s, full ckpt should fit in 8 GB."""
        peak, _ = estimate_peak_vram_mb(
            1.0, batch_size=1, chunk_duration_s=60,
            optimizer_type="adamw8bit", rank=16,
        )
        peak_gb = peak / 1024
        self.assertLess(peak_gb, 8.0)


# ---------------------------------------------------------------------------
# VRAM verdict
# ---------------------------------------------------------------------------

class TestVramVerdict(unittest.TestCase):

    def test_comfortable_is_green(self):
        self.assertEqual(vram_verdict(6000, 24000), "green")

    def test_tight_is_yellow(self):
        self.assertEqual(vram_verdict(20000, 24000), "yellow")

    def test_over_is_red(self):
        self.assertEqual(vram_verdict(23500, 24000), "red")

    def test_way_over_is_red(self):
        self.assertEqual(vram_verdict(30000, 24000), "red")

    def test_zero_gpu_is_green(self):
        """Can't detect GPU -> don't block."""
        self.assertEqual(vram_verdict(99999, 0), "green")

    def test_exact_80_pct_is_yellow(self):
        self.assertEqual(vram_verdict(19200, 24000), "yellow")

    def test_system_used_tightens_budget(self):
        """6 GB peak on 24 GB card is green, but not if 20 GB is already used."""
        self.assertEqual(vram_verdict(6000, 24000, system_used_mb=0), "green")
        self.assertEqual(vram_verdict(6000, 24000, system_used_mb=20000), "red")

    def test_system_used_makes_yellow(self):
        """10 GB peak on 24 GB card: green normally, yellow with 12 GB system use."""
        self.assertEqual(vram_verdict(10000, 24000, system_used_mb=0), "green")
        # 10000 / (24000 - 12000) = 83% -> yellow
        self.assertEqual(vram_verdict(10000, 24000, system_used_mb=12000), "yellow")


class TestSystemVramUsed(unittest.TestCase):

    def test_normal_case(self):
        self.assertAlmostEqual(system_vram_used_mb(24000, 20000), 4000.0)

    def test_none_total_returns_zero(self):
        self.assertEqual(system_vram_used_mb(None, 20000), 0.0)

    def test_none_free_returns_zero(self):
        self.assertEqual(system_vram_used_mb(24000, None), 0.0)

    def test_both_none_returns_zero(self):
        self.assertEqual(system_vram_used_mb(None, None), 0.0)

    def test_no_negative(self):
        """Even if free > total (shouldn't happen), clamp to 0."""
        self.assertEqual(system_vram_used_mb(24000, 30000), 0.0)


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

    def test_free_vram_tightens_budget(self):
        r_total, _ = suggest_checkpointing(16 * 1024)
        r_free, _ = suggest_checkpointing(16 * 1024, vram_free_mb=7 * 1024)
        self.assertGreaterEqual(r_free, r_total)

    def test_8bit_optimizer_loosens_budget(self):
        r_full, _ = suggest_checkpointing(
            24 * 1024, chunk_duration_s=60, optimizer_type="adamw", rank=64,
        )
        r_8bit, _ = suggest_checkpointing(
            24 * 1024, chunk_duration_s=60, optimizer_type="adamw8bit", rank=64,
        )
        self.assertLessEqual(r_8bit, r_full)


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

    def test_accepts_all_kwargs(self):
        opts = build_checkpointing_options(
            16 * 1024, batch_size=1, chunk_duration_s=120,
            attn_backend="flash_attention_2", adapter_type="lokr",
            rank=64, target_mlp=True, optimizer_type="adamw8bit",
        )
        self.assertEqual(len(opts), 5)


# ---------------------------------------------------------------------------
# detect_attn_backend (mocked)
# ---------------------------------------------------------------------------

class TestDetectAttnBackend(unittest.TestCase):

    @patch("sidestep_engine.models.loader._flash_attention_unavailable_reason", return_value=None)
    def test_returns_flash_when_available(self, mock_reason):
        result = detect_attn_backend("cuda:0", "bf16")
        self.assertEqual(result, "flash_attention_2")

    @patch("sidestep_engine.models.loader._flash_attention_unavailable_reason",
           return_value="no flash_attn package")
    def test_returns_sdpa_when_flash_unavailable(self, mock_reason):
        result = detect_attn_backend("cuda:0", "bf16")
        self.assertEqual(result, "sdpa")

    def test_cpu_returns_sdpa(self):
        result = detect_attn_backend("cpu", "fp32")
        self.assertEqual(result, "sdpa")


if __name__ == "__main__":
    unittest.main()
