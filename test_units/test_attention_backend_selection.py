"""Tests for attention backend preflight decisions."""

from __future__ import annotations

import unittest
from unittest.mock import patch


class TestAttentionBackendSelection(unittest.TestCase):
    """Cover FA2 availability checks and fallback candidate selection."""

    def test_cpu_device_skips_flash_attention(self) -> None:
        from sidestep_engine.models.loader import _flash_attention_unavailable_reason

        reason = _flash_attention_unavailable_reason("cpu", "bf16")
        assert reason is not None
        self.assertIn("not CUDA", reason)

    def test_fp32_precision_skips_flash_attention(self) -> None:
        from sidestep_engine.models.loader import _flash_attention_unavailable_reason

        reason = _flash_attention_unavailable_reason("cuda:0", "fp32")
        assert reason is not None
        self.assertIn("fp32", reason)

    def test_candidate_list_starts_with_sdpa_when_fa2_unavailable(self) -> None:
        from sidestep_engine.models.loader import _choose_attention_candidates

        with patch(
            "sidestep_engine.models.loader._flash_attention_unavailable_reason",
            return_value="missing flash_attn",
        ):
            candidates, reason = _choose_attention_candidates("cuda:0", "bf16")
            self.assertEqual(candidates[0], "sdpa")
            self.assertEqual(reason, "missing flash_attn")


if __name__ == "__main__":
    unittest.main()
