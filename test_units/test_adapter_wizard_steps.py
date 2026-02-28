"""Tests for per-adapter wizard step functions.

Verifies each step_* function populates expected answer keys.
Uses mock prompts to avoid interactive input.
"""

import unittest
from contextlib import ExitStack
from unittest.mock import patch

from sidestep_engine.ui.flows.train_steps_adapter import (
    step_lora,
    step_dora,
    step_lokr,
    step_loha,
    step_oft,
    ADAPTER_STEP_MAP,
    ADAPTER_LABEL_MAP,
)

# All modules that import prompt helpers directly need patching at their level.
_PROMPT_MODULES = [
    "sidestep_engine.ui.prompt_helpers",
    "sidestep_engine.ui.flows.train_steps_adapter_lora",
    "sidestep_engine.ui.flows.train_steps_adapter_dora",
    "sidestep_engine.ui.flows.train_steps_adapter_lokr",
    "sidestep_engine.ui.flows.train_steps_adapter_loha",
    "sidestep_engine.ui.flows.train_steps_adapter_oft",
    "sidestep_engine.ui.flows.train_steps_adapter",
]

_MOCK_FNS = dict(
    ask=lambda *a, **kw: kw.get("default", ""),
    ask_bool=lambda *a, **kw: kw.get("default", False),
    menu=lambda *a, **kw: "both",
    section=lambda *a, **kw: None,
    print_message=lambda *a, **kw: None,
    print_rich=lambda *a, **kw: None,
)


def _patch_prompts():
    """Return a context manager that patches prompt helpers across all modules."""
    stack = ExitStack()
    for mod in _PROMPT_MODULES:
        for name, fn in _MOCK_FNS.items():
            try:
                stack.enter_context(patch(f"{mod}.{name}", side_effect=fn))
            except AttributeError:
                pass  # module doesn't import this name directly
    return stack


class TestStepLora(unittest.TestCase):
    """Test step_lora populates expected keys."""

    def test_basic_mode_defaults(self):
        a = {"config_mode": "basic"}
        with _patch_prompts():
            step_lora(a)
        self.assertIn("rank", a)
        self.assertIn("alpha", a)
        self.assertEqual(a["dropout"], 0.1)
        self.assertEqual(a["attention_type"], "both")
        self.assertTrue(a["target_mlp"])

    def test_fisher_map_skips_rank(self):
        a = {
            "config_mode": "basic",
            "_fisher_map_cached": True,
            "_fisher_map_cached_dir": "/tmp/fake",
            "dataset_dir": "/tmp/fake",
        }
        with _patch_prompts():
            step_lora(a)
        self.assertEqual(a["dropout"], 0.1)
        self.assertNotIn("rank", a)


class TestStepDora(unittest.TestCase):
    """Test step_dora sets use_dora=True."""

    def test_sets_use_dora(self):
        a = {"config_mode": "basic"}
        with _patch_prompts():
            step_dora(a)
        self.assertTrue(a.get("use_dora"))
        self.assertIn("rank", a)

    def test_fisher_map_with_dora(self):
        a = {"config_mode": "basic", "_fisher_map_cached": True}
        with _patch_prompts():
            step_dora(a)
        self.assertTrue(a["use_dora"])
        self.assertEqual(a["dropout"], 0.1)


class TestStepLokr(unittest.TestCase):
    """Test step_lokr populates LoKR keys."""

    def test_basic_mode_defaults(self):
        a = {"config_mode": "basic"}
        with _patch_prompts():
            step_lokr(a)
        self.assertIn("lokr_linear_dim", a)
        self.assertIn("lokr_linear_alpha", a)
        self.assertIn("lokr_factor", a)
        self.assertFalse(a["lokr_decompose_both"])
        self.assertEqual(a["attention_type"], "both")


class TestStepLoha(unittest.TestCase):
    """Test step_loha populates LoHA keys."""

    def test_basic_mode_defaults(self):
        a = {"config_mode": "basic"}
        with _patch_prompts():
            step_loha(a)
        self.assertIn("loha_linear_dim", a)
        self.assertIn("loha_linear_alpha", a)
        self.assertIn("loha_factor", a)
        self.assertFalse(a["loha_use_tucker"])
        self.assertTrue(a["target_mlp"])


class TestStepOft(unittest.TestCase):
    """Test step_oft populates OFT keys."""

    def test_basic_mode_defaults(self):
        a = {"config_mode": "basic"}
        with _patch_prompts():
            step_oft(a)
        self.assertIn("oft_block_size", a)
        self.assertFalse(a["oft_coft"])
        self.assertAlmostEqual(a["oft_eps"], 6e-5)
        self.assertTrue(a["target_mlp"])


class TestAdapterMaps(unittest.TestCase):
    """Test ADAPTER_STEP_MAP and ADAPTER_LABEL_MAP completeness."""

    def test_all_adapters_in_step_map(self):
        for adapter in ("lora", "dora", "lokr", "loha", "oft"):
            self.assertIn(adapter, ADAPTER_STEP_MAP)
            self.assertTrue(callable(ADAPTER_STEP_MAP[adapter]))

    def test_all_adapters_in_label_map(self):
        for adapter in ("lora", "dora", "lokr", "loha", "oft"):
            self.assertIn(adapter, ADAPTER_LABEL_MAP)
            self.assertIsInstance(ADAPTER_LABEL_MAP[adapter], str)

    def test_oft_label_experimental(self):
        self.assertIn("Experimental", ADAPTER_LABEL_MAP["oft"])


if __name__ == "__main__":
    unittest.main()
