"""Tests for per-attention-type projection selection.

Covers both the wizard flow (_resolve_wizard_projections in flows_common)
and the CLI path (resolve_target_modules in cli/validation), verifying
that split self/cross projections are merged with correct prefixes and
that backward-compatible single-list paths still work.
"""

from __future__ import annotations

import unittest


class TestResolveWizardProjections(unittest.TestCase):
    """Unit tests for _resolve_wizard_projections."""

    def _resolve(self, answers: dict) -> list:
        from acestep.training_v2.ui.flows_common import _resolve_wizard_projections
        return _resolve_wizard_projections(answers)

    # -- "both" with split projections ---

    def test_both_split_same_projections(self):
        """Both attention types, same projections for each."""
        a = {
            "attention_type": "both",
            "self_target_modules_str": "q_proj k_proj v_proj o_proj",
            "cross_target_modules_str": "q_proj k_proj v_proj o_proj",
        }
        result = self._resolve(a)
        self.assertEqual(result, [
            "self_attn.q_proj", "self_attn.k_proj",
            "self_attn.v_proj", "self_attn.o_proj",
            "cross_attn.q_proj", "cross_attn.k_proj",
            "cross_attn.v_proj", "cross_attn.o_proj",
        ])

    def test_both_split_different_projections(self):
        """Both attention types, different projections for each."""
        a = {
            "attention_type": "both",
            "self_target_modules_str": "q_proj v_proj",
            "cross_target_modules_str": "q_proj k_proj v_proj o_proj",
        }
        result = self._resolve(a)
        self.assertEqual(result, [
            "self_attn.q_proj", "self_attn.v_proj",
            "cross_attn.q_proj", "cross_attn.k_proj",
            "cross_attn.v_proj", "cross_attn.o_proj",
        ])

    def test_both_split_only_self_provided(self):
        """Only self projections key present; cross falls back to default."""
        a = {
            "attention_type": "both",
            "self_target_modules_str": "q_proj",
        }
        result = self._resolve(a)
        self.assertEqual(result, [
            "self_attn.q_proj",
            "cross_attn.q_proj", "cross_attn.k_proj",
            "cross_attn.v_proj", "cross_attn.o_proj",
        ])

    def test_both_split_preserves_qualified_modules(self):
        """Already-qualified module names (containing '.') pass through."""
        a = {
            "attention_type": "both",
            "self_target_modules_str": "q_proj self_attn.custom_layer",
            "cross_target_modules_str": "v_proj",
        }
        result = self._resolve(a)
        self.assertEqual(result, [
            "self_attn.q_proj", "self_attn.custom_layer",
            "cross_attn.v_proj",
        ])

    # -- "both" without split (backward compat) ---

    def test_both_no_split_uses_single_string(self):
        """Backward compat: no split keys means single string passthrough."""
        a = {
            "attention_type": "both",
            "target_modules_str": "q_proj v_proj",
        }
        result = self._resolve(a)
        self.assertEqual(result, ["q_proj", "v_proj"])

    def test_both_no_keys_uses_default(self):
        """No projection keys at all; uses default."""
        a = {"attention_type": "both"}
        result = self._resolve(a)
        self.assertEqual(result, ["q_proj", "k_proj", "v_proj", "o_proj"])

    # -- "self" and "cross" (single-type paths) ---

    def test_self_returns_unprefixed(self):
        """Single 'self' type returns unprefixed (config_builder prefixes)."""
        a = {
            "attention_type": "self",
            "target_modules_str": "q_proj v_proj",
        }
        result = self._resolve(a)
        self.assertEqual(result, ["q_proj", "v_proj"])

    def test_cross_returns_unprefixed(self):
        """Single 'cross' type returns unprefixed (config_builder prefixes)."""
        a = {
            "attention_type": "cross",
            "target_modules_str": "q_proj k_proj",
        }
        result = self._resolve(a)
        self.assertEqual(result, ["q_proj", "k_proj"])

    def test_self_with_default(self):
        """'self' with no target_modules_str uses default."""
        a = {"attention_type": "self"}
        result = self._resolve(a)
        self.assertEqual(result, ["q_proj", "k_proj", "v_proj", "o_proj"])

    # -- Missing attention_type ---

    def test_missing_attention_type_defaults_to_both(self):
        """No attention_type key defaults to 'both', no split keys -> passthrough."""
        a = {"target_modules_str": "q_proj"}
        result = self._resolve(a)
        self.assertEqual(result, ["q_proj"])


class TestBuildTrainNamespaceProjections(unittest.TestCase):
    """Verify that build_train_namespace passes resolved projections through."""

    def _minimal_answers(self, **overrides) -> dict:
        """Return minimal wizard answers dict with required keys."""
        base = {
            "checkpoint_dir": "/tmp/ckpt",
            "model_variant": "turbo",
            "dataset_dir": "/tmp/ds",
            "output_dir": "/tmp/out",
        }
        base.update(overrides)
        return base

    def test_split_projections_in_namespace(self):
        """Split projections land in namespace.target_modules pre-resolved."""
        from acestep.training_v2.ui.flows_common import build_train_namespace

        a = self._minimal_answers(
            attention_type="both",
            self_target_modules_str="q_proj",
            cross_target_modules_str="v_proj o_proj",
        )
        ns = build_train_namespace(a)
        self.assertEqual(ns.target_modules, [
            "self_attn.q_proj",
            "cross_attn.v_proj", "cross_attn.o_proj",
        ])

    def test_single_string_in_namespace(self):
        """Single target_modules_str (non-both) lands in namespace."""
        from acestep.training_v2.ui.flows_common import build_train_namespace

        a = self._minimal_answers(
            attention_type="self",
            target_modules_str="q_proj v_proj",
        )
        ns = build_train_namespace(a)
        self.assertEqual(ns.target_modules, ["q_proj", "v_proj"])


class TestResolveTargetModulesCLI(unittest.TestCase):
    """Unit tests for resolve_target_modules (CLI path in validation.py)."""

    def _resolve(self, target_modules, attention_type, **kwargs):
        from acestep.training_v2.cli.validation import resolve_target_modules
        return resolve_target_modules(target_modules, attention_type, **kwargs)

    # -- "both" with split kwargs ---

    def test_both_split_different(self):
        """Split self/cross with different projections."""
        result = self._resolve(
            ["q_proj", "k_proj", "v_proj", "o_proj"], "both",
            self_target_modules=["q_proj", "v_proj"],
            cross_target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        )
        self.assertEqual(result, [
            "self_attn.q_proj", "self_attn.v_proj",
            "cross_attn.q_proj", "cross_attn.k_proj",
            "cross_attn.v_proj", "cross_attn.o_proj",
        ])

    def test_both_split_only_self(self):
        """Only --self-target-modules provided; cross falls back to default."""
        result = self._resolve(
            ["q_proj", "k_proj", "v_proj", "o_proj"], "both",
            self_target_modules=["q_proj"],
        )
        self.assertEqual(result, [
            "self_attn.q_proj",
            "cross_attn.q_proj", "cross_attn.k_proj",
            "cross_attn.v_proj", "cross_attn.o_proj",
        ])

    def test_both_split_only_cross(self):
        """Only --cross-target-modules provided; self falls back to default."""
        result = self._resolve(
            ["q_proj", "k_proj", "v_proj", "o_proj"], "both",
            cross_target_modules=["v_proj"],
        )
        self.assertEqual(result, [
            "self_attn.q_proj", "self_attn.k_proj",
            "self_attn.v_proj", "self_attn.o_proj",
            "cross_attn.v_proj",
        ])

    def test_both_split_preserves_qualified(self):
        """Already-qualified modules pass through unchanged."""
        result = self._resolve(
            ["q_proj"], "both",
            self_target_modules=["q_proj", "self_attn.custom"],
            cross_target_modules=["v_proj"],
        )
        self.assertEqual(result, [
            "self_attn.q_proj", "self_attn.custom",
            "cross_attn.v_proj",
        ])

    # -- "both" without split (backward compat) ---

    def test_both_no_split_passthrough(self):
        """No split kwargs means passthrough (existing behavior)."""
        result = self._resolve(["q_proj", "v_proj"], "both")
        self.assertEqual(result, ["q_proj", "v_proj"])

    # -- "self" and "cross" ---

    def test_self_prefixes(self):
        """'self' adds self_attn. prefix."""
        result = self._resolve(["q_proj", "v_proj"], "self")
        self.assertEqual(result, ["self_attn.q_proj", "self_attn.v_proj"])

    def test_cross_prefixes(self):
        """'cross' adds cross_attn. prefix."""
        result = self._resolve(["q_proj", "v_proj"], "cross")
        self.assertEqual(result, ["cross_attn.q_proj", "cross_attn.v_proj"])

    def test_self_ignores_split_kwargs(self):
        """Split kwargs are ignored when attention_type is not 'both'."""
        result = self._resolve(
            ["q_proj"], "self",
            self_target_modules=["v_proj"],
            cross_target_modules=["o_proj"],
        )
        self.assertEqual(result, ["self_attn.q_proj"])

    def test_self_preserves_qualified(self):
        """Already-qualified modules pass through for single type."""
        result = self._resolve(["q_proj", "self_attn.custom"], "self")
        self.assertEqual(result, ["self_attn.q_proj", "self_attn.custom"])


class TestMLPTargetingWizard(unittest.TestCase):
    """Tests for MLP targeting in the wizard flow."""

    def _resolve(self, answers: dict) -> list:
        from acestep.training_v2.ui.flows_common import _resolve_wizard_projections
        return _resolve_wizard_projections(answers)

    def test_mlp_appended_when_enabled(self):
        """MLP modules appended when target_mlp is True."""
        a = {
            "attention_type": "self",
            "target_modules_str": "q_proj v_proj",
            "target_mlp": True,
        }
        result = self._resolve(a)
        self.assertEqual(result, [
            "q_proj", "v_proj",
            "gate_proj", "up_proj", "down_proj",
        ])

    def test_mlp_not_appended_by_default(self):
        """MLP modules NOT appended when target_mlp is absent/False."""
        a = {
            "attention_type": "self",
            "target_modules_str": "q_proj v_proj",
        }
        result = self._resolve(a)
        self.assertEqual(result, ["q_proj", "v_proj"])

    def test_mlp_with_both_split(self):
        """MLP modules appended to split self/cross projections."""
        a = {
            "attention_type": "both",
            "self_target_modules_str": "q_proj",
            "cross_target_modules_str": "v_proj",
            "target_mlp": True,
        }
        result = self._resolve(a)
        self.assertEqual(result, [
            "self_attn.q_proj",
            "cross_attn.v_proj",
            "gate_proj", "up_proj", "down_proj",
        ])

    def test_mlp_deduplication(self):
        """MLP modules not duplicated if already present."""
        a = {
            "attention_type": "self",
            "target_modules_str": "q_proj gate_proj",
            "target_mlp": True,
        }
        result = self._resolve(a)
        # gate_proj should not appear twice
        self.assertEqual(result, [
            "q_proj", "gate_proj",
            "up_proj", "down_proj",
        ])


class TestMLPTargetingCLI(unittest.TestCase):
    """Tests for MLP targeting via CLI resolve_target_modules."""

    def _resolve(self, target_modules, attention_type, **kwargs):
        from acestep.training_v2.cli.validation import resolve_target_modules
        return resolve_target_modules(target_modules, attention_type, **kwargs)

    def test_mlp_appended_when_enabled(self):
        """MLP modules appended when target_mlp=True."""
        result = self._resolve(["q_proj", "v_proj"], "both", target_mlp=True)
        self.assertEqual(result, [
            "q_proj", "v_proj",
            "gate_proj", "up_proj", "down_proj",
        ])

    def test_mlp_not_appended_by_default(self):
        """MLP modules NOT appended when target_mlp is default (False)."""
        result = self._resolve(["q_proj", "v_proj"], "both")
        self.assertEqual(result, ["q_proj", "v_proj"])

    def test_mlp_with_self_attention(self):
        """MLP + self-only attention combines correctly."""
        result = self._resolve(["q_proj"], "self", target_mlp=True)
        self.assertEqual(result, [
            "self_attn.q_proj",
            "gate_proj", "up_proj", "down_proj",
        ])

    def test_mlp_with_split_projections(self):
        """MLP + split self/cross projections."""
        result = self._resolve(
            ["q_proj"], "both",
            self_target_modules=["q_proj"],
            cross_target_modules=["v_proj"],
            target_mlp=True,
        )
        self.assertEqual(result, [
            "self_attn.q_proj",
            "cross_attn.v_proj",
            "gate_proj", "up_proj", "down_proj",
        ])

    def test_mlp_deduplication(self):
        """MLP modules not duplicated if already in target_modules."""
        result = self._resolve(
            ["q_proj", "gate_proj"], "both", target_mlp=True,
        )
        self.assertEqual(result, [
            "q_proj", "gate_proj",
            "up_proj", "down_proj",
        ])


class TestNamespaceMLPTargeting(unittest.TestCase):
    """Verify that build_train_namespace passes target_mlp through."""

    def _minimal_answers(self, **overrides) -> dict:
        base = {
            "checkpoint_dir": "/tmp/ckpt",
            "model_variant": "turbo",
            "dataset_dir": "/tmp/ds",
            "output_dir": "/tmp/out",
        }
        base.update(overrides)
        return base

    def test_target_mlp_in_namespace(self):
        """target_mlp flag lands in namespace."""
        from acestep.training_v2.ui.flows_common import build_train_namespace

        a = self._minimal_answers(target_mlp=True)
        ns = build_train_namespace(a)
        self.assertTrue(ns.target_mlp)

    def test_target_mlp_default_false(self):
        """target_mlp defaults to False in namespace."""
        from acestep.training_v2.ui.flows_common import build_train_namespace

        a = self._minimal_answers()
        ns = build_train_namespace(a)
        self.assertFalse(ns.target_mlp)

    def test_mlp_modules_in_target_modules(self):
        """When target_mlp=True, MLP modules are in target_modules."""
        from acestep.training_v2.ui.flows_common import build_train_namespace

        a = self._minimal_answers(
            attention_type="self",
            target_modules_str="q_proj",
            target_mlp=True,
        )
        ns = build_train_namespace(a)
        self.assertIn("gate_proj", ns.target_modules)
        self.assertIn("up_proj", ns.target_modules)
        self.assertIn("down_proj", ns.target_modules)


if __name__ == "__main__":
    unittest.main()
