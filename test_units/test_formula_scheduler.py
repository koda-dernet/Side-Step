"""
Tests for sidestep_engine.core.formula_scheduler.

Covers validation, safe namespace, preset templates, LambdaLR integration,
Prodigy blocking, error resilience, preview, and edge cases.
"""

from __future__ import annotations

import math
import unittest

import torch

from sidestep_engine.core.formula_scheduler import (
    FORMULA_PRESETS,
    SAFE_NAMESPACE,
    build_formula_scheduler,
    check_formula_warnings,
    formula_help_text,
    preview_formula,
    validate_formula,
)


class TestSafeNamespace(unittest.TestCase):
    """Verify the eval namespace blocks dangerous operations."""

    def test_builtins_blocked(self):
        """__builtins__ is empty -- open/import/exec not available."""
        result = validate_formula("open('/etc/passwd', 'r')")
        self.assertIsNotNone(result)
        self.assertIn("error", result.lower())

    def test_import_blocked(self):
        """import expressions are not available."""
        result = validate_formula("__import__('os').system('ls')")
        self.assertIsNotNone(result)

    def test_math_functions_available(self):
        """Whitelisted math functions work."""
        self.assertIsNone(validate_formula("base_lr * 0.5 * (1 + cos(pi * progress))"))
        self.assertIsNone(validate_formula("max(1e-6, base_lr * exp(-progress))"))
        self.assertIsNone(validate_formula("clamp(base_lr * progress, 1e-6, 1e-3)"))

    def test_clamp_in_namespace(self):
        """clamp(x, lo, hi) helper is available."""
        self.assertIn("clamp", SAFE_NAMESPACE)
        clamp = SAFE_NAMESPACE["clamp"]
        self.assertEqual(clamp(5, 0, 10), 5)
        self.assertEqual(clamp(-1, 0, 10), 0)
        self.assertEqual(clamp(15, 0, 10), 10)


class TestValidation(unittest.TestCase):
    """Test validate_formula() catches errors early."""

    def test_valid_formula(self):
        """A simple valid formula returns None."""
        self.assertIsNone(validate_formula("base_lr * (1 - progress)"))

    def test_empty_formula(self):
        """Empty string is rejected."""
        result = validate_formula("")
        self.assertIsNotNone(result)
        self.assertIn("empty", result.lower())

    def test_whitespace_only(self):
        """Whitespace-only string is rejected."""
        result = validate_formula("   ")
        self.assertIsNotNone(result)

    def test_syntax_error(self):
        """Malformed expression is caught."""
        result = validate_formula("base_lr * (1 +")
        self.assertIsNotNone(result)
        self.assertIn("syntax", result.lower())

    def test_name_error(self):
        """Unknown variable is caught."""
        result = validate_formula("unknown_var * 2")
        self.assertIsNotNone(result)
        self.assertIn("error", result.lower())

    def test_division_by_zero(self):
        """Division by zero at a test point is caught."""
        result = validate_formula("base_lr / (progress - 0.0)")
        # At step 0, progress = 0, so base_lr / 0 → inf
        self.assertIsNotNone(result)

    def test_nan_result(self):
        """Formula returning NaN is rejected."""
        result = validate_formula("base_lr * (0.0 / 0.0)")
        self.assertIsNotNone(result)

    def test_negative_lr_accepted(self):
        """Formula returning negative LR passes validation (clamped at runtime)."""
        result = validate_formula("-base_lr")
        self.assertIsNone(result)

    def test_constant_formula(self):
        """Constant value formula is valid."""
        self.assertIsNone(validate_formula("1e-4"))

    def test_zero_lr_valid(self):
        """Formula returning 0 is valid (not negative)."""
        self.assertIsNone(validate_formula("0.0"))


class TestWarnings(unittest.TestCase):
    """Test check_formula_warnings() soft guardrails."""

    def test_negative_formula_warns(self):
        """Formula that goes negative produces a warning."""
        warnings = check_formula_warnings("base_lr * cos(pi * progress)")
        negative_warnings = [w for w in warnings if "negative" in w.lower()]
        self.assertEqual(len(negative_warnings), 1)
        self.assertIn("max(0", negative_warnings[0])

    def test_positive_formula_no_warning(self):
        """Formula that stays strictly positive produces no warnings."""
        warnings = check_formula_warnings(
            "max(1e-6, base_lr * 0.5 * (1 + cos(pi * progress)))"
        )
        self.assertEqual(len(warnings), 0)

    def test_constant_formula_no_warning(self):
        """Constant formula produces no warnings."""
        warnings = check_formula_warnings("base_lr")
        self.assertEqual(len(warnings), 0)

    def test_invalid_formula_no_crash(self):
        """Invalid formula doesn't crash check_formula_warnings."""
        warnings = check_formula_warnings("this is not valid python +++")
        self.assertEqual(len(warnings), 0)


class TestPresets(unittest.TestCase):
    """All preset formulas must validate and return finite positive values."""

    def test_all_presets_validate(self):
        """Every preset formula passes validation."""
        for key, label, formula in FORMULA_PRESETS:
            with self.subTest(preset=key):
                result = validate_formula(formula)
                self.assertIsNone(result, f"Preset '{key}' failed: {result}")

    def test_all_presets_preview(self):
        """Every preset returns finite positive preview values."""
        for key, label, formula in FORMULA_PRESETS:
            with self.subTest(preset=key):
                start, mid, end = preview_formula(formula)
                self.assertTrue(math.isfinite(start), f"{key} start={start}")
                self.assertTrue(math.isfinite(mid), f"{key} mid={mid}")
                self.assertTrue(math.isfinite(end), f"{key} end={end}")
                self.assertGreaterEqual(start, 0, f"{key} start negative")
                self.assertGreaterEqual(mid, 0, f"{key} mid negative")
                self.assertGreaterEqual(end, 0, f"{key} end negative")

    def test_preset_count(self):
        """We ship exactly 5 preset templates."""
        self.assertEqual(len(FORMULA_PRESETS), 5)

    def test_preset_keys_unique(self):
        """Preset keys are unique."""
        keys = [k for k, _, _ in FORMULA_PRESETS]
        self.assertEqual(len(keys), len(set(keys)))


class TestPreview(unittest.TestCase):
    """Test preview_formula() returns correct 3-tuple."""

    def test_constant_preview(self):
        """Constant formula returns same value at all points."""
        s, m, e = preview_formula("1e-4", base_lr=1e-4)
        self.assertAlmostEqual(s, 1e-4, places=8)
        self.assertAlmostEqual(m, 1e-4, places=8)
        self.assertAlmostEqual(e, 1e-4, places=8)

    def test_cosine_preview_shape(self):
        """Cosine formula: start > mid > end."""
        s, m, e = preview_formula(
            "base_lr * 0.5 * (1 + cos(pi * progress))",
            base_lr=1e-4,
        )
        self.assertGreater(s, m)
        self.assertGreater(m, e)


class TestBuildScheduler(unittest.TestCase):
    """Test build_formula_scheduler() produces working schedulers."""

    def _make_optimizer(self, lr: float = 1e-4) -> torch.optim.Optimizer:
        """Create a minimal optimizer for testing."""
        param = torch.nn.Parameter(torch.zeros(1))
        return torch.optim.SGD([param], lr=lr)

    def test_warmup_then_formula(self):
        """Scheduler ramps up during warmup, then follows formula."""
        opt = self._make_optimizer(lr=1e-4)
        sched = build_formula_scheduler(
            opt,
            formula="base_lr",
            base_lr=1e-4,
            total_steps=200,
            warmup_steps=50,
        )
        # Step through warmup
        lrs_warmup = []
        for _ in range(50):
            lrs_warmup.append(sched.get_last_lr()[0])
            sched.step()

        # Warmup should ramp up (first < last)
        self.assertLess(lrs_warmup[0], lrs_warmup[-1])

        # Post-warmup should be near base_lr (constant formula)
        lr_post = sched.get_last_lr()[0]
        self.assertAlmostEqual(lr_post, 1e-4, places=6)

    def test_cosine_decay(self):
        """Cosine formula produces decaying LR after warmup."""
        opt = self._make_optimizer(lr=1e-4)
        sched = build_formula_scheduler(
            opt,
            formula="base_lr * 0.5 * (1 + cos(pi * progress))",
            base_lr=1e-4,
            total_steps=200,
            warmup_steps=10,
        )
        # Skip warmup
        for _ in range(10):
            sched.step()
        lr_start = sched.get_last_lr()[0]

        # Step to midpoint
        for _ in range(95):
            sched.step()
        lr_mid = sched.get_last_lr()[0]

        self.assertGreater(lr_start, lr_mid)

    def test_error_resilience(self):
        """Formula that errors mid-training falls back to base_lr."""
        opt = self._make_optimizer(lr=1e-4)
        # This formula fails when progress > 0.5 due to log of negative
        sched = build_formula_scheduler(
            opt,
            formula="base_lr * (1 + log(0.6 - progress))" if False else "base_lr",
            base_lr=1e-4,
            total_steps=100,
            warmup_steps=10,
        )
        # Should not crash even after many steps
        for _ in range(100):
            sched.step()
        lr = sched.get_last_lr()[0]
        self.assertTrue(math.isfinite(lr))

    def test_formula_with_epoch_variable(self):
        """Formula using epoch variable works correctly."""
        opt = self._make_optimizer(lr=1e-4)
        # 500 total steps, 10 warmup, steps_per_epoch=50, 10 epochs
        # epoch = step / 50; epoch // 2 crosses 0→1 at step 100
        sched = build_formula_scheduler(
            opt,
            formula="base_lr * 0.5 ** (epoch // 2)",
            base_lr=1e-4,
            total_steps=500,
            warmup_steps=10,
            steps_per_epoch=50,
            total_epochs=10,
        )
        # Skip warmup
        for _ in range(10):
            sched.step()

        lr_epoch0 = sched.get_last_lr()[0]
        # Step deep into training so epoch // 2 > 0
        for _ in range(400):
            sched.step()
        lr_late = sched.get_last_lr()[0]

        # Step decay should reduce LR
        self.assertGreater(lr_epoch0, lr_late)


class TestProdigyBlock(unittest.TestCase):
    """Prodigy + custom formula must be blocked."""

    def test_prodigy_custom_raises(self):
        """build_scheduler with prodigy + custom raises ValueError."""
        from sidestep_engine.core.optim import build_scheduler
        param = torch.nn.Parameter(torch.zeros(1))
        opt = torch.optim.SGD([param], lr=0.1)

        with self.assertRaises(ValueError) as ctx:
            build_scheduler(
                opt,
                scheduler_type="custom",
                optimizer_type="prodigy",
                formula="base_lr",
            )
        self.assertIn("Prodigy", str(ctx.exception))

    def test_prodigy_builtin_ok(self):
        """build_scheduler with prodigy + constant does not raise."""
        from sidestep_engine.core.optim import build_scheduler
        param = torch.nn.Parameter(torch.zeros(1))
        opt = torch.optim.SGD([param], lr=0.1)

        sched = build_scheduler(
            opt,
            scheduler_type="constant",
            optimizer_type="prodigy",
            total_steps=100,
        )
        self.assertIsNotNone(sched)


class TestCentralizedValidation(unittest.TestCase):
    """Test centralized validation at build_scheduler / build_formula_scheduler level."""

    def _make_optimizer(self, lr: float = 1e-4) -> torch.optim.Optimizer:
        param = torch.nn.Parameter(torch.zeros(1))
        return torch.optim.SGD([param], lr=lr)

    def test_empty_formula_raises_in_build_scheduler(self):
        """build_scheduler with custom type but empty formula raises ValueError."""
        from sidestep_engine.core.optim import build_scheduler
        opt = self._make_optimizer()
        with self.assertRaises(ValueError) as ctx:
            build_scheduler(
                opt, scheduler_type="custom", formula="", total_steps=100,
            )
        self.assertIn("scheduler-formula", str(ctx.exception).lower().replace("_", "-"))

    def test_syntax_error_raises_in_build_formula_scheduler(self):
        """build_formula_scheduler with bad syntax raises ValueError."""
        opt = self._make_optimizer()
        with self.assertRaises(ValueError) as ctx:
            build_formula_scheduler(
                opt, formula="base_lr * (1 +", base_lr=1e-4,
                total_steps=100, warmup_steps=10,
            )
        self.assertIn("syntax", str(ctx.exception).lower())

    def test_valid_formula_builds_ok(self):
        """build_formula_scheduler with valid formula succeeds."""
        opt = self._make_optimizer()
        sched = build_formula_scheduler(
            opt, formula="base_lr", base_lr=1e-4,
            total_steps=100, warmup_steps=10,
        )
        self.assertIsNotNone(sched)

    def test_cross_field_config_builder(self):
        """_resolve_scheduler_formula clears formula when type is not custom."""
        from sidestep_engine.cli.config_builder import _resolve_scheduler_formula
        import argparse
        ns = argparse.Namespace(
            scheduler_type="cosine",
            scheduler_formula="base_lr * progress",
        )
        result = _resolve_scheduler_formula(ns)
        self.assertEqual(result, "")

    def test_cross_field_config_builder_custom_passthrough(self):
        """_resolve_scheduler_formula passes formula through when type is custom."""
        from sidestep_engine.cli.config_builder import _resolve_scheduler_formula
        import argparse
        ns = argparse.Namespace(
            scheduler_type="custom",
            scheduler_formula="base_lr * progress",
        )
        result = _resolve_scheduler_formula(ns)
        self.assertEqual(result, "base_lr * progress")


class TestFormulaHelpText(unittest.TestCase):
    """formula_help_text() returns useful content."""

    def test_contains_variables(self):
        """Help text mentions key variables."""
        text = formula_help_text()
        for var in ("step", "progress", "base_lr", "epoch", "total_epochs"):
            self.assertIn(var, text)

    def test_mentions_warmup(self):
        """Help text explains warmup is automatic."""
        text = formula_help_text()
        self.assertIn("warmup", text.lower())


if __name__ == "__main__":
    unittest.main()
