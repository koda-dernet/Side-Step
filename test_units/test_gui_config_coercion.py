"""Tests for GUI → config type coercion.

The GUI sends all form values as strings. The _coerce_type helper in
config_builder.py must cast them to match argparse default types before
they reach dataclass constructors.
"""

from __future__ import annotations

import argparse
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from sidestep_engine.cli.config_builder import _coerce_type


class TestCoerceType(unittest.TestCase):
    """Unit tests for the _coerce_type helper."""

    def test_str_to_int(self):
        self.assertEqual(_coerce_type("64", 0), 64)

    def test_str_to_int_float_format(self):
        self.assertEqual(_coerce_type("64.0", 0), 64)

    def test_str_to_float(self):
        self.assertAlmostEqual(_coerce_type("3e-4", 0.0), 3e-4)

    def test_str_to_float_plain(self):
        self.assertAlmostEqual(_coerce_type("0.1", 0.0), 0.1)

    def test_str_to_bool_true(self):
        self.assertTrue(_coerce_type("true", False))

    def test_str_to_bool_false(self):
        self.assertFalse(_coerce_type("false", True))

    def test_str_to_bool_one(self):
        self.assertTrue(_coerce_type("1", False))

    def test_already_correct_type(self):
        self.assertEqual(_coerce_type(64, 0), 64)
        self.assertEqual(_coerce_type(3e-4, 0.0), 3e-4)
        self.assertTrue(_coerce_type(True, False))

    def test_none_reference_numeric_coerced(self):
        self.assertEqual(_coerce_type("0", None), 0)
        self.assertEqual(_coerce_type("60", None), 60)
        self.assertAlmostEqual(_coerce_type("3.5", None), 3.5)

    def test_none_reference_path_passthrough(self):
        self.assertEqual(_coerce_type("/some/path", None), "/some/path")
        self.assertEqual(_coerce_type("", None), "")

    def test_none_reference_name_passthrough(self):
        self.assertEqual(_coerce_type("turbo", None), "turbo")

    def test_none_value_passthrough(self):
        self.assertIsNone(_coerce_type(None, 0))

    def test_unconvertible_passthrough(self):
        self.assertEqual(_coerce_type("not_a_number", 0), "not_a_number")

    def test_str_to_str_passthrough(self):
        self.assertEqual(_coerce_type("hello", "default"), "hello")


class TestApplyConfigFileCoercion(unittest.TestCase):
    """Integration test: string values in a JSON config file get coerced."""

    def test_string_rank_coerced_to_int(self):
        from sidestep_engine.cli.config_builder import (
            _apply_config_file,
            _populate_defaults_cache,
        )
        _populate_defaults_cache()

        # Simulate a GUI-produced config with string values
        gui_config = {
            "rank": "64",
            "alpha": "128",
            "dropout": "0.1",
            "lr": "3e-4",
            "epochs": "1000",
            "batch_size": "1",
            "save_every": "50",
        }

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as f:
            json.dump(gui_config, f)
            config_path = f.name

        try:
            ns = argparse.Namespace(config=config_path)
            # Set all attrs to None so _apply_config_file will apply values
            for dest in [
                "rank", "alpha", "dropout", "learning_rate",
                "epochs", "batch_size", "save_every",
            ]:
                setattr(ns, dest, None)

            _apply_config_file(ns)

            self.assertIsInstance(ns.rank, int)
            self.assertEqual(ns.rank, 64)
            self.assertIsInstance(ns.alpha, int)
            self.assertEqual(ns.alpha, 128)
            self.assertIsInstance(ns.learning_rate, float)
            self.assertAlmostEqual(ns.learning_rate, 3e-4)
            self.assertIsInstance(ns.epochs, int)
            self.assertEqual(ns.epochs, 1000)
        finally:
            Path(config_path).unlink(missing_ok=True)


if __name__ == "__main__":
    unittest.main()
