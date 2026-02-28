"""Tests for flow-level preset subset loading."""

from __future__ import annotations

import unittest
from unittest.mock import patch


class TestFlowPresetSubset(unittest.TestCase):
    def test_applies_only_allowed_fields(self) -> None:
        from sidestep_engine.ui.flows.common import offer_load_preset_subset

        answers = {"rank": 32}
        with (
            patch(
                "sidestep_engine.ui.presets.list_presets",
                return_value=[{"name": "quick_test", "builtin": False, "description": ""}],
            ),
            patch(
                "sidestep_engine.ui.presets.load_preset",
                return_value={
                    "rank": 64,
                    "alpha": 128,
                    "optimizer_type": "adamw",
                },
            ),
            patch("sidestep_engine.ui.flows.common.menu", return_value="quick_test"),
        ):
            offer_load_preset_subset(
                answers,
                allowed_fields={"rank", "alpha"},
            )

        self.assertEqual(answers["rank"], 64)
        self.assertEqual(answers["alpha"], 128)
        self.assertNotIn("optimizer_type", answers)

    def test_preserve_fields_not_overwritten(self) -> None:
        from sidestep_engine.ui.flows.common import offer_load_preset_subset

        answers = {"rank": 32}
        with (
            patch(
                "sidestep_engine.ui.presets.list_presets",
                return_value=[{"name": "quick_test", "builtin": False, "description": ""}],
            ),
            patch(
                "sidestep_engine.ui.presets.load_preset",
                return_value={"rank": 64, "alpha": 128},
            ),
            patch("sidestep_engine.ui.flows.common.menu", return_value="quick_test"),
        ):
            offer_load_preset_subset(
                answers,
                allowed_fields={"rank", "alpha"},
                preserve_fields={"rank"},
            )

        self.assertEqual(answers["rank"], 32)
        self.assertEqual(answers["alpha"], 128)


if __name__ == "__main__":
    unittest.main()
