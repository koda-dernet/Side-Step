"""Tests for PP++ compatibility constants and gating logic.

Includes regression tests for the _check_fisher_map bug where
incompatible adapters would see a false 'PP++ map detected' message.
"""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from sidestep_engine.ui.flows.common import (
    PP_COMPATIBLE_ADAPTERS,
    is_pp_compatible,
)


class TestPPCompatibility(unittest.TestCase):
    """Test PP_COMPATIBLE_ADAPTERS and is_pp_compatible helper."""

    def test_lora_compatible(self):
        self.assertTrue(is_pp_compatible("lora"))

    def test_dora_compatible(self):
        self.assertTrue(is_pp_compatible("dora"))

    def test_lokr_not_compatible(self):
        self.assertFalse(is_pp_compatible("lokr"))

    def test_loha_not_compatible(self):
        self.assertFalse(is_pp_compatible("loha"))

    def test_oft_not_compatible(self):
        self.assertFalse(is_pp_compatible("oft"))

    def test_unknown_not_compatible(self):
        self.assertFalse(is_pp_compatible("unknown"))

    def test_frozenset_membership(self):
        self.assertIn("lora", PP_COMPATIBLE_ADAPTERS)
        self.assertIn("dora", PP_COMPATIBLE_ADAPTERS)
        self.assertNotIn("lokr", PP_COMPATIBLE_ADAPTERS)
        self.assertNotIn("loha", PP_COMPATIBLE_ADAPTERS)
        self.assertNotIn("oft", PP_COMPATIBLE_ADAPTERS)

    def test_frozenset_is_immutable(self):
        with self.assertRaises(AttributeError):
            PP_COMPATIBLE_ADAPTERS.add("lokr")  # type: ignore[attr-defined]


class TestCheckFisherMapGating(unittest.TestCase):
    """Regression: _check_fisher_map must NOT set _fisher_map_detected
    for PP++-incompatible adapters even when fisher_map.json exists."""

    def _make_dataset_with_fisher(self) -> str:
        """Create a temp dir with a fisher_map.json and a dummy .pt file."""
        tmp = tempfile.mkdtemp()
        fisher = {
            "rank_pattern": {"layer.0": 64, "layer.1": 32},
            "rank_budget": {"min": 32, "max": 64},
        }
        (Path(tmp) / "fisher_map.json").write_text(json.dumps(fisher))
        (Path(tmp) / "dummy.pt").write_bytes(b"\x00")
        return tmp

    @patch("sidestep_engine.ui.flows.train.print_message")
    def test_lora_with_fisher_sets_detected(self, mock_msg):
        """LoRA + fisher_map → _fisher_map_detected = True."""
        from sidestep_engine.ui.flows.train import _check_fisher_map
        tmp = self._make_dataset_with_fisher()
        answers = {"dataset_dir": tmp}
        _check_fisher_map(answers, "lora")
        self.assertTrue(answers["_fisher_map_detected"])

    @patch("sidestep_engine.ui.flows.train.print_message")
    def test_dora_with_fisher_sets_detected(self, mock_msg):
        """DoRA + fisher_map → _fisher_map_detected = True."""
        from sidestep_engine.ui.flows.train import _check_fisher_map
        tmp = self._make_dataset_with_fisher()
        answers = {"dataset_dir": tmp}
        _check_fisher_map(answers, "dora")
        self.assertTrue(answers["_fisher_map_detected"])

    @patch("sidestep_engine.ui.flows.train.print_message")
    def test_loha_with_fisher_NOT_detected(self, mock_msg):
        """LoHA + fisher_map → _fisher_map_detected must stay False."""
        from sidestep_engine.ui.flows.train import _check_fisher_map
        tmp = self._make_dataset_with_fisher()
        answers = {"dataset_dir": tmp}
        _check_fisher_map(answers, "loha")
        self.assertFalse(answers["_fisher_map_detected"])

    @patch("sidestep_engine.ui.flows.train.print_message")
    def test_oft_with_fisher_NOT_detected(self, mock_msg):
        """OFT + fisher_map → _fisher_map_detected must stay False."""
        from sidestep_engine.ui.flows.train import _check_fisher_map
        tmp = self._make_dataset_with_fisher()
        answers = {"dataset_dir": tmp}
        _check_fisher_map(answers, "oft")
        self.assertFalse(answers["_fisher_map_detected"])

    @patch("sidestep_engine.ui.flows.train.print_message")
    def test_lokr_with_fisher_NOT_detected(self, mock_msg):
        """LoKR + fisher_map → _fisher_map_detected must stay False."""
        from sidestep_engine.ui.flows.train import _check_fisher_map
        tmp = self._make_dataset_with_fisher()
        answers = {"dataset_dir": tmp}
        _check_fisher_map(answers, "lokr")
        self.assertFalse(answers["_fisher_map_detected"])

    @patch("sidestep_engine.ui.flows.train.print_message")
    def test_loha_with_fisher_shows_warning(self, mock_msg):
        """LoHA + fisher_map → user sees incompatibility warning."""
        from sidestep_engine.ui.flows.train import _check_fisher_map
        tmp = self._make_dataset_with_fisher()
        answers = {"dataset_dir": tmp}
        _check_fisher_map(answers, "loha")
        # Check that a warning about LOHA was printed
        warn_calls = [
            c for c in mock_msg.call_args_list
            if c.kwargs.get("kind") == "warn"
        ]
        self.assertTrue(
            len(warn_calls) > 0,
            "Expected a 'warn' message about incompatibility",
        )
        self.assertIn("LOHA", warn_calls[0].args[0])


if __name__ == "__main__":
    unittest.main()
