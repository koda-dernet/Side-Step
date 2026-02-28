"""Tests for PP++ chain ordering fix.

Verifies that _offer_pp_train_chain is only called AFTER the fisher
analysis has completed (i.e. after yield ns dispatches the namespace),
not before.  Also verifies the chain correctly checks for fisher_map.json
on disk before offering training.
"""

from __future__ import annotations

import argparse
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch


class TestOfferPPTrainChainRequiresFisherMap(unittest.TestCase):
    """_offer_pp_train_chain should return None when no fisher_map.json exists."""

    def test_returns_none_when_no_fisher_map(self) -> None:
        """Chain offer should be skipped when fisher_map.json is absent."""
        from sidestep_engine.ui.wizard import _offer_pp_train_chain

        with tempfile.TemporaryDirectory() as td:
            ns = argparse.Namespace(dataset_dir=td)
            result = _offer_pp_train_chain(ns, {})
            self.assertIsNone(result)

    def test_returns_none_when_no_dataset_dir(self) -> None:
        """Chain offer should be skipped when dataset_dir is missing."""
        from sidestep_engine.ui.wizard import _offer_pp_train_chain

        ns = argparse.Namespace(dataset_dir=None)
        result = _offer_pp_train_chain(ns, {})
        self.assertIsNone(result)

    def test_offers_chain_when_fisher_map_exists(self) -> None:
        """Chain offer should proceed when fisher_map.json is on disk."""
        from sidestep_engine.ui.wizard import _offer_pp_train_chain

        with tempfile.TemporaryDirectory() as td:
            fisher_path = Path(td) / "fisher_map.json"
            fisher_path.write_text(json.dumps({
                "rank_pattern": {"mod.a": 64},
                "rank_budget": {"min": 16, "max": 128},
            }))

            ns = argparse.Namespace(dataset_dir=td)
            mock_ns = argparse.Namespace(subcommand="train")

            with (
                patch("sidestep_engine.ui.prompt_helpers.ask_bool", return_value=True),
                patch("sidestep_engine.ui.wizard._pick_adapter_type", return_value="lora"),
                patch("sidestep_engine.ui.wizard.wizard_train", return_value=mock_ns),
                patch("sidestep_engine.ui.prompt_helpers.print_message"),
                patch("sidestep_engine.ui.prompt_helpers.print_rich"),
            ):
                result = _offer_pp_train_chain(ns, {})
                self.assertIs(result, mock_ns)

    def test_returns_none_when_user_declines(self) -> None:
        """Chain offer should return None when user says no to training."""
        from sidestep_engine.ui.wizard import _offer_pp_train_chain

        with tempfile.TemporaryDirectory() as td:
            fisher_path = Path(td) / "fisher_map.json"
            fisher_path.write_text(json.dumps({"rank_pattern": {}}))

            ns = argparse.Namespace(dataset_dir=td)

            with (
                patch("sidestep_engine.ui.prompt_helpers.ask_bool", return_value=False),
                patch("sidestep_engine.ui.prompt_helpers.print_message"),
                patch("sidestep_engine.ui.prompt_helpers.print_rich"),
            ):
                result = _offer_pp_train_chain(ns, {})
                self.assertIsNone(result)


class TestSessionLoopPPChainOrdering(unittest.TestCase):
    """The session loop should dispatch fisher BEFORE offering the chain."""

    def test_pp_chain_after_dispatch(self) -> None:
        """Verify fisher ns is yielded before chain offer runs."""
        from sidestep_engine.ui.wizard import run_wizard_session

        call_order: list[str] = []

        fisher_ns = argparse.Namespace(
            subcommand="analyze",
            dataset_dir="/fake/dir",
            preprocess=False,
            tensor_output=None,
        )

        with (
            patch("sidestep_engine.ui.banner.show_banner"),
            patch("sidestep_engine.ui.wizard._ensure_first_run_done"),
            patch("sidestep_engine.ui.wizard._main_menu", side_effect=[fisher_ns, None]),
            patch(
                "sidestep_engine.ui.wizard._offer_pp_train_chain",
                side_effect=lambda ns, sd: (call_order.append("chain_offer"), None)[1],
            ) as mock_chain,
            patch(
                "sidestep_engine.ui.wizard._update_session_defaults",
            ),
        ):
            yielded = []
            for ns in run_wizard_session():
                call_order.append(f"dispatch:{ns.subcommand}")
                yielded.append(ns)

        # Fisher (analyze) dispatch must happen before chain offer
        self.assertEqual(call_order, ["dispatch:analyze", "chain_offer"])
        self.assertEqual(len(yielded), 1)
        self.assertEqual(yielded[0].subcommand, "analyze")

    def test_non_fisher_ns_does_not_trigger_chain(self) -> None:
        """Non-fisher namespaces should not trigger the PP++ chain offer."""
        from sidestep_engine.ui.wizard import run_wizard_session

        train_ns = argparse.Namespace(
            subcommand="train",
            preprocess=False,
            tensor_output=None,
        )

        with (
            patch("sidestep_engine.ui.banner.show_banner"),
            patch("sidestep_engine.ui.wizard._ensure_first_run_done"),
            patch("sidestep_engine.ui.wizard._main_menu", side_effect=[train_ns, None]),
            patch(
                "sidestep_engine.ui.wizard._offer_pp_train_chain",
            ) as mock_chain,
            patch("sidestep_engine.ui.wizard._update_session_defaults"),
        ):
            for _ in run_wizard_session():
                pass

        mock_chain.assert_not_called()


if __name__ == "__main__":
    unittest.main()
