"""Tests for wizard session carry-over defaults and chain context."""

from __future__ import annotations

import argparse
import tempfile
import unittest
from unittest.mock import patch

_TMP = tempfile.gettempdir()


class TestWizardSessionCarryover(unittest.TestCase):
    def test_session_defaults_carry_forward_between_actions(self) -> None:
        from acestep.training_v2.ui import wizard

        preprocess_ns = argparse.Namespace(
            preprocess=True,
            tensor_output=f"{_TMP}/tensors",
            checkpoint_dir=f"{_TMP}/checkpoints",
            model_variant="acestep-v15-base",
            dataset_dir=f"{_TMP}/tensors",
        )

        calls: list[dict] = []
        menu_results = iter([preprocess_ns, None])

        def _main_menu(session_defaults=None):
            calls.append(dict(session_defaults or {}))
            return next(menu_results)

        with (
            patch("acestep.training_v2.ui.banner.show_banner"),
            patch("acestep.training_v2.ui.wizard._ensure_first_run_done"),
            patch("acestep.training_v2.ui.wizard._main_menu", side_effect=_main_menu),
            patch("acestep.training_v2.ui.prompt_helpers.ask_bool", return_value=False),
        ):
            gen = wizard.run_wizard_session()
            first = next(gen)
            self.assertEqual(first.tensor_output, f"{_TMP}/tensors")
            with self.assertRaises(StopIteration):
                next(gen)

        self.assertEqual(calls[0], {})
        self.assertEqual(calls[1].get("dataset_dir"), f"{_TMP}/tensors")
        self.assertEqual(calls[1].get("checkpoint_dir"), f"{_TMP}/checkpoints")
        self.assertEqual(calls[1].get("model_variant"), "acestep-v15-base")

    def test_chain_training_uses_carried_context(self) -> None:
        from acestep.training_v2.ui import wizard

        preprocess_ns = argparse.Namespace(
            preprocess=True,
            tensor_output=f"{_TMP}/tensors",
            checkpoint_dir=f"{_TMP}/checkpoints",
            model_variant="acestep-v15-base",
            dataset_dir=f"{_TMP}/tensors",
        )
        chain_ns = argparse.Namespace(
            preprocess=False,
            checkpoint_dir=f"{_TMP}/checkpoints",
            model_variant="acestep-v15-base",
            dataset_dir=f"{_TMP}/tensors",
        )
        captured: dict = {}

        def _wizard_train(**kwargs):
            captured.update(kwargs.get("preset", {}))
            return chain_ns

        with (
            patch("acestep.training_v2.ui.banner.show_banner"),
            patch("acestep.training_v2.ui.wizard._ensure_first_run_done"),
            patch(
                "acestep.training_v2.ui.wizard._main_menu",
                side_effect=[preprocess_ns, None],
            ),
            patch("acestep.training_v2.ui.prompt_helpers.ask_bool", return_value=True),
            patch("acestep.training_v2.ui.wizard.menu", return_value="lora"),
            patch("acestep.training_v2.ui.wizard.wizard_train", side_effect=_wizard_train),
        ):
            outputs = list(wizard.run_wizard_session())

        self.assertEqual(len(outputs), 2)
        self.assertEqual(outputs[0].tensor_output, f"{_TMP}/tensors")
        self.assertEqual(outputs[1].dataset_dir, f"{_TMP}/tensors")
        self.assertEqual(captured.get("dataset_dir"), f"{_TMP}/tensors")
        self.assertEqual(captured.get("checkpoint_dir"), f"{_TMP}/checkpoints")
        self.assertEqual(captured.get("model_variant"), "acestep-v15-base")


if __name__ == "__main__":
    unittest.main()
