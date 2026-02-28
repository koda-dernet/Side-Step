"""Tests for Preprocessing++ wizard guardrails and defaults."""

from __future__ import annotations

import inspect
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


class TestCheckpointDefaultsFromSettings(unittest.TestCase):
    """Checkpoint prompt defaults should honor saved settings."""

    def test_fisher_step_model_uses_settings_checkpoint_default(self) -> None:
        from sidestep_engine.ui.flows import fisher as flows_fisher

        with tempfile.TemporaryDirectory() as td:
            ckpt_dir = str(Path(td) / "ckpt")
            ds_dir = str(Path(td) / "data")
            Path(ckpt_dir).mkdir(parents=True, exist_ok=True)
            Path(ds_dir).mkdir(parents=True, exist_ok=True)
            (Path(ds_dir) / "dummy.pt").write_bytes(b"x")

            ask_path_calls: list[tuple[str, str | None]] = []

            def _ask_path(label, default=None, **kwargs):
                ask_path_calls.append((label, default))
                if "Checkpoint directory" in label:
                    return ckpt_dir
                return ds_dir

            with (
                patch("sidestep_engine.settings.get_checkpoint_dir", return_value=ckpt_dir),
                patch("sidestep_engine.ui.flows.wizard_shared_steps.ask_path", side_effect=_ask_path),
                patch(
                    "sidestep_engine.ui.flows.wizard_shared_steps.pick_model",
                    return_value=("acestep-v15-base", SimpleNamespace(base_model="base")),
                ),
                patch("sidestep_engine.ui.flows.wizard_shared_steps.describe_preprocessed_dataset_issue", return_value=None),
            ):
                answers = {}
                flows_fisher._step_model(answers)

            self.assertEqual(ask_path_calls[0][0], "Checkpoint directory")
            self.assertEqual(ask_path_calls[0][1], ckpt_dir)

    def test_preprocess_step_model_uses_settings_checkpoint_default(self) -> None:
        from sidestep_engine.ui.flows import preprocess as flows_preprocess

        with tempfile.TemporaryDirectory() as td:
            ckpt_dir = str(Path(td) / "ckpt")
            Path(ckpt_dir).mkdir(parents=True, exist_ok=True)

            ask_path_calls: list[tuple[str, str | None]] = []

            def _ask_path(label, default=None, **kwargs):
                ask_path_calls.append((label, default))
                return ckpt_dir

            with (
                patch("sidestep_engine.settings.get_checkpoint_dir", return_value=ckpt_dir),
                patch("sidestep_engine.ui.flows.preprocess.ask_path", side_effect=_ask_path),
                patch(
                    "sidestep_engine.ui.flows.wizard_shared_steps.pick_model",
                    return_value=("acestep-v15-base", SimpleNamespace(base_model="base", is_official=True)),
                ),
            ):
                answers = {}
                flows_preprocess._step_model(answers)

            self.assertEqual(ask_path_calls[0][0], "Checkpoint directory")
            self.assertEqual(ask_path_calls[0][1], ckpt_dir)


class TestPreprocessingPPTurboGate(unittest.TestCase):
    """Turbo-specific opt-in should gate Preprocessing++ flow."""

    def test_turbo_decline_reprompts_model_selection(self) -> None:
        from sidestep_engine.ui.flows import fisher as flows_fisher

        with tempfile.TemporaryDirectory() as td:
            ckpt_dir = str(Path(td) / "ckpt")
            ds_dir = str(Path(td) / "data")
            Path(ckpt_dir).mkdir(parents=True, exist_ok=True)
            Path(ds_dir).mkdir(parents=True, exist_ok=True)
            (Path(ds_dir) / "dummy.pt").write_bytes(b"x")

            def _ask_path(label, default=None, **kwargs):
                if "Checkpoint directory" in label:
                    return ckpt_dir
                return ds_dir

            def _ask(label, default=None, **kwargs):
                if "Base rank" in label:
                    return 64
                if "Minimum rank" in label:
                    return 16
                if "Maximum rank" in label:
                    return 128
                if "Proceed?" in label:
                    return "y"
                return default

            # _warn_non_base_variant calls ask_bool from wizard_shared_steps namespace
            _decline_variant_then_accept = [False]  # decline turbo variant
            def _ws_ask_bool(label, **kw):
                if "variant" in label.lower():
                    if _decline_variant_then_accept:
                        _decline_variant_then_accept.pop()
                        return False
                    return True
                return True

            with (
                patch("sidestep_engine.settings.get_checkpoint_dir", return_value=ckpt_dir),
                patch("sidestep_engine.ui.flows.wizard_shared_steps.ask_path", side_effect=_ask_path),
                patch(
                    "sidestep_engine.ui.flows.wizard_shared_steps.pick_model",
                    side_effect=[
                        ("acestep-v15-turbo", SimpleNamespace(base_model="turbo")),
                        ("acestep-v15-base", SimpleNamespace(base_model="base")),
                    ],
                ) as mock_pick,
                patch(
                    "sidestep_engine.ui.flows.fisher.ask_bool",
                    side_effect=[True, False, True, True],
                ) as mock_ask_bool,
                patch("sidestep_engine.ui.flows.wizard_shared_steps.ask_bool", side_effect=_ws_ask_bool),
                patch("sidestep_engine.ui.flows.fisher.ask", side_effect=_ask),
                patch("sidestep_engine.ui.flows.fisher.menu", return_value="texture"),
                patch("sidestep_engine.ui.flows.common.offer_load_preset_subset"),
                patch("sidestep_engine.ui.flows.wizard_shared_steps.describe_preprocessed_dataset_issue", return_value=None),
            ):
                ns = flows_fisher.wizard_preprocessing_pp()

            self.assertEqual(mock_pick.call_count, 2)
            self.assertEqual(ns.model_variant, "acestep-v15-base")

    def test_turbo_accept_continues(self) -> None:
        from sidestep_engine.ui.flows import fisher as flows_fisher

        with tempfile.TemporaryDirectory() as td:
            ckpt_dir = str(Path(td) / "ckpt")
            ds_dir = str(Path(td) / "data")
            Path(ckpt_dir).mkdir(parents=True, exist_ok=True)
            Path(ds_dir).mkdir(parents=True, exist_ok=True)
            (Path(ds_dir) / "dummy.pt").write_bytes(b"x")

            def _ask_path(label, default=None, **kwargs):
                if "Checkpoint directory" in label:
                    return ckpt_dir
                return ds_dir

            def _ask(label, default=None, **kwargs):
                if "Base rank" in label:
                    return 64
                if "Minimum rank" in label:
                    return 16
                if "Maximum rank" in label:
                    return 128
                if "Proceed?" in label:
                    return "y"
                return default

            with (
                patch("sidestep_engine.settings.get_checkpoint_dir", return_value=ckpt_dir),
                patch("sidestep_engine.ui.flows.wizard_shared_steps.ask_path", side_effect=_ask_path),
                patch(
                    "sidestep_engine.ui.flows.wizard_shared_steps.pick_model",
                    return_value=("acestep-v15-turbo", SimpleNamespace(base_model="turbo")),
                ),
                patch("sidestep_engine.ui.flows.fisher.ask_bool", return_value=True) as mock_ask_bool,
                patch("sidestep_engine.ui.flows.wizard_shared_steps.ask_bool", return_value=True),
                patch("sidestep_engine.ui.flows.fisher.ask", side_effect=_ask),
                patch("sidestep_engine.ui.flows.fisher.menu", return_value="texture"),
                patch("sidestep_engine.ui.flows.common.offer_load_preset_subset"),
                patch("sidestep_engine.ui.flows.wizard_shared_steps.describe_preprocessed_dataset_issue", return_value=None),
            ):
                ns = flows_fisher.wizard_preprocessing_pp()

            turbo_calls = [
                c for c in mock_ask_bool.call_args_list
                if "turbo" in str(c).lower()
            ]
            self.assertEqual(len(turbo_calls), 1)
            self.assertEqual(ns.model_variant, "acestep-v15-turbo")


class TestCompatibilityMessaging(unittest.TestCase):
    """Compatibility check should not mention removed vanilla mode."""

    def test_compat_source_has_no_vanilla_warning_text(self) -> None:
        from sidestep_engine import _compat

        src = inspect.getsource(_compat.check_compatibility)
        self.assertNotIn("Vanilla training mode", src)
        self.assertNotIn("Base ACE-Step not detected", src)


if __name__ == "__main__":
    unittest.main()

