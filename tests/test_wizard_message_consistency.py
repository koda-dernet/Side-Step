"""Tests for cross-flow dataset/model message consistency helpers."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


class TestWizardMessageConsistency(unittest.TestCase):
    def test_fisher_uses_shared_dataset_issue_helper(self) -> None:
        from acestep.training_v2.ui import flows_fisher

        with tempfile.TemporaryDirectory() as td:
            ckpt_dir = str(Path(td) / "ckpt")
            ds_dir = str(Path(td) / "data")
            Path(ckpt_dir).mkdir(parents=True, exist_ok=True)
            Path(ds_dir).mkdir(parents=True, exist_ok=True)

            def _ask_path(label, default=None, **kwargs):
                if "Checkpoint directory" in label:
                    return ckpt_dir
                return ds_dir

            with (
                patch("acestep.training_v2.settings.get_checkpoint_dir", return_value=ckpt_dir),
                patch("acestep.training_v2.ui.flows_fisher.ask_path", side_effect=_ask_path),
                patch(
                    "acestep.training_v2.model_discovery.pick_model",
                    return_value=("acestep-v15-base", SimpleNamespace(base_model="base")),
                ),
                patch(
                    "acestep.training_v2.ui.flows_common.describe_preprocessed_dataset_issue",
                    side_effect=["manifest.json is invalid JSON: bad", None],
                ),
                patch("acestep.training_v2.ui.flows_common.show_dataset_issue") as show_issue,
            ):
                answers = {}
                flows_fisher._step_model(answers)

            show_issue.assert_called_once()

    def test_estimate_uses_shared_dataset_issue_helper(self) -> None:
        from acestep.training_v2.ui import flows_estimate

        with tempfile.TemporaryDirectory() as td:
            ckpt_dir = str(Path(td) / "ckpt")
            ds_dir = str(Path(td) / "data")
            Path(ckpt_dir).mkdir(parents=True, exist_ok=True)
            Path(ds_dir).mkdir(parents=True, exist_ok=True)

            def _ask_path(label, default=None, **kwargs):
                if "Checkpoint directory" in label:
                    return ckpt_dir
                return ds_dir

            with (
                patch("acestep.training_v2.settings.get_checkpoint_dir", return_value=ckpt_dir),
                patch("acestep.training_v2.ui.flows_estimate.ask_path", side_effect=_ask_path),
                patch(
                    "acestep.training_v2.model_discovery.pick_model",
                    return_value=("acestep-v15-base", SimpleNamespace(base_model="base")),
                ),
                patch(
                    "acestep.training_v2.ui.flows_common.describe_preprocessed_dataset_issue",
                    side_effect=["manifest.json is invalid JSON: bad", None],
                ),
                patch("acestep.training_v2.ui.flows_common.show_dataset_issue") as show_issue,
            ):
                answers = {}
                flows_estimate._step_model(answers)

            show_issue.assert_called_once()


if __name__ == "__main__":
    unittest.main()
