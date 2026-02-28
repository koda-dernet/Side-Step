"""Tests for cross-flow dataset/model message consistency helpers."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


class TestWizardMessageConsistency(unittest.TestCase):
    def test_fisher_uses_shared_dataset_issue_helper(self) -> None:
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

            with (
                patch("sidestep_engine.settings.get_checkpoint_dir", return_value=ckpt_dir),
                patch("sidestep_engine.ui.flows.wizard_shared_steps.ask_path", side_effect=_ask_path),
                patch(
                    "sidestep_engine.ui.flows.wizard_shared_steps.pick_model",
                    return_value=("acestep-v15-base", SimpleNamespace(base_model="base")),
                ),
                patch(
                    "sidestep_engine.ui.flows.wizard_shared_steps.describe_preprocessed_dataset_issue",
                    side_effect=["manifest.json is invalid JSON: bad", None],
                ),
                patch("sidestep_engine.ui.flows.wizard_shared_steps.show_dataset_issue") as show_issue,
            ):
                answers = {}
                flows_fisher._step_model(answers)

            show_issue.assert_called_once()



class TestShowWhatsChangedNotice(unittest.TestCase):
    """Verify the beta 0.9.0 workflow-change notice prints correctly."""

    def test_notice_prints_non_rich(self) -> None:
        """Non-Rich path prints the expected text."""
        from sidestep_engine.ui.flows.wizard_shared_steps import show_whats_changed_notice

        with patch("sidestep_engine.ui.prompt_helpers.is_rich_active", return_value=False):
            with patch("builtins.print") as mock_print:
                show_whats_changed_notice()
        # Collect all printed output
        output = " ".join(str(c[0][0]) for c in mock_print.call_args_list if c[0])
        self.assertIn("beta 0.9.0", output)
        self.assertIn("raw audio folder", output)
        self.assertIn("no longer needed", output)

    def test_notice_prints_rich(self) -> None:
        """Rich path prints via console.print with expected content."""
        from sidestep_engine.ui.flows.wizard_shared_steps import show_whats_changed_notice

        mock_console = SimpleNamespace(print=lambda *a, **kw: None)
        with (
            patch("sidestep_engine.ui.prompt_helpers.is_rich_active", return_value=True),
            patch("sidestep_engine.ui.prompt_helpers.console", mock_console),
        ):
            # Should not raise
            show_whats_changed_notice()


if __name__ == "__main__":
    unittest.main()
