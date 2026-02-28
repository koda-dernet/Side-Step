"""Tests for training review UX helpers."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

_TMP = tempfile.gettempdir()


class TestTrainingReviewFlow(unittest.TestCase):
    def test_review_menu_can_jump_to_training_section(self) -> None:
        from sidestep_engine.ui.flows import train as flows_train

        answers = {
            "adapter_type": "lora",
            "model_variant": "acestep-v15-base",
            "dataset_dir": f"{_TMP}/data",
            "output_dir": f"{_TMP}/out",
            "learning_rate": 1e-4,
            "batch_size": 1,
            "gradient_accumulation": 4,
            "epochs": 100,
            "rank": 64,
            "alpha": 128,
            "dropout": 0.1,
            "target_modules_str": "q_proj k_proj v_proj o_proj",
        }
        steps = [
            ("Required Settings", flows_train.step_required),
            ("LoRA Settings", flows_train.step_lora),
            ("Training Settings", flows_train.step_training),
            ("Logging & Checkpoints", flows_train.step_logging),
            ("Latent Chunking", flows_train.step_chunk_duration),
        ]

        with patch("sidestep_engine.ui.flows.train.menu", return_value="edit_training"):
            idx = flows_train._review_and_confirm(answers, "basic", steps)
        self.assertEqual(idx, 2)

    def test_review_cancel_raises_goback(self) -> None:
        from sidestep_engine.ui.flows import train as flows_train
        from sidestep_engine.ui.prompt_helpers import GoBack

        answers = {"adapter_type": "lora"}
        steps = [("Required Settings", flows_train.step_required)]

        with patch("sidestep_engine.ui.flows.train.menu", return_value="cancel"):
            with self.assertRaises(GoBack):
                flows_train._review_and_confirm(answers, "basic", steps)

    @patch("sidestep_engine.ui.prompt_helpers.ask_bool", return_value=False)
    def test_check_fisher_map_marks_recommendation_without_running_analysis(
        self, _mock_ask: MagicMock,
    ) -> None:
        from sidestep_engine.ui.flows import train as flows_train

        with tempfile.TemporaryDirectory() as td:
            d = Path(td)
            (d / "sample.pt").write_bytes(b"x")
            answers = {"dataset_dir": str(d)}
            flows_train._check_fisher_map(answers, adapter_type="lora")

        self.assertFalse(answers.get("_fisher_map_detected", True))
        self.assertTrue(answers.get("_pp_recommended", False))
        self.assertEqual(answers.get("_pp_sample_count"), 1)

    def test_check_fisher_map_detects_existing_map(self) -> None:
        from sidestep_engine.ui.flows import train as flows_train

        with tempfile.TemporaryDirectory() as td:
            d = Path(td)
            (d / "fisher_map.json").write_text(
                json.dumps(
                    {
                        "rank_pattern": {"x": 64},
                        "rank_budget": {"min": 16, "max": 128},
                    }
                ),
                encoding="utf-8",
            )
            answers = {"dataset_dir": str(d)}
            flows_train._check_fisher_map(answers, adapter_type="lora")

        self.assertTrue(answers.get("_fisher_map_detected", False))
        self.assertFalse(answers.get("_pp_recommended", True))


if __name__ == "__main__":
    unittest.main()
