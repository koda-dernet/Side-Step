"""Tests for TensorBoard utilities and versioned log directory management."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import torch
import torch.nn as nn

from acestep.training_v2.tensorboard_utils import (
    TrainingLogger,
    resolve_latest_versioned_log_dir,
    resolve_versioned_log_dir,
)


class TestVersionedLogDirectories(unittest.TestCase):
    """Test versioned log directory resolution."""

    def test_resolve_versioned_log_dir_first_run(self):
        """First run should create _v0 directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_root = Path(tmpdir)
            result = resolve_versioned_log_dir(log_root, "jazz")

            self.assertEqual(result.name, "jazz_v0")
            self.assertTrue(result.exists())

    def test_resolve_versioned_log_dir_second_run(self):
        """Second run should create _v1 directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_root = Path(tmpdir)
            (log_root / "jazz_v0").mkdir(parents=True)

            result = resolve_versioned_log_dir(log_root, "jazz")

            self.assertEqual(result.name, "jazz_v1")
            self.assertTrue(result.exists())

    def test_resolve_versioned_log_dir_gaps_in_versions(self):
        """Should find next version even with gaps."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_root = Path(tmpdir)
            (log_root / "jazz_v0").mkdir(parents=True)
            (log_root / "jazz_v2").mkdir(parents=True)
            # Missing v1, should create v3

            result = resolve_versioned_log_dir(log_root, "jazz")

            self.assertEqual(result.name, "jazz_v3")
            self.assertTrue(result.exists())

    def test_resolve_versioned_log_dir_race_condition(self):
        """Should handle race condition by incrementing version."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_root = Path(tmpdir)
            (log_root / "jazz_v0").mkdir(parents=True)

            # Simulate race condition by creating v1 during resolution
            with patch.object(Path, "mkdir") as mock_mkdir:
                def side_effect(*args, **kwargs):
                    if kwargs.get("exist_ok") is False:
                        raise FileExistsError()
                    else:
                        real_path = Path(tmpdir) / "jazz_v2"
                        real_path.mkdir(parents=True, exist_ok=True)

                mock_mkdir.side_effect = side_effect
                result = resolve_versioned_log_dir(log_root, "jazz")

                # Should fall back to v2
                self.assertEqual(result.name, "jazz_v2")

    def test_resolve_versioned_log_dir_special_characters(self):
        """Should handle run names with special characters."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_root = Path(tmpdir)
            result = resolve_versioned_log_dir(log_root, "my-run_2024")

            self.assertTrue(result.name.startswith("my-run_2024_v"))
            self.assertTrue(result.exists())

    def test_resolve_latest_versioned_log_dir_none_exists(self):
        """Should return None when no versioned dirs exist."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_root = Path(tmpdir)
            result = resolve_latest_versioned_log_dir(log_root, "jazz")

            self.assertIsNone(result)

    def test_resolve_latest_versioned_log_dir_single(self):
        """Should find single versioned directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_root = Path(tmpdir)
            v0 = log_root / "jazz_v0"
            v0.mkdir(parents=True)

            result = resolve_latest_versioned_log_dir(log_root, "jazz")

            self.assertEqual(result, v0)

    def test_resolve_latest_versioned_log_dir_multiple(self):
        """Should find highest version number."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_root = Path(tmpdir)
            (log_root / "jazz_v0").mkdir(parents=True)
            (log_root / "jazz_v1").mkdir(parents=True)
            v5 = log_root / "jazz_v5"
            v5.mkdir(parents=True)

            result = resolve_latest_versioned_log_dir(log_root, "jazz")

            self.assertEqual(result, v5)

    def test_resolve_latest_versioned_log_dir_ignores_other_runs(self):
        """Should ignore directories for other run names."""
        with tempfile.TemporaryDirectory() as tmpdir:
            log_root = Path(tmpdir)
            jazz_v2 = log_root / "jazz_v2"
            jazz_v2.mkdir(parents=True)
            (log_root / "rock_v5").mkdir(parents=True)

            result = resolve_latest_versioned_log_dir(log_root, "jazz")

            self.assertEqual(result, jazz_v2)

    def test_resolve_latest_versioned_log_dir_nonexistent_root(self):
        """Should return None when log root doesn't exist."""
        result = resolve_latest_versioned_log_dir("/nonexistent/path", "jazz")
        self.assertIsNone(result)


class TestTrainingLogger(unittest.TestCase):
    """Test TrainingLogger wrapper."""

    def test_logger_creation_without_tensorboard(self):
        """Should create silent logger when TensorBoard not available."""
        with patch("acestep.training_v2.tensorboard_utils._TB_AVAILABLE", False):
            with tempfile.TemporaryDirectory() as tmpdir:
                logger = TrainingLogger(tmpdir, enabled=True)

                # All methods should be silent no-ops
                logger.log_scalar("test", 1.0, 0)
                logger.log_loss(0.5, 0)
                logger.log_lr(1e-4, 0)
                logger.flush()
                logger.close()

                self.assertIsNone(logger._writer)

    def test_logger_creation_disabled(self):
        """Should not create writer when disabled."""
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = TrainingLogger(tmpdir, enabled=False)

            self.assertIsNone(logger._writer)

    @patch("acestep.training_v2.tensorboard_utils._TB_AVAILABLE", True)
    @patch("acestep.training_v2.tensorboard_utils.SummaryWriter")
    def test_logger_creation_with_tensorboard(self, mock_writer):
        """Should create writer when TensorBoard is available."""
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = TrainingLogger(tmpdir, enabled=True)

            mock_writer.assert_called_once()
            self.assertIsNotNone(logger._writer)

    @patch("acestep.training_v2.tensorboard_utils._TB_AVAILABLE", True)
    @patch("acestep.training_v2.tensorboard_utils.SummaryWriter")
    def test_log_scalar(self, mock_writer_class):
        """Should log scalar values."""
        mock_writer = MagicMock()
        mock_writer_class.return_value = mock_writer

        with tempfile.TemporaryDirectory() as tmpdir:
            logger = TrainingLogger(tmpdir)
            logger.log_scalar("test/metric", 1.5, 100)

            mock_writer.add_scalar.assert_called_once_with("test/metric", 1.5, global_step=100)

    @patch("acestep.training_v2.tensorboard_utils._TB_AVAILABLE", True)
    @patch("acestep.training_v2.tensorboard_utils.SummaryWriter")
    def test_log_loss(self, mock_writer_class):
        """Should log loss values."""
        mock_writer = MagicMock()
        mock_writer_class.return_value = mock_writer

        with tempfile.TemporaryDirectory() as tmpdir:
            logger = TrainingLogger(tmpdir)
            logger.log_loss(0.5, 100)

            mock_writer.add_scalar.assert_called_once_with("train/loss", 0.5, global_step=100)

    @patch("acestep.training_v2.tensorboard_utils._TB_AVAILABLE", True)
    @patch("acestep.training_v2.tensorboard_utils.SummaryWriter")
    def test_log_lr(self, mock_writer_class):
        """Should log learning rate."""
        mock_writer = MagicMock()
        mock_writer_class.return_value = mock_writer

        with tempfile.TemporaryDirectory() as tmpdir:
            logger = TrainingLogger(tmpdir)
            logger.log_lr(1e-4, 100)

            mock_writer.add_scalar.assert_called_once_with("train/lr", 1e-4, global_step=100)

    @patch("acestep.training_v2.tensorboard_utils._TB_AVAILABLE", True)
    @patch("acestep.training_v2.tensorboard_utils.SummaryWriter")
    def test_log_per_layer_grad_norms(self, mock_writer_class):
        """Should compute and log per-parameter gradient norms."""
        mock_writer = MagicMock()
        mock_writer_class.return_value = mock_writer

        # Create a simple model with gradients
        model = nn.Linear(10, 5)
        for param in model.parameters():
            param.grad = torch.randn_like(param)

        with tempfile.TemporaryDirectory() as tmpdir:
            logger = TrainingLogger(tmpdir)
            norms = logger.log_per_layer_grad_norms(model, step=100)

            # Should return dict of norms
            self.assertIsInstance(norms, dict)
            self.assertGreater(len(norms), 0)

            # Should have called add_scalar for each parameter
            self.assertGreater(mock_writer.add_scalar.call_count, 0)

    @patch("acestep.training_v2.tensorboard_utils._TB_AVAILABLE", True)
    @patch("acestep.training_v2.tensorboard_utils.SummaryWriter")
    def test_log_estimation_scores(self, mock_writer_class):
        """Should log estimation scores."""
        mock_writer = MagicMock()
        mock_writer_class.return_value = mock_writer

        with tempfile.TemporaryDirectory() as tmpdir:
            logger = TrainingLogger(tmpdir)
            scores = {"module1": 0.5, "module2": 0.8}
            logger.log_estimation_scores(scores, step=100)

            self.assertEqual(mock_writer.add_scalar.call_count, 2)

    @patch("acestep.training_v2.tensorboard_utils._TB_AVAILABLE", True)
    @patch("acestep.training_v2.tensorboard_utils.SummaryWriter")
    def test_log_param_histogram(self, mock_writer_class):
        """Should log parameter histograms."""
        mock_writer = MagicMock()
        mock_writer_class.return_value = mock_writer

        model = nn.Linear(10, 5)

        with tempfile.TemporaryDirectory() as tmpdir:
            logger = TrainingLogger(tmpdir)
            logger.log_param_histogram(model, step=100)

            # Should call add_histogram for each trainable parameter
            self.assertGreater(mock_writer.add_histogram.call_count, 0)

    @patch("acestep.training_v2.tensorboard_utils._TB_AVAILABLE", True)
    @patch("acestep.training_v2.tensorboard_utils.SummaryWriter")
    def test_context_manager(self, mock_writer_class):
        """Should work as context manager."""
        mock_writer = MagicMock()
        mock_writer_class.return_value = mock_writer

        with tempfile.TemporaryDirectory() as tmpdir:
            with TrainingLogger(tmpdir) as logger:
                logger.log_loss(0.5, 0)

            # Should close writer on exit
            mock_writer.close.assert_called_once()

    @patch("acestep.training_v2.tensorboard_utils._TB_AVAILABLE", True)
    @patch("acestep.training_v2.tensorboard_utils.SummaryWriter")
    def test_flush(self, mock_writer_class):
        """Should flush writer."""
        mock_writer = MagicMock()
        mock_writer_class.return_value = mock_writer

        with tempfile.TemporaryDirectory() as tmpdir:
            logger = TrainingLogger(tmpdir)
            logger.flush()

            mock_writer.flush.assert_called_once()

    @patch("acestep.training_v2.tensorboard_utils._TB_AVAILABLE", True)
    @patch("acestep.training_v2.tensorboard_utils.SummaryWriter")
    def test_close(self, mock_writer_class):
        """Should close and clear writer."""
        mock_writer = MagicMock()
        mock_writer_class.return_value = mock_writer

        with tempfile.TemporaryDirectory() as tmpdir:
            logger = TrainingLogger(tmpdir)
            logger.close()

            mock_writer.close.assert_called_once()
            self.assertIsNone(logger._writer)


if __name__ == "__main__":
    unittest.main()