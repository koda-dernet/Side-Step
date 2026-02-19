"""Tests for turbo auto-select training: discrete sampling, is_turbo detection, and training_step branching."""

from __future__ import annotations

import argparse
import tempfile
import unittest
from unittest.mock import MagicMock, patch

import torch

_TMP = tempfile.gettempdir()


class TestDiscreteTimestepSampling(unittest.TestCase):
    """Verify sample_discrete_timesteps returns valid turbo schedule values."""

    def test_output_shape(self):
        """Each call returns (t, r) tensors of shape [batch_size]."""
        from acestep.training_v2.timestep_sampling import sample_discrete_timesteps

        t, r = sample_discrete_timesteps(batch_size=16, device="cpu", dtype=torch.float32)
        self.assertEqual(t.shape, (16,))
        self.assertEqual(r.shape, (16,))

    def test_r_equals_t(self):
        """r must equal t (use_meanflow=False for all ACE-Step variants)."""
        from acestep.training_v2.timestep_sampling import sample_discrete_timesteps

        t, r = sample_discrete_timesteps(batch_size=32, device="cpu", dtype=torch.float32)
        self.assertTrue(torch.equal(t, r))

    def test_values_in_schedule(self):
        """Every sampled value must be one of the 8 turbo timesteps."""
        from acestep.training_v2.timestep_sampling import (
            TURBO_SHIFT3_TIMESTEPS,
            sample_discrete_timesteps,
        )

        t, _ = sample_discrete_timesteps(batch_size=1000, device="cpu", dtype=torch.float32)
        # Build reference set from the same torch tensor to avoid fp precision mismatch
        ref = torch.tensor(TURBO_SHIFT3_TIMESTEPS, dtype=torch.float32)
        unique_vals = t.unique()
        for val in unique_vals:
            self.assertTrue(
                torch.any(torch.isclose(ref, val)),
                f"Sampled value {val.item()} is not in TURBO_SHIFT3_TIMESTEPS",
            )

    def test_schedule_length(self):
        """The turbo schedule has exactly 8 values."""
        from acestep.training_v2.timestep_sampling import TURBO_SHIFT3_TIMESTEPS

        self.assertEqual(len(TURBO_SHIFT3_TIMESTEPS), 8)

    def test_custom_schedule(self):
        """Passing a custom schedule overrides the default."""
        from acestep.training_v2.timestep_sampling import sample_discrete_timesteps

        custom = [0.5, 0.25]
        t, _ = sample_discrete_timesteps(
            batch_size=100, device="cpu", dtype=torch.float32, timesteps=custom,
        )
        for val in t.tolist():
            self.assertIn(val, custom)

    def test_batch_size_one(self):
        """Edge case: batch_size=1 must still work."""
        from acestep.training_v2.timestep_sampling import sample_discrete_timesteps

        t, r = sample_discrete_timesteps(batch_size=1, device="cpu", dtype=torch.float32)
        self.assertEqual(t.shape, (1,))
        self.assertEqual(r.shape, (1,))


class TestIsTurboDetection(unittest.TestCase):
    """Verify is_turbo is correctly auto-detected from model variant and config."""

    def _build_args(self, **overrides):
        """Build a minimal argparse.Namespace for config_builder.build_configs."""
        defaults = dict(
            checkpoint_dir=f"{_TMP}/fake_checkpoints",
            model_variant="turbo",
            base_model="turbo",
            device="cpu",
            precision="fp32",
            adapter_type="lora",
            rank=64,
            alpha=128,
            dropout=0.1,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            attention_type="both",
            bias="none",
            num_workers=0,
            pin_memory=False,
            prefetch_factor=0,
            persistent_workers=False,
            learning_rate=1e-4,
            batch_size=1,
            gradient_accumulation=4,
            epochs=100,
            warmup_steps=100,
            weight_decay=0.01,
            max_grad_norm=1.0,
            seed=42,
            output_dir=f"{_TMP}/fake_output",
            save_every=10,
            resume_from=None,
            log_dir=None,
            log_every=10,
            log_heavy_every=50,
            shift=3.0,
            num_inference_steps=8,
            optimizer_type="adamw",
            scheduler_type="cosine",
            gradient_checkpointing=True,
            offload_encoder=False,
            cfg_ratio=0.15,
            estimate_batches=None,
            top_k=16,
            granularity="module",
            module_config=None,
            auto_estimate=False,
            estimate_output=None,
            preprocess=False,
            audio_dir=None,
            dataset_dir=f"{_TMP}/fake_dataset",
            dataset_json=None,
            tensor_output=None,
            max_duration=240.0,
        )
        defaults.update(overrides)
        return argparse.Namespace(**defaults)

    @patch("acestep.training_v2.cli.config_builder._resolve_model_config_path")
    @patch("acestep.training_v2.cli.config_builder.detect_gpu")
    def test_turbo_variant_detected(self, mock_gpu, mock_config_path):
        """base_model='turbo' should set is_turbo=True."""
        from acestep.training_v2.cli.config_builder import build_configs
        from pathlib import Path

        mock_gpu.return_value = MagicMock(device="cpu", precision="fp32")
        mock_config_path.return_value = Path("/nonexistent/config.json")

        args = self._build_args(base_model="turbo", model_variant="turbo")
        _, train_cfg = build_configs(args)
        self.assertTrue(train_cfg.is_turbo)

    @patch("acestep.training_v2.cli.config_builder._resolve_model_config_path")
    @patch("acestep.training_v2.cli.config_builder.detect_gpu")
    def test_base_variant_not_turbo(self, mock_gpu, mock_config_path):
        """base_model='base' with 50 inference steps should set is_turbo=False."""
        from acestep.training_v2.cli.config_builder import build_configs
        from pathlib import Path

        mock_gpu.return_value = MagicMock(device="cpu", precision="fp32")
        mock_config_path.return_value = Path("/nonexistent/config.json")

        args = self._build_args(
            base_model="base", model_variant="base",
            shift=1.0, num_inference_steps=50,
        )
        _, train_cfg = build_configs(args)
        self.assertFalse(train_cfg.is_turbo)

    @patch("acestep.training_v2.cli.config_builder._resolve_model_config_path")
    @patch("acestep.training_v2.cli.config_builder.detect_gpu")
    def test_turbo_finetune_detected_by_steps(self, mock_gpu, mock_config_path):
        """A custom fine-tune with 8 inference steps should be detected as turbo."""
        from acestep.training_v2.cli.config_builder import build_configs
        from pathlib import Path

        mock_gpu.return_value = MagicMock(device="cpu", precision="fp32")
        mock_config_path.return_value = Path("/nonexistent/config.json")

        args = self._build_args(
            base_model="my-custom-finetune", model_variant="my-custom-finetune",
            num_inference_steps=8,
        )
        _, train_cfg = build_configs(args)
        self.assertTrue(train_cfg.is_turbo)

    @patch("acestep.training_v2.cli.config_builder._resolve_model_config_path")
    @patch("acestep.training_v2.cli.config_builder.detect_gpu")
    def test_non_turbo_finetune_detected_by_steps(self, mock_gpu, mock_config_path):
        """A custom fine-tune with non-8 steps should be treated as base/SFT."""
        from acestep.training_v2.cli.config_builder import build_configs
        from pathlib import Path

        mock_gpu.return_value = MagicMock(device="cpu", precision="fp32")
        mock_config_path.return_value = Path("/nonexistent/config.json")

        args = self._build_args(
            base_model="my-custom-finetune", model_variant="my-custom-finetune",
            num_inference_steps=50, shift=1.0,
        )
        _, train_cfg = build_configs(args)
        self.assertFalse(train_cfg.is_turbo)


class TestTrainingStepBranching(unittest.TestCase):
    """Verify FixedLoRAModule branches between discrete and continuous sampling."""

    @patch("acestep.training_v2.fixed_lora_module.sample_discrete_timesteps")
    @patch("acestep.training_v2.fixed_lora_module.sample_timesteps")
    def test_turbo_uses_discrete(self, mock_continuous, mock_discrete):
        """When is_turbo=True, training_step must call sample_discrete_timesteps."""
        from acestep.training_v2.fixed_lora_module import FixedLoRAModule
        from acestep.training_v2.configs import LoRAConfigV2, TrainingConfigV2

        bsz = 2
        seq_len = 10
        hidden = 64

        mock_discrete.return_value = (
            torch.tensor([0.5, 0.3]),
            torch.tensor([0.5, 0.3]),
        )

        model = MagicMock()
        model.config = MagicMock()
        model.null_condition_emb = torch.randn(1, 1, 128)
        model.decoder = MagicMock()
        model.decoder.return_value = (torch.randn(bsz, seq_len, hidden),)

        lora_cfg = LoRAConfigV2(r=8, alpha=16)
        train_cfg = TrainingConfigV2(
            is_turbo=True, output_dir=f"{_TMP}/test",
        )

        with patch.object(FixedLoRAModule, "_inject_lora"):
            module = FixedLoRAModule(model, lora_cfg, train_cfg, "cpu", torch.float32)
            module.model = model

        batch = {
            "target_latents": torch.randn(bsz, seq_len, hidden),
            "attention_mask": torch.ones(bsz, seq_len),
            "encoder_hidden_states": torch.randn(bsz, 5, 128),
            "encoder_attention_mask": torch.ones(bsz, 5),
            "context_latents": torch.randn(bsz, seq_len, 128),
        }

        module.training_step(batch)
        mock_discrete.assert_called_once()
        mock_continuous.assert_not_called()

    @patch("acestep.training_v2.fixed_lora_module.sample_discrete_timesteps")
    @patch("acestep.training_v2.fixed_lora_module.sample_timesteps")
    def test_base_uses_continuous(self, mock_continuous, mock_discrete):
        """When is_turbo=False, training_step must call sample_timesteps."""
        from acestep.training_v2.fixed_lora_module import FixedLoRAModule
        from acestep.training_v2.configs import LoRAConfigV2, TrainingConfigV2

        bsz = 2
        seq_len = 10
        hidden = 64

        mock_continuous.return_value = (
            torch.tensor([0.4, 0.6]),
            torch.tensor([0.4, 0.6]),
        )

        model = MagicMock()
        model.config = MagicMock()
        model.null_condition_emb = torch.randn(1, 1, 128)
        model.decoder = MagicMock()
        model.decoder.return_value = (torch.randn(bsz, seq_len, hidden),)

        lora_cfg = LoRAConfigV2(r=8, alpha=16)
        train_cfg = TrainingConfigV2(
            is_turbo=False, output_dir=f"{_TMP}/test",
        )

        with patch.object(FixedLoRAModule, "_inject_lora"):
            module = FixedLoRAModule(model, lora_cfg, train_cfg, "cpu", torch.float32)
            module.model = model

        batch = {
            "target_latents": torch.randn(bsz, seq_len, hidden),
            "attention_mask": torch.ones(bsz, seq_len),
            "encoder_hidden_states": torch.randn(bsz, 5, 128),
            "encoder_attention_mask": torch.ones(bsz, 5),
            "context_latents": torch.randn(bsz, seq_len, 128),
        }

        module.training_step(batch)
        mock_continuous.assert_called_once()
        mock_discrete.assert_not_called()


class TestWizardTurboDetection(unittest.TestCase):
    """Verify _is_turbo_variant helper in flows_train."""

    def test_turbo_base_model(self):
        from acestep.training_v2.ui.flows_train import _is_turbo_variant

        self.assertTrue(_is_turbo_variant({"base_model": "turbo"}))
        self.assertTrue(_is_turbo_variant({"base_model": "TURBO"}))
        self.assertTrue(_is_turbo_variant({"base_model": "acestep-v15-turbo"}))

    def test_base_model(self):
        from acestep.training_v2.ui.flows_train import _is_turbo_variant

        self.assertFalse(_is_turbo_variant({"base_model": "base", "num_inference_steps": 50}))
        self.assertFalse(_is_turbo_variant({"base_model": "sft", "num_inference_steps": 50}))

    def test_turbo_finetune_by_steps(self):
        from acestep.training_v2.ui.flows_train import _is_turbo_variant

        self.assertTrue(_is_turbo_variant({"base_model": "my-finetune", "num_inference_steps": 8}))

    def test_non_turbo_finetune_by_steps(self):
        from acestep.training_v2.ui.flows_train import _is_turbo_variant

        self.assertFalse(_is_turbo_variant({"base_model": "my-finetune", "num_inference_steps": 50}))

    def test_unknown_base_defaults_turbo(self):
        """When base_model is not set and default inference steps is 8, treat as turbo."""
        from acestep.training_v2.ui.flows_train import _is_turbo_variant

        self.assertTrue(_is_turbo_variant({}))


class TestWizardLossWeightingGuard(unittest.TestCase):
    """Ensure turbo namespace never carries stale min_snr loss weighting."""

    def test_turbo_forces_loss_weighting_none(self):
        from acestep.training_v2.ui.flows_common import build_train_namespace

        answers = {
            "checkpoint_dir": f"{_TMP}/ckpt",
            "model_variant": "turbo",
            "base_model": "turbo",
            "dataset_dir": f"{_TMP}/data",
            "output_dir": f"{_TMP}/out",
            "loss_weighting": "min_snr",
            "snr_gamma": 5.0,
        }
        ns = build_train_namespace(answers)
        self.assertEqual(ns.loss_weighting, "none")


if __name__ == "__main__":
    unittest.main()
