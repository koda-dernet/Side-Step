"""Tests for Fisher map auto-detection in config_builder."""

from __future__ import annotations

import json
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path
from unittest.mock import patch


def _make_model_config(ckpt_dir: str, variant: str = "turbo") -> None:
    """Create a minimal model config.json so config_builder won't fail."""
    variant_dir_map = {
        "turbo": "acestep-v15-turbo",
        "base": "acestep-v15-base",
        "sft": "acestep-v15-sft",
    }
    sub = variant_dir_map.get(variant, variant)
    d = Path(ckpt_dir) / sub
    d.mkdir(parents=True, exist_ok=True)
    cfg = {"timestep_mu": -0.4, "timestep_sigma": 1.0, "data_proportion": 0.5}
    (d / "config.json").write_text(json.dumps(cfg))


class TestFisherAutoDetection(unittest.TestCase):
    """Verify config_builder auto-detects and applies fisher_map.json."""

    def _make_fisher_map(self, dataset_dir: str, **overrides) -> Path:
        data = {
            "version": 2,
            "model_variant": "turbo",
            "target_modules": ["self_attn.q_proj", "self_attn.v_proj"],
            "rank_pattern": {
                "layers.0.self_attn.q_proj": 96,
                "layers.0.self_attn.v_proj": 48,
            },
            "alpha_pattern": {
                "layers.0.self_attn.q_proj": 192,
                "layers.0.self_attn.v_proj": 96,
            },
            "rank_budget": {"min": 16, "max": 128},
        }
        data.update(overrides)
        path = Path(dataset_dir) / "fisher_map.json"
        path.write_text(json.dumps(data))
        return path

    def _make_args(self, dataset_dir: str, ckpt_dir: str, **overrides) -> Namespace:
        defaults = dict(
            dataset_dir=dataset_dir,
            checkpoint_dir=ckpt_dir,
            model_variant="turbo",
            base_model=None,
            adapter_type="lora",
            rank=64,
            alpha=128,
            dropout=0.1,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
            bias="none",
            attention_type="both",
            self_target_modules=None,
            cross_target_modules=None,
            target_mlp=False,
            shift=3.0,
            num_inference_steps=8,
            ignore_fisher_map=False,
            device="auto",
            precision="auto",
            learning_rate=1e-4,
            batch_size=1,
            gradient_accumulation=4,
            epochs=100,
            warmup_steps=100,
            weight_decay=0.01,
            max_grad_norm=1.0,
            seed=42,
            output_dir="./output",
            save_every=10,
            resume_from=None,
            log_dir=None,
            log_every=10,
            log_heavy_every=50,
            optimizer_type="adamw",
            scheduler_type="cosine",
            gradient_checkpointing=True,
            offload_encoder=False,
            preprocess=False,
            audio_dir=None,
            dataset_json=None,
            tensor_output=None,
            max_duration=0,
            normalize="none",
            num_workers=0,
            pin_memory=False,
            prefetch_factor=0,
            persistent_workers=False,
            cfg_ratio=0.15,
            loss_weighting="none",
            snr_gamma=5.0,
            save_best=True,
            save_best_after=200,
            early_stop_patience=0,
            chunk_duration=None,
        )
        defaults.update(overrides)
        return Namespace(**defaults)

    def test_fisher_map_applied(self):
        """When fisher_map.json exists, adapter_cfg should be overridden."""
        from acestep.training_v2.cli.config_builder import build_configs

        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt_dir = str(Path(tmpdir) / "ckpts")
            ds_dir = str(Path(tmpdir) / "data")
            Path(ds_dir).mkdir()
            _make_model_config(ckpt_dir, "turbo")
            self._make_fisher_map(ds_dir)
            args = self._make_args(ds_dir, ckpt_dir)
            adapter_cfg, _train = build_configs(args)

            self.assertEqual(adapter_cfg.target_modules, ["self_attn.q_proj", "self_attn.v_proj"])
            self.assertEqual(adapter_cfg.rank_pattern["layers.0.self_attn.q_proj"], 96)
            self.assertEqual(adapter_cfg.r, 16)  # safety fallback

    def test_no_fisher_map_unchanged(self):
        """Without fisher_map.json, adapter_cfg stays at flat rank."""
        from acestep.training_v2.cli.config_builder import build_configs

        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt_dir = str(Path(tmpdir) / "ckpts")
            ds_dir = str(Path(tmpdir) / "data")
            Path(ds_dir).mkdir()
            _make_model_config(ckpt_dir, "turbo")
            args = self._make_args(ds_dir, ckpt_dir)
            adapter_cfg, _train = build_configs(args)

            self.assertEqual(adapter_cfg.r, 64)
            self.assertEqual(adapter_cfg.rank_pattern, {})

    def test_ignore_flag_bypasses(self):
        """--ignore-fisher-map should prevent loading."""
        from acestep.training_v2.cli.config_builder import build_configs

        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt_dir = str(Path(tmpdir) / "ckpts")
            ds_dir = str(Path(tmpdir) / "data")
            Path(ds_dir).mkdir()
            _make_model_config(ckpt_dir, "turbo")
            self._make_fisher_map(ds_dir)
            args = self._make_args(ds_dir, ckpt_dir, ignore_fisher_map=True)
            adapter_cfg, _train = build_configs(args)

            self.assertEqual(adapter_cfg.rank_pattern, {})
            self.assertEqual(adapter_cfg.r, 64)

    def test_lokr_skips_fisher_map(self):
        """LoKR adapter should ignore fisher_map.json (LoRA only)."""
        from acestep.training_v2.cli.config_builder import build_configs

        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt_dir = str(Path(tmpdir) / "ckpts")
            ds_dir = str(Path(tmpdir) / "data")
            Path(ds_dir).mkdir()
            _make_model_config(ckpt_dir, "turbo")
            self._make_fisher_map(ds_dir)
            args = self._make_args(
                ds_dir, ckpt_dir,
                adapter_type="lokr",
                lokr_linear_dim=64, lokr_linear_alpha=128,
                lokr_factor=-1, lokr_decompose_both=False,
                lokr_use_tucker=False, lokr_use_scalar=False,
                lokr_weight_decompose=False,
            )
            adapter_cfg, _train = build_configs(args)
            self.assertFalse(
                hasattr(adapter_cfg, "rank_pattern") and adapter_cfg.rank_pattern
            )

    def test_log_heavy_every_is_clamped_non_negative(self):
        """Negative CLI value should be clamped to 0 (disabled)."""
        from acestep.training_v2.cli.config_builder import build_configs

        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt_dir = str(Path(tmpdir) / "ckpts")
            ds_dir = str(Path(tmpdir) / "data")
            Path(ds_dir).mkdir()
            _make_model_config(ckpt_dir, "turbo")
            args = self._make_args(ds_dir, ckpt_dir, log_heavy_every=-9)
            _adapter_cfg, train_cfg = build_configs(args)
            self.assertEqual(train_cfg.log_heavy_every, 0)


if __name__ == "__main__":
    unittest.main()
