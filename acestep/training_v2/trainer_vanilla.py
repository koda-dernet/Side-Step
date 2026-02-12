"""
VanillaTrainer -- Thin adapter wrapping the original LoRATrainer for TUI use.

The original ``acestep/training/trainer.py`` ``LoRATrainer`` requires a
``dit_handler`` shim.  This module provides a ``VanillaTrainer`` class with
the same interface as ``FixedTrainer`` so both can be used interchangeably
from the TUI training monitor.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any, Dict, Generator, Optional

import torch

from acestep.training.configs import LoRAConfig, TrainingConfig
from acestep.training.trainer import LoRATrainer
from acestep.training_v2.model_loader import load_decoder_for_training

logger = logging.getLogger(__name__)


class _HandlerShim:
    """Minimal shim satisfying the ``LoRATrainer`` constructor."""

    def __init__(self, model: torch.nn.Module, device: str, dtype: torch.dtype) -> None:
        self.model = model
        self.device = device
        self.dtype = dtype
        self.quantization = None


class VanillaTrainer:
    """Adapter that wraps the upstream ``LoRATrainer`` to match FixedTrainer's interface."""

    def __init__(
        self,
        lora_config,
        training_config,
        progress_callback=None,
    ) -> None:
        self.lora_config = lora_config
        self.training_config = training_config
        self.progress_callback = progress_callback

    def train(self) -> None:
        """Run vanilla training, calling progress_callback for each update."""
        cfg = self.training_config

        # Map V2 config fields to base LoRAConfig / TrainingConfig
        lora_cfg = LoRAConfig(
            r=getattr(self.lora_config, "rank", 64),
            alpha=getattr(self.lora_config, "alpha", 128),
            dropout=getattr(self.lora_config, "dropout", 0.0),
            target_modules=getattr(self.lora_config, "target_modules", ["to_q", "to_k", "to_v", "to_out.0"]),
            bias=getattr(self.lora_config, "bias", "none"),
        )
        # Windows uses spawn-based multiprocessing which breaks DataLoader workers
        num_workers = getattr(cfg, "num_workers", 4)
        if sys.platform == "win32" and num_workers > 0:
            logger.info("[Side-Step] Windows detected -- setting num_workers=0 (spawn incompatible)")
            num_workers = 0

        train_cfg = TrainingConfig(
            learning_rate=getattr(cfg, "learning_rate", 1e-4),
            batch_size=getattr(cfg, "batch_size", 1),
            gradient_accumulation_steps=getattr(cfg, "gradient_accumulation_steps", 4),
            max_epochs=getattr(cfg, "max_epochs", getattr(cfg, "epochs", 100)),
            warmup_steps=getattr(cfg, "warmup_steps", 500),
            weight_decay=getattr(cfg, "weight_decay", 0.01),
            max_grad_norm=getattr(cfg, "max_grad_norm", 1.0),
            seed=getattr(cfg, "seed", 42),
            output_dir=getattr(cfg, "output_dir", "./lora_output"),
            save_every_n_epochs=getattr(cfg, "save_every_n_epochs", 10),
            num_workers=num_workers,
            pin_memory=getattr(cfg, "pin_memory", True),
        )

        # Resolve device / precision
        from acestep.training_v2.gpu_utils import detect_gpu

        gpu = detect_gpu(
            requested_device=getattr(cfg, "device", "auto"),
            requested_precision=getattr(cfg, "precision", "auto"),
        )
        dtype_map = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}

        model = load_decoder_for_training(
            checkpoint_dir=getattr(cfg, "checkpoint_dir", "./checkpoints"),
            variant=getattr(cfg, "model_variant", getattr(cfg, "variant", "turbo")),
            device=gpu.device,
            precision=gpu.precision,
        )

        handler = _HandlerShim(
            model=model,
            device=gpu.device,
            dtype=dtype_map.get(gpu.precision, torch.bfloat16),
        )

        trainer = LoRATrainer(handler, lora_cfg, train_cfg)
        dataset_dir = getattr(cfg, "dataset_dir", "")
        resume_from = getattr(cfg, "resume_from", None)

        for update in trainer.train_from_preprocessed(
            tensor_dir=dataset_dir,
            resume_from=resume_from,
        ):
            if self.progress_callback:
                # Upstream yields (step, loss, msg) tuples
                step, loss, msg = update if len(update) == 3 else (update[0], update[1], "")
                should_continue = self.progress_callback(
                    epoch=0, step=step, loss=loss, lr=0.0, is_epoch_end=False,
                )
                if should_continue is False:
                    break
