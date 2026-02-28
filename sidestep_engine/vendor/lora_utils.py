"""LoRA Utilities for ACE-Step (vendored for standalone Side-Step).

Provides utilities for injecting LoRA adapters into the DiT decoder model.
Uses PEFT (Parameter-Efficient Fine-Tuning) library for LoRA implementation.
"""

import json
import os
from pathlib import Path
from typing import Optional, Dict, Any, Tuple
from loguru import logger
import types

import torch
import torch.nn as nn

try:
    from peft import (
        get_peft_model,
        LoraConfig,
        TaskType,
        PeftModel,
        PeftConfig,
    )
    PEFT_AVAILABLE = True
except ImportError:
    PEFT_AVAILABLE = False
    logger.warning("PEFT library not installed. LoRA training will not be available.")

from sidestep_engine.vendor.configs import LoRAConfig


def check_peft_available() -> bool:
    """Check if PEFT library is available."""
    return PEFT_AVAILABLE


def inject_lora_into_dit(
    model,
    lora_config: LoRAConfig,
) -> Tuple[Any, Dict[str, Any]]:
    """Inject LoRA adapters into the DiT decoder of the model.

    Args:
        model: The AceStepConditionGenerationModel
        lora_config: LoRA configuration

    Returns:
        Tuple of (peft_model, info_dict)
    """
    if not PEFT_AVAILABLE:
        raise ImportError("PEFT library is required for LoRA training. Install with: pip install peft")

    # Get the decoder (DiT model). Previous failed training runs may leave
    # Fabric/PEFT wrappers attached; unwrap to a clean base module first.
    decoder = model.decoder
    while hasattr(decoder, "_forward_module"):
        decoder = decoder._forward_module
    if hasattr(decoder, "base_model"):
        base_model = decoder.base_model
        if hasattr(base_model, "model"):
            decoder = base_model.model
        else:
            decoder = base_model
    if hasattr(decoder, "model") and isinstance(decoder.model, nn.Module):
        decoder = decoder.model
    model.decoder = decoder

    # PEFT may call enable_input_require_grads() when is_gradient_checkpointing
    # is true. AceStepDiTModel doesn't implement get_input_embeddings, so the
    # default implementation raises NotImplementedError. Guard this path.
    if hasattr(decoder, "enable_input_require_grads"):
        orig_enable_input_require_grads = decoder.enable_input_require_grads

        def _safe_enable_input_require_grads(self):
            try:
                result = orig_enable_input_require_grads()
                try:
                    self._acestep_input_grads_hook_enabled = True
                except Exception:
                    pass
                return result
            except NotImplementedError:
                try:
                    self._acestep_input_grads_hook_enabled = False
                except Exception:
                    pass
                if not getattr(self, "_acestep_input_grads_warning_emitted", False):
                    logger.info(
                        "Skipping enable_input_require_grads for decoder: "
                        "get_input_embeddings is not implemented (expected for DiT)"
                    )
                    try:
                        self._acestep_input_grads_warning_emitted = True
                    except Exception:
                        pass
                return None

        decoder.enable_input_require_grads = types.MethodType(
            _safe_enable_input_require_grads, decoder
        )

    # Avoid PEFT auto-prep path on non-embedding diffusion decoder.
    if hasattr(decoder, "is_gradient_checkpointing"):
        try:
            decoder.is_gradient_checkpointing = False
        except Exception:
            pass

    # Create PEFT LoRA config
    rank_pat = getattr(lora_config, "rank_pattern", None) or {}
    alpha_pat = getattr(lora_config, "alpha_pattern", None) or {}
    # When adaptive ranks are active, global r is the SAFETY NET (rank_min).
    # Every real module rank lives in rank_pattern.  If a module slips
    # through target matching without a pattern entry it gets the safe
    # minimum rank instead of an expensive fallback.
    if rank_pat:
        fallback_r = getattr(lora_config, "rank_min", lora_config.r)
        fallback_alpha = fallback_r * 2
    else:
        fallback_r = lora_config.r
        fallback_alpha = lora_config.alpha

    use_dora = getattr(lora_config, "use_dora", False)
    peft_lora_config = LoraConfig(
        r=fallback_r,
        lora_alpha=fallback_alpha,
        lora_dropout=lora_config.dropout,
        target_modules=lora_config.target_modules,
        rank_pattern=rank_pat,
        alpha_pattern=alpha_pat,
        bias=lora_config.bias,
        task_type=TaskType.FEATURE_EXTRACTION,
        use_dora=use_dora,
    )

    # Apply LoRA to the decoder
    peft_decoder = get_peft_model(decoder, peft_lora_config)

    # Replace the decoder in the original model
    model.decoder = peft_decoder

    # Freeze all non-LoRA parameters
    for name, param in model.named_parameters():
        if 'lora_' not in name:
            param.requires_grad = False

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    info = {
        "total_params": total_params,
        "trainable_params": trainable_params,
        "trainable_ratio": trainable_params / total_params if total_params > 0 else 0,
        "lora_r": lora_config.r,
        "lora_alpha": lora_config.alpha,
        "target_modules": lora_config.target_modules,
    }

    logger.info(f"LoRA injected into DiT decoder:")
    logger.info(f"  Total parameters: {total_params:,}")
    logger.info(f"  Trainable parameters: {trainable_params:,} ({info['trainable_ratio']:.2%})")
    logger.info(f"  LoRA rank: {lora_config.r}, alpha: {lora_config.alpha}")

    return model, info


def _unwrap_decoder(model):
    """Unwrap Fabric / PEFT wrappers to reach the real PEFT-injected decoder.

    During training, ``Fabric.setup()`` wraps the decoder in a
    ``_FabricModule``.  Calling ``save_pretrained`` on the wrapper
    instead of the real PEFT model produces empty or corrupted
    safetensors files.  This helper peels the wrapper(s) off.
    """
    decoder = model.decoder if hasattr(model, "decoder") else model
    while hasattr(decoder, "_forward_module"):
        decoder = decoder._forward_module
    return decoder


def _scrub_adapter_config(output_dir: str) -> None:
    """Remove trainer-specific paths from adapter_config.json so adapters are portable."""
    cfg_path = os.path.join(output_dir, "adapter_config.json")
    if not os.path.isfile(cfg_path):
        return
    try:
        with open(cfg_path, "r") as f:
            cfg = json.load(f)
        if cfg.get("base_model_name_or_path"):
            cfg["base_model_name_or_path"] = ""
            with open(cfg_path, "w") as f:
                json.dump(cfg, f, indent=2)
    except Exception as exc:
        logger.debug("Could not scrub adapter_config.json: %s", exc)


def save_lora_weights(
    model,
    output_dir: str,
    save_full_model: bool = False,
) -> str:
    """Save LoRA adapter weights.

    Args:
        model: Model with LoRA adapters
        output_dir: Directory to save weights
        save_full_model: Whether to save the full model state dict

    Returns:
        Path to saved weights
    """
    import sys
    if sys.platform == "win32" and len(os.path.abspath(output_dir)) > 240:
        logger.warning(
            "Output path is very long (%d chars) -- this may fail on Windows. "
            "Try a shorter output path, or enable long path support: "
            "run 'reg add HKLM\\SYSTEM\\CurrentControlSet\\Control\\FileSystem "
            "/v LongPathsEnabled /t REG_DWORD /d 1 /f' in an admin terminal "
            "and restart.",
            len(os.path.abspath(output_dir)),
        )
    os.makedirs(output_dir, exist_ok=True)

    # Unwrap Fabric wrapper so PEFT's save_pretrained sees the real model
    decoder = _unwrap_decoder(model)

    if hasattr(decoder, 'save_pretrained'):
        decoder.save_pretrained(output_dir)
        _scrub_adapter_config(output_dir)
        logger.info(f"LoRA adapter saved to {output_dir}")
        return output_dir
    elif save_full_model:
        # Save full model state dict (larger file)
        model_path = str(Path(output_dir) / "model.pt")
        torch.save(decoder.state_dict(), model_path)
        logger.info(f"Full model state dict saved to {model_path}")
        return model_path
    else:
        # Extract only LoRA parameters
        lora_state_dict = {}
        for name, param in decoder.named_parameters():
            if 'lora_' in name:
                lora_state_dict[name] = param.data.clone()

        if not lora_state_dict:
            logger.warning("No LoRA parameters found to save!")
            return ""

        lora_path = str(Path(output_dir) / "lora_weights.pt")
        torch.save(lora_state_dict, lora_path)
        logger.info(f"LoRA weights saved to {lora_path}")
        return lora_path


def load_lora_weights(
    model,
    lora_path: str,
    lora_config: Optional[LoRAConfig] = None,
) -> Any:
    """Load LoRA adapter weights into the model.

    Args:
        model: The base model (without LoRA)
        lora_path: Path to saved LoRA weights (adapter or .pt file)
        lora_config: LoRA configuration (required if loading from .pt file)

    Returns:
        Model with LoRA weights loaded
    """
    if not os.path.exists(lora_path):
        raise FileNotFoundError(f"LoRA weights not found: {lora_path}")

    # Check if it's a PEFT adapter directory
    if os.path.isdir(lora_path):
        if not PEFT_AVAILABLE:
            raise ImportError("PEFT library is required to load adapter. Install with: pip install peft")

        # Load PEFT adapter
        peft_config = PeftConfig.from_pretrained(lora_path)
        model.decoder = PeftModel.from_pretrained(model.decoder, lora_path)
        logger.info(f"LoRA adapter loaded from {lora_path}")

    elif lora_path.endswith('.pt'):
        # Load from PyTorch state dict
        if lora_config is None:
            raise ValueError("lora_config is required when loading from .pt file")

        # First inject LoRA structure
        model, _ = inject_lora_into_dit(model, lora_config)

        # Load weights
        lora_state_dict = torch.load(lora_path, map_location='cpu', weights_only=True)

        # Load into model
        model_state = model.state_dict()
        for name, param in lora_state_dict.items():
            if name in model_state:
                model_state[name].copy_(param)
            else:
                logger.warning(f"Unexpected key in LoRA state dict: {name}")

        logger.info(f"LoRA weights loaded from {lora_path}")

    else:
        raise ValueError(f"Unsupported LoRA weight format: {lora_path}")

    return model


def save_training_checkpoint(
    model,
    optimizer,
    scheduler,
    epoch: int,
    global_step: int,
    output_dir: str,
) -> str:
    """Save a training checkpoint including LoRA weights and training state.

    Args:
        model: Model with LoRA adapters
        optimizer: Optimizer state
        scheduler: Scheduler state
        epoch: Current epoch number
        global_step: Current global step
        output_dir: Directory to save checkpoint

    Returns:
        Path to saved checkpoint directory
    """
    os.makedirs(output_dir, exist_ok=True)

    # Save LoRA adapter weights
    adapter_path = save_lora_weights(model, output_dir)

    # Save training state (optimizer, scheduler, epoch, step)
    training_state = {
        "epoch": epoch,
        "global_step": global_step,
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
    }

    state_path = str(Path(output_dir) / "training_state.pt")
    torch.save(training_state, state_path)

    logger.info(f"Training checkpoint saved to {output_dir} (epoch {epoch}, step {global_step})")
    return output_dir


def load_training_checkpoint(
    checkpoint_dir: str,
    optimizer=None,
    scheduler=None,
    device: torch.device = None,
) -> Dict[str, Any]:
    """Load training checkpoint.

    Args:
        checkpoint_dir: Directory containing checkpoint files
        optimizer: Optimizer instance to load state into (optional)
        scheduler: Scheduler instance to load state into (optional)
        device: Device to load tensors to

    Returns:
        Dictionary with checkpoint info.
    """
    result = {
        "epoch": 0,
        "global_step": 0,
        "adapter_path": None,
        "loaded_optimizer": False,
        "loaded_scheduler": False,
    }

    # Normalize: if user pointed to a file inside the checkpoint dir,
    # use the containing directory instead.
    if os.path.isfile(checkpoint_dir):
        checkpoint_dir = os.path.dirname(checkpoint_dir)

    # Find adapter path
    adapter_path = str(Path(checkpoint_dir) / "adapter")
    if os.path.exists(adapter_path):
        result["adapter_path"] = adapter_path
    elif os.path.exists(checkpoint_dir):
        result["adapter_path"] = checkpoint_dir

    # Load training state
    state_path = str(Path(checkpoint_dir) / "training_state.pt")
    if os.path.exists(state_path):
        map_location = device if device else "cpu"
        training_state = torch.load(state_path, map_location=map_location, weights_only=False)

        result["epoch"] = training_state.get("epoch", 0)
        result["global_step"] = training_state.get("global_step", 0)

        if optimizer is not None and "optimizer_state_dict" in training_state:
            try:
                optimizer.load_state_dict(training_state["optimizer_state_dict"])
                result["loaded_optimizer"] = True
                logger.info("Optimizer state loaded from checkpoint")
            except Exception as e:
                logger.warning(f"Failed to load optimizer state: {e}")

        if scheduler is not None and "scheduler_state_dict" in training_state:
            try:
                scheduler.load_state_dict(training_state["scheduler_state_dict"])
                result["loaded_scheduler"] = True
                logger.info("Scheduler state loaded from checkpoint")
            except Exception as e:
                logger.warning(f"Failed to load scheduler state: {e}")

        logger.info(f"Loaded checkpoint from epoch {result['epoch']}, step {result['global_step']}")
    else:
        # Fallback: extract epoch from path
        import re
        match = re.search(r'epoch_(\d+)', checkpoint_dir)
        if match:
            result["epoch"] = int(match.group(1))
            logger.info(f"No training_state.pt found, extracted epoch {result['epoch']} from path")

    return result


