"""OFT utilities for Side-Step.

Integrates PEFT OFT (Orthogonal Fine-Tuning) adapters with the ACE-Step
decoder.  OFT constrains weight updates to orthogonal transformations,
preserving pretrained knowledge.  Experimental for audio DiT models.
"""

import types
from typing import Any, Dict, Tuple

import torch
import torch.nn as nn
from loguru import logger

from sidestep_engine.vendor.configs import OFTConfig

try:
    from peft import get_peft_model, TaskType
    from peft import OFTConfig as PeftOFTConfig

    PEFT_OFT_AVAILABLE = True
except ImportError:
    PEFT_OFT_AVAILABLE = False
    logger.warning(
        "PEFT OFT support not available. "
        "Install/upgrade with: pip install peft>=0.12.0"
    )


def check_peft_oft_available() -> bool:
    """Check if PEFT OFT is importable."""
    return PEFT_OFT_AVAILABLE


def inject_oft_into_dit(
    model: Any,
    oft_config: OFTConfig,
) -> Tuple[Any, Dict[str, Any]]:
    """Inject OFT adapters into the decoder via PEFT.

    Returns:
        Tuple of (model, info_dict).
    """
    if not PEFT_OFT_AVAILABLE:
        raise ImportError(
            "PEFT >= 0.12.0 is required for OFT training. "
            "Install with: pip install peft>=0.12.0"
        )

    decoder = model.decoder

    # Unwrap any existing PEFT/Fabric wrappers.
    while hasattr(decoder, "_forward_module"):
        decoder = decoder._forward_module
    if hasattr(decoder, "base_model"):
        base = decoder.base_model
        decoder = base.model if hasattr(base, "model") else base
    if hasattr(decoder, "model") and isinstance(decoder.model, nn.Module):
        decoder = decoder.model
    model.decoder = decoder

    # Guard enable_input_require_grads (same pattern as lora_utils).
    if hasattr(decoder, "enable_input_require_grads"):
        orig = decoder.enable_input_require_grads

        def _safe_enable(self):
            try:
                return orig()
            except NotImplementedError:
                return None

        decoder.enable_input_require_grads = types.MethodType(_safe_enable, decoder)

    if hasattr(decoder, "is_gradient_checkpointing"):
        try:
            decoder.is_gradient_checkpointing = False
        except Exception:
            pass

    peft_oft_config = PeftOFTConfig(
        r=0,
        oft_block_size=oft_config.block_size,
        target_modules=oft_config.target_modules,
        coft=oft_config.coft,
        eps=oft_config.eps,
        task_type=TaskType.FEATURE_EXTRACTION,
    )

    peft_decoder = get_peft_model(decoder, peft_oft_config)
    model.decoder = peft_decoder

    # Freeze non-OFT parameters.
    for name, param in model.named_parameters():
        if "oft_" not in name:
            param.requires_grad = False

    total_params = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    info: Dict[str, Any] = {
        "total_params": total_params,
        "trainable_params": trainable,
        "trainable_ratio": trainable / total_params if total_params > 0 else 0.0,
        "block_size": oft_config.block_size,
        "coft": oft_config.coft,
        "algo": "oft",
        "target_modules": oft_config.target_modules,
    }
    logger.info(
        f"OFT injected: {trainable:,}/{total_params:,} trainable "
        f"({info['trainable_ratio']:.2%})"
    )
    return model, info
