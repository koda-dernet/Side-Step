"""LoHA utilities for Side-Step.

Integrates LyCORIS LoHA (Low-Rank Hadamard Product) adapters with the
ACE-Step decoder.  Mirrors ``lokr_utils.py`` but uses ``algo="loha"``.
"""

import json
import os
from typing import Any, Dict, Optional, Tuple

import torch
from loguru import logger

from sidestep_engine.vendor.configs import LoHAConfig

try:
    from lycoris import LycorisNetwork, create_lycoris

    LYCORIS_AVAILABLE = True
except ImportError:
    LYCORIS_AVAILABLE = False
    LycorisNetwork = Any  # type: ignore[assignment,misc]


def check_lycoris_available() -> bool:
    """Check if LyCORIS is importable."""
    return LYCORIS_AVAILABLE


def inject_loha_into_dit(
    model: Any,
    loha_config: LoHAConfig,
    multiplier: float = 1.0,
) -> Tuple[Any, "LycorisNetwork", Dict[str, Any]]:
    """Inject LoHA adapters into the decoder.

    Returns:
        Tuple of (model, lycoris_network, info_dict).
    """
    if not LYCORIS_AVAILABLE:
        raise ImportError(
            "LyCORIS library is required for LoHA training. "
            "Install with: pip install lycoris-lora"
        )

    decoder = model.decoder

    for _, param in model.named_parameters():
        param.requires_grad = False

    LycorisNetwork.apply_preset(
        {
            "unet_target_name": loha_config.target_modules,
            "target_name": loha_config.target_modules,
        }
    )

    lycoris_net = create_lycoris(
        decoder,
        multiplier,
        linear_dim=loha_config.linear_dim,
        linear_alpha=loha_config.linear_alpha,
        algo="loha",
        factor=loha_config.factor,
        use_tucker=loha_config.use_tucker,
        use_scalar=loha_config.use_scalar,
        full_matrix=loha_config.full_matrix,
        bypass_mode=loha_config.bypass_mode,
        rs_lora=loha_config.rs_lora,
    )

    lycoris_net.apply_to()
    decoder._lycoris_net = lycoris_net

    # LyCORIS apply_preset + create_lycoris already handles target-module
    # selection.  Do NOT re-filter by name matching here — that can
    # accidentally freeze valid adapter tensors, causing silent quality
    # regression with many all-zero saved tensors.
    param_list: list[torch.nn.Parameter] = []
    for module in getattr(lycoris_net, "loras", []) or []:
        for p in module.parameters():
            p.requires_grad = True
            param_list.append(p)

    if not param_list:
        for p in lycoris_net.parameters():
            p.requires_grad = True
            param_list.append(p)

    unique = {id(p): p for p in param_list}
    total_params = sum(p.numel() for p in model.parameters())
    loha_params = sum(p.numel() for p in unique.values())
    trainable = sum(p.numel() for p in unique.values() if p.requires_grad)

    info: Dict[str, Any] = {
        "total_params": total_params,
        "loha_params": loha_params,
        "trainable_params": trainable,
        "trainable_ratio": trainable / total_params if total_params > 0 else 0.0,
        "linear_dim": loha_config.linear_dim,
        "linear_alpha": loha_config.linear_alpha,
        "algo": "loha",
        "target_modules": loha_config.target_modules,
    }
    logger.info(
        f"LoHA injected: {trainable:,}/{total_params:,} trainable "
        f"({info['trainable_ratio']:.2%})"
    )
    return model, lycoris_net, info


def save_loha_weights(
    lycoris_net: "LycorisNetwork",
    output_dir: str,
    dtype: Optional[torch.dtype] = None,
    metadata: Optional[Dict[str, Any]] = None,
) -> str:
    """Save LoHA weights to safetensors."""
    os.makedirs(output_dir, exist_ok=True)
    weights_path = os.path.join(output_dir, "loha_weights.safetensors")

    save_meta: Dict[str, str] = {"algo": "loha", "format": "lycoris"}
    if metadata:
        for k, v in metadata.items():
            if v is None:
                continue
            save_meta[k] = v if isinstance(v, str) else json.dumps(v, ensure_ascii=True)

    lycoris_net.save_weights(weights_path, dtype=dtype, metadata=save_meta)
    logger.info(f"LoHA weights saved to {weights_path}")
    return weights_path


def load_loha_weights(lycoris_net: "LycorisNetwork", weights_path: str) -> Dict[str, Any]:
    """Load LoHA weights into an injected LyCORIS network."""
    if not os.path.exists(weights_path):
        raise FileNotFoundError(f"LoHA weights not found: {weights_path}")
    result = lycoris_net.load_weights(weights_path)
    logger.info(f"LoHA weights loaded from {weights_path}")
    return result
