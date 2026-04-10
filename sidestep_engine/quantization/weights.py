"""Optimum-quanto weight quantization for ACE-Step models (ai-toolkit–aligned).

Applies in-place weight quantization to a loaded ``nn.Module`` tree, then
``freeze`` from optimum-quanto. Intended to run **after** ``from_pretrained``
and **before** PEFT LoRA injection.

Requires optional dependency: ``pip install 'side-step[quantize]'``.
"""

from __future__ import annotations

import logging
from fnmatch import fnmatch
from typing import Any, List, Optional, Union

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

# Already-quantized module class names — skip (matches ai-toolkit/util/quantize.py).
Q_MODULES = frozenset(
    {
        "QLinear",
        "QConv2d",
        "QEmbedding",
        "QBatchNorm2d",
        "QLayerNorm",
        "QConvTranspose2d",
        "QEmbeddingBag",
    }
)


class TorchAOQType:
    """Wrapper for torchao weight-only configs (same idea as toolkit ``aotype``)."""

    def __init__(self, name: str, config: object) -> None:
        self.name = name
        self.config = config


# PEFT LoRA + torchao-quantized ``nn.Linear`` (AffineQuantizedTensor) is broken in common
# peft/torchao combos: ``TorchaoLoraLinear`` requires ``get_apply_tensor_subclass`` but
# ``dispatch_torchao`` does not pass it. Training therefore supports **optimum-quanto**
# qtypes only (QLinear / Q* modules), not torchao ``quantize_`` weights.
_TORCHAO_TO_QUANTO_HINT: dict[str, str] = {
    "int8": "qint8",
    "float8": "qfloat8",
    "uint8": "qint8",
    "uint7": "qint8",
    "uint6": "qint4",
    "uint5": "qint4",
    "uint4": "qint4",
    "uint3": "qint2",
    "uint2": "qint2",
}


def _torchao_qtypes() -> dict[str, object]:
    """Lazy-import torchao quantization configs."""
    from torchao.quantization.quant_api import (
        Float8WeightOnlyConfig,
        Int8WeightOnlyConfig,
        UIntXWeightOnlyConfig,
        quantize_ as torchao_quantize_,
    )

    # noqa: F841 — quantize_ used by callers via module attribute
    _ = torchao_quantize_
    return {
        "uint2": UIntXWeightOnlyConfig(torch.uint2),
        "uint3": UIntXWeightOnlyConfig(torch.uint3),
        "uint4": UIntXWeightOnlyConfig(torch.uint4),
        "uint5": UIntXWeightOnlyConfig(torch.uint5),
        "uint6": UIntXWeightOnlyConfig(torch.uint6),
        "uint7": UIntXWeightOnlyConfig(torch.uint7),
        "uint8": UIntXWeightOnlyConfig(torch.uint8),
        "int8": Int8WeightOnlyConfig(),
        "float8": Float8WeightOnlyConfig(),
    }


def get_qtype(qtype: Union[str, object]) -> Union[object, TorchAOQType]:
    """Resolve a string qtype to optimum-quanto ``qtype`` or a torchao config wrapper.

    Quanto names (e.g. ``qfloat8``) are resolved first so torchao is not imported
    unless a torch-only key (``int8``, ``float8``, ``uint4``, …) is requested.

    For training, use :func:`apply_weight_quantization`, which rejects ``TorchAOQType``
    so only Quanto dtypes are applied before PEFT LoRA.

    Raises:
        ValueError: If the string is unknown.
    """
    from optimum.quanto.tensor import qtypes

    if not isinstance(qtype, str):
        return qtype

    if qtype in qtypes:
        return qtypes[qtype]

    tao = _torchao_qtypes()
    if qtype in tao:
        return TorchAOQType(qtype, tao[qtype])

    raise ValueError(
        f"Unknown qtype {qtype!r}. Use a quanto name (e.g. qfloat8, qint8) or "
        f"torchao keys: {sorted(tao.keys())}."
    )


def _register_qlinear_input_contiguous_hooks(model: nn.Module) -> int:
    """Force QLinear inputs to be contiguous before ``F.linear``.

    CUDA **qint4** (TinyGemm / ``_weight_int4pack_mm``) requires the activation
    matrix ``A`` to be contiguous. Attention and other blocks often pass
    strided views into ``q_proj`` / ``k_proj``; PEFT LoRA preserves that layout
    into the wrapped ``QLinear``, which then crashes inside quanto. A
    forward-pre hook fixes this with a cheap copy only when needed.
    """
    registered = 0

    def _pre_hook(_mod: nn.Module, args: tuple[Any, ...]) -> tuple[Any, ...] | None:
        if not args:
            return args
        x = args[0]
        if isinstance(x, torch.Tensor) and not x.is_contiguous():
            return (x.contiguous(),) + tuple(args[1:])
        return args

    for m in model.modules():
        if m.__class__.__name__ == "QLinear":
            m.register_forward_pre_hook(_pre_hook)
            registered += 1
    if registered:
        logger.info(
            "[quantization] registered contiguous input pre-hooks on %d QLinear module(s)",
            registered,
        )
    return registered


def normalize_qtype_string(qtype: str, *, device: str) -> str:
    """Apply MPS adjustments: qfloat8 → qint8 (quanto-native, no torchao required)."""
    q = qtype.strip()
    if torch.backends.mps.is_available() and (
        device == "mps" or device.startswith("mps:")
    ):
        if q == "qfloat8":
            logger.info("[quantization] MPS: using qint8 instead of qfloat8")
            return "qint8"
    return q


def quantize_module_tree(
    model: torch.nn.Module,
    weights: Optional[Union[str, object, TorchAOQType]] = None,
    *,
    activations: Optional[str] = None,
    include: Optional[Union[str, List[str]]] = None,
    exclude: Optional[Union[str, List[str]]] = None,
) -> None:
    """Quantize submodules in-place (ai-toolkit ``quantize`` semantics)."""
    from optimum.quanto.quantize import _quantize_submodule

    if isinstance(weights, str):
        w = get_qtype(weights)
    else:
        w = weights

    if include is not None:
        include = [include] if isinstance(include, str) else include
    if exclude is not None:
        exclude = [exclude] if isinstance(exclude, str) else exclude

    for name, m in model.named_modules():
        if include is not None and not any(fnmatch(name, p) for p in include):
            continue
        if exclude is not None and any(fnmatch(name, p) for p in exclude):
            continue
        try:
            if m.__class__.__name__ in Q_MODULES:
                continue
            if isinstance(w, TorchAOQType):
                from torchao.quantization.quant_api import quantize_ as torchao_quantize_

                torchao_quantize_(m, w.config)
            else:
                _quantize_submodule(
                    model,
                    name,
                    m,
                    weights=w,
                    activations=activations,
                    optimizer=None,
                )
        except Exception as exc:
            logger.warning("[quantization] skipped %s: %s", name, exc)


def apply_weight_quantization(
    model: torch.nn.Module,
    qtype: str,
    *,
    device_hint: str = "cuda",
) -> None:
    """Run ``quantize`` + ``freeze`` on *model* for VRAM-friendly training prep.

    Only **optimum-quanto** qtype names are supported (e.g. ``qfloat8``, ``qint8``).
    TorchAO-only keys (``int8``, ``float8``, ``uint4``, …) are rejected: they yield
    torchao tensor subclasses that trigger a broken PEFT LoRA path. Use the Quanto
    names above instead.

    After ``freeze``, registers forward-pre hooks on every ``QLinear`` so activations
    are contiguous — required for CUDA **qint4** / int4 mm with non-contiguous
    attention layouts.

    Args:
        model: Full ACE condition model (encoder + decoder), already on device.
        qtype: Quanto qtype name (e.g. ``qfloat8``, ``qint8``, ``qint4``).
        device_hint: Used for MPS qfloat8 → qint8 swap.
    """
    try:
        from optimum.quanto import freeze
    except ImportError as exc:
        raise ImportError(
            "optimum-quanto is required for weight quantization. "
            "Install with: uv pip install 'side-step[quantize]'"
        ) from exc

    resolved = normalize_qtype_string(qtype, device=device_hint)
    w = get_qtype(resolved)
    if isinstance(w, TorchAOQType):
        hint = _TORCHAO_TO_QUANTO_HINT.get(w.name, "qfloat8 or qint8")
        raise ValueError(
            f"Weight qtype {w.name!r} uses TorchAO weight-only quantization, which is not "
            f"compatible with PEFT LoRA in this release. Use an optimum-quanto name instead "
            f"(try {hint!r}, or see optimum.quanto.tensor.qtypes)."
        )
    logger.info("[quantization] applying weight quantization (qtype=%s)", resolved)
    quantize_module_tree(model, weights=w)
    freeze(model)
    _register_qlinear_input_contiguous_hooks(model)
    logger.info("[quantization] optimum.quanto.freeze() applied to model")
