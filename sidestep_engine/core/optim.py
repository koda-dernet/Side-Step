"""
Side-Step Optimizer & Scheduler Factories

Provides ``build_optimizer()`` and ``build_scheduler()`` so that
``trainer_fixed.py`` doesn't need to hard-code AdamW / CosineAnnealing.

Supported optimizers:
    adamw       -- torch.optim.AdamW (default, fused on CUDA)
    adamw8bit   -- bitsandbytes.optim.AdamW8bit (optional dep)
    adafactor   -- transformers.optimization.Adafactor
    prodigy     -- prodigyopt.Prodigy (optional dep, auto-tunes LR)

Supported schedulers:
    cosine              -- warmup + CosineAnnealingLR (single smooth decay)
    cosine_restarts     -- warmup + CosineAnnealingWarmRestarts (cyclical)
    linear              -- warmup + LinearLR decay to near-zero
    constant            -- warmup then flat LR
    constant_with_warmup -- alias for constant
"""

from __future__ import annotations

import inspect
import logging
import math
from typing import Dict, Iterable, List, Optional, Union

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    CosineAnnealingWarmRestarts,
    ConstantLR,
    LinearLR,
    SequentialLR,
)

logger = logging.getLogger(__name__)

_PRODIGY_DEFAULT_LR = 1.0


def _sanitize_scalar_hparams(lr: float, weight_decay: float) -> tuple[float, float]:
    """Sanitize optimizer scalar hyperparameters to safe finite values."""
    safe_lr = float(lr)
    safe_wd = float(weight_decay)

    if not math.isfinite(safe_lr) or safe_lr <= 0.0:
        logger.warning(
            "[Side-Step] Invalid learning rate (%s) -- using 1e-4",
            lr,
        )
        safe_lr = 1e-4

    if not math.isfinite(safe_wd) or safe_wd < 0.0:
        logger.warning(
            "[Side-Step] Invalid weight decay (%s) -- using 0.01",
            weight_decay,
        )
        safe_wd = 0.01

    return safe_lr, safe_wd


def _build_prodigy_kwargs(
    prodigy_cls,
    params: Iterable,
    lr: float,
    weight_decay: float,
) -> dict:
    """Build kwargs for prodigy with optional stability flags if supported."""
    kwargs = {
        "params": params,
        "lr": lr,
        "weight_decay": weight_decay,
    }
    try:
        sig = inspect.signature(prodigy_cls.__init__)
    except Exception:
        return kwargs

    optional_safe_args = {
        "safeguard_warmup": True,
        "use_bias_correction": True,
        "betas": (0.9, 0.99),
        "eps": 1e-8,
    }
    for key, value in optional_safe_args.items():
        if key in sig.parameters:
            kwargs[key] = value
    return kwargs


# ---------------------------------------------------------------------------
# Per-layer-type param classification
# ---------------------------------------------------------------------------

_MLP_SUFFIXES = ("gate_proj", "up_proj", "down_proj")


def classify_trainable_params(
    model: nn.Module,
    lycoris_net: Optional[nn.Module] = None,
) -> Dict[str, List[torch.nn.Parameter]]:
    """Classify trainable parameters into self_attn, cross_attn, and mlp groups.

    Classification rules (matching ``fisher/modules.py`` conventions):
      - MLP: parameter name contains a suffix in ``_MLP_SUFFIXES``
      - Cross-attention: ``.cross_attn.`` appears in the parameter name
      - Self-attention: everything else

    Falls back to the LyCORIS network when the model yields no trainable
    params (same as ``_collect_trainable_params`` in ``trainer.py``).

    Returns:
        ``{"self_attn": [...], "cross_attn": [...], "mlp": [...]}``
        Each list contains unique ``nn.Parameter`` objects.
    """
    groups: Dict[str, Dict[int, torch.nn.Parameter]] = {
        "self_attn": {},
        "cross_attn": {},
        "mlp": {},
    }

    def _classify(name: str, param: torch.nn.Parameter) -> None:
        if not param.requires_grad:
            return
        pid = id(param)
        short = name.rsplit(".", 1)[-1] if "." in name else name
        if short in _MLP_SUFFIXES:
            groups["mlp"][pid] = param
        elif ".cross_attn." in name:
            groups["cross_attn"][pid] = param
        else:
            groups["self_attn"][pid] = param

    found_any = False
    for name, param in model.named_parameters():
        if param.requires_grad:
            found_any = True
            _classify(name, param)

    if not found_any and lycoris_net is not None:
        for m in getattr(lycoris_net, "loras", []) or []:
            for name, param in m.named_parameters():
                _classify(name, param)
        if not any(groups[k] for k in groups):
            for name, param in lycoris_net.named_parameters():
                _classify(name, param)

    return {k: list(v.values()) for k, v in groups.items()}


def build_layer_param_groups(
    classified: Dict[str, List[torch.nn.Parameter]],
    base_lr: float,
    lr_scale_self_attn: float = 1.0,
    lr_scale_cross_attn: float = 1.0,
    lr_scale_mlp: float = 1.0,
) -> Union[List[dict], List[torch.nn.Parameter]]:
    """Build optimizer param groups with per-layer-type LR scaling.

    When all scales are 1.0, returns a flat parameter list (identical to
    the old code path -- no param-group overhead in the optimizer).

    Args:
        classified: Output of :func:`classify_trainable_params`.
        base_lr: Base learning rate.
        lr_scale_self_attn: Multiplier for self-attention params.
        lr_scale_cross_attn: Multiplier for cross-attention params.
        lr_scale_mlp: Multiplier for MLP/FFN params.

    Returns:
        Either a flat param list or a list of param-group dicts.
    """
    all_default = (
        abs(lr_scale_self_attn - 1.0) < 1e-9
        and abs(lr_scale_cross_attn - 1.0) < 1e-9
        and abs(lr_scale_mlp - 1.0) < 1e-9
    )

    if all_default:
        # Flat list -- identical to previous behavior
        flat: List[torch.nn.Parameter] = []
        for params in classified.values():
            flat.extend(params)
        return flat

    groups: List[dict] = []
    _scales = {
        "self_attn": lr_scale_self_attn,
        "cross_attn": lr_scale_cross_attn,
        "mlp": lr_scale_mlp,
    }
    for group_name, scale in _scales.items():
        params = classified.get(group_name, [])
        if params:
            groups.append({
                "params": params,
                "lr": base_lr * scale,
                "_group_name": group_name,  # metadata (ignored by optimizer)
            })

    if groups:
        _summary = ", ".join(
            f"{g['_group_name']}={len(g['params'])}p@{g['lr']:.2e}"
            for g in groups
        )
        logger.info("[Side-Step] Per-layer LR groups: %s", _summary)

    return groups


# ---------------------------------------------------------------------------
# Optimizer factory
# ---------------------------------------------------------------------------

def build_optimizer(
    params: Iterable,
    optimizer_type: str = "adamw",
    lr: float = 1e-4,
    weight_decay: float = 0.01,
    device_type: str = "cuda",
) -> torch.optim.Optimizer:
    """Create an optimizer from a string key.

    ``params`` can be a flat iterable of parameters *or* a list of
    param-group dicts (as returned by :func:`build_layer_param_groups`).
    When param-group dicts are provided, each group keeps its own ``lr``;
    the *lr* argument is used only as a fallback for groups that don't
    specify one.

    Falls back to AdamW when an optional dependency is missing.
    """
    optimizer_type = optimizer_type.lower().strip()
    lr, weight_decay = _sanitize_scalar_hparams(lr, weight_decay)

    if optimizer_type == "adamw8bit":
        try:
            from bitsandbytes.optim import AdamW8bit
            logger.info("[Side-Step] Using AdamW8bit optimizer (lower VRAM)")
            return AdamW8bit(params, lr=lr, weight_decay=weight_decay)
        except ImportError:
            logger.warning(
                "[Side-Step] bitsandbytes not installed -- falling back to AdamW. "
                "Install with: pip install bitsandbytes>=0.45.0"
            )
            optimizer_type = "adamw"

    if optimizer_type == "adafactor":
        try:
            from transformers.optimization import Adafactor
            logger.info("[Side-Step] Using Adafactor optimizer (minimal state memory)")
            return Adafactor(
                params,
                lr=lr,
                weight_decay=weight_decay,
                scale_parameter=False,
                relative_step=False,
            )
        except ImportError:
            logger.warning(
                "[Side-Step] transformers not installed -- falling back to AdamW"
            )
            optimizer_type = "adamw"

    if optimizer_type == "prodigy":
        try:
            from prodigyopt import Prodigy
            requested_lr = lr
            if abs(lr - 1e-4) < 1e-12:
                lr = _PRODIGY_DEFAULT_LR

            logger.info(
                "[Side-Step] Using Prodigy optimizer (requested_lr=%.6f, effective_lr=%.6f, wd=%.6f)",
                requested_lr,
                lr,
                weight_decay,
            )
            return Prodigy(**_build_prodigy_kwargs(Prodigy, params, lr, weight_decay))
        except ImportError:
            logger.warning(
                "[Side-Step] prodigyopt not installed -- falling back to AdamW. "
                "Install with: pip install prodigyopt>=1.1.2"
            )
            optimizer_type = "adamw"

    # Default: AdamW
    kwargs = {"lr": lr, "weight_decay": weight_decay}
    if device_type == "cuda":
        kwargs["fused"] = True
    logger.info("[Side-Step] Using AdamW optimizer")
    return AdamW(params, **kwargs)


# ---------------------------------------------------------------------------
# Scheduler factory
# ---------------------------------------------------------------------------

def build_scheduler(
    optimizer: torch.optim.Optimizer,
    scheduler_type: str = "cosine",
    total_steps: int = 1000,
    warmup_steps: int = 500,
    lr: float = 1e-4,
    optimizer_type: str = "adamw",
    n_restarts: int = 4,
    formula: str = "",
    steps_per_epoch: int = 100,
    total_epochs: int = 10,
    warmup_start_factor: float = 0.1,
    cosine_eta_min_ratio: float = 0.01,
):
    """Create a learning rate scheduler from a string key.

    When the optimizer is Prodigy, defaults to constant schedule
    (Prodigy manages LR internally).  Custom formulas are blocked
    for Prodigy.

    Args:
        n_restarts: Number of cosine restart cycles for the
            ``cosine_restarts`` scheduler.  Ignored by other types.
        formula: Python math expression for the ``custom`` scheduler.
        steps_per_epoch: Steps per epoch (passed to custom formula).
        total_epochs: Total training epochs (passed to custom formula).
    """
    scheduler_type = scheduler_type.lower().strip()
    optimizer_type = optimizer_type.lower().strip()

    # Prodigy handles its own LR -- custom formulas are incompatible
    if optimizer_type == "prodigy" and scheduler_type == "custom":
        raise ValueError(
            "Prodigy optimizer manages LR internally -- "
            "custom LR formulas are not compatible. "
            "Use a different optimizer or a built-in scheduler."
        )
    if optimizer_type == "prodigy" and scheduler_type not in ("constant", "constant_with_warmup"):
        logger.info(
            "[Side-Step] Prodigy optimizer detected -- overriding scheduler to 'constant' "
            "(Prodigy adapts LR internally)"
        )
        scheduler_type = "constant"

    # Clamp warmup to avoid exceeding total
    _max_warmup = max(1, total_steps // 10)
    if warmup_steps > _max_warmup:
        logger.warning(
            "[Side-Step] Warmup steps (%d) clamped to %d (10%% of %d total steps)",
            warmup_steps, _max_warmup, total_steps,
        )
        warmup_steps = _max_warmup

    warmup_sched = LinearLR(
        optimizer,
        start_factor=warmup_start_factor,
        end_factor=1.0,
        total_iters=warmup_steps,
    )

    remaining = max(1, total_steps - warmup_steps)

    if scheduler_type in ("constant", "constant_with_warmup"):
        main_sched = ConstantLR(optimizer, factor=1.0, total_iters=total_steps)
    elif scheduler_type == "linear":
        main_sched = LinearLR(
            optimizer,
            start_factor=1.0,
            end_factor=0.01,
            total_iters=remaining,
        )
    elif scheduler_type == "custom":
        if not formula or not formula.strip():
            raise ValueError(
                "scheduler_type is 'custom' but no --scheduler-formula was provided. "
                "Either set --scheduler-formula or use a built-in scheduler type."
            )
        from sidestep_engine.core.formula_scheduler import build_formula_scheduler
        return build_formula_scheduler(
            optimizer,
            formula=formula,
            base_lr=lr,
            total_steps=total_steps,
            warmup_steps=warmup_steps,
            steps_per_epoch=steps_per_epoch,
            total_epochs=total_epochs,
            warmup_start_factor=warmup_start_factor,
        )
    elif scheduler_type == "cosine_restarts":
        # Cyclical cosine: LR resets to peak multiple times during training.
        # T_0 = cycle length = remaining / n_restarts.
        main_sched = CosineAnnealingWarmRestarts(
            optimizer,
            T_0=max(1, remaining // max(1, n_restarts)),
            T_mult=1,
            eta_min=lr * cosine_eta_min_ratio,
        )
    else:
        # cosine (default) -- single smooth decay to eta_min, no restarts.
        main_sched = CosineAnnealingLR(
            optimizer,
            T_max=remaining,
            eta_min=lr * cosine_eta_min_ratio,
        )

    return SequentialLR(optimizer, [warmup_sched, main_sched], milestones=[warmup_steps])
