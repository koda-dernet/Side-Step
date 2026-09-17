"""Rose: Range-Of-Slice Equilibration optimizer (vendored).

Vendored from https://github.com/MatthewK78/Rose (commit 2026-04-22).

Rose is a stateless optimizer that rescales gradients using a per-slice
``|max| - min`` range.  Unlike Adam, it maintains **zero per-parameter
state** between steps: no momentum, no variance, no step counters.

This means optimizer VRAM usage is effectively zero — all GPU memory
goes to model weights, gradients, and activations.

Original author: Matthew Everet Kieren
Licensed under the Apache License, Version 2.0.
"""

import torch


class Rose(torch.optim.Optimizer):
    """Rose: Range-Of-Slice Equilibration optimizer.
    In loving memory of my mother, Rose Kieren.

    Copyright 2026 Matthew Everet Kieren. All Rights Reserved.
    Licensed under the Apache License, Version 2.0.

    Rose rescales gradients using a per-slice `|max| - min` range
    computed by reducing all dimensions beyond the leading axis.
    Unlike Adam and other stateful optimizers, Rose maintains no
    per-parameter state between steps: no momentum buffers,
    variance estimates, or step counters.

    Args:
        params (iterable):
            Iterable of model parameters or parameter-group dictionaries.

        lr (float):
            The global step size. Because this optimizer uses range-based
            normalization rather than Adam's RMS-based normalization, the
            same `lr` value can correspond to very different effective
            update sizes. Tune `lr` independently rather than relying on
            Adam defaults.

        weight_decay (float or None, optional) [1e-4]:
            A decoupled multiplicative weight-decay coefficient applied
            separately from the adaptive gradient step.

        wd_schedule (bool or float, optional) [False]:
            Scales weight decay proportionally with the learning-rate
            schedule so that decay weakens as the learning rate drops.

        centralize (bool, optional) [True]:
            Removes shared offsets from gradient slices before the range
            computation.

        stabilize (bool, optional) [True]:
            Computes a trust factor from the coefficient of variation of
            the per-slice range tensor.

        bf16_sr (bool or torch.Generator, optional) [True]:
            Improves BF16 training by using stochastic rounding.

        compute_dtype (torch.dtype, str, or None, optional) [fp64]:
            Promotes parameters and gradients to this dtype for the
            update step, then casts them back on write-back.
    """
    def __init__(
        self,
        params,
        lr: float,
        *,
        weight_decay: float | None = 1e-4,
        wd_schedule: bool | float = False,
        centralize: bool = True,
        stabilize: bool = True,
        bf16_sr: bool | torch.Generator = True,
        compute_dtype: torch.dtype | str | None = "fp64"
    ):
        if lr < 0.0:
            raise ValueError(f"\nInvalid learning rate: {lr}") from None
        if weight_decay is not None and weight_decay < 0.0:
            raise ValueError(f"\nInvalid weight_decay: {weight_decay}") from None

        if isinstance(bf16_sr, torch.Generator):
            self.bf16_sr_gen = bf16_sr
            bf16_sr = True
        else:
            self.bf16_sr_gen = None

        if isinstance(compute_dtype, str):
            dtype_lookup: dict[str, torch.dtype | None] = {
                "float16": torch.float16, "fp16": torch.float16,
                "float32": torch.float32, "fp32": torch.float32,
                "float64": torch.float64, "fp64": torch.float64,
                "bfloat16": torch.bfloat16, "bf16": torch.bfloat16,
                "none": None, "null": None
            }
            try:
                compute_dtype = dtype_lookup[compute_dtype.strip().lower()]
            except KeyError:
                raise ValueError(
                    f"\nInvalid compute_dtype string: {compute_dtype!r}.\n"
                    f"Valid options: {sorted(dtype_lookup)}"
                ) from None

        if bf16_sr and compute_dtype not in (torch.float32, torch.float64, None):
            raise ValueError(
                f"\nbf16_sr=True has no useful effect when compute_dtype is {compute_dtype}.\n"
                f"Use torch.float32, torch.float64, or None (same as fp32) instead."
            ) from None

        defaults = dict(
            lr=lr,
            centralize=centralize,
            stabilize=stabilize,
            weight_decay=weight_decay,
            wd_schedule=wd_schedule,
            bf16_sr=bf16_sr,
            compute_dtype=compute_dtype
        )
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None) -> torch.Tensor | None:
        """Perform a single Rose optimization step."""
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            use_stabilize = group["stabilize"]
            use_centralize = group["centralize"]
            bf16_sr = group["bf16_sr"]
            compute_dtype = group["compute_dtype"]

            # --- Decoupled Weight Decay Factor ---
            weight_decay = group["weight_decay"]
            wd_schedule = group["wd_schedule"]
            group.setdefault("initial_lr", lr)

            if weight_decay and wd_schedule:
                wd_lr = lr / (
                    wd_schedule if isinstance(wd_schedule, float)
                    else group.get("max_lr", group.get("initial_lr"))
                )
            else:
                wd_lr = lr

            wd_factor = None if not weight_decay else max(0.0, 1.0 - wd_lr * weight_decay)

            for p in group["params"]:
                if p.grad is None:
                    continue
                if p.grad.is_sparse:
                    raise RuntimeError("Rose does not support sparse gradients")

                # --- Precision Handling ---
                use_bf16_sr = bf16_sr and p.dtype is torch.bfloat16
                fp32 = use_bf16_sr and not compute_dtype
                grad = p.grad.to(dtype=torch.float32 if fp32 else compute_dtype)
                param = p.to(dtype=torch.float32 if fp32 else compute_dtype)

                # --- Decoupled Multiplicative Weight Decay ---
                if wd_factor is not None:
                    param.mul_(wd_factor)

                if grad.ndim == 0:
                    # --- 0D Scalar ---
                    param.add_(grad.sign(), alpha=-lr)

                elif grad.ndim == 1:
                    # --- Vectors / Degenerate Slices ---
                    g_min, g_max = grad.aminmax()
                    denom = g_max.abs_().sub_(g_min)
                    denom.masked_fill_(denom == 0.0, 1.0)
                    param.addcdiv_(grad, denom, value=-lr)

                else:
                    # --- Active Axes: all axes except the first ---
                    active_axes = tuple(range(1, grad.ndim))

                    # --- Gradient Centralization ---
                    if use_centralize:
                        if grad is not p.grad:
                            grad.sub_(grad.mean(dim=active_axes, keepdim=True))
                        else:
                            grad = grad.sub(grad.mean(dim=active_axes, keepdim=True))

                    # --- Per-slice Range: `R = |max(g)| - min(g)` ---
                    raw_scale = (
                        grad.amax(dim=active_axes, keepdim=True).abs_()
                        .sub_(grad.amin(dim=active_axes, keepdim=True))
                    )

                    if use_stabilize:
                        # --- Coefficient-of-Variation Trust Gating ---
                        std, mean = torch.std_mean(raw_scale, correction=0)
                        trust = mean.div(std.add_(mean).masked_fill_(mean == 0.0, 1.0))
                        denom = mean.lerp(raw_scale, trust)
                    else:
                        denom = raw_scale

                    # --- Update: θ -= lr · g / D(g) ---
                    denom.masked_fill_(denom == 0.0, 1.0)  # SGD fallback
                    param.addcdiv_(grad, denom, value=-lr)

                if use_bf16_sr:
                    # --- BF16 stochastic rounding ---
                    param = param.to(dtype=torch.float32)
                    p.copy_(
                        torch.empty_like(p, dtype=torch.int32)
                        .random_(0, 0x10000, generator=self.bf16_sr_gen)
                        .add_(param.view(dtype=torch.int32))
                        .bitwise_and_(-0x10000)
                        .view(dtype=torch.float32)
                    )

                elif param is not p:
                    p.copy_(param)

        return loss
