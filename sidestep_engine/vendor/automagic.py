"""Automagic: Per-element adaptive learning rate optimizer (vendored).

Vendored from ostris/ai-toolkit (Apache 2.0 License).
Source: https://huggingface.co/spaces/rahul7star/ai-toolkit
Original author: ostris (https://github.com/ostris/ai-toolkit)

Automagic maintains a per-element learning rate mask that bumps up when
gradient signs agree between steps, and down when they disagree.  The
LR mask is compressed to int8 via Auto8bitTensor to save memory.

Uses Adafactor-style row/column factored second moments for 2D+ params.

Side-Step adaptations:
  - QBytesTensor (optimum.quanto) made optional — not needed for bf16 LoRA
  - Parameter swapping disabled by default
  - Merged optimizer_utils.py inline to eliminate external dependency
"""

from __future__ import annotations

import random
from typing import List, Optional

import torch
from torch import Tensor


# ---------------------------------------------------------------------------
# QBytesTensor — optional, only needed for quanto-quantized models
# ---------------------------------------------------------------------------
try:
    from optimum.quanto import QBytesTensor as _QBytesTensor
except ImportError:
    _QBytesTensor = None  # Side-Step doesn't use quanto; this path is unused


# =========================================================================
# Utilities (from optimizer_utils.py)
# =========================================================================

def _compute_scale_for_dtype(tensor: Tensor, dtype: torch.dtype) -> float:
    """Compute appropriate scale for the given tensor and target dtype."""
    if dtype == torch.int8:
        abs_max = torch.max(torch.abs(tensor))
        return abs_max / 127.0 if abs_max > 0 else 1.0
    elif dtype == torch.uint8:
        max_val = torch.max(tensor)
        min_val = torch.min(tensor)
        range_val = max_val - min_val
        return range_val / 255.0 if range_val > 0 else 1.0
    elif dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
        abs_max = torch.max(torch.abs(tensor))
        if dtype == torch.float8_e4m3fn:
            max_representable = 448.0
        else:
            max_representable = 57344.0
        return abs_max / max_representable if abs_max > 0 else 1.0
    else:
        raise ValueError(f"Unsupported dtype for quantization: {dtype}")


def _quantize_tensor(tensor: Tensor, dtype: torch.dtype):
    """Quantize a floating-point tensor to the target dtype with scaling."""
    scale = _compute_scale_for_dtype(tensor, dtype)
    if dtype == torch.int8:
        quantized_data = torch.clamp(torch.round(tensor / scale), -128, 127).to(dtype)
    elif dtype == torch.uint8:
        quantized_data = torch.clamp(torch.round(tensor / scale), 0, 255).to(dtype)
    elif dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
        scaled_tensor = tensor / scale
        quantized_data = scaled_tensor.to(dtype)
    else:
        raise ValueError(f"Unsupported dtype for quantization: {dtype}")
    return quantized_data, scale


def _update_parameter(target, result_float: Tensor) -> None:
    """Update a parameter tensor, handling both regular and QBytesTensor."""
    if _QBytesTensor is not None and isinstance(target, _QBytesTensor):
        target_dtype = target._data.dtype
        device = target._data.device
        result_float = result_float.to(device)
        quantized_data, new_scale = _quantize_tensor(result_float, target_dtype)
        target._data.copy_(quantized_data)
        target._scale.copy_(new_scale)
    else:
        target.copy_(result_float)


def _get_format_params(dtype: torch.dtype) -> tuple[int, int]:
    """Returns (mantissa_bits, total_bits) for each format."""
    lookup = {
        torch.float32: (23, 32),
        torch.bfloat16: (7, 16),
        torch.float16: (10, 16),
        torch.float8_e4m3fn: (3, 8),
        torch.float8_e5m2: (2, 8),
        torch.int8: (0, 8),
    }
    if dtype not in lookup:
        raise ValueError(f"Unsupported dtype: {dtype}")
    return lookup[dtype]


def copy_stochastic(
    target: Tensor,
    source: Tensor,
    eps: Optional[float] = None,
) -> None:
    """Performs stochastic rounding from source tensor to target tensor."""
    with torch.no_grad():
        if target.dtype == torch.float32:
            target.copy_(source)
            return

        if target.dtype == torch.int8:
            scaled = source * 127.0
            noise = torch.rand_like(scaled) - 0.5
            rounded = torch.round(scaled + noise)
            clamped = torch.clamp(rounded, -127, 127)
            target.copy_(clamped.to(torch.int8))
            return

        mantissa_bits, _ = _get_format_params(target.dtype)
        source_int = source.view(dtype=torch.int32)
        bits_to_round = 23 - mantissa_bits

        rand = torch.randint_like(
            source, dtype=torch.int32, low=0, high=(1 << bits_to_round),
        )

        result = source_int.clone()
        result.add_(rand)
        mask = (-1) << bits_to_round
        result.bitwise_and_(mask)

        if eps is not None:
            eps_int = torch.tensor(eps, dtype=torch.float32).view(dtype=torch.int32)
            zero_mask = (result.abs() < eps_int)
            result[zero_mask] = torch.sign(source_int[zero_mask]) * eps_int

        result_float = result.view(dtype=torch.float32)

        if target.dtype == torch.float8_e4m3fn:
            result_float.clamp_(-448.0, 448.0)
        elif target.dtype == torch.float8_e5m2:
            result_float.clamp_(-57344.0, 57344.0)

        _update_parameter(target, result_float)
        del result, rand, source_int


class Auto8bitTensor:
    """Quantizes a float tensor to int8 with a single scalar scale factor."""

    def __init__(self, data, *args, **kwargs):
        if isinstance(data, dict):
            self._load_from_state_dict(data)
        else:
            abs_max = data.abs().max().item()
            scale = abs_max / 127.0 if abs_max > 0 else 1.0
            self.quantized = (data / scale).round().clamp(-127, 127).to(torch.int8)
            self.scale = scale
            self.orig_dtype = data.dtype

    def dequantize(self) -> Tensor:
        return self.quantized.to(dtype=torch.float32) * self.scale

    def to(self, *args, **kwargs):
        dtype = None
        if args and isinstance(args[0], torch.dtype):
            dtype = args[0]
            args = args[1:]
        elif "dtype" in kwargs:
            dtype = kwargs.pop("dtype")
        if dtype is not None:
            return self.dequantize().to(dtype=dtype, *args, **kwargs)
        return self.dequantize().to(*args, **kwargs)

    def state_dict(self):
        return {
            "quantized": self.quantized,
            "scale": self.scale,
            "orig_dtype": self.orig_dtype,
        }

    def _load_from_state_dict(self, state_dict):
        self.quantized = state_dict["quantized"]
        self.scale = state_dict["scale"]
        self.orig_dtype = state_dict["orig_dtype"]

    def __str__(self):
        return f"Auto8bitTensor({self.dequantize()})"


def _stochastic_grad_accumulation(param):
    """Post-accumulate gradient hook for stochastic rounding in non-fp32."""
    if hasattr(param, "_accum_grad"):
        grad_fp32 = param._accum_grad.clone().to(torch.float32)
        grad_fp32.add_(param.grad.to(torch.float32))
        copy_stochastic(param._accum_grad, grad_fp32)
        del grad_fp32
        del param.grad
    else:
        param._accum_grad = param.grad.clone()
        del param.grad


# =========================================================================
# Automagic Optimizer
# =========================================================================

class Automagic(torch.optim.Optimizer):
    """Per-element adaptive learning rate optimizer.

    Maintains a per-element LR mask that increases when gradient signs
    agree between steps and decreases when they disagree.  The LR mask
    is compressed to int8 to save memory.

    Uses Adafactor-style factored second moments for 2D+ parameters.

    Args:
        params: Iterable of parameters or param-group dicts.
        lr (float): Starting learning rate.  Auto-managed per-element.
        min_lr (float): Minimum per-element LR (default 1e-7).
        max_lr (float): Maximum per-element LR (default 1e-3).
        lr_bump (float): Amount to adjust per-element LR each step.
        eps: Epsilon for numerical stability.
        clip_threshold (float): RMS-based update clipping threshold.
        beta2 (float): Exponential decay rate for second moments.
        weight_decay (float): Decoupled weight decay coefficient.
    """

    def __init__(
        self,
        params,
        lr=1e-6,
        min_lr=1e-7,
        max_lr=1e-3,
        lr_bump=1e-6,
        eps=(1e-30, 1e-3),
        clip_threshold=1.0,
        beta2=0.999,
        weight_decay=0.0,
        do_paramiter_swapping=False,
        paramiter_swapping_factor=0.1,
    ):
        self.lr = lr
        if self.lr > 1e-3:
            import logging
            logging.getLogger(__name__).warning(
                "[Automagic] Start LR %.1e is very high — clamping to 1e-6. "
                "Automagic self-manages LR (not like AdamW).",
                lr,
            )
            self.lr = 1e-6
        self.min_lr = min_lr
        self.max_lr = max_lr
        self.lr_bump = lr_bump

        defaults = {
            "lr": lr,
            "eps": eps,
            "clip_threshold": clip_threshold,
            "beta2": beta2,
            "weight_decay": weight_decay,
        }
        super().__init__(params, defaults)

        self.base_lrs: List[float] = [lr for _ in self.param_groups]
        self.is_stochastic_rounding_accumulation = False

        # Setup stochastic grad accum hooks for non-fp32 params
        for group in self.param_groups:
            for param in group["params"]:
                if param.requires_grad and param.dtype != torch.float32:
                    self.is_stochastic_rounding_accumulation = True
                    param.register_post_accumulate_grad_hook(
                        _stochastic_grad_accumulation
                    )

        # Parameter swapping (disabled by default for LoRA training)
        self.do_paramiter_swapping = do_paramiter_swapping
        self.paramiter_swapping_factor = paramiter_swapping_factor
        self._total_paramiter_size = 0
        for group in self.param_groups:
            for param in group["params"]:
                self._total_paramiter_size += torch.numel(param)

        if self.do_paramiter_swapping:
            self._swap_paramiters()

    def _swap_paramiters(self):
        """Randomly freeze/unfreeze a subset of parameters."""
        all_params = []
        for group in self.param_groups:
            for param in group["params"]:
                param.requires_grad_(False)
                param.grad = None
                all_params.append(param)
        random.shuffle(all_params)

        target = int(self._total_paramiter_size * self.paramiter_swapping_factor)
        total = 0
        for param in all_params:
            total += torch.numel(param)
            if total >= target:
                break
            else:
                param.requires_grad_(True)

    @staticmethod
    def _get_lr(param_group, param_state):
        if "avg_lr" in param_state:
            return param_state["avg_lr"]
        return 0.0

    def _get_group_lr(self, group):
        group_lrs = [
            self._get_lr(group, self.state[p]) for p in group["params"]
        ]
        if not group_lrs:
            return self.lr
        return sum(group_lrs) / len(group_lrs)

    @staticmethod
    def _rms(tensor):
        return tensor.norm(2) / (tensor.numel() ** 0.5)

    @staticmethod
    def _approx_sq_grad(exp_avg_sq_row, exp_avg_sq_col):
        r_factor = (
            exp_avg_sq_row / exp_avg_sq_row.mean(dim=-1, keepdim=True)
        ).rsqrt_().unsqueeze(-1)
        c_factor = exp_avg_sq_col.unsqueeze(-2).rsqrt()
        return torch.mul(r_factor, c_factor)

    def _step_hook(self):
        """Copy stochastically rounded grads before step."""
        if not self.is_stochastic_rounding_accumulation:
            return
        for group in self.param_groups:
            for param in group["params"]:
                if param.requires_grad and hasattr(param, "_accum_grad"):
                    param.grad = param._accum_grad
                    del param._accum_grad

    def get_learning_rates(self):
        """Return per-group average learning rates."""
        lrs = [self._get_group_lr(group) for group in self.param_groups]
        return lrs if lrs else self.base_lrs

    def get_avg_learning_rate(self):
        lrs = self.get_learning_rates()
        return sum(lrs) / len(lrs)

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single Automagic optimization step."""
        self._step_hook()
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None or not p.requires_grad:
                    continue

                grad = p.grad
                if grad.dtype != torch.float32:
                    grad = grad.to(torch.float32)
                if grad.is_sparse:
                    raise RuntimeError("Automagic does not support sparse gradients.")

                state = self.state[p]
                grad_shape = grad.shape
                factored = len(grad_shape) >= 2

                # State initialization
                if len(state) == 0:
                    self._initialize_state(p)
                else:
                    if factored:
                        if "exp_avg_sq_row" not in state or "exp_avg_sq_col" not in state:
                            state["exp_avg_sq_row"] = torch.zeros(p.shape[:-1]).to(grad)
                            state["exp_avg_sq_col"] = torch.zeros(
                                p.shape[:-2] + p.shape[-1:]
                            ).to(grad)
                        else:
                            state["exp_avg_sq_row"] = state["exp_avg_sq_row"].to(grad)
                            state["exp_avg_sq_col"] = state["exp_avg_sq_col"].to(grad)
                    else:
                        if "exp_avg_sq" not in state:
                            state["exp_avg_sq"] = torch.zeros_like(grad)
                        else:
                            state["exp_avg_sq"] = state["exp_avg_sq"].to(grad)

                p_data_fp32 = p
                if _QBytesTensor is not None and isinstance(p_data_fp32, _QBytesTensor):
                    p_data_fp32 = p_data_fp32.dequantize()
                if p.dtype != torch.float32:
                    p_data_fp32 = p_data_fp32.clone().float()

                if "step" not in state:
                    state["step"] = 0
                state["step"] += 1
                state["RMS"] = self._rms(p_data_fp32)

                beta2 = group["beta2"]
                eps = group["eps"]
                if isinstance(eps, (tuple, list)):
                    eps = eps[0]
                update = (grad ** 2) + eps

                if factored:
                    exp_avg_sq_row = state["exp_avg_sq_row"]
                    exp_avg_sq_col = state["exp_avg_sq_col"]
                    exp_avg_sq_row.mul_(beta2).add_(
                        update.mean(dim=-1), alpha=(1.0 - beta2)
                    )
                    exp_avg_sq_col.mul_(beta2).add_(
                        update.mean(dim=-2), alpha=(1.0 - beta2)
                    )
                    update = self._approx_sq_grad(exp_avg_sq_row, exp_avg_sq_col)
                    update.mul_(grad)
                else:
                    exp_avg_sq = state["exp_avg_sq"]
                    exp_avg_sq.mul_(beta2).add_(update, alpha=(1.0 - beta2))
                    update = exp_avg_sq.rsqrt().mul_(grad)

                update.div_(
                    (self._rms(update) / group["clip_threshold"]).clamp_(min=1.0)
                )

                # Ensure state is initialized
                if "last_polarity" not in state or "lr_mask" not in state:
                    self._initialize_state(p)

                # Sign agreement
                last_polarity = state["last_polarity"]
                current_polarity = (update > 0).to(torch.bool)
                sign_agreement = torch.where(
                    last_polarity == current_polarity, 1, -1
                )
                state["last_polarity"] = current_polarity

                lr_mask = state["lr_mask"].to(torch.float32)

                # Per-element LR adjustment
                new_lr = torch.where(
                    sign_agreement > 0,
                    lr_mask + self.lr_bump,
                    lr_mask - self.lr_bump,
                )
                new_lr = torch.clamp(new_lr, min=self.min_lr, max=self.max_lr)

                update.mul_(new_lr)

                state["lr_mask"] = Auto8bitTensor(new_lr)
                state["avg_lr"] = torch.mean(new_lr)

                if group["weight_decay"] != 0:
                    weight_decay_update = (
                        p_data_fp32 * (-group["weight_decay"]) * new_lr
                    )
                    p_data_fp32.add_(weight_decay_update)

                p_data_fp32.add_(-update)

                if p.dtype != torch.float32:
                    copy_stochastic(p, p_data_fp32)

        return loss

    def _initialize_state(self, p):
        """Initialize optimizer state for a parameter."""
        state = self.state[p]
        state["step"] = 0

        if "lr_mask" not in state:
            state["lr_mask"] = Auto8bitTensor(
                torch.ones(p.shape, device=p.device, dtype=torch.float32) * self.lr
            )
        state["avg_lr"] = torch.mean(state["lr_mask"].to(torch.float32))

        if "last_polarity" not in state:
            state["last_polarity"] = torch.zeros(
                p.shape, dtype=torch.bool, device=p.device
            )

        factored = len(p.shape) >= 2
        if factored:
            state["exp_avg_sq_row"] = torch.zeros(p.shape[:-1]).to(p)
            state["exp_avg_sq_col"] = torch.zeros(
                p.shape[:-2] + p.shape[-1:]
            ).to(p)
        else:
            state["exp_avg_sq"] = torch.zeros_like(p)

        state["RMS"] = 0

    def state_dict(self, *args, **kwargs):
        orig = super().state_dict(*args, **kwargs)
        new_state = {}
        for p, state in orig["state"].items():
            save_state = {k: v for k, v in state.items() if k != "lr_mask"}
            if "lr_mask" in state:
                save_state["lr_mask"] = state["lr_mask"].state_dict()
            new_state[p] = save_state
        orig["state"] = new_state
        return orig

    def load_state_dict(self, state_dict, strict=True):
        is_valid = False
        if "state" in state_dict and isinstance(state_dict["state"], dict):
            for _, param_state in state_dict["state"].items():
                if isinstance(param_state, dict) and "lr_mask" in param_state:
                    is_valid = True
                    break

        if not is_valid:
            return

        state_dict_copy = {
            "state": {},
            "param_groups": state_dict["param_groups"],
        }
        for param_id, param_state in state_dict["state"].items():
            state_dict_copy["state"][param_id] = {
                k: v for k, v in param_state.items() if k != "lr_mask"
            }

        super().load_state_dict(state_dict_copy)

        current_params = []
        for group in self.param_groups:
            for p in group["params"]:
                if p.requires_grad:
                    current_params.append(p)

        saved_param_ids = list(state_dict["state"].keys())

        for i, current_param in enumerate(current_params):
            if i >= len(saved_param_ids):
                break

            saved_state = state_dict["state"][saved_param_ids[i]]
            if "lr_mask" not in saved_state:
                continue

            if current_param not in self.state:
                self._initialize_state(current_param)

            current_state = self.state[current_param]
            saved_lr_mask = saved_state["lr_mask"]

            try:
                if (
                    "quantized" in saved_lr_mask
                    and saved_lr_mask["quantized"].shape == current_param.shape
                ):
                    current_state["lr_mask"] = Auto8bitTensor(saved_lr_mask)
                else:
                    current_state["lr_mask"] = Auto8bitTensor(
                        torch.ones(current_param.shape, device=current_param.device, dtype=torch.float32)
                        * self.lr
                    )
            except Exception:
                current_state["lr_mask"] = Auto8bitTensor(
                    torch.ones(current_param.shape, device=current_param.device, dtype=torch.float32)
                    * self.lr
                )
