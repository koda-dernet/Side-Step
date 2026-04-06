"""
FixedLoRAModule -- adapter training step for ACE-Step 1.5

This module contains the ``FixedLoRAModule`` (nn.Module) responsible for
the per-step training logic. Timestep sampling is controlled by
``timestep_mode``: continuous (logit-normal, default for all variants)
or discrete (8-step turbo schedule). CFG dropout is applied for all modes.

Also includes small device/dtype/precision helpers.
"""

from __future__ import annotations

import logging
from collections import deque
from typing import Any, Dict, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

# Vendored ACE-Step utilities (standalone -- no base ACE-Step install needed)
from sidestep_engine.vendor.lora_utils import (
    check_peft_available,
    inject_lora_into_dit,
)
from sidestep_engine.vendor.lokr_utils import (
    check_lycoris_available,
    inject_lokr_into_dit,
)
from sidestep_engine.vendor.loha_utils import (
    check_lycoris_available as check_lycoris_available_loha,
    inject_loha_into_dit,
)
from sidestep_engine.vendor.oft_utils import (
    check_peft_oft_available,
    inject_oft_into_dit,
)

# V2 modules
from sidestep_engine.core.configs import (
    LoRAConfigV2, LoKRConfigV2, LoHAConfigV2, OFTConfigV2, TrainingConfigV2,
)
from sidestep_engine.core.timestep_sampling import (
    apply_cfg_dropout,
    sample_discrete_timesteps,
    sample_timesteps,
)
from sidestep_engine.core.types import TrainingUpdate

# Union type for adapter configs
AdapterConfig = Union[LoRAConfigV2, LoKRConfigV2, LoHAConfigV2, OFTConfigV2]


class _LastLossAccessor:
    """Lightweight wrapper that provides ``[-1]`` and bool access.

    Avoids storing an unbounded list of floats while keeping backward
    compatibility with code that reads ``module.training_losses[-1]``
    or checks ``if module.training_losses:``.
    """

    def __init__(self, module: "FixedLoRAModule") -> None:
        self._module = module
        self._has_value = False

    def append(self, value: float) -> None:
        self._module.last_training_loss = value
        self._has_value = True

    def __getitem__(self, idx: int) -> float:
        if idx == -1 or idx == 0:
            return self._module.last_training_loss
        raise IndexError("only index -1 or 0 is supported")

    def __bool__(self) -> bool:
        return self._has_value

    def __len__(self) -> int:
        return 1 if self._has_value else 0

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _normalize_device_type(device: Any) -> str:
    if isinstance(device, torch.device):
        return device.type
    if isinstance(device, str):
        return device.split(":", 1)[0]
    return str(device)



def _cuda_supports_bf16() -> bool:
    """Return True only on Ampere+ GPUs (sm_80+) with native bf16 hardware."""
    if not torch.cuda.is_available():
        return False
    try:
        return torch.cuda.is_bf16_supported()
    except AttributeError:
        major, _ = torch.cuda.get_device_capability()
        return major >= 8


def _select_compute_dtype(device_type: str, precision: str = "auto") -> torch.dtype:
    """Return the torch dtype for model weights and compute.

    Args:
        device_type: "cuda", "mps", "xpu", or "cpu".
        precision:   "auto" | "bf16" | "fp16" | "fp32".  When "auto",
                     selects the best precision for the detected hardware
                     (bf16 on Ampere+, fp16 on older CUDA/MPS).
    """
    if precision == "fp32":
        return torch.float32
    if precision == "fp16":
        return torch.float16
    if precision == "bf16":
        return torch.bfloat16
    # "auto": pick the best dtype the hardware supports
    if device_type in ("cuda", "xpu"):
        if device_type == "cuda" and not _cuda_supports_bf16():
            return torch.float16   # T4 / Turing / Volta fallback
        return torch.bfloat16
    if device_type == "mps":
        return torch.float16
    return torch.float32


# ===========================================================================
# FixedLoRAModule -- corrected training step
# ===========================================================================

class FixedLoRAModule(nn.Module):
    """Variant-aware adapter training module.

    Supports both LoRA (PEFT) and LoKR (LyCORIS) adapters.  The training
    step is identical for both -- only the injection and weight format differ.

    Training uses the same flow-matching regime across variants:

        1. Load pre-computed tensors.
        2. Apply **CFG dropout** on ``encoder_hidden_states`` when enabled.
        3. Sample noise ``x1`` and continuous timestep ``t`` via
           ``sample_timesteps()`` (logit-normal).
        4. Interpolate ``x_t = t * x1 + (1 - t) * x0``.
        5. Forward through decoder, compute flow matching loss.
    """

    def __init__(
        self,
        model: nn.Module,
        adapter_config: AdapterConfig,
        training_config: TrainingConfigV2,
        device: torch.device,
        dtype: torch.dtype,
    ) -> None:
        super().__init__()

        self.adapter_config = adapter_config
        self.adapter_type = training_config.adapter_type
        self.training_config = training_config
        self.device = torch.device(device) if isinstance(device, str) else device
        self.device_type = _normalize_device_type(self.device)
        self.dtype = _select_compute_dtype(
            self.device_type,
            getattr(training_config, "precision", "auto"),
        )
        self.transfer_non_blocking = self.device_type in ("cuda", "xpu")

        # LyCORIS network reference (only set for LoKR/LoHA)
        self.lycoris_net: Any = None
        self.adapter_info: Dict[str, Any] = {}

        # -- Adapter injection -----------------------------------------------
        if self.adapter_type == "lokr":
            self._inject_lokr(model, adapter_config)  # type: ignore[arg-type]
        elif self.adapter_type == "loha":
            self._inject_loha(model, adapter_config)  # type: ignore[arg-type]
        elif self.adapter_type == "oft":
            self._inject_oft(model, adapter_config)  # type: ignore[arg-type]
        else:
            # lora and dora both use PEFT LoRA injection (dora sets use_dora=True)
            self._inject_lora(model, adapter_config)  # type: ignore[arg-type]

        # Backward-compat alias
        self.lora_info = self.adapter_info

        # Model config (for timestep params read at runtime)
        self.config = model.config

        # -- Null condition embedding for CFG dropout ------------------------
        # ``model.null_condition_emb`` is a Parameter on the top-level model
        # (not the decoder).
        if hasattr(model, "null_condition_emb"):
            self._null_cond_emb = model.null_condition_emb
        else:
            self._null_cond_emb = None
            logger.warning(
                "[WARN] model.null_condition_emb not found -- CFG dropout disabled"
            )

        # -- Training strategy -------------------------------------------------
        self._is_turbo: bool = getattr(training_config, "is_turbo", False)
        self._timestep_mode: str = getattr(training_config, "timestep_mode", "continuous")

        # Timestep sampling params (used for all variants)
        self._timestep_mu = training_config.timestep_mu
        self._timestep_sigma = training_config.timestep_sigma
        self._data_proportion = training_config.data_proportion
        self._cfg_ratio = training_config.cfg_ratio
        self._loss_weighting = training_config.loss_weighting
        self._snr_gamma = training_config.snr_gamma
        self._loss_fn = training_config.loss_fn
        self._huber_delta = training_config.huber_delta
        self._channel_balance = training_config.channel_balance
        self._dynamic_channel_balance = training_config.dynamic_channel_balance
        self._vae_channel_prior = training_config.vae_channel_prior
        self._latent_noise_scale = training_config.latent_noise
        self._t_bias = training_config.t_bias
        self._legacy_loss = training_config.legacy_loss

        # Legacy mode overrides: revert everything to flat MSE
        if self._legacy_loss:
            self._loss_weighting = "none"
            self._loss_fn = "mse"
            self._channel_balance = False
            self._latent_noise_scale = 0.0

        # Per-channel weights and std (set by trainer from channel_stats.json)
        self._channel_weights: Optional[torch.Tensor] = None  # [64]
        self._channel_std: Optional[torch.Tensor] = None       # [64]

        # Running EMA of per-channel loss for dynamic rebalancing (lazy init)
        self._ema_ch_loss: Optional[torch.Tensor] = None  # [64]

        # Adaptive timestep sampler (set by trainer when enabled, None = off)
        self._adaptive_sampler = None

        _variant_label = "Turbo" if self._is_turbo else "Base/SFT"
        _mode_label = (
            "continuous logit-normal" if self._timestep_mode == "continuous"
            else "discrete 8-step"
        )
        logger.info(
            "[OK] %s detected -- using %s sampling + CFG dropout "
            "(ratio=%.2f), loss_weighting=%s, loss_fn=%s%s",
            _variant_label, _mode_label,
            self._cfg_ratio, self._loss_weighting, self._loss_fn,
            " [LEGACY]" if self._legacy_loss else "",
        )

        # Rolling buffer of sampled timesteps (CPU tensors) for TensorBoard
        # histogram logging.  Capped so memory stays bounded even over
        # very long runs.
        self._timestep_buffer: deque[torch.Tensor] = deque(maxlen=100)

        # When gradient checkpointing is enabled via wrapper layers that don't
        # expose enable_input_require_grads(), force at least one forward input
        # to require grad so checkpointed segments keep a valid autograd graph.
        self.force_input_grads_for_checkpointing: bool = False

        # Book-keeping -- store only the most recent loss to avoid
        # unbounded memory growth over long training runs.
        self.last_training_loss: float = 0.0

        # Backward-compat: property provides list-like [-1] access
        # for callers that read ``training_losses[-1]``.
        self.training_losses = _LastLossAccessor(self)

        # One-shot flag: emit detailed NaN diagnostic only once per run
        self._nan_diagnosed = False

        # When True, training_step skips all mutable side-effects
        # (adaptive sampler, timestep buffer, EMA, metrics, loss history).
        # Set by the validation runner to prevent validation from corrupting
        # training state.
        self._eval_mode: bool = False

        # Auxiliary metrics stashed by training_step for TB logging.
        # Populated every step, drained by the training loop at log_every.
        self._step_metrics: Dict[str, float] = {}

    # -----------------------------------------------------------------------
    # Adapter injection helpers
    # -----------------------------------------------------------------------

    def _inject_lora(self, model: nn.Module, cfg: LoRAConfigV2) -> None:
        """Inject LoRA adapters via PEFT.

        Raises:
            RuntimeError: If PEFT is not installed.
        """
        if not check_peft_available():
            raise RuntimeError(
                "PEFT is required for LoRA training but is not installed.\n"
                "Install it with:  uv pip install peft"
            )
        self.model, self.adapter_info = inject_lora_into_dit(model, cfg)
        logger.info(
            "[OK] LoRA injected: %s trainable params",
            f"{self.adapter_info['trainable_params']:,}",
        )

    def _inject_lokr(self, model: nn.Module, cfg: LoKRConfigV2) -> None:
        """Inject LoKR adapters via LyCORIS.

        After injection, explicitly moves the model to the target device
        so that newly created LoKR parameters (which LyCORIS creates on
        CPU) end up on GPU before training begins.

        Raises:
            RuntimeError: If LyCORIS is not installed.
        """
        if not check_lycoris_available():
            raise RuntimeError(
                "LyCORIS is required for LoKR training but is not installed.\n"
                "Install it with:  uv pip install lycoris-lora"
            )
        self.model, self.lycoris_net, self.adapter_info = inject_lokr_into_dit(
            model, cfg,
        )
        # LyCORIS creates adapter parameters on CPU.  Move the entire
        # model to the target device so all parameters (including the
        # new LoKR ones) are co-located before training setup.
        self.model = self.model.to(self.device)
        logger.info(
            "[OK] LoKR injected: %s trainable params (moved to %s)",
            f"{self.adapter_info['trainable_params']:,}",
            self.device,
        )

    def _inject_loha(self, model: nn.Module, cfg: LoHAConfigV2) -> None:
        """Inject LoHA adapters via LyCORIS.

        Raises:
            RuntimeError: If LyCORIS is not installed.
        """
        if not check_lycoris_available_loha():
            raise RuntimeError(
                "LyCORIS is required for LoHA training but is not installed.\n"
                "Install it with:  uv pip install lycoris-lora"
            )
        self.model, self.lycoris_net, self.adapter_info = inject_loha_into_dit(
            model, cfg,
        )
        self.model = self.model.to(self.device)
        logger.info(
            "[OK] LoHA injected: %s trainable params (moved to %s)",
            f"{self.adapter_info['trainable_params']:,}",
            self.device,
        )

    def _inject_oft(self, model: nn.Module, cfg: OFTConfigV2) -> None:
        """Inject OFT adapters via PEFT.

        Raises:
            RuntimeError: If PEFT OFT is not available.
        """
        if not check_peft_oft_available():
            raise RuntimeError(
                "PEFT >= 0.12.0 is required for OFT training.\n"
                "Install it with:  uv pip install peft>=0.12.0"
            )
        self.model, self.adapter_info = inject_oft_into_dit(model, cfg)
        logger.info(
            "[OK] OFT injected: %s trainable params",
            f"{self.adapter_info['trainable_params']:,}",
        )

    # -----------------------------------------------------------------------
    # Timestep buffer helpers
    # -----------------------------------------------------------------------

    def drain_timestep_buffer(self) -> Optional[torch.Tensor]:
        """Return accumulated timesteps and clear the buffer.

        Concatenates all stored per-batch timestep tensors into a single
        1-D tensor suitable for ``SummaryWriter.add_histogram()``.

        Returns:
            Concatenated CPU tensor, or ``None`` if the buffer is empty.
        """
        if not self._timestep_buffer:
            return None
        result = torch.cat(list(self._timestep_buffer))
        self._timestep_buffer.clear()
        return result

    def drain_step_metrics(self) -> Dict[str, float]:
        """Return and clear auxiliary metrics from the last training step.

        Called by the training loop at ``log_every`` intervals to feed
        TensorBoard without changing ``training_step``'s return type.
        """
        m = dict(self._step_metrics)
        self._step_metrics.clear()
        return m

    # -----------------------------------------------------------------------
    # Training step
    # -----------------------------------------------------------------------

    def training_step(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Single training step.

        Timestep sampling follows ``timestep_mode``: continuous uses
        logit-normal sampling, discrete uses the 8-step turbo schedule.
        CFG dropout is applied whenever the model provides a null-condition
        embedding and the configured ratio is greater than zero.

        Args:
            batch: Dict with keys ``target_latents``, ``attention_mask``,
                ``encoder_hidden_states``, ``encoder_attention_mask``,
                ``context_latents``.

        Returns:
            Scalar loss tensor (``float32`` for stable backward).
        """
        nb = self.transfer_non_blocking

        target_latents = batch["target_latents"].to(self.device, dtype=self.dtype, non_blocking=nb)
        attention_mask = batch["attention_mask"].to(self.device, dtype=self.dtype, non_blocking=nb)
        encoder_hidden_states = batch["encoder_hidden_states"].to(self.device, dtype=self.dtype, non_blocking=nb)
        encoder_attention_mask = batch["encoder_attention_mask"].to(self.device, dtype=self.dtype, non_blocking=nb)
        context_latents = batch["context_latents"].to(self.device, dtype=self.dtype, non_blocking=nb)

        bsz = target_latents.shape[0]

        # ---- Reference training path for all variants -----------------
        if self._null_cond_emb is not None and self._cfg_ratio > 0.0:
            encoder_hidden_states = apply_cfg_dropout(
                encoder_hidden_states,
                self._null_cond_emb,
                cfg_ratio=self._cfg_ratio,
            )
        if self._timestep_mode == "discrete":
            t, _r = sample_discrete_timesteps(
                batch_size=bsz,
                device=self.device,
                dtype=self.dtype,
            )
        elif self._adaptive_sampler is not None:
            t, _r = self._adaptive_sampler.sample(
                batch_size=bsz,
                base_sampler=sample_timesteps,
                device=self.device,
                dtype=self.dtype,
                data_proportion=self._data_proportion,
                timestep_mu=self._timestep_mu,
                timestep_sigma=self._timestep_sigma,
                use_meanflow=False,
            )
        else:
            t, _r = sample_timesteps(
                batch_size=bsz,
                device=self.device,
                dtype=self.dtype,
                data_proportion=self._data_proportion,
                timestep_mu=self._timestep_mu,
                timestep_sigma=self._timestep_sigma,
                use_meanflow=False,
            )

        # Record sampled timesteps for TensorBoard histogram logging.
        if not self._eval_mode:
            self._timestep_buffer.append(t.detach().cpu())

        # ---- Per-channel latent noise regularization ----------------------
        x0 = target_latents  # data
        if self._latent_noise_scale > 0 and self._channel_std is not None:
            ch_std = self._channel_std.to(
                device=x0.device, dtype=x0.dtype,
            )
            x0 = x0 + torch.randn_like(x0) * (self._latent_noise_scale * ch_std)

        # ---- Flow matching noise ----------------------------------------
        x1 = torch.randn_like(x0)  # noise
        t_ = t.unsqueeze(-1).unsqueeze(-1)

        # ---- Interpolate x_t -------------------------------------------
        xt = t_ * x1 + (1.0 - t_) * x0
        if self.force_input_grads_for_checkpointing:
            xt = xt.requires_grad_(True)

        # ---- Decoder forward -------------------------------------------
        decoder_outputs = self.model.decoder(
            hidden_states=xt,
            timestep=t,
            timestep_r=t,  # r = t
            attention_mask=attention_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            context_latents=context_latents,
        )

        # ---- Flow matching loss ----------------------------------------
        flow = x1 - x0
        pred = decoder_outputs[0]

        # Per-element loss computation
        _is_x0 = self._loss_fn.startswith("x0_")
        if _is_x0:
            # x₀-prediction: loss on reconstructed clean latent
            # x0_pred = xt - t * pred_velocity; target = x0
            _error = (xt - t_ * pred) - x0  # [B, T, 64]
        else:
            # Velocity-prediction: loss on predicted flow
            _error = pred - flow  # [B, T, 64]

        _fn = self._loss_fn.replace("x0_", "")  # strip x0_ prefix
        if _fn == "huber":
            per_element = F.smooth_l1_loss(
                _error, torch.zeros_like(_error),
                beta=self._huber_delta, reduction="none",
            )
        elif _fn == "pseudo_huber":
            _d = self._huber_delta
            per_element = _d * _d * (torch.sqrt(1.0 + (_error / _d) ** 2) - 1.0)
        else:
            per_element = _error ** 2  # MSE

        # Per-channel fidelity balancing
        if self._channel_balance and self._channel_weights is not None:
            cw = self._channel_weights.to(
                device=per_element.device, dtype=per_element.dtype,
            )
            # Dynamic rebalancing: blend static VAE prior with live loss EMA
            if self._dynamic_channel_balance:
                with torch.no_grad():
                    _pch = per_element.detach().mean(dim=(0, 1))  # [64]
                    if self._eval_mode:
                        # Use current EMA without updating it
                        _dyn_w = (self._ema_ch_loss / self._ema_ch_loss.mean().clamp(min=1e-8)) if self._ema_ch_loss is not None else torch.ones_like(_pch)
                    else:
                        if self._ema_ch_loss is None:
                            self._ema_ch_loss = _pch.clone()
                        else:
                            self._ema_ch_loss = self._ema_ch_loss.to(
                                device=_pch.device, dtype=_pch.dtype,
                            )
                            self._ema_ch_loss.lerp_(_pch, 0.01)
                        _dyn_w = self._ema_ch_loss / self._ema_ch_loss.mean().clamp(min=1e-8)
                    cw = 0.5 * cw + 0.5 * _dyn_w
            per_element = per_element * cw  # [B, T, 64] * [64]

        # Attention-mask-weighted loss (skip zero-padded positions)
        if not self._legacy_loss:
            mask = attention_mask.unsqueeze(-1)  # [B, T, 1]
            masked_sum = (per_element * mask).sum()
            n_valid = mask.sum() * per_element.shape[-1]
            per_sample_loss_raw = (per_element * mask).sum(dim=(-1, -2)) / (
                mask.sum(dim=(-1, -2)) * per_element.shape[-1]
            ).clamp(min=1e-8)
        else:
            masked_sum = per_element.sum()
            n_valid = torch.tensor(
                per_element.numel(), device=per_element.device,
                dtype=per_element.dtype,
            )
            per_sample_loss_raw = per_element.mean(dim=(-1, -2))

        # Timestep weighting
        if self._loss_weighting == "flow_snr":
            t_f32 = t.float().clamp(min=1e-4, max=1.0 - 1e-4)
            w = ((1.0 - t_f32) ** self._t_bias) / (t_f32 * (1.0 - t_f32))
            w = w.clamp(max=self._snr_gamma)
            _w_pre_norm = w.clone()  # snapshot before normalization for telemetry
            w = w / w.mean().clamp(min=1e-8)  # normalize to preserve scale
            diffusion_loss = (w.to(per_sample_loss_raw.dtype) * per_sample_loss_raw).mean()
        elif self._loss_weighting == "min_snr":
            t_f32 = t.float().clamp(min=1e-4, max=1.0 - 1e-4)
            snr = ((1.0 - t_f32) / t_f32) ** 2
            snr = snr.clamp(max=1e6)
            weights = torch.clamp(snr, max=self._snr_gamma) / snr.clamp(min=1e-6)
            diffusion_loss = (weights.to(per_sample_loss_raw.dtype) * per_sample_loss_raw).mean()
        else:
            diffusion_loss = masked_sum / n_valid.clamp(min=1e-8)

        # Update adaptive sampler with per-sample losses (if enabled)
        if self._adaptive_sampler is not None and not self._eval_mode:
            with torch.no_grad():
                self._adaptive_sampler.update(t, per_sample_loss_raw.detach())

        # -- Auxiliary metrics for TensorBoard --------------------------
        if not self._eval_mode:
            sm = self._step_metrics
            sm["fidelity/timestep_mean"] = float(t.mean())
            sm["fidelity/raw_loss"] = float(per_sample_loss_raw.mean())
            sm["fidelity/weighted_loss"] = float(diffusion_loss)
            if self._loss_weighting == "flow_snr":
                sm["fidelity/snr_weight_mean"] = float(_w_pre_norm.mean())
                sm["fidelity/snr_weight_max"] = float(_w_pre_norm.max())
                if _w_pre_norm.numel() > 1:
                    sm["fidelity/snr_weight_spread"] = float(
                        _w_pre_norm.max() / _w_pre_norm.min().clamp(min=1e-8)
                    )
            elif self._loss_weighting == "min_snr":
                sm["fidelity/snr_weight_mean"] = float(weights.mean())
                sm["fidelity/snr_weight_max"] = float(weights.max())
            if self._channel_balance and self._channel_weights is not None:
                per_ch = per_element.mean(dim=(0, 1))  # [64]
                sm["fidelity/ch_loss_max"] = float(per_ch.max())
                sm["fidelity/ch_loss_min"] = float(per_ch.min())
                sm["fidelity/ch_loss_ratio"] = float(
                    per_ch.max() / per_ch.min().clamp(min=1e-8)
                )
                if self._dynamic_channel_balance and self._ema_ch_loss is not None:
                    sm["fidelity/ch_dynamic_blend_range"] = float(
                        cw.max() / cw.min().clamp(min=1e-8)
                    )
            # Arrangement vs detail split (always logged)
            _t_flat = t.detach()
            _loss_flat = per_sample_loss_raw.detach()
            _hi = _t_flat > 0.5
            _lo = ~_hi
            if _hi.any():
                sm["fidelity/arrangement_loss"] = float(_loss_flat[_hi].mean())
            if _lo.any():
                sm["fidelity/detail_loss"] = float(_loss_flat[_lo].mean())

        # fp32 for stable backward
        diffusion_loss = diffusion_loss.float()

        if torch.isnan(diffusion_loss) or torch.isinf(diffusion_loss):
            logger.warning(
                "[WARN] NaN/Inf loss detected (step will be skipped by trainer)"
            )
            if not self._nan_diagnosed:
                self._nan_diagnosed = True
                diag_parts = []
                for _k in ("target_latents", "encoder_hidden_states",
                           "context_latents"):
                    _v = batch.get(_k)
                    if _v is not None:
                        has_nan = bool(torch.isnan(_v).any())
                        has_inf = bool(torch.isinf(_v).any())
                        if has_nan or has_inf:
                            diag_parts.append(
                                f"{_k}: nan={has_nan} inf={has_inf}"
                            )
                if diag_parts:
                    logger.warning(
                        "[DIAG] Corrupted input tensors: %s  "
                        "(re-run preprocessing to fix)",
                        ", ".join(diag_parts),
                    )
                else:
                    logger.warning(
                        "[DIAG] Input tensors are clean -- NaN likely "
                        "from model forward pass (precision/overflow)"
                    )
            return diffusion_loss

        if not self._eval_mode:
            self.training_losses.append(diffusion_loss.item())
        return diffusion_loss
