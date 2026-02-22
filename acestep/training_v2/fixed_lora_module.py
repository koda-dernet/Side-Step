"""
FixedLoRAModule -- Variant-aware adapter training step for ACE-Step 1.5

This module contains the ``FixedLoRAModule`` (nn.Module) responsible for
the per-step training logic.  The training strategy is auto-selected
based on the model variant:

- **Turbo**: discrete 8-step timestep sampling, no CFG dropout.
- **Base / SFT**: continuous logit-normal timestep sampling + CFG dropout.

Also includes small device/dtype/precision helpers used by both the
Fabric and basic training loops.
"""

from __future__ import annotations

import logging
from collections import deque
from contextlib import nullcontext
from typing import Any, Dict, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

# Vendored ACE-Step utilities (standalone -- no base ACE-Step install needed)
from acestep.training_v2._vendor.lora_utils import (
    check_peft_available,
    inject_lora_into_dit,
)
from acestep.training_v2._vendor.lokr_utils import (
    check_lycoris_available,
    inject_lokr_into_dit,
)

# V2 modules
from acestep.training_v2.configs import LoRAConfigV2, LoKRConfigV2, TrainingConfigV2
from acestep.training_v2.timestep_sampling import (
    apply_cfg_dropout,
    sample_discrete_timesteps,
    sample_timesteps,
)
from acestep.training_v2.ui import TrainingUpdate

# Union type for adapter configs
AdapterConfig = Union[LoRAConfigV2, LoKRConfigV2]


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


def _select_compute_dtype(device_type: str) -> torch.dtype:
    if device_type in ("cuda", "xpu"):
        return torch.bfloat16
    return torch.float32


def _select_fabric_precision(device_type: str) -> str:
    if device_type in ("cuda", "xpu"):
        return "bf16-mixed"
    if device_type == "mps":
        # "16-mixed" activates a GradScaler whose _unscale_grads_ crashes on
        # MPS tensors.  Use "32-true" instead -- the training step's own
        # torch.autocast still provides fp16 forward-pass benefits.
        return "32-true"
    return "32-true"


# ===========================================================================
# FixedLoRAModule -- corrected training step
# ===========================================================================

class FixedLoRAModule(nn.Module):
    """Variant-aware adapter training module.

    Supports both LoRA (PEFT) and LoKR (LyCORIS) adapters.  The training
    step is identical for both -- only the injection and weight format differ.

    The timestep sampling strategy is auto-selected from ``is_turbo``:

    **Turbo** (``is_turbo=True``):
        1. Load pre-computed tensors.
        2. Sample noise ``x1`` and a discrete timestep ``t`` from the
           turbo 8-step schedule (uniform over steps).
        3. Interpolate ``x_t = t * x1 + (1 - t) * x0``.
        4. Forward through decoder, compute flow matching loss.

    **Base / SFT** (``is_turbo=False``):
        1. Load pre-computed tensors.
        2. Apply **CFG dropout** on ``encoder_hidden_states``.
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
        self.dtype = _select_compute_dtype(self.device_type)
        self.transfer_non_blocking = self.device_type in ("cuda", "xpu")

        # LyCORIS network reference (only set for LoKR)
        self.lycoris_net: Any = None
        self.adapter_info: Dict[str, Any] = {}

        # -- Adapter injection -----------------------------------------------
        if self.adapter_type == "lokr":
            self._inject_lokr(model, adapter_config)  # type: ignore[arg-type]
        else:
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

        # -- Variant-aware training strategy -----------------------------------
        self._is_turbo: bool = getattr(training_config, "is_turbo", False)

        # Timestep sampling params (used by base/sft continuous sampling)
        self._timestep_mu = training_config.timestep_mu
        self._timestep_sigma = training_config.timestep_sigma
        self._data_proportion = training_config.data_proportion
        self._cfg_ratio = training_config.cfg_ratio
        self._loss_weighting = training_config.loss_weighting
        self._snr_gamma = training_config.snr_gamma

        if self._is_turbo:
            logger.info(
                "[OK] Turbo detected -- using discrete 8-step sampling, "
                "CFG dropout disabled"
            )
        else:
            logger.info(
                "[OK] Base/SFT detected -- using continuous logit-normal "
                "sampling + CFG dropout (ratio=%.2f), loss_weighting=%s",
                self._cfg_ratio,
                self._loss_weighting,
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
        CPU) end up on GPU before Fabric wraps the model.

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
        # new LoKR ones) are co-located before Fabric setup.
        self.model = self.model.to(self.device)
        logger.info(
            "[OK] LoKR injected: %s trainable params (moved to %s)",
            f"{self.adapter_info['trainable_params']:,}",
            self.device,
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

    # -----------------------------------------------------------------------
    # Training step
    # -----------------------------------------------------------------------

    def training_step(self, batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Single training step with variant-aware timestep sampling.

        Turbo models use discrete 8-step sampling (no CFG dropout).
        Base/SFT models use continuous logit-normal sampling + CFG dropout.

        Args:
            batch: Dict with keys ``target_latents``, ``attention_mask``,
                ``encoder_hidden_states``, ``encoder_attention_mask``,
                ``context_latents``.

        Returns:
            Scalar loss tensor (``float32`` for stable backward).
        """
        # Mixed-precision context
        if self.device_type in ("cuda", "xpu", "mps"):
            autocast_ctx = torch.autocast(device_type=self.device_type, dtype=self.dtype)
        else:
            autocast_ctx = nullcontext()

        with autocast_ctx:
            nb = self.transfer_non_blocking

            target_latents = batch["target_latents"].to(self.device, dtype=self.dtype, non_blocking=nb)
            attention_mask = batch["attention_mask"].to(self.device, dtype=self.dtype, non_blocking=nb)
            encoder_hidden_states = batch["encoder_hidden_states"].to(self.device, dtype=self.dtype, non_blocking=nb)
            encoder_attention_mask = batch["encoder_attention_mask"].to(self.device, dtype=self.dtype, non_blocking=nb)
            context_latents = batch["context_latents"].to(self.device, dtype=self.dtype, non_blocking=nb)

            bsz = target_latents.shape[0]

            if self._is_turbo:
                # ---- Turbo: discrete 8-step, no CFG dropout ----------------
                t, r = sample_discrete_timesteps(
                    batch_size=bsz,
                    device=self.device,
                    dtype=self.dtype,
                )
            else:
                # ---- Base/SFT: CFG dropout + continuous sampling -----------
                if self._null_cond_emb is not None and self._cfg_ratio > 0.0:
                    encoder_hidden_states = apply_cfg_dropout(
                        encoder_hidden_states,
                        self._null_cond_emb,
                        cfg_ratio=self._cfg_ratio,
                    )
                t, r = sample_timesteps(
                    batch_size=bsz,
                    device=self.device,
                    dtype=self.dtype,
                    data_proportion=self._data_proportion,
                    timestep_mu=self._timestep_mu,
                    timestep_sigma=self._timestep_sigma,
                    use_meanflow=False,
                )

            # Record sampled timesteps for TensorBoard histogram logging.
            self._timestep_buffer.append(t.detach().cpu())

            # ---- Flow matching noise ----------------------------------------
            x1 = torch.randn_like(target_latents)  # noise
            x0 = target_latents  # data
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

            if self._loss_weighting == "min_snr":
                per_sample_loss = F.mse_loss(
                    decoder_outputs[0], flow, reduction="none",
                )
                per_sample_loss = per_sample_loss.mean(dim=(-1, -2))
                # Upcast to fp32 to avoid overflow on fp16 devices (MPS):
                # ((1-t)/t)^2 can reach ~1e8 which exceeds fp16 max (65504).
                t_f32 = t.float().clamp(min=1e-4, max=1.0 - 1e-4)
                snr = ((1.0 - t_f32) / t_f32) ** 2
                snr = snr.clamp(max=1e6)
                weights = torch.clamp(snr, max=self._snr_gamma) / snr.clamp(min=1e-6)
                diffusion_loss = (weights.to(per_sample_loss.dtype) * per_sample_loss).mean()
            else:
                diffusion_loss = F.mse_loss(decoder_outputs[0], flow)

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

        self.training_losses.append(diffusion_loss.item())
        return diffusion_loss
