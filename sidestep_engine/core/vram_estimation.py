"""VRAM estimation for ACE-Step training.

Estimates peak GPU memory from first principles using ground-truth
architecture constants measured directly from the model safetensors.

The dominant cost during training is **activation memory** -- tensors
saved by autograd for the backward pass.  The MLP intermediates
(gate, up, silu, intermediate at ``B * S * 6144``) account for
roughly 50-55% of per-layer activation memory, followed by the
attention QKV projections and SDPA outputs.

Gradient checkpointing eliminates stored activations for checkpointed
layers (recomputes them during backward), making it the single most
impactful VRAM knob.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# =========================================================================
# Architecture constants -- measured from ACE-Step v1.5 safetensors
# =========================================================================

_HIDDEN: int = 2048
_INTERMEDIATE: int = 6144
_NUM_HEADS: int = 16
_NUM_KV_HEADS: int = 8       # GQA: 8 KV heads vs 16 Q heads
_HEAD_DIM: int = 128
_SLIDING_WINDOW: int = 128
_NUM_LAYERS: int = 24
_PATCH_SIZE: int = 2
_LATENT_FPS: float = 25.0    # 48000 Hz / 1920x downsampling
_DEFAULT_CHUNK_S: int = 60
_MAX_AUDIO_S: int = 240         # worst-case when chunking is disabled
_DEFAULT_NUM_LAYERS: int = _NUM_LAYERS

# Model weights (bf16) -- measured from model.safetensors
_MODEL_WEIGHTS_MB: float = 4566.0       # full model.safetensors (decoder + internal encoders + tokenizer + detokenizer)
_ENCODER_OFFLOAD_SAVE_MB: float = 0.0   # internal encoders load with the model; external Qwen3 is preprocessing-only

# Optimizer state: bytes per trainable parameter
_OPTIM_BPP: Dict[str, float] = {
    "adamw":     12.0,   # m (fp32) + v (fp32) + master copy (fp32)
    "prodigy":   16.0,   # m + v + s + d states (all fp32)
    "adamw8bit":  6.2,   # m (int8) + v (int8) + master (fp32) + quantization overhead
    "adafactor":  1.5,   # factored second-moment states
}

# CUDA context + cuDNN workspace + misc driver allocations
_CUDA_OVERHEAD_MB: float = 500.0

# PyTorch allocator fragmentation (reserved but unused memory)
_FRAGMENTATION_RATIO: float = 0.12

# Flash attention discount on stored activations: FA2 avoids
# materialising full attention scores, saving ~15-20% on the
# attention portion.  Since attention is ~25-30% of total activation
# memory, net effect is ~5% overall.
_FLASH_ATTN_DISCOUNT: float = 0.95

# SDPA backward peak: temporary tiles during score recomputation.
# Expressed as a fraction of one layer's persistent activations.
_SDPA_BACKWARD_PEAK_RATIO: float = 0.08

# Cross-attention context length (encoder output sequence).
# Varies by conditioning, but 512 is a reasonable average.
_CONTEXT_LEN: int = 512

# Legacy aliases -- kept so ``selective_checkpointing.py`` re-exports
# don't break.  Tests should migrate to the new API.
_MB_PER_LAYER_60S: float = 380.0         # deprecated
_BACKWARD_MULTIPLIER: float = 2.0        # deprecated
_MODEL_OVERHEAD_OFFLOAD_MB: float = _MODEL_WEIGHTS_MB
_MODEL_OVERHEAD_NO_OFFLOAD_MB: float = _MODEL_WEIGHTS_MB


# =========================================================================
# Attention backend detection
# =========================================================================

def detect_attn_backend(device: str = "auto", precision: str = "auto") -> str:
    """Detect the best available attention backend without loading a model.

    Returns ``'flash_attention_2'``, ``'sdpa'``, or ``'eager'``.
    """
    resolved_device = device
    resolved_precision = precision
    if device == "auto" or precision == "auto":
        try:
            from sidestep_engine.models.gpu_utils import detect_gpu
            info = detect_gpu(requested_device=device, requested_precision=precision)
            resolved_device = info.device
            resolved_precision = info.precision
        except Exception:
            return "sdpa"
    try:
        from sidestep_engine.models.loader import (
            _flash_attention_unavailable_reason,
        )
        if _flash_attention_unavailable_reason(resolved_device, resolved_precision) is None:
            return "flash_attention_2"
    except Exception:
        pass
    return "sdpa"


# =========================================================================
# Per-layer activation memory (the core model)
# =========================================================================

def _activation_bytes_per_layer(
    batch_size: int,
    seq_len: int,
    is_full_attention: bool = True,
    attn_backend: str = "sdpa",
) -> int:
    """Estimate autograd-saved tensors for one decoder layer (bytes).

    Accounts for every tensor PyTorch saves during the forward pass
    for use in backward:

    Self-attention:
      - RMSNorm input, projection input (shared), Q, K, V,
        SDPA output, logsumexp, o_proj input, residual

    Cross-attention:
      - Same structure, K/V come from encoder context

    MLP (SwiGLU):
      - RMSNorm input, projection input (shared), gate output,
        up output, silu output, intermediate (for down_proj), residual

    The MLP intermediates (4 tensors at B*S*6144*2 each) typically
    dominate, accounting for ~53% of per-layer activation memory.
    """
    B, S, H, I = batch_size, seq_len, _HIDDEN, _INTERMEDIATE
    C = _CONTEXT_LEN
    bpe = 2  # bf16

    # -- Self-attention saved tensors --
    sa = 0
    sa += B * S * H * bpe           # norm input
    sa += B * S * H * bpe           # proj input (shared for q/k/v)
    sa += B * _NUM_HEADS * S * _HEAD_DIM * bpe      # Q
    sa += B * _NUM_KV_HEADS * S * _HEAD_DIM * bpe   # K
    sa += B * _NUM_KV_HEADS * S * _HEAD_DIM * bpe   # V
    sa += B * _NUM_HEADS * S * _HEAD_DIM * bpe       # SDPA output
    sa += B * _NUM_HEADS * S * 4                      # logsumexp (fp32)
    sa += B * S * H * bpe           # o_proj input
    sa += B * S * H * bpe           # residual

    # -- Cross-attention saved tensors --
    ca = 0
    ca += B * S * H * bpe           # norm input
    ca += B * S * H * bpe           # Q proj input
    ca += B * C * H * bpe           # K/V proj input (encoder states)
    ca += B * _NUM_HEADS * S * _HEAD_DIM * bpe      # Q
    ca += B * _NUM_KV_HEADS * C * _HEAD_DIM * bpe   # K (encoder length)
    ca += B * _NUM_KV_HEADS * C * _HEAD_DIM * bpe   # V (encoder length)
    ca += B * _NUM_HEADS * S * _HEAD_DIM * bpe       # SDPA output
    ca += B * _NUM_HEADS * S * 4                      # logsumexp (fp32)
    ca += B * S * H * bpe           # o_proj input
    ca += B * S * H * bpe           # residual

    # -- MLP (SwiGLU) saved tensors --
    mlp = 0
    mlp += B * S * H * bpe          # norm input
    mlp += B * S * H * bpe          # proj input (shared gate/up)
    mlp += B * S * I * bpe          # gate_proj output
    mlp += B * S * I * bpe          # up_proj output
    mlp += B * S * I * bpe          # silu(gate) output
    mlp += B * S * I * bpe          # intermediate (input to down_proj)
    mlp += B * S * H * bpe          # residual

    total = sa + ca + mlp

    # Flash attention: avoids materialising some intermediates
    if attn_backend == "flash_attention_2":
        total = int(total * _FLASH_ATTN_DISCOUNT)

    return total


def _seq_len_from_chunk(chunk_duration_s: Optional[int]) -> int:
    """Convert chunk duration (seconds) to decoder sequence length.

    - ``None`` → default 60 s (matches default training config).
    - ``0`` → 240 s worst-case (chunking disabled, full audio length).
    - ``> 0`` → use the given value directly.
    """
    if chunk_duration_s is None:
        seconds = _DEFAULT_CHUNK_S
    elif chunk_duration_s <= 0:
        seconds = _MAX_AUDIO_S
    else:
        seconds = chunk_duration_s
    latent_frames = int(seconds * _LATENT_FPS)
    return latent_frames // _PATCH_SIZE


# =========================================================================
# Public activation estimate
# =========================================================================

def estimate_activation_mb(
    num_uncheckpointed: int,
    batch_size: int = 1,
    chunk_duration_s: Optional[int] = None,
    attn_backend: str = "sdpa",
) -> float:
    """Estimate activation VRAM in MB.

    Args:
        num_uncheckpointed: Decoder layers NOT checkpointed.
        batch_size: Training batch size.
        chunk_duration_s: Chunk duration in seconds (``None`` = 60 s default).
        attn_backend: ``'flash_attention_2'``, ``'sdpa'``, or ``'eager'``.
    """
    seq_len = _seq_len_from_chunk(chunk_duration_s)

    # Half the layers use full attention, half use sliding.
    # For simplicity, all layers use the same per-layer cost since
    # the sliding window only affects the temporary backward peak,
    # not the persistent saved tensors (SDPA doesn't materialise
    # the full score matrix for either path).
    per_layer = _activation_bytes_per_layer(
        batch_size, seq_len, is_full_attention=True, attn_backend=attn_backend,
    )

    # Persistent activations for non-checkpointed layers
    persistent = num_uncheckpointed * per_layer

    # During backward, SDPA recomputes attention scores in tiles,
    # creating a temporary peak for the layer being processed.
    backward_peak = int(per_layer * _SDPA_BACKWARD_PEAK_RATIO)

    total = persistent + backward_peak
    return total / (1024 * 1024)


# =========================================================================
# Trainable parameter estimation
# =========================================================================

def _estimate_trainable_params(
    adapter_type: str = "lora",
    rank: int = 8,
    target_mlp: bool = False,
    num_layers: int = _DEFAULT_NUM_LAYERS,
    attention_type: str = "both",
) -> int:
    """Estimate trainable parameter count for a given adapter config.

    Uses the real GQA projection dimensions:
      q_proj: [2048, 2048]   k_proj: [1024, 2048]
      v_proj: [1024, 2048]   o_proj: [2048, 2048]
    """
    if adapter_type == "oft":
        mods_per_attn = 4
        attn_blocks = 2 if attention_type == "both" else 1
        mods = mods_per_attn * attn_blocks
        if target_mlp:
            mods += 3
        return mods * num_layers * rank * rank

    # LoRA / DoRA / LoKR / LoHA: rank * (in_features + out_features) per target
    lora_per_attn_block = (
        rank * (_HIDDEN + _HIDDEN)                     # q_proj: 2048 -> 2048
        + rank * (_HIDDEN + _NUM_KV_HEADS * _HEAD_DIM)  # k_proj: 2048 -> 1024
        + rank * (_HIDDEN + _NUM_KV_HEADS * _HEAD_DIM)  # v_proj: 2048 -> 1024
        + rank * (_HIDDEN + _HIDDEN)                     # o_proj: 2048 -> 2048
    )

    if attention_type == "both":
        attn_params = lora_per_attn_block * 2 * num_layers
    else:
        attn_params = lora_per_attn_block * num_layers

    mlp_params = 0
    if target_mlp:
        mlp_per_layer = (
            rank * (_HIDDEN + _INTERMEDIATE)      # gate_proj: 2048 -> 6144
            + rank * (_HIDDEN + _INTERMEDIATE)    # up_proj:   2048 -> 6144
            + rank * (_INTERMEDIATE + _HIDDEN)    # down_proj: 6144 -> 2048
        )
        mlp_params = mlp_per_layer * num_layers

    return attn_params + mlp_params


# =========================================================================
# Optimizer state estimate
# =========================================================================

def estimate_optimizer_state_mb(
    adapter_type: str = "lora",
    rank: int = 8,
    target_mlp: bool = False,
    optimizer_type: str = "adamw",
    num_layers: int = _DEFAULT_NUM_LAYERS,
) -> float:
    """Estimate optimizer state VRAM in MB."""
    params = _estimate_trainable_params(adapter_type, rank, target_mlp, num_layers)
    bpp = _OPTIM_BPP.get(optimizer_type, 12.0)
    return params * bpp / (1024 * 1024)


# =========================================================================
# Peak VRAM (combined)
# =========================================================================

def estimate_peak_vram_mb(
    checkpointing_ratio: float = 1.0,
    batch_size: int = 1,
    chunk_duration_s: Optional[int] = None,
    attn_backend: str = "sdpa",
    offload_encoder: bool = True,
    adapter_type: str = "lora",
    rank: int = 8,
    target_mlp: bool = False,
    optimizer_type: str = "adamw",
    num_layers: int = _DEFAULT_NUM_LAYERS,
) -> Tuple[float, Dict[str, float]]:
    """Estimate peak training VRAM with a detailed breakdown.

    Returns ``(peak_mb, breakdown)`` where *breakdown* has keys
    ``model_mb``, ``activation_mb``, ``optimizer_mb``, ``gradient_mb``,
    ``adapter_mb``, ``cuda_overhead_mb``, ``peak_mb``, and optionally
    ``gpu_total_mb`` (added by GUI endpoint).
    """
    model_mb = _MODEL_WEIGHTS_MB

    # Checkpointing: determine how many layers store full activations
    ckpt = max(0, min(num_layers, round(num_layers * checkpointing_ratio)))
    uncheckpointed = num_layers - ckpt

    if ckpt > 0 and uncheckpointed == 0:
        uncheckpointed = 1  # recomputation window always holds one layer

    activation_mb = estimate_activation_mb(
        uncheckpointed, batch_size, chunk_duration_s, attn_backend,
    )

    # When checkpointing is active, checkpointed layers still store
    # their input hidden states (B * S * H * 2 per layer).
    if ckpt > 0:
        seq_len = _seq_len_from_chunk(chunk_duration_s)
        ckpt_input_bytes = ckpt * batch_size * seq_len * _HIDDEN * 2
        activation_mb += ckpt_input_bytes / (1024 * 1024)

    optimizer_mb = estimate_optimizer_state_mb(
        adapter_type, rank, target_mlp, optimizer_type, num_layers,
    )

    params = _estimate_trainable_params(adapter_type, rank, target_mlp, num_layers)
    gradient_mb = params * 4.0 / (1024 * 1024)   # fp32 gradients
    adapter_mb = params * 2.0 / (1024 * 1024)     # bf16 weights

    subtotal = model_mb + activation_mb + optimizer_mb + gradient_mb + adapter_mb + _CUDA_OVERHEAD_MB
    fragmentation_mb = subtotal * _FRAGMENTATION_RATIO
    peak = subtotal + fragmentation_mb

    return peak, {
        "model_mb": model_mb,
        "activation_mb": activation_mb,
        "optimizer_mb": optimizer_mb,
        "gradient_mb": gradient_mb,
        "adapter_mb": adapter_mb,
        "cuda_overhead_mb": _CUDA_OVERHEAD_MB,
        "fragmentation_mb": fragmentation_mb,
        "peak_mb": peak,
    }


# =========================================================================
# VRAM verdict (soft gate)
# =========================================================================

def vram_verdict(
    peak_mb: float,
    gpu_total_mb: float,
    system_used_mb: float = 0.0,
) -> str:
    """Return a verdict for the estimated peak VRAM vs available GPU.

    Args:
        peak_mb: Estimated peak training VRAM (our workload only).
        gpu_total_mb: Total GPU VRAM.
        system_used_mb: VRAM already consumed by other processes
            (desktop compositor, other apps, etc.).  When provided,
            the effective ceiling is ``gpu_total_mb - system_used_mb``.

    Returns ``'green'``, ``'yellow'``, or ``'red'``.

    - **green**: estimated < 80% of effective VRAM -- comfortable.
    - **yellow**: 80-95% -- tight, may work.
    - **red**: >95% -- will very likely OOM.

    This is NEVER a hard gate.  The user can always proceed.
    """
    effective = gpu_total_mb - max(0.0, system_used_mb)
    if effective <= 0:
        return "green"  # can't check, don't block
    ratio = peak_mb / effective
    if ratio < 0.80:
        return "green"
    if ratio < 0.95:
        return "yellow"
    return "red"


def system_vram_used_mb(
    gpu_total_mb: Optional[float],
    gpu_free_mb: Optional[float],
) -> float:
    """Compute how much VRAM other processes are consuming.

    Returns 0 if either value is unavailable.
    """
    if gpu_total_mb is None or gpu_free_mb is None:
        return 0.0
    return max(0.0, gpu_total_mb - gpu_free_mb)


# =========================================================================
# Suggestion / menu builders
# =========================================================================

def suggest_checkpointing(
    vram_total_mb: float,
    batch_size: int = 1,
    chunk_duration_s: Optional[int] = None,
    attn_backend: str = "sdpa",
    offload_encoder: bool = True,
    num_layers: int = _DEFAULT_NUM_LAYERS,
    vram_free_mb: Optional[float] = None,
    adapter_type: str = "lora",
    rank: int = 8,
    target_mlp: bool = False,
    optimizer_type: str = "adamw",
) -> Tuple[float, str]:
    """Suggest a checkpointing ratio that fits in the VRAM budget.

    Iterates from least to most checkpointing, returning the first
    ratio whose estimated peak VRAM fits within 90% of available memory.

    Returns ``(ratio, reason)`` with a human-readable reason string.
    """
    budget = (vram_free_mb if vram_free_mb is not None else vram_total_mb) * 0.90

    for ratio in (0.0, 0.25, 0.5, 0.75, 1.0):
        peak, _ = estimate_peak_vram_mb(
            ratio, batch_size, chunk_duration_s, attn_backend,
            offload_encoder, adapter_type, rank, target_mlp,
            optimizer_type, num_layers,
        )
        if peak <= budget:
            label = {0.0: "Off", 0.25: "Minimal", 0.5: "Partial",
                     0.75: "Most", 1.0: "Full"}[ratio]
            return ratio, f"{label} — est. ~{peak:.0f} MB fits in ~{budget:.0f} MB budget"

    return 1.0, "Full — VRAM is very tight, checkpoint all layers"


def build_checkpointing_options(
    vram_total_mb: Optional[float],
    batch_size: int = 1,
    chunk_duration_s: Optional[int] = None,
    num_layers: int = _DEFAULT_NUM_LAYERS,
    attn_backend: str = "sdpa",
    adapter_type: str = "lora",
    rank: int = 8,
    target_mlp: bool = False,
    optimizer_type: str = "adamw",
    offload_encoder: bool = True,
) -> List[Tuple[float, str, float]]:
    """Build menu options with estimated peak VRAM per checkpointing level.

    Returns a list of ``(ratio, label, est_mb)`` tuples, most to least
    checkpointing.
    """
    _LEVELS = [
        (1.0, "Full", "safest, ~20% slower"),
        (0.75, "Most", "~10% slower"),
        (0.5, "Partial", "balanced"),
        (0.25, "Minimal", "faster, more VRAM"),
        (0.0, "Off", "fastest, most VRAM"),
    ]
    options = []
    for ratio, name, speed in _LEVELS:
        peak, _ = estimate_peak_vram_mb(
            ratio, batch_size, chunk_duration_s, attn_backend,
            offload_encoder, adapter_type, rank, target_mlp,
            optimizer_type, num_layers,
        )
        ckpt = max(0, min(num_layers, round(num_layers * ratio)))
        label = f"{name} — {ckpt}/{num_layers} layers, ~{peak:.0f} MB est. ({speed})"
        options.append((ratio, label, peak))
    return options
