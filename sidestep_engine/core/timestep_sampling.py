"""
Timestep Sampling and CFG Dropout for ACE-Step Training V2

Provides two sampling strategies:

- **Continuous logit-normal** (base/sft): reimplements ``sample_t_r()``
  from ``modeling_acestep_v15_turbo.py`` lines 169-194.
- **Discrete 8-step** (turbo): samples uniformly from the turbo
  ``shift=3.0`` inference schedule so the LoRA trains at exactly the
  timestep values used during inference.

Also provides ``apply_cfg_dropout()`` matching lines 1691-1699 of the
same file.
"""

from __future__ import annotations

from typing import List, Optional

import torch


# ---------------------------------------------------------------------------
# Turbo discrete timestep schedule (shift=3.0, 8 inference steps)
# ---------------------------------------------------------------------------

TURBO_SHIFT3_TIMESTEPS: List[float] = [
    1.0, 0.9545454545454546, 0.9, 0.8333333333333334,
    0.75, 0.6428571428571429, 0.5, 0.3,
]
"""Discrete timestep schedule for turbo inference with ``shift=3.0``.

Matches ``SHIFT_TIMESTEPS[3.0]`` in
``modeling_acestep_v15_turbo.py`` line 1822.
"""


# ---------------------------------------------------------------------------
# Discrete timestep sampling (turbo)
# ---------------------------------------------------------------------------

def sample_discrete_timesteps(
    batch_size: int,
    device: torch.device | str,
    dtype: torch.dtype,
    timesteps: Optional[List[float]] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample uniformly from a discrete timestep schedule.

    Each sample in the batch independently picks one of the schedule
    values with equal probability.  Used for turbo LoRA training so the
    adapter trains at exactly the timestep values it will see during
    8-step inference.

    Args:
        batch_size: Number of samples.
        device: Torch device.
        dtype: Tensor dtype.
        timesteps: Schedule values.  Defaults to
            :data:`TURBO_SHIFT3_TIMESTEPS`.

    Returns:
        ``(t, r)`` -- each of shape ``[batch_size]``.  ``r == t``
        (matching ``use_meanflow=False``).
    """
    if timesteps is None:
        timesteps = TURBO_SHIFT3_TIMESTEPS
    ts = torch.tensor(timesteps, device=device, dtype=dtype)
    indices = torch.randint(0, ts.shape[0], (batch_size,), device=device)
    t = ts[indices]
    return t, t  # r = t


# ---------------------------------------------------------------------------
# Continuous logit-normal timestep sampling (base/sft)
# ---------------------------------------------------------------------------

def sample_timesteps(
    batch_size: int,
    device: torch.device | str,
    dtype: torch.dtype,
    data_proportion: float = 0.0,
    timestep_mu: float = -0.4,
    timestep_sigma: float = 1.0,
    use_meanflow: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample timestep ``t`` and ``r`` for flow matching training.

    This is a faithful reimplementation of ``sample_t_r()`` from
    ``checkpoints/acestep-v15-turbo/modeling_acestep_v15_turbo.py``
    lines 169-194.

    All ACE-Step model variants call this with ``use_meanflow=False``
    during training, which forces ``data_proportion=1.0`` and therefore
    ``r = t`` for every sample in the batch.

    Args:
        batch_size: Number of samples.
        device: Torch device.
        dtype: Tensor dtype (e.g. ``torch.bfloat16``).
        data_proportion: Proportion of data samples (from model config).
        timestep_mu: Mean for logit-normal sampling (from model config).
        timestep_sigma: Std  for logit-normal sampling (from model config).
        use_meanflow: Whether to use mean-flow (``False`` for all current
            ACE-Step variants during training).

    Returns:
        ``(t, r)`` -- each of shape ``[batch_size]``.
    """
    # Logit-normal sampling via sigmoid(N(mu, sigma))
    t = torch.sigmoid(
        torch.randn((batch_size,), device=device, dtype=dtype) * timestep_sigma + timestep_mu
    )
    r = torch.sigmoid(
        torch.randn((batch_size,), device=device, dtype=dtype) * timestep_sigma + timestep_mu
    )

    # Assign t = max, r = min for each pair
    t, r = torch.maximum(t, r), torch.minimum(t, r)

    # When use_meanflow is False the model forces data_proportion = 1.0,
    # which makes r = t for *every* sample (the zero_mask covers the full
    # batch).
    if not use_meanflow:
        data_proportion = 1.0

    data_size = int(batch_size * data_proportion)
    zero_mask = torch.arange(batch_size, device=device) < data_size
    r = torch.where(zero_mask, t, r)

    return t, r


# ---------------------------------------------------------------------------
# CFG dropout
# ---------------------------------------------------------------------------

def apply_cfg_dropout(
    encoder_hidden_states: torch.Tensor,
    null_condition_emb: torch.Tensor,
    cfg_ratio: float = 0.15,
) -> torch.Tensor:
    """Apply classifier-free guidance dropout to condition embeddings.

    Faithful reimplementation of lines 1691-1699 of
    ``modeling_acestep_v15_turbo.py``:

    .. code-block:: python

        full_cfg_condition_mask = torch.where(
            (torch.rand(size=(bsz,), device=device, dtype=dtype) < cfg_ratio),
            torch.zeros(size=(bsz,), device=device, dtype=dtype),
            torch.ones(size=(bsz,), device=device, dtype=dtype),
        ).view(-1, 1, 1)
        encoder_hidden_states = torch.where(
            full_cfg_condition_mask > 0,
            encoder_hidden_states,
            self.null_condition_emb.expand_as(encoder_hidden_states),
        )

    Args:
        encoder_hidden_states: Condition embeddings ``[B, L, D]``.
        null_condition_emb: Null (unconditional) embedding.  Will be
            ``expand_as``'d to match ``encoder_hidden_states``.
        cfg_ratio: Probability of replacing a sample's condition with
            the null embedding (default 0.15).

    Returns:
        Modified ``encoder_hidden_states`` with some samples replaced.
    """
    bsz = encoder_hidden_states.shape[0]
    device = encoder_hidden_states.device
    dtype = encoder_hidden_states.dtype

    # Per-sample mask: 0 = drop condition (replace with null), 1 = keep
    full_cfg_condition_mask = torch.where(
        torch.rand(size=(bsz,), device=device, dtype=dtype) < cfg_ratio,
        torch.zeros(size=(bsz,), device=device, dtype=dtype),
        torch.ones(size=(bsz,), device=device, dtype=dtype),
    ).view(-1, 1, 1)

    encoder_hidden_states = torch.where(
        full_cfg_condition_mask > 0,
        encoder_hidden_states,
        null_condition_emb.expand_as(encoder_hidden_states),
    )

    return encoder_hidden_states
