"""Fisher diagonal estimation engine.

Performs forward+backward runs through the decoder, accumulating
squared gradients (Fisher diagonal) per targetable module.
Supports timestep-focus masking and multi-pass processing for low VRAM.
"""

from __future__ import annotations

import gc
import logging
from contextlib import nullcontext
from typing import Callable, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from acestep.training_v2.timestep_sampling import sample_timesteps

logger = logging.getLogger(__name__)


def sample_focused_timestep(
    batch_size: int,
    focus: str,
    device: torch.device,
    dtype: torch.dtype,
    timestep_mu: float = -0.4,
    timestep_sigma: float = 1.0,
    data_proportion: float = 0.5,
) -> torch.Tensor:
    """Sample timesteps within the requested focus range.

    Args:
        batch_size: Number of timestep samples.
        focus: ``"texture"`` (0,0.4), ``"structure"`` (0.6,1.0),
            ``"balanced"`` (training distribution), or ``"low,high"``.
        device: Target device.
        dtype: Target dtype.

    Returns:
        Tensor of shape ``[batch_size]`` in the focus range.
    """
    if focus == "texture":
        low, high = 0.0, 0.4
    elif focus == "structure":
        low, high = 0.6, 1.0
    elif focus == "balanced":
        t, _ = sample_timesteps(
            batch_size,
            device,
            dtype,
            data_proportion=data_proportion,
            timestep_mu=timestep_mu,
            timestep_sigma=timestep_sigma,
            use_meanflow=False,
        )
        return t
    else:
        parts = focus.split(",")
        low, high = float(parts[0]), float(parts[1])

    return low + torch.rand((batch_size,), device=device, dtype=dtype) * (high - low)


def single_fisher_run(
    model: nn.Module,
    loader_factory: Callable,
    param_to_module: Dict[str, str],
    target_names: List[str],
    max_batches: int,
    timestep_focus: str,
    device: torch.device,
    dtype: torch.dtype,
    chunk_groups: Optional[List[Tuple[str, List[str]]]] = None,
    patience: int = 5,
    top_k_for_convergence: int = 20,
    progress_callback: Optional[Callable] = None,
    cancel_check: Optional[Callable] = None,
    timestep_mu: float = -0.4,
    timestep_sigma: float = 1.0,
    data_proportion: float = 0.5,
) -> Dict[str, float]:
    """Execute one Fisher estimation run over the dataset.

    When *chunk_groups* is provided, each group of modules is processed
    in a **separate full pass** over the data (fresh DataLoader each
    time).  Only one group's gradients exist on GPU at a time, cutting
    peak gradient memory from ~3.75 GB to ~0.8-2.3 GB.

    Args:
        model: Full ACE-Step model with ``.decoder``.
        loader_factory: Callable that returns a fresh DataLoader each
            time (e.g. ``data_module.train_dataloader``).
        param_to_module: ``{param_name: module_name}`` mapping.
        target_names: All targetable module names.
        max_batches: Upper bound on batches to process.
        timestep_focus: Timestep focus mode string.
        device: Torch device.
        dtype: Torch dtype.
        chunk_groups: If set, run a separate data pass per group
            (for low VRAM).  ``None`` = full mode (one pass).
        patience: Early-stop when top-K ranking stable for this many
            consecutive batches.
        top_k_for_convergence: Number of top modules tracked for
            convergence checking.
        progress_callback: ``(batch, total, msg) -> None``.
        cancel_check: ``() -> bool``.

    Returns:
        ``{module_name: fisher_score}`` (sum-of-squared-grads / param-count).
    """
    if chunk_groups:
        return _run_chunked(
            model, loader_factory, param_to_module, target_names,
            max_batches, timestep_focus, device, dtype,
            chunk_groups, patience, top_k_for_convergence,
            progress_callback, cancel_check,
            timestep_mu=timestep_mu,
            timestep_sigma=timestep_sigma,
            data_proportion=data_proportion,
        )
    return _run_full(
        model, loader_factory(), param_to_module, target_names,
        max_batches, timestep_focus, device, dtype,
        patience, top_k_for_convergence,
        progress_callback, cancel_check,
        timestep_mu=timestep_mu,
        timestep_sigma=timestep_sigma,
        data_proportion=data_proportion,
    )


def _run_full(
    model, loader, param_to_module, target_names,
    max_batches, timestep_focus, device, dtype,
    patience, top_k_for_convergence,
    progress_callback, cancel_check,
    timestep_mu=-0.4, timestep_sigma=1.0, data_proportion=0.5,
) -> Dict[str, float]:
    """All modules in one backward pass per batch (needs more VRAM)."""
    autocast_ctx = _make_autocast(device, dtype)
    fisher_accum: Dict[str, float] = {n: 0.0 for n in target_names}
    param_count: Dict[str, int] = {n: 0 for n in target_names}
    all_set = set(target_names)
    prev_ranking: List[str] = []
    stable_count = 0
    oom_count = 0
    batches_done = 0

    for batch in loader:
        if batches_done >= max_batches:
            break
        if cancel_check and cancel_check():
            break
        try:
            _run_one_backward(
                model, batch, param_to_module, all_set,
                fisher_accum, param_count, timestep_focus,
                device, dtype, autocast_ctx,
                    timestep_mu=timestep_mu,
                    timestep_sigma=timestep_sigma,
                    data_proportion=data_proportion,
            )
            oom_count = 0
        except torch.cuda.OutOfMemoryError:
            oom_count += 1
            _clear_cache(device)
            logger.warning("OOM on batch %d (oom_count=%d)", batches_done, oom_count)
            if oom_count >= 2:
                logger.warning(
                    "Repeated OOM. Stopping run early at %d/%d batches.",
                    batches_done, max_batches,
                )
                break
            continue

        batches_done += 1
        if progress_callback:
            progress_callback(batches_done, max_batches, "fisher")

        snapshot = _current_ranking(fisher_accum, param_count, top_k_for_convergence)
        if snapshot == prev_ranking:
            stable_count += 1
            if stable_count >= patience:
                logger.info("Fisher converged after %d batches", batches_done)
                break
        else:
            stable_count = 0
        prev_ranking = snapshot

    return _normalise(fisher_accum, param_count, target_names, batches_done)


def _run_chunked(
    model, loader_factory, param_to_module, target_names,
    max_batches, timestep_focus, device, dtype,
    chunk_groups, patience, top_k_for_convergence,
    progress_callback, cancel_check,
    timestep_mu=-0.4, timestep_sigma=1.0, data_proportion=0.5,
) -> Dict[str, float]:
    """Process each module group in a separate full data pass.

    A fresh DataLoader is created for each group so every group sees
    the same data.  Only one group's gradients exist on GPU at a time,
    cutting peak gradient memory from ~3.75 GB to ~0.8-2.3 GB.
    """
    autocast_ctx = _make_autocast(device, dtype)
    fisher_accum: Dict[str, float] = {n: 0.0 for n in target_names}
    param_count: Dict[str, int] = {n: 0 for n in target_names}
    total_batches_done = 0

    total_groups = len(chunk_groups)
    for g_idx, (group_name, group_modules) in enumerate(chunk_groups):
        group_set = set(group_modules)
        logger.info(
            "Chunked pass %d/%d (%s, %d modules)",
            g_idx + 1, total_groups, group_name, len(group_modules),
        )
        oom_count = 0
        batches_done = 0

        # Fresh loader for each group so each sees the full dataset
        for batch in loader_factory():
            if batches_done >= max_batches:
                break
            if cancel_check and cancel_check():
                break
            try:
                _run_one_backward(
                    model, batch, param_to_module, group_set,
                    fisher_accum, param_count, timestep_focus,
                    device, dtype, autocast_ctx,
                    timestep_mu=timestep_mu,
                    timestep_sigma=timestep_sigma,
                    data_proportion=data_proportion,
                )
                oom_count = 0
            except torch.cuda.OutOfMemoryError:
                oom_count += 1
                _clear_cache(device)
                logger.warning(
                    "OOM on %s batch %d (oom_count=%d)",
                    group_name, batches_done, oom_count,
                )
                if oom_count >= 2:
                    logger.warning(
                        "Repeated OOM on %s. Stopping this pass early.", group_name,
                    )
                    break
                continue

            batches_done += 1
            if progress_callback:
                done_so_far = g_idx * max_batches + batches_done
                total_work = total_groups * max_batches
                progress_callback(done_so_far, total_work, f"fisher/{group_name}")

        total_batches_done = max(total_batches_done, batches_done)

        # Clean up between passes
        _clear_cache(device)

    return _normalise(fisher_accum, param_count, target_names, total_batches_done)


# ---------------------------------------------------------------------------
# Core single-batch forward+backward
# ---------------------------------------------------------------------------

def _run_one_backward(
    model: nn.Module,
    batch: dict,
    param_to_module: Dict[str, str],
    active_modules: set,
    fisher_accum: Dict[str, float],
    param_count: Dict[str, int],
    timestep_focus: str,
    device: torch.device,
    dtype: torch.dtype,
    autocast_ctx,
    timestep_mu: float = -0.4,
    timestep_sigma: float = 1.0,
    data_proportion: float = 0.5,
) -> None:
    """Single forward+backward, accumulating Fisher for *active_modules*."""
    for pname, param in model.named_parameters():
        mod = param_to_module.get(pname)
        param.requires_grad_(mod is not None and mod in active_modules)

    target_latents = attention_mask = encoder_hidden_states = None
    encoder_attention_mask = context_latents = None
    x0 = x1 = xt = t = t_ = decoder_outputs = flow = loss = None

    try:
        target_latents = batch["target_latents"].to(device, dtype=dtype)
        attention_mask = batch["attention_mask"].to(device, dtype=dtype)
        encoder_hidden_states = batch["encoder_hidden_states"].to(device, dtype=dtype)
        encoder_attention_mask = batch["encoder_attention_mask"].to(device, dtype=dtype)
        context_latents = batch["context_latents"].to(device, dtype=dtype)

        bsz = target_latents.shape[0]

        with autocast_ctx:
            x0 = target_latents
            x1 = torch.randn_like(x0)
            t = sample_focused_timestep(
                bsz,
                timestep_focus,
                device,
                dtype,
                timestep_mu=timestep_mu,
                timestep_sigma=timestep_sigma,
                data_proportion=data_proportion,
            )
            t_ = t.unsqueeze(-1).unsqueeze(-1)
            xt = t_ * x1 + (1.0 - t_) * x0
            xt.requires_grad_(True)

            decoder_outputs = model.decoder(
                hidden_states=xt, timestep=t, timestep_r=t,
                attention_mask=attention_mask,
                encoder_hidden_states=encoder_hidden_states,
                encoder_attention_mask=encoder_attention_mask,
                context_latents=context_latents,
            )
            flow = x1 - x0
            loss = F.mse_loss(decoder_outputs[0], flow)

        loss.backward()

        with torch.no_grad():
            for pname, param in model.named_parameters():
                if param.grad is not None and pname in param_to_module:
                    mod = param_to_module[pname]
                    if mod in active_modules:
                        fisher_accum[mod] += (
                            param.grad.detach().float().pow(2).sum().item()
                        )
                        param_count[mod] += param.grad.numel()
    finally:
        model.zero_grad(set_to_none=True)
        for param in model.parameters():
            param.requires_grad_(False)
        del target_latents, attention_mask, encoder_hidden_states
        del encoder_attention_mask, context_latents
        del x0, x1, xt, t, t_, decoder_outputs, flow, loss
        _clear_cache(device)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _normalise(
    accum: Dict[str, float], counts: Dict[str, int],
    names: List[str], batches: int,
) -> Dict[str, float]:
    """Normalise accumulated squared gradients to per-parameter Fisher.

    ``counts[n]`` already includes one ``numel()`` per processed batch,
    so dividing by *counts* alone gives the correct per-element,
    per-batch average: ``E_x[(dL/dθ_i)²]``.
    """
    scores: Dict[str, float] = {}
    for name in names:
        pc = counts.get(name, 0)
        scores[name] = accum[name] / pc if pc > 0 else 0.0
    return scores


def _current_ranking(
    accum: Dict[str, float], counts: Dict[str, int], k: int,
) -> List[str]:
    """Rank modules by per-parameter Fisher (accum / counts)."""
    scores = {}
    for n in accum:
        c = counts.get(n, 0)
        scores[n] = accum[n] / c if c > 0 else 0.0
    ranked = sorted(scores, key=scores.get, reverse=True)  # type: ignore[arg-type]
    return ranked[:k]


def _make_autocast(device: torch.device, dtype: torch.dtype):
    dev = str(device)
    if "cuda" in dev:
        return torch.autocast("cuda", dtype=dtype)
    if "xpu" in dev:
        return torch.autocast("xpu", dtype=dtype)
    if "mps" in dev:
        return torch.autocast("mps", dtype=dtype)
    return nullcontext()


def _clear_cache(device: torch.device) -> None:
    gc.collect()
    dev = str(device)
    if "cuda" in dev and torch.cuda.is_available():
        torch.cuda.empty_cache()
    elif "mps" in dev and hasattr(torch, "mps"):
        torch.mps.empty_cache()
    elif "xpu" in dev and hasattr(torch, "xpu"):
        torch.xpu.empty_cache()
