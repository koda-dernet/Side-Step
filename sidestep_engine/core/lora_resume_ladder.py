"""PEFT LoRA resume helpers: load adapter weights via a strategy ladder.

The resume path historically used ``decoder.load_state_dict(sd, strict=False)``
which silently drops LoRA keys when the on-disk format
(``base_model.model.<m>.lora_A.weight``) does not match the live PEFT-wrapped
model's format (``base_model.model.<m>.lora_A.default.weight``), causing
resume to appear successful while actually starting from fresh weights.

This module centralises a robust loader with three outcomes:

1. ``peft.set_peft_model_state_dict`` -- official adapter-name-aware loader.
2. Auto-probe remap -- tries four common key transforms and keeps the
   variant that overlaps the live model's keys the most.
3. Hard failure -- raises ``RuntimeError`` if no strategy matched a single
   LoRA key, so callers never resume with stale weights.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Tuple

import torch.nn as nn

logger = logging.getLogger(__name__)

_LORA_MARKERS = (".lora_", ".lora_embedding_")
_REMAP_VARIANTS = ("passthrough", "inject_adapter", "add_base_prefix", "prefix_and_inject")


@dataclass
class ResumeResult:
    """Outcome of a LoRA resume load."""

    strategy: str
    matched_lora_keys: int
    total_lora_keys: int
    warnings: List[str] = field(default_factory=list)


def _count_lora_keys(keys: Iterable[str]) -> int:
    """Return the number of keys that look like LoRA parameter names."""
    return sum(1 for k in keys if any(m in k for m in _LORA_MARKERS))


def _try_set_peft_state_dict(
    decoder: nn.Module, state_dict: Dict[str, Any], adapter_name: str,
) -> Optional[ResumeResult]:
    """Primary strategy: defer to PEFT's adapter-name-aware loader."""
    try:
        from peft import set_peft_model_state_dict
    except ImportError:
        return None
    total = _count_lora_keys(state_dict.keys())
    try:
        incompat = set_peft_model_state_dict(decoder, state_dict, adapter_name=adapter_name)
    except Exception as exc:  # pragma: no cover - defensive
        return ResumeResult("set_peft_model_state_dict", 0, total,
                            [f"set_peft_model_state_dict raised: {exc}"])
    unexpected = list(getattr(incompat, "unexpected_keys", []) or [])
    lora_unex = _count_lora_keys(unexpected)
    matched = max(0, total - lora_unex)
    warns = [f"{lora_unex} LoRA key(s) in checkpoint did not match live model"] if lora_unex else []
    return ResumeResult("set_peft_model_state_dict", matched, total, warns)


def _remap_key(key: str, variant: str, adapter_name: str) -> str:
    """Rewrite *key* per variant. See module docstring for variant semantics."""
    out = key
    if variant in ("add_base_prefix", "prefix_and_inject") and not out.startswith("base_model.model."):
        out = "base_model.model." + out
    if variant in ("inject_adapter", "prefix_and_inject") and any(m in out for m in _LORA_MARKERS):
        for suffix in (".weight", ".bias"):
            marker = f".{adapter_name}{suffix}"
            if out.endswith(suffix) and marker not in out:
                out = out[: -len(suffix)] + marker
                break
    return out


def _try_auto_remap(
    decoder: nn.Module, state_dict: Dict[str, Any], adapter_name: str,
) -> ResumeResult:
    """Fallback: probe four key transforms, keep the best-overlapping one."""
    live_keys = set(decoder.state_dict().keys())
    total = _count_lora_keys(state_dict.keys())
    keys = list(state_dict.keys())
    best_variant, best_score, best_mapping = "passthrough", -1, {}
    for variant in _REMAP_VARIANTS:
        mapping = {k: _remap_key(k, variant, adapter_name) for k in keys}
        score = sum(1 for v in mapping.values() if v in live_keys)
        if score > best_score:
            best_variant, best_score, best_mapping = variant, score, mapping
    strategy = f"auto_remap:{best_variant}"
    if best_score <= 0:
        return ResumeResult(strategy, 0, total,
                            ["auto_remap found no overlap between checkpoint and live keys"])
    remapped = {best_mapping[k]: v for k, v in state_dict.items() if best_mapping[k] in live_keys}
    decoder.load_state_dict(remapped, strict=False)
    return ResumeResult(strategy, _count_lora_keys(remapped.keys()), total, [])


def load_lora_resume_weights(
    decoder: nn.Module,
    state_dict: Dict[str, Any],
    *,
    adapter_name: str = "default",
) -> ResumeResult:
    """Load *state_dict* into PEFT-wrapped *decoder* via a strategy ladder.

    Args:
        decoder: Live LoRA-injected model (a ``PeftModel`` or equivalent).
        state_dict: Weights loaded from ``adapter_model.safetensors`` /
            ``adapter_model.bin``.
        adapter_name: PEFT adapter name used when the live model was wrapped.
            Defaults to ``"default"`` (PEFT's default).

    Returns:
        ``ResumeResult`` describing which strategy succeeded and how many
        LoRA keys were loaded.

    Raises:
        RuntimeError: if *state_dict* holds no LoRA keys, or if every
            strategy matched zero LoRA keys.
    """
    total = _count_lora_keys(state_dict.keys())
    if total == 0:
        raise RuntimeError(
            "LoRA resume failed: checkpoint state_dict contains no LoRA keys "
            "(searched for '.lora_' / '.lora_embedding_'). The file may be "
            "corrupted, empty, or from a different adapter format (e.g. LoKR)."
        )

    primary = _try_set_peft_state_dict(decoder, state_dict, adapter_name)
    if primary is not None and primary.matched_lora_keys > 0:
        return primary

    fallback = _try_auto_remap(decoder, state_dict, adapter_name)
    if fallback.matched_lora_keys > 0:
        if primary is not None:
            fallback.warnings.insert(0,
                f"primary strategy (set_peft_model_state_dict) matched 0/{total} keys; "
                f"fell back to {fallback.strategy}")
        return fallback

    details: List[str] = []
    if primary is not None:
        details.append(f"set_peft_model_state_dict matched 0/{total}")
    details.append(f"{fallback.strategy} matched 0/{total}")
    raise RuntimeError(
        "LoRA resume failed: no strategy matched any LoRA keys ("
        + "; ".join(details) + "). The checkpoint's adapter config likely "
        "differs from the live model (different rank, target_modules, or type)."
    )
