"""Adaptive rank assignment from Fisher + spectral signals.

Selects which modules to include (by Fisher threshold) and sizes
each module's LoRA rank using ``base_rank * sqrt(fisher * spectral)``
centred on the median of included modules.
"""

from __future__ import annotations

import logging
import math
from typing import Dict, List, Tuple

logger = logging.getLogger(__name__)


def assign_ranks(
    fisher_scores: Dict[str, float],
    spectral_ranks: Dict[str, int],
    base_rank: int = 64,
    rank_min: int = 16,
    rank_max: int = 128,
    inclusion_percentile: float = 0.55,
) -> Tuple[List[str], Dict[str, int], Dict[str, int]]:
    """Select modules and assign adaptive LoRA ranks.

    Args:
        fisher_scores: ``{module_name: fisher_score}``.
        spectral_ranks: ``{module_name: effective_rank}`` (-1 = unknown).
        base_rank: User's chosen base rank (median target).
        rank_min: Floor for assigned ranks.
        rank_max: Ceiling for assigned ranks.
        inclusion_percentile: Top fraction of modules to include by
            Fisher score (e.g. 0.55 = top 55%).

    Returns:
        ``(target_modules, rank_pattern, alpha_pattern)`` where
        *target_modules* is a list of short module suffixes for PEFT,
        and the patterns map module-name keys to int values.
    """
    if not fisher_scores:
        return [], {}, {}

    included = _select_modules(fisher_scores, inclusion_percentile)
    if not included:
        return [], {}, {}

    rank_pattern: Dict[str, int] = {}
    alpha_pattern: Dict[str, int] = {}

    f_vals = [fisher_scores[n] for n in included]
    f_median = _median(f_vals) or 1e-12

    s_vals = [spectral_ranks.get(n, -1) for n in included]
    s_valid = [v for v in s_vals if v > 0]
    s_median = _median(s_valid) if s_valid else 1.0

    for name in included:
        f_factor = fisher_scores[name] / f_median
        s_raw = spectral_ranks.get(name, -1)
        s_factor = (s_raw / s_median) if s_raw > 0 else 1.0
        raw = base_rank * math.sqrt(f_factor * s_factor)
        rank = _round_to_multiple(raw, 8)
        rank = max(rank_min, min(rank_max, rank))
        peft_key = _to_peft_key(name)
        rank_pattern[peft_key] = rank
        alpha_pattern[peft_key] = rank * 2

    target_modules = _derive_target_module_suffixes(included)
    return target_modules, rank_pattern, alpha_pattern


_CATEGORY_MARKERS = {
    "cross_attn": ".cross_attn.",
    "mlp": ".mlp.",
}

_MIN_CATEGORY_FRACTION = 0.10


def _classify_module(name: str) -> str:
    """Return the category of a module: 'cross_attn', 'mlp', or 'self_attn'."""
    for cat, marker in _CATEGORY_MARKERS.items():
        if marker in name:
            return cat
    return "self_attn"


def _select_modules(
    fisher_scores: Dict[str, float],
    percentile: float,
) -> List[str]:
    """Return module names in the top *percentile* by Fisher score.

    Guarantees minimum representation per module category (self_attn,
    cross_attn, mlp) so that no category is entirely excluded.  For each
    category present in *fisher_scores*, at least 10% of that category's
    modules (minimum 1) are included even if they fall below the global
    percentile cutoff.
    """
    ranked = sorted(fisher_scores, key=fisher_scores.get, reverse=True)  # type: ignore[arg-type]
    n = max(1, int(len(ranked) * percentile))
    selected = set(ranked[:n])

    categories: Dict[str, List[str]] = {}
    for name in ranked:
        cat = _classify_module(name)
        categories.setdefault(cat, []).append(name)

    for cat, members in categories.items():
        count_before = sum(1 for m in members if m in selected)
        floor = max(1, int(len(members) * _MIN_CATEGORY_FRACTION))
        if count_before >= floor:
            continue
        for m in members:
            if m not in selected:
                selected.add(m)
                count_before += 1
            if count_before >= floor:
                break
        logger.info(
            "Category '%s' underrepresented; promoted to %d module(s) "
            "(floor=%d, total_in_category=%d)",
            cat, count_before, floor, len(members),
        )

    return [name for name in ranked if name in selected]


def _to_peft_key(full_name: str) -> str:
    """Convert a full module name to the key PEFT uses for rank_pattern.

    PEFT matches rank_pattern keys as substrings of the full module path.
    We strip the leading ``decoder.`` prefix if present.
    """
    if full_name.startswith("decoder."):
        return full_name[len("decoder."):]
    return full_name


def _derive_target_module_suffixes(included: List[str]) -> List[str]:
    """Extract unique short suffixes needed for PEFT ``target_modules``.

    E.g. ``decoder.layers.0.self_attn.q_proj`` → ``self_attn.q_proj``.
    Deduplicates so PEFT's regex matching can find all layers.
    """
    seen = set()
    result = []
    for name in included:
        parts = name.split(".")
        # Find the suffix starting from the attention/mlp component
        for i, p in enumerate(parts):
            if p in ("self_attn", "cross_attn", "mlp"):
                suffix = ".".join(parts[i:])
                if suffix not in seen:
                    seen.add(suffix)
                    result.append(suffix)
                break
    return sorted(result)


def _round_to_multiple(value: float, multiple: int) -> int:
    """Round *value* to the nearest *multiple*."""
    return int(round(value / multiple)) * multiple


def _median(values: list) -> float:
    """Return the median of a list of numbers."""
    if not values:
        return 0.0
    s = sorted(values)
    mid = len(s) // 2
    if len(s) % 2 == 0:
        return (s[mid - 1] + s[mid]) / 2.0
    return float(s[mid])
