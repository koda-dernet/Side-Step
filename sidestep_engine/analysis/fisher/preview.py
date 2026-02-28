"""Mandatory preview for Fisher analysis results.

Prints a summary table of module selection, rank distribution, and
parameter counts, then asks the user to confirm before saving.
"""

from __future__ import annotations

import sys
from collections import Counter
from typing import Any, Dict, List, Optional


def print_preview(
    fisher_scores: Dict[str, float],
    fisher_stds: Dict[str, float],
    spectral_ranks: Dict[str, int],
    rank_pattern: Dict[str, int],
    alpha_pattern: Dict[str, int],
    target_modules: List[str],
    excluded: List[str],
    base_rank: int,
    rank_min: int,
    rank_max: int,
    total_batches: int,
    num_runs: int,
    variant: str,
    timestep_focus: str,
    num_analyzed: int,
    sample_coverage: Optional[List[Dict[str, Any]]] = None,
) -> None:
    """Print the analysis preview to stdout."""
    _focus_desc = _describe_focus(timestep_focus)
    _divider = "\u2500" * 55

    print(f"\n  Side-Step Fisher + Spectral Analysis")
    print(f"  {_divider}")
    print(f"  Model: {variant} | Batches: {total_batches} "
          f"({num_runs} runs) | Modules analyzed: {num_analyzed}")
    print(f"  Timestep focus: {_focus_desc}")
    if sample_coverage:
        avg_cov = sum(float(r.get("coverage_ratio", 0.0)) for r in sample_coverage) / max(len(sample_coverage), 1)
        avg_count = sum(int(r.get("selected_count", 0)) for r in sample_coverage) / max(len(sample_coverage), 1)
        total_count = int(sample_coverage[0].get("total_count", 0))
        print(
            f"  Sample coverage: {avg_count:.1f}/{total_count} per run "
            f"(avg {avg_cov * 100:.1f}%)"
        )

    # Module selection breakdown
    n_incl = len(rank_pattern)
    n_excl = len(excluded)
    sa = sum(1 for k in rank_pattern if "self_attn" in k)
    ca = sum(1 for k in rank_pattern if "cross_attn" in k)
    mlp = sum(1 for k in rank_pattern if "mlp" in k or "gate_proj" in k
              or "up_proj" in k or "down_proj" in k)

    print(f"\n  Module Selection: {n_incl} included, {n_excl} excluded")
    print(f"    Self-attention:  {sa} modules")
    print(f"    Cross-attention: {ca} modules")
    print(f"    MLP:             {mlp} modules")

    # Rank distribution histogram
    dist = Counter(rank_pattern.values())
    print(f"\n  Rank Distribution (base rank = {base_rank}):")
    for r in sorted(dist):
        marker = "  \u2190 base" if r == base_rank else ""
        tag = " (min)" if r == rank_min else (" (max)" if r == rank_max else "")
        print(f"    rank {r:>4d}: {dist[r]:>3d} modules{tag}{marker}")

    # Top 5 modules by assigned rank
    top5 = sorted(rank_pattern.items(), key=lambda x: x[1], reverse=True)[:5]
    if top5:
        print(f"\n  Top 5 (highest rank assigned):")
        for name, rank in top5:
            f_score = fisher_scores.get(f"decoder.{name}", fisher_scores.get(name, 0.0))
            er = spectral_ranks.get(f"decoder.{name}", spectral_ranks.get(name, -1))
            er_str = str(er) if er > 0 else "?"
            print(f"    {name:<48s}  F={_fmt_fisher(f_score)}  ER={er_str:>3s}  "
                  f"\u2192 rank {rank}")

    # Low-confidence modules
    lc = _low_confidence_modules(fisher_scores, fisher_stds)
    if lc:
        print(f"\n  Low-Confidence Modules (cross-run std > 50% of mean):")
        for name, mean, std in lc[:3]:
            print(f"    \u26a0 {name}  F={_fmt_fisher(mean)}\u00b1{_fmt_fisher(std)}")

    print(f"  {_divider}\n")


def ask_confirmation(default_yes: bool = False) -> bool:
    """Prompt the user to confirm saving the Fisher map.

    Args:
        default_yes: If True, default answer is yes.

    Returns:
        True if the user confirms.
    """
    prompt = "  Save fisher_map.json? [y/N] " if not default_yes else "  Save fisher_map.json? [Y/n] "
    try:
        answer = input(prompt).strip().lower()
    except (KeyboardInterrupt, EOFError):
        print()
        return False
    if not answer:
        return default_yes
    return answer in ("y", "yes")


def _fmt_fisher(val: float) -> str:
    """Format a Fisher score for display, using scientific notation for tiny values."""
    if val == 0.0:
        return "0       "
    if abs(val) >= 0.001:
        return f"{val:.5f}"
    return f"{val:.2e}"


def _describe_focus(focus: str) -> str:
    """Human-readable description of the timestep focus."""
    if focus == "texture":
        return "texture (t < 0.4) -- measuring style/timbre sensitivity"
    if focus == "structure":
        return "structure (t > 0.6) -- measuring rhythm/arrangement sensitivity"
    if focus == "balanced":
        return "balanced -- full timestep distribution"
    return f"custom ({focus})"


def _low_confidence_modules(
    scores: Dict[str, float],
    stds: Dict[str, float],
) -> list:
    """Return modules where cross-run std exceeds 50% of the mean."""
    result = []
    for name, mean in scores.items():
        std = stds.get(name, 0.0)
        if mean > 0 and std > 0.5 * mean:
            result.append((name, mean, std))
    return sorted(result, key=lambda x: x[2] / max(x[1], 1e-12), reverse=True)
