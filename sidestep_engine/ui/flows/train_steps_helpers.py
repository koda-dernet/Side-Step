"""
Training step estimation and warmup ratio helpers.

Extracted from ``train_steps_training.py`` to meet the module LOC policy.
"""

from __future__ import annotations

import math

from sidestep_engine.training_defaults import DEFAULT_EPOCHS
from sidestep_engine.ui.prompt_helpers import _esc, print_message


def show_dataset_step_estimate(a: dict) -> None:
    """Show dataset size and estimated steps/epoch after key params are set."""
    from pathlib import Path
    dataset_dir = a.get("dataset_dir")
    if not dataset_dir:
        return
    try:
        pt_count = len(list(Path(dataset_dir).glob("*.pt")))
    except Exception:
        return
    if pt_count == 0:
        return
    batch = a.get("batch_size", 1)
    accum = a.get("gradient_accumulation", 4)
    repeats = a.get("dataset_repeats", 1)
    effective_batch = max(1, batch * accum)
    effective_samples = pt_count * max(1, repeats)
    steps_per_epoch = max(1, math.ceil(effective_samples / effective_batch))
    epochs = a.get("epochs", DEFAULT_EPOCHS)
    total_steps = steps_per_epoch * epochs
    max_steps = a.get("max_steps", 0)

    info = (
        f"  Dataset: {pt_count} samples"
        + (f" × {repeats} repeats" if repeats > 1 else "")
        + f", ~{steps_per_epoch} optimizer steps/epoch, "
        f"~{total_steps} total steps for {epochs} epochs."
    )
    if max_steps > 0:
        est_epochs = max(1, math.ceil(max_steps / steps_per_epoch))
        info += f"\n  Max steps override: training will stop at {max_steps} steps (~{est_epochs} epochs)."
    print_message(info, kind="dim")


def estimate_total_steps(a: dict) -> int:
    """Estimate total optimizer steps from current answers. Returns 0 if unknown."""
    from pathlib import Path
    dataset_dir = a.get("dataset_dir")
    if not dataset_dir:
        return 0
    try:
        pt_count = len(list(Path(dataset_dir).glob("*.pt")))
    except Exception:
        return 0
    if pt_count == 0:
        return 0
    repeats = max(1, a.get("dataset_repeats", 1))
    effective_batch = max(1, a.get("batch_size", 1) * a.get("gradient_accumulation", 4))
    steps_per_epoch = max(1, math.ceil((pt_count * repeats) / effective_batch))
    max_steps = a.get("max_steps", 0)
    if max_steps > 0:
        return max_steps
    return steps_per_epoch * a.get("epochs", DEFAULT_EPOCHS)


def warn_warmup_ratio(a: dict) -> None:
    """Warn if warmup_steps is more than 25% of estimated total steps."""
    warmup = a.get("warmup_steps", 0)
    if warmup <= 0:
        return
    total = estimate_total_steps(a)
    if total <= 0:
        return
    ratio = warmup / total
    if ratio > 0.25:
        pct = int(ratio * 100)
        msg = (
            f"  Warmup ({warmup} steps) is {pct}% of estimated total steps ({total}).\n"
            f"  This means the model spends a large portion of training ramping up.\n"
            f"  Consider lowering warmup to ~{max(1, total // 10)} steps."
        )
        print_message(msg, kind="warn")


def smart_save_best_default(a: dict) -> int:
    """Return a sensible default for save_best_after based on dataset size.

    Ensures tracking starts after warmup is complete plus a small buffer,
    so the model has stabilized before we begin comparing losses.
    """
    warmup = a.get("warmup_steps", 100)
    total = estimate_total_steps(a)
    if total <= 0:
        return max(warmup + 10, 200)
    return max(warmup + 10, min(200, total // 10))
