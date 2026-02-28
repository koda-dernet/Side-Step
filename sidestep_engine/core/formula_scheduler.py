"""
Custom LR schedule via user-defined math formulas.

Provides a safe ``eval()``-based LambdaLR scheduler that lets users write
arbitrary post-warmup LR curves.  Warmup is auto-prepended using the same
``SequentialLR`` pattern as all built-in schedulers.

Available formula variables (post-warmup phase):
    step, total_steps (remaining after warmup), progress (0→1),
    epoch, total_epochs, steps_per_epoch, base_lr.
Math functions: cos, sin, exp, log, sqrt, pi, e, min, max, abs, clamp.
"""

from __future__ import annotations

import logging
import math
from typing import Optional

import torch
from torch.optim.lr_scheduler import LambdaLR, LinearLR, SequentialLR

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Safe eval namespace
# ---------------------------------------------------------------------------

SAFE_NAMESPACE: dict = {
    "pi": math.pi,
    "e": math.e,
    "cos": math.cos,
    "sin": math.sin,
    "exp": math.exp,
    "log": math.log,
    "sqrt": math.sqrt,
    "pow": pow,
    "min": min,
    "max": max,
    "abs": abs,
    "clamp": lambda x, lo, hi: max(lo, min(hi, x)),
}

# ---------------------------------------------------------------------------
# Preset formula templates
# ---------------------------------------------------------------------------

FORMULA_PRESETS: list[tuple[str, str, str]] = [
    (
        "cosine",
        "Cosine decay (same shape as built-in)",
        "base_lr * 0.5 * (1 + cos(pi * progress))",
    ),
    (
        "cosine_floor",
        "Cosine decay with floor (never below 1e-6)",
        "max(1e-6, base_lr * 0.5 * (1 + cos(pi * progress)))",
    ),
    (
        "constant_then_cosine",
        "Constant for 30% of training, then cosine decay",
        "base_lr if progress < 0.3 else base_lr * 0.5 * (1 + cos(pi * (progress - 0.3) / 0.7))",
    ),
    (
        "step_decay",
        "Halve LR every 10 epochs",
        "base_lr * 0.5 ** (epoch // 10)",
    ),
    (
        "linear_floor",
        "Linear decay but never below 1e-5",
        "max(1e-5, base_lr * (1 - progress))",
    ),
]


def _eval_formula(
    code: object,
    step: int,
    total_steps: int,
    base_lr: float,
    steps_per_epoch: int,
    total_epochs: int,
) -> float:
    """Evaluate a compiled formula at one point and return the raw float."""
    progress = step / max(1, total_steps)
    epoch = step / max(1, steps_per_epoch)
    ns = {
        **SAFE_NAMESPACE,
        "step": step,
        "total_steps": total_steps,
        "base_lr": base_lr,
        "progress": progress,
        "epoch": epoch,
        "total_epochs": total_epochs,
        "steps_per_epoch": steps_per_epoch,
    }
    return float(eval(code, {"__builtins__": {}}, ns))  # noqa: S307


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate_formula(
    formula: str,
    base_lr: float = 1e-4,
    total_steps: int = 1000,
    warmup_steps: int = 100,
    steps_per_epoch: int = 100,
    total_epochs: int = 10,
) -> Optional[str]:
    """Compile and test-evaluate a formula at three points.

    Returns an error message string on failure, or ``None`` on success.
    """
    if not formula or not formula.strip():
        return "Formula cannot be empty"
    try:
        code = compile(formula.strip(), "<lr_formula>", "eval")
    except SyntaxError as exc:
        return f"Syntax error in formula: {exc}"

    remaining = max(1, total_steps - warmup_steps)
    test_points = [0, remaining // 2, max(0, remaining - 1)]
    for pt in test_points:
        try:
            result = _eval_formula(
                code, pt, remaining, base_lr, steps_per_epoch, total_epochs,
            )
        except Exception as exc:
            return f"Formula error at step {pt}: {exc}"
        if not math.isfinite(result):
            return f"Formula returned {result} at step {pt} (must be finite)"
    return None


_NEAR_ZERO_THRESHOLD = 1e-8


def check_formula_warnings(
    formula: str,
    base_lr: float = 1e-4,
    total_steps: int = 1000,
    warmup_steps: int = 100,
    steps_per_epoch: int = 100,
    total_epochs: int = 10,
) -> list[str]:
    """Return soft warnings about a formula that passed validation.

    Checks for negative values (clamped to 0 at runtime), near-zero
    results, and flatlined tail.
    """
    warnings: list[str] = []
    try:
        code = compile(formula.strip(), "<lr_formula>", "eval")
    except SyntaxError:
        return warnings

    remaining = max(1, total_steps - warmup_steps)
    test_points = [0, remaining // 4, remaining // 2,
                   3 * remaining // 4, max(0, remaining - 1)]
    neg_steps: list[int] = []
    near_zero_steps: list[int] = []
    for pt in test_points:
        try:
            result = _eval_formula(
                code, pt, remaining, base_lr, steps_per_epoch, total_epochs,
            )
            if result < 0:
                neg_steps.append(pt)
            elif abs(result) < _NEAR_ZERO_THRESHOLD:
                near_zero_steps.append(pt)
        except Exception:
            logger.debug("Formula check error at test point %d", pt, exc_info=True)

    if neg_steps:
        warnings.append(
            f"Formula goes negative at {len(neg_steps)} of {len(test_points)} "
            f"test points (will be clamped to 0). "
            f"Consider wrapping with max(0, ...) for clarity."
        )
    if near_zero_steps:
        warnings.append(
            f"Formula returns near-zero (<{_NEAR_ZERO_THRESHOLD}) at "
            f"{len(near_zero_steps)} of {len(test_points)} test points. "
            f"Training may stall — check your formula logic."
        )
    return warnings


# ---------------------------------------------------------------------------
# Preview
# ---------------------------------------------------------------------------

def preview_formula(
    formula: str,
    base_lr: float = 1e-4,
    total_steps: int = 1000,
    warmup_steps: int = 100,
    steps_per_epoch: int = 100,
    total_epochs: int = 10,
) -> tuple[float, float, float]:
    """Return ``(start_lr, mid_lr, end_lr)`` for the post-warmup phase."""
    code = compile(formula.strip(), "<lr_formula>", "eval")
    remaining = max(1, total_steps - warmup_steps)
    start = max(0.0, _eval_formula(code, 0, remaining, base_lr, steps_per_epoch, total_epochs))
    mid = max(0.0, _eval_formula(code, remaining // 2, remaining, base_lr, steps_per_epoch, total_epochs))
    end = max(0.0, _eval_formula(code, max(0, remaining - 1), remaining, base_lr, steps_per_epoch, total_epochs))
    return (start, mid, end)


# ---------------------------------------------------------------------------
# Scheduler builder
# ---------------------------------------------------------------------------

def build_formula_scheduler(
    optimizer: torch.optim.Optimizer,
    formula: str,
    base_lr: float,
    total_steps: int,
    warmup_steps: int,
    steps_per_epoch: int = 100,
    total_epochs: int = 10,
    warmup_start_factor: float = 0.1,
) -> SequentialLR:
    """Build a warmup + custom-formula LR scheduler.

    The warmup phase uses the same ``LinearLR`` ramp as built-in schedulers.
    The post-warmup phase evaluates *formula* via ``LambdaLR``.

    Args:
        optimizer: The optimizer to schedule.
        formula: Python math expression returning the desired absolute LR.
        base_lr: The configured learning rate (used to convert to multiplier).
        total_steps: Total optimizer steps for the entire run.
        warmup_steps: Steps for the linear warmup phase.
        steps_per_epoch: Steps per epoch (for the ``epoch`` variable).
        total_epochs: Total training epochs (for the ``total_epochs`` variable).

    Returns:
        A ``SequentialLR`` with warmup then formula-driven scheduling.

    Raises:
        ValueError: If *formula* is empty or fails validation.
    """
    err = validate_formula(
        formula, base_lr, total_steps, warmup_steps,
        steps_per_epoch, total_epochs,
    )
    if err:
        raise ValueError(f"Invalid custom LR formula: {err}")
    code = compile(formula.strip(), "<lr_formula>", "eval")
    remaining = max(1, total_steps - warmup_steps)

    _MAX_CONSECUTIVE_ERRORS = 10
    _state = {"consecutive_errors": 0, "permanently_disabled": False}

    def lr_lambda(step: int) -> float:
        if _state["permanently_disabled"]:
            return 1.0
        try:
            result = _eval_formula(
                code, step, remaining, base_lr, steps_per_epoch, total_epochs,
            )
            if not math.isfinite(result) or result < 0:
                _state["consecutive_errors"] += 1
                if _state["consecutive_errors"] == 1:
                    logger.warning(
                        "[Side-Step] Custom formula returned %s at step %d "
                        "-- falling back to base_lr",
                        result, step,
                    )
                if _state["consecutive_errors"] >= _MAX_CONSECUTIVE_ERRORS:
                    _state["permanently_disabled"] = True
                    logger.error(
                        "[Side-Step] Custom formula failed %d consecutive times. "
                        "Permanently reverting to base_lr for the rest of training.",
                        _MAX_CONSECUTIVE_ERRORS,
                    )
                return 1.0
            _state["consecutive_errors"] = 0
            return max(0.0, result) / base_lr
        except Exception as exc:
            _state["consecutive_errors"] += 1
            if _state["consecutive_errors"] == 1:
                logger.warning(
                    "[Side-Step] Custom formula error at step %d: %s "
                    "-- falling back to base_lr",
                    step, exc,
                )
            if _state["consecutive_errors"] >= _MAX_CONSECUTIVE_ERRORS:
                _state["permanently_disabled"] = True
                logger.exception(
                    "[Side-Step] Custom formula failed %d consecutive times. "
                    "Permanently reverting to base_lr for the rest of training.",
                    _MAX_CONSECUTIVE_ERRORS,
                )
            return 1.0

    warmup_sched = LinearLR(
        optimizer, start_factor=warmup_start_factor, end_factor=1.0, total_iters=warmup_steps,
    )
    main_sched = LambdaLR(optimizer, lr_lambda)

    return SequentialLR(
        optimizer, [warmup_sched, main_sched], milestones=[warmup_steps],
    )


# ---------------------------------------------------------------------------
# Wizard helpers
# ---------------------------------------------------------------------------

def formula_help_text() -> str:
    """Return a dim-hint string explaining formula variables and math."""
    return (
        "Your formula controls the LR after warmup completes.\n"
        "  Warmup is automatic (same linear ramp as built-in schedulers).\n"
        "  Variables: step, total_steps, progress (0→1), epoch,\n"
        "             total_epochs, steps_per_epoch, base_lr\n"
        "  Math: cos, sin, exp, log, sqrt, pi, min, max, abs, clamp(x, lo, hi)"
    )
