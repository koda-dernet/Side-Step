"""
Core training wizard steps: hyperparameters, CFG, and estimation helpers.

Extracted from ``train_steps.py`` to meet the module LOC policy.
"""

from __future__ import annotations

from sidestep_engine.ui.flows.train_steps_helpers import (
    estimate_total_steps as _estimate_total_steps,
    show_dataset_step_estimate as _show_dataset_step_estimate,
    smart_save_best_default as _smart_save_best_default,
    warn_warmup_ratio as _warn_warmup_ratio,
)
from sidestep_engine.training_defaults import DEFAULT_EPOCHS, DEFAULT_LEARNING_RATE
from sidestep_engine.ui.flows.train_steps_required import _has_fisher_map
from sidestep_engine.ui.prompt_helpers import (
    _esc,
    ask,
    print_message,
    section,
)


# ---- Helpers ----------------------------------------------------------------

def _default_shift(a: dict) -> float:
    """Return default shift value based on selected model variant."""
    base = a.get("base_model", a.get("model_variant", "turbo"))
    if isinstance(base, str) and "turbo" in base.lower():
        return 3.0
    return 1.0


def _default_inference_steps(a: dict) -> int:
    """Return default num_inference_steps based on selected model variant."""
    base = a.get("base_model", a.get("model_variant", "turbo"))
    if isinstance(base, str) and "turbo" in base.lower():
        return 8
    return 50


# ---- Steps ------------------------------------------------------------------

def step_training(a: dict) -> None:
    """Core training hyperparameters.

    In basic mode, shift and inference steps are auto-detected from the
    model variant and not prompted.  seed is also skipped (default 42).
    """
    section("Training Settings (press Enter for defaults)")
    _is_basic = a.get("config_mode") == "basic"

    _pp_active = _has_fisher_map(a)
    _lr_default = 5e-5 if _pp_active else DEFAULT_LEARNING_RATE
    if _pp_active:
        print_message(
            "Preprocessing++ detected -- adaptive ranks overfit faster.\n"
            "  Recommended learning rate: ~5e-5 (lower than usual).",
            kind="warn",
        )

    # LR clarity: explain what it controls
    _lr_label = (
        "Optimizer learning rate (global, applied to all trainable adapter weights)"
        if not _is_basic
        else "Learning rate"
    )
    if not _is_basic:
        print_message(
            "ACE-Step uses a single DiT (diffusion transformer), not UNet+TE.\n"
            "  This LR applies uniformly to all adapter parameters (self-attn + cross-attn + MLP/FFN).",
            kind="dim",
        )

    def _validate_lr(v: float) -> str | None:
        if v <= 0:
            return "Learning rate must be > 0 (you cannot train without a learning rate)."
        if v > 1.0:
            return "Learning rate > 1.0 is almost certainly too high. Typical range: 1e-5 to 1e-3."
        return None

    a["learning_rate"] = ask(
        _lr_label, default=a.get("learning_rate", _lr_default),
        type_fn=float, allow_back=True, validate_fn=_validate_lr,
    )

    if _pp_active and a["learning_rate"] > 1e-4:
        print_message(
            f"Learning rate {a['learning_rate']:.1e} is high for Preprocessing++.\n"
            "  This may cause overfitting or garbled output. Consider <= 1e-4.",
            kind="warn",
        )
    _pos_int = lambda v: "Must be >= 1." if v < 1 else None

    a["batch_size"] = ask("Batch size", default=a.get("batch_size", 1), type_fn=int, allow_back=True, validate_fn=_pos_int)
    a["gradient_accumulation"] = ask("Gradient accumulation", default=a.get("gradient_accumulation", 4), type_fn=int, allow_back=True, validate_fn=_pos_int)
    a["epochs"] = ask(
        "Max epochs (each epoch = one full pass through your dataset)",
        default=a.get("epochs", DEFAULT_EPOCHS), type_fn=int, allow_back=True, validate_fn=_pos_int,
    )

    # Training duration strategy
    if not _is_basic:
        print_message(
            "Max steps sets a hard stop regardless of epochs.  0 = disabled (epochs only).\n"
            "  Use max_steps when you want precise control over training length.\n"
            "  Use epochs + dataset_repeats to control how many times the model sees your data.",
            kind="dim",
        )

        a["max_steps"] = ask(
            "Max optimizer steps (0 = use epochs only)",
            default=a.get("max_steps", 0), type_fn=int, allow_back=True,
            validate_fn=lambda v: "Must be >= 0." if v < 0 else None,
        )
        a["dataset_repeats"] = ask(
            "Dataset repeats per epoch (multiply effective dataset size)",
            default=a.get("dataset_repeats", 1), type_fn=int, allow_back=True, validate_fn=_pos_int,
        )
    else:
        a.setdefault("max_steps", 0)
        a.setdefault("dataset_repeats", 1)

    a["warmup_steps"] = ask(
        "Warmup steps (LR ramps from 0 to target over this many steps)",
        default=a.get("warmup_steps", 100), type_fn=int, allow_back=True,
        validate_fn=lambda v: "Must be >= 0." if v < 0 else None,
    )

    # Show dataset size and step estimate
    _show_dataset_step_estimate(a)

    # Warn if warmup is a large fraction of estimated total steps
    _warn_warmup_ratio(a)

    # Basic mode: auto-detect seed, shift, inference steps from model
    if _is_basic:
        a.setdefault("seed", 42)
        a["shift"] = a.get("shift", _default_shift(a))
        a["num_inference_steps"] = a.get("num_inference_steps", _default_inference_steps(a))
        return

    a["seed"] = ask("Seed", default=a.get("seed", 42), type_fn=int, allow_back=True)

    # Shift & inference steps -- auto-default from model variant
    a["shift"] = ask(
        "Shift (turbo=3.0, base/sft=1.0)",
        default=a.get("shift", _default_shift(a)),
        type_fn=float, allow_back=True,
    )
    a["num_inference_steps"] = ask(
        "Inference steps (turbo=8, base/sft=50)",
        default=a.get("num_inference_steps", _default_inference_steps(a)),
        type_fn=int, allow_back=True,
    )


def step_cfg(a: dict) -> None:
    """CFG dropout and loss weighting (fixed mode only)."""
    section("Corrected Training Settings (press Enter for defaults)")
    a["cfg_ratio"] = ask("CFG dropout ratio", default=a.get("cfg_ratio", 0.15), type_fn=float, allow_back=True)
    a["loss_weighting"] = ask(
        "Loss weighting (none / min_snr -- min_snr can yield better results on SFT and base models)",
        default=a.get("loss_weighting", "none"), allow_back=True,
    )
    if a["loss_weighting"] == "min_snr":
        a["snr_gamma"] = ask("SNR gamma", default=a.get("snr_gamma", 5.0), type_fn=float, allow_back=True)
