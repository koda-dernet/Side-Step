"""
'All the Levers' wizard step — comprehensive training knobs panel.

Surfaces every mathematical and training-dynamics parameter in one place,
including values that are normally hardcoded or model-derived.  Only shown
in advanced wizard mode.  All defaults match existing behavior exactly.
"""

from __future__ import annotations

from sidestep_engine.ui.flows.common import is_turbo
from sidestep_engine.ui.prompt_helpers import (
    ask,
    ask_bool,
    print_message,
    section,
)


# ---- Sub-section helpers ---------------------------------------------------

def _levers_flow_math(a: dict) -> None:
    """Flow matching math knobs."""
    section("Flow Matching Math")

    _is_turbo = is_turbo(a)

    print_message(
        "Timestep mu/sigma control the logit-normal sampling distribution.\n"
        "  These are read from the model config — change only if you know\n"
        "  what you're doing.",
        kind="dim",
    )
    _mu_default = a.get("timestep_mu", -0.4)
    a["timestep_mu"] = ask(
        "Timestep mu (from model config)",
        default=_mu_default, type_fn=float,
    )
    if a["timestep_mu"] != _mu_default:
        print_message(
            "Overriding model default — this changes the sampling distribution.",
            kind="warn",
        )

    _sigma_default = a.get("timestep_sigma", 1.0)
    a["timestep_sigma"] = ask(
        "Timestep sigma (from model config)",
        default=_sigma_default, type_fn=float,
        validate_fn=lambda v: "Must be > 0" if v <= 0 else None,
    )

    if not _is_turbo:
        a["cfg_ratio"] = ask(
            "CFG dropout ratio (0 = disabled)",
            default=a.get("cfg_ratio", 0.15), type_fn=float,
            validate_fn=lambda v: "Must be >= 0 and <= 1" if v < 0 or v > 1 else None,
        )
        a["loss_weighting"] = ask(
            "Loss weighting (none / min_snr)",
            default=a.get("loss_weighting", "none"),
            choices=["none", "min_snr"],
        )
        if a["loss_weighting"] == "min_snr":
            a["snr_gamma"] = ask(
                "SNR gamma clamp",
                default=a.get("snr_gamma", 5.0), type_fn=float,
                validate_fn=lambda v: "Must be > 0" if v <= 0 else None,
            )


def _levers_optimizer(a: dict) -> None:
    """Optimizer internal knobs."""
    section("Optimizer Internals")

    print_message(
        "These control warmup ramp and cosine decay floor.\n"
        "  Defaults match the current training loop exactly.",
        kind="dim",
    )

    a["warmup_start_factor"] = ask(
        "Warmup start factor (LR starts at base_lr × this)",
        default=a.get("warmup_start_factor", 0.1), type_fn=float,
        validate_fn=lambda v: "Must be > 0 and <= 1" if v <= 0 or v > 1 else None,
    )
    a["cosine_eta_min_ratio"] = ask(
        "Cosine eta_min ratio (LR decays to base_lr × this)",
        default=a.get("cosine_eta_min_ratio", 0.01), type_fn=float,
        validate_fn=lambda v: "Must be >= 0 and <= 1" if v < 0 or v > 1 else None,
    )
    a["cosine_restarts_count"] = ask(
        "Cosine restarts count (cosine_restarts scheduler only)",
        default=a.get("cosine_restarts_count", 4), type_fn=int,
        validate_fn=lambda v: "Must be >= 1" if v < 1 else None,
    )
    a["weight_decay"] = ask(
        "Weight decay",
        default=a.get("weight_decay", 0.01), type_fn=float,
    )
    a["max_grad_norm"] = ask(
        "Max gradient norm",
        default=a.get("max_grad_norm", 1.0), type_fn=float,
        validate_fn=lambda v: "Must be > 0" if v <= 0 else None,
    )


def _levers_experimental(a: dict) -> None:
    """New experimental features."""
    section("Experimental Enhancements")

    print_message(
        "EMA maintains a smoothed copy of adapter weights — often produces\n"
        "  higher-quality output. ~5 MB overhead at rank 8 (stored on CPU).",
        kind="dim",
    )
    a["ema_decay"] = ask(
        "EMA decay (0 = off, 0.9999 = typical, 0.999 = faster)",
        default=a.get("ema_decay", 0.0), type_fn=float,
        validate_fn=lambda v: "Must be >= 0 and < 1" if v < 0 or v >= 1 else None,
    )

    print_message(
        "\nValidation holds out data to detect overfitting.\n"
        "  Best-model selection uses val loss when enabled.",
        kind="dim",
    )
    a["val_split"] = ask(
        "Validation split (0 = off, 0.1 = 10% held out)",
        default=a.get("val_split", 0.0), type_fn=float,
        validate_fn=lambda v: "Must be >= 0 and <= 0.5" if v < 0 or v > 0.5 else None,
    )

    if not is_turbo(a):
        print_message(
            "\nAdaptive timestep focuses training on regions where the model\n"
            "  struggles most. Mixes with the base logit-normal distribution.",
            kind="dim",
        )
        a["adaptive_timestep_ratio"] = ask(
            "Adaptive timestep ratio (0 = off, 0.3 = recommended)",
            default=a.get("adaptive_timestep_ratio", 0.0), type_fn=float,
            validate_fn=lambda v: "Must be >= 0 and <= 1" if v < 0 or v > 1 else None,
        )
    else:
        a.setdefault("adaptive_timestep_ratio", 0.0)


def _levers_tracking(a: dict) -> None:
    """Training dynamics / tracking knobs."""
    section("Tracking")

    a["save_best_every_n_steps"] = ask(
        "Step-level best-model check interval (0 = epoch only)",
        default=a.get("save_best_every_n_steps", 0), type_fn=int,
        validate_fn=lambda v: "Must be >= 0" if v < 0 else None,
    )
    a["early_stop_patience"] = ask(
        "Early stop patience (0 = disabled)",
        default=a.get("early_stop_patience", 0), type_fn=int,
        validate_fn=lambda v: "Must be >= 0" if v < 0 else None,
    )


# ---- Main step function ---------------------------------------------------

def step_all_the_levers(a: dict) -> None:
    """All the Levers — comprehensive training knobs panel.

    Surfaces every mathematical and training-dynamics parameter in one
    place.  Only shown in advanced wizard mode.
    """
    section("All the Levers")
    print_message(
        "Every training math knob in one place. Defaults match the\n"
        "  current loop exactly — change only what you want to tweak.",
        kind="dim",
    )

    _levers_flow_math(a)
    _levers_optimizer(a)
    _levers_experimental(a)
    _levers_tracking(a)
