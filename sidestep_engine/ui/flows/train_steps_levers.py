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
        "Timestep mode controls how training timesteps are sampled:\n"
        "  continuous = logit-normal distribution (recommended for all variants)\n"
        "  discrete  = 8-step turbo inference schedule (legacy turbo behavior)",
        kind="dim",
    )
    a["timestep_mode"] = ask(
        "Timestep sampling mode",
        default=a.get("timestep_mode", "continuous"),
        choices=["continuous", "discrete"],
    )

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
    if a["ema_decay"] > 0:
        a["ema_start_step"] = ask(
            "EMA start step (tracking begins at this step, 0 = immediate)",
            default=a.get("ema_start_step", 2000), type_fn=int,
            validate_fn=lambda v: "Must be >= 0" if v < 0 else None,
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

    print_message(
        "\nFlow matching fidelity controls. These are new defaults that improve\n"
        "  training quality. Set legacy_loss=True to revert everything.",
        kind="dim",
    )
    a["t_bias"] = ask(
        "Timestep bias toward detail (0 = symmetric, 0.5 = recommended)",
        default=a.get("t_bias", 0.5), type_fn=float,
        validate_fn=lambda v: "Must be >= 0 and <= 2" if v < 0 or v > 2 else None,
    )
    a["channel_balance"] = ask(
        "Per-channel fidelity balancing (True/False)",
        default=a.get("channel_balance", True), type_fn=bool,
    )
    a["vae_channel_prior"] = ask(
        "Use VAE decoder importance in channel weights (True/False)",
        default=a.get("vae_channel_prior", True), type_fn=bool,
    )
    a["latent_noise"] = ask(
        "Per-channel latent noise scale (0 = off, 0.02 = recommended)",
        default=a.get("latent_noise", 0.02), type_fn=float,
        validate_fn=lambda v: "Must be >= 0" if v < 0 else None,
    )
    a["legacy_loss"] = ask(
        "Legacy loss mode (reverts all fidelity changes, True/False)",
        default=a.get("legacy_loss", False), type_fn=bool,
    )


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
    a["target_loss_floor"] = ask(
        "Target loss LR floor (0.01 = 1%)",
        default=a.get("target_loss_floor", 0.01), type_fn=float,
        validate_fn=lambda v: "Must be > 0 and <= 1" if v <= 0 or v > 1 else None,
    )
    _sched_warmup = a.get("warmup_steps", 0)
    _cruise_default = max(a.get("target_loss_warmup", 50), _sched_warmup) if _sched_warmup > 0 else a.get("target_loss_warmup", 50)
    _cruise_min = _sched_warmup if _sched_warmup > 0 else 0
    _cruise_hint = f" (min {_sched_warmup}, linked to scheduler warmup)" if _sched_warmup > 0 else ""
    a["target_loss_warmup"] = ask(
        f"Cruise control warmup steps{_cruise_hint}",
        default=_cruise_default, type_fn=int,
        validate_fn=lambda v, _m=_cruise_min: f"Must be >= {_m} (scheduler warmup)" if v < _m else ("Must be >= 0" if v < 0 else None),
    )
    a["target_loss_smoothing"] = ask(
        "Cruise control smoothing (EMA beta)",
        default=a.get("target_loss_smoothing", 0.98), type_fn=float,
        validate_fn=lambda v: "Must be > 0 and < 1" if v <= 0 or v >= 1 else None,
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
