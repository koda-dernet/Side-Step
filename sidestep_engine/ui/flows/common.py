"""
Shared utilities for wizard flow modules.

Contains the Namespace builder for training flows, which maps the wizard
``answers`` dict to the ``argparse.Namespace`` expected by dispatch.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from sidestep_engine.ui.prompt_helpers import DEFAULT_NUM_WORKERS, menu, print_message
from sidestep_engine.training_defaults import (
    DEFAULT_ALPHA,
    DEFAULT_ATTENTION_TYPE,
    DEFAULT_BATCH_SIZE,
    DEFAULT_BIAS,
    DEFAULT_CFG_RATIO,
    DEFAULT_CHUNK_DECAY_EVERY,
    DEFAULT_COSINE_ETA_MIN_RATIO,
    DEFAULT_COSINE_RESTARTS_COUNT,
    DEFAULT_DATASET_REPEATS,
    DEFAULT_DROPOUT,
    DEFAULT_EARLY_STOP_PATIENCE,
    DEFAULT_EMA_DECAY,
    DEFAULT_EMA_START_STEP,
    DEFAULT_TARGET_LOSS,
    DEFAULT_TARGET_LOSS_FLOOR,
    DEFAULT_TARGET_LOSS_WARMUP,
    DEFAULT_TARGET_LOSS_SMOOTHING,
    DEFAULT_EPOCHS,
    DEFAULT_GRADIENT_ACCUMULATION,
    DEFAULT_GRADIENT_CHECKPOINTING,
    DEFAULT_GRADIENT_CHECKPOINTING_RATIO,
    DEFAULT_LEARNING_RATE,
    DEFAULT_LOG_EVERY,
    DEFAULT_LOG_HEAVY_EVERY,
    DEFAULT_LOHA_FACTOR,
    DEFAULT_LOHA_LINEAR_ALPHA,
    DEFAULT_LOHA_LINEAR_DIM,
    DEFAULT_LOHA_USE_SCALAR,
    DEFAULT_LOHA_USE_TUCKER,
    DEFAULT_LOKR_DECOMPOSE_BOTH,
    DEFAULT_LOKR_FACTOR,
    DEFAULT_LOKR_LINEAR_ALPHA,
    DEFAULT_LOKR_LINEAR_DIM,
    DEFAULT_LOKR_USE_SCALAR,
    DEFAULT_LOKR_USE_TUCKER,
    DEFAULT_LOKR_WEIGHT_DECOMPOSE,
    DEFAULT_LOSS_WEIGHTING,
    DEFAULT_MAX_GRAD_NORM,
    DEFAULT_MAX_STEPS,
    DEFAULT_OFFLOAD_ENCODER,
    DEFAULT_OFT_BLOCK_SIZE,
    DEFAULT_OFT_COFT,
    DEFAULT_OFT_EPS,
    DEFAULT_OPTIMIZER_TYPE,
    DEFAULT_PERSISTENT_WORKERS,
    DEFAULT_PIN_MEMORY,
    DEFAULT_PREFETCH_FACTOR,
    DEFAULT_RANK,
    DEFAULT_SAVE_BEST,
    DEFAULT_SAVE_BEST_AFTER,
    DEFAULT_SAVE_BEST_EVERY_N_STEPS,
    DEFAULT_SAVE_EVERY,
    DEFAULT_SCHEDULER_FORMULA,
    DEFAULT_SCHEDULER_TYPE,
    DEFAULT_SEED,
    DEFAULT_SNR_GAMMA,
    DEFAULT_LOSS_FN,
    DEFAULT_HUBER_DELTA,
    DEFAULT_CHANNEL_BALANCE,
    DEFAULT_VAE_CHANNEL_PRIOR,
    DEFAULT_LATENT_NOISE,
    DEFAULT_T_BIAS,
    DEFAULT_LEGACY_LOSS,
    DEFAULT_STRICT_RESUME,
    DEFAULT_TARGET_MLP,
    DEFAULT_TIMESTEP_MODE,
    DEFAULT_ADAPTIVE_TIMESTEP_RATIO,
    DEFAULT_VAL_SPLIT,
    DEFAULT_WARMUP_START_FACTOR,
    DEFAULT_WARMUP_STEPS,
    DEFAULT_WEIGHT_DECAY,
)


from sidestep_engine.core.constants import (  # noqa: F401 -- re-export
    PP_COMPATIBLE_ADAPTERS,
    is_pp_compatible,
    is_turbo,
)

_DEFAULT_PROJECTIONS = "q_proj k_proj v_proj o_proj"


# Import the canonical set so wizard detection matches what preprocessing supports.
from sidestep_engine.data.preprocess_discovery import AUDIO_EXTENSIONS as _AUDIO_EXTENSIONS


def has_raw_audio_only(dataset_dir: str) -> bool:
    """Return True if *dataset_dir* has audio files but no .pt tensors."""
    from sidestep_engine.core.dataset_validator import validate_dataset
    status = validate_dataset(dataset_dir)
    return status.kind == "raw_audio"


def describe_preprocessed_dataset_issue(dataset_dir: str) -> str | None:
    """Return a user-facing issue string when dataset_dir is not train-ready.

    Delegates to the shared ``core.dataset_validator`` for consistency.
    Returns ``None`` when the dataset looks fine.
    """
    from sidestep_engine.core.dataset_validator import validate_dataset
    status = validate_dataset(dataset_dir)
    if status.kind in ("preprocessed", "mixed") and not status.issues:
        return None
    if status.issues:
        return status.issues[0]
    if status.kind == "raw_audio":
        return None  # raw audio is valid (wizard offers auto-preprocess)
    return f"Dataset directory appears empty or invalid: {dataset_dir}"


def show_dataset_issue(issue: str) -> None:
    """Print a standardized dataset issue message with recovery tips."""
    print_message(issue, kind="warn")
    if "invalid JSON" in issue or "manifest.json" in issue:
        print_message(
            "Tip: regenerate tensors/manifest, or fix Windows JSON paths using / or escaped \\\\.",
            kind="dim",
        )


def show_model_picker_fallback_hint() -> None:
    """Print a consistent hint when checkpoint model discovery is empty."""
    print_message(
        "No model directories found in that checkpoint path.",
        kind="warn",
    )
    print_message(
        "Enter the folder name manually (examples: turbo, base, sft, or your fine-tune folder name).",
        kind="dim",
    )


def offer_load_preset_subset(
    answers: dict,
    *,
    allowed_fields: set[str],
    prompt: str = "Load preset defaults for this flow?",
    preserve_fields: set[str] | None = None,
) -> None:
    """Optionally load a training preset and apply only overlapping fields."""
    from sidestep_engine.ui.presets import list_presets, load_preset

    presets = list_presets()
    if not presets:
        return

    options: list[tuple[str, str]] = [("keep", "Keep current defaults")]
    for p in presets:
        tag = " (built-in)" if p["builtin"] else ""
        desc = f" -- {p['description']}" if p["description"] else ""
        options.append((p["name"], f"{p['name']}{tag}{desc}"))

    choice = menu(prompt, options, default=1, allow_back=True)
    if choice == "keep":
        return

    data = load_preset(choice)
    if not data:
        print_message(f"Could not load preset '{choice}'.", kind="warn")
        return

    preserved = preserve_fields or set()
    applied = 0
    for key, value in data.items():
        if key in allowed_fields and key not in preserved:
            answers[key] = value
            applied += 1

    if applied:
        print_message(f"Loaded {applied} preset values from '{choice}'.", kind="ok")


def _resolve_wizard_projections(a: dict) -> list:
    """Build the ``target_modules`` list from wizard answers.

    When ``attention_type == "both"`` and the wizard collected separate
    self/cross projection strings, each set is prefixed with its attention
    path (``self_attn.`` / ``cross_attn.``) and the two are merged into
    one list.  Modules that already contain a ``.`` are passed through
    unchanged (assumed fully qualified).

    When a single ``target_modules_str`` is present (the "self" or "cross"
    path, or backward-compatible answers), it is split and returned as-is;
    the downstream ``resolve_target_modules`` call in ``config_builder``
    will add the appropriate prefix.

    When ``target_mlp`` is True, MLP module names (gate_proj, up_proj,
    down_proj) are appended (deduplicated).
    """
    # Resume flow can provide a pre-resolved list directly from
    # saved adapter config; preserve it as-is.
    direct = a.get("target_modules")
    if isinstance(direct, list) and direct:
        return list(direct)

    attention_type = a.get("attention_type", "both")
    has_split = "self_target_modules_str" in a or "cross_target_modules_str" in a

    if attention_type == "both" and has_split:
        self_mods = (a.get("self_target_modules_str") or _DEFAULT_PROJECTIONS).split()
        cross_mods = (a.get("cross_target_modules_str") or _DEFAULT_PROJECTIONS).split()
        resolved = []
        for m in self_mods:
            resolved.append(m if "." in m else f"self_attn.{m}")
        for m in cross_mods:
            resolved.append(m if "." in m else f"cross_attn.{m}")
    else:
        resolved = (a.get("target_modules_str") or _DEFAULT_PROJECTIONS).split()

    if a.get("target_mlp", False):
        mlp_modules = ["gate_proj", "up_proj", "down_proj"]
        existing = set(resolved)
        for m in mlp_modules:
            if m not in existing:
                resolved.append(m)

    return resolved


def build_train_namespace(a: dict, mode: str = "train") -> argparse.Namespace:
    """Convert a wizard answers dict into an argparse.Namespace for dispatch.

    Args:
        a: Wizard answers dict populated by step functions.
        mode: Training subcommand (always ``'train'``; turbo vs base/sft
            is auto-detected from the model variant).

    Returns:
        A fully populated ``argparse.Namespace``.
    """
    target_modules = _resolve_wizard_projections(a)
    nw = a.get("num_workers", DEFAULT_NUM_WORKERS)
    is_turbo_model = is_turbo(a)
    loss_weighting = DEFAULT_LOSS_WEIGHTING if is_turbo_model else a.get("loss_weighting", DEFAULT_LOSS_WEIGHTING)
    return argparse.Namespace(
        subcommand="train",
        plain=False,
        yes=False,
        _from_wizard=True,
        # Adapter selection
        adapter_type=a.get("adapter_type", "lora"),
        checkpoint_dir=a["checkpoint_dir"],
        model_variant=a["model_variant"],
        base_model=a.get("base_model", a["model_variant"]),
        device=a.get("device", "auto"),
        precision=a.get("precision", "auto"),
        dataset_dir=a["dataset_dir"],
        num_workers=nw,
        pin_memory=a.get("pin_memory", DEFAULT_PIN_MEMORY),
        prefetch_factor=a.get("prefetch_factor", DEFAULT_PREFETCH_FACTOR if nw > 0 else 0),
        persistent_workers=a.get("persistent_workers", nw > 0),
        learning_rate=a.get("learning_rate", DEFAULT_LEARNING_RATE),
        batch_size=a.get("batch_size", DEFAULT_BATCH_SIZE),
        gradient_accumulation=a.get("gradient_accumulation", DEFAULT_GRADIENT_ACCUMULATION),
        epochs=a.get("epochs", DEFAULT_EPOCHS),
        warmup_steps=a.get("warmup_steps", DEFAULT_WARMUP_STEPS),
        weight_decay=a.get("weight_decay", DEFAULT_WEIGHT_DECAY),
        max_grad_norm=a.get("max_grad_norm", DEFAULT_MAX_GRAD_NORM),
        seed=a.get("seed", DEFAULT_SEED),
        # LoRA args
        rank=a.get("rank", DEFAULT_RANK),
        alpha=a.get("alpha", DEFAULT_ALPHA),
        dropout=a.get("dropout", DEFAULT_DROPOUT),
        target_modules=target_modules,
        attention_type=a.get("attention_type", DEFAULT_ATTENTION_TYPE),
        target_mlp=a.get("target_mlp", DEFAULT_TARGET_MLP),
        bias=a.get("bias", DEFAULT_BIAS),
        # LoKR args
        lokr_linear_dim=a.get("lokr_linear_dim", DEFAULT_LOKR_LINEAR_DIM),
        lokr_linear_alpha=a.get("lokr_linear_alpha", DEFAULT_LOKR_LINEAR_ALPHA),
        lokr_factor=a.get("lokr_factor", DEFAULT_LOKR_FACTOR),
        lokr_decompose_both=a.get("lokr_decompose_both", DEFAULT_LOKR_DECOMPOSE_BOTH),
        lokr_use_tucker=a.get("lokr_use_tucker", DEFAULT_LOKR_USE_TUCKER),
        lokr_use_scalar=a.get("lokr_use_scalar", DEFAULT_LOKR_USE_SCALAR),
        lokr_weight_decompose=a.get("lokr_weight_decompose", DEFAULT_LOKR_WEIGHT_DECOMPOSE),
        # LoHA args
        loha_linear_dim=a.get("loha_linear_dim", DEFAULT_LOHA_LINEAR_DIM),
        loha_linear_alpha=a.get("loha_linear_alpha", DEFAULT_LOHA_LINEAR_ALPHA),
        loha_factor=a.get("loha_factor", DEFAULT_LOHA_FACTOR),
        loha_use_tucker=a.get("loha_use_tucker", DEFAULT_LOHA_USE_TUCKER),
        loha_use_scalar=a.get("loha_use_scalar", DEFAULT_LOHA_USE_SCALAR),
        # OFT args
        oft_block_size=a.get("oft_block_size", DEFAULT_OFT_BLOCK_SIZE),
        oft_coft=a.get("oft_coft", DEFAULT_OFT_COFT),
        oft_eps=a.get("oft_eps", DEFAULT_OFT_EPS),
        # DoRA flag (set by adapter_type dispatch, not a separate arg)
        use_dora=a.get("use_dora", False),
        # Output / checkpoints
        output_dir=a["output_dir"],
        save_every=a.get("save_every", DEFAULT_SAVE_EVERY),
        save_best=a.get("save_best", DEFAULT_SAVE_BEST),
        save_best_after=a.get("save_best_after", DEFAULT_SAVE_BEST_AFTER),
        early_stop_patience=a.get("early_stop_patience", DEFAULT_EARLY_STOP_PATIENCE),
        target_loss=a.get("target_loss", DEFAULT_TARGET_LOSS),
        target_loss_floor=a.get("target_loss_floor", DEFAULT_TARGET_LOSS_FLOOR),
        target_loss_warmup=a.get("target_loss_warmup", DEFAULT_TARGET_LOSS_WARMUP),
        target_loss_smoothing=a.get("target_loss_smoothing", DEFAULT_TARGET_LOSS_SMOOTHING),
        resume_from=a.get("resume_from"),
        strict_resume=a.get("strict_resume", DEFAULT_STRICT_RESUME),
        run_name=a.get("run_name"),
        log_dir=a.get("log_dir"),
        log_every=a.get("log_every", DEFAULT_LOG_EVERY),
        log_heavy_every=max(0, int(a.get("log_heavy_every", DEFAULT_LOG_HEAVY_EVERY))),
        shift=a.get("shift", 3.0 if is_turbo_model else 1.0),
        num_inference_steps=a.get("num_inference_steps", 8 if is_turbo_model else 50),
        optimizer_type=a.get("optimizer_type", DEFAULT_OPTIMIZER_TYPE),
        scheduler_type=a.get("scheduler_type", DEFAULT_SCHEDULER_TYPE),
        scheduler_formula=a.get("scheduler_formula", DEFAULT_SCHEDULER_FORMULA),
        gradient_checkpointing=a.get("gradient_checkpointing", DEFAULT_GRADIENT_CHECKPOINTING),
        gradient_checkpointing_ratio=a.get("gradient_checkpointing_ratio", DEFAULT_GRADIENT_CHECKPOINTING_RATIO),
        offload_encoder=a.get("offload_encoder", DEFAULT_OFFLOAD_ENCODER),
        chunk_duration=a.get("chunk_duration"),
        max_latent_length=a.get("max_latent_length"),
        crop_mode=a.get("crop_mode"),
        chunk_decay_every=a.get("chunk_decay_every", DEFAULT_CHUNK_DECAY_EVERY),
        dataset_repeats=a.get("dataset_repeats", DEFAULT_DATASET_REPEATS),
        max_steps=a.get("max_steps", DEFAULT_MAX_STEPS),
        preprocess=False,
        audio_dir=None,
        dataset_json=None,
        tensor_output=None,
        max_duration=0,
        normalize="none",
        cfg_ratio=a.get("cfg_ratio", DEFAULT_CFG_RATIO),
        loss_weighting=loss_weighting,
        snr_gamma=a.get("snr_gamma", DEFAULT_SNR_GAMMA),
        loss_fn=a.get("loss_fn", DEFAULT_LOSS_FN),
        huber_delta=a.get("huber_delta", DEFAULT_HUBER_DELTA),
        channel_balance=a.get("channel_balance", DEFAULT_CHANNEL_BALANCE),
        vae_channel_prior=a.get("vae_channel_prior", DEFAULT_VAE_CHANNEL_PRIOR),
        latent_noise=a.get("latent_noise", DEFAULT_LATENT_NOISE),
        t_bias=a.get("t_bias", DEFAULT_T_BIAS),
        legacy_loss=a.get("legacy_loss", DEFAULT_LEGACY_LOSS),
        timestep_mode=a.get("timestep_mode", DEFAULT_TIMESTEP_MODE),
        # "All the Levers" enhancements
        ema_decay=a.get("ema_decay", DEFAULT_EMA_DECAY),
        ema_start_step=a.get("ema_start_step", DEFAULT_EMA_START_STEP),
        val_split=a.get("val_split", DEFAULT_VAL_SPLIT),
        adaptive_timestep_ratio=a.get("adaptive_timestep_ratio", DEFAULT_ADAPTIVE_TIMESTEP_RATIO),
        warmup_start_factor=a.get("warmup_start_factor", DEFAULT_WARMUP_START_FACTOR),
        cosine_eta_min_ratio=a.get("cosine_eta_min_ratio", DEFAULT_COSINE_ETA_MIN_RATIO),
        cosine_restarts_count=a.get("cosine_restarts_count", DEFAULT_COSINE_RESTARTS_COUNT),
        save_best_every_n_steps=a.get("save_best_every_n_steps", DEFAULT_SAVE_BEST_EVERY_N_STEPS),
        timestep_mu=a.get("timestep_mu"),
        timestep_sigma=a.get("timestep_sigma"),
    )
