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
    DEFAULT_EPOCHS,
    DEFAULT_LEARNING_RATE,
    DEFAULT_OPTIMIZER_TYPE,
    DEFAULT_SAVE_EVERY,
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
    loss_weighting = "none" if is_turbo_model else a.get("loss_weighting", "none")
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
        pin_memory=a.get("pin_memory", True),
        prefetch_factor=a.get("prefetch_factor", 2 if nw > 0 else 0),
        persistent_workers=a.get("persistent_workers", nw > 0),
        learning_rate=a.get("learning_rate", DEFAULT_LEARNING_RATE),
        batch_size=a.get("batch_size", 1),
        gradient_accumulation=a.get("gradient_accumulation", 4),
        epochs=a.get("epochs", DEFAULT_EPOCHS),
        warmup_steps=a.get("warmup_steps", 100),
        weight_decay=a.get("weight_decay", 0.01),
        max_grad_norm=a.get("max_grad_norm", 1.0),
        seed=a.get("seed", 42),
        # LoRA args
        rank=a.get("rank", 64),
        alpha=a.get("alpha", 128),
        dropout=a.get("dropout", 0.1),
        target_modules=target_modules,
        attention_type=a.get("attention_type", "both"),
        target_mlp=a.get("target_mlp", True),
        bias=a.get("bias", "none"),
        # LoKR args
        lokr_linear_dim=a.get("lokr_linear_dim", 64),
        lokr_linear_alpha=a.get("lokr_linear_alpha", 128),
        lokr_factor=a.get("lokr_factor", -1),
        lokr_decompose_both=a.get("lokr_decompose_both", False),
        lokr_use_tucker=a.get("lokr_use_tucker", False),
        lokr_use_scalar=a.get("lokr_use_scalar", False),
        lokr_weight_decompose=a.get("lokr_weight_decompose", False),
        # LoHA args
        loha_linear_dim=a.get("loha_linear_dim", 64),
        loha_linear_alpha=a.get("loha_linear_alpha", 128),
        loha_factor=a.get("loha_factor", -1),
        loha_use_tucker=a.get("loha_use_tucker", False),
        loha_use_scalar=a.get("loha_use_scalar", False),
        # OFT args
        oft_block_size=a.get("oft_block_size", 64),
        oft_coft=a.get("oft_coft", False),
        oft_eps=a.get("oft_eps", 6e-5),
        # DoRA flag (set by adapter_type dispatch, not a separate arg)
        use_dora=a.get("use_dora", False),
        # Output / checkpoints
        output_dir=a["output_dir"],
        save_every=a.get("save_every", DEFAULT_SAVE_EVERY),
        save_best=a.get("save_best", True),
        save_best_after=a.get("save_best_after", 200),
        early_stop_patience=a.get("early_stop_patience", 0),
        resume_from=a.get("resume_from"),
        strict_resume=a.get("strict_resume", True),
        run_name=a.get("run_name"),
        log_dir=a.get("log_dir"),
        log_every=a.get("log_every", 10),
        log_heavy_every=max(0, int(a.get("log_heavy_every", 50))),
        shift=a.get("shift", 3.0 if is_turbo_model else 1.0),
        num_inference_steps=a.get("num_inference_steps", 8 if is_turbo_model else 50),
        optimizer_type=a.get("optimizer_type", DEFAULT_OPTIMIZER_TYPE),
        scheduler_type=a.get("scheduler_type", "cosine"),
        scheduler_formula=a.get("scheduler_formula", ""),
        gradient_checkpointing=a.get("gradient_checkpointing", True),
        gradient_checkpointing_ratio=a.get("gradient_checkpointing_ratio", 1.0),
        offload_encoder=a.get("offload_encoder", True),
        chunk_duration=a.get("chunk_duration"),
        chunk_decay_every=a.get("chunk_decay_every", 10),
        dataset_repeats=a.get("dataset_repeats", 1),
        max_steps=a.get("max_steps", 0),
        preprocess=False,
        audio_dir=None,
        dataset_json=None,
        tensor_output=None,
        max_duration=0,
        normalize="none",
        cfg_ratio=a.get("cfg_ratio", 0.15),
        loss_weighting=loss_weighting,
        snr_gamma=a.get("snr_gamma", 5.0),
        # "All the Levers" enhancements
        ema_decay=a.get("ema_decay", 0.0),
        val_split=a.get("val_split", 0.0),
        adaptive_timestep_ratio=a.get("adaptive_timestep_ratio", 0.0),
        warmup_start_factor=a.get("warmup_start_factor", 0.1),
        cosine_eta_min_ratio=a.get("cosine_eta_min_ratio", 0.01),
        cosine_restarts_count=a.get("cosine_restarts_count", 4),
        save_best_every_n_steps=a.get("save_best_every_n_steps", 0),
        timestep_mu=a.get("timestep_mu"),
        timestep_sigma=a.get("timestep_sigma"),
    )
