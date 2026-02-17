"""
Shared utilities for wizard flow modules.

Contains the Namespace builder for training flows, which maps the wizard
``answers`` dict to the ``argparse.Namespace`` expected by dispatch.
"""

from __future__ import annotations

import argparse

from acestep.training_v2.ui.prompt_helpers import DEFAULT_NUM_WORKERS


_DEFAULT_PROJECTIONS = "q_proj k_proj v_proj o_proj"


def _is_turbo(a: dict) -> bool:
    """Quick name-based turbo check for default selection."""
    base = a.get("base_model", a.get("model_variant", "turbo"))
    label = base.lower() if isinstance(base, str) else ""
    if "base" in label or "sft" in label:
        return False
    return True


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
    attention_type = a.get("attention_type", "both")
    has_split = "self_target_modules_str" in a or "cross_target_modules_str" in a

    if attention_type == "both" and has_split:
        self_mods = a.get("self_target_modules_str", _DEFAULT_PROJECTIONS).split()
        cross_mods = a.get("cross_target_modules_str", _DEFAULT_PROJECTIONS).split()
        resolved = []
        for m in self_mods:
            resolved.append(m if "." in m else f"self_attn.{m}")
        for m in cross_mods:
            resolved.append(m if "." in m else f"cross_attn.{m}")
    else:
        resolved = a.get("target_modules_str", _DEFAULT_PROJECTIONS).split()

    if a.get("target_mlp", False):
        mlp_modules = ["gate_proj", "up_proj", "down_proj"]
        existing = set(resolved)
        for m in mlp_modules:
            if m not in existing:
                resolved.append(m)

    return resolved


def build_train_namespace(a: dict, mode: str = "fixed") -> argparse.Namespace:
    """Convert a wizard answers dict into an argparse.Namespace for dispatch.

    Args:
        a: Wizard answers dict populated by step functions.
        mode: Training subcommand (always ``'fixed'``; turbo vs base/sft
            is auto-detected from the model variant).

    Returns:
        A fully populated ``argparse.Namespace``.
    """
    target_modules = _resolve_wizard_projections(a)
    nw = a.get("num_workers", DEFAULT_NUM_WORKERS)
    return argparse.Namespace(
        subcommand="fixed",
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
        learning_rate=a.get("learning_rate", 1e-4),
        batch_size=a.get("batch_size", 1),
        gradient_accumulation=a.get("gradient_accumulation", 4),
        epochs=a.get("epochs", 100),
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
        target_mlp=a.get("target_mlp", False),
        bias=a.get("bias", "none"),
        # LoKR args
        lokr_linear_dim=a.get("lokr_linear_dim", 64),
        lokr_linear_alpha=a.get("lokr_linear_alpha", 128),
        lokr_factor=a.get("lokr_factor", -1),
        lokr_decompose_both=a.get("lokr_decompose_both", False),
        lokr_use_tucker=a.get("lokr_use_tucker", False),
        lokr_use_scalar=a.get("lokr_use_scalar", False),
        lokr_weight_decompose=a.get("lokr_weight_decompose", False),
        # Output / checkpoints
        output_dir=a["output_dir"],
        save_every=a.get("save_every", 10),
        resume_from=a.get("resume_from"),
        log_dir=a.get("log_dir"),
        log_every=a.get("log_every", 10),
        log_heavy_every=a.get("log_heavy_every", 50),
        sample_every_n_epochs=a.get("sample_every_n_epochs", 0),
        shift=a.get("shift", 3.0 if _is_turbo(a) else 1.0),
        num_inference_steps=a.get("num_inference_steps", 8 if _is_turbo(a) else 50),
        optimizer_type=a.get("optimizer_type", "adamw"),
        scheduler_type=a.get("scheduler_type", "cosine"),
        gradient_checkpointing=a.get("gradient_checkpointing", True),
        offload_encoder=a.get("offload_encoder", False),
        chunk_duration=a.get("chunk_duration"),
        preprocess=False,
        audio_dir=None,
        dataset_json=None,
        tensor_output=None,
        max_duration=0,
        normalize="none",
        cfg_ratio=a.get("cfg_ratio", 0.15),
        loss_weighting=a.get("loss_weighting", "none"),
        snr_gamma=a.get("snr_gamma", 5.0),
        estimate_batches=None,
        top_k=16,
        granularity="module",
        module_config=None,
        auto_estimate=False,
        estimate_output=None,
    )
