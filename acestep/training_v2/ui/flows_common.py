"""
Shared utilities for wizard flow modules.

Contains the Namespace builder for training flows, which maps the wizard
``answers`` dict to the ``argparse.Namespace`` expected by dispatch.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from acestep.training_v2.ui.prompt_helpers import DEFAULT_NUM_WORKERS, menu, print_message


_DEFAULT_PROJECTIONS = "q_proj k_proj v_proj o_proj"


def describe_preprocessed_dataset_issue(dataset_dir: str) -> str | None:
    """Return a user-facing issue string when dataset_dir is not train-ready.

    This is a lightweight preflight check used by wizard flows to catch
    obvious misconfiguration before training/analysis starts.
    """
    d = Path(dataset_dir)
    if not d.is_dir():
        return f"Dataset directory not found: {d}"

    if any(d.glob("*.pt")):
        return None

    manifest_path = d / "manifest.json"
    if not manifest_path.is_file():
        return "No .pt files found and manifest.json is missing"

    try:
        raw = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        return (
            f"manifest.json is invalid JSON: {exc}. "
            "Escape backslashes (\\\\) or use forward slashes (/)."
        )
    except Exception as exc:
        return f"Failed to read manifest.json: {exc}"

    if not isinstance(raw, dict):
        return "manifest.json must be a JSON object with a 'samples' list"
    samples = raw.get("samples", [])
    if not isinstance(samples, list):
        return "manifest.json field 'samples' must be a list"
    if not samples:
        return "manifest.json has no samples and no .pt files were found in the directory"
    if not any(isinstance(s, str) and s.strip() for s in samples):
        return "manifest.json samples are empty or not string paths"
    return None


def show_dataset_issue(issue: str) -> None:
    """Print a standardized dataset issue message with recovery tips."""
    print_message(issue, style="yellow")
    if "invalid JSON" in issue or "manifest.json" in issue:
        print_message(
            "Tip: regenerate tensors/manifest, or fix Windows JSON paths using / or escaped \\\\.",
            style="dim",
        )


def show_model_picker_fallback_hint() -> None:
    """Print a consistent hint when checkpoint model discovery is empty."""
    print_message(
        "No model directories found in that checkpoint path.",
        style="yellow",
    )
    print_message(
        "Enter the folder name manually (examples: turbo, base, sft, or your fine-tune folder name).",
        style="dim",
    )


def offer_load_preset_subset(
    answers: dict,
    *,
    allowed_fields: set[str],
    prompt: str = "Load preset defaults for this flow?",
    preserve_fields: set[str] | None = None,
) -> None:
    """Optionally load a training preset and apply only overlapping fields."""
    from acestep.training_v2.ui.presets import list_presets, load_preset

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
        print_message(f"Could not load preset '{choice}'.", style="yellow")
        return

    preserved = preserve_fields or set()
    applied = 0
    for key, value in data.items():
        if key in allowed_fields and key not in preserved:
            answers[key] = value
            applied += 1

    if applied:
        print_message(f"Loaded {applied} preset values from '{choice}'.", style="green")


def _is_turbo(a: dict) -> bool:
    """Variant check with metadata fallback for unknown custom names."""
    base = a.get("base_model", a.get("model_variant", "turbo"))
    label = base.lower() if isinstance(base, str) else ""
    if "turbo" in label:
        return True
    if "base" in label or "sft" in label:
        return False
    infer_steps = a.get("num_inference_steps", 8)
    try:
        return int(infer_steps) == 8
    except (TypeError, ValueError):
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
    is_turbo = _is_turbo(a)
    loss_weighting = "none" if is_turbo else a.get("loss_weighting", "none")
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
        save_best=a.get("save_best", True),
        save_best_after=a.get("save_best_after", 200),
        early_stop_patience=a.get("early_stop_patience", 0),
        resume_from=a.get("resume_from"),
        strict_resume=a.get("strict_resume", True),
        run_name=a.get("run_name"),
        log_dir=a.get("log_dir"),
        log_every=a.get("log_every", 10),
        log_heavy_every=max(0, int(a.get("log_heavy_every", 50))),
        shift=a.get("shift", 3.0 if is_turbo else 1.0),
        num_inference_steps=a.get("num_inference_steps", 8 if is_turbo else 50),
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
        loss_weighting=loss_weighting,
        snr_gamma=a.get("snr_gamma", 5.0),
        estimate_batches=None,
        top_k=16,
        granularity="module",
        module_config=None,
        auto_estimate=False,
        estimate_output=None,
    )
