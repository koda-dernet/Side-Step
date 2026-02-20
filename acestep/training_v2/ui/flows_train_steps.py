"""
Individual wizard steps for the training flow.

Each step function takes an ``answers`` dict and writes user responses
into it.  All prompts support ``allow_back=True`` for go-back navigation.
"""

from __future__ import annotations

from acestep.training_v2.ui import console, is_rich_active
from acestep.training_v2.ui.prompt_helpers import (
    IS_WINDOWS,
    DEFAULT_NUM_WORKERS,
    _esc,
    menu,
    ask,
    ask_path,
    ask_bool,
    ask_output_path,
    native_path,
    section,
)


# ---- Helpers ----------------------------------------------------------------

def _has_fisher_map(a: dict) -> bool:
    """Return True if a valid Preprocessing++ map exists in dataset dir."""
    from pathlib import Path
    ds = a.get("dataset_dir")
    if not ds:
        return False
    p = Path(ds) / "fisher_map.json"
    if not p.is_file():
        return False
    try:
        import json
        data = json.loads(p.read_text(encoding="utf-8"))
        return bool(data.get("rank_pattern"))
    except Exception:
        return False


# ---- Basic steps ------------------------------------------------------------

def step_config_mode(a: dict) -> None:
    """Choose basic vs advanced configuration depth."""
    a["config_mode"] = menu(
        "How much do you want to configure?",
        [
            ("basic", "Basic (recommended defaults, fewer questions)"),
            ("advanced", "Advanced (all settings exposed)"),
        ],
        default=1,
        allow_back=True,
    )


def step_required(a: dict) -> None:
    """Collect required paths and model selection."""
    from acestep.training_v2.settings import get_checkpoint_dir
    from acestep.training_v2.model_discovery import (
        pick_model,
        prompt_base_model,
    )
    from acestep.training_v2.ui.flows_common import (
        describe_preprocessed_dataset_issue,
        show_dataset_issue,
        show_model_picker_fallback_hint,
    )

    section("Required Settings")

    # Checkpoint dir: prefer settings default over hardcoded fallback
    ckpt_default = a.get("checkpoint_dir") or get_checkpoint_dir() or native_path("./checkpoints")
    a["checkpoint_dir"] = ask_path(
        "Checkpoint directory", default=ckpt_default,
        must_exist=True, allow_back=True,
    )

    # Model selection via interactive picker (replaces hardcoded choices)
    result = pick_model(a["checkpoint_dir"])
    if result is None:
        # No models found -- fall back to manual entry
        show_model_picker_fallback_hint()
        a["model_variant"] = ask(
            "Model variant or folder name", default=a.get("model_variant", "turbo"),
            allow_back=True,
        )
        a["base_model"] = a["model_variant"]
    else:
        name, info = result
        a["model_variant"] = name
        a["base_model"] = info.base_model

        # If fine-tune with unknown base, ask the user
        if not info.is_official and info.base_model == "unknown":
            a["base_model"] = prompt_base_model(name)

    while True:
        a["dataset_dir"] = ask_path(
            "Dataset directory (preprocessed .pt files)",
            default=a.get("dataset_dir"),
            must_exist=True, allow_back=True,
        )
        issue = describe_preprocessed_dataset_issue(a["dataset_dir"])
        if issue is None:
            break
        show_dataset_issue(issue)
    a["output_dir"] = ask_output_path(
        "Output directory for adapter weights",
        default=a.get("output_dir"),
        required=True,
        allow_back=True,
    )


_DEFAULT_PROJECTIONS = "q_proj k_proj v_proj o_proj"


def _ask_attention_type(a: dict) -> None:
    """Prompt for attention layer targeting."""
    a["attention_type"] = menu(
        "Which attention layers to target?",
        [
            ("both", "Both self-attention and cross-attention"),
            ("self", "Self-attention only (audio patterns)"),
            ("cross", "Cross-attention only (text conditioning)"),
        ],
        default=1,
        allow_back=True,
    )


def _ask_projections(a: dict) -> None:
    """Prompt for target projections, splitting by attention type when 'both'.

    When the user selects "both", asks separately for self-attention and
    cross-attention projections so they can be configured independently.
    When "self" or "cross", asks once as a single set.
    """
    if a.get("attention_type") == "both":
        a["self_target_modules_str"] = ask(
            "Self-attention projections",
            default=a.get("self_target_modules_str", _DEFAULT_PROJECTIONS),
            allow_back=True,
        )
        a["cross_target_modules_str"] = ask(
            "Cross-attention projections",
            default=a.get("cross_target_modules_str", _DEFAULT_PROJECTIONS),
            allow_back=True,
        )
    else:
        a["target_modules_str"] = ask(
            "Target projections",
            default=a.get("target_modules_str", _DEFAULT_PROJECTIONS),
            allow_back=True,
        )


def step_lora(a: dict) -> None:
    """LoRA hyperparameters.

    When a Preprocessing++ map is detected in the dataset directory, rank / alpha /
    target-module questions are skipped because the map will
    override them in ``build_configs``.
    """
    if _has_fisher_map(a):
        section("LoRA Settings (Preprocessing++ guided -- rank & targets locked)")
        a["dropout"] = ask(
            "Dropout", default=a.get("dropout", 0.1),
            type_fn=float, allow_back=True,
        )
        return

    section("LoRA Settings (press Enter for defaults)")
    a["rank"] = ask("Rank", default=a.get("rank", 64), type_fn=int, allow_back=True)
    a["alpha"] = ask("Alpha", default=a.get("alpha", 128), type_fn=int, allow_back=True)
    a["dropout"] = ask("Dropout", default=a.get("dropout", 0.1), type_fn=float, allow_back=True)

    _ask_attention_type(a)
    _ask_projections(a)
    a["target_mlp"] = ask_bool(
        "Also target MLP/FFN layers (gate_proj, up_proj, down_proj)?",
        default=a.get("target_mlp", False),
        allow_back=True,
    )


def step_lokr(a: dict) -> None:
    """LoKR hyperparameters (LyCORIS Kronecker adapter)."""
    section("LoKR Settings (press Enter for defaults)")
    a["lokr_linear_dim"] = ask("Linear dimension", default=a.get("lokr_linear_dim", 64), type_fn=int, allow_back=True)
    a["lokr_linear_alpha"] = ask("Linear alpha", default=a.get("lokr_linear_alpha", 128), type_fn=int, allow_back=True)
    a["lokr_factor"] = ask("Factor (-1 = auto)", default=a.get("lokr_factor", -1), type_fn=int, allow_back=True)

    a["lokr_decompose_both"] = ask_bool(
        "Decompose both Kronecker factors?",
        default=a.get("lokr_decompose_both", False),
        allow_back=True,
    )
    a["lokr_use_tucker"] = ask_bool(
        "Use Tucker decomposition?",
        default=a.get("lokr_use_tucker", False),
        allow_back=True,
    )
    a["lokr_use_scalar"] = ask_bool(
        "Use scalar scaling?",
        default=a.get("lokr_use_scalar", False),
        allow_back=True,
    )
    a["lokr_weight_decompose"] = ask_bool(
        "Enable DoRA-style weight decomposition?",
        default=a.get("lokr_weight_decompose", False),
        allow_back=True,
    )

    _ask_attention_type(a)
    _ask_projections(a)
    a["target_mlp"] = ask_bool(
        "Also target MLP/FFN layers (gate_proj, up_proj, down_proj)?",
        default=a.get("target_mlp", False),
        allow_back=True,
    )


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


def step_training(a: dict) -> None:
    """Core training hyperparameters."""
    section("Training Settings (press Enter for defaults)")

    _pp_active = _has_fisher_map(a)
    _lr_default = 5e-5 if _pp_active else 1e-4
    if _pp_active:
        _pp_hint = (
            "  Preprocessing++ detected -- adaptive ranks overfit faster.\n"
            "  Recommended learning rate: ~5e-5 (lower than usual).\n"
        )
        if is_rich_active() and console is not None:
            console.print(f"  [yellow]{_esc(_pp_hint)}[/]")
        else:
            print(_pp_hint)

    a["learning_rate"] = ask("Learning rate", default=a.get("learning_rate", _lr_default), type_fn=float, allow_back=True)

    if _pp_active and a["learning_rate"] > 1e-4:
        _lr_warn = (
            f"  Learning rate {a['learning_rate']:.1e} is high for Preprocessing++.\n"
            "  This may cause overfitting or garbled output. Consider <= 1e-4.\n"
        )
        if is_rich_active() and console is not None:
            console.print(f"  [yellow]{_esc(_lr_warn)}[/]")
        else:
            print(_lr_warn)
    a["batch_size"] = ask("Batch size", default=a.get("batch_size", 1), type_fn=int, allow_back=True)
    a["gradient_accumulation"] = ask("Gradient accumulation", default=a.get("gradient_accumulation", 4), type_fn=int, allow_back=True)
    a["epochs"] = ask("Max epochs", default=a.get("epochs", 100), type_fn=int, allow_back=True)
    a["warmup_steps"] = ask("Warmup steps", default=a.get("warmup_steps", 100), type_fn=int, allow_back=True)
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


def step_logging(a: dict) -> None:
    """Logging and checkpoint settings."""
    import os

    section("Logging & Checkpoints (press Enter for defaults)")
    a["save_every"] = ask("Save checkpoint every N epochs", default=a.get("save_every", 10), type_fn=int, allow_back=True)
    a["save_best"] = ask_bool("Auto-save best model (smoothed loss)", default=a.get("save_best", True), allow_back=True)
    if a["save_best"]:
        a["save_best_after"] = ask("Start best-model tracking after epoch", default=a.get("save_best_after", 200), type_fn=int, allow_back=True)
        a["early_stop_patience"] = ask("Early stop patience (0=disabled)", default=a.get("early_stop_patience", 0), type_fn=int, allow_back=True)
    else:
        a["save_best_after"] = a.get("save_best_after", 200)
        a["early_stop_patience"] = 0
    a["log_every"] = ask("Log metrics every N steps", default=a.get("log_every", 10), type_fn=int, allow_back=True)
    a["log_heavy_every"] = ask(
        "Log gradient norms every N steps (0=disabled)",
        default=a.get("log_heavy_every", 50),
        type_fn=int,
        allow_back=True,
    )
    if a["log_heavy_every"] < 0:
        a["log_heavy_every"] = 0
    resume_raw = ask("Resume from checkpoint path (leave empty to skip)", default=a.get("resume_from"), allow_back=True)
    if resume_raw in (None, "None", ""):
        a["resume_from"] = None
    else:
        # Normalize: if user pointed to a file (e.g. adapter_config.json),
        # use the containing directory instead.
        if os.path.isfile(resume_raw):
            parent = os.path.dirname(resume_raw)
            if is_rich_active() and console is not None:
                console.print(
                    f"  [yellow]That's a file -- using checkpoint directory: {_esc(parent)}[/]"
                )
            else:
                print(f"  That's a file -- using checkpoint directory: {parent}")
            if ask_bool("Use this directory for resume?", default=True, allow_back=True):
                resume_raw = parent
            else:
                resume_raw = ask_path(
                    "Resume checkpoint directory",
                    default=parent,
                    must_exist=True,
                    allow_back=True,
                )
        else:
            # Directory path: validate it exists
            resume_raw = ask_path(
                "Resume checkpoint directory",
                default=resume_raw,
                must_exist=True,
                allow_back=True,
            )
        a["resume_from"] = resume_raw
        a["strict_resume"] = ask_bool(
            "Strict resume? (abort on state mismatch)",
            default=a.get("strict_resume", True),
            allow_back=True,
        )


def step_run_name(a: dict) -> None:
    """Prompt for an optional user-chosen run name."""
    section("Run Name (optional)")
    raw = ask(
        "Run name (used for output dir / TB logs, leave empty for auto)",
        default=a.get("run_name"),
        allow_back=True,
    )
    if raw in (None, "None", ""):
        a["run_name"] = None
    else:
        a["run_name"] = raw


# ---- Advanced steps ---------------------------------------------------------

def step_advanced_device(a: dict) -> None:
    """Advanced: device and precision."""
    section("Device & Precision (Advanced, press Enter for defaults)")
    a["device"] = ask("Device", default=a.get("device", "auto"), choices=["auto", "cuda", "cuda:0", "cuda:1", "mps", "xpu", "cpu"], allow_back=True)
    a["precision"] = ask("Precision", default=a.get("precision", "auto"), choices=["auto", "bf16", "fp16", "fp32"], allow_back=True)


def step_advanced_optimizer(a: dict) -> None:
    """Advanced: optimizer and scheduler."""
    section("Optimizer & Scheduler (press Enter for defaults)")
    a["optimizer_type"] = menu(
        "Which optimizer to use?",
        [
            ("adamw", "AdamW (default, reliable)"),
            ("adamw8bit", "AdamW 8-bit (saves ~30% optimizer VRAM, needs bitsandbytes)"),
            ("adafactor", "Adafactor (minimal state memory)"),
            ("prodigy", "Prodigy (auto-tunes LR -- start around 0.1, needs prodigyopt)"),
        ],
        default=1,
        allow_back=True,
    )
    if a["optimizer_type"] == "prodigy":
        a["learning_rate"] = ask(
            "Learning rate (Prodigy: start around 0.1, lower if unstable)",
            default=0.1,
            type_fn=float,
            allow_back=True,
        )

    a["scheduler_type"] = menu(
        "LR scheduler?",
        [
            ("cosine", "Cosine Annealing (smooth decay to near-zero, most popular)"),
            ("cosine_restarts", "Cosine with Restarts (cyclical decay, LR resets periodically)"),
            ("linear", "Linear (steady decay to near-zero)"),
            ("constant", "Constant (flat LR after warmup)"),
            ("constant_with_warmup", "Constant with Warmup (explicit warmup then flat)"),
        ],
        default=1,
        allow_back=True,
    )


def step_advanced_vram(a: dict) -> None:
    """Advanced: VRAM savings."""
    section("VRAM Savings (Advanced, press Enter for defaults)")
    a["gradient_checkpointing"] = ask_bool(
        "Enable gradient checkpointing? (saves ~40-60% activation VRAM, ~10-30% slower)",
        default=a.get("gradient_checkpointing", True),
        allow_back=True,
    )
    a["offload_encoder"] = ask_bool(
        "Offload encoder/VAE to CPU? (saves ~2-4GB VRAM after setup)",
        default=a.get("offload_encoder", False),
        allow_back=True,
    )


def step_advanced_training(a: dict) -> None:
    """Advanced: weight decay, grad norm, bias."""
    section("Advanced Training Settings (press Enter for defaults)")
    a["weight_decay"] = ask("Weight decay", default=a.get("weight_decay", 0.01), type_fn=float, allow_back=True)
    a["max_grad_norm"] = ask("Max gradient norm", default=a.get("max_grad_norm", 1.0), type_fn=float, allow_back=True)
    a["bias"] = ask("Bias training mode", default=a.get("bias", "none"), choices=["none", "all", "lora_only"], allow_back=True)


def step_advanced_dataloader(a: dict) -> None:
    """Advanced: DataLoader tuning."""
    section("Data Loading (Advanced, press Enter for defaults)")
    a["num_workers"] = ask("DataLoader workers", default=a.get("num_workers", DEFAULT_NUM_WORKERS), type_fn=int, allow_back=True)
    if IS_WINDOWS and a["num_workers"] > 0:
        if is_rich_active() and console is not None:
            console.print("  [yellow]Warning: Windows detected -- forcing num_workers=0[/]")
        else:
            print("  Warning: Windows detected -- forcing num_workers=0")
        a["num_workers"] = 0
    a["pin_memory"] = ask_bool("Pin memory for GPU transfer?", default=a.get("pin_memory", True), allow_back=True)
    a["prefetch_factor"] = ask("Prefetch factor", default=a.get("prefetch_factor", 2 if a["num_workers"] > 0 else 0), type_fn=int, allow_back=True)
    a["persistent_workers"] = ask_bool("Keep workers alive between epochs?", default=a.get("persistent_workers", a["num_workers"] > 0), allow_back=True)


def step_advanced_logging(a: dict) -> None:
    """Advanced: TensorBoard logging."""
    section("Advanced Logging (press Enter for defaults)")
    log_dir_raw = ask("TensorBoard log directory (leave empty for default)", default=a.get("log_dir"), allow_back=True)
    if log_dir_raw in (None, "None", "") or not str(log_dir_raw).strip():
        a["log_dir"] = None
    else:
        a["log_dir"] = ask_output_path(
            "TensorBoard log directory",
            default=str(log_dir_raw).strip(),
            required=True,
            allow_back=True,
        )


def step_chunk_duration(a: dict) -> None:
    """Latent chunking for data augmentation and VRAM savings."""
    section("Latent Chunking (optional)")

    if is_rich_active() and console is not None:
        console.print(
            "  [dim]Latent chunking slices preprocessed tensors into random\n"
            "  fixed-length windows each iteration, providing data augmentation\n"
            "  (the model sees different parts of each song every epoch) and\n"
            "  reducing VRAM usage for long songs.\n\n"
            "  [bold yellow]Warning:[/][dim] Chunks shorter than 60 seconds can hurt\n"
            "  training quality instead of enriching it. Use shorter chunks\n"
            "  only if you need to reduce VRAM and understand the trade-off.\n"
            "  Leave disabled (0) if your songs are already short or you have\n"
            "  enough VRAM.[/]"
        )
    else:
        print(
            "  Latent chunking slices preprocessed tensors into random\n"
            "  fixed-length windows each iteration, providing data augmentation\n"
            "  (the model sees different parts of each song every epoch) and\n"
            "  reducing VRAM usage for long songs.\n\n"
            "  WARNING: Chunks shorter than 60 seconds can hurt training quality\n"
            "  instead of enriching it. Use shorter chunks only if you need to\n"
            "  reduce VRAM and understand the trade-off.\n"
            "  Leave disabled (0) if your songs are already short or you have\n"
            "  enough VRAM."
        )

    while True:
        chunk = ask(
            "Chunk duration in seconds (0 = disabled, recommended: 60)",
            default=a.get("chunk_duration", 0),
            type_fn=int, allow_back=True,
        )

        if chunk <= 0:
            a["chunk_duration"] = None
            return

        if chunk < 60:
            if is_rich_active() and console is not None:
                console.print(
                    f"  [bold yellow]Caution:[/] {chunk}s chunks are below the recommended\n"
                    "  60s minimum. This may reduce training quality, especially for\n"
                    "  full-length inference. Consider using 60s or higher."
                )
            else:
                print(
                    f"  Caution: {chunk}s chunks are below the recommended 60s minimum.\n"
                    "  This may reduce training quality, especially for full-length\n"
                    "  inference. Consider using 60s or higher."
                )
            if not ask_bool("Use this chunk size anyway?", default=False, allow_back=True):
                continue

        a["chunk_duration"] = chunk
        return
