"""
Interactive wizard for ACE-Step Training V2.

Launched when ``python train.py`` is run with no subcommand.  Walks the
user through all settings using Rich prompts, then returns a populated
``argparse.Namespace`` identical to what the CLI args would produce.

Uses only ``rich.prompt`` -- no extra dependencies.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Optional

from acestep.training_v2.ui import console, is_rich_active


# ---- Menu helper ------------------------------------------------------------

def _menu(
    title: str,
    options: list[tuple[str, str]],
    default: int = 1,
) -> str:
    """Display a numbered menu and return the chosen key.

    Args:
        title: Prompt text.
        options: List of ``(key, label)`` tuples.
        default: 1-based default index.

    Returns:
        The ``key`` of the chosen option.
    """
    if is_rich_active() and console is not None:
        console.print()
        console.print(f"  [bold]{title}[/]\n")
        for i, (key, label) in enumerate(options, 1):
            marker = "[bold cyan]>[/]" if i == default else " "
            tag = "  [dim](default)[/]" if i == default else ""
            console.print(f"    {marker} [bold]{i}[/]. {label}{tag}")
        console.print()

        from rich.prompt import IntPrompt
        while True:
            choice = IntPrompt.ask(
                "  Choice",
                default=default,
                console=console,
            )
            if 1 <= choice <= len(options):
                return options[choice - 1][0]
            console.print(f"  [red]Please enter a number between 1 and {len(options)}[/]")
    else:
        print(f"\n  {title}\n")
        for i, (key, label) in enumerate(options, 1):
            tag = " (default)" if i == default else ""
            print(f"    {i}. {label}{tag}")
        print()
        while True:
            try:
                raw = input(f"  Choice [{default}]: ").strip()
                choice = int(raw) if raw else default
                if 1 <= choice <= len(options):
                    return options[choice - 1][0]
                print(f"  Please enter a number between 1 and {len(options)}")
            except ValueError:
                print(f"  Please enter a number between 1 and {len(options)}")


def _ask(
    label: str,
    default: Any = None,
    required: bool = False,
    type_fn: type = str,
    choices: Optional[list] = None,
) -> Any:
    """Ask for a single value with an optional default.

    Args:
        label: Prompt text.
        default: Default value (None = required).
        required: If True, empty input is rejected.
        type_fn: Cast function (str, int, float).
        choices: Optional list of valid string values.

    Returns:
        The user's input, cast to ``type_fn``.
    """
    if choices:
        choice_str = f" ({'/'.join(str(c) for c in choices)})"
    else:
        choice_str = ""

    if is_rich_active() and console is not None:
        from rich.prompt import Prompt, IntPrompt, FloatPrompt

        prompt_cls = Prompt
        if type_fn is int:
            prompt_cls = IntPrompt
        elif type_fn is float:
            prompt_cls = FloatPrompt

        while True:
            result = prompt_cls.ask(
                f"  {label}{choice_str}",
                default=default if default is not None else ...,
                console=console,
            )
            if result is ...:
                if required:
                    console.print("  [red]This field is required[/]")
                    continue
                return None  # optional field, user pressed Enter to skip
            if required and not str(result).strip():
                console.print("  [red]This field is required[/]")
                continue
            if choices and str(result) not in [str(c) for c in choices]:
                console.print(f"  [red]Must be one of: {', '.join(str(c) for c in choices)}[/]")
                continue
            return type_fn(result) if not isinstance(result, type_fn) else result
    else:
        default_str = f" [{default}]" if default is not None else ""
        while True:
            raw = input(f"  {label}{choice_str}{default_str}: ").strip()
            if not raw and default is not None:
                return default
            if not raw and required:
                print("  This field is required")
                continue
            try:
                val = type_fn(raw)
                if choices and str(val) not in [str(c) for c in choices]:
                    print(f"  Must be one of: {', '.join(str(c) for c in choices)}")
                    continue
                return val
            except (ValueError, TypeError):
                print(f"  Invalid input, expected {type_fn.__name__}")


def _ask_path(
    label: str,
    default: Optional[str] = None,
    must_exist: bool = False,
) -> str:
    """Ask for a filesystem path, optionally validating existence."""
    while True:
        val = _ask(label, default=default, required=True)
        if must_exist and not Path(val).exists():
            if is_rich_active() and console is not None:
                console.print(f"  [red]Path not found: {val}[/]")
            else:
                print(f"  Path not found: {val}")
            continue
        return val


def _section(title: str) -> None:
    """Print a section header."""
    if is_rich_active() and console is not None:
        console.print(f"\n  [bold cyan]--- {title} ---[/]\n")
    else:
        print(f"\n  --- {title} ---\n")


# ---- Wizard flows -----------------------------------------------------------

def _ask_bool(label: str, default: bool = True) -> bool:
    """Ask for a yes/no boolean value."""
    choices = ["yes", "no"]
    default_str = "yes" if default else "no"
    result = _ask(label, default=default_str, choices=choices)
    return result.lower() in ("yes", "y", "true", "1")


def _wizard_train(mode: str) -> argparse.Namespace:
    """Interactive wizard for training (fixed or vanilla)."""

    # Ask for configuration mode first
    config_mode = _menu(
        "How much do you want to configure?",
        [
            ("basic", "Basic (recommended defaults, fewer questions)"),
            ("advanced", "Advanced (all settings exposed)"),
        ],
        default=1,
    )

    _section("Required Settings")
    checkpoint_dir = _ask_path("Checkpoint directory", default="./checkpoints", must_exist=True)
    model_variant = _ask("Model variant", default="turbo", choices=["turbo", "base", "sft"])
    dataset_dir = _ask_path("Dataset directory (preprocessed .pt files)", must_exist=True)
    output_dir = _ask("Output directory for LoRA weights", required=True)

    _section("LoRA Settings (press Enter for defaults)")
    rank = _ask("Rank", default=64, type_fn=int)
    alpha = _ask("Alpha", default=128, type_fn=int)
    dropout = _ask("Dropout", default=0.1, type_fn=float)

    # Attention type selection
    attention_type = _menu(
        "Which attention layers to target?",
        [
            ("both", "Both self-attention and cross-attention"),
            ("self", "Self-attention only (audio patterns)"),
            ("cross", "Cross-attention only (text conditioning)"),
        ],
        default=1,
    )

    target_modules_str = _ask("Target projections", default="q_proj k_proj v_proj o_proj")
    target_modules = target_modules_str.split()

    _section("Training Settings (press Enter for defaults)")
    learning_rate = _ask("Learning rate", default=1e-4, type_fn=float)
    batch_size = _ask("Batch size", default=1, type_fn=int)
    gradient_accumulation = _ask("Gradient accumulation", default=4, type_fn=int)
    epochs = _ask("Max epochs", default=100, type_fn=int)
    warmup_steps = _ask("Warmup steps", default=100, type_fn=int)
    seed = _ask("Seed", default=42, type_fn=int)

    cfg_ratio = 0.15
    if mode == "fixed":
        _section("Corrected Training Settings")
        cfg_ratio = _ask("CFG dropout ratio", default=0.15, type_fn=float)

    _section("Logging & Checkpoints (press Enter for defaults)")
    save_every = _ask("Save checkpoint every N epochs", default=10, type_fn=int)
    log_every = _ask("Log metrics every N steps", default=10, type_fn=int)
    resume_from = _ask("Resume from checkpoint path (leave empty to skip)", default=None)
    if resume_from == "None" or resume_from == "":
        resume_from = None

    # ---- Advanced mode settings (defaults used in basic mode) ----
    device = "auto"
    precision = "auto"
    weight_decay = 0.01
    max_grad_norm = 1.0
    bias = "none"
    # Windows uses spawn-based multiprocessing which breaks DataLoader workers
    num_workers = 0 if sys.platform == "win32" else 4
    pin_memory = True
    prefetch_factor = 2
    persistent_workers = num_workers > 0
    log_dir = None
    log_heavy_every = 50
    sample_every_n_epochs = 0
    optimizer_type = "adamw"
    scheduler_type = "cosine"
    gradient_checkpointing = False
    offload_encoder = False

    if config_mode == "advanced":
        _section("Device & Precision (Advanced)")
        device = _ask("Device", default="auto", choices=["auto", "cuda", "cuda:0", "cuda:1", "mps", "xpu", "cpu"])
        precision = _ask("Precision", default="auto", choices=["auto", "bf16", "fp16", "fp32"])

        _section("Optimizer & Scheduler")
        optimizer_type = _menu(
            "Which optimizer to use?",
            [
                ("adamw", "AdamW (default, reliable)"),
                ("adamw8bit", "AdamW 8-bit (saves ~30% optimizer VRAM, needs bitsandbytes)"),
                ("adafactor", "Adafactor (minimal state memory)"),
                ("prodigy", "Prodigy (auto-tunes LR -- set LR to 1.0, needs prodigyopt)"),
            ],
            default=1,
        )
        if optimizer_type == "prodigy":
            learning_rate = _ask("Learning rate (Prodigy: use 1.0)", default=1.0, type_fn=float)

        scheduler_type = _menu(
            "LR scheduler?",
            [
                ("cosine", "Cosine Annealing (gradual decay, most popular)"),
                ("linear", "Linear (steady decay to near-zero)"),
                ("constant", "Constant (flat LR after warmup)"),
                ("constant_with_warmup", "Constant with Warmup (explicit warmup then flat)"),
            ],
            default=1,
        )

        _section("VRAM Savings (Advanced)")
        gradient_checkpointing = _ask_bool(
            "Enable gradient checkpointing? (saves ~40-60% activation VRAM, ~30% slower)",
            default=False,
        )
        offload_encoder = _ask_bool(
            "Offload encoder/VAE to CPU? (saves ~2-4GB VRAM after setup)",
            default=False,
        )

        _section("Advanced Training Settings")
        weight_decay = _ask("Weight decay (L2 regularization)", default=0.01, type_fn=float)
        max_grad_norm = _ask("Max gradient norm (clipping)", default=1.0, type_fn=float)
        bias = _ask("Bias training mode", default="none", choices=["none", "all", "lora_only"])

        _section("Data Loading (Advanced)")
        num_workers = _ask("DataLoader workers", default=4, type_fn=int)
        pin_memory = _ask_bool("Pin memory for GPU transfer?", default=True)
        prefetch_factor = _ask("Prefetch factor", default=2, type_fn=int)
        persistent_workers = _ask_bool("Keep workers alive between epochs?", default=True)

        _section("Advanced Logging")
        log_dir = _ask("TensorBoard log directory (leave empty for default)", default=None)
        if log_dir == "None" or log_dir == "":
            log_dir = None
        log_heavy_every = _ask("Log gradient norms every N steps", default=50, type_fn=int)
        sample_every_n_epochs = _ask("Generate sample every N epochs (0=disabled)", default=0, type_fn=int)

    # Build namespace matching what argparse would produce
    ns = argparse.Namespace(
        subcommand=mode,
        plain=False,
        yes=True,  # Skip confirmation -- wizard already confirmed implicitly
        _from_wizard=True,  # Tells subcommand runners to skip banner/warning
        # Model
        checkpoint_dir=checkpoint_dir,
        model_variant=model_variant,
        # Device
        device=device,
        precision=precision,
        # Data
        dataset_dir=dataset_dir,
        num_workers=num_workers,
        pin_memory=pin_memory,
        prefetch_factor=prefetch_factor,
        persistent_workers=persistent_workers,
        # Training
        learning_rate=learning_rate,
        batch_size=batch_size,
        gradient_accumulation=gradient_accumulation,
        epochs=epochs,
        warmup_steps=warmup_steps,
        weight_decay=weight_decay,
        max_grad_norm=max_grad_norm,
        seed=seed,
        # LoRA
        rank=rank,
        alpha=alpha,
        dropout=dropout,
        target_modules=target_modules,
        attention_type=attention_type,
        bias=bias,
        # Checkpointing
        output_dir=output_dir,
        save_every=save_every,
        resume_from=resume_from,
        # Logging
        log_dir=log_dir,
        log_every=log_every,
        log_heavy_every=log_heavy_every,
        sample_every_n_epochs=sample_every_n_epochs,
        # Optimizer / scheduler / VRAM
        optimizer_type=optimizer_type,
        scheduler_type=scheduler_type,
        gradient_checkpointing=gradient_checkpointing,
        offload_encoder=offload_encoder,
        # Preprocessing (not used in training)
        preprocess=False,
        audio_dir=None,
        dataset_json=None,
        tensor_output=None,
        max_duration=240.0,
        # Fixed-specific
        cfg_ratio=cfg_ratio,
        # Estimation/selective (not used)
        estimate_batches=None,
        top_k=16,
        granularity="module",
        module_config=None,
        auto_estimate=False,
        estimate_output=None,
    )
    return ns


def _wizard_preprocess() -> argparse.Namespace:
    """Interactive wizard for preprocessing."""
    _section("Preprocessing Settings")
    checkpoint_dir = _ask_path("Checkpoint directory", default="./checkpoints", must_exist=True)
    model_variant = _ask("Model variant", default="turbo", choices=["turbo", "base", "sft"])
    audio_dir = _ask_path("Audio directory (source audio files)", must_exist=True)
    dataset_json = _ask("Dataset JSON file (optional, leave empty to skip)", default=None)
    if dataset_json == "None" or dataset_json == "":
        dataset_json = None
    elif dataset_json and not Path(dataset_json).exists():
        if is_rich_active() and console is not None:
            console.print(f"  [yellow]Warning: {dataset_json} not found[/]")
        else:
            print(f"  Warning: {dataset_json} not found")
    tensor_output = _ask("Output directory for .pt tensor files", required=True)
    max_duration = _ask("Max audio duration in seconds", default=240.0, type_fn=float)

    ns = argparse.Namespace(
        subcommand="fixed",
        plain=False,
        yes=True,
        _from_wizard=True,
        checkpoint_dir=checkpoint_dir,
        model_variant=model_variant,
        device="auto",
        precision="auto",
        dataset_dir=tensor_output,  # Will be populated after preprocessing
        num_workers=4,
        pin_memory=True,
        prefetch_factor=2,
        persistent_workers=True,
        learning_rate=1e-4,
        batch_size=1,
        gradient_accumulation=4,
        epochs=100,
        warmup_steps=100,
        weight_decay=0.01,
        max_grad_norm=1.0,
        seed=42,
        rank=64,
        alpha=128,
        dropout=0.1,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        attention_type="both",
        bias="none",
        output_dir="./lora_output",
        save_every=10,
        resume_from=None,
        log_dir=None,
        log_every=10,
        log_heavy_every=50,
        sample_every_n_epochs=0,
        optimizer_type="adamw",
        scheduler_type="cosine",
        gradient_checkpointing=False,
        offload_encoder=False,
        preprocess=True,
        audio_dir=audio_dir,
        dataset_json=dataset_json,
        tensor_output=tensor_output,
        max_duration=max_duration,
        cfg_ratio=0.15,
        estimate_batches=None,
        top_k=16,
        granularity="module",
        module_config=None,
        auto_estimate=False,
        estimate_output=None,
    )
    return ns


def _wizard_estimate() -> argparse.Namespace:
    """Interactive wizard for gradient sensitivity estimation."""
    _section("Gradient Sensitivity Estimation")

    if is_rich_active() and console is not None:
        console.print(
            "  [dim]Estimates which LoRA layers learn fastest for your dataset.\n"
            "  Results are saved as JSON and can be used to guide rank selection.[/]\n"
        )

    checkpoint_dir = _ask_path("Checkpoint directory", default="./checkpoints", must_exist=True)
    model_variant = _ask("Model variant", default="turbo", choices=["turbo", "base", "sft"])
    dataset_dir = _ask_path("Dataset directory (preprocessed .pt files)", must_exist=True)

    _section("Estimation Parameters")
    estimate_batches = _ask("Number of estimation batches", default=5, type_fn=int)
    top_k = _ask("Top-K layers to highlight", default=16, type_fn=int)
    granularity = _ask("Granularity", default="module", choices=["module", "layer"])

    _section("LoRA Settings (for estimation)")
    rank = _ask("Rank", default=64, type_fn=int)
    alpha = _ask("Alpha", default=128, type_fn=int)
    dropout = _ask("Dropout", default=0.1, type_fn=float)

    estimate_output = _ask("Output JSON path", default="./estimate_results.json")

    ns = argparse.Namespace(
        subcommand="estimate",
        plain=False,
        yes=True,
        _from_wizard=True,
        checkpoint_dir=checkpoint_dir,
        model_variant=model_variant,
        device="auto",
        precision="auto",
        dataset_dir=dataset_dir,
        num_workers=4,
        pin_memory=True,
        prefetch_factor=2,
        persistent_workers=True,
        learning_rate=1e-4,
        batch_size=1,
        gradient_accumulation=4,
        epochs=1,
        warmup_steps=0,
        weight_decay=0.01,
        max_grad_norm=1.0,
        seed=42,
        rank=rank,
        alpha=alpha,
        dropout=dropout,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        attention_type="both",
        bias="none",
        output_dir="./estimate_output",
        save_every=999,
        resume_from=None,
        log_dir=None,
        log_every=10,
        log_heavy_every=50,
        sample_every_n_epochs=0,
        optimizer_type="adamw",
        scheduler_type="cosine",
        gradient_checkpointing=False,
        offload_encoder=False,
        preprocess=False,
        audio_dir=None,
        dataset_json=None,
        tensor_output=None,
        max_duration=240.0,
        cfg_ratio=0.15,
        estimate_batches=estimate_batches,
        top_k=top_k,
        granularity=granularity,
        module_config=None,
        auto_estimate=False,
        estimate_output=estimate_output,
    )
    return ns


# ---- Main wizard entry point ------------------------------------------------

def run_wizard() -> Optional[argparse.Namespace]:
    """Launch the interactive wizard.

    Returns:
        A populated ``argparse.Namespace`` ready for the normal dispatch
        logic, or ``None`` if the user chose to exit.
    """
    # Show banner
    from acestep.training_v2.ui.banner import show_banner
    show_banner(subcommand="interactive")

    try:
        action = _menu(
            "What would you like to do?",
            [
                ("fixed", "Train a LoRA (corrected timesteps + CFG dropout)"),
                ("vanilla", "Train a LoRA (original behavior)"),
                ("preprocess", "Preprocess audio data"),
                ("estimate", "Run gradient estimation"),
                ("exit", "Exit"),
            ],
            default=1,
        )

        if action == "exit":
            return None

        if action == "estimate":
            return _wizard_estimate()

        if action == "preprocess":
            return _wizard_preprocess()

        # fixed or vanilla
        return _wizard_train(mode=action)

    except (KeyboardInterrupt, EOFError):
        if is_rich_active() and console is not None:
            console.print("\n  [dim]Aborted.[/]")
        else:
            print("\n  Aborted.")
        return None
