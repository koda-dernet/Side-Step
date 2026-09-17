"""
Wizard review summary and CLI command export.

Builds a grouped parameter table from wizard answers (before config
objects exist) and generates a reproducible CLI command string.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from sidestep_engine.training_defaults import TRAINING_DEFAULTS
from sidestep_engine.ui import console, is_rich_active
from sidestep_engine.ui.prompt_helpers import _esc, print_message, print_rich


# ---- Answer-key → display grouping ------------------------------------------

_LORA_KEYS: List[Tuple[str, str]] = [
    ("rank", "Rank"),
    ("alpha", "Alpha"),
    ("dropout", "Dropout"),
    ("attention_type", "Attention targeting"),
    ("target_mlp", "MLP/FFN layers"),
    ("bias", "Bias"),
]

_DORA_KEYS: List[Tuple[str, str]] = [
    ("rank", "Rank"),
    ("alpha", "Alpha"),
    ("dropout", "Dropout"),
    ("attention_type", "Attention targeting"),
    ("target_mlp", "MLP/FFN layers"),
    ("bias", "Bias"),
]

_LOKR_KEYS: List[Tuple[str, str]] = [
    ("lokr_linear_dim", "Linear dimension"),
    ("lokr_linear_alpha", "Linear alpha"),
    ("lokr_factor", "Factor"),
    ("attention_type", "Attention targeting"),
    ("target_mlp", "MLP/FFN layers"),
]

_LOHA_KEYS: List[Tuple[str, str]] = [
    ("loha_linear_dim", "Linear dimension"),
    ("loha_linear_alpha", "Linear alpha"),
    ("loha_factor", "Factor"),
    ("attention_type", "Attention targeting"),
    ("target_mlp", "MLP/FFN layers"),
]

_OFT_KEYS: List[Tuple[str, str]] = [
    ("oft_block_size", "Block size"),
    ("oft_coft", "Constrained OFT"),
    ("oft_eps", "Epsilon"),
    ("attention_type", "Attention targeting"),
    ("target_mlp", "MLP/FFN layers"),
]

# Defaults for non-default highlighting.
# Canonical source: sidestep_engine.training_defaults.TRAINING_DEFAULTS
_DEFAULTS: Dict[str, Any] = {
    **TRAINING_DEFAULTS,
    # Keys that the wizard uses but aren't training parameters
    "shift": 3.0,
    "num_inference_steps": 8,
    "chunk_duration": None,
    "max_latent_length": None,
    "crop_mode": None,
    "resume_from": None,
    "run_name": None,
    "log_dir": None,
}


_GROUPS_TEMPLATE: List[Tuple[str, List[Tuple[str, str]]]] = [
    ("Model", [
        ("model_variant", "Model variant"),
        ("adapter_type", "Adapter"),
        ("checkpoint_dir", "Checkpoint dir"),
        ("dataset_dir", "Dataset dir"),
    ]),
    ("Device", [
        ("device", "Device"),
        ("precision", "Precision"),
    ]),
    # Adapter group inserted dynamically
    ("Training", [
        ("learning_rate", "Learning rate"),
        ("optimizer_type", "Optimizer"),
        ("scheduler_type", "LR scheduler"),
        ("scheduler_formula", "Custom formula"),
        ("batch_size", "Batch size"),
        ("gradient_accumulation", "Grad accumulation"),
        ("_effective_batch", "Effective batch"),
        ("epochs", "Max epochs"),
        ("max_steps", "Max optimizer steps"),
        ("dataset_repeats", "Dataset repeats"),
        ("warmup_steps", "Warmup steps"),
        ("weight_decay", "Weight decay"),
        ("max_grad_norm", "Max grad norm"),
        ("shift", "Noise shift"),
        ("num_inference_steps", "Inference steps"),
        ("seed", "Seed"),
    ]),
    ("VRAM", [
        ("gradient_checkpointing", "Gradient checkpointing"),
        ("gradient_checkpointing_ratio", "Checkpointing ratio"),
        ("offload_encoder", "Offload encoder"),
    ]),
    ("Sequence Cropping", [
        ("crop_mode", "Crop mode"),
        ("chunk_duration", "Chunk duration (s)"),
        ("max_latent_length", "Max latent length (frames)"),
        ("chunk_decay_every", "Coverage decay interval"),
    ]),
    ("Corrected Training", [
        ("cfg_ratio", "CFG dropout ratio"),
        ("loss_fn", "Loss function"),
        ("huber_delta", "Huber/Pseudo-Huber delta"),
        ("loss_weighting", "Loss weighting"),
        ("snr_gamma", "SNR gamma"),
    ]),
    ("All the Levers", [
        ("ema_decay", "EMA decay"),
        ("ema_start_step", "EMA start step"),
        ("val_split", "Validation split"),
        ("adaptive_timestep_ratio", "Adaptive timestep ratio"),
        ("t_bias", "Timestep bias"),
        ("channel_balance", "Channel balancing"),
        ("vae_channel_prior", "VAE channel prior"),
        ("latent_noise", "Latent noise scale"),
        ("legacy_loss", "Legacy loss mode"),
        ("warmup_start_factor", "Warmup start factor"),
        ("cosine_eta_min_ratio", "Cosine eta_min ratio"),
        ("cosine_restarts_count", "Cosine restarts count"),
        ("save_best_every_n_steps", "Step-level best interval"),
    ]),
    ("Checkpointing", [
        ("output_dir", "Output dir"),
        ("run_name", "Run name"),
        ("save_every", "Save every N epochs"),
        ("save_best", "Save best model"),
        ("save_best_after", "Best-model tracking after"),
        ("early_stop_patience", "Early stop patience"),
        ("target_loss", "Target loss"),
        ("target_loss_floor", "Target loss LR floor"),
        ("target_loss_warmup", "Cruise warmup steps"),
        ("target_loss_smoothing", "Cruise smoothing (EMA β)"),
        ("resume_from", "Resume from"),
    ]),
    ("Logging", [
        ("log_dir", "TensorBoard dir"),
        ("log_every", "Log every N steps"),
        ("log_heavy_every", "Grad norms every N steps"),
    ]),
]


_ADAPTER_GROUP_MAP: Dict[str, Tuple[str, List[Tuple[str, str]]]] = {
    "lora": ("LoRA", _LORA_KEYS),
    "dora": ("DoRA", _DORA_KEYS),
    "lokr": ("LoKR", _LOKR_KEYS),
    "loha": ("LoHA", _LOHA_KEYS),
    "oft": ("OFT [Experimental]", _OFT_KEYS),
}


def _build_groups(answers: dict) -> List[Tuple[str, List[Tuple[str, str]]]]:
    """Return ordered groups with the correct adapter section inserted."""
    adapter = answers.get("adapter_type", "lora")
    adapter_group = _ADAPTER_GROUP_MAP.get(adapter, ("LoRA", _LORA_KEYS))
    groups = []
    for name, keys in _GROUPS_TEMPLATE:
        groups.append((name, keys))
        if name == "Device":
            groups.append(adapter_group)
    return groups


def _is_default(key: str, value: Any) -> bool:
    """Return True when *value* matches the known default for *key*."""
    if key not in _DEFAULTS:
        return True  # unknown keys are not highlighted
    return value == _DEFAULTS[key]


def _resolve(key: str, answers: dict) -> Any:
    """Resolve a display key from the answers dict.

    Handles computed pseudo-keys like ``_effective_batch``.
    """
    if key == "_effective_batch":
        bs = answers.get("batch_size", 1)
        ga = answers.get("gradient_accumulation", 4)
        try:
            return int(bs) * int(ga)
        except (TypeError, ValueError):
            return None
    if key == "scheduler_formula":
        val = answers.get(key, "")
        if not val:
            return None
        return val[:60] + "..." if len(val) > 60 else val
    return answers.get(key)


def _fmt(value: Any) -> str:
    """Format a single value for display."""
    if value is None:
        return "(auto)"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, float) and 0 < abs(value) < 0.001:
        return f"{value:.1e}"
    if isinstance(value, float):
        return f"{value:g}"
    if isinstance(value, list):
        return ", ".join(str(v) for v in value)
    return str(value)


def show_review_table(answers: dict) -> None:
    """Print a full grouped review table from wizard answers.

    Non-default values are highlighted in yellow (Rich) or marked
    with ``*`` (plain text) so the user can quickly spot changes.
    """
    groups = _build_groups(answers)

    if is_rich_active() and console is not None:
        from rich.table import Table
        from rich.panel import Panel

        table = Table(show_header=True, header_style="bold",
                      border_style="dim", pad_edge=True, expand=False)
        table.add_column("Parameter", style="dim", min_width=22)
        table.add_column("Value", min_width=30)

        for group_name, keys in groups:
            table.add_row(f"[bold cyan]{group_name}[/]", "")
            for key, label in keys:
                value = _resolve(key, answers)
                formatted = _fmt(value)
                if _is_default(key, value):
                    table.add_row(f"  {label}", _esc(formatted))
                else:
                    table.add_row(f"  {label}", f"[bold yellow]{_esc(formatted)}[/]")

        console.print(Panel(table, title="[bold]Training Configuration[/]",
                            border_style="blue", padding=(0, 1)))
    else:
        print("\n  Training Configuration")
        print("  " + "=" * 50)
        for group_name, keys in groups:
            print(f"\n  [{group_name}]")
            for key, label in keys:
                value = _resolve(key, answers)
                formatted = _fmt(value)
                marker = " *" if not _is_default(key, value) else ""
                print(f"    {label:.<26s} {formatted}{marker}")
        print("  " + "=" * 50)
        print("  (* = non-default value)")


# ---- CLI command export ------------------------------------------------------

# Keys that are internal to the wizard and should not appear in CLI output.
_SKIP_KEYS = {
    "_from_wizard", "_auto_preprocess_audio_dir", "_fisher_map_detected",
    "_pp_recommended", "_pp_sample_count", "_turbo_selected",
    "config_mode", "base_model",
    "self_target_modules_str", "cross_target_modules_str",
    "target_modules_str",
    "use_dora",
}

# answer-key → CLI flag name (only where they differ).
_KEY_TO_FLAG = {
    "model_variant": "model",
    "adapter_type": "adapter",
    "learning_rate": "lr",
    "save_every": "save-every",
    "log_every": "log-every",
    "log_heavy_every": "log-heavy-every",
}


def _cli_flag(key: str) -> str:
    """Convert an answer key to its CLI flag name."""
    if key in _KEY_TO_FLAG:
        return f"--{_KEY_TO_FLAG[key]}"
    return f"--{key.replace('_', '-')}"


def build_cli_command(answers: dict) -> str:
    """Build a reproducible CLI command string from wizard answers."""
    parts = ["sidestep train"]

    for key, value in sorted(answers.items()):
        if key.startswith("_") or key in _SKIP_KEYS:
            continue
        if value is None:
            continue
        if isinstance(value, bool):
            if value:
                parts.append(_cli_flag(key))
            else:
                parts.append(f"--no-{key.replace('_', '-')}")
            continue
        if isinstance(value, list):
            parts.append(f"{_cli_flag(key)} {' '.join(str(v) for v in value)}")
            continue
        if key == "scheduler_formula" and value:
            import shlex
            parts.append(f"{_cli_flag(key)} {shlex.quote(str(value))}")
            continue
        parts.append(f"{_cli_flag(key)} {value}")

    return " \\\n  ".join(parts)


def save_cli_command(answers: dict) -> Optional[Path]:
    """Save the CLI command to a file in the output directory.

    Returns the path written, or None if the output dir is not set.
    """
    output_dir = answers.get("output_dir")
    if not output_dir:
        return None

    path = Path(output_dir) / "cli_command.txt"
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        cmd = build_cli_command(answers)
        path.write_text(cmd + "\n", encoding="utf-8")
    except OSError:
        return None

    print_rich(f"\n[dim]Session CLI command saved to[/] [bold]{_esc(str(path))}[/]")

    return path
