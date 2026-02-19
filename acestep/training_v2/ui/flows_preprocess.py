"""
Wizard flow for preprocessing audio into tensors.

Uses a step-list pattern for go-back navigation.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Callable

from acestep.training_v2.ui import console, is_rich_active
from acestep.training_v2.ui.prompt_helpers import (
    DEFAULT_NUM_WORKERS,
    GoBack,
    _esc,
    ask,
    ask_path,
    menu,
    native_path,
    section,
    step_indicator,
)


# ---- Steps ------------------------------------------------------------------

def _step_model(a: dict) -> None:
    """Checkpoint directory and model selection."""
    from acestep.training_v2.settings import get_checkpoint_dir
    from acestep.training_v2.model_discovery import pick_model, prompt_base_model
    from acestep.training_v2.ui.flows_common import show_model_picker_fallback_hint

    section("Preprocessing Settings")

    ckpt_default = a.get("checkpoint_dir") or get_checkpoint_dir() or native_path("./checkpoints")
    a["checkpoint_dir"] = ask_path(
        "Checkpoint directory", default=ckpt_default,
        must_exist=True, allow_back=True,
    )

    # Model picker (replaces hardcoded turbo/base/sft choices)
    result = pick_model(a["checkpoint_dir"])
    if result is None:
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
        if not info.is_official and info.base_model == "unknown":
            a["base_model"] = prompt_base_model(name)


def _step_source(a: dict) -> None:
    """Dataset JSON or audio directory."""
    if is_rich_active() and console is not None:
        console.print(
            "\n  [dim]A dataset JSON provides lyrics, genre, BPM, key, and "
            "other metadata for each audio file, plus audio file paths.\n"
            "  Without it, you will be asked for an audio directory and all "
            "tracks will default to [Instrumental] with no genre/BPM info.[/]"
        )
    else:
        print(
            "\n  A dataset JSON provides lyrics, genre, BPM, key, and "
            "other metadata for each audio file, plus audio file paths.\n"
            "  Without it, you will be asked for an audio directory and all "
            "tracks will default to [Instrumental] with no genre/BPM info."
        )

    a["dataset_json"] = _ask_dataset_json(a.get("dataset_json"))

    if not a.get("dataset_json"):
        if is_rich_active() and console is not None:
            console.print("  [dim]Subdirectories will be scanned recursively.[/]")
        else:
            print("  Subdirectories will be scanned recursively.")
        a["audio_dir"] = ask_path(
            "Audio directory (source audio files)",
            default=a.get("audio_dir"),
            must_exist=True, allow_back=True,
        )
    else:
        a["audio_dir"] = None


def _ask_dataset_json(default: str | None) -> str | None:
    """Prompt for a dataset JSON path with search-nearby fallback.

    When the entered path is not found, searches common sibling
    directories (``datasets/``, ``data/``) and CWD for a matching
    filename before giving up.  Lets the user retry instead of
    silently falling through to audio-directory mode.
    """
    from acestep.training_v2.ui.prompt_helpers import ask_bool

    while True:
        dataset_json = ask(
            "Dataset JSON file (leave empty to skip)",
            default=default, allow_back=True,
        )
        if dataset_json in (None, "None", ""):
            return None

        resolved = _resolve_dataset_json(dataset_json)
        if resolved is not None:
            return str(resolved)

        # Not found -- offer to retry
        _print_not_found(dataset_json)
        if ask_bool("Try a different path?", default=True):
            default = dataset_json  # keep what they typed as the new default
            continue
        return None


def _resolve_dataset_json(raw_path: str) -> Path | None:
    """Try to find the dataset JSON, searching nearby if needed."""
    candidate = Path(raw_path).expanduser()

    # 1. Exact path (absolute or relative to CWD)
    resolved = candidate.resolve()
    if resolved.is_file():
        return resolved

    # 2. If they gave just a filename, search common subdirectories
    search_dirs = [
        Path.cwd(),
        Path.cwd() / "datasets",
        Path.cwd() / "data",
    ]
    name = candidate.name
    for d in search_dirs:
        p = (d / name).resolve()
        if p.is_file():
            _print_found_nearby(raw_path, p)
            return p

    # 3. Glob for the filename anywhere one level deep from CWD
    for match in sorted(Path.cwd().glob(f"*/{name}")):
        if match.is_file():
            _print_found_nearby(raw_path, match)
            return match

    return None


def _print_found_nearby(original: str, found: Path) -> None:
    """Tell the user we found their file at a different path."""
    if is_rich_active() and console is not None:
        console.print(f"  [yellow]'{_esc(original)}' not at that exact path,[/]")
        console.print(f"  [green]but found it at: {_esc(found)}[/]")
    else:
        print(f"  '{original}' not at that exact path,")
        print(f"  but found it at: {found}")


def _print_not_found(path: str) -> None:
    """Tell the user the file was not found anywhere."""
    searched = [
        str(Path.cwd()),
        str(Path.cwd() / "datasets"),
        str(Path.cwd() / "data"),
        f"{Path.cwd()}/*/",
    ]
    if is_rich_active() and console is not None:
        console.print(f"  [red]Not found: {_esc(path)}[/]")
        console.print("  [dim]Searched:[/]")
        for s in searched:
            console.print(f"    [dim]- {_esc(s)}[/]")
        console.print("  [dim]Tip: paste an absolute path if you're unsure.[/]")
    else:
        print(f"  Not found: {path}")
        print("  Searched:")
        for s in searched:
            print(f"    - {s}")
        print("  Tip: paste an absolute path if you're unsure.")


def _step_output(a: dict) -> None:
    """Output directory for tensor files."""
    a["tensor_output"] = ask(
        "Output directory for .pt tensor files",
        default=a.get("tensor_output"),
        required=True, allow_back=True,
    )


def _step_normalize(a: dict) -> None:
    """Audio normalization before VAE encoding."""
    from acestep.training_v2.ui.prompt_helpers import ask_bool

    if is_rich_active() and console is not None:
        console.print(
            "\n  [dim]Normalization ensures consistent loudness across training audio.\n"
            "  Peak normalizes to -1.0 dBFS (matches ACE-Step output).\n"
            "  LUFS normalizes to -14 LUFS (broadcast standard, requires pyloudnorm).\n"
            "  If unsure, 'peak' is a safe default.[/]"
        )
    else:
        print(
            "\n  Normalization ensures consistent loudness across training audio.\n"
            "  Peak normalizes to -1.0 dBFS (matches ACE-Step output).\n"
            "  LUFS normalizes to -14 LUFS (broadcast standard, requires pyloudnorm).\n"
            "  If unsure, 'peak' is a safe default."
        )

    if ask_bool("Normalize audio before encoding?", default=True, allow_back=True):
        a["normalize"] = menu(
            "Normalization method",
            [
                ("peak", "Peak (-1.0 dBFS, no extra deps, matches ACE-Step)"),
                ("lufs", "LUFS (-14 LUFS, perceptually uniform, needs pyloudnorm)"),
            ],
            default=1,
            allow_back=True,
        )
    else:
        a["normalize"] = "none"

    if a.get("normalize") == "lufs":
        from acestep.training_v2.ui.dependency_check import (
            ensure_optional_dependencies,
            required_preprocess_optionals,
        )

        unresolved = ensure_optional_dependencies(
            required_preprocess_optionals("lufs"),
            interactive=True,
            allow_install_prompt=True,
        )
        if unresolved:
            if ask_bool(
                "LUFS dependency still missing. Switch normalization to peak so preprocessing can continue?",
                default=True,
                allow_back=True,
            ):
                a["normalize"] = "peak"


def _step_scan_durations(a: dict) -> None:
    """Scan audio files and show per-song duration feedback."""
    from acestep.training_v2.audio_duration import get_audio_duration, detect_max_duration
    from acestep.training_v2.preprocess_discovery import discover_audio_files

    section("Audio Duration Scan")

    audio_files = discover_audio_files(
        a.get("audio_dir"), a.get("dataset_json"),
    )

    if not audio_files:
        if is_rich_active() and console is not None:
            console.print("  [yellow]No audio files found to scan.[/]")
        else:
            print("  No audio files found to scan.")
        a["max_duration"] = 0
        return

    # Show per-song durations
    durations = {}
    for af in audio_files:
        dur = get_audio_duration(str(af))
        durations[af.name] = dur

    if is_rich_active() and console is not None:
        console.print(f"  [bold]Found {len(audio_files)} audio files:[/]\n")
        for name, dur in sorted(durations.items()):
            m, s = divmod(dur, 60)
            console.print(f"    {name:<40s}  {m}m {s:02d}s  ({dur}s)")
        longest = max(durations.values()) if durations else 0
        console.print(f"\n  [bold green]Longest clip: {longest}s[/]")
        console.print("  [dim]Max duration will be set automatically to match.[/]")
    else:
        print(f"  Found {len(audio_files)} audio files:\n")
        for name, dur in sorted(durations.items()):
            m, s = divmod(dur, 60)
            print(f"    {name:<40s}  {m}m {s:02d}s  ({dur}s)")
        longest = max(durations.values()) if durations else 0
        print(f"\n  Longest clip: {longest}s")
        print("  Max duration will be set automatically to match.")

    a["max_duration"] = 0  # signal auto-detect to the preprocess pipeline


# ---- Step list and runner ---------------------------------------------------

_STEPS: list[tuple[str, Callable[..., Any]]] = [
    ("Model & Checkpoint", _step_model),
    ("Audio Source", _step_source),
    ("Output Settings", _step_output),
    ("Audio Normalization", _step_normalize),
    ("Duration Scan", _step_scan_durations),
]


def wizard_preprocess(preset: dict | None = None) -> argparse.Namespace:
    """Interactive wizard for preprocessing.

    Args:
        preset: Optional pre-filled defaults (session carry-over).

    Returns:
        A populated ``argparse.Namespace`` with ``preprocess=True``.

    Raises:
        GoBack: If the user backs out of the first step.
    """
    answers: dict = dict(preset) if preset else {}
    total = len(_STEPS)
    i = 0

    while i < total:
        label, step_fn = _STEPS[i]
        try:
            step_indicator(i + 1, total, label)
            step_fn(answers)
            i += 1
        except GoBack:
            if i == 0:
                raise  # bubble to main menu
            i -= 1

    return argparse.Namespace(
        subcommand="fixed",
        plain=False,
        yes=True,
        _from_wizard=True,
        adapter_type="lora",
        checkpoint_dir=answers["checkpoint_dir"],
        model_variant=answers["model_variant"],
        base_model=answers.get("base_model", answers["model_variant"]),
        device="auto",
        precision="auto",
        dataset_dir=answers.get("tensor_output", ""),
        num_workers=DEFAULT_NUM_WORKERS,
        pin_memory=True,
        prefetch_factor=2 if DEFAULT_NUM_WORKERS > 0 else 0,
        persistent_workers=DEFAULT_NUM_WORKERS > 0,
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
        output_dir=native_path("./lora_output"),
        save_every=10,
        resume_from=None,
        log_dir=None,
        log_every=10,
        log_heavy_every=50,
        optimizer_type="adamw",
        scheduler_type="cosine",
        gradient_checkpointing=True,
        offload_encoder=False,
        preprocess=True,
        audio_dir=answers.get("audio_dir"),
        dataset_json=answers.get("dataset_json"),
        tensor_output=answers.get("tensor_output"),
        max_duration=answers.get("max_duration", 0),
        normalize=answers.get("normalize", "none"),
        cfg_ratio=0.15,
        loss_weighting="none",
        snr_gamma=5.0,
        estimate_batches=None,
        top_k=16,
        granularity="module",
        module_config=None,
        auto_estimate=False,
        estimate_output=None,
    )
