"""
Error presentation for the ACE-Step Training V2 CLI.

Wraps exceptions in Rich Panels with actionable suggestions.
Falls back to plain ``[FAIL]`` messages when Rich is unavailable.
"""

from __future__ import annotations

import sys
import traceback
from acestep.training_v2.ui import console, is_rich_active

# ---- Suggestion database ----------------------------------------------------

_SUGGESTIONS: dict[str, list[str]] = {
    # CUDA OOM
    "CUDA out of memory": [
        "Reduce --batch-size (try 1)",
        "Reduce --rank (try 32 or 16)",
        "Reduce --gradient-accumulation",
        "Close other GPU-consuming processes",
    ],
    "OutOfMemoryError": [
        "Reduce --batch-size (try 1)",
        "Reduce --rank (try 32 or 16)",
    ],
    # Model loading
    "No such file or directory": [
        "Check --checkpoint-dir points to the correct path",
        "Verify the model variant directory exists (e.g. acestep-v15-turbo/)",
    ],
    "not found": [
        "Check that all required paths exist",
        "Run preprocessing first if .pt files are missing",
    ],
    # LoRA / PEFT
    "peft": [
        "Install PEFT:  pip install peft",
    ],
    "PeftModel": [
        "Install PEFT:  pip install peft",
    ],
    # Fabric
    "lightning": [
        "Install Lightning Fabric:  pip install lightning",
        "Or the trainer will fall back to a basic PyTorch loop",
    ],
    # Flash attention
    "flash_attn": [
        "Flash Attention 2 is optional. The model will fall back to SDPA",
        "To install:  pip install flash-attn --no-build-isolation",
    ],
    "attn_implementation": [
        "Attention backend fallback is normal on unsupported setups",
        "Flash Attention 2 needs CUDA + Ampere (RTX 30xx+) + bf16/fp16",
        "SDPA fallback is valid, but can use more VRAM than Flash Attention 2",
    ],
    # Dtype
    "bfloat16": [
        "Your GPU may not support bf16.  Try --precision fp16",
    ],
    # Dataset manifest / JSON path handling
    "Invalid \\escape": [
        "manifest.json contains an unescaped Windows path",
        "Use escaped backslashes (\\\\) or forward slashes (/)",
        "Example: C:/data/song.pt or C:\\\\data\\\\song.pt",
    ],
    "JSONDecodeError": [
        "A JSON file is malformed (commonly manifest.json or dataset JSON)",
        "Validate JSON syntax and quote/escape all path strings",
    ],
    "manifest.json": [
        "Verify manifest.json is valid JSON and contains a 'samples' list",
        "Re-run preprocessing to regenerate manifest.json if needed",
    ],
    # Preprocessing++
    "Fisher analysis": [
        "Try reducing dataset scope with --fisher-batches",
        "Ensure gradient checkpointing is working (check GPU driver)",
    ],
    "Preprocessing++": [
        "Try reducing dataset scope with --fisher-batches",
        "Ensure gradient checkpointing is working (check GPU driver)",
    ],
    "fisher_map": [
        "Re-run Preprocessing++: train.py fisher --dataset-dir <dir>",
        "Or bypass with --ignore-fisher-map",
    ],
    # Generic
    "Permission denied": [
        "Check file/directory permissions on --output-dir",
    ],
    "No space left on device": [
        "Free up disk space or change --output-dir to a different drive",
    ],
}


def _find_suggestions(error_str: str) -> list[str]:
    """Match error text against known patterns and return suggestions."""
    suggestions: list[str] = []
    lower = error_str.lower()
    for pattern, tips in _SUGGESTIONS.items():
        if pattern.lower() in lower:
            suggestions.extend(tips)
    return suggestions


# ---- Public API -------------------------------------------------------------

def handle_error(
    exc: BaseException,
    context: str = "Training",
    show_traceback: bool = False,
) -> None:
    """Display an error with suggestions.

    Args:
        exc: The exception that occurred.
        context: Short label (e.g. "Model loading", "Training").
        show_traceback: If True, include the full traceback.
    """
    error_str = str(exc)
    exc_type = type(exc).__name__
    suggestions = _find_suggestions(error_str)

    if is_rich_active() and console is not None:
        _show_rich(exc, exc_type, error_str, context, suggestions, show_traceback)
    else:
        _show_plain(exc, exc_type, error_str, context, suggestions, show_traceback)


def show_info(msg: str) -> None:
    """Display an info message."""
    if is_rich_active() and console is not None:
        console.print(f"[bold blue][INFO][/] {msg}")
    else:
        print(f"[INFO] {msg}", file=sys.stderr)



# ---- Rich rendering --------------------------------------------------------

def _show_rich(
    exc: BaseException,
    exc_type: str,
    error_str: str,
    context: str,
    suggestions: list[str],
    show_traceback: bool,
) -> None:
    from rich.panel import Panel
    from rich.text import Text

    assert console is not None

    body = Text()
    body.append(f"{exc_type}: ", style="bold red")
    body.append(f"{error_str}\n", style="red")

    if suggestions:
        body.append("\nSuggested fixes:\n", style="bold yellow")
        for i, tip in enumerate(suggestions, 1):
            body.append(f"  {i}. {tip}\n", style="yellow")

    if show_traceback:
        body.append("\nTraceback:\n", style="dim")
        tb_str = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
        body.append(tb_str, style="dim red")

    console.print(
        Panel(
            body,
            title=f"[bold red]{context} Error[/]",
            border_style="red",
            padding=(0, 1),
        )
    )


# ---- Plain rendering -------------------------------------------------------

def _show_plain(
    exc: BaseException,
    exc_type: str,
    error_str: str,
    context: str,
    suggestions: list[str],
    show_traceback: bool,
) -> None:
    print(f"\n[Side-Step] {context} Error: {exc_type}: {error_str}", file=sys.stderr)
    if suggestions:
        print("       Suggested fixes:", file=sys.stderr)
        for i, tip in enumerate(suggestions, 1):
            print(f"         {i}. {tip}", file=sys.stderr)
    if show_traceback:
        traceback.print_exception(type(exc), exc, exc.__traceback__, file=sys.stderr)
    print(file=sys.stderr)
