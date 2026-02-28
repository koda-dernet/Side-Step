"""
Logging, checkpoint, run-name, and chunk-duration wizard steps.

Extracted from ``train_steps.py`` to meet the module LOC policy.
"""

from __future__ import annotations

import os
from sidestep_engine.training_defaults import DEFAULT_SAVE_EVERY

from sidestep_engine.ui.flows.train_steps_helpers import smart_save_best_default as _smart_save_best_default
from sidestep_engine.ui.prompt_helpers import (
    _esc,
    ask,
    ask_bool,
    ask_output_path,
    ask_path,
    print_message,
    print_rich,
    section,
)


def step_logging(a: dict) -> None:
    """Logging and checkpoint settings.

    In basic mode, only save_every and save_best are prompted.
    Resume, heavy logging, and strict_resume are skipped (defaults applied).
    """
    _is_basic = a.get("config_mode") == "basic"

    section("Logging & Checkpoints (press Enter for defaults)")
    a["save_every"] = ask(
        "Save checkpoint every N epochs",
        default=a.get("save_every", DEFAULT_SAVE_EVERY),
        type_fn=int,
        allow_back=True,
    )

    # Basic mode: auto-default save_best, logging, and resume
    if _is_basic:
        a.setdefault("save_best", True)
        a["save_best_after"] = _smart_save_best_default(a)
        a.setdefault("early_stop_patience", 0)
        a.setdefault("log_every", 10)
        a.setdefault("log_heavy_every", 50)
        a.setdefault("resume_from", None)
        return

    a["save_best"] = ask_bool("Auto-save best model (smoothed loss)", default=a.get("save_best", True), allow_back=True)
    if a["save_best"]:
        _sba_default = a.get("save_best_after") or _smart_save_best_default(a)
        a["save_best_after"] = ask("Start best-model tracking after epoch", default=_sba_default, type_fn=int, allow_back=True)
        a["early_stop_patience"] = ask("Early stop patience (0=disabled)", default=a.get("early_stop_patience", 0), type_fn=int, allow_back=True)
    else:
        a["save_best_after"] = a.get("save_best_after") or _smart_save_best_default(a)
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
            print_message(
                f"That's a file -- using checkpoint directory: {parent}",
                kind="warn",
            )
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


def step_chunk_duration(a: dict) -> None:
    """Latent chunking for data augmentation and VRAM savings.

    In basic mode, chunking is disabled by default without prompting.
    """
    if a.get("config_mode") == "basic":
        a.setdefault("chunk_duration", None)
        return

    section("Latent Chunking (optional)")

    print_message(
        "Latent chunking slices preprocessed tensors into random\n"
        "  fixed-length windows each iteration, providing data augmentation\n"
        "  (the model sees different parts of each song every epoch) and\n"
        "  reducing VRAM usage for long songs.",
        kind="dim",
    )
    print_rich(
        "\n  [bold yellow]Warning:[/][dim] Chunks shorter than 60 seconds can hurt\n"
        "  training quality instead of enriching it. Use shorter chunks\n"
        "  only if you need to reduce VRAM and understand the trade-off.\n"
        "  Leave disabled (0) if your songs are already short or you have\n"
        "  enough VRAM.[/]"
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

        if 0 < chunk < 10:
            print_rich(
                f"[bold red]Rejected:[/] {chunk}s is below the hard minimum of 10 seconds.\n"
                "  Chunks this short produce degenerate latents. Use at least 10s."
            )
            continue

        if chunk < 60:
            print_rich(
                f"[bold yellow]Caution:[/] {chunk}s chunks are below the recommended\n"
                "  60s minimum. This may reduce training quality, especially for\n"
                "  full-length inference. Consider using 60s or higher."
            )
            if not ask_bool("Use this chunk size anyway?", default=False, allow_back=True):
                continue

        a["chunk_duration"] = chunk

        # Coverage decay: configurable in advanced, auto in basic
        if a.get("config_mode") != "basic":
            a["chunk_decay_every"] = ask(
                "Coverage decay interval (epochs, 0=off)",
                default=a.get("chunk_decay_every", 10),
                type_fn=int, allow_back=True,
            )
        else:
            a.setdefault("chunk_decay_every", 10)
        return
