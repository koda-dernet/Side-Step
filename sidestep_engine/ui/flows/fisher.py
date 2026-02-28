"""
Wizard flow for Preprocessing++.

Uses the same step-list pattern as ``flows_estimate.py`` with
go-back navigation.  Simpler than the estimate flow: no LoRA
settings needed (rank is the only relevant parameter), always
module-level, always saves inside dataset_dir.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any, Callable

logger = logging.getLogger(__name__)

from sidestep_engine.ui.prompt_helpers import (
    GoBack,
    _esc,
    ask,
    ask_bool,
    ask_path,
    blank_line,
    menu,
    native_path,
    print_message,
    section,
    step_indicator,
)
from sidestep_engine.ui.flows.common import is_turbo


def _step_model(a: dict) -> None:
    """Collect checkpoint, variant, and dataset directory.

    Raises:
        GoBack: If the user presses back.
    """
    from sidestep_engine.ui.prompt_helpers import print_message
    from sidestep_engine.ui.flows.wizard_shared_steps import (
        ask_model_and_checkpoint,
        ask_dataset_folder,
    )

    section("Preprocessing++")
    print_message(
        "Scans your dataset and auto-picks what to train, then assigns\n"
        "  adaptive ranks for stronger, cleaner fine-tuning.\n"
        "  It saves preprocessing++ metadata in your dataset directory\n"
        "  and training uses it automatically.\n\n"
        "  This works TOO well and it is DISGUSTINGLY POWERFUL, if you use this,\n"
        "  use a lower learning rate than usual, or you might overfit.\n\n"
        "  Compatible adapters: LoRA, DoRA.\n"
        "  LoKR, LoHA, and OFT do not support PP++ because they lack\n"
        "  PEFT's per-module rank_pattern mechanism.",
        kind="dim",
    )

    ask_model_and_checkpoint(a, default_variant="turbo", prompt_base_model=False)
    a["_turbo_selected"] = is_turbo(a)
    ask_dataset_folder(a, allow_audio=True)


def _run_pp_inline_preprocess(answers: dict) -> None:
    """Run preprocessing inline when PP++ detects raw audio.

    Delegates to the shared ``run_inline_preprocess`` runner.

    Raises:
        SystemExit: If preprocessing fails and the user declines.
    """
    from sidestep_engine.ui.flows.inline_preprocess import run_inline_preprocess

    try:
        run_inline_preprocess(answers, label="Auto-Preprocessing (for Preprocessing++)")
    except RuntimeError as exc:
        print_message(f"\nPreprocessing failed: {exc}", kind="error")
        if not ask_bool("Continue to Preprocessing++ anyway (requires .pt tensors)?", default=False):
            raise SystemExit(1) from exc


def _step_focus(a: dict) -> None:
    """Timestep focus selection.

    Raises:
        GoBack: If the user presses back.
    """
    section("Timestep Focus")
    print_message(
        "Controls which aspect of the audio the analysis targets:\n"
        "    balanced   -- full timestep range (recommended)\n"
        "    texture    -- timbre, sonic character (style transfer only)\n"
        "    structure  -- rhythm, tempo, beat grid",
        kind="dim",
    )

    a["timestep_focus"] = menu(
        "Timestep focus",
        [
            ("balanced", "Balanced (recommended)"),
            ("texture", "Texture / style transfer"),
            ("structure", "Structure / rhythm transfer"),
        ],
        default=1,
        allow_back=True,
    )


def _step_rank_budget(a: dict) -> None:
    """Rank budget parameters with sanity validation.

    Raises:
        GoBack: If the user presses back.
    """
    _RANK_FLOOR = 4
    _RANK_SOFT_CEILING = 512

    section("Rank Budget")

    while True:
        a["rank"] = ask(
            "Base rank (median target)", default=a.get("rank", 64),
            type_fn=int, allow_back=True,
        )
        a["rank_min"] = ask(
            "Minimum rank", default=a.get("rank_min", 16),
            type_fn=int, allow_back=True,
        )
        a["rank_max"] = ask(
            "Maximum rank", default=a.get("rank_max", 128),
            type_fn=int, allow_back=True,
        )

        errors: list[str] = []
        if a["rank_min"] < _RANK_FLOOR:
            errors.append(f"Minimum rank must be >= {_RANK_FLOOR} (got {a['rank_min']})")
        if a["rank"] < 1 or a["rank_max"] < 1:
            errors.append("Rank values must be positive")
        if not (a["rank_min"] <= a["rank"] <= a["rank_max"]):
            errors.append(
                f"Must satisfy rank_min <= rank <= rank_max "
                f"({a['rank_min']} <= {a['rank']} <= {a['rank_max']})"
            )

        if errors:
            for e in errors:
                print_message(e, kind="error")
            blank_line()
            continue

        if a["rank_max"] > _RANK_SOFT_CEILING:
            print_message(
                f"Maximum rank {a['rank_max']} is very high (recommended <= {_RANK_SOFT_CEILING}).\n"
                "  Extremely high ranks waste VRAM and can degrade quality.",
                kind="warn",
            )
            if not ask_bool("Continue with this rank budget?", default=True, allow_back=True):
                continue

        break


def _step_confirm(a: dict) -> None:
    """Show estimated time and confirm.

    Raises:
        GoBack: If the user presses back.
    """
    ds = Path(a["dataset_dir"])
    n_files = len(list(ds.glob("*.pt")))
    est_min = max(0.5, n_files * 0.12)

    section("Confirm")

    if n_files < 5:
        print_message(
            f"Very small dataset ({n_files} sample{'s' if n_files != 1 else ''}).\n"
            "  Fisher analysis needs variety to rank modules reliably.\n"
            "  Results may be unreliable with fewer than 5 samples.",
            kind="warn",
        )
        if not ask_bool(
            "Continue with a small dataset?",
            default=False,
            allow_back=True,
        ):
            raise GoBack

    print_message(f"Dataset:  {n_files} preprocessed samples")
    print_message(f"Focus:    {a.get('timestep_focus', 'balanced')}")
    print_message(
        f"Ranks:    {a.get('rank', 64)} (base), "
        f"{a.get('rank_min', 16)}-{a.get('rank_max', 128)}"
    )
    print_message(f"Est time: ~{est_min:.0f} min")

    # VRAM pre-check
    _PP_MIN_VRAM_MB = 7000
    try:
        from sidestep_engine.models.gpu_utils import get_available_vram_mb
        free_mb = get_available_vram_mb()
        if free_mb is not None and free_mb < _PP_MIN_VRAM_MB:
            print_message(
                f"Low GPU memory ({free_mb:.0f} MB free).\n"
                f"  Preprocessing++ typically needs ~{_PP_MIN_VRAM_MB // 1000} GB free VRAM.\n"
                "  Close other GPU consumers or expect possible out-of-memory errors.",
                kind="warn",
            )
            blank_line()
    except Exception as exc:
        logger.debug("VRAM pre-check skipped: %s", exc)

    if a.get("_turbo_selected"):
        print_message(
            "Turbo selected: Preprocessing++ can destabilize Turbo training.\n"
            "Base models are strongly recommended for this workflow.",
            kind="warn",
        )
        if not ask_bool(
            "Proceed with Preprocessing++ on turbo anyway?",
            default=False,
            allow_back=True,
        ):
            a["_force_model_repick"] = True
            raise GoBack

    if not ask_bool("Proceed?", default=True, allow_back=True):
        raise GoBack


_STEPS: list[tuple[str, Callable[..., Any]]] = [
    ("Model & Dataset", _step_model),
    ("Timestep Focus", _step_focus),
    ("Rank Budget", _step_rank_budget),
    ("Confirm", _step_confirm),
]


def wizard_preprocessing_pp(preset: dict | None = None) -> argparse.Namespace:
    """Interactive wizard for Preprocessing++.

    Args:
        preset: Optional pre-filled defaults (session carry-over).

    Returns:
        A populated ``argparse.Namespace`` for the fisher subcommand.

    Raises:
        GoBack: If the user backs out of the first step.
    """
    from sidestep_engine.ui.flows.common import offer_load_preset_subset

    answers: dict = dict(preset) if preset else {}
    offer_load_preset_subset(
        answers,
        allowed_fields={"rank", "rank_min", "rank_max", "timestep_focus"},
        prompt="Load a preset for rank/focus defaults?",
        preserve_fields={"checkpoint_dir", "model_variant", "dataset_dir"},
    )
    total = len(_STEPS)
    i = 0

    while i < total:
        label, step_fn = _STEPS[i]
        try:
            step_indicator(i + 1, total, label)
            step_fn(answers)
            i += 1
        except GoBack:
            if answers.pop("_force_model_repick", False):
                i = 0
                continue
            if i == 0:
                raise
            i -= 1

    # Run inline preprocessing if raw audio was detected
    if answers.get("_auto_preprocess_audio_dir"):
        _run_pp_inline_preprocess(answers)

    return argparse.Namespace(
        subcommand="analyze",
        plain=False,
        yes=True,
        _from_wizard=True,
        checkpoint_dir=answers["checkpoint_dir"],
        model_variant=answers["model_variant"],
        device="auto",
        precision="auto",
        dataset_dir=answers["dataset_dir"],
        rank=answers.get("rank", 64),
        rank_min=answers.get("rank_min", 16),
        rank_max=answers.get("rank_max", 128),
        timestep_focus=answers.get("timestep_focus", "balanced"),
        runs=None,
        batches=None,
        convergence_patience=5,
        fisher_output=None,
    )


def wizard_fisher(preset: dict | None = None) -> argparse.Namespace:
    """Backward-compatible alias for ``wizard_preprocessing_pp``."""
    return wizard_preprocessing_pp(preset=preset)
