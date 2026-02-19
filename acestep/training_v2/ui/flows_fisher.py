"""
Wizard flow for Preprocessing++.

Uses the same step-list pattern as ``flows_estimate.py`` with
go-back navigation.  Simpler than the estimate flow: no LoRA
settings needed (rank is the only relevant parameter), always
module-level, always saves inside dataset_dir.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Callable

from acestep.training_v2.ui import console, is_rich_active
from acestep.training_v2.ui.prompt_helpers import (
    GoBack,
    _esc,
    ask,
    ask_bool,
    ask_path,
    menu,
    native_path,
    section,
    step_indicator,
)


def _is_turbo_selection(model_variant: str, base_model: str | None = None) -> bool:
    """Return True when selected model/base indicates turbo."""
    labels = [model_variant, base_model or ""]
    for label in labels:
        if isinstance(label, str) and "turbo" in label.lower():
            return True
    return False


def _step_model(a: dict) -> None:
    """Collect checkpoint, variant, and dataset directory.

    Raises:
        GoBack: If the user presses back.
    """
    section("Preprocessing++")
    _msg = (
        "  Scans your dataset and auto-picks what to train, then assigns\n"
        "  adaptive ranks for stronger, cleaner fine-tuning.\n"
        "  It saves preprocessing++ metadata in your dataset directory\n"
        "  and training uses it automatically.\n\n"
        "  This works TOO well and it is DISGUSTINGLY POWERFUL, if you use this,\n"
        "  use a lower learning rate than usual, or you might overfit.\n"
    )
    if is_rich_active() and console is not None:
        console.print(f"  [dim]{_msg}[/]")
    else:
        print(_msg)

    from acestep.training_v2.settings import get_checkpoint_dir
    from acestep.training_v2.model_discovery import pick_model
    from acestep.training_v2.ui.flows_common import (
        describe_preprocessed_dataset_issue,
        show_dataset_issue,
        show_model_picker_fallback_hint,
    )

    ckpt_default = a.get("checkpoint_dir") or get_checkpoint_dir() or native_path("./checkpoints")
    a["checkpoint_dir"] = ask_path(
        "Checkpoint directory",
        default=ckpt_default,
        must_exist=True, allow_back=True,
    )

    result = pick_model(a["checkpoint_dir"])
    base_model = None
    if result is None:
        show_model_picker_fallback_hint()
        a["model_variant"] = ask(
            "Model variant or folder name", default=a.get("model_variant", "turbo"),
            allow_back=True,
        )
    else:
        a["model_variant"] = result[0]
        base_model = getattr(result[1], "base_model", None)
    a["_turbo_selected"] = _is_turbo_selection(a["model_variant"], base_model)

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


def _step_focus(a: dict) -> None:
    """Timestep focus selection.

    Raises:
        GoBack: If the user presses back.
    """
    section("Timestep Focus")
    _msg = (
        "  Controls which aspect of the audio the analysis targets:\n"
        "    texture   -- timbre, sonic character (default for style transfer)\n"
        "    structure  -- rhythm, tempo, beat grid\n"
        "    balanced   -- full timestep range (no focus)\n"
    )
    if is_rich_active() and console is not None:
        console.print(f"  [dim]{_msg}[/]")
    else:
        print(_msg)

    a["timestep_focus"] = menu(
        "Timestep focus",
        [
            ("texture", "Texture / style transfer (recommended)"),
            ("structure", "Structure / rhythm transfer"),
            ("balanced", "Balanced (no focus)"),
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
                if is_rich_active() and console is not None:
                    console.print(f"  [red]{_esc(e)}[/]")
                else:
                    print(f"  {e}")
            print()
            continue

        if a["rank_max"] > _RANK_SOFT_CEILING:
            warn = (
                f"Maximum rank {a['rank_max']} is very high (recommended <= {_RANK_SOFT_CEILING}).\n"
                "  Extremely high ranks waste VRAM and can degrade quality."
            )
            if is_rich_active() and console is not None:
                console.print(f"  [yellow]{_esc(warn)}[/]")
            else:
                print(f"  {warn}")
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
        warn = (
            f"Very small dataset ({n_files} sample{'s' if n_files != 1 else ''}).\n"
            "  Fisher analysis needs variety to rank modules reliably.\n"
            "  Results may be unreliable with fewer than 5 samples."
        )
        if is_rich_active() and console is not None:
            console.print(f"  [yellow]{_esc(warn)}[/]")
        else:
            print(f"  {warn}")
        if not ask_bool(
            "Continue with a small dataset?",
            default=False,
            allow_back=True,
        ):
            raise GoBack

    print(f"  Dataset:  {n_files} preprocessed samples")
    print(f"  Focus:    {a.get('timestep_focus', 'texture')}")
    print(f"  Ranks:    {a.get('rank', 64)} (base), "
          f"{a.get('rank_min', 16)}-{a.get('rank_max', 128)}")
    print(f"  Est time: ~{est_min:.0f} min\n")

    # VRAM pre-check
    _PP_MIN_VRAM_MB = 7000
    try:
        from acestep.training_v2.gpu_utils import get_available_vram_mb
        free_mb = get_available_vram_mb()
        if free_mb is not None and free_mb < _PP_MIN_VRAM_MB:
            vram_warn = (
                f"Low GPU memory ({free_mb:.0f} MB free).\n"
                f"  Preprocessing++ typically needs ~{_PP_MIN_VRAM_MB // 1000} GB free VRAM.\n"
                "  Close other GPU consumers or expect possible out-of-memory errors."
            )
            if is_rich_active() and console is not None:
                console.print(f"  [yellow]{_esc(vram_warn)}[/]")
            else:
                print(f"  {vram_warn}")
            print()
    except Exception:
        pass

    if a.get("_turbo_selected"):
        caution = (
            "Turbo selected: Preprocessing++ can destabilize Turbo training.\n"
            "Base models are strongly recommended for this workflow."
        )
        if is_rich_active() and console is not None:
            console.print(f"  [yellow]{_esc(caution)}[/]")
        else:
            print(f"  {caution}")
        if not ask_bool(
            "Proceed with Preprocessing++ on turbo anyway?",
            default=False,
            allow_back=True,
        ):
            a["_force_model_repick"] = True
            raise GoBack

    confirm = ask(
        "Proceed?", default="y",
        choices=["y", "n"], allow_back=True,
    )
    if confirm.lower() != "y":
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
    from acestep.training_v2.ui.flows_common import offer_load_preset_subset

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

    return argparse.Namespace(
        subcommand="fisher",
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
        timestep_focus=answers.get("timestep_focus", "texture"),
        fisher_runs=None,
        fisher_batches=None,
        convergence_patience=5,
        fisher_output=None,
    )


def wizard_fisher(preset: dict | None = None) -> argparse.Namespace:
    """Backward-compatible alias for ``wizard_preprocessing_pp``."""
    return wizard_preprocessing_pp(preset=preset)
