"""
Shared wizard step helpers for model selection and dataset folder prompts.

Extracted from duplicated logic across ``train_steps``, ``fisher``,
``estimate``, and ``preprocess`` to provide a single source of truth.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple

from sidestep_engine.ui.prompt_helpers import (
    GoBack,
    _esc,
    ask,
    ask_bool,
    ask_path,
    menu,
    native_path,
    print_message,
    print_rich,
)
from sidestep_engine.ui.flows.common import (
    _AUDIO_EXTENSIONS,
    describe_preprocessed_dataset_issue,
    show_dataset_issue,
    show_model_picker_fallback_hint,
)


# ---------------------------------------------------------------------------
# Interactive model picker (moved here from models.discovery to break the
# models → ui import cycle).
# ---------------------------------------------------------------------------

def pick_model(
    checkpoint_dir: "str | Path",
) -> "Optional[Tuple[str, 'ModelInfo']]":
    """Interactive model selector with fuzzy search.

    Scans *checkpoint_dir*, lists all discovered models, and lets the
    user pick by number or type a name to search.

    Returns ``(model_name, ModelInfo)`` or ``None`` if no models found.

    Raises:
        GoBack: When user types 'b'/'back'.
    """
    from sidestep_engine.models.discovery import (
        ModelInfo, scan_models, fuzzy_search, warn_if_no_weights,
    )
    from sidestep_engine.ui import console, is_rich_active

    models = scan_models(checkpoint_dir)
    if not models:
        return None

    options = []
    for m in models:
        tag = "(official)" if m.is_official else f"(custom, base: {m.base_model})"
        options.append((m.name, f"{m.name}  {tag}"))

    options.append(("__search__", "Search by name..."))

    choice = menu(
        "Select a model to train on",
        options,
        default=1,
        allow_back=True,
    )

    if choice == "__search__":
        result = _search_loop(models)
        if result is not None:
            _ui_warn_no_weights(result[1].path, result[0])
        return result

    for m in models:
        if m.name == choice:
            _ui_warn_no_weights(m.path, m.name)
            return (m.name, m)

    return None


def _search_loop(models: list) -> "Optional[Tuple[str, object]]":
    """Fuzzy-search sub-flow for model selection."""
    from sidestep_engine.models.discovery import fuzzy_search
    from sidestep_engine.ui import console, is_rich_active

    while True:
        query = ask("Enter model name (or part of it)", allow_back=True)
        hits = fuzzy_search(query, models)

        if not hits:
            _msg = "  No matches found. Try a different search term."
            if is_rich_active() and console is not None:
                console.print(f"  [yellow]{_msg}[/]")
            else:
                print(_msg)
            continue

        options = []
        for m in hits:
            tag = "(official)" if m.is_official else f"(custom, base: {m.base_model})"
            options.append((m.name, f"{m.name}  {tag}"))
        title = (
            "Matched 1 model — pick one"
            if len(hits) == 1
            else "Multiple matches — pick one"
        )
        try:
            choice = menu(title, options, default=1, allow_back=True)
        except GoBack:
            continue
        for m in hits:
            if m.name == choice:
                return (m.name, m)


def _ui_warn_no_weights(model_path: Path, model_name: str) -> None:
    """Rich-formatted warning when a model directory has no weight files."""
    from sidestep_engine.models.discovery import _has_weight_files
    from sidestep_engine.ui import console, is_rich_active

    if _has_weight_files(model_path):
        return
    _msg = (
        f"  [yellow]Warning: '{_esc(model_name)}' has no model.safetensors, "
        "pytorch_model.bin, or *.safetensors.[/]\n"
        "  [dim]This may be an incomplete download. Loading will likely fail.[/]"
    )
    if is_rich_active() and console is not None:
        console.print(_msg)
    else:
        print(f"  Warning: '{model_name}' has no weight files. This may be an incomplete download.")


def prompt_base_model(model_name: str) -> str:
    """Ask the user which base model a fine-tune descends from.

    Returns ``"turbo"``, ``"base"``, or ``"sft"``.
    """
    from sidestep_engine.ui import console, is_rich_active

    if is_rich_active() and console is not None:
        console.print(
            f"\n  [yellow]'{_esc(model_name)}' appears to be a custom fine-tune.[/]"
        )
        console.print(
            "  [dim]Knowing the base model helps condition timestep sampling.[/]\n"
        )
    else:
        print(f"\n  '{model_name}' appears to be a custom fine-tune.")
        print("  Knowing the base model helps condition timestep sampling.\n")

    return menu(
        "Which base model was this fine-tune trained from?",
        [
            ("turbo", "Turbo (8-step accelerated)"),
            ("base", "Base (full diffusion)"),
            ("sft", "SFT (supervised fine-tune)"),
        ],
        default=1,
        allow_back=True,
    )


# ---------------------------------------------------------------------------
# Wizard step: model + checkpoint directory
# ---------------------------------------------------------------------------

def ask_model_and_checkpoint(
    answers: dict,
    *,
    default_variant: str = "turbo",
    prompt_base_model: bool = True,
) -> None:
    """Prompt for checkpoint directory and model variant.

    Populates ``answers["checkpoint_dir"]``, ``answers["model_variant"]``,
    and optionally ``answers["base_model"]`` via the interactive model
    picker.  Falls back to manual entry when no models are found.

    Args:
        answers: Mutable wizard answers dict.
        default_variant: Default model variant when manual entry is needed.
        prompt_base_model: If ``True`` and the picked model is an
            unofficial fine-tune with unknown base, ask the user to
            identify it.

    Raises:
        GoBack: If the user presses back at any prompt.
    """
    from sidestep_engine.settings import get_checkpoint_dir

    ckpt_default = (
        answers.get("checkpoint_dir")
        or get_checkpoint_dir()
        or native_path("./checkpoints")
    )
    answers["checkpoint_dir"] = ask_path(
        "Checkpoint directory",
        default=ckpt_default,
        must_exist=True,
        allow_back=True,
    )

    result = pick_model(answers["checkpoint_dir"])
    if result is None:
        show_model_picker_fallback_hint()
        answers["model_variant"] = ask(
            "Model variant or folder name",
            default=answers.get("model_variant", default_variant),
            allow_back=True,
        )
        answers["base_model"] = answers["model_variant"]
        _warn_non_base_variant(answers)
        return

    name, info = result
    answers["model_variant"] = name
    answers["base_model"] = getattr(info, "base_model", name)

    if prompt_base_model and not info.is_official and info.base_model == "unknown":
        # Module-level prompt_base_model is shadowed by the kwarg, so
        # reference via globals().
        _prompt_fn = globals()["prompt_base_model"]
        answers["base_model"] = _prompt_fn(name)

    _warn_non_base_variant(answers)


def _warn_non_base_variant(answers: dict) -> None:
    """Warn when the selected variant is not 'base' and ask for confirmation.

    Community consensus is that training on the **base** variant produces
    the best results.  Training on turbo or SFT works but is suboptimal.
    If the user declines, raises :class:`GoBack` so the model picker
    re-appears.
    """
    from sidestep_engine.ui.prompt_helpers import GoBack

    base = answers.get("base_model", answers.get("model_variant", ""))
    label = base.lower() if isinstance(base, str) else ""
    if "base" in label:
        return

    if "sft" in label:
        reason = "SFT models respond well to prompts but produce lower-quality LoRAs than base."
    elif "turbo" in label:
        reason = "Turbo is optimised for fast inference, not training. LoRAs trained on base transfer to turbo at inference time."
    else:
        reason = "Training on the base variant is strongly recommended for best results."

    print_message(
        f"\nYou selected a non-base variant. {reason}",
        kind="warn",
    )
    print_message(
        "The community strongly recommends: always train on base.",
        kind="warn",
    )
    if not ask_bool("Continue with this variant anyway?", default=False):
        answers["_force_model_repick"] = True
        raise GoBack()


def show_whats_changed_notice() -> None:
    """Display a one-time notice about the workflow change from beta 0.9.0.

    Explains that users no longer need to preprocess separately before
    training — they can point directly to a raw audio folder.
    """
    print_rich(
        "\n[bold white]What changed from beta 0.9.0:[/]\n"
        "  Previously you had to preprocess your audio into .pt tensors\n"
        "  [bold]before[/] training (via the Preprocess menu). That extra step\n"
        "  is no longer needed.\n"
        "\n  [bold green]Now:[/] Just point to your [bold]raw audio folder[/] \u2013 Side-Step\n"
        "  will auto-build a dataset.json from your .txt sidecars, ask you\n"
        "  about normalization, and preprocess everything inline.\n"
        "\n  Already have .pt tensors from a previous run? You can still\n"
        "  point to those and skip straight to training."
    )


def ask_dataset_folder(
    answers: dict,
    *,
    allow_audio: bool = True,
) -> None:
    """Prompt for dataset folder, validating contents.

    Accepts a folder of preprocessed ``.pt`` tensors or (when
    *allow_audio* is ``True``) a folder of raw audio files.  When raw
    audio is detected, sets ``answers["_auto_preprocess_audio_dir"]``.

    Populates ``answers["dataset_dir"]``.

    Args:
        answers: Mutable wizard answers dict.
        allow_audio: Whether to accept raw audio folders.

    Raises:
        GoBack: If the user presses back at the prompt.
    """
    while True:
        answers["dataset_dir"] = ask_path(
            "Dataset folder (audio files OR preprocessed .pt tensors)",
            default=answers.get("dataset_dir"),
            must_exist=True,
            allow_back=True,
        )

        ds = Path(answers["dataset_dir"])
        has_pt = any(ds.glob("*.pt"))
        has_audio = any(
            f.suffix.lower() in _AUDIO_EXTENSIONS
            for f in ds.iterdir() if f.is_file()
        )

        if has_pt:
            # Prefer preprocessed tensors even if raw audio also exists
            issue = describe_preprocessed_dataset_issue(answers["dataset_dir"])
            if issue is None:
                return
            show_dataset_issue(issue)
            continue

        if has_audio and allow_audio:
            answers["_auto_preprocess_audio_dir"] = answers["dataset_dir"]
            return

        if has_audio and not allow_audio:
            print_message(
                "This folder contains raw audio. Please preprocess first.",
                kind="error",
            )
            continue

        print_message(
            "No audio files or .pt tensors found in this folder.",
            kind="error",
        )
        exts = ", ".join(sorted(_AUDIO_EXTENSIONS))
        print_message(f"Supported audio formats: {exts}", kind="dim")
