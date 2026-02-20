"""
Wizard flow for gradient sensitivity estimation.

Uses a step-list pattern for go-back navigation.
"""

from __future__ import annotations

import argparse
from typing import Any, Callable

from acestep.training_v2.ui import console, is_rich_active
from acestep.training_v2.ui.prompt_helpers import (
    DEFAULT_NUM_WORKERS,
    GoBack,
    ask,
    ask_path,
    ask_output_path,
    native_path,
    section,
    step_indicator,
)


# ---- Steps ------------------------------------------------------------------

def _step_model(a: dict) -> None:
    """Checkpoint, variant, and dataset.

    Args:
        a: Mutated answers dict.  Sets ``checkpoint_dir``,
            ``model_variant``, and ``dataset_dir``.

    Raises:
        GoBack: If the user presses back.
    """
    section("Gradient Sensitivity Estimation")
    if is_rich_active() and console is not None:
        console.print(
            "  [dim]Estimates which LoRA layers learn fastest for your dataset.\n"
            "  Results are saved as JSON and can be used to guide rank selection.[/]\n"
        )
    else:
        print(
            "  Estimates which LoRA layers learn fastest for your dataset.\n"
            "  Results are saved as JSON and can be used to guide rank selection.\n"
        )

    from acestep.training_v2.settings import get_checkpoint_dir
    from acestep.training_v2.model_discovery import pick_model
    from acestep.training_v2.ui.flows_common import (
        describe_preprocessed_dataset_issue,
        show_dataset_issue,
        show_model_picker_fallback_hint,
    )

    ckpt_default = a.get("checkpoint_dir") or get_checkpoint_dir() or native_path("./checkpoints")
    a["checkpoint_dir"] = ask_path(
        "Checkpoint directory", default=ckpt_default,
        must_exist=True, allow_back=True,
    )

    result = pick_model(a["checkpoint_dir"])
    if result is None:
        show_model_picker_fallback_hint()
        a["model_variant"] = ask(
            "Model variant or folder name", default=a.get("model_variant", "base"),
            allow_back=True,
        )
    else:
        a["model_variant"] = result[0]

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


def _step_params(a: dict) -> None:
    """Estimation parameters.

    Args:
        a: Mutated answers dict.  Sets ``estimate_batches``,
            ``top_k``, and ``granularity``.

    Raises:
        GoBack: If the user presses back.
    """
    section("Estimation Parameters (press Enter for defaults)")
    a["estimate_batches"] = ask("Number of estimation batches", default=a.get("estimate_batches", 5), type_fn=int, allow_back=True)
    a["top_k"] = ask("Top-K layers to highlight", default=a.get("top_k", 16), type_fn=int, allow_back=True)
    a["granularity"] = ask("Granularity", default=a.get("granularity", "module"), choices=["module", "layer"], allow_back=True)


def _step_lora(a: dict) -> None:
    """LoRA settings used during estimation.

    Args:
        a: Mutated answers dict.  Sets ``rank``, ``alpha``,
            and ``dropout``.

    Raises:
        GoBack: If the user presses back.
    """
    section("LoRA Settings (press Enter for defaults)")
    a["rank"] = ask("Rank", default=a.get("rank", 64), type_fn=int, allow_back=True)
    a["alpha"] = ask("Alpha", default=a.get("alpha", 128), type_fn=int, allow_back=True)
    a["dropout"] = ask("Dropout", default=a.get("dropout", 0.1), type_fn=float, allow_back=True)


def _step_output(a: dict) -> None:
    """Output path.

    Args:
        a: Mutated answers dict.  Sets ``estimate_output``.

    Raises:
        GoBack: If the user presses back.
    """
    result = ask_output_path(
        "Output JSON path",
        default=a.get("estimate_output", native_path("./estimate_results.json")),
        required=False,
        allow_back=True,
        for_file=True,
    )
    a["estimate_output"] = result or native_path("./estimate_results.json")


# ---- Step list and runner ---------------------------------------------------

_STEPS: list[tuple[str, Callable[..., Any]]] = [
    ("Model & Dataset", _step_model),
    ("Estimation Parameters", _step_params),
    ("LoRA Settings", _step_lora),
    ("Output", _step_output),
]


def wizard_estimate(preset: dict | None = None) -> argparse.Namespace:
    """Interactive wizard for gradient estimation.

    Args:
        preset: Optional pre-filled defaults (session carry-over).

    Returns:
        A populated ``argparse.Namespace`` for the estimate subcommand.

    Raises:
        GoBack: If the user backs out of the first step.
    """
    from acestep.training_v2.ui.flows_common import offer_load_preset_subset

    answers: dict = dict(preset) if preset else {}
    offer_load_preset_subset(
        answers,
        allowed_fields={"rank", "alpha", "dropout"},
        prompt="Load a preset for LoRA estimation defaults?",
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
            if i == 0:
                raise  # bubble to main menu
            i -= 1

    return argparse.Namespace(
        subcommand="estimate",
        plain=False,
        yes=True,
        _from_wizard=True,
        checkpoint_dir=answers["checkpoint_dir"],
        model_variant=answers["model_variant"],
        device="auto",
        precision="auto",
        dataset_dir=answers["dataset_dir"],
        num_workers=DEFAULT_NUM_WORKERS,
        pin_memory=True,
        prefetch_factor=2 if DEFAULT_NUM_WORKERS > 0 else 0,
        persistent_workers=DEFAULT_NUM_WORKERS > 0,
        learning_rate=1e-4,
        batch_size=1,
        gradient_accumulation=4,
        epochs=1,
        warmup_steps=0,
        weight_decay=0.01,
        max_grad_norm=1.0,
        seed=42,
        rank=answers.get("rank", 64),
        alpha=answers.get("alpha", 128),
        dropout=answers.get("dropout", 0.1),
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
        attention_type="both",
        bias="none",
        output_dir=native_path("./estimate_output"),
        save_every=999,
        resume_from=None,
        log_dir=None,
        log_every=10,
        log_heavy_every=50,
        optimizer_type="adamw",
        scheduler_type="cosine",
        gradient_checkpointing=True,
        offload_encoder=False,
        preprocess=False,
        audio_dir=None,
        dataset_json=None,
        tensor_output=None,
        max_duration=240.0,
        cfg_ratio=0.15,
        estimate_batches=answers.get("estimate_batches", 5),
        top_k=answers.get("top_k", 16),
        granularity=answers.get("granularity", "module"),
        module_config=None,
        auto_estimate=False,
        estimate_output=answers.get("estimate_output", native_path("./estimate_results.json")),
    )
