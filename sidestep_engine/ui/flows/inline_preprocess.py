"""
Inline preprocessing helpers for the training wizard.

When a user points the training wizard at a raw-audio folder (instead of
preprocessed ``.pt`` tensors), these helpers orchestrate:

1. Displaying a per-file sidecar metadata summary.
2. Asking about the trigger tag / activation word.
3. Asking about audio normalization.
4. Showing a duration scan.
5. Running the two-pass preprocessing pipeline.

Sidecar metadata (``.txt`` files) is read directly by the preprocess
pipeline — no intermediate ``dataset.json`` is created.  If a
pre-existing ``dataset.json`` is found in the audio folder, it is
honoured (e.g. from the upstream ACE-Step builder or Side-Step's
"Export dataset JSON" tool).

Display/prompt helpers live in ``inline_preprocess_prompts`` to keep
this orchestrator within the LOC policy.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from sidestep_engine.ui.prompt_helpers import (
    ask_path,
    print_message,
    section,
)

# Re-export prompt helpers so existing callers are not broken
from sidestep_engine.ui.flows.inline_preprocess_prompts import (  # noqa: F401
    show_sidecar_summary,
    ask_trigger_tag,
    ask_normalization,
    show_duration_scan,
    _metadata_flags,
    _format_marks,
)


# ---------------------------------------------------------------------------
# Pre-existing dataset.json detection
# ---------------------------------------------------------------------------

def detect_existing_json(audio_dir: str) -> Optional[str]:
    """Return path to a pre-existing ``dataset.json`` in *audio_dir*, or None."""
    dj_path = Path(audio_dir) / "dataset.json"
    if dj_path.is_file():
        print_message("Found existing dataset.json \u2014 using it for metadata.", kind="ok")
        return str(dj_path)
    return None


# ---------------------------------------------------------------------------
# Shared inline preprocessing runner
# ---------------------------------------------------------------------------

def run_inline_preprocess(
    answers: dict,
    *,
    label: str = "Auto-Preprocessing",
) -> None:
    """Run two-pass preprocessing inline when a wizard detects raw audio.

    Orchestrates dataset building, metadata display, trigger tag,
    normalization, duration scan, and pipeline dispatch.  Updates
    ``answers['dataset_dir']`` to the tensor output on success.

    Args:
        answers: Wizard answers dict (must contain ``dataset_dir``,
            ``checkpoint_dir``, ``model_variant``).
        label: Section header label shown to the user.

    Raises:
        RuntimeError: If preprocessing fails and the user declines to
            continue.
    """
    from sidestep_engine.settings import get_preprocessed_tensors_dir

    audio_dir = answers["dataset_dir"]
    run_name = answers.get("run_name") or Path(audio_dir).name
    default_out = str(Path(get_preprocessed_tensors_dir()) / run_name)

    section(label)
    print_message("Preprocessing raw audio into .pt tensors so the next step can proceed.", kind="dim")

    # Honour a pre-existing dataset.json (e.g. from upstream ACE-Step builder
    # or Side-Step's "Export dataset JSON" tool).  Otherwise the preprocess
    # pipeline reads .txt sidecars directly — no intermediate JSON needed.
    dataset_json = detect_existing_json(audio_dir)
    show_sidecar_summary(audio_dir)

    # Trigger tag / activation word
    custom_tag, tag_position = ask_trigger_tag(audio_dir)

    normalize, target_db, target_lufs = ask_normalization()
    show_duration_scan(audio_dir, dataset_json)

    print_message(f"Tensors will be saved to: {default_out}", kind="dim")
    tensor_output = ask_path(
        "Output directory for preprocessed tensors",
        default=default_out,
        must_exist=False,
        allow_back=False,
    )

    try:
        from sidestep_engine.data.preprocess import preprocess_audio_files

        def _progress(current: int, total: int, message: str) -> None:
            print_message(f"{message} ({current}/{total})", kind="dim")

        print_message("\nStarting two-pass preprocessing ...", kind="info")
        result = preprocess_audio_files(
            audio_dir=audio_dir if not dataset_json else None,
            output_dir=tensor_output,
            checkpoint_dir=answers["checkpoint_dir"],
            variant=answers["model_variant"],
            max_duration=0,
            dataset_json=dataset_json,
            device="auto",
            precision="auto",
            normalize=normalize,
            target_db=target_db,
            target_lufs=target_lufs,
            progress_callback=_progress,
            custom_tag=custom_tag,
            tag_position=tag_position,
        )
        print_message(f"\nPreprocessing complete: {result['processed']}/{result['total']} samples", kind="ok")
        print_message(f"Output: {result['output_dir']}", kind="dim")
        answers["dataset_dir"] = str(result["output_dir"])

    except Exception as exc:
        print_message(f"Preprocessing failed: {exc}", kind="error")
        raise RuntimeError(str(exc)) from exc
