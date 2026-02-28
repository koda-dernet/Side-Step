"""
Prompts and display helpers for inline preprocessing.

Extracted from ``inline_preprocess.py`` to keep both modules within the
LOC policy.  Contains:

- Per-file sidecar metadata summary display.
- Trigger tag / activation word prompt.
- Audio normalization prompt.
- Duration scan display.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from sidestep_engine.ui import console, is_rich_active
from sidestep_engine.ui.prompt_helpers import (
    _esc,
    ask,
    ask_bool,
    menu,
    print_message,
    print_rich,
)


# ---------------------------------------------------------------------------
# Per-file sidecar metadata summary
# ---------------------------------------------------------------------------

def show_sidecar_summary(audio_dir: str) -> None:
    """Print a per-file metadata table by reading ``.txt`` sidecars directly.

    Shows ✓/– indicators for caption, lyrics, BPM, and genre for each
    sample so the user can verify their sidecar data was picked up.

    Args:
        audio_dir: Path to the folder containing audio + sidecar files.
    """
    from sidestep_engine.data.preprocess_discovery import AUDIO_EXTENSIONS
    from sidestep_engine.data.dataset_builder import load_sidecar_metadata
    from sidestep_engine.data.sidecar_metadata import normalize_sidecar

    audio_files = sorted(
        f for f in Path(audio_dir).rglob("*")
        if f.is_file() and f.suffix.lower() in AUDIO_EXTENSIONS
    )
    if not audio_files:
        return

    use_rich = is_rich_active() and console is not None
    print_message("\nPer-file metadata:", kind="banner")
    for af in audio_files:
        raw = load_sidecar_metadata(af)
        sample = normalize_sidecar(raw, af) if raw else {}
        flags = _metadata_flags(sample)
        marks = _format_marks(flags, use_rich)
        if use_rich:
            print_rich(f"  {_esc(af.name):<40s}  {marks}")
        else:
            print_message(f"  {af.name:<40s}  {marks}")


def _metadata_flags(sample: dict) -> dict[str, bool]:
    """Extract boolean presence flags from a sample dict."""
    return {
        "caption": bool(sample.get("caption", "").strip()),
        "lyrics": (
            bool(sample.get("lyrics", "").strip())
            and sample.get("lyrics") != "[Instrumental]"
        ),
        "bpm": sample.get("bpm") is not None,
        "genre": bool(sample.get("genre", "").strip()),
        "key": bool(sample.get("keyscale", "").strip()),
        "sig": bool(sample.get("timesignature", "").strip()),
    }


def _format_marks(flags: dict[str, bool], use_rich: bool) -> str:
    """Format metadata presence flags as a ✓/– indicator string."""
    parts: list[str] = []
    for key in ("caption", "lyrics", "bpm", "genre", "key", "sig"):
        ok = flags.get(key, False)
        if use_rich:
            mark = "[green]✓[/]" if ok else ("[red]–[/]" if key == "caption" else "[dim]–[/]")
        else:
            mark = "✓" if ok else "–"
        parts.append(f"{mark} {key}")
    return "  ".join(parts)


# ---------------------------------------------------------------------------
# Trigger tag / activation word
# ---------------------------------------------------------------------------

def ask_trigger_tag(audio_dir: str) -> tuple[str, str]:
    """Prompt for a trigger tag (activation word) to prepend/append to captions.

    Scans existing sidecars for ``custom_tag`` values and pre-fills the
    default.  Writes the chosen tag back to sidecars that don't have one.

    Returns:
        ``(custom_tag, tag_position)`` — tag may be empty (no tag).
    """
    from sidestep_engine.ui.flows.build_dataset import (
        scan_sidecar_tags,
        writeback_tag_to_sidecars,
    )

    print_message(
        "\nA trigger tag (activation word) lets you invoke your fine-tune at\n"
        "  inference time by including the tag in the prompt.  Leave empty\n"
        "  if you don't need one.",
        kind="dim",
    )

    detected_tags = scan_sidecar_tags(audio_dir)
    tag_default = ""
    if len(detected_tags) == 1:
        tag_default = next(iter(detected_tags))
        print_message(f"  Found trigger tag in sidecars: {tag_default}", kind="ok")
    elif len(detected_tags) > 1:
        print_message(
            f"  Warning: found {len(detected_tags)} different trigger tags "
            f"in sidecars: {', '.join(sorted(detected_tags))}\n"
            f"  Using multiple activation tags may confuse the model during inference.",
            kind="warn",
        )
        tag_default = sorted(detected_tags)[0]

    tag = ask(
        "Trigger tag / activation word (leave empty for none)",
        default=tag_default,
        allow_back=False,
    )

    if tag and tag != tag_default:
        writeback_tag_to_sidecars(audio_dir, tag)

    tag_position = "prepend"
    if tag:
        tag_position = menu(
            "Tag position in the caption",
            [
                ("prepend", "Prepend (tag comes before caption)"),
                ("append", "Append (tag comes after caption)"),
                ("replace", "Replace (tag replaces caption entirely)"),
            ],
            default=1,
            allow_back=False,
        )

    return tag, tag_position


# ---------------------------------------------------------------------------
# Normalization prompt
# ---------------------------------------------------------------------------

def ask_normalization() -> tuple[str, float, float]:
    """Prompt the user for audio normalization settings.

    Returns:
        ``(method, target_db, target_lufs)`` where method is one of
        ``"none"``, ``"peak"``, or ``"lufs"``.
    """
    print_message(
        "\nNormalization ensures consistent loudness across training audio.\n"
        "  Peak normalizes to -1.0 dBFS (matches ACE-Step output).\n"
        "  LUFS normalizes to -14 LUFS (broadcast standard, requires pyloudnorm).\n"
        "  If unsure, 'peak' is a safe default.",
        kind="dim",
    )

    target_db = -1.0
    target_lufs = -14.0

    if not ask_bool("Normalize audio before encoding?", default=True, allow_back=False):
        return "none", target_db, target_lufs

    method = menu(
        "Normalization method",
        [
            ("peak", "Peak (-1.0 dBFS, no extra deps, matches ACE-Step)"),
            ("lufs", "LUFS (-14 LUFS, perceptually uniform, needs pyloudnorm)"),
        ],
        default=1,
        allow_back=False,
    )

    if method == "lufs":
        method = _ensure_lufs_deps(method)

    if method == "peak":
        target_db = float(ask(
            "Peak target (dBFS)", default=-1.0, type_fn=float, allow_back=False,
        ))
    elif method == "lufs":
        target_lufs = float(ask(
            "LUFS target", default=-14.0, type_fn=float, allow_back=False,
        ))

    return method, target_db, target_lufs


def _ensure_lufs_deps(method: str) -> str:
    """Check LUFS dependencies; offer fallback to peak if missing."""
    from sidestep_engine.ui.dependency_check import (
        ensure_optional_dependencies,
        required_preprocess_optionals,
    )

    unresolved = ensure_optional_dependencies(
        required_preprocess_optionals("lufs"),
        interactive=True,
        allow_install_prompt=True,
    )
    if unresolved and ask_bool(
        "LUFS dependency still missing. Switch to peak?",
        default=True, allow_back=False,
    ):
        return "peak"
    return method


# ---------------------------------------------------------------------------
# Duration scan
# ---------------------------------------------------------------------------

def show_duration_scan(audio_dir: str, dataset_json: Optional[str]) -> None:
    """Display per-file durations and longest clip.

    Args:
        audio_dir: Path to the audio folder.
        dataset_json: Optional path to dataset JSON for file discovery.
    """
    from sidestep_engine.data.audio_duration import get_audio_duration
    from sidestep_engine.data.preprocess_discovery import discover_audio_files

    try:
        audio_files = discover_audio_files(
            audio_dir if not dataset_json else None,
            dataset_json,
        )
    except (FileNotFoundError, OSError):
        return

    if not audio_files:
        return

    durations: dict[str, int] = {}
    for af in audio_files:
        durations[af.name] = get_audio_duration(str(af))

    print_message(f"\nDuration scan ({len(audio_files)} files):", kind="banner")
    for name, dur in sorted(durations.items()):
        m, s = divmod(dur, 60)
        print_message(f"  {name:<40s}  {int(m)}m {s:02d}s  ({dur}s)")
    longest = max(durations.values()) if durations else 0
    print_message(f"\nLongest clip: {longest}s", kind="ok")
