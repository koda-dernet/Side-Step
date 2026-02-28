"""
Wizard flow for the folder-based dataset builder.

Walks the user through building a ``dataset.json`` from a folder of audio
files with sidecar metadata, then offers to chain into preprocessing.
"""

from __future__ import annotations

from sidestep_engine.ui.prompt_helpers import (
    GoBack,
    _esc,
    ask,
    ask_path,
    ask_output_path,
    menu,
    print_message,
    print_rich,
    section,
)


def wizard_build_dataset() -> dict:
    """Interactive wizard for building dataset metadata from a folder.

    Offers two modes:
        1. **AI-enriched sidecars** — generates captions and fetches
           lyrics per song using LLM + Genius APIs.
        2. **Legacy JSON builder** — builds a ``dataset.json`` from
           existing sidecar files (no API calls).

    Returns:
        A dict with results, or raises ``GoBack`` if the user backs out.
    """
    section("Build Dataset")

    mode = menu(
        "Choose a mode",
        [
            ("ai", "AI-enriched sidecars (generate captions + fetch lyrics)"),
            ("legacy", "Export ACE-Step JSON (build dataset.json from existing files)"),
        ],
        default=1,
        allow_back=True,
    )

    if mode == "ai":
        from sidestep_engine.ui.flows.build_dataset_ai import wizard_build_dataset_ai
        audio_dir = wizard_build_dataset_ai()
        return {"audio_dir": audio_dir, "mode": "ai"} if audio_dir else {}

    print_message(
        "\nBuilds an ACE-Step-compatible dataset.json from your audio folder.\n"
        "  You only need this if you plan to use upstream ACE-Step tools,\n"
        "  share your dataset with others, or want a portable metadata snapshot.\n"
        "  Side-Step reads .txt sidecars directly during training \u2014 no JSON required.",
        kind="dim",
    )
    _print_explanation()

    # Step 1: Input directory
    input_dir = ask_path(
        "Audio folder to scan (subdirs included)",
        must_exist=True,
        allow_back=True,
    )

    # Auto-discovery: check for existing dataset.json
    from pathlib import Path
    existing_dj = Path(input_dir) / "dataset.json"
    if existing_dj.is_file():
        try:
            import json
            data = json.loads(existing_dj.read_text(encoding="utf-8"))
            n_samples = len(data.get("samples", []))
            meta = data.get("metadata", {})
            created = meta.get("created_at", "unknown")
            print_message(
                f"\nFound existing dataset.json ({n_samples} samples, created {created}).",
                kind="ok",
            )

            from sidestep_engine.ui.prompt_helpers import ask_bool
            if ask_bool("Use this existing dataset.json instead of rebuilding?", default=True, allow_back=True):
                return {
                    "dataset_json": str(existing_dj),
                    "input_dir": input_dir,
                    "stats": {"total": n_samples, "skipped": 0, "with_metadata": 0, "reused": True},
                }
        except Exception:
            pass  # ignore parse errors, let user rebuild

    # Step 2: Trigger tag — pre-fill from sidecars if available
    detected_tags = scan_sidecar_tags(input_dir)
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
        "Trigger tag (leave empty for none)",
        default=tag_default,
        allow_back=True,
    )

    # Write-back: persist new tag to all sidecars
    if tag and tag != tag_default:
        writeback_tag_to_sidecars(input_dir, tag)

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
            allow_back=True,
        )

    # Step 3: Genre ratio
    genre_ratio_str = ask(
        "Genre ratio (% of samples using genre as prompt, 0-100)",
        default="0",
        allow_back=True,
    )
    try:
        genre_ratio = max(0, min(100, int(genre_ratio_str or "0")))
    except ValueError:
        genre_ratio = 0

    # Step 4: Dataset name
    name = ask(
        "Dataset name (used in the JSON metadata block)",
        default="local_dataset",
        allow_back=True,
    )

    # Step 5: Output path
    output = ask_output_path(
        "Output JSON path (leave empty for <folder>/dataset.json)",
        default="",
        required=False,
        allow_back=True,
        for_file=True,
    )
    if not output or output.strip() == "":
        output = None

    # Build it
    from sidestep_engine.data.dataset_builder import build_dataset

    try:
        out_path, stats = build_dataset(
            input_dir=input_dir,
            tag=tag,
            tag_position=tag_position,
            name=name,
            output=output,
            genre_ratio=genre_ratio,
        )
    except FileNotFoundError as exc:
        _print_error(str(exc))
        raise GoBack()
    except Exception as exc:
        _print_error(f"Build failed: {exc}")
        raise GoBack()

    # Summary
    _print_success(out_path, stats)

    return {
        "dataset_json": str(out_path),
        "total": stats["total"],
        "with_metadata": stats["with_metadata"],
    }


# ---- Helpers ----------------------------------------------------------------

def _print_explanation() -> None:
    """Explain how the dataset builder works with concrete examples."""
    msg_rich = (
        "\n  [bold]How it works:[/]\n"
        "  Point this at a folder containing your audio files (.wav, .mp3, .flac, etc.)\n"
        "  and matching text files with metadata. A dataset.json will be generated\n"
        "  that you can feed directly into the preprocessing step.\n\n"
        "  [bold]How to organise your files:[/]\n"
        "  Each audio file can have a matching .txt file [bold]with the same name[/].\n"
        "  For example, if your song is [cyan]MyTrack.wav[/], the metadata goes in [cyan]MyTrack.txt[/].\n\n"
        "  [bold]Option A[/] -- Single .txt per song (recommended):\n\n"
        "    [dim]my_songs/[/]\n"
        "      [cyan]MyTrack.wav[/]\n"
        "      [cyan]MyTrack.txt[/]          [dim]<-- key: value pairs[/]\n"
        "      [cyan]AnotherSong.mp3[/]\n"
        "      [cyan]AnotherSong.txt[/]\n\n"
        "    Inside the .txt:\n"
        "      [green]caption: dreamy ambient synth pad, reverb-heavy[/]\n"
        "      [green]genre: ambient, electronic[/]\n"
        "      [green]bpm: 90[/]\n"
        "      [green]key: C minor[/]\n"
        "      [green]lyrics:[/]\n"
        "      [green]\\[Verse][/]\n"
        "      [green]Floating through the stars tonight ...[/]\n\n"
        "  [bold]Option B[/] -- Separate caption + lyrics files (ACE-Step upstream):\n\n"
        "    [dim]my_songs/[/]\n"
        "      [cyan]MyTrack.wav[/]\n"
        "      [cyan]MyTrack.caption.txt[/]   [dim]<-- one line: the caption[/]\n"
        "      [cyan]MyTrack.lyrics.txt[/]    [dim]<-- full lyrics[/]\n\n"
        "  [bold]Option C[/] -- Audio only (no text files):\n"
        "    Songs with no matching .txt are included as instrumentals.\n"
        "    The caption is derived from the filename (e.g. \"My Track\").\n"
    )
    msg_plain = (
        "\n  How it works:\n"
        "  Point this at a folder containing your audio files (.wav, .mp3, .flac, etc.)\n"
        "  and matching text files with metadata. A dataset.json will be generated\n"
        "  that you can feed directly into the preprocessing step.\n\n"
        "  How to organise your files:\n"
        "  Each audio file can have a matching .txt file with the same name.\n"
        "  For example, if your song is MyTrack.wav, the metadata goes in MyTrack.txt.\n\n"
        "  Option A -- Single .txt per song (recommended):\n\n"
        "    my_songs/\n"
        "      MyTrack.wav\n"
        "      MyTrack.txt          <-- key: value pairs\n"
        "      AnotherSong.mp3\n"
        "      AnotherSong.txt\n\n"
        "    Inside the .txt:\n"
        "      caption: dreamy ambient synth pad, reverb-heavy\n"
        "      genre: ambient, electronic\n"
        "      bpm: 90\n"
        "      key: C minor\n"
        "      lyrics:\n"
        "      [Verse]\n"
        "      Floating through the stars tonight ...\n\n"
        "  Option B -- Separate caption + lyrics files (ACE-Step upstream):\n\n"
        "    my_songs/\n"
        "      MyTrack.wav\n"
        "      MyTrack.caption.txt   <-- one line: the caption\n"
        "      MyTrack.lyrics.txt    <-- full lyrics\n\n"
        "  Option C -- Audio only (no text files):\n"
        "    Songs with no matching .txt are included as instrumentals.\n"
        "    The caption is derived from the filename (e.g. \"My Track\").\n"
    )
    print_rich(msg_rich)


def _print_success(out_path, stats: dict) -> None:
    """Print build success summary."""
    print_message("\nDataset built successfully!", kind="ok")
    print_message(f"  Total samples:   {stats['total']}")
    print_message(f"  With metadata:   {stats['with_metadata']}")
    print_message(f"  Output:          {out_path}")
    print_rich(
        f"\n[dim]You can now preprocess with:[/]\n"
        f"  [bold]sidestep train --preprocess "
        f"--dataset-json {_esc(out_path)} ...[/]\n"
        f"  [dim]Or select 'Preprocess audio' from the main menu.[/]"
    )


def _print_error(msg: str) -> None:
    """Print an error message."""
    print_message(f"\nError: {msg}", kind="error")


def scan_sidecar_tags(input_dir: str) -> set[str]:
    """Scan ``.txt`` sidecars in *input_dir* for ``custom_tag`` values.

    Returns the set of distinct non-empty tags found.
    """
    from pathlib import Path
    from sidestep_engine.data.dataset_builder import load_sidecar_metadata
    from sidestep_engine.data.preprocess_discovery import AUDIO_EXTENSIONS

    tags: set[str] = set()
    base = Path(input_dir)
    for af in base.rglob("*"):
        if af.is_file() and af.suffix.lower() in AUDIO_EXTENSIONS:
            meta = load_sidecar_metadata(af)
            tag = meta.get("custom_tag", "").strip()
            if tag:
                tags.add(tag)
    return tags


def writeback_tag_to_sidecars(input_dir: str, tag: str) -> None:
    """Write *tag* as ``custom_tag`` into every sidecar in *input_dir*.

    Uses ``sidecar_io.merge_fields`` with ``fill_missing`` policy so
    existing per-sample tags are not overwritten — only files without
    a ``custom_tag`` get the new value.
    """
    from pathlib import Path
    from sidestep_engine.data.preprocess_discovery import AUDIO_EXTENSIONS
    from sidestep_engine.data.sidecar_io import (
        read_sidecar,
        merge_fields,
        write_sidecar,
        sidecar_path_for,
    )

    base = Path(input_dir)
    updated = 0
    for af in sorted(base.rglob("*")):
        if not (af.is_file() and af.suffix.lower() in AUDIO_EXTENSIONS):
            continue
        sc = sidecar_path_for(af)
        if not sc.exists():
            continue  # don't create new sidecars just for a tag
        existing = read_sidecar(sc)
        if existing.get("custom_tag", "").strip():
            continue  # already has a tag — don't overwrite
        merged = merge_fields(existing, {"custom_tag": tag}, policy="fill_missing")
        write_sidecar(sc, merged)
        updated += 1
    if updated:
        print_message(f"  Wrote custom_tag to {updated} sidecar(s).", kind="ok")
