"""
Wizard flow for the folder-based dataset builder.

Walks the user through building a ``dataset.json`` from a folder of audio
files with sidecar metadata, then offers to chain into preprocessing.
"""

from __future__ import annotations

from acestep.training_v2.ui import console, is_rich_active
from acestep.training_v2.ui.prompt_helpers import (
    GoBack,
    _esc,
    ask,
    ask_path,
    ask_output_path,
    menu,
    section,
)


def wizard_build_dataset() -> dict:
    """Interactive wizard for building a dataset JSON from a folder.

    Returns:
        A dict with ``dataset_json`` path and build stats, or raises
        ``GoBack`` if the user backs out.
    """
    section("Build Dataset from Folder")

    _print_explanation()

    # Step 1: Input directory
    input_dir = ask_path(
        "Audio folder to scan (subdirs included)",
        must_exist=True,
        allow_back=True,
    )

    # Step 2: Trigger tag (optional)
    tag = ask(
        "Trigger tag (leave empty for none)",
        default="",
        allow_back=True,
    )

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
    from acestep.training_v2.dataset_builder import build_dataset

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
    if is_rich_active() and console is not None:
        console.print(msg_rich)
    else:
        print(msg_plain)


def _print_success(out_path, stats: dict) -> None:
    """Print build success summary."""
    if is_rich_active() and console is not None:
        console.print(f"\n  [bold green]Dataset built successfully![/]")
        console.print(f"    Total samples:   {stats['total']}")
        console.print(f"    With metadata:   {stats['with_metadata']}")
        console.print(f"    Output:          {_esc(out_path)}")
        console.print(
            f"\n  [dim]You can now preprocess with:[/]\n"
            f"  [bold]python train.py fixed --preprocess "
            f"--dataset-json {_esc(out_path)} ...[/]\n"
            f"  [dim]Or select 'Preprocess audio' from the main menu.[/]\n"
        )
    else:
        print(f"\n  Dataset built successfully!")
        print(f"    Total samples:   {stats['total']}")
        print(f"    With metadata:   {stats['with_metadata']}")
        print(f"    Output:          {out_path}")
        print(
            f"\n  You can now preprocess with:\n"
            f"  python train.py fixed --preprocess "
            f"--dataset-json {out_path} ...\n"
            f"  Or select 'Preprocess audio' from the main menu.\n"
        )


def _print_error(msg: str) -> None:
    """Print an error message."""
    if is_rich_active() and console is not None:
        console.print(f"\n  [red]Error:[/] {_esc(msg)}\n")
    else:
        print(f"\n  Error: {msg}\n")
