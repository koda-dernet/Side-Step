"""AI sidecar wizard — step functions and orchestrator."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from sidestep_engine.data.preprocess_discovery import AUDIO_EXTENSIONS
from sidestep_engine.ui.flows.build_dataset_ai_batch import (
    _resolve_or_prompt, _save_setting,
    _validate_caption_key, _validate_genius_token,
)
from sidestep_engine.ui.prompt_helpers import (
    GoBack, ask, ask_path, menu, print_message, section,
)
# Note: menu still used by _step_provider_keys and _step_overwrite_policy

logger = logging.getLogger(__name__)



def _step_audio_folder(a: dict) -> None:
    """Step 1: select audio folder and show scan summary."""
    section("Audio Folder")
    a["audio_dir"] = ask_path(
        "Audio folder", default=a.get("audio_dir"),
        must_exist=True, allow_back=True,
    )
    audio_dir = Path(a["audio_dir"])
    audio_files = sorted(
        f for f in audio_dir.rglob("*")
        if f.is_file() and f.suffix.lower() in AUDIO_EXTENSIONS
    )
    a["_audio_files"] = audio_files

    txt_count = sum(1 for f in audio_files if f.with_suffix(".txt").is_file())
    print_message(f"\nFound {len(audio_files)} audio files, {txt_count} with existing sidecars.\n")


def _step_provider_keys(a: dict) -> None:
    """Step 3: caption provider, model, API keys."""
    from sidestep_engine.data.caption_config import (
        DEFAULT_GEMINI_MODEL, DEFAULT_OPENAI_MODEL, GEMINI_MODEL_SUGGESTIONS,
    )
    from sidestep_engine.settings import (
        get_caption_provider, get_gemini_api_key, get_gemini_model,
        get_genius_api_token, get_hf_token,
        get_music_flamingo_url, get_transcriber_server_url,
        get_openai_api_key, get_openai_base_url, get_openai_model,
    )

    section("Caption Provider & API Keys")

    a["caption_provider"] = menu(
        "Caption provider",
        [
            ("gemini", "Gemini"),
            ("openai", "OpenAI / OpenAI-compatible"),
            ("local_8-10gb", "Local 8-10GB (Qwen2.5-Omni 4-bit)"),
            ("local_12gb", "Local 12GB (Qwen2.5-Omni 8-bit)"),
            ("local_16gb", "Local 16GB+ (Qwen2.5-Omni bf16)"),
            ("music_flamingo", "Music Flamingo"),
            ("skip", "Skip AI captions"),
        ],
        default=1 if get_caption_provider() == "gemini" else 2,
        allow_back=True,
    )
    _save_setting("caption_provider", a["caption_provider"])

    # Resolve caption API key + model
    a["_caption_key"] = None
    if a["caption_provider"] in ("local_8-10gb", "local_12gb", "local_16gb"):
        print_message(
            "Local model: Qwen2.5-Omni-7B (~15 GB download on first use).\n"
            "No API key required. Run the installer script to pre-download.",
            kind="dim",
        )
    elif a["caption_provider"] == "music_flamingo":
        a["_music_flamingo_url"] = _resolve_or_prompt(
            get_music_flamingo_url(),
            "Music Flamingo server URL",
            "music_flamingo_url",
        )
        a["_hf_token"] = _resolve_or_prompt(
            get_hf_token(),
            "Hugging Face token (optional)",
            "hf_token",
        )
    elif a["caption_provider"] == "gemini":
        a["_caption_key"] = _resolve_or_prompt(
            get_gemini_api_key(), "Gemini API key", "gemini_api_key",
        )
        default_gm = get_gemini_model() or DEFAULT_GEMINI_MODEL
        print_message(
            "Available models: " + ", ".join(GEMINI_MODEL_SUGGESTIONS),
            kind="dim",
        )
        a["_gemini_model"] = ask(
            "Gemini model", default=default_gm, allow_back=True,
        ) or default_gm
        _save_setting("gemini_model", a["_gemini_model"])
        _validate_caption_key("gemini", a["_caption_key"])
    elif a["caption_provider"] == "openai":
        a["_caption_key"] = _resolve_or_prompt(
            get_openai_api_key(), "OpenAI API key", "openai_api_key",
        )
        default_base = get_openai_base_url() or ""
        base_url = ask(
            "Base URL (empty = api.openai.com)",
            default=default_base, allow_back=True,
        ) or None
        a["_openai_base_url"] = base_url
        if base_url:
            _save_setting("openai_base_url", base_url)
        default_om = get_openai_model() or DEFAULT_OPENAI_MODEL
        a["_openai_model"] = ask(
            "Model name", default=default_om, allow_back=True,
        ) or default_om
        _save_setting("openai_model", a["_openai_model"])
        _validate_caption_key("openai", a["_caption_key"],
                              base_url=a.get("_openai_base_url"),
                              model=a["_openai_model"])

    # Lyrics provider
    a["_lyrics_provider"] = menu(
        "Lyrics provider",
        [
            ("genius", "Genius"),
            ("transcriber_server", "Transcriber Server"),
            ("music_flamingo", "Music Flamingo"),
            ("none", "Skip lyrics"),
        ],
        default=1,
        allow_back=True,
    )
    if a["_lyrics_provider"] == "genius":
        a["_genius_token"] = _resolve_or_prompt(
            get_genius_api_token(),
            "Genius API token (leave empty to skip lyrics)",
            "genius_api_token",
        )
        _validate_genius_token(a.get("_genius_token", ""))
    elif a["_lyrics_provider"] == "transcriber_server":
        a["_transcriber_server_url"] = _resolve_or_prompt(
            get_transcriber_server_url(),
            "Transcriber Server URL",
            "transcriber_server_url",
        )
    elif a["_lyrics_provider"] == "music_flamingo":
        if not a.get("_music_flamingo_url"):
            a["_music_flamingo_url"] = _resolve_or_prompt(
                get_music_flamingo_url(),
                "Music Flamingo server URL",
                "music_flamingo_url",
            )
        if not a.get("_hf_token"):
            a["_hf_token"] = _resolve_or_prompt(
                get_hf_token(),
                "Hugging Face token (optional)",
                "hf_token",
            )


def _step_artist_resolution(a: dict) -> None:
    """Step 4: default artist for lyrics lookups."""
    if a.get("_lyrics_provider", "none") == "none":
        return
    section("Artist Resolution")
    a["default_artist"] = ask(
        "Default artist (empty = per-song detection)",
        default=a.get("default_artist", ""), allow_back=True,
    )


def _step_audio_analysis(a: dict) -> None:
    """Step: offer local offline audio analysis (BPM, key, signature)."""
    from sidestep_engine.ui.prompt_helpers import ask_bool

    section("Local Audio Analysis")
    print_message(
        "Side-Step can run local offline analysis to extract BPM, musical key,\n"
        "and time signature from your audio files using Demucs + BeatNet + librosa.\n"
        "This requires no API keys but uses GPU VRAM for stem separation.",
        kind="dim",
    )
    a["_run_audio_analysis"] = ask_bool(
        "Run local audio analysis (BPM / key / signature)?",
        default=True,
    )
    if a["_run_audio_analysis"]:
        a["_audio_analyze_device"] = ask(
            "Device for Demucs stem separation",
            default="auto",
            allow_back=True,
        ) or "auto"


def _step_overwrite_policy(a: dict) -> None:
    """Step 5: overwrite policy for existing sidecars."""
    section("Overwrite Policy")
    a["policy"] = menu("How to handle existing sidecar data?", [
        ("fill_missing", "Fill missing only (default)"),
        ("overwrite_caption", "Overwrite captions only"),
        ("overwrite_all", "Overwrite all generated fields"),
    ], default=1, allow_back=True)


def wizard_build_dataset_ai() -> Optional[str]:
    """Run the AI sidecar authoring wizard.

    Returns:
        The audio directory path (for chaining to preprocess), or
        ``None`` if cancelled.
    """
    from sidestep_engine.ui.flows.build_dataset_ai_batch import run_batch

    a: dict = {}
    steps = [
        _step_audio_folder,
        _step_audio_analysis,
        _step_provider_keys,
        _step_artist_resolution,
        _step_overwrite_policy,
    ]

    i = 0
    while i < len(steps):
        try:
            steps[i](a)
            i += 1
        except GoBack:
            if i == 0:
                return None
            i -= 1

    stats = run_batch(a)

    section("Summary")
    total = sum(stats.values())
    print_message(f"{total} songs scanned")
    print_message(f"{stats['written']} sidecars written/updated", kind="ok")
    print_message(f"{stats['skipped']} skipped", kind="dim")
    if stats["failed"]:
        print_message(f"{stats['failed']} failed (see log)", kind="error")

    return a.get("audio_dir")
