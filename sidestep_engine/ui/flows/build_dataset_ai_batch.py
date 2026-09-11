"""
Batch processing and provider factory for the AI sidecar wizard.

Extracted from ``build_dataset_ai.py`` to keep module sizes within
policy.  Handles building caption/lyrics callables from wizard answers
and running the per-song enrichment loop.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Callable, Dict, Optional

from sidestep_engine.data.caption_config import (
    DEFAULT_GEMINI_MODEL,
    DEFAULT_OPENAI_MODEL,
)

from sidestep_engine.ui.prompt_helpers import print_message, section

logger = logging.getLogger(__name__)


def _caption_generation_kwargs(values: dict[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for source_key, target_key in (
        ("_caption_temperature", "temperature"),
        ("_caption_max_tokens", "max_tokens"),
        ("_caption_top_p", "top_p"),
        ("_caption_presence_penalty", "presence_penalty"),
        ("_caption_frequency_penalty", "frequency_penalty"),
        ("_caption_repetition_penalty", "repetition_penalty"),
    ):
        value = values.get(source_key)
        if value is not None:
            result[target_key] = value
    return result


def _resolve_or_prompt(
    existing: Optional[str], label: str, settings_key: str,
) -> Optional[str]:
    """Use existing key if set, otherwise prompt and offer to save."""
    from sidestep_engine.ui.prompt_helpers import ask, ask_bool
    if existing:
        print_message(f"{label}: found in settings", kind="ok")
        return existing
    key = ask(label, default="", allow_back=True) or None
    if key and ask_bool("Save to settings?", default=True):
        _save_setting(settings_key, key)
    return key


def _save_setting(key_name: str, value: Any) -> None:
    """Persist a single setting value to disk."""
    from sidestep_engine.settings import load_settings, save_settings
    data = load_settings() or {}
    data[key_name] = value
    save_settings(data)


def _validate_caption_key(provider: str, key: str, **kwargs: Any) -> None:
    """Best-effort validation of a caption API key, with user feedback."""
    if not key:
        return
    print_message("Validating key...", kind="dim")
    try:
        if provider == "gemini":
            from sidestep_engine.data.caption_provider_gemini import validate_key
            ok = validate_key(key)
        elif provider == "openai":
            from sidestep_engine.data.caption_provider_openai import validate_key
            ok = validate_key(key, **kwargs)
        else:
            return
        if ok:
            print_message("Key validated", kind="ok")
        else:
            print_message("Key validation failed — check the key and try again", kind="error")
    except Exception as exc:
        print_message(f"Validation error: {exc}", kind="warn")


def _validate_genius_token(token: str) -> None:
    """Best-effort validation of a Genius API token, with user feedback."""
    if not token:
        return
    print_message("Validating token...", kind="dim")
    try:
        from sidestep_engine.data.lyrics_provider_genius import validate_token
        if validate_token(token):
            print_message("Token validated", kind="ok")
        else:
            print_message("Token validation failed — lyrics will be skipped", kind="warn")
    except Exception as exc:
        print_message(f"Validation error: {exc}", kind="warn")




def build_caption_fn(
    answers: dict,
) -> Optional[Callable[[str, str, str, Path], Optional[str]]]:
    """Build the caption generator callable from wizard answers.

    Args:
        answers: Wizard answer dict with ``caption_provider``,
            ``_caption_key``, and optional OpenAI settings.

    Returns:
        A callable ``(title, artist, lyrics_excerpt, audio_path) -> str|None``,
        or ``None`` if captions are skipped.
    """
    provider = answers.get("caption_provider", "skip")
    key = answers.get("_caption_key")
    generation_kwargs = _caption_generation_kwargs(answers)

    # Local providers don't need an API key
    if provider in ("local_8-10gb", "local_12gb", "local_16gb"):
        from sidestep_engine.data.caption_provider_local import generate_caption as _local_cap
        tier = {"local_8-10gb": "8-10gb", "local_12gb": "12gb"}.get(provider, "16gb")
        return lambda title, artist, excerpt, audio_path: _local_cap(
            title, artist, audio_path=audio_path, lyrics_excerpt=excerpt,
            tier=tier,
            max_new_tokens=generation_kwargs.get("max_tokens"),
            temperature=generation_kwargs.get("temperature"),
            top_p=generation_kwargs.get("top_p"),
            repetition_penalty=generation_kwargs.get("repetition_penalty"),
        )

    if provider in ("skip", "lyrics_only") or not key:
        return None

    if provider == "gemini":
        from sidestep_engine.data.caption_provider_gemini import generate_caption
        gemini_model = answers.get("_gemini_model", DEFAULT_GEMINI_MODEL)
        return lambda title, artist, excerpt, audio_path: generate_caption(
            title, artist, key, audio_path=audio_path, lyrics_excerpt=excerpt,
            model=gemini_model,
            temperature=generation_kwargs.get("temperature"),
            max_tokens=generation_kwargs.get("max_tokens"),
            top_p=generation_kwargs.get("top_p"),
        )

    if provider == "openai":
        from sidestep_engine.data.caption_provider_openai import generate_caption
        return lambda title, artist, excerpt, audio_path: generate_caption(
            title, artist, key, audio_path=audio_path, lyrics_excerpt=excerpt,
            base_url=answers.get("_openai_base_url"),
            model=answers.get("_openai_model", DEFAULT_OPENAI_MODEL),
            temperature=generation_kwargs.get("temperature"),
            max_tokens=generation_kwargs.get("max_tokens"),
            top_p=generation_kwargs.get("top_p"),
            presence_penalty=generation_kwargs.get("presence_penalty"),
            frequency_penalty=generation_kwargs.get("frequency_penalty"),
        )

    return None


def build_lyrics_fn(
    answers: dict,
) -> Optional[Callable[[str, str], Optional[str]]]:
    """Build the lyrics fetcher callable from wizard answers.

    Args:
        answers: Wizard answer dict with ``_lyrics_provider`` and
            provider-specific keys.

    Returns:
        A callable ``(artist, title) -> str|None``, or ``None`` if
        lyrics fetching is skipped.
    """
    lyrics_provider = answers.get("_lyrics_provider", "genius")

    if lyrics_provider == "genius":
        token = answers.get("_genius_token")
        if not token:
            return None
        from sidestep_engine.data.lyrics_provider_genius import fetch_lyrics
        return lambda artist, title: fetch_lyrics(artist, title, token)

    if lyrics_provider == "transcriber_server":
        server_url = answers.get("_transcriber_server_url", "")
        if not server_url:
            return None
        from sidestep_engine.data.lyrics_provider_server import fetch_lyrics_from_server
        return lambda artist, title: fetch_lyrics_from_server(
            "", server_url=server_url, artist=artist, title=title,
        )

    if lyrics_provider == "music_flamingo":
        server_url = answers.get("_music_flamingo_url", "")
        hf_token = answers.get("_hf_token", "")
        if not server_url:
            return None
        from sidestep_engine.data.lyrics_provider_music_flamingo import fetch_lyrics_from_music_flamingo
        return lambda artist, title: fetch_lyrics_from_music_flamingo(
            "", server_url=server_url, artist=artist, title=title,
            hf_token=hf_token or None,
        )

    return None


def build_metadata_fn(
    answers: dict,
) -> Optional[Callable[[Path], Optional[dict]]]:
    """Build the metadata provider callable from wizard answers.

    Args:
        answers: Wizard answer dict with ``caption_provider`` and
            Music Flamingo settings.

    Returns:
        A callable ``(audio_path) -> dict|None``, or ``None`` if
        metadata generation is skipped.
    """
    provider = answers.get("caption_provider", "skip")
    if provider != "music_flamingo":
        return None
    server_url = answers.get("_music_flamingo_url", "")
    hf_token = answers.get("_hf_token", "")
    if not server_url:
        return None
    from sidestep_engine.data.metadata_provider_music_flamingo import (
        fetch_music_flamingo_metadata,
    )
    return lambda audio_path: fetch_music_flamingo_metadata(
        str(audio_path), server_url=server_url, hf_token=hf_token or None,
    )


def build_audio_analyze_fn(
    answers: dict,
) -> Optional[Callable[[Path], Optional[dict]]]:
    """Build the local audio analysis callable from wizard answers.

    Args:
        answers: Wizard answer dict with ``_run_audio_analysis`` flag
            and optional ``_audio_analyze_device``.

    Returns:
        A callable ``(audio_path) -> dict|None``, or ``None`` if
        audio analysis is skipped.
    """
    if not answers.get("_run_audio_analysis"):
        return None

    device = answers.get("_audio_analyze_device", "auto")

    def _analyze(audio_path: Path) -> Optional[dict]:
        from sidestep_engine.analysis.audio_analysis import analyze_audio
        return analyze_audio(audio_path, device=device)

    return _analyze


def run_batch(answers: dict) -> Dict[str, int]:
    """Process all songs and return summary stats.

    Args:
        answers: Wizard answer dict (must contain ``_audio_files``,
            ``default_artist``, ``policy``, and provider keys).

    Returns:
        Dict with ``written``, ``skipped``, ``failed`` counts.
    """
    from sidestep_engine.data.enrich_song import enrich_one

    section("Processing")
    audio_files = answers.get("_audio_files", [])
    total = len(audio_files)

    caption_fn = build_caption_fn(answers)
    lyrics_fn = build_lyrics_fn(answers)
    metadata_fn = build_metadata_fn(answers)
    audio_analyze_fn = build_audio_analyze_fn(answers)

    stats: Dict[str, int] = {"written": 0, "skipped": 0, "failed": 0}
    for i, af in enumerate(audio_files, 1):
        print_message(f"[{i}/{total}] {af.name}", kind="banner")
        result = enrich_one(
            af,
            default_artist=answers.get("default_artist", ""),
            caption_fn=caption_fn,
            lyrics_fn=lyrics_fn,
            metadata_fn=metadata_fn,
            audio_analyze_fn=audio_analyze_fn,
            policy=answers.get("policy", "fill_missing"),
        )
        status = result["status"]
        stats[status] = stats.get(status, 0) + 1
        if status == "failed":
            print_message(f"  Error: {result.get('error', 'unknown')}", kind="error")
        else:
            print_message(f"  → {status}", kind="ok" if status == "written" else "dim")
        for warn in result.get("warnings", []):
            print_message(f"  ⚠ {warn}", kind="warn")

    return stats
