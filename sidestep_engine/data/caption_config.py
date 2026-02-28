"""
Shared caption generation configuration.

Provides the system prompt (user-overridable from file), default
generation parameters, and prompt building utilities used by all
caption providers.  Custom prompts are loaded from
``~/.config/sidestep/caption_prompt.txt`` (or platform equivalent).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

# ── Default generation parameters ────────────────────────────────────
DEFAULT_TEMPERATURE: float = 0.7
DEFAULT_MAX_TOKENS: int = 2048
DEFAULT_GEMINI_MODEL: str = "gemini-2.5-flash"
DEFAULT_OPENAI_MODEL: str = "gpt-4o"

GEMINI_MODEL_SUGGESTIONS: list[str] = [
    "gemini-2.5-flash",
    "gemini-2.5-flash-lite",
    "gemini-2.5-pro",
    "gemini-3-flash-preview",
    "gemini-3-pro-preview",
]

_DEFAULT_SYSTEM_PROMPT = (
    "You are a music metadata assistant. You will receive a song's "
    "title, artist, and optionally a lyrics excerpt. When an audio "
    "file is attached, listen to it carefully.\n\n"
    "Return EXACTLY the following key: value lines (no extra text):\n"
    "  caption: <1-2 sentence description of style, mood, sonic character>\n"
    "  genre: <comma-separated genre tags, e.g. 'hip hop, rap'>\n"
    "  bpm: <estimated BPM as integer, e.g. 120>\n"
    "  key: <musical key, e.g. 'C minor' or 'F# major'>\n"
    "  signature: <time signature, e.g. '4/4'>\n\n"
    "Rules:\n"
    "- Do NOT include the artist name or song title in the caption.\n"
    "- Focus on instrumentation, tempo feel, and emotional tone.\n"
    "- If you cannot determine a field, write 'N/A' as the value.\n"
    "- Output ONLY the five key: value lines, nothing else."
)


def _prompt_override_path() -> Path:
    """Path to the user's custom system prompt file."""
    from sidestep_engine.settings import settings_dir
    return settings_dir() / "caption_prompt.txt"


def get_system_prompt() -> str:
    """Return the system prompt, loading from file override if present.

    Checks ``~/.config/sidestep/caption_prompt.txt`` (or platform
    equivalent).  Falls back to the built-in default.

    Returns:
        System prompt string.
    """
    p = _prompt_override_path()
    if p.is_file():
        try:
            text = p.read_text(encoding="utf-8").strip()
            if text:
                logger.debug("Loaded custom caption prompt from %s", p)
                return text
        except OSError as exc:
            logger.warning("Failed to read custom prompt: %s", exc)
    return _DEFAULT_SYSTEM_PROMPT


def build_user_prompt(
    title: str, artist: str, lyrics_excerpt: str = "",
) -> str:
    """Build the user prompt from song metadata.

    Args:
        title: Song title.
        artist: Artist name.
        lyrics_excerpt: Optional lyrics text (truncated to 500 chars).

    Returns:
        Formatted user prompt string.
    """
    prompt = f"Title: {title}\nArtist: {artist}"
    if lyrics_excerpt:
        prompt += f"\nLyrics excerpt:\n{lyrics_excerpt[:500]}"
    return prompt


# Keys we expect from the structured AI response
_STRUCTURED_KEYS = frozenset({"caption", "genre", "bpm", "key", "signature"})


def parse_structured_response(text: str) -> dict[str, str]:
    """Parse a structured ``key: value`` AI response into a dict.

    Falls back to treating the entire text as a caption if no
    recognised keys are found (backward compat with plain-text
    prompts or user-overridden system prompts).

    Args:
        text: Raw response text from the caption provider.

    Returns:
        Dict with sidecar-compatible keys.  Values that are
        ``"N/A"`` or empty are omitted.
    """
    if not text:
        return {}

    result: dict[str, str] = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or ":" not in line:
            continue
        key, _, value = line.partition(":")
        key = key.strip().lower()
        value = value.strip()
        if key in _STRUCTURED_KEYS and value and value.lower() != "n/a":
            result[key] = value

    # Fallback: if no structured keys found, treat whole text as caption
    if not result:
        result["caption"] = text.strip()

    return result
