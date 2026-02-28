"""
Generate audio captions using Google's Gemini API.

Uploads audio files for multimodal analysis via ``client.files.upload``
and generates natural-language captions.  Falls back to text-only
when no audio is provided or upload fails.

Uses the ``google-genai`` SDK (``google.genai``).
"""

from __future__ import annotations

import logging
import re
import time
from pathlib import Path
from typing import Any, Optional

from sidestep_engine.data.caption_config import (
    DEFAULT_GEMINI_MODEL,
    DEFAULT_MAX_TOKENS,
    DEFAULT_TEMPERATURE,
    build_user_prompt,
    get_system_prompt,
)

logger = logging.getLogger(__name__)

_MAX_RETRIES = 3
_RETRY_BACKOFF_BASE = 2.0
_UPLOAD_TIMEOUT_S = 120

_QUOTA_RE = re.compile(r"Quota exceeded", re.IGNORECASE)
_RATE_LIMIT_RE = re.compile(r"(rate.?limit|resource.?exhausted|429)", re.IGNORECASE)


def _make_client(api_key: str) -> Any:
    """Create and return a ``google.genai.Client``.

    Raises:
        ImportError: If ``google-genai`` is not installed.
    """
    from google import genai
    return genai.Client(api_key=api_key)


def _simplify_error(exc: Exception) -> str:
    """Condense a verbose Gemini API error into a short user-facing string.

    Detects quota / rate-limit patterns and returns a one-liner.
    Falls back to the first 120 characters for unknown errors.
    """
    raw = str(exc)
    if _QUOTA_RE.search(raw):
        return (
            "Gemini quota exceeded (free-tier limit reached). "
            "See https://ai.google.dev/gemini-api/docs/rate-limits"
        )
    if _RATE_LIMIT_RE.search(raw):
        return (
            "Gemini rate limit hit — wait a moment and retry. "
            "See https://ai.google.dev/gemini-api/docs/rate-limits"
        )
    if len(raw) > 120:
        return raw[:120] + "…"
    return raw


def _upload_audio(client: Any, audio_path: Path) -> Any:
    """Upload an audio file to Gemini and poll until processed.

    Raises ``RuntimeError`` on timeout (``_UPLOAD_TIMEOUT_S``) or failure.
    """
    logger.debug("Uploading audio: %s", audio_path.name)
    uploaded = client.files.upload(file=str(audio_path))
    deadline = time.monotonic() + _UPLOAD_TIMEOUT_S
    while getattr(uploaded, "state", None) and uploaded.state.name == "PROCESSING":
        if time.monotonic() > deadline:
            raise RuntimeError(
                f"Gemini audio processing timed out after {_UPLOAD_TIMEOUT_S}s: "
                f"{audio_path.name}"
            )
        time.sleep(2)
        uploaded = client.files.get(name=uploaded.name)
    state_name = getattr(getattr(uploaded, "state", None), "name", "")
    if state_name == "FAILED":
        raise RuntimeError(f"Gemini audio processing failed: {audio_path.name}")
    return uploaded


def _delete_uploaded(client: Any, file_obj: Any) -> None:
    """Best-effort cleanup of a Gemini uploaded file."""
    try:
        client.files.delete(name=file_obj.name)
        logger.debug("Deleted uploaded file: %s", file_obj.name)
    except Exception as exc:
        logger.debug("Could not delete uploaded file: %s", exc)


def _safe_extract_text(response: Any) -> str:
    """Extract text from a Gemini response bypassing the broken ``.text`` accessor.

    The SDK raises ``ValueError`` on ``response.text`` when finish_reason is
    MAX_TOKENS or SAFETY with empty parts (google-gemini/issues/280).
    """
    try:
        candidates = response.candidates
        if not candidates:
            return ""
        parts = candidates[0].content.parts
        if not parts:
            return ""
        return "".join(p.text for p in parts if hasattr(p, "text") and p.text)
    except (AttributeError, IndexError, TypeError):
        # Last resort: try the .text accessor (works in simple cases)
        try:
            return response.text or ""
        except (ValueError, AttributeError):
            return ""


def _check_finish_reason(response: Any, title: str, artist: str) -> None:
    """Log a warning if the Gemini response was truncated or filtered."""
    try:
        candidate = response.candidates[0]
        fr = candidate.finish_reason
        # finish_reason enum: 1=STOP, 2=MAX_TOKENS, 3=SAFETY, 4=RECITATION, 5=OTHER
        fr_name = getattr(fr, "name", str(fr))
        if fr_name == "MAX_TOKENS" or fr == 2:
            logger.warning(
                "Gemini response truncated (MAX_TOKENS) for: %s - %s. "
                "Increase DEFAULT_MAX_TOKENS or shorten the prompt.",
                artist, title,
            )
        elif fr_name == "SAFETY" or fr == 3:
            ratings = getattr(candidate, "safety_ratings", [])
            logger.warning(
                "Gemini response filtered (SAFETY) for: %s - %s. "
                "Safety ratings: %s", artist, title, ratings,
            )
        elif fr_name == "RECITATION" or fr == 4:
            logger.warning(
                "Gemini response blocked (RECITATION/copyright) for: %s - %s",
                artist, title,
            )
    except (AttributeError, IndexError):
        pass


def generate_caption(
    title: str,
    artist: str,
    api_key: str,
    *,
    audio_path: Optional[Path] = None,
    lyrics_excerpt: str = "",
    model: str = DEFAULT_GEMINI_MODEL,
    max_retries: int = _MAX_RETRIES,
) -> Optional[str]:
    """Generate a caption for a song using Gemini.

    Uploads audio for multimodal analysis when *audio_path* is given.
    Returns caption string or ``None`` on failure.
    """
    try:
        client = _make_client(api_key)
    except ImportError:
        logger.error("google-genai is not installed. "
                     "Install with: pip install google-genai")
        return None

    user_prompt = build_user_prompt(title, artist, lyrics_excerpt)

    # Upload audio (graceful fallback on failure)
    audio_file = None
    if audio_path and audio_path.is_file():
        try:
            audio_file = _upload_audio(client, audio_path)
        except Exception as exc:
            logger.warning("Audio upload failed, text-only fallback: %s", exc)

    content = [audio_file, user_prompt] if audio_file else [user_prompt]
    config = {
        "system_instruction": get_system_prompt(),
        "temperature": DEFAULT_TEMPERATURE,
        "max_output_tokens": DEFAULT_MAX_TOKENS,
    }

    try:
        for attempt in range(max_retries):
            try:
                response = client.models.generate_content(
                    model=model, contents=content, config=config,
                )
                _check_finish_reason(response, title, artist)
                text = _safe_extract_text(response).strip()
                if text:
                    return text
                logger.warning("Gemini returned empty for: %s - %s", artist, title)
                return None
            except Exception as exc:
                wait = _RETRY_BACKOFF_BASE ** attempt
                short = _simplify_error(exc)
                logger.warning("Gemini error (attempt %d/%d): %s — retrying in %.1fs",
                               attempt + 1, max_retries, short, wait)
                if attempt < max_retries - 1:
                    time.sleep(wait)

        logger.error("Gemini failed after %d attempts: %s - %s",
                     max_retries, artist, title)
        return None
    finally:
        if audio_file is not None:
            _delete_uploaded(client, audio_file)


def validate_key(api_key: str) -> bool:
    """Check whether a Gemini API key is valid via ``client.models.list()``."""
    try:
        client = _make_client(api_key)
    except ImportError:
        return False
    try:
        page = client.models.list(config={"page_size": 1})
        return len(list(page)) > 0
    except Exception:
        return False
