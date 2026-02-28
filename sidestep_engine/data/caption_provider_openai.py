"""
Generate audio captions using the OpenAI-compatible API.

Supports audio file upload for multimodal models (e.g.
``gpt-4o-audio-preview``) via base64 encoding.  Text-only endpoints
work transparently when no audio is provided.  Custom endpoints are
configured via ``base_url``.
"""

from __future__ import annotations

import base64
import logging
import time
from pathlib import Path
from typing import Any, Optional

from sidestep_engine.data.caption_config import (
    DEFAULT_MAX_TOKENS,
    DEFAULT_OPENAI_MODEL,
    DEFAULT_TEMPERATURE,
    build_user_prompt,
    get_system_prompt,
)

logger = logging.getLogger(__name__)

_MAX_RETRIES = 3
_RETRY_BACKOFF_BASE = 2.0
_MAX_AUDIO_SIZE_MB = 20
_AUDIO_FMT_MAP = {"mp3": "mp3", "wav": "wav", "flac": "flac",
                   "ogg": "ogg", "m4a": "m4a"}


def _get_openai() -> Any:
    """Import and return the ``openai`` module.

    Raises:
        ImportError: If the package is not installed.
    """
    import openai
    return openai


def _build_audio_part(audio_path: Path) -> Optional[dict]:
    """Build a base64-encoded audio content part, or ``None`` if too large/unsupported."""
    size_mb = audio_path.stat().st_size / (1024 * 1024)
    if size_mb > _MAX_AUDIO_SIZE_MB:
        logger.warning("Audio too large for OpenAI upload (%.1fMB > %dMB): %s",
                       size_mb, _MAX_AUDIO_SIZE_MB, audio_path.name)
        return None
    fmt = audio_path.suffix.lstrip(".").lower()
    if fmt not in _AUDIO_FMT_MAP:
        logger.warning("Unsupported audio format for OpenAI: .%s", fmt)
        return None
    data = base64.standard_b64encode(audio_path.read_bytes()).decode("utf-8")
    return {"type": "input_audio",
            "input_audio": {"data": data, "format": _AUDIO_FMT_MAP[fmt]}}


def generate_caption(
    title: str,
    artist: str,
    api_key: str,
    *,
    audio_path: Optional[Path] = None,
    lyrics_excerpt: str = "",
    model: str = DEFAULT_OPENAI_MODEL,
    base_url: Optional[str] = None,
    max_retries: int = _MAX_RETRIES,
) -> Optional[str]:
    """Generate a caption using an OpenAI-compatible API.

    Includes audio as multimodal input when *audio_path* meets size/format
    constraints.  Returns caption string or ``None`` on failure.
    """
    try:
        openai = _get_openai()
    except ImportError:
        logger.error("openai is not installed. "
                     "Install with: pip install openai")
        return None

    client_kwargs: dict[str, Any] = {"api_key": api_key}
    if base_url:
        client_kwargs["base_url"] = base_url
    client = openai.OpenAI(**client_kwargs)

    user_prompt = build_user_prompt(title, artist, lyrics_excerpt)

    # Build user content (text-only or multimodal)
    audio_part = None
    if audio_path and audio_path.is_file():
        audio_part = _build_audio_part(audio_path)
    if audio_part:
        user_content: Any = [{"type": "text", "text": user_prompt}, audio_part]
    else:
        user_content = user_prompt

    messages = [
        {"role": "system", "content": get_system_prompt()},
        {"role": "user", "content": user_content},
    ]

    for attempt in range(max_retries):
        try:
            # Try max_completion_tokens first (required by o1/o3 models,
            # works on all modern SDKs).  Fall back to max_tokens for
            # older SDKs or custom endpoints that reject the new param.
            try:
                response = client.chat.completions.create(
                    model=model, messages=messages,
                    max_completion_tokens=DEFAULT_MAX_TOKENS,
                    temperature=DEFAULT_TEMPERATURE,
                )
            except (TypeError, Exception) as _mct_err:
                # TypeError: older openai SDK doesn't know the param
                # BadRequestError: endpoint rejects the param
                if "max_completion_tokens" in str(_mct_err) or isinstance(_mct_err, TypeError):
                    response = client.chat.completions.create(
                        model=model, messages=messages,
                        max_tokens=DEFAULT_MAX_TOKENS,
                        temperature=DEFAULT_TEMPERATURE,
                    )
                else:
                    raise

            # Check finish_reason for truncation
            choice = response.choices[0]
            if getattr(choice, "finish_reason", None) == "length":
                logger.warning(
                    "OpenAI response truncated (finish_reason=length) for: "
                    "%s - %s. Increase DEFAULT_MAX_TOKENS or shorten the prompt.",
                    artist, title,
                )

            # Safe content extraction (content can be None on refusal)
            content = getattr(choice.message, "content", None)
            text = content.strip() if content else ""
            if text:
                return text
            # Check for refusal (OpenAI models can set refusal field)
            refusal = getattr(choice.message, "refusal", None)
            if refusal:
                logger.warning("OpenAI refused request for: %s - %s: %s",
                               artist, title, refusal)
            else:
                logger.warning("OpenAI returned empty for: %s - %s", artist, title)
            return None
        except Exception as exc:
            wait = _RETRY_BACKOFF_BASE ** attempt
            logger.warning("OpenAI error (attempt %d/%d): %s — retrying in %.1fs",
                           attempt + 1, max_retries, exc, wait)
            if attempt < max_retries - 1:
                time.sleep(wait)

    logger.error("OpenAI failed after %d attempts: %s - %s",
                 max_retries, artist, title)
    return None


def validate_key(
    api_key: str, base_url: Optional[str] = None,
    model: str = DEFAULT_OPENAI_MODEL,
) -> bool:
    """Check whether an OpenAI-compatible API key is valid."""
    try:
        openai = _get_openai()
    except ImportError:
        return False
    try:
        kwargs: dict[str, Any] = {"api_key": api_key}
        if base_url:
            kwargs["base_url"] = base_url
        client = openai.OpenAI(**kwargs)
        client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": "test"}],
            max_tokens=1,
        )
        return True
    except Exception:
        return False
