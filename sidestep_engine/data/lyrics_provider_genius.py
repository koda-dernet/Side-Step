"""
Fetch song lyrics from the Genius API via ``lyricsgenius``.

Wraps the ``lyricsgenius`` library with timeout, retry, and
sanitization integration for the AI dataset builder.
"""

from __future__ import annotations

import logging
import time
from typing import Optional

logger = logging.getLogger(__name__)

# Retry configuration
_MAX_RETRIES = 3
_RETRY_BACKOFF_BASE = 2.0
_REQUEST_TIMEOUT = 15


def fetch_lyrics(
    artist: str,
    title: str,
    api_token: str,
    *,
    max_retries: int = _MAX_RETRIES,
    timeout: int = _REQUEST_TIMEOUT,
) -> Optional[str]:
    """Fetch lyrics for a song from Genius.

    Uses ``lyricsgenius`` to search and retrieve full lyrics.
    Retries with exponential backoff on transient failures.

    Args:
        artist: Artist name for the search query.
        title: Song title for the search query.
        api_token: Genius API access token.
        max_retries: Maximum number of retry attempts.
        timeout: Request timeout in seconds.

    Returns:
        Raw lyrics text, or ``None`` if not found or on error.
    """
    try:
        import lyricsgenius
    except ImportError:
        logger.error(
            "lyricsgenius is not installed. "
            "Install it with: pip install lyricsgenius"
        )
        return None

    genius = lyricsgenius.Genius(
        api_token,
        timeout=timeout,
        verbose=False,
        remove_section_headers=False,
    )

    for attempt in range(max_retries):
        try:
            song = genius.search_song(title, artist)
            if song is None:
                logger.info(
                    "No lyrics found on Genius: %s - %s", artist, title
                )
                return None
            return _clean_genius_text(song.lyrics)

        except Exception as exc:
            wait = _RETRY_BACKOFF_BASE ** attempt
            logger.warning(
                "Genius API error (attempt %d/%d): %s — retrying in %.1fs",
                attempt + 1, max_retries, exc, wait,
            )
            if attempt < max_retries - 1:
                time.sleep(wait)

    logger.error("Genius API failed after %d attempts: %s - %s",
                 max_retries, artist, title)
    return None


def _clean_genius_text(raw: str) -> str:
    """Strip Genius page artifacts from lyrics text.

    Removes the song title header and trailing metadata that
    ``lyricsgenius`` sometimes includes.

    Args:
        raw: Raw lyrics string from lyricsgenius.

    Returns:
        Cleaned lyrics text.
    """
    if not raw:
        return raw

    lines = raw.strip().splitlines()

    # lyricsgenius prepends "<N> Contributors<title> Lyrics" or
    # "<title> Lyrics" as the first line — skip it
    if lines and ("Lyrics" in lines[0] or "Contributors" in lines[0]):
        lines = lines[1:]

    # Strip trailing "Embed" or "<N>Embed" line
    if lines and lines[-1].rstrip().endswith("Embed"):
        lines = lines[:-1]

    return "\n".join(lines).strip()


def validate_token(api_token: str, timeout: int = 10) -> bool:
    """Check whether a Genius API token is valid.

    Makes a lightweight search request to verify authentication.

    Args:
        api_token: Genius API access token to validate.
        timeout: Request timeout in seconds.

    Returns:
        ``True`` if the token is valid, ``False`` otherwise.
    """
    try:
        import lyricsgenius
    except ImportError:
        return False

    try:
        genius = lyricsgenius.Genius(
            api_token, timeout=timeout, verbose=False
        )
        genius.search_song("test", "test")
        return True
    except Exception:
        return False
