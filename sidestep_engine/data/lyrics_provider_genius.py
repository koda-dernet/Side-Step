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

# Silence the third-party ``lyricsgenius`` logger without relying on the
# ``verbose`` constructor kwarg -- it was removed from ``Genius.__init__``
# in recent releases and raises ``TypeError``.
logging.getLogger("lyricsgenius").setLevel(logging.WARNING)

# Retry configuration
_MAX_RETRIES = 3
_RETRY_BACKOFF_BASE = 2.0
_REQUEST_TIMEOUT = 15


def _is_wrapped_encoding_error(exc: Exception) -> bool:
    """Return True if *exc* (or its cause chain) is a Unicode encoding error.

    Libraries like ``lyricsgenius`` and ``requests`` sometimes wrap
    ``UnicodeEncodeError`` inside their own exception types.  This helper
    walks the ``__cause__`` / ``__context__`` chain and also checks the
    stringified message as a fallback.
    """
    cur: BaseException | None = exc
    for _ in range(6):
        if cur is None:
            break
        if isinstance(cur, (UnicodeEncodeError, UnicodeDecodeError)):
            return True
        cur = getattr(cur, "__cause__", None) or getattr(cur, "__context__", None)
    msg = str(exc).lower()
    return "codec can't encode" in msg or "codec can't decode" in msg


def _safe_ascii_query(text: str) -> str:
    """Normalize a search query for safe HTTP transport.

    Applies NFC normalization and transliterates non-ASCII characters
    to their closest ASCII equivalents where possible (e.g. accented
    letters → base letters).  This prevents ``latin-1`` / ``ascii``
    codec errors deep inside the ``requests`` / ``lyricsgenius`` stack.
    """
    import unicodedata
    # NFC normalize, then NFKD decompose to split accents from base chars
    text = unicodedata.normalize("NFC", text)
    decomposed = unicodedata.normalize("NFKD", text)
    # Keep only ASCII characters (strips accents/diacritics)
    ascii_approx = decomposed.encode("ascii", "ignore").decode("ascii")
    return ascii_approx.strip() or text.strip()


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

    # Sanitize queries to avoid encoding errors in the HTTP transport
    artist = _safe_ascii_query(artist)
    title = _safe_ascii_query(title)
    # API tokens are pure ASCII; strip invisible Unicode from copy-paste
    # ("Bearer <token>" header must be latin-1 safe)
    api_token = api_token.encode("ascii", "ignore").decode("ascii").strip()

    genius = lyricsgenius.Genius(
        api_token,
        timeout=timeout,
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

        except (UnicodeEncodeError, UnicodeDecodeError) as exc:
            # Encoding errors are deterministic — retrying won't help.
            # Typically caused by non-ASCII characters in Genius API
            # response data that lyricsgenius passes into HTTP headers.
            logger.warning(
                "Genius encoding error (non-retryable): %s — %s - %s",
                exc, artist, title,
            )
            return None
        except Exception as exc:
            if _is_wrapped_encoding_error(exc):
                logger.warning(
                    "Genius encoding error (non-retryable): %s — %s - %s",
                    exc, artist, title,
                )
                return None
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
        # Strip invisible non-ASCII from copy-paste (tokens are pure ASCII)
        api_token = api_token.encode("ascii", "ignore").decode("ascii").strip()
        genius = lyricsgenius.Genius(api_token, timeout=timeout)
        genius.search_song("test", "test")
        return True
    except Exception:
        return False
