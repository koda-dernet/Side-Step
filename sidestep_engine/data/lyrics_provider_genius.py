"""
Fetch song lyrics from the Genius API via ``lyricsgenius``.

Wraps the ``lyricsgenius`` library with timeout, retry, and
sanitization integration for the AI dataset builder.
"""

from __future__ import annotations

import logging
import re
import time
from typing import Optional

logger = logging.getLogger(__name__)

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


def _clean_search_title(title: str) -> str:
    """Strip parenthetical features and suffixes from a song title.

    Removes common patterns like ``(w asteria)``, ``(feat. X)``,
    ``(Official Audio)``, ``(Remix)`` etc. that confuse Genius search.
    """
    # Remove parenthetical/bracketed content containing feature tags
    # or common noise like "Official Audio", "Lyric Video", etc.
    title = re.sub(
        r"\s*[\(\[]\s*"
        r"(?:w[/\s]|feat\.?\s|ft\.?\s|with\s|prod\.?\s|official|lyric|audio|video|remix|live|acoustic)"
        r"[^)\]]*[\)\]]",
        "", title, flags=re.IGNORECASE,
    )
    # Also strip trailing " - Topic" or " - Remix" style suffixes
    title = re.sub(r"\s*[-–—]\s*(?:official|lyric|audio|video|topic).*$", "", title, flags=re.IGNORECASE)
    return title.strip()


def _clean_search_artist(artist: str) -> str:
    """Extract the primary artist from a multi-artist string.

    Takes just the first artist from comma/slash/ampersand-separated
    lists like ``6arelyhuman, asteria`` → ``6arelyhuman``.
    """
    if not artist:
        return artist
    # Split on common multi-artist separators
    primary = re.split(r"\s*[,/&]\s*|\s+(?:x|X|vs\.?|and|feat\.?|ft\.?)\s+", artist)[0]
    return primary.strip() or artist.strip()


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
            "Install with: uv pip install 'side-step[captions]' "
            "(or: pip install lyricsgenius)"
        )
        return None

    # Clean queries for better Genius search accuracy
    title = _clean_search_title(title)
    artist = _clean_search_artist(artist)
    # Sanitize for safe HTTP transport
    artist = _safe_ascii_query(artist)
    title = _safe_ascii_query(title)
    # API tokens are pure ASCII; strip invisible Unicode from copy-paste
    # ("Bearer <token>" header must be latin-1 safe)
    api_token = api_token.encode("ascii", "ignore").decode("ascii").strip()

    genius = lyricsgenius.Genius(
        api_token,
        timeout=timeout,
        verbose=False,
        remove_section_headers=False,
    )

    # Try the normal order first
    result = _try_search(genius, title, artist, max_retries)
    if result is not None:
        return result

    # Filename parsing can swap artist/title ("Title - Artist" vs
    # "Artist - Title").  Try the swapped order as a fallback.
    swapped_title = _clean_search_title(artist)
    swapped_artist = _clean_search_artist(title)
    swapped_title = _safe_ascii_query(swapped_title)
    swapped_artist = _safe_ascii_query(swapped_artist)
    if (swapped_title != title or swapped_artist != artist) and swapped_title:
        logger.info(
            "Genius: retrying with swapped artist/title: '%s' - '%s'",
            swapped_artist, swapped_title,
        )
        result = _try_search(genius, swapped_title, swapped_artist, max_retries)
        if result is not None:
            return result

    logger.info("Genius: no valid lyrics found for '%s' - '%s'", artist, title)
    return None


def _try_search(genius: object, title: str, artist: str, max_retries: int) -> Optional[str]:
    """Attempt a Genius search with validation.

    Returns cleaned lyrics text, or ``None`` if the result is invalid
    (title mismatch, profile page, not found, or errors).
    """
    for attempt in range(max_retries):
        try:
            song = genius.search_song(title, artist)  # type: ignore[union-attr]
            if song is None:
                logger.info(
                    "No lyrics found on Genius: '%s' - '%s'", artist, title
                )
                return None

            # Validate the result is an actual song, not an artist/profile page
            if not _titles_match(title, getattr(song, "title", "")):
                logger.info(
                    "Genius result title mismatch: searched '%s' but got '%s'",
                    title, getattr(song, "title", "?"),
                )
                return None

            cleaned = _clean_genius_text(song.lyrics)
            if _is_profile_page(cleaned):
                logger.info(
                    "Genius returned an artist/profile page instead of lyrics: '%s' - '%s'",
                    artist, title,
                )
                return None

            return cleaned

        except (UnicodeEncodeError, UnicodeDecodeError) as exc:
            # Encoding errors are deterministic — retrying won't help.
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

    logger.error("Genius API failed after %d attempts: '%s' - '%s'",
                 max_retries, artist, title)
    return None


def _normalize_for_match(text: str) -> str:
    """Lowercase, strip punctuation/whitespace for fuzzy title comparison."""
    text = re.sub(r"[^\w\s]", "", text.lower())
    return " ".join(text.split())


def _titles_match(query_title: str, result_title: str) -> bool:
    """Check if the Genius result title plausibly matches the search query.

    Compares normalized word sets.  When one side is a single word
    (likely an artist name, not a song), requires exact word-set
    equality.  Otherwise requires ≥50% bidirectional overlap to
    tolerate suffixes like "(Official Audio)" or "(Remix)".
    """
    q = _normalize_for_match(query_title)
    r = _normalize_for_match(result_title)
    if not q or not r:
        return False
    if q == r:
        return True
    q_words = set(q.split())
    r_words = set(r.split())
    if not q_words or not r_words:
        return False
    # Single-word on either side → must be an exact word-set match.
    # This prevents artist-name pages ("6arelyhuman") from matching
    # multi-word queries ("6arelyhuman asteria").
    if len(q_words) == 1 or len(r_words) == 1:
        return q_words == r_words
    # Multi-word: bidirectional ≥50% overlap
    common = len(q_words & r_words)
    return (common / len(q_words)) >= 0.5 and (common / len(r_words)) >= 0.5


# Phrases that indicate a Genius artist/profile page, not actual lyrics
_PROFILE_PAGE_MARKERS = [
    "this page is currently a w.i.p",
    "welcome to the",
    "how do you access a song",
    "by clicking the blue dot",
    "missing/incomplete lyrics",
    "only available on soundcloud",
    "tracks & release dates",
    "if a song that's missing here",
    "= lyrics complete",
    "add a song",
    "comment here to add it",
]


def _is_profile_page(text: str) -> bool:
    """Detect if text is a Genius artist/profile page rather than lyrics."""
    if not text:
        return False
    lower = text[:1500].lower()  # check only the start
    hits = sum(1 for marker in _PROFILE_PAGE_MARKERS if marker in lower)
    return hits >= 2


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
        genius = lyricsgenius.Genius(
            api_token, timeout=timeout, verbose=False
        )
        genius.search_song("test", "test")
        return True
    except Exception:
        return False
