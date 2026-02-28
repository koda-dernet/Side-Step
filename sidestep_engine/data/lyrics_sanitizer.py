"""
Sanitize lyrics section headers by removing performer/artist mentions.

Genius lyrics often include performer names in section tags, e.g.
``[Verse 1: Flume]`` or ``[Chorus - KUCKA]``.  These are noise for
training and should be stripped down to just the section label.
"""

from __future__ import annotations

import re

# Matches section headers like [Verse], [Chorus 1], [Bridge - Artist],
# [Verse 1: Artist feat. Other], [Pre-Chorus (Artist)], etc.
# Captures the section label (group 1) and discards everything after
# a separator (: - — – ,) or inside parentheses at the end.
_HEADER_RE = re.compile(
    r"\["
    r"([A-Za-z][A-Za-z0-9 -]*?)"      # group 1: section label (may contain hyphens)
    r"(?:"
    r"\s*[:—–,]\s*.+"                   # separator + performer text (not bare -)
    r"|\s+-\s+.+"                       # spaced dash: " - Artist"
    r"|\s*\([^)]*\)"                    # parenthesized performer
    r")?"
    r"\]"
)


def sanitize_headers(lyrics: str) -> str:
    """Remove performer mentions from lyrics section headers.

    Transforms headers like ``[Verse 1: Artist]``, ``[Chorus - KUCKA]``,
    ``[Bridge (feat. Someone)]`` into plain ``[Verse 1]``, ``[Chorus]``,
    ``[Bridge]``.

    Lines that are not section headers are passed through unchanged.

    Args:
        lyrics: Raw lyrics text, possibly with Genius-style headers.

    Returns:
        Cleaned lyrics with performer-free section headers.
    """
    if not lyrics:
        return lyrics

    lines = lyrics.splitlines()
    result: list[str] = []

    for line in lines:
        stripped = line.strip()
        if stripped.startswith("[") and stripped.endswith("]"):
            match = _HEADER_RE.fullmatch(stripped)
            if match:
                label = match.group(1).strip()
                result.append(f"[{label}]")
                continue
        result.append(line)

    return "\n".join(result)
