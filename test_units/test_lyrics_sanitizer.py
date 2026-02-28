"""Tests for lyrics section header sanitization."""

from __future__ import annotations

import unittest

from sidestep_engine.data.lyrics_sanitizer import sanitize_headers


class TestSanitizeHeaders(unittest.TestCase):
    """Test sanitize_headers with various Genius-style header patterns."""

    def test_plain_header_unchanged(self):
        """Headers without performer mentions pass through."""
        self.assertEqual(sanitize_headers("[Verse]"), "[Verse]")
        self.assertEqual(sanitize_headers("[Chorus]"), "[Chorus]")
        self.assertEqual(sanitize_headers("[Verse 1]"), "[Verse 1]")

    def test_colon_separator(self):
        """[Verse 1: Artist] → [Verse 1]."""
        self.assertEqual(
            sanitize_headers("[Verse 1: Flume]"), "[Verse 1]"
        )

    def test_dash_separator(self):
        """[Chorus - KUCKA] → [Chorus]."""
        self.assertEqual(
            sanitize_headers("[Chorus - KUCKA]"), "[Chorus]"
        )

    def test_em_dash_separator(self):
        """[Bridge — Artist Name] → [Bridge]."""
        self.assertEqual(
            sanitize_headers("[Bridge — Artist Name]"), "[Bridge]"
        )

    def test_en_dash_separator(self):
        """[Outro – DJ Snake] → [Outro]."""
        self.assertEqual(
            sanitize_headers("[Outro – DJ Snake]"), "[Outro]"
        )

    def test_parenthesized_performer(self):
        """[Pre-Chorus (feat. Someone)] → [Pre-Chorus]."""
        self.assertEqual(
            sanitize_headers("[Pre-Chorus (feat. Someone)]"),
            "[Pre-Chorus]",
        )

    def test_multiline_lyrics_preserved(self):
        """Non-header lines are untouched."""
        lyrics = (
            "[Verse 1: Flume]\n"
            "Floating through the stars tonight\n"
            "Lost in neon light\n"
            "\n"
            "[Chorus - KUCKA]\n"
            "We can make it if we try"
        )
        expected = (
            "[Verse 1]\n"
            "Floating through the stars tonight\n"
            "Lost in neon light\n"
            "\n"
            "[Chorus]\n"
            "We can make it if we try"
        )
        self.assertEqual(sanitize_headers(lyrics), expected)

    def test_empty_string(self):
        """Empty input returns empty output."""
        self.assertEqual(sanitize_headers(""), "")

    def test_none_passthrough(self):
        """None input returns None (falsy passthrough)."""
        self.assertIsNone(sanitize_headers(None))

    def test_instrumental_tag_unchanged(self):
        """[Instrumental] is not a performer tag."""
        self.assertEqual(
            sanitize_headers("[Instrumental]"), "[Instrumental]"
        )

    def test_comma_separated_performer(self):
        """[Hook, Artist] → [Hook]."""
        self.assertEqual(
            sanitize_headers("[Hook, Artist]"), "[Hook]"
        )

    def test_complex_section_number(self):
        """[Verse 2: Artist feat. Other] → [Verse 2]."""
        self.assertEqual(
            sanitize_headers("[Verse 2: Artist feat. Other]"),
            "[Verse 2]",
        )

    def test_non_header_brackets_unchanged(self):
        """Lines with brackets that aren't section headers pass through."""
        line = "I said [something] to you"
        self.assertEqual(sanitize_headers(line), line)


if __name__ == "__main__":
    unittest.main()
