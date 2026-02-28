"""Tests for Genius lyrics provider (mocked API calls)."""

from __future__ import annotations

import unittest
from unittest.mock import MagicMock, patch

from sidestep_engine.data.lyrics_provider_genius import (
    _clean_genius_text,
    fetch_lyrics,
)


class TestCleanGeniusText(unittest.TestCase):
    """Test Genius page artifact removal."""

    def test_strips_title_header(self):
        """First line with 'Lyrics' is removed."""
        raw = "Song Title Lyrics\n[Verse]\nHello world"
        self.assertEqual(_clean_genius_text(raw), "[Verse]\nHello world")

    def test_strips_contributors_header(self):
        """First line with 'Contributors' is removed."""
        raw = "3 ContributorsSong Title Lyrics\n[Verse]\nHello"
        self.assertEqual(_clean_genius_text(raw), "[Verse]\nHello")

    def test_strips_trailing_embed(self):
        """Trailing 'Embed' or '<N>Embed' line is removed."""
        raw = "[Verse]\nHello world\n123Embed"
        self.assertEqual(_clean_genius_text(raw), "[Verse]\nHello world")

    def test_clean_text_passthrough(self):
        """Already-clean text passes through."""
        raw = "[Verse]\nHello world"
        self.assertEqual(_clean_genius_text(raw), raw)

    def test_empty_string(self):
        """Empty input returns empty."""
        self.assertEqual(_clean_genius_text(""), "")


class TestFetchLyrics(unittest.TestCase):
    """Test fetch_lyrics with mocked lyricsgenius."""

    @patch("sidestep_engine.data.lyrics_provider_genius.lyricsgenius", create=True)
    def test_success(self, mock_lg_module):
        """Successful fetch returns cleaned lyrics."""
        mock_song = MagicMock()
        mock_song.lyrics = "Song Lyrics\n[Verse]\nHello world\n42Embed"
        mock_genius = MagicMock()
        mock_genius.search_song.return_value = mock_song
        mock_lg_module.Genius.return_value = mock_genius

        with patch.dict("sys.modules", {"lyricsgenius": mock_lg_module}):
            result = fetch_lyrics("Artist", "Song", "fake-token", max_retries=1)

        self.assertEqual(result, "[Verse]\nHello world")

    @patch("sidestep_engine.data.lyrics_provider_genius.lyricsgenius", create=True)
    def test_not_found(self, mock_lg_module):
        """Returns None when no song is found."""
        mock_genius = MagicMock()
        mock_genius.search_song.return_value = None
        mock_lg_module.Genius.return_value = mock_genius

        with patch.dict("sys.modules", {"lyricsgenius": mock_lg_module}):
            result = fetch_lyrics("Artist", "Unknown", "fake-token", max_retries=1)

        self.assertIsNone(result)

    @patch("sidestep_engine.data.lyrics_provider_genius.lyricsgenius", create=True)
    @patch("sidestep_engine.data.lyrics_provider_genius.time.sleep")
    def test_retry_on_error(self, mock_sleep, mock_lg_module):
        """Retries on transient errors with backoff."""
        mock_genius = MagicMock()
        mock_genius.search_song.side_effect = [
            ConnectionError("timeout"),
            ConnectionError("timeout"),
        ]
        mock_lg_module.Genius.return_value = mock_genius

        with patch.dict("sys.modules", {"lyricsgenius": mock_lg_module}):
            result = fetch_lyrics("Artist", "Song", "fake-token", max_retries=2)

        self.assertIsNone(result)
        self.assertEqual(mock_sleep.call_count, 1)


if __name__ == "__main__":
    unittest.main()
