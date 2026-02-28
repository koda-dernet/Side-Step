"""Tests for per-song enrichment pipeline."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from sidestep_engine.data.enrich_song import enrich_one, parse_filename


class TestParseFilename(unittest.TestCase):
    """Test artist/title extraction from filenames."""

    def test_artist_dash_title(self):
        """Standard 'Artist - Title' pattern."""
        artist, title = parse_filename(Path("Flume - Say It.mp3"))
        self.assertEqual(artist, "Flume")
        self.assertEqual(title, "Say It")

    def test_title_only(self):
        """No dash → empty artist, stem as title."""
        artist, title = parse_filename(Path("Some Track.wav"))
        self.assertEqual(artist, "")
        self.assertEqual(title, "Some Track")


class TestEnrichOne(unittest.TestCase):
    """Test the enrichment pipeline with mocked providers."""

    def _make_audio(self, tmpdir: str, name: str = "Artist - Song.mp3") -> Path:
        """Create a dummy audio file."""
        p = Path(tmpdir) / name
        p.write_bytes(b"\x00" * 10)
        return p

    def test_caption_written(self):
        """Caption from caption_fn is written to sidecar."""
        with tempfile.TemporaryDirectory() as d:
            af = self._make_audio(d)
            result = enrich_one(
                af,
                caption_fn=lambda t, a, l, p: "A great track",
            )
            self.assertEqual(result["status"], "written")
            sc = af.with_suffix(".txt")
            self.assertTrue(sc.exists())
            content = sc.read_text(encoding="utf-8")
            self.assertIn("A great track", content)

    def test_lyrics_fn_called_without_artist(self):
        """F9: Lyrics fn is called even when no artist detected."""
        with tempfile.TemporaryDirectory() as d:
            af = self._make_audio(d, name="SomeTrack.mp3")
            lyrics_fn = MagicMock(return_value="Hello world lyrics")
            result = enrich_one(
                af,
                lyrics_fn=lyrics_fn,
                caption_fn=lambda t, a, l, p: "cap",
            )
            # lyrics_fn should be called with empty artist
            lyrics_fn.assert_called_once()
            args = lyrics_fn.call_args[0]
            self.assertEqual(args[0], "")  # artist
            self.assertEqual(args[1], "SomeTrack")  # title

    def test_warnings_on_no_lyrics(self):
        """F9: Warnings included when lyrics not found."""
        with tempfile.TemporaryDirectory() as d:
            af = self._make_audio(d)
            result = enrich_one(
                af,
                lyrics_fn=lambda a, t: None,
                caption_fn=lambda t, a, l, p: "cap",
            )
            self.assertIn("warnings", result)
            self.assertTrue(any("Not found" in w for w in result["warnings"]))

    def test_warnings_on_empty_caption(self):
        """F9: Warning when caption_fn returns None."""
        with tempfile.TemporaryDirectory() as d:
            af = self._make_audio(d)
            result = enrich_one(
                af,
                caption_fn=lambda t, a, l, p: None,
            )
            self.assertIn("warnings", result)
            self.assertTrue(any("Caption returned empty" in w for w in result["warnings"]))

    def test_extra_keys_preserved_on_rerun(self):
        """F6: Re-running enrichment doesn't drop user-added keys."""
        with tempfile.TemporaryDirectory() as d:
            af = self._make_audio(d)
            sc = af.with_suffix(".txt")
            # Manually write a sidecar with extra keys
            sc.write_text(
                "caption: old\ngenre: rock\nbpm: 120\nkey: \nsignature: \n"
                "repeat: 3\nlyrics:\n",
                encoding="utf-8",
            )
            result = enrich_one(
                af,
                caption_fn=lambda t, a, l, p: "new caption",
                policy="overwrite_caption",
            )
            self.assertEqual(result["status"], "written")
            content = sc.read_text(encoding="utf-8")
            self.assertIn("new caption", content)
            self.assertIn("repeat: 3", content)

    def test_skipped_when_nothing_new(self):
        """Returns 'skipped' when no new fields generated."""
        with tempfile.TemporaryDirectory() as d:
            af = self._make_audio(d)
            result = enrich_one(af)
            self.assertEqual(result["status"], "skipped")

    def test_fill_missing_skips_complete_sidecar(self):
        """Regression: fill_missing skips files with all generated fields populated."""
        with tempfile.TemporaryDirectory() as d:
            af = self._make_audio(d)
            sc = af.with_suffix(".txt")
            sc.write_text(
                "caption: Existing caption\ngenre: rock\nbpm: 120\n"
                "key: C minor\nsignature: 4/4\nlyrics:\nHello world\n",
                encoding="utf-8",
            )
            # caption_fn should NOT be called (early skip)
            caption_fn = MagicMock(return_value="Should not be called")
            result = enrich_one(
                af,
                caption_fn=caption_fn,
                policy="fill_missing",
            )
            self.assertEqual(result["status"], "skipped")
            caption_fn.assert_not_called()

    def test_fill_missing_processes_incomplete_sidecar(self):
        """fill_missing processes files missing some generated fields."""
        with tempfile.TemporaryDirectory() as d:
            af = self._make_audio(d)
            sc = af.with_suffix(".txt")
            # Missing genre, bpm, key, signature
            sc.write_text("caption: Existing\nlyrics:\nHello\n", encoding="utf-8")
            result = enrich_one(
                af,
                caption_fn=lambda t, a, l, p: (
                    "caption: Updated\ngenre: pop\nbpm: 120\n"
                    "key: C major\nsignature: 4/4"
                ),
                policy="fill_missing",
            )
            self.assertEqual(result["status"], "written")
            content = sc.read_text(encoding="utf-8")
            # Existing caption preserved (fill_missing)
            self.assertIn("Existing", content)
            # Missing fields filled
            self.assertIn("genre: pop", content)


if __name__ == "__main__":
    unittest.main()
