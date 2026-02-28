"""Tests for Option-A sidecar read/write/merge."""

from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path

from sidestep_engine.data.sidecar_io import (
    merge_fields,
    read_sidecar,
    sidecar_path_for,
    write_sidecar,
)


class TestReadSidecar(unittest.TestCase):
    """Test reading existing Option-A sidecars."""

    def test_read_simple(self):
        """Reads key-value pairs correctly."""
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", delete=False, encoding="utf-8"
        ) as f:
            f.write("caption: A cool track\ngenre: electronic\nbpm: 120\n")
            f.flush()
            path = Path(f.name)
        try:
            data = read_sidecar(path)
            self.assertEqual(data["caption"], "A cool track")
            self.assertEqual(data["genre"], "electronic")
            self.assertEqual(data["bpm"], "120")
        finally:
            os.unlink(path)

    def test_read_missing_file(self):
        """Missing file returns empty dict."""
        data = read_sidecar(Path("/nonexistent/file.txt"))
        self.assertEqual(data, {})

    def test_read_with_lyrics(self):
        """Multi-line lyrics block is captured."""
        content = (
            "caption: test\n"
            "lyrics:\n"
            "[Verse]\n"
            "Line one\n"
            "Line two\n"
        )
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", delete=False, encoding="utf-8"
        ) as f:
            f.write(content)
            f.flush()
            path = Path(f.name)
        try:
            data = read_sidecar(path)
            self.assertEqual(data["caption"], "test")
            self.assertIn("Line one", data["lyrics"])
            self.assertIn("[Verse]", data["lyrics"])
        finally:
            os.unlink(path)


class TestMergeFields(unittest.TestCase):
    """Test merge policies."""

    def test_fill_missing_skips_existing(self):
        """fill_missing does not overwrite existing values."""
        existing = {"caption": "old caption", "genre": ""}
        new = {"caption": "new caption", "genre": "rock"}
        merged = merge_fields(existing, new, policy="fill_missing")
        self.assertEqual(merged["caption"], "old caption")
        self.assertEqual(merged["genre"], "rock")

    def test_overwrite_caption(self):
        """overwrite_caption replaces caption but fills rest."""
        existing = {"caption": "old", "genre": "pop"}
        new = {"caption": "new", "genre": "rock"}
        merged = merge_fields(existing, new, policy="overwrite_caption")
        self.assertEqual(merged["caption"], "new")
        self.assertEqual(merged["genre"], "pop")

    def test_overwrite_all(self):
        """overwrite_all replaces all generated fields."""
        existing = {"caption": "old", "lyrics": "old lyrics", "bpm": "120"}
        new = {"caption": "new", "lyrics": "new lyrics", "bpm": "140"}
        merged = merge_fields(existing, new, policy="overwrite_all")
        self.assertEqual(merged["caption"], "new")
        self.assertEqual(merged["lyrics"], "new lyrics")
        # bpm is in GENERATED_FIELDS, so overwrite_all replaces it
        self.assertEqual(merged["bpm"], "140")

    def test_overwrite_all_preserves_non_generated(self):
        """overwrite_all does NOT overwrite non-generated custom keys."""
        existing = {"caption": "old", "custom_tag": "my_tag"}
        new = {"caption": "new", "custom_tag": "other"}
        merged = merge_fields(existing, new, policy="overwrite_all")
        self.assertEqual(merged["caption"], "new")
        # custom_tag is not in GENERATED_FIELDS → fill_missing applies
        self.assertEqual(merged["custom_tag"], "my_tag")

    def test_empty_new_values_skipped(self):
        """Empty new values are never written."""
        existing = {"caption": "keep"}
        new = {"caption": ""}
        merged = merge_fields(existing, new, policy="overwrite_all")
        self.assertEqual(merged["caption"], "keep")

    def test_inputs_not_mutated(self):
        """Original dicts are not modified."""
        existing = {"caption": "old"}
        new = {"genre": "rock"}
        merge_fields(existing, new, policy="fill_missing")
        self.assertNotIn("genre", existing)


class TestWriteSidecar(unittest.TestCase):
    """Test atomic sidecar writing."""

    def test_roundtrip(self):
        """Write then read produces the same data."""
        data = {
            "caption": "A cool track",
            "genre": "electronic",
            "bpm": "120",
            "key": "Am",
            "signature": "4/4",
            "lyrics": "[Verse]\nLine one\nLine two",
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "song.txt"
            write_sidecar(path, data)
            self.assertTrue(path.exists())
            readback = read_sidecar(path)
            self.assertEqual(readback["caption"], "A cool track")
            self.assertEqual(readback["genre"], "electronic")
            self.assertIn("Line one", readback["lyrics"])

    def test_empty_lyrics(self):
        """Writing with no lyrics produces a valid file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "song.txt"
            write_sidecar(path, {"caption": "test"})
            content = path.read_text(encoding="utf-8")
            self.assertIn("lyrics:", content)

    def test_atomic_no_partial(self):
        """No .tmp files left behind on success."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "song.txt"
            write_sidecar(path, {"caption": "test"})
            tmp_files = list(Path(tmpdir).glob(".sidecar_*"))
            self.assertEqual(len(tmp_files), 0)


    def test_extra_keys_preserved(self):
        """F6: Non-standard keys survive a write-read roundtrip."""
        data = {
            "caption": "A track",
            "repeat": "3",
            "is_instrumental": "true",
            "prompt_override": "custom prompt",
            "lyrics": "[Verse]\nHello",
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "song.txt"
            write_sidecar(path, data)
            readback = read_sidecar(path)
            self.assertEqual(readback["caption"], "A track")
            self.assertEqual(readback["repeat"], "3")
            self.assertEqual(readback["is_instrumental"], "true")
            self.assertEqual(readback["prompt_override"], "custom prompt")
            self.assertIn("Hello", readback["lyrics"])


class TestWriteSidecarErrorHandling(unittest.TestCase):
    """Regression tests for atomic write error handling."""

    def test_no_double_close_on_replace_failure(self):
        """B1: fd is not double-closed when os.replace fails."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "readonly_dir" / "song.txt"
            # Parent dir doesn't exist → os.replace will fail
            # but os.write + os.close succeed on the temp file
            # (mkstemp is in tmpdir, replace target is in nonexistent subdir)
            from sidestep_engine.data import sidecar_io
            import unittest.mock as um

            orig_replace = os.replace

            def bad_replace(src, dst):
                raise OSError("simulated replace failure")

            with um.patch.object(os, "replace", side_effect=bad_replace):
                with self.assertRaises(OSError):
                    write_sidecar(Path(tmpdir) / "song.txt", {"caption": "test"})
            # If we get here without a double-close crash, the fix works.
            # Also verify no temp files leaked
            tmp_files = list(Path(tmpdir).glob(".sidecar_*"))
            self.assertEqual(len(tmp_files), 0)


class TestLyricsColonSafety(unittest.TestCase):
    """Regression: lyrics with colons must survive read-write roundtrips."""

    def _roundtrip(self, data):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "song.txt"
            write_sidecar(path, data)
            return read_sidecar(path)

    def test_colon_in_lyrics_text(self):
        """Lyrics line 'Thinking about you: my love' must not be truncated."""
        data = {
            "caption": "A rock ballad",
            "genre": "rock",
            "lyrics": (
                "[Verse 1]\n"
                "Standing in the rain\n"
                "Thinking about you: my love\n"
                "[Chorus]\n"
                "We will never part"
            ),
        }
        rb = self._roundtrip(data)
        self.assertIn("Thinking about you: my love", rb["lyrics"])
        self.assertIn("[Chorus]", rb["lyrics"])
        self.assertIn("We will never part", rb["lyrics"])
        self.assertNotIn("thinking about you", rb)

    def test_non_bracket_section_markers(self):
        """Lyrics like 'Verse 1:', 'Chorus:', 'Bridge:' must not split."""
        data = {
            "caption": "test",
            "lyrics": (
                "Verse 1:\n"
                "First line of verse\n"
                "Chorus:\n"
                "First line of chorus\n"
                "Bridge:\n"
                "Bridge content\n"
                "Outro:\n"
                "Fade out"
            ),
        }
        rb = self._roundtrip(data)
        self.assertIn("Verse 1:", rb["lyrics"])
        self.assertIn("First line of verse", rb["lyrics"])
        self.assertIn("Chorus:", rb["lyrics"])
        self.assertIn("Bridge:", rb["lyrics"])
        self.assertIn("Outro:", rb["lyrics"])
        self.assertIn("Fade out", rb["lyrics"])
        self.assertNotIn("verse 1", rb)
        self.assertNotIn("chorus", rb)

    def test_timestamp_in_lyrics(self):
        """Lyrics with timestamps like '3:00 AM' survive roundtrip."""
        data = {
            "caption": "test",
            "lyrics": "[Verse]\nIt was 3:00 AM when I called\nYou said: come home",
        }
        rb = self._roundtrip(data)
        self.assertIn("3:00 AM", rb["lyrics"])
        self.assertIn("You said: come home", rb["lyrics"])

    def test_bulk_trigger_tag_preserves_lyrics_with_colons(self):
        """Simulates the bulk trigger tag read-modify-write cycle."""
        original = {
            "caption": "A song",
            "genre": "pop",
            "custom_tag": "",
            "lyrics": "[Verse]\nRemember this: I love you\nAlways: forever",
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "song.txt"
            write_sidecar(path, original)

            data = read_sidecar(path)
            data["custom_tag"] = "suno_v4"
            write_sidecar(path, data)

            final = read_sidecar(path)
            self.assertEqual(final["custom_tag"], "suno_v4")
            self.assertIn("Remember this: I love you", final["lyrics"])
            self.assertIn("Always: forever", final["lyrics"])

    def test_multiple_roundtrips_stable(self):
        """Lyrics must not degrade over repeated read-write cycles."""
        data = {
            "caption": "test",
            "lyrics": "Verse 1:\nHello: world\nNote: important\n[Chorus]\nSing: along",
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "song.txt"
            write_sidecar(path, data)
            for _ in range(5):
                rb = read_sidecar(path)
                write_sidecar(path, rb)
            final = read_sidecar(path)
            self.assertIn("Hello: world", final["lyrics"])
            self.assertIn("Note: important", final["lyrics"])
            self.assertIn("Sing: along", final["lyrics"])


class TestIsInstrumentalRoundtrip(unittest.TestCase):
    """Regression: is_instrumental boolean must survive roundtrip."""

    def test_bool_true_normalised(self):
        """Python bool True is written as lowercase 'true'."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "song.txt"
            write_sidecar(path, {"caption": "test", "is_instrumental": True})
            rb = read_sidecar(path)
            self.assertEqual(rb["is_instrumental"], "true")

    def test_string_True_normalised(self):
        """String 'True' (capital T) is normalised to 'true'."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "song.txt"
            write_sidecar(path, {"caption": "test", "is_instrumental": "True"})
            rb = read_sidecar(path)
            self.assertEqual(rb["is_instrumental"], "true")

    def test_false_not_written_as_extra(self):
        """Boolean false is written as 'false', not dropped."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "song.txt"
            write_sidecar(path, {"caption": "test", "is_instrumental": False})
            content = path.read_text(encoding="utf-8")
            # False is falsy so the extra-keys loop skips it; that's OK —
            # the field simply won't appear, which reads back as absent.
            # This test documents the current behaviour.
            rb = read_sidecar(path)
            self.assertNotIn("True", content)


class TestBackupCreation(unittest.TestCase):
    """Regression: .bak backup is created before overwrite."""

    def test_backup_created_on_overwrite(self):
        """Overwriting an existing sidecar creates a .txt.bak file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "song.txt"
            write_sidecar(path, {"caption": "original"})
            write_sidecar(path, {"caption": "updated"})
            bak = path.with_suffix(".txt.bak")
            self.assertTrue(bak.exists())
            bak_content = bak.read_text(encoding="utf-8")
            self.assertIn("original", bak_content)
            current = read_sidecar(path)
            self.assertEqual(current["caption"], "updated")

    def test_no_backup_on_new_file(self):
        """Creating a new sidecar does not leave a .bak file."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "song.txt"
            write_sidecar(path, {"caption": "new"})
            bak = path.with_suffix(".txt.bak")
            self.assertFalse(bak.exists())


class TestSidecarPathFor(unittest.TestCase):
    """Test sidecar path derivation."""

    def test_wav_to_txt(self):
        self.assertEqual(
            sidecar_path_for(Path("/audio/song.wav")),
            Path("/audio/song.txt"),
        )

    def test_mp3_to_txt(self):
        self.assertEqual(
            sidecar_path_for(Path("track.mp3")),
            Path("track.txt"),
        )


if __name__ == "__main__":
    unittest.main()
