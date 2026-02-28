"""Tests for the folder-based dataset builder."""

from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


class TestParseTxtMetadata(unittest.TestCase):
    """Verify key:value text file parsing."""

    def test_basic_key_value(self):
        """Simple key:value pairs are parsed correctly."""
        from sidestep_engine.data.dataset_builder import parse_txt_metadata

        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("caption: A cool jazz track\ngenre: Jazz\nbpm: 120\n")
            f.flush()
            path = Path(f.name)

        try:
            meta = parse_txt_metadata(path)
            self.assertEqual(meta["caption"], "A cool jazz track")
            self.assertEqual(meta["genre"], "Jazz")
            self.assertEqual(meta["bpm"], "120")
        finally:
            os.unlink(path)

    def test_multiline_lyrics(self):
        """Lyrics spanning multiple lines are captured as one value."""
        from sidestep_engine.data.dataset_builder import parse_txt_metadata

        content = (
            "caption: My Song\n"
            "lyrics:\n"
            "[Verse 1]\n"
            "Hello world\n"
            "This is a test\n"
            "[Chorus]\n"
            "La la la\n"
        )
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write(content)
            f.flush()
            path = Path(f.name)

        try:
            meta = parse_txt_metadata(path)
            self.assertEqual(meta["caption"], "My Song")
            self.assertIn("[Verse 1]", meta["lyrics"])
            self.assertIn("[Chorus]", meta["lyrics"])
        finally:
            os.unlink(path)

    def test_missing_file_returns_empty(self):
        """Non-existent file returns empty dict."""
        from sidestep_engine.data.dataset_builder import parse_txt_metadata

        result = parse_txt_metadata(Path("/nonexistent/file.txt"))
        self.assertEqual(result, {})

    def test_keys_lowercased(self):
        """Keys are normalised to lowercase."""
        from sidestep_engine.data.dataset_builder import parse_txt_metadata

        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("Caption: Test\nGenre: Rock\n")
            f.flush()
            path = Path(f.name)

        try:
            meta = parse_txt_metadata(path)
            self.assertIn("caption", meta)
            self.assertIn("genre", meta)
        finally:
            os.unlink(path)

    def test_windows_bom_and_crlf(self):
        """UTF-8 BOM + CRLF line endings (Windows Notepad) parse correctly."""
        from sidestep_engine.data.dataset_builder import parse_txt_metadata

        # BOM + CRLF content simulating a Windows-created .txt
        raw = (
            b"\xef\xbb\xbf"  # UTF-8 BOM
            b"caption: A cool jazz track\r\n"
            b"genre: Jazz\r\n"
            b"bpm: 120\r\n"
            b"key: Am\r\n"
            b"lyrics:\r\n"
            b"[Verse 1]\r\n"
            b"Hello world\r\n"
        )
        with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as f:
            f.write(raw)
            f.flush()
            path = Path(f.name)

        try:
            meta = parse_txt_metadata(path)
            self.assertEqual(meta.get("caption"), "A cool jazz track")
            self.assertEqual(meta.get("genre"), "Jazz")
            self.assertEqual(meta.get("bpm"), "120")
            self.assertEqual(meta.get("key"), "Am")
            self.assertIn("[Verse 1]", meta.get("lyrics", ""))
            self.assertIn("Hello world", meta.get("lyrics", ""))
        finally:
            os.unlink(path)

    def test_leading_whitespace_on_key_lines(self):
        """Keys with minor leading whitespace still parse as keys."""
        from sidestep_engine.data.dataset_builder import parse_txt_metadata

        content = "  caption: Indented caption\n  genre: Rock\n  bpm: 90\n"
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write(content)
            f.flush()
            path = Path(f.name)

        try:
            meta = parse_txt_metadata(path)
            self.assertEqual(meta.get("caption"), "Indented caption")
            self.assertEqual(meta.get("genre"), "Rock")
            self.assertEqual(meta.get("bpm"), "90")
        finally:
            os.unlink(path)

    def test_bom_only_file_returns_empty(self):
        """A file containing only a BOM (no actual content) returns {}."""
        from sidestep_engine.data.dataset_builder import parse_txt_metadata

        with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as f:
            f.write(b"\xef\xbb\xbf")
            f.flush()
            path = Path(f.name)

        try:
            self.assertEqual(parse_txt_metadata(path), {})
        finally:
            os.unlink(path)


class TestLoadSidecarMetadata(unittest.TestCase):
    """Verify auto-detection of metadata file conventions."""

    def test_txt_key_value_detected(self):
        """Convention 1: song.txt with key:value pairs."""
        from sidestep_engine.data.dataset_builder import load_sidecar_metadata

        with tempfile.TemporaryDirectory() as td:
            audio = Path(td) / "song.wav"
            audio.touch()
            txt = Path(td) / "song.txt"
            txt.write_text("caption: My Caption\ngenre: Pop\n")

            meta = load_sidecar_metadata(audio)
            self.assertEqual(meta["caption"], "My Caption")
            self.assertEqual(meta["genre"], "Pop")

    def test_caption_txt_detected(self):
        """Convention 2: song.caption.txt + song.lyrics.txt."""
        from sidestep_engine.data.dataset_builder import load_sidecar_metadata

        with tempfile.TemporaryDirectory() as td:
            audio = Path(td) / "track.wav"
            audio.touch()
            Path(td, "track.caption.txt").write_text("An ambient track")
            Path(td, "track.lyrics.txt").write_text("[Verse]\nDrifting away")

            meta = load_sidecar_metadata(audio)
            self.assertEqual(meta["caption"], "An ambient track")
            self.assertIn("Drifting away", meta["lyrics"])

    def test_no_metadata_returns_empty(self):
        """Convention 3: no sidecar files returns empty dict."""
        from sidestep_engine.data.dataset_builder import load_sidecar_metadata

        with tempfile.TemporaryDirectory() as td:
            audio = Path(td) / "orphan.wav"
            audio.touch()

            meta = load_sidecar_metadata(audio)
            self.assertEqual(meta, {})


class TestBuildDataset(unittest.TestCase):
    """Verify end-to-end dataset JSON generation."""

    def test_builds_valid_json(self):
        """A folder with audio + txt should produce valid dataset.json."""
        from sidestep_engine.data.dataset_builder import build_dataset

        with tempfile.TemporaryDirectory() as td:
            # Create a fake audio file (extension matters, content does not)
            audio = Path(td) / "mysong.wav"
            audio.write_bytes(b"\x00" * 100)
            txt = Path(td) / "mysong.txt"
            txt.write_text("caption: Test Song\ngenre: Electronic\nbpm: 128\n")

            with patch("sidestep_engine.data.dataset_builder.get_audio_duration", return_value=180):
                out_path, stats = build_dataset(
                    input_dir=td, tag="test_tag", tag_position="prepend", name="test_ds",
                )

            self.assertTrue(out_path.exists())
            data = json.loads(out_path.read_text())
            self.assertEqual(data["metadata"]["name"], "test_ds")
            self.assertEqual(data["metadata"]["custom_tag"], "test_tag")
            self.assertEqual(len(data["samples"]), 1)

            sample = data["samples"][0]
            self.assertEqual(sample["caption"], "Test Song")
            self.assertEqual(sample["genre"], "Electronic")
            self.assertEqual(sample["bpm"], 128)
            self.assertEqual(sample["duration"], 180)
            self.assertEqual(sample["custom_tag"], "test_tag")

    def test_no_audio_raises(self):
        """Empty directory should raise FileNotFoundError."""
        from sidestep_engine.data.dataset_builder import build_dataset

        with tempfile.TemporaryDirectory() as td:
            with self.assertRaises(FileNotFoundError):
                build_dataset(input_dir=td)

    def test_instrumental_detection(self):
        """Song with no lyrics should be marked instrumental."""
        from sidestep_engine.data.dataset_builder import build_dataset

        with tempfile.TemporaryDirectory() as td:
            audio = Path(td) / "beat.flac"
            audio.write_bytes(b"\x00" * 100)
            txt = Path(td) / "beat.txt"
            txt.write_text("caption: A beat\n")

            with patch("sidestep_engine.data.dataset_builder.get_audio_duration", return_value=60):
                out_path, _ = build_dataset(input_dir=td)

            data = json.loads(out_path.read_text())
            self.assertTrue(data["samples"][0]["is_instrumental"])
            self.assertEqual(data["samples"][0]["lyrics"], "[Instrumental]")


class TestGenreRatio(unittest.TestCase):
    """Verify genre_ratio is threaded through to the output JSON."""

    def test_genre_ratio_written_to_metadata(self):
        """genre_ratio=30 should appear in the metadata block."""
        from sidestep_engine.data.dataset_builder import build_dataset

        with tempfile.TemporaryDirectory() as td:
            audio = Path(td) / "track.wav"
            audio.write_bytes(b"\x00" * 100)
            Path(td, "track.txt").write_text("caption: A track\ngenre: Rock\n")

            with patch("sidestep_engine.data.dataset_builder.get_audio_duration", return_value=120):
                out_path, _ = build_dataset(input_dir=td, genre_ratio=30)

            data = json.loads(out_path.read_text())
            self.assertEqual(data["metadata"]["genre_ratio"], 30)

    def test_genre_ratio_defaults_to_zero(self):
        """Omitting genre_ratio should write 0 (backward compat)."""
        from sidestep_engine.data.dataset_builder import build_dataset

        with tempfile.TemporaryDirectory() as td:
            audio = Path(td) / "song.wav"
            audio.write_bytes(b"\x00" * 100)
            Path(td, "song.txt").write_text("caption: A song\n")

            with patch("sidestep_engine.data.dataset_builder.get_audio_duration", return_value=60):
                out_path, _ = build_dataset(input_dir=td)

            data = json.loads(out_path.read_text())
            self.assertEqual(data["metadata"]["genre_ratio"], 0)

    def test_genre_ratio_does_not_affect_samples(self):
        """genre_ratio is metadata-only; per-sample fields are unchanged."""
        from sidestep_engine.data.dataset_builder import build_dataset

        with tempfile.TemporaryDirectory() as td:
            audio = Path(td) / "beat.wav"
            audio.write_bytes(b"\x00" * 100)
            Path(td, "beat.txt").write_text("caption: My Beat\ngenre: Jazz\nbpm: 90\n")

            with patch("sidestep_engine.data.dataset_builder.get_audio_duration", return_value=180):
                out_path, _ = build_dataset(input_dir=td, genre_ratio=50)

            data = json.loads(out_path.read_text())
            sample = data["samples"][0]
            self.assertEqual(sample["caption"], "My Beat")
            self.assertEqual(sample["genre"], "Jazz")
            self.assertEqual(sample["bpm"], 90)
            self.assertNotIn("genre_ratio", sample)


if __name__ == "__main__":
    unittest.main()
