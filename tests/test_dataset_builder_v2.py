"""Tests for dataset builder and metadata parsing."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from acestep.training_v2.dataset_builder import (
    build_dataset,
    load_sidecar_metadata,
    parse_txt_metadata,
)


class TestParseTxtMetadata(unittest.TestCase):
    """Test key: value metadata file parsing."""

    def test_parse_simple_key_values(self):
        """Should parse simple key: value pairs."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("caption: Test Song\n")
            f.write("genre: Rock\n")
            f.write("bpm: 120\n")
            f.flush()
            path = Path(f.name)

        try:
            result = parse_txt_metadata(path)
            self.assertEqual(result["caption"], "Test Song")
            self.assertEqual(result["genre"], "Rock")
            self.assertEqual(result["bpm"], "120")
        finally:
            path.unlink()

    def test_parse_multiline_values(self):
        """Should handle multi-line values (e.g., lyrics)."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("caption: Test Song\n")
            f.write("lyrics: Line 1\n")
            f.write("Line 2\n")
            f.write("Line 3\n")
            f.write("genre: Pop\n")
            f.flush()
            path = Path(f.name)

        try:
            result = parse_txt_metadata(path)
            self.assertEqual(result["caption"], "Test Song")
            self.assertIn("Line 1", result["lyrics"])
            self.assertIn("Line 2", result["lyrics"])
            self.assertIn("Line 3", result["lyrics"])
            self.assertEqual(result["genre"], "Pop")
        finally:
            path.unlink()

    def test_parse_keys_normalized_to_lowercase(self):
        """Keys should be normalized to lowercase."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("Caption: Test\n")
            f.write("GENRE: Rock\n")
            f.write("BPM: 120\n")
            f.flush()
            path = Path(f.name)

        try:
            result = parse_txt_metadata(path)
            self.assertIn("caption", result)
            self.assertIn("genre", result)
            self.assertIn("bpm", result)
        finally:
            path.unlink()

    def test_parse_empty_file(self):
        """Empty file should return empty dict."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.flush()
            path = Path(f.name)

        try:
            result = parse_txt_metadata(path)
            self.assertEqual(result, {})
        finally:
            path.unlink()

    def test_parse_nonexistent_file(self):
        """Nonexistent file should return empty dict."""
        result = parse_txt_metadata(Path("/nonexistent/file.txt"))
        self.assertEqual(result, {})

    def test_parse_lyrics_with_section_markers(self):
        """Should handle lyrics with [Verse], [Chorus], etc. markers."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("caption: Test Song\n")
            f.write("lyrics:\n")
            f.write("[Verse 1]\n")
            f.write("First line\n")
            f.write("[Chorus]\n")
            f.write("Chorus line\n")
            f.flush()
            path = Path(f.name)

        try:
            result = parse_txt_metadata(path)
            # Section markers should be preserved as continuation lines
            self.assertIn("[Verse 1]", result["lyrics"])
            self.assertIn("[Chorus]", result["lyrics"])
        finally:
            path.unlink()

    def test_parse_value_with_colon(self):
        """Values containing colons should be preserved."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("caption: Test: The Song\n")
            f.flush()
            path = Path(f.name)

        try:
            result = parse_txt_metadata(path)
            self.assertEqual(result["caption"], "Test: The Song")
        finally:
            path.unlink()


class TestLoadSidecarMetadata(unittest.TestCase):
    """Test auto-detection of metadata file conventions."""

    def test_convention_1_key_value_txt(self):
        """Should load key: value .txt file (Side-Step convention)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            audio_path = Path(tmpdir) / "song.wav"
            audio_path.touch()
            txt_path = Path(tmpdir) / "song.txt"
            txt_path.write_text("caption: Test Song\ngenre: Rock\n")

            result = load_sidecar_metadata(audio_path)
            self.assertEqual(result["caption"], "Test Song")
            self.assertEqual(result["genre"], "Rock")

    def test_convention_2_caption_and_lyrics_txt(self):
        """Should load separate .caption.txt and .lyrics.txt files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            audio_path = Path(tmpdir) / "song.wav"
            audio_path.touch()
            caption_path = Path(tmpdir) / "song.caption.txt"
            lyrics_path = Path(tmpdir) / "song.lyrics.txt"
            caption_path.write_text("Test Caption")
            lyrics_path.write_text("Test Lyrics")

            result = load_sidecar_metadata(audio_path)
            self.assertEqual(result["caption"], "Test Caption")
            self.assertEqual(result["lyrics"], "Test Lyrics")

    def test_convention_3_lyrics_only_txt(self):
        """Should use .txt for lyrics when no caption file exists."""
        with tempfile.TemporaryDirectory() as tmpdir:
            audio_path = Path(tmpdir) / "song.wav"
            audio_path.touch()
            txt_path = Path(tmpdir) / "song.txt"
            txt_path.write_text("Just lyrics here")

            result = load_sidecar_metadata(audio_path)
            self.assertEqual(result.get("lyrics", ""), "Just lyrics here")

    def test_no_metadata_files(self):
        """Should return empty dict when no metadata files exist."""
        with tempfile.TemporaryDirectory() as tmpdir:
            audio_path = Path(tmpdir) / "song.wav"
            audio_path.touch()

            result = load_sidecar_metadata(audio_path)
            self.assertEqual(result, {})

    def test_priority_key_value_over_separate_files(self):
        """Key: value .txt should take priority over separate caption/lyrics files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            audio_path = Path(tmpdir) / "song.wav"
            audio_path.touch()
            txt_path = Path(tmpdir) / "song.txt"
            caption_path = Path(tmpdir) / "song.caption.txt"

            txt_path.write_text("caption: From txt\n")
            caption_path.write_text("From caption.txt")

            result = load_sidecar_metadata(audio_path)
            # Key: value format should win
            self.assertEqual(result["caption"], "From txt")


class TestBuildDataset(unittest.TestCase):
    """Test full dataset building from audio directory."""

    def _create_dummy_audio_file(self, path: Path):
        """Create a dummy audio file."""
        # Create a minimal WAV file header (44 bytes)
        with open(path, "wb") as f:
            # RIFF header
            f.write(b"RIFF")
            f.write((36).to_bytes(4, "little"))  # chunk size
            f.write(b"WAVE")
            f.write(b"fmt ")
            f.write((16).to_bytes(4, "little"))  # subchunk1 size
            f.write((1).to_bytes(2, "little"))   # audio format (PCM)
            f.write((1).to_bytes(2, "little"))   # num channels
            f.write((44100).to_bytes(4, "little"))  # sample rate
            f.write((88200).to_bytes(4, "little"))  # byte rate
            f.write((2).to_bytes(2, "little"))   # block align
            f.write((16).to_bytes(2, "little"))  # bits per sample
            f.write(b"data")
            f.write((0).to_bytes(4, "little"))   # data size

    def test_build_dataset_simple(self):
        """Should build dataset from audio files."""
        with tempfile.TemporaryDirectory() as tmpdir:
            audio_dir = Path(tmpdir)
            audio1 = audio_dir / "song1.wav"
            audio2 = audio_dir / "song2.wav"
            self._create_dummy_audio_file(audio1)
            self._create_dummy_audio_file(audio2)

            output_path, stats = build_dataset(
                input_dir=str(audio_dir),
                tag="test_tag",
                name="test_dataset",
            )

            self.assertTrue(output_path.exists())
            self.assertEqual(stats["total"], 2)
            self.assertEqual(stats["skipped"], 0)

            # Verify JSON structure
            data = json.loads(output_path.read_text())
            self.assertEqual(len(data["samples"]), 2)
            self.assertEqual(data["metadata"]["name"], "test_dataset")
            self.assertEqual(data["metadata"]["custom_tag"], "test_tag")

    def test_build_dataset_with_metadata(self):
        """Should include sidecar metadata in dataset."""
        with tempfile.TemporaryDirectory() as tmpdir:
            audio_dir = Path(tmpdir)
            audio1 = audio_dir / "song1.wav"
            self._create_dummy_audio_file(audio1)
            txt1 = audio_dir / "song1.txt"
            txt1.write_text("caption: My Caption\ngenre: Rock\nbpm: 120\n")

            output_path, stats = build_dataset(
                input_dir=str(audio_dir),
                tag="",
                name="test_dataset",
            )

            data = json.loads(output_path.read_text())
            sample = data["samples"][0]
            self.assertEqual(sample["caption"], "My Caption")
            self.assertEqual(sample["genre"], "Rock")
            self.assertEqual(sample["bpm"], 120)
            self.assertEqual(stats["with_metadata"], 1)

    def test_build_dataset_fallback_caption_from_filename(self):
        """Should derive caption from filename when no metadata exists."""
        with tempfile.TemporaryDirectory() as tmpdir:
            audio_dir = Path(tmpdir)
            audio1 = audio_dir / "my_test_song.wav"
            self._create_dummy_audio_file(audio1)

            output_path, _ = build_dataset(
                input_dir=str(audio_dir),
                tag="",
                name="test_dataset",
            )

            data = json.loads(output_path.read_text())
            sample = data["samples"][0]
            # Underscores and hyphens should be replaced with spaces
            self.assertEqual(sample["caption"], "my test song")

    def test_build_dataset_instrumental_detection(self):
        """Should detect instrumental tracks."""
        with tempfile.TemporaryDirectory() as tmpdir:
            audio_dir = Path(tmpdir)
            audio1 = audio_dir / "song1.wav"
            self._create_dummy_audio_file(audio1)
            txt1 = audio_dir / "song1.txt"
            txt1.write_text("caption: Song\nis_instrumental: true\n")

            output_path, _ = build_dataset(
                input_dir=str(audio_dir),
                tag="",
                name="test_dataset",
            )

            data = json.loads(output_path.read_text())
            sample = data["samples"][0]
            self.assertTrue(sample["is_instrumental"])
            self.assertEqual(sample["lyrics"], "[Instrumental]")

    def test_build_dataset_empty_directory(self):
        """Should raise error when no audio files found."""
        with tempfile.TemporaryDirectory() as tmpdir:
            with self.assertRaises(FileNotFoundError):
                build_dataset(
                    input_dir=str(tmpdir),
                    tag="",
                    name="test_dataset",
                )

    def test_build_dataset_recursive_scan(self):
        """Should recursively find audio files in subdirectories."""
        with tempfile.TemporaryDirectory() as tmpdir:
            audio_dir = Path(tmpdir)
            subdir = audio_dir / "subdir"
            subdir.mkdir()
            audio1 = audio_dir / "song1.wav"
            audio2 = subdir / "song2.wav"
            self._create_dummy_audio_file(audio1)
            self._create_dummy_audio_file(audio2)

            output_path, stats = build_dataset(
                input_dir=str(audio_dir),
                tag="",
                name="test_dataset",
            )

            self.assertEqual(stats["total"], 2)

    def test_build_dataset_custom_tag(self):
        """Custom tag should be applied to all samples."""
        with tempfile.TemporaryDirectory() as tmpdir:
            audio_dir = Path(tmpdir)
            audio1 = audio_dir / "song1.wav"
            self._create_dummy_audio_file(audio1)

            output_path, _ = build_dataset(
                input_dir=str(audio_dir),
                tag="custom_tag",
                name="test_dataset",
            )

            data = json.loads(output_path.read_text())
            sample = data["samples"][0]
            self.assertEqual(sample["custom_tag"], "custom_tag")
            self.assertEqual(data["metadata"]["custom_tag"], "custom_tag")

    def test_build_dataset_bpm_parsing(self):
        """BPM should be parsed as integer."""
        with tempfile.TemporaryDirectory() as tmpdir:
            audio_dir = Path(tmpdir)
            audio1 = audio_dir / "song1.wav"
            self._create_dummy_audio_file(audio1)
            txt1 = audio_dir / "song1.txt"
            txt1.write_text("caption: Song\nbpm: 120.5\n")

            output_path, _ = build_dataset(
                input_dir=str(audio_dir),
                tag="",
                name="test_dataset",
            )

            data = json.loads(output_path.read_text())
            sample = data["samples"][0]
            self.assertEqual(sample["bpm"], 120)
            self.assertIsInstance(sample["bpm"], int)

    def test_build_dataset_invalid_bpm(self):
        """Invalid BPM should be set to None."""
        with tempfile.TemporaryDirectory() as tmpdir:
            audio_dir = Path(tmpdir)
            audio1 = audio_dir / "song1.wav"
            self._create_dummy_audio_file(audio1)
            txt1 = audio_dir / "song1.txt"
            txt1.write_text("caption: Song\nbpm: invalid\n")

            output_path, _ = build_dataset(
                input_dir=str(audio_dir),
                tag="",
                name="test_dataset",
            )

            data = json.loads(output_path.read_text())
            sample = data["samples"][0]
            self.assertIsNone(sample["bpm"])


if __name__ == "__main__":
    unittest.main()