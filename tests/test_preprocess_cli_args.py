"""Tests for preprocessing CLI arguments (target-db, target-lufs)."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch


class TestPreprocessCliArgs(unittest.TestCase):
    """Verify --target-db and --target-lufs parse and pass through."""

    def test_target_db_target_lufs_parse(self):
        """CLI parses --target-db and --target-lufs correctly."""
        from acestep.training_v2.cli.common import build_root_parser

        parser = build_root_parser()
        with tempfile.TemporaryDirectory() as tmp:
            ckpt = Path(tmp) / "ckpt"
            ckpt.mkdir()
            (ckpt / "acestep-v15-base").mkdir()
            audio = Path(tmp) / "audio"
            audio.mkdir()
            tensors = Path(tmp) / "tensors"
            tensors.mkdir()

            args1 = parser.parse_args([
                "fixed",
                "--preprocess",
                "--checkpoint-dir", str(ckpt),
                "--dataset-dir", str(tensors),
                "--output-dir", str(Path(tmp) / "out"),
                "--audio-dir", str(audio),
                "--tensor-output", str(tensors),
                "--normalize", "peak",
                "--target-db", "-3",
            ])
            self.assertEqual(args1.target_db, -3.0)
            self.assertEqual(args1.target_lufs, -14.0)

            args2 = parser.parse_args([
                "fixed",
                "--preprocess",
                "--checkpoint-dir", str(ckpt),
                "--dataset-dir", str(tensors),
                "--output-dir", str(Path(tmp) / "out"),
                "--audio-dir", str(audio),
                "--tensor-output", str(tensors),
                "--normalize", "lufs",
                "--target-lufs", "-18",
            ])
            self.assertEqual(args2.target_db, -1.0)
            self.assertEqual(args2.target_lufs, -18.0)

    def test_defaults_when_omitted(self):
        """When --target-db and --target-lufs omitted, defaults apply."""
        from acestep.training_v2.cli.common import build_root_parser

        parser = build_root_parser()
        with tempfile.TemporaryDirectory() as tmp:
            ckpt = Path(tmp) / "ckpt"
            ckpt.mkdir()
            (ckpt / "acestep-v15-base").mkdir()
            audio = Path(tmp) / "audio"
            audio.mkdir()
            tensors = Path(tmp) / "tensors"
            tensors.mkdir()

            args = parser.parse_args([
                "fixed",
                "--preprocess",
                "--checkpoint-dir", str(ckpt),
                "--dataset-dir", str(tensors),
                "--output-dir", str(Path(tmp) / "out"),
                "--audio-dir", str(audio),
                "--tensor-output", str(tensors),
            ])
            self.assertEqual(args.target_db, -1.0)
            self.assertEqual(args.target_lufs, -14.0)

    def test_preprocess_audio_files_passes_normalize_params(self):
        """preprocess_audio_files forwards target_db and target_lufs to _normalize_audio."""
        from acestep.training_v2.preprocess import preprocess_audio_files

        with tempfile.TemporaryDirectory() as tmp:
            ckpt = Path(tmp) / "ckpt"
            ckpt.mkdir()
            (ckpt / "acestep-v15-base").mkdir()
            audio_dir = Path(tmp) / "audio"
            audio_dir.mkdir()
            # Create a minimal valid WAV so discovery finds something
            import wave
            wav_path = audio_dir / "test.wav"
            with wave.open(str(wav_path), "wb") as w:
                w.setnchannels(2)
                w.setsampwidth(2)
                w.setframerate(48000)
                w.writeframes(b"\x00" * (2 * 2 * 4800))  # 0.1 s of silence

            out_dir = Path(tmp) / "out"
            out_dir.mkdir()

            # Mock _pass1_light to avoid loading real models
            captured_kwargs = {}

            def capture_pass1(**kwargs):
                captured_kwargs.update(kwargs)
                return [], 0

            with patch(
                "acestep.training_v2.preprocess._pass1_light",
                side_effect=capture_pass1,
            ):
                preprocess_audio_files(
                    audio_dir=str(audio_dir),
                    output_dir=str(out_dir),
                    checkpoint_dir=str(ckpt),
                    variant="base",
                    normalize="peak",
                    target_db=-3.0,
                    target_lufs=-18.0,
                )

            self.assertIn("target_db", captured_kwargs)
            self.assertIn("target_lufs", captured_kwargs)
            self.assertEqual(captured_kwargs["target_db"], -3.0)
            self.assertEqual(captured_kwargs["target_lufs"], -18.0)


if __name__ == "__main__":
    unittest.main()
