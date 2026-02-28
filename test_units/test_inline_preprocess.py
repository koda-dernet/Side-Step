"""Tests for inline preprocessing helpers and related modules.

Covers:
- Pre-existing dataset.json detection.
- Sidecar metadata fallback in preprocess_discovery.
- Upstream compat fields in dataset_builder.
- Per-file metadata summary formatting.
- Metadata flags extraction (including key + signature columns).
- Smart save_best_after and warmup ratio warnings (L1).
- Signature alias in dataset builder.
- CoverageChunkSampler: weighted offset, decay, persistence.
"""

from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


class TestDetectExistingJson(unittest.TestCase):
    """Verify detect_existing_json finds pre-existing dataset.json."""

    @patch("sidestep_engine.ui.flows.inline_preprocess.print_message")
    def test_returns_path_when_exists(self, _mock_pm):
        from sidestep_engine.ui.flows.inline_preprocess import detect_existing_json

        d = tempfile.mkdtemp(prefix="sidestep_test_dj_")
        try:
            dj = {"metadata": {"name": "test"}, "samples": []}
            Path(d, "dataset.json").write_text(json.dumps(dj), encoding="utf-8")
            result = detect_existing_json(d)
            self.assertEqual(result, str(Path(d) / "dataset.json"))
        finally:
            import shutil
            shutil.rmtree(d, ignore_errors=True)

    @patch("sidestep_engine.ui.flows.inline_preprocess.print_message")
    def test_returns_none_when_missing(self, _mock_pm):
        from sidestep_engine.ui.flows.inline_preprocess import detect_existing_json

        d = tempfile.mkdtemp(prefix="sidestep_test_nodj_")
        try:
            result = detect_existing_json(d)
            self.assertIsNone(result)
        finally:
            import shutil
            shutil.rmtree(d, ignore_errors=True)


class TestSidecarMetadataFallback(unittest.TestCase):
    """Verify preprocess_discovery reads sidecars when no JSON is available."""

    def test_sidecars_read_without_json(self):
        """load_sample_metadata with dataset_json=None should read .txt sidecars."""
        from sidestep_engine.data.preprocess_discovery import load_sample_metadata

        d = tempfile.mkdtemp(prefix="sidestep_test_sc_")
        try:
            Path(d, "song1.wav").write_bytes(b"\x00" * 100)
            Path(d, "song1.txt").write_text(
                "caption: A dreamy ambient track\ngenre: ambient\nbpm: 90\n",
                encoding="utf-8",
            )
            Path(d, "song2.mp3").write_bytes(b"\x00" * 100)
            # song2 has no sidecar — should get defaults

            audio_files = sorted(Path(d).rglob("*"), key=lambda f: f.name)
            audio_files = [f for f in audio_files if f.suffix.lower() in {".wav", ".mp3"}]
            meta = load_sample_metadata(None, audio_files)

            self.assertIn("song1.wav", meta)
            self.assertEqual(meta["song1.wav"]["caption"], "A dreamy ambient track")
            self.assertEqual(meta["song1.wav"]["genre"], "ambient")
            self.assertEqual(meta["song1.wav"]["bpm"], 90)

            # song2 should have filename-derived defaults
            self.assertIn("song2.mp3", meta)
            self.assertEqual(meta["song2.mp3"]["caption"], "song2")
            self.assertEqual(meta["song2.mp3"]["lyrics"], "[Instrumental]")
        finally:
            import shutil
            shutil.rmtree(d, ignore_errors=True)


class TestSidecarDurationFallback(unittest.TestCase):
    """Prove duration is computed from audio file, not silently zero."""

    def test_duration_computed_from_audio(self):
        """When sidecar has no duration, get_audio_duration should be called."""
        from sidestep_engine.data.sidecar_metadata import load_sidecars_for_files

        d = tempfile.mkdtemp(prefix="sidestep_test_dur_")
        try:
            Path(d, "song.wav").write_bytes(b"\x00" * 100)
            Path(d, "song.txt").write_text(
                "caption: Test\nbpm: 120\n", encoding="utf-8",
            )
            audio_files = [Path(d, "song.wav")]

            with patch(
                "sidestep_engine.data.audio_duration.get_audio_duration",
                return_value=237,
            ) as mock_dur:
                meta = load_sidecars_for_files(audio_files)

            mock_dur.assert_called_once()
            self.assertEqual(meta["song.wav"]["duration"], 237)
        finally:
            import shutil
            shutil.rmtree(d, ignore_errors=True)

    def test_duration_from_sidecar_respected(self):
        """When sidecar explicitly provides duration, it should be used."""
        from sidestep_engine.data.sidecar_metadata import normalize_sidecar

        raw = {"caption": "Test", "duration": "180"}
        result = normalize_sidecar(raw, Path("/fake/song.wav"))
        self.assertEqual(result["duration"], 180)


class TestUpstreamCompatFields(unittest.TestCase):
    """Verify dataset_builder produces upstream-compatible JSON fields."""

    def test_upstream_fields_present(self):
        from sidestep_engine.data.dataset_builder import build_dataset

        d = tempfile.mkdtemp(prefix="sidestep_test_compat_")
        try:
            Path(d, "song.wav").write_bytes(b"\x00" * 100)
            Path(d, "song.txt").write_text(
                "caption: Test\ngenre: pop\nbpm: 120\n",
                encoding="utf-8",
            )
            out_json = str(Path(d, "dataset.json"))
            _, stats = build_dataset(input_dir=d, output=out_json)
            data = json.loads(Path(out_json).read_text(encoding="utf-8"))
            sample = data["samples"][0]

            # Upstream compat fields
            self.assertIn("id", sample)
            self.assertEqual(len(sample["id"]), 8)  # hex[:8]
            self.assertIn("raw_lyrics", sample)
            self.assertIn("formatted_lyrics", sample)
            self.assertEqual(sample["formatted_lyrics"], "")
            self.assertIn("language", sample)
            self.assertEqual(sample["language"], "unknown")
            self.assertIn("labeled", sample)
            self.assertTrue(sample["labeled"])  # has caption

            # Metadata block
            meta = data["metadata"]
            self.assertIn("all_instrumental", meta)
        finally:
            import shutil
            shutil.rmtree(d, ignore_errors=True)


class TestMetadataFlags(unittest.TestCase):
    """Verify _metadata_flags extraction logic."""

    def test_all_present(self):
        from sidestep_engine.ui.flows.inline_preprocess import _metadata_flags

        sample = {
            "caption": "test", "lyrics": "Hello", "bpm": 120, "genre": "pop",
            "keyscale": "C major", "timesignature": "4/4",
        }
        flags = _metadata_flags(sample)
        self.assertTrue(flags["caption"])
        self.assertTrue(flags["lyrics"])
        self.assertTrue(flags["bpm"])
        self.assertTrue(flags["genre"])
        self.assertTrue(flags["key"])
        self.assertTrue(flags["sig"])

    def test_instrumental_lyrics_false(self):
        """[Instrumental] lyrics should flag as absent."""
        from sidestep_engine.ui.flows.inline_preprocess import _metadata_flags

        sample = {"caption": "test", "lyrics": "[Instrumental]", "bpm": None, "genre": ""}
        flags = _metadata_flags(sample)
        self.assertTrue(flags["caption"])
        self.assertFalse(flags["lyrics"])
        self.assertFalse(flags["bpm"])
        self.assertFalse(flags["genre"])

    def test_empty_sample(self):
        from sidestep_engine.ui.flows.inline_preprocess import _metadata_flags

        flags = _metadata_flags({})
        self.assertFalse(flags["caption"])
        self.assertFalse(flags["lyrics"])
        self.assertFalse(flags["bpm"])
        self.assertFalse(flags["genre"])
        self.assertFalse(flags["key"])
        self.assertFalse(flags["sig"])


class TestFormatMarks(unittest.TestCase):
    """Verify _format_marks output for both plain and Rich modes."""

    def test_plain_output(self):
        from sidestep_engine.ui.flows.inline_preprocess import _format_marks

        flags = {
            "caption": True, "lyrics": False, "bpm": True,
            "genre": False, "key": True, "sig": False,
        }
        result = _format_marks(flags, use_rich=False)
        self.assertIn("✓ caption", result)
        self.assertIn("– lyrics", result)
        self.assertIn("✓ bpm", result)
        self.assertIn("– genre", result)
        self.assertIn("✓ key", result)
        self.assertIn("– sig", result)

    def test_rich_output(self):
        from sidestep_engine.ui.flows.inline_preprocess import _format_marks

        flags = {
            "caption": True, "lyrics": False, "bpm": True,
            "genre": False, "key": True, "sig": False,
        }
        result = _format_marks(flags, use_rich=True)
        self.assertIn("[green]✓[/]", result)
        self.assertIn("[dim]–[/]", result)


class TestSmartDefaults(unittest.TestCase):
    """Verify _smart_save_best_default and _warn_warmup_ratio."""

    def test_smart_save_best_small_dataset(self):
        """Small dataset should start tracking after warmup + buffer."""
        from sidestep_engine.ui.flows.train_steps import _smart_save_best_default

        d = tempfile.mkdtemp(prefix="sidestep_test_pt_")
        try:
            for i in range(5):
                Path(d, f"sample_{i}.pt").write_bytes(b"\x00")
            # 5 samples, batch=1, accum=4 => 2 steps/epoch, 100 epochs => 200 total
            # warmup=100 => max(100+10, min(200, 200//10)) = max(110, 20) = 110
            answers = {
                "dataset_dir": d, "batch_size": 1,
                "gradient_accumulation": 4, "epochs": 100, "max_steps": 0,
                "warmup_steps": 100,
            }
            default = _smart_save_best_default(answers)
            self.assertEqual(default, 110)
        finally:
            import shutil
            shutil.rmtree(d, ignore_errors=True)

    def test_smart_save_best_large_dataset(self):
        """Large dataset: warmup+10 wins when larger than total//10 cap."""
        from sidestep_engine.ui.flows.train_steps import _smart_save_best_default

        d = tempfile.mkdtemp(prefix="sidestep_test_pt_lg_")
        try:
            for i in range(100):
                Path(d, f"sample_{i}.pt").write_bytes(b"\x00")
            # 100 samples, batch=1, accum=4 => 25 steps/epoch, 100 epochs => 2500 total
            answers = {
                "dataset_dir": d, "batch_size": 1,
                "gradient_accumulation": 4, "epochs": 100, "max_steps": 0,
            }
            default = _smart_save_best_default(answers)
            # max(10, min(200, 2500 // 10)) = max(10, 200) = 200
            self.assertEqual(default, 200)
        finally:
            import shutil
            shutil.rmtree(d, ignore_errors=True)

    def test_smart_save_best_unknown_dir(self):
        """Unknown dataset dir falls back to 200."""
        from sidestep_engine.ui.flows.train_steps import _smart_save_best_default

        default = _smart_save_best_default({"dataset_dir": "/nonexistent"})
        self.assertEqual(default, 200)

    def test_warn_warmup_high_ratio(self):
        """High warmup ratio prints a warning (doesn't crash)."""
        from sidestep_engine.ui.flows.train_steps import _warn_warmup_ratio

        d = tempfile.mkdtemp(prefix="sidestep_test_pt_wu_")
        try:
            for i in range(4):
                Path(d, f"s_{i}.pt").write_bytes(b"\x00")
            # 4 samples, batch=1, accum=4 => 1 step/epoch, 10 epochs => 10 total
            # warmup=5 => 50% — should trigger warning
            answers = {
                "dataset_dir": d, "batch_size": 1,
                "gradient_accumulation": 4, "epochs": 10,
                "max_steps": 0, "warmup_steps": 5,
            }
            # Should not raise; just prints a warning
            _warn_warmup_ratio(answers)
        finally:
            import shutil
            shutil.rmtree(d, ignore_errors=True)


class TestScanSidecarTags(unittest.TestCase):
    """Verify _scan_sidecar_tags detects custom_tag from sidecars."""

    def test_single_tag_detected(self):
        from sidestep_engine.ui.flows.build_dataset import scan_sidecar_tags

        d = tempfile.mkdtemp(prefix="sidestep_test_tag_")
        try:
            Path(d, "song1.wav").write_bytes(b"\x00" * 100)
            Path(d, "song1.txt").write_text(
                "caption: Test\ncustom_tag: my_trigger\n", encoding="utf-8",
            )
            Path(d, "song2.wav").write_bytes(b"\x00" * 100)
            Path(d, "song2.txt").write_text(
                "caption: Test2\ncustom_tag: my_trigger\n", encoding="utf-8",
            )
            tags = scan_sidecar_tags(d)
            self.assertEqual(tags, {"my_trigger"})
        finally:
            import shutil
            shutil.rmtree(d, ignore_errors=True)

    def test_multiple_tags_detected(self):
        from sidestep_engine.ui.flows.build_dataset import scan_sidecar_tags

        d = tempfile.mkdtemp(prefix="sidestep_test_mtag_")
        try:
            Path(d, "s1.wav").write_bytes(b"\x00" * 100)
            Path(d, "s1.txt").write_text("caption: A\ncustom_tag: tag_a\n", encoding="utf-8")
            Path(d, "s2.wav").write_bytes(b"\x00" * 100)
            Path(d, "s2.txt").write_text("caption: B\ncustom_tag: tag_b\n", encoding="utf-8")
            tags = scan_sidecar_tags(d)
            self.assertEqual(tags, {"tag_a", "tag_b"})
        finally:
            import shutil
            shutil.rmtree(d, ignore_errors=True)

    def test_no_tags(self):
        from sidestep_engine.ui.flows.build_dataset import scan_sidecar_tags

        d = tempfile.mkdtemp(prefix="sidestep_test_notag_")
        try:
            Path(d, "s.wav").write_bytes(b"\x00" * 100)
            Path(d, "s.txt").write_text("caption: Test\n", encoding="utf-8")
            tags = scan_sidecar_tags(d)
            self.assertEqual(tags, set())
        finally:
            import shutil
            shutil.rmtree(d, ignore_errors=True)


class TestWritebackTag(unittest.TestCase):
    """Verify _writeback_tag_to_sidecars writes tag without overwriting existing."""

    @patch("sidestep_engine.ui.flows.build_dataset.print_message")
    def test_writes_to_empty_sidecars(self, _mock_pm):
        from sidestep_engine.ui.flows.build_dataset import writeback_tag_to_sidecars
        from sidestep_engine.data.sidecar_io import read_sidecar

        d = tempfile.mkdtemp(prefix="sidestep_test_wb_")
        try:
            Path(d, "s.wav").write_bytes(b"\x00" * 100)
            Path(d, "s.txt").write_text("caption: Test\n", encoding="utf-8")
            writeback_tag_to_sidecars(d, "new_trigger")
            data = read_sidecar(Path(d, "s.txt"))
            self.assertEqual(data.get("custom_tag"), "new_trigger")
        finally:
            import shutil
            shutil.rmtree(d, ignore_errors=True)

    @patch("sidestep_engine.ui.flows.build_dataset.print_message")
    def test_does_not_overwrite_existing_tag(self, _mock_pm):
        from sidestep_engine.ui.flows.build_dataset import writeback_tag_to_sidecars
        from sidestep_engine.data.sidecar_io import read_sidecar

        d = tempfile.mkdtemp(prefix="sidestep_test_wb_exist_")
        try:
            Path(d, "s.wav").write_bytes(b"\x00" * 100)
            Path(d, "s.txt").write_text(
                "caption: Test\ncustom_tag: original\n", encoding="utf-8",
            )
            writeback_tag_to_sidecars(d, "new_trigger")
            data = read_sidecar(Path(d, "s.txt"))
            self.assertEqual(data.get("custom_tag"), "original")
        finally:
            import shutil
            shutil.rmtree(d, ignore_errors=True)


    @patch("sidestep_engine.ui.flows.build_dataset.print_message")
    def test_skips_files_without_sidecar(self, _mock_pm):
        """Audio files with no .txt sidecar should not get a new one created."""
        from sidestep_engine.ui.flows.build_dataset import writeback_tag_to_sidecars

        d = tempfile.mkdtemp(prefix="sidestep_test_wb_nosc_")
        try:
            Path(d, "nosidecar.wav").write_bytes(b"\x00" * 100)
            writeback_tag_to_sidecars(d, "my_trigger")
            self.assertFalse(Path(d, "nosidecar.txt").exists())
        finally:
            import shutil
            shutil.rmtree(d, ignore_errors=True)


class TestCustomTagPriority(unittest.TestCase):
    """Verify build_dataset prefers per-sample sidecar custom_tag over wizard tag."""

    def test_sidecar_tag_wins_over_wizard(self):
        from sidestep_engine.data.dataset_builder import build_dataset

        d = tempfile.mkdtemp(prefix="sidestep_test_tagpri_")
        try:
            Path(d, "s.wav").write_bytes(b"\x00" * 100)
            Path(d, "s.txt").write_text(
                "caption: Test\ncustom_tag: sidecar_tag\n", encoding="utf-8",
            )
            out = str(Path(d, "ds.json"))
            _, _ = build_dataset(input_dir=d, tag="wizard_tag", output=out)
            data = json.loads(Path(out).read_text(encoding="utf-8"))
            self.assertEqual(data["samples"][0]["custom_tag"], "sidecar_tag")
        finally:
            import shutil
            shutil.rmtree(d, ignore_errors=True)

    def test_wizard_tag_used_when_no_sidecar_tag(self):
        from sidestep_engine.data.dataset_builder import build_dataset

        d = tempfile.mkdtemp(prefix="sidestep_test_tagfall_")
        try:
            Path(d, "s.wav").write_bytes(b"\x00" * 100)
            Path(d, "s.txt").write_text("caption: Test\n", encoding="utf-8")
            out = str(Path(d, "ds.json"))
            _, _ = build_dataset(input_dir=d, tag="wizard_tag", output=out)
            data = json.loads(Path(out).read_text(encoding="utf-8"))
            self.assertEqual(data["samples"][0]["custom_tag"], "wizard_tag")
        finally:
            import shutil
            shutil.rmtree(d, ignore_errors=True)


class TestSignatureAlias(unittest.TestCase):
    """Verify 'signature' is accepted as an alias for 'timesignature'."""

    def test_signature_parsed_from_sidecar(self):
        """A .txt sidecar with 'signature: 4/4' should populate timesignature."""
        from sidestep_engine.data.dataset_builder import parse_txt_metadata

        d = tempfile.mkdtemp(prefix="sidestep_test_sig_")
        try:
            txt = Path(d, "song.txt")
            txt.write_text(
                "caption: Test song\nsignature: 4/4\nbpm: 120\n",
                encoding="utf-8",
            )
            meta = parse_txt_metadata(txt)
            self.assertEqual(meta.get("signature"), "4/4")
        finally:
            import shutil
            shutil.rmtree(d, ignore_errors=True)

    def test_signature_alias_in_build(self):
        """build_dataset should map 'signature' to 'timesignature' field."""
        from sidestep_engine.data.dataset_builder import build_dataset

        d = tempfile.mkdtemp(prefix="sidestep_test_sig_build_")
        try:
            Path(d, "song.wav").write_bytes(b"\x00" * 100)
            Path(d, "song.txt").write_text(
                "caption: Sig test\nsignature: 3/4\n",
                encoding="utf-8",
            )
            out_json = str(Path(d, "dataset.json"))
            _, stats = build_dataset(input_dir=d, output=out_json)
            data = json.loads(Path(out_json).read_text(encoding="utf-8"))
            sample = data["samples"][0]
            self.assertEqual(sample["timesignature"], "3/4")
        finally:
            import shutil
            shutil.rmtree(d, ignore_errors=True)


@unittest.skipUnless(HAS_TORCH, "torch not installed")
class TestCoverageChunkSampler(unittest.TestCase):
    """Verify CoverageChunkSampler offset selection and state management."""

    def test_sample_offset_in_range(self):
        """Returned offset must be in [0, total - chunk)."""
        from sidestep_engine.data.chunk_sampler import CoverageChunkSampler

        sampler = CoverageChunkSampler(n_bins=8, decay_every=0)
        for _ in range(50):
            offset = sampler.sample_offset("file_a", total_frames=1000, chunk_frames=200)
            self.assertGreaterEqual(offset, 0)
            self.assertLess(offset, 800)

    def test_coverage_biases_unseen(self):
        """After many samples, bins with fewer hits should be favored."""
        from sidestep_engine.data.chunk_sampler import CoverageChunkSampler

        sampler = CoverageChunkSampler(n_bins=4, decay_every=0)
        # Manually saturate bin 0
        hist = sampler._get_or_create("bias_test")
        hist[0] = 1000

        bin_hits = [0] * 4
        for _ in range(200):
            offset = sampler.sample_offset("bias_test", total_frames=400, chunk_frames=100)
            bin_idx = min(3, int(offset / 75))
            bin_hits[bin_idx] += 1

        # Bin 0 should be much less likely than others
        self.assertLess(bin_hits[0], sum(bin_hits[1:]))

    def test_decay_halves_counts(self):
        """notify_epoch at decay boundary should halve histogram."""
        from sidestep_engine.data.chunk_sampler import CoverageChunkSampler

        sampler = CoverageChunkSampler(n_bins=4, decay_every=5)
        hist = sampler._get_or_create("decay_test")
        hist[:] = 100

        sampler.notify_epoch(5)
        self.assertEqual(hist[0].item(), 50)

    def test_state_dict_roundtrip(self):
        """state_dict / load_state_dict preserves coverage histograms."""
        from sidestep_engine.data.chunk_sampler import CoverageChunkSampler

        sampler = CoverageChunkSampler(n_bins=8, decay_every=10)
        for _ in range(20):
            sampler.sample_offset("file_1", 500, 100)
        state = sampler.state_dict()

        sampler2 = CoverageChunkSampler(n_bins=8, decay_every=10)
        sampler2.load_state_dict(state)

        self.assertEqual(
            sampler._histograms["file_1"].tolist(),
            sampler2._histograms["file_1"].tolist(),
        )

    def test_backward_compatible_load(self):
        """Loading empty or missing state should not crash."""
        from sidestep_engine.data.chunk_sampler import CoverageChunkSampler

        sampler = CoverageChunkSampler()
        sampler.load_state_dict({})  # empty state
        sampler.load_state_dict(None)  # type: ignore  # malformed
        self.assertEqual(len(sampler._histograms), 0)

    def test_short_sample_returns_zero(self):
        """When total_frames <= chunk_frames, offset should be 0."""
        from sidestep_engine.data.chunk_sampler import CoverageChunkSampler

        sampler = CoverageChunkSampler()
        self.assertEqual(sampler.sample_offset("short", 100, 100), 0)
        self.assertEqual(sampler.sample_offset("short", 50, 100), 0)


if __name__ == "__main__":
    unittest.main()
