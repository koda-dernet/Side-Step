"""Tests for audio duration detection and auto-max-duration."""

from __future__ import annotations

import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch


class TestGetAudioDuration(unittest.TestCase):
    """Verify get_audio_duration chains torchcodec -> soundfile -> 0."""

    @patch("sidestep_engine.data.audio_duration.logger")
    def test_returns_int_from_soundfile(self, _mock_log):
        """Duration from soundfile is truncated to int."""
        from sidestep_engine.data.audio_duration import get_audio_duration

        mock_info = MagicMock()
        mock_info.duration = 187.654

        with patch.dict("sys.modules", {"torchcodec": None, "torchcodec.decoders": None}):
            with patch("soundfile.info", return_value=mock_info):
                result = get_audio_duration("/fake/song.wav")

        self.assertIsInstance(result, int)
        self.assertEqual(result, 187)

    def test_returns_zero_on_all_failures(self):
        """Returns 0 when both backends fail."""
        from sidestep_engine.data.audio_duration import get_audio_duration

        with patch.dict("sys.modules", {"torchcodec": None, "torchcodec.decoders": None}):
            with patch("soundfile.info", side_effect=RuntimeError("no file")):
                result = get_audio_duration("/nonexistent/file.wav")

        self.assertEqual(result, 0)


class TestDetectMaxDuration(unittest.TestCase):
    """Verify detect_max_duration picks the longest clip."""

    def test_empty_list_returns_zero(self):
        """Empty file list should return 0."""
        from sidestep_engine.data.audio_duration import detect_max_duration

        self.assertEqual(detect_max_duration([]), 0)

    def test_picks_longest(self):
        """Should return the maximum duration across files."""
        from sidestep_engine.data.audio_duration import detect_max_duration

        durations = {"/a.wav": 60, "/b.wav": 180, "/c.wav": 120}

        def mock_dur(path):
            return durations.get(path, 0)

        with patch("sidestep_engine.data.audio_duration.get_audio_duration", side_effect=mock_dur):
            result = detect_max_duration([Path(p) for p in durations])

        self.assertEqual(result, 180)

    def test_all_failures_returns_zero(self):
        """When every probe fails (returns 0), detect returns 0."""
        from sidestep_engine.data.audio_duration import detect_max_duration

        with patch("sidestep_engine.data.audio_duration.get_audio_duration", return_value=0):
            result = detect_max_duration([Path("/a.wav"), Path("/b.wav")])

        self.assertEqual(result, 0)


if __name__ == "__main__":
    unittest.main()
