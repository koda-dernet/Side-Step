"""Tests for shared caption configuration module."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from sidestep_engine.data.caption_config import (
    DEFAULT_GEMINI_MODEL,
    DEFAULT_MAX_TOKENS,
    DEFAULT_OPENAI_MODEL,
    DEFAULT_TEMPERATURE,
    build_user_prompt,
    get_system_prompt,
    parse_structured_response,
)

_CFG = "sidestep_engine.data.caption_config"


class TestGetSystemPrompt(unittest.TestCase):
    """Test system prompt loading with file override."""

    def test_default_prompt_content(self):
        """Default system prompt contains key instructions."""
        prompt = get_system_prompt()
        self.assertIn("Do NOT include the artist name", prompt)
        self.assertIn("caption", prompt.lower())

    @patch(f"{_CFG}._prompt_override_path")
    def test_loads_custom_prompt_from_file(self, mock_path):
        """Custom prompt file overrides the default."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("Custom prompt text for testing")
            f.flush()
            mock_path.return_value = Path(f.name)
            result = get_system_prompt()
        self.assertEqual(result, "Custom prompt text for testing")
        Path(f.name).unlink()

    @patch(f"{_CFG}._prompt_override_path")
    def test_empty_file_uses_default(self, mock_path):
        """Empty override file falls back to default."""
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("")
            f.flush()
            mock_path.return_value = Path(f.name)
            result = get_system_prompt()
        self.assertIn("Do NOT include the artist name", result)
        Path(f.name).unlink()

    @patch(f"{_CFG}._prompt_override_path")
    def test_missing_file_uses_default(self, mock_path):
        """Non-existent file falls back to default."""
        mock_path.return_value = Path("/tmp/nonexistent_caption_prompt_xyz.txt")
        result = get_system_prompt()
        self.assertIn("Do NOT include the artist name", result)


class TestBuildUserPrompt(unittest.TestCase):
    """Test user prompt construction."""

    def test_basic_prompt(self):
        """Prompt includes title and artist."""
        prompt = build_user_prompt("My Song", "My Artist")
        self.assertIn("Title: My Song", prompt)
        self.assertIn("Artist: My Artist", prompt)

    def test_with_lyrics(self):
        """Lyrics excerpt is appended when provided."""
        prompt = build_user_prompt("Song", "Artist", "some lyrics here")
        self.assertIn("some lyrics here", prompt)
        self.assertIn("Lyrics excerpt:", prompt)

    def test_without_lyrics(self):
        """No lyrics section when excerpt is empty."""
        prompt = build_user_prompt("Song", "Artist", "")
        self.assertNotIn("Lyrics excerpt:", prompt)

    def test_lyrics_truncated_at_500(self):
        """Lyrics are truncated to 500 characters."""
        long_lyrics = "x" * 1000
        prompt = build_user_prompt("Song", "Artist", long_lyrics)
        # The lyrics in the prompt should be at most 500 chars
        lyrics_part = prompt.split("Lyrics excerpt:\n")[1]
        self.assertEqual(len(lyrics_part), 500)


class TestDefaults(unittest.TestCase):
    """Test default generation parameters."""

    def test_temperature_is_float(self):
        """Temperature default is a float."""
        self.assertIsInstance(DEFAULT_TEMPERATURE, float)
        self.assertGreater(DEFAULT_TEMPERATURE, 0)
        self.assertLessEqual(DEFAULT_TEMPERATURE, 1.0)

    def test_max_tokens_is_int(self):
        """Max tokens default is a positive integer."""
        self.assertIsInstance(DEFAULT_MAX_TOKENS, int)
        self.assertGreater(DEFAULT_MAX_TOKENS, 0)

    def test_default_model_names_are_strings(self):
        """Default model names are non-empty strings."""
        self.assertIsInstance(DEFAULT_GEMINI_MODEL, str)
        self.assertTrue(len(DEFAULT_GEMINI_MODEL) > 0)
        self.assertIsInstance(DEFAULT_OPENAI_MODEL, str)
        self.assertTrue(len(DEFAULT_OPENAI_MODEL) > 0)


class TestParseStructuredResponse(unittest.TestCase):
    """Test structured AI response parsing."""

    def test_full_structured_response(self):
        """All five fields are extracted correctly."""
        text = (
            "caption: A dreamy lo-fi track with warm pads\n"
            "genre: ambient, lo-fi\n"
            "bpm: 85\n"
            "key: C minor\n"
            "signature: 4/4"
        )
        result = parse_structured_response(text)
        self.assertEqual(result["caption"], "A dreamy lo-fi track with warm pads")
        self.assertEqual(result["genre"], "ambient, lo-fi")
        self.assertEqual(result["bpm"], "85")
        self.assertEqual(result["key"], "C minor")
        self.assertEqual(result["signature"], "4/4")

    def test_plain_text_fallback(self):
        """Plain text without key: value lines → treated as caption."""
        text = "A dreamy lo-fi ambient track with warm pads and soft rain"
        result = parse_structured_response(text)
        self.assertEqual(result["caption"], text)
        self.assertEqual(len(result), 1)

    def test_na_values_omitted(self):
        """Fields with 'N/A' are excluded from the result."""
        text = (
            "caption: Bright pop track\n"
            "genre: pop\n"
            "bpm: N/A\n"
            "key: n/a\n"
            "signature: 4/4"
        )
        result = parse_structured_response(text)
        self.assertNotIn("bpm", result)
        self.assertNotIn("key", result)
        self.assertEqual(result["caption"], "Bright pop track")
        self.assertEqual(result["signature"], "4/4")

    def test_partial_response(self):
        """Only some fields present → only those returned."""
        text = "caption: A mellow jazz tune\ngenre: jazz"
        result = parse_structured_response(text)
        self.assertEqual(result["caption"], "A mellow jazz tune")
        self.assertEqual(result["genre"], "jazz")
        self.assertNotIn("bpm", result)

    def test_empty_input(self):
        """Empty string returns empty dict."""
        self.assertEqual(parse_structured_response(""), {})

    def test_ignores_unknown_keys(self):
        """Keys not in _STRUCTURED_KEYS are ignored."""
        text = "caption: A track\nfoo: bar\ngenre: rock"
        result = parse_structured_response(text)
        self.assertNotIn("foo", result)
        self.assertEqual(result["genre"], "rock")

    def test_whitespace_handling(self):
        """Leading/trailing whitespace on keys and values is stripped."""
        text = "  caption :  Warm synth pads  \n  bpm : 120  "
        result = parse_structured_response(text)
        self.assertEqual(result["caption"], "Warm synth pads")
        self.assertEqual(result["bpm"], "120")


if __name__ == "__main__":
    unittest.main()
