"""Tests for OpenAI-compatible caption provider (mocked API calls)."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from sidestep_engine.data.caption_provider_openai import (
    _MAX_AUDIO_SIZE_MB,
    generate_caption,
)

_MOD = "sidestep_engine.data.caption_provider_openai"


def _make_mock_openai(
    response_text: str = "A caption",
    finish_reason: str = "stop",
    content_none: bool = False,
    refusal: str | None = None,
) -> MagicMock:
    """Create a mock openai module with configured response."""
    mock_openai = MagicMock()
    mock_message = MagicMock()
    mock_message.content = None if content_none else response_text
    mock_message.refusal = refusal
    mock_choice = MagicMock()
    mock_choice.message = mock_message
    mock_choice.finish_reason = finish_reason
    mock_response = MagicMock()
    mock_response.choices = [mock_choice]
    mock_openai.OpenAI.return_value.chat.completions.create.return_value = mock_response
    return mock_openai


class TestGenerateCaption(unittest.TestCase):
    """Test generate_caption with mocked openai."""

    @patch(f"{_MOD}._get_openai")
    def test_success(self, mock_get):
        """Successful generation returns caption text."""
        mock_get.return_value = _make_mock_openai(
            "Ethereal electronic track with glitchy synths"
        )
        result = generate_caption("Song", "Artist", "fake-key", max_retries=1)
        self.assertEqual(result, "Ethereal electronic track with glitchy synths")

    @patch(f"{_MOD}._get_openai")
    def test_empty_response_returns_none(self, mock_get):
        """Empty API response returns None."""
        mock_get.return_value = _make_mock_openai("")
        result = generate_caption("Song", "Artist", "fake-key", max_retries=1)
        self.assertIsNone(result)

    @patch(f"{_MOD}._get_openai")
    def test_custom_base_url(self, mock_get):
        """Custom base_url is passed to the OpenAI client."""
        mock_openai = _make_mock_openai("A caption")
        mock_get.return_value = mock_openai
        generate_caption(
            "Song", "Artist", "fake-key",
            base_url="https://api.mistral.ai/v1", max_retries=1,
        )
        call_kwargs = mock_openai.OpenAI.call_args[1]
        self.assertEqual(call_kwargs["base_url"], "https://api.mistral.ai/v1")

    @patch(f"{_MOD}._get_openai")
    def test_custom_model(self, mock_get):
        """Custom model name is passed to create()."""
        mock_openai = _make_mock_openai("A caption")
        mock_get.return_value = mock_openai
        generate_caption(
            "Song", "Artist", "fake-key",
            model="mistral-large-latest", max_retries=1,
        )
        create_kwargs = (
            mock_openai.OpenAI.return_value.chat.completions.create.call_args[1]
        )
        self.assertEqual(create_kwargs["model"], "mistral-large-latest")

    @patch(f"{_MOD}._get_openai")
    def test_prompt_includes_lyrics(self, mock_get):
        """Lyrics excerpt is included in the user message."""
        mock_openai = _make_mock_openai("A caption")
        mock_get.return_value = mock_openai
        generate_caption(
            "Song", "Artist", "fake-key",
            lyrics_excerpt="Hello world lyrics", max_retries=1,
        )
        create_kwargs = (
            mock_openai.OpenAI.return_value.chat.completions.create.call_args[1]
        )
        user_msg = create_kwargs["messages"][1]["content"]
        self.assertIn("Hello world lyrics", user_msg)

    @patch(f"{_MOD}._get_openai")
    def test_audio_included_as_base64(self, mock_get):
        """Audio file is base64-encoded and included in multimodal content."""
        mock_openai = _make_mock_openai("Caption with audio")
        mock_get.return_value = mock_openai

        with tempfile.NamedTemporaryFile(suffix=".mp3") as f:
            f.write(b"\x00" * 100)
            f.flush()
            generate_caption(
                "Song", "Artist", "fake-key",
                audio_path=Path(f.name), max_retries=1,
            )
        create_kwargs = (
            mock_openai.OpenAI.return_value.chat.completions.create.call_args[1]
        )
        user_content = create_kwargs["messages"][1]["content"]
        # Multimodal: content is a list with text + audio parts
        self.assertIsInstance(user_content, list)
        self.assertEqual(user_content[0]["type"], "text")
        self.assertEqual(user_content[1]["type"], "input_audio")
        self.assertEqual(user_content[1]["input_audio"]["format"], "mp3")

    @patch(f"{_MOD}._get_openai")
    def test_no_audio_when_path_none(self, mock_get):
        """Without audio_path, user content is plain text."""
        mock_openai = _make_mock_openai("Text-only caption")
        mock_get.return_value = mock_openai
        generate_caption("Song", "Artist", "fake-key", max_retries=1)
        create_kwargs = (
            mock_openai.OpenAI.return_value.chat.completions.create.call_args[1]
        )
        user_content = create_kwargs["messages"][1]["content"]
        self.assertIsInstance(user_content, str)

    def test_shared_system_prompt_used(self):
        """Provider uses the shared caption_config system prompt."""
        from sidestep_engine.data.caption_config import get_system_prompt
        prompt = get_system_prompt()
        self.assertIn("Do NOT include the artist name", prompt)


class TestSafeContentExtraction(unittest.TestCase):
    """Regression: content=None must not crash with AttributeError."""

    @patch(f"{_MOD}._get_openai")
    def test_none_content_returns_none(self, mock_get):
        """When message.content is None, return None (not crash)."""
        mock_get.return_value = _make_mock_openai(content_none=True)
        result = generate_caption("Song", "Artist", "fake-key", max_retries=1)
        self.assertIsNone(result)

    @patch(f"{_MOD}._get_openai")
    def test_refusal_logged_returns_none(self, mock_get):
        """When model refuses, return None (not crash)."""
        mock_get.return_value = _make_mock_openai(
            content_none=True, refusal="Content policy violation",
        )
        result = generate_caption("Song", "Artist", "fake-key", max_retries=1)
        self.assertIsNone(result)


class TestFinishReasonChecking(unittest.TestCase):
    """Regression: truncated OpenAI responses must still return text."""

    @patch(f"{_MOD}._get_openai")
    def test_length_finish_returns_partial(self, mock_get):
        """finish_reason=length: return partial text (don't discard)."""
        mock_get.return_value = _make_mock_openai(
            "Partial caption text", finish_reason="length",
        )
        result = generate_caption("Song", "Artist", "fake-key", max_retries=1)
        self.assertEqual(result, "Partial caption text")


if __name__ == "__main__":
    unittest.main()
