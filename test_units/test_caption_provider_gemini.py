"""Tests for Gemini caption provider (mocked google.genai client)."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

from sidestep_engine.data.caption_provider_gemini import (
    _safe_extract_text,
    _simplify_error,
    generate_caption,
)

_MOD = "sidestep_engine.data.caption_provider_gemini"


def _make_mock_response(
    text: str = "A caption", finish_reason_name: str = "STOP",
    empty_parts: bool = False,
) -> MagicMock:
    """Create a mock GenerateContentResponse with candidates/parts structure."""
    response = MagicMock()
    candidate = MagicMock()
    fr = MagicMock()
    fr.name = finish_reason_name
    candidate.finish_reason = fr
    if empty_parts:
        candidate.content.parts = []
    else:
        part = MagicMock()
        part.text = text
        candidate.content.parts = [part]
    candidate.safety_ratings = []
    response.candidates = [candidate]
    if empty_parts:
        response.text = property(lambda s: (_ for _ in ()).throw(ValueError("no parts")))
    else:
        response.text = text
    return response


def _make_mock_client(response_text: str = "A caption") -> MagicMock:
    """Create a mock google.genai.Client with configured response."""
    client = MagicMock()
    mock_response = _make_mock_response(response_text)
    client.models.generate_content.return_value = mock_response
    return client


class TestGenerateCaption(unittest.TestCase):
    """Test generate_caption with mocked google.genai client."""

    @patch(f"{_MOD}._make_client")
    def test_success(self, mock_mc):
        """Successful generation returns caption text."""
        mock_mc.return_value = _make_mock_client(
            "Ethereal electronic track with glitchy synths"
        )
        result = generate_caption("Song", "Artist", "fake-key", max_retries=1)
        self.assertEqual(result, "Ethereal electronic track with glitchy synths")

    @patch(f"{_MOD}._make_client")
    def test_empty_response_returns_none(self, mock_mc):
        """Empty API response returns None."""
        mock_mc.return_value = _make_mock_client("")
        result = generate_caption("Song", "Artist", "fake-key", max_retries=1)
        self.assertIsNone(result)

    @patch(f"{_MOD}._make_client")
    def test_prompt_includes_lyrics_excerpt(self, mock_mc):
        """When lyrics_excerpt is provided, it's included in the prompt."""
        client = _make_mock_client("A caption")
        mock_mc.return_value = client
        generate_caption(
            "Song", "Artist", "fake-key",
            lyrics_excerpt="Hello world lyrics", max_retries=1,
        )
        call_kwargs = client.models.generate_content.call_args
        contents = call_kwargs.kwargs.get("contents") or call_kwargs[1].get("contents")
        prompt_text = contents[-1] if isinstance(contents, list) else contents
        self.assertIn("Hello world lyrics", prompt_text)

    @patch(f"{_MOD}._make_client")
    def test_audio_upload_called_when_path_provided(self, mock_mc):
        """Audio file is uploaded when audio_path points to a real file."""
        client = _make_mock_client("Caption with audio")
        mock_mc.return_value = client
        mock_file = MagicMock()
        mock_file.state = None  # no PROCESSING state
        client.files.upload.return_value = mock_file

        with tempfile.NamedTemporaryFile(suffix=".mp3") as f:
            f.write(b"\x00" * 100)
            f.flush()
            result = generate_caption(
                "Song", "Artist", "fake-key",
                audio_path=Path(f.name), max_retries=1,
            )
        self.assertEqual(result, "Caption with audio")
        client.files.upload.assert_called_once()
        call_kwargs = client.models.generate_content.call_args
        contents = call_kwargs.kwargs.get("contents") or call_kwargs[1].get("contents")
        self.assertEqual(len(contents), 2)
        self.assertIs(contents[0], mock_file)

    @patch(f"{_MOD}._make_client")
    def test_no_audio_upload_when_path_none(self, mock_mc):
        """No upload attempted when audio_path is None."""
        client = _make_mock_client("Text-only caption")
        mock_mc.return_value = client
        generate_caption("Song", "Artist", "fake-key", max_retries=1)
        client.files.upload.assert_not_called()

    @patch(f"{_MOD}._make_client")
    def test_uploaded_file_deleted_after_use(self, mock_mc):
        """Uploaded file is cleaned up after caption generation."""
        client = _make_mock_client("ok")
        mock_mc.return_value = client
        mock_file = MagicMock()
        mock_file.state = None
        client.files.upload.return_value = mock_file

        with tempfile.NamedTemporaryFile(suffix=".mp3") as f:
            f.write(b"\x00" * 100)
            f.flush()
            generate_caption(
                "Song", "Artist", "fake-key",
                audio_path=Path(f.name), max_retries=1,
            )
        client.files.delete.assert_called_once_with(name=mock_file.name)

    @patch(f"{_MOD}.time")
    @patch(f"{_MOD}._make_client")
    def test_upload_timeout_raises(self, mock_mc, mock_time):
        """Upload polling times out if processing exceeds deadline."""
        client = _make_mock_client("ok")
        mock_mc.return_value = client
        mock_file = MagicMock()
        mock_file.state.name = "PROCESSING"
        client.files.upload.return_value = mock_file
        client.files.get.return_value = mock_file
        mock_time.monotonic.side_effect = [0, 0, 200]
        mock_time.sleep = MagicMock()

        with tempfile.NamedTemporaryFile(suffix=".mp3") as f:
            f.write(b"\x00" * 100)
            f.flush()
            result = generate_caption(
                "Song", "Artist", "fake-key",
                audio_path=Path(f.name), max_retries=1,
            )
        # Caption still generated (text-only fallback)
        self.assertEqual(result, "ok")

    def test_shared_system_prompt_used(self):
        """Provider uses the shared caption_config system prompt."""
        from sidestep_engine.data.caption_config import get_system_prompt
        prompt = get_system_prompt()
        self.assertIn("Do NOT include the artist name", prompt)

    @patch(f"{_MOD}._make_client")
    def test_validate_key_uses_list_models(self, mock_mc):
        """validate_key uses client.models.list() for auth check."""
        from sidestep_engine.data.caption_provider_gemini import validate_key
        client = MagicMock()
        mock_mc.return_value = client
        client.models.list.return_value = [MagicMock()]
        result = validate_key("good-key")
        self.assertTrue(result)
        client.models.list.assert_called_once()

    @patch(f"{_MOD}._make_client")
    def test_validate_key_returns_false_on_error(self, mock_mc):
        """validate_key returns False when list raises."""
        from sidestep_engine.data.caption_provider_gemini import validate_key
        client = MagicMock()
        mock_mc.return_value = client
        client.models.list.side_effect = Exception("bad key")
        result = validate_key("fake-key")
        self.assertFalse(result)


class TestSafeExtractText(unittest.TestCase):
    """Regression: _safe_extract_text must not crash on MAX_TOKENS/SAFETY."""

    def test_normal_response(self):
        """Normal response with text parts returns concatenated text."""
        resp = _make_mock_response("Hello world")
        self.assertEqual(_safe_extract_text(resp), "Hello world")

    def test_empty_parts_returns_empty(self):
        """MAX_TOKENS with empty parts returns empty string (not ValueError)."""
        resp = _make_mock_response(empty_parts=True, finish_reason_name="MAX_TOKENS")
        result = _safe_extract_text(resp)
        self.assertEqual(result, "")

    def test_no_candidates_returns_empty(self):
        """Response with no candidates returns empty string."""
        resp = MagicMock()
        resp.candidates = []
        self.assertEqual(_safe_extract_text(resp), "")


class TestFinishReasonChecking(unittest.TestCase):
    """Regression: truncated Gemini responses must be logged."""

    @patch(f"{_MOD}._make_client")
    def test_max_tokens_still_returns_partial_text(self, mock_mc):
        """MAX_TOKENS with partial text: return it (don't discard)."""
        client = MagicMock()
        mock_mc.return_value = client
        resp = _make_mock_response("Partial caption", finish_reason_name="MAX_TOKENS")
        client.models.generate_content.return_value = resp
        result = generate_caption("Song", "Artist", "fake-key", max_retries=1)
        self.assertEqual(result, "Partial caption")


class TestSimplifyError(unittest.TestCase):
    """_simplify_error condenses verbose Gemini errors."""

    def test_quota_exceeded(self):
        """Quota exceeded error returns one-liner with docs link."""
        exc = Exception(
            "Quota exceeded for metric: generativelanguage.googleapis.com/"
            "generate_content_free_tier_input_token_count, limit: 0, "
            "model: gemini-3-pro\n[links { description: \"Learn more\" }]"
        )
        result = _simplify_error(exc)
        self.assertIn("quota exceeded", result.lower())
        self.assertIn("rate-limits", result)
        self.assertLess(len(result), 120)

    def test_rate_limit(self):
        """Rate limit / 429 error returns one-liner."""
        exc = Exception("429 Resource Exhausted: too many requests")
        result = _simplify_error(exc)
        self.assertIn("rate limit", result.lower())

    def test_short_error_unchanged(self):
        """Short errors pass through unchanged."""
        exc = Exception("Connection refused")
        self.assertEqual(_simplify_error(exc), "Connection refused")

    def test_long_error_truncated(self):
        """Errors > 120 chars are truncated with ellipsis."""
        exc = Exception("A" * 200)
        result = _simplify_error(exc)
        self.assertEqual(len(result), 121)  # 120 + '\u2026'
        self.assertTrue(result.endswith("\u2026"))


if __name__ == "__main__":
    unittest.main()
