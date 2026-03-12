"""Music Flamingo lyrics provider (experimental)."""

from __future__ import annotations

import json
from urllib.error import HTTPError, URLError

from sidestep_engine.data.http_utils import validate_http_url
from sidestep_engine.data.lyrics_provider_server import _extract_payload_text
from sidestep_engine.data.metadata_provider_music_flamingo import (
    _auth_headers,
    _call_gradio_http,
    _call_local_music_flamingo,
    _http_post_multipart,
    _looks_like_gradio_space,
    _looks_like_local_server,
    _normalize_root_url,
    _resolve_generic_endpoint,
)

_DEFAULT_TIMEOUT_S = 240.0
_DEFAULT_PROMPT = "Return only the song lyrics or transcript text. No JSON. No labels. No commentary."


def fetch_lyrics_from_music_flamingo(
    audio_path: str,
    *,
    server_url: str,
    title: str = "",
    artist: str = "",
    prompt: str = _DEFAULT_PROMPT,
    timeout_s: float = _DEFAULT_TIMEOUT_S,
    hf_token: str | None = None,
) -> str:
    root_url = _normalize_root_url(server_url)
    if not root_url:
        raise ValueError("Music Flamingo URL is not configured")
    full_prompt = prompt
    if artist or title:
        hints = []
        if artist:
            hints.append(f"artist={artist}")
        if title:
            hints.append(f"title={title}")
        full_prompt = f"{prompt} {'; '.join(hints)}"
    try:
        if _looks_like_local_server(root_url):
            raw = _call_local_music_flamingo(root_url, audio_path, full_prompt, timeout_s, hf_token=hf_token)
        elif _looks_like_gradio_space(root_url):
            raw = _call_gradio_http(root_url, audio_path, full_prompt, timeout_s, hf_token=hf_token)
        else:
            endpoint = validate_http_url(_resolve_generic_endpoint(root_url))
            headers = _auth_headers(hf_token, endpoint)
            fields = {"prompt": full_prompt, "mode": "lyrics", "return_json": "false"}
            raw = _http_post_multipart(endpoint, "file", audio_path, fields, timeout_s, headers=headers)
    except HTTPError as exc:
        details = exc.read().decode("utf-8", errors="replace") if hasattr(exc, "read") else str(exc)
        raise RuntimeError(f"HTTP {getattr(exc, 'code', '?')}: {details}") from exc
    except URLError as exc:
        raise RuntimeError(str(exc.reason or exc)) from exc
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Malformed JSON response: {exc}") from exc

    text = _extract_payload_text(raw).strip()
    if not text:
        raise RuntimeError("Music Flamingo returned no usable lyrics")
    return text
