"""HTTP lyrics provider backed by a configurable transcription server."""

from __future__ import annotations

import json
import logging
import os
import re
from collections.abc import Iterable
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from sidestep_engine.data.http_utils import build_multipart, validate_http_url

logger = logging.getLogger(__name__)
_DEFAULT_TIMEOUT_S = 180.0
_HEADER_ONLY_RE = re.compile(r"^\[[^\]]+\]$")
_TERMINAL_RE = re.compile(r"[.!?…:;)](?:[\"']+)?$")


def _resolve_endpoint(server_url: str) -> str:
    base = str(server_url or "").strip().rstrip("/")
    if not base:
        return ""
    tail = base.rsplit("/", 1)[-1].lower()
    if tail in {"transcribe", "lyrics", "predict", "infer"}:
        return validate_http_url(base)
    return validate_http_url(base + "/transcribe")




def _post_requests(endpoint: str, audio_path: str, fields: dict[str, str], timeout_s: float) -> dict[str, Any]:
    import requests

    endpoint = validate_http_url(endpoint)
    with open(audio_path, "rb") as f:
        files = {"file": (os.path.basename(audio_path), f, "application/octet-stream")}
        resp = requests.post(endpoint, files=files, data=fields, timeout=timeout_s)
        resp.raise_for_status()
        return resp.json()


def _post_urllib(endpoint: str, audio_path: str, fields: dict[str, str], timeout_s: float) -> dict[str, Any]:
    endpoint = validate_http_url(endpoint)
    body, boundary = build_multipart(fields, "file", audio_path)
    req = Request(endpoint, method="POST", data=body, headers={"Content-Type": f"multipart/form-data; boundary={boundary}"})
    with urlopen(req, timeout=timeout_s) as resp:
        return json.loads(resp.read().decode("utf-8", errors="replace"))


def _normalize_spaces(text: str) -> str:
    return re.sub(r"\s+", " ", str(text or "")).strip()


def _normalize_text(text: str) -> str:
    lines = [line.rstrip() for line in str(text or "").replace("\r\n", "\n").replace("\r", "\n").split("\n")]
    cleaned: list[str] = []
    blank = False
    for line in lines:
        s = line.strip()
        if not s:
            if cleaned and not blank:
                cleaned.append("")
            blank = True
            continue
        if _HEADER_ONLY_RE.fullmatch(s):
            continue
        cleaned.append(s)
        blank = False
    return "\n".join(cleaned).strip()


def _coerce_json_string(value: str) -> Any:
    s = str(value or "").strip()
    if not s:
        return ""
    if s[:1] not in "[{":
        return s
    try:
        return json.loads(s)
    except Exception:
        return s


def _segment_gap(seg: dict[str, Any]) -> float:
    try:
        start = float(seg.get("start")) if seg.get("start") is not None else None
        end = float(seg.get("end")) if seg.get("end") is not None else None
        if start is None or end is None:
            return 0.0
        return max(0.0, end - start)
    except Exception:
        return 0.0


def _iter_segments(segments: Any) -> Iterable[dict[str, Any]]:
    if not isinstance(segments, list):
        return []
    out: list[dict[str, Any]] = []
    for seg in segments:
        if isinstance(seg, str):
            s = _normalize_spaces(seg)
            if s:
                out.append({"text": s})
            continue
        if isinstance(seg, dict):
            for key in ("text", "lyrics", "content", "line", "transcript"):
                val = seg.get(key)
                if isinstance(val, str) and val.strip():
                    out.append({"text": _normalize_spaces(val), "start": seg.get("start"), "end": seg.get("end")})
                    break
    return out


def _flush_phrase(buf: list[str], out: list[str]) -> None:
    if not buf:
        return
    line = _normalize_spaces(" ".join(buf))
    if line:
        out.append(line)
    buf.clear()


def _join_segment_lines(segments: Iterable[dict[str, Any]]) -> str:
    rendered: list[str] = []
    buf: list[str] = []
    wc = 0
    prev_end = None

    for seg in list(segments):
        text = _normalize_spaces(seg.get("text") or "")
        if not text or _HEADER_ONLY_RE.fullmatch(text):
            continue

        start = seg.get("start")
        gap = None
        if prev_end is not None and start is not None:
            try:
                gap = float(start) - float(prev_end)
            except Exception:
                gap = None

        words = len(text.split())
        should_break = False
        if gap is not None and gap >= 1.6:
            should_break = True
        elif gap is not None and gap >= 0.85 and wc >= 4:
            should_break = True
        elif wc >= 8:
            should_break = True
        elif _TERMINAL_RE.search(" ".join(buf)):
            should_break = True

        if should_break and buf:
            _flush_phrase(buf, rendered)
            wc = 0
            if gap is not None and gap >= 1.6 and rendered and rendered[-1] != "":
                rendered.append("")

        buf.append(text)
        wc += words
        prev_end = seg.get("end") if seg.get("end") is not None else prev_end

        if _TERMINAL_RE.search(text) or wc >= 10:
            _flush_phrase(buf, rendered)
            wc = 0

    _flush_phrase(buf, rendered)
    return _normalize_text("\n".join(rendered))



def _reflow_text_block(text: str) -> str:
    norm = _normalize_text(text)
    if not norm:
        return ""

    paras = norm.split("\n\n")
    out: list[str] = []
    for para in paras:
        lines = [ln.strip() for ln in para.split("\n") if ln.strip()]
        if not lines:
            continue

        # Preserve genuine line breaks whenever the source already looks lyric-like.
        short_lines = sum(1 for ln in lines if len(ln.split()) <= 8)
        if len(lines) >= 3 and short_lines >= max(2, len(lines) // 2):
            out.append("\n".join(lines))
            continue

        # For prose-like blocks, wrap softly into lyric-ish lines instead of one giant slab.
        words = _normalize_spaces(" ".join(lines)).split()
        if not words:
            continue
        wrapped: list[str] = []
        chunk: list[str] = []
        for word in words:
            chunk.append(word)
            joined = " ".join(chunk)
            if len(chunk) >= 8 or _TERMINAL_RE.search(joined):
                wrapped.append(joined)
                chunk = []
        if chunk:
            wrapped.append(" ".join(chunk))
        out.append("\n".join(wrapped))

    return _normalize_text("\n\n".join(out))


def _extract_payload_text(payload: Any) -> str:
    if payload is None:
        return ""
    if isinstance(payload, str):
        parsed = _coerce_json_string(payload)
        if parsed != payload:
            return _extract_payload_text(parsed)
        return _reflow_text_block(payload)
    if isinstance(payload, list):
        return _join_segment_lines(_iter_segments(payload))
    if not isinstance(payload, dict):
        return ""

    for key in ("lyrics", "text", "transcript", "result", "output"):
        val = payload.get(key)
        if isinstance(val, str) and val.strip():
            parsed = _coerce_json_string(val)
            if parsed != val:
                text = _extract_payload_text(parsed)
                if text:
                    return text
            return _reflow_text_block(val)

    for key in ("segments", "lyrics_segments", "transcript_segments"):
        text = _join_segment_lines(_iter_segments(payload.get(key)))
        if text:
            return text

    for key in ("data", "prediction", "response"):
        if key in payload:
            text = _extract_payload_text(payload[key])
            if text:
                return text
    return ""


def fetch_lyrics_from_server(
    audio_path: str,
    *,
    server_url: str,
    title: str = "",
    artist: str = "",
    timeout_s: float = _DEFAULT_TIMEOUT_S,
) -> str:
    endpoint = _resolve_endpoint(server_url)
    if not endpoint:
        raise ValueError("Transcriber Server URL is not configured")
    fields = {"mode": "lyrics"}
    if title:
        fields["title"] = title
    if artist:
        fields["artist"] = artist
    try:
        try:
            payload = _post_requests(endpoint, audio_path, fields, timeout_s)
        except ImportError as exc:
            logger.debug("requests unavailable, falling back to urllib: %s", exc)
            payload = _post_urllib(endpoint, audio_path, fields, timeout_s)
    except HTTPError as exc:
        details = exc.read().decode("utf-8", errors="replace") if hasattr(exc, "read") else str(exc)
        raise RuntimeError(f"HTTP {getattr(exc, 'code', '?')}: {details}") from exc
    except URLError as exc:
        raise RuntimeError(str(exc.reason or exc)) from exc
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Malformed JSON response: {exc}") from exc
    text = _extract_payload_text(payload)
    if not text:
        raise RuntimeError("Server returned no usable lyrics")
    return text
