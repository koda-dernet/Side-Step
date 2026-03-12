"""Music Flamingo metadata provider.

Supports two transport styles:
- generic HTTP endpoints that accept multipart file uploads
- Hugging Face / Gradio Spaces over the public HTTP API, without gradio_client
"""

from __future__ import annotations

import ast
import http.cookiejar
import ipaddress
import json
import logging
import os
import re
import uuid
from itertools import islice
from pathlib import Path
from typing import Any, Dict, Optional, Tuple
from urllib.error import HTTPError, URLError
from urllib.parse import urlparse
from urllib.request import HTTPCookieProcessor, Request, build_opener, urlopen

from sidestep_engine.data.http_utils import build_multipart, validate_http_url

logger = logging.getLogger(__name__)

_DEFAULT_TIMEOUT_S = 240.0
_DEFAULT_PROMPT = "Return one JSON object with: caption, genres, bpm, key_scale, timesignature, vocal_language, is_instrumental. No extra text."
_DEFAULT_PROMPT_RICH = "Return one JSON object with: caption, genres, bpm, key_scale, timesignature, vocal_language, is_instrumental. Caption must describe only the sound, groove, instrumentation, and vocal character of this song. Do not mention the song title or artist. No extra text."
_DEFAULT_PROMPT_STRUCTURED_RETRY = "Return one JSON object with: caption, genres, bpm, key_scale, timesignature, vocal_language, is_instrumental. Include explicit numeric bpm and explicit key_scale, timesignature, and vocal_language when detectable. No extra text."
_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)
_HF_SPACE_RE = re.compile(r"https?://huggingface\.co/spaces/([^/]+)/([^/?#]+)", re.I)


def _auth_headers(hf_token: str | None, target_url: str | None = None) -> Dict[str, str]:
    token = str(hf_token or "").strip()
    headers = {
        "Accept": "application/json, text/event-stream, text/plain;q=0.9, */*;q=0.8",
        "User-Agent": "Side-Step Music Flamingo/1.0",
    }
    parsed = urlparse(str(target_url or "").strip())
    host = (parsed.hostname or "").lower()
    if parsed.scheme and parsed.netloc:
        origin = f"{parsed.scheme}://{parsed.netloc}"
        headers["Origin"] = origin
        headers["Referer"] = origin + "/"
    is_hf_target = host.endswith("hf.space") or host.endswith("huggingface.co")
    if token and is_hf_target:
        bearer = f"Bearer {token}"
        headers["Authorization"] = bearer
        headers["X-HF-Authorization"] = bearer
    return headers




def _build_cookie_opener():
    jar = http.cookiejar.CookieJar()
    return build_opener(HTTPCookieProcessor(jar))


def _open_request(req: Request, timeout_s: float, opener=None):
    validate_http_url(req.full_url)
    if opener is not None:
        return opener.open(req, timeout=timeout_s)
    return urlopen(req, timeout=timeout_s)

def _normalize_root_url(server_url: str) -> str:
    return str(server_url or "").strip().rstrip("/")


def _space_subdomain_url(server_url: str) -> str:
    base = str(server_url or "").strip().rstrip("/")
    if not base:
        return ""
    match = _HF_SPACE_RE.match(base)
    if match:
        return f"https://{match.group(1)}-{match.group(2)}.hf.space"
    if "/spaces/" in base and base.startswith("http"):
        parsed = urlparse(base)
        parts = [p for p in parsed.path.split("/") if p]
        if len(parts) >= 3 and parts[0] == "spaces":
            return f"{parsed.scheme}://{parts[1]}-{parts[2]}.hf.space"
    return base


def _looks_like_gradio_space(root_url: str) -> bool:
    low = (root_url or "").lower()
    return low.endswith(".hf.space") or "/gradio_api" in low or "/spaces/" in low


def _looks_like_local_server(root_url: str) -> bool:
    try:
        parsed = urlparse(str(root_url or "").strip())
        host = (parsed.hostname or "").strip().lower()
    except Exception:
        host = ""
    if not host:
        return False
    if host in {"localhost", "127.0.0.1", "::1", "0.0.0.0", "host.docker.internal"}:
        return True
    try:
        ip = ipaddress.ip_address(host)
        return bool(ip.is_private or ip.is_loopback)
    except Exception:
        return False


def _resolve_local_caption_endpoint(server_url: str) -> str:
    base = _normalize_root_url(server_url)
    if not base:
        return ""
    tail = base.rsplit("/", 1)[-1].lower()
    if tail in {"caption", "analyze", "infer", "predict"}:
        return base
    return base + "/caption"


def _resolve_generic_endpoint(server_url: str) -> str:
    base = _normalize_root_url(server_url)
    if not base:
        return ""
    tail = base.rsplit("/", 1)[-1].lower()
    if tail in {"infer", "predict", "analyze", "metadata", "metas"}:
        return base
    return base + "/infer"




def _http_get_json(url: str, timeout_s: float, headers: Optional[Dict[str, str]] = None, opener=None) -> Dict[str, Any]:
    req = Request(url, method="GET", headers=headers or {})
    with _open_request(req, timeout_s, opener=opener) as resp:
        raw = resp.read().decode("utf-8", errors="replace")
    return json.loads(raw)


def _http_post_json(url: str, payload: Dict[str, Any], timeout_s: float, headers: Optional[Dict[str, str]] = None, opener=None) -> Any:
    hdrs = {"Content-Type": "application/json"}
    if headers:
        hdrs.update(headers)
    req = Request(url, method="POST", data=json.dumps(payload).encode("utf-8"), headers=hdrs)
    with _open_request(req, timeout_s, opener=opener) as resp:
        raw = resp.read().decode("utf-8", errors="replace")
    try:
        return json.loads(raw)
    except Exception:
        return raw


def _http_post_multipart(
    url: str,
    file_field_name: str,
    file_path: str,
    fields: Dict[str, str],
    timeout_s: float,
    headers: Optional[Dict[str, str]] = None,
    opener=None,
) -> Any:
    url = validate_http_url(url)
    body, boundary = build_multipart(fields, file_field_name, file_path)
    hdrs = {"Content-Type": f"multipart/form-data; boundary={boundary}"}
    if headers:
        hdrs.update(headers)
    req = Request(url, method="POST", data=body, headers=hdrs)
    with _open_request(req, timeout_s, opener=opener) as resp:
        raw = resp.read().decode("utf-8", errors="replace")
    try:
        return json.loads(raw)
    except Exception:
        return raw


def _candidate_root_urls(root_url: str) -> list[str]:
    base = (root_url or "").strip().rstrip("/")
    if not base:
        return []
    alt = _space_subdomain_url(base)
    seen: set[str] = set()
    urls: list[str] = []
    preferred = []
    if alt and alt != base:
        preferred.append(alt)
    preferred.append(base)
    # For Hugging Face Spaces, prefer the embedded Gradio runtime only.
    if "/spaces/" in base:
        preferred = [u for u in preferred if u.endswith('.hf.space')]
    for url in preferred:
        u = (url or "").strip().rstrip("/")
        if u and u not in seen:
            seen.add(u)
            urls.append(u)
    return urls


def _candidate_config_urls(root_url: str) -> list[str]:
    base = root_url.rstrip("/")
    urls = [base + "/config"]
    if not base.endswith("/gradio_api"):
        urls.append(base + "/gradio_api/config")
    return urls


def _load_gradio_config(root_url: str, timeout_s: float, headers: Optional[Dict[str, str]] = None, opener=None) -> Tuple[str, Dict[str, Any]]:
    errors: list[str] = []
    for candidate_root in _candidate_root_urls(root_url):
        for url in _candidate_config_urls(candidate_root):
            try:
                cfg = _http_get_json(url, min(timeout_s, 30.0), headers=headers, opener=opener)
                if isinstance(cfg, dict) and cfg.get("dependencies"):
                    return candidate_root, cfg
            except Exception as exc:
                errors.append(f"{url}: {exc}")
    raise RuntimeError("Music Flamingo config discovery failed: " + " | ".join(errors[-4:]))


def _component_by_id(config: Dict[str, Any], component_id: int) -> Dict[str, Any]:
    for comp in config.get("components") or []:
        if comp.get("id") == component_id:
            return comp
    return {}


def _score_dependency(config: Dict[str, Any], dep: Dict[str, Any]) -> int:
    if not dep.get("backend_fn"):
        return -1000
    inputs = dep.get("inputs") or []
    outputs = dep.get("outputs") or []
    score = 0
    if len(inputs) == 3:
        score += 20
    if len(outputs) == 1:
        score += 10
    input_types = [(_component_by_id(config, cid).get("type") or "").lower() for cid in inputs]
    output_types = [(_component_by_id(config, cid).get("type") or "").lower() for cid in outputs]
    if any(t == "audio" for t in input_types):
        score += 30
    score += 8 * sum(1 for t in input_types if t == "textbox")
    if any(t == "textbox" for t in output_types):
        score += 8
    api_name = dep.get("api_name")
    if isinstance(api_name, str):
        if api_name.strip("/").lower() in {"predict", "infer"}:
            score += 10
        else:
            score += 4
    if dep.get("queue", True):
        score += 3
    return score


def _resolve_gradio_endpoint(config: Dict[str, Any]) -> Tuple[int, str]:
    deps = config.get("dependencies") or []
    if not deps:
        raise RuntimeError("Music Flamingo config has no dependencies")
    best_index = 0
    best_score = -10**9
    best_dep: Dict[str, Any] = {}
    for idx, dep in enumerate(deps):
        score = _score_dependency(config, dep)
        if score > best_score:
            best_score = score
            best_index = dep.get("id", idx)
            best_dep = dep
    api_name = best_dep.get("api_name")
    if isinstance(api_name, str) and api_name.strip():
        endpoint = "/" + api_name.strip("/")
    else:
        endpoint = "/predict"
    return int(best_index), endpoint


def _src_prefixed(root_url: str, config: Dict[str, Any]) -> str:
    base = root_url.rstrip("/")
    api_prefix = str(config.get("api_prefix") or "").strip("/")
    if api_prefix:
        return base + "/" + api_prefix
    return base


def _coerce_uploaded_audio_variants(upload_data: Dict[str, Any]) -> list[Any]:
    path = str((upload_data or {}).get("path") or "").strip()
    variants: list[Any] = []
    if upload_data:
        canonical = dict(upload_data)
        canonical.setdefault("meta", {"_type": "gradio.FileData"})
        canonical.setdefault("is_stream", False)
        variants.append(canonical)
    if path:
        variants.append({"path": path, "meta": {"_type": "gradio.FileData"}, "is_stream": False})
        variants.append(path)
    out: list[Any] = []
    seen: set[str] = set()
    for v in variants:
        key = repr(v)
        if key not in seen:
            seen.add(key)
            out.append(v)
    return out


def _call_gradio_run(src_prefixed: str, api_name: str, upload_data: Dict[str, Any], prompt: str, timeout_s: float, headers: Optional[Dict[str, str]] = None, opener=None) -> Any:
    run_url = src_prefixed.rstrip("/") + "/run" + api_name
    errors: list[str] = []
    for audio_value in _coerce_uploaded_audio_variants(upload_data):
        payloads = [
            {"audio_path": audio_value, "youtube_url": "", "prompt_text": prompt},
            {"data": [audio_value, "", prompt]},
        ]
        for payload in payloads:
            try:
                response = _http_post_json(run_url, payload, timeout_s, headers=headers, opener=opener)
            except Exception as exc:
                errors.append(str(exc))
                continue
            if isinstance(response, dict):
                if response.get("error") not in (None, "", []):
                    errors.append(str(response.get("error")))
                    continue
                if "output" in response:
                    return response.get("output")
                data = response.get("data")
                if data not in (None, "", []):
                    return data
            elif response not in (None, "", []):
                return response
    raise RuntimeError("Music Flamingo /run failed: " + " | ".join(errors[-4:]))


def _call_gradio_call(src_prefixed: str, api_name: str, upload_data: Dict[str, Any], prompt: str, timeout_s: float, headers: Optional[Dict[str, str]] = None, opener=None) -> Any:
    call_url = src_prefixed.rstrip("/") + "/call" + api_name
    errors: list[str] = []
    for audio_value in _coerce_uploaded_audio_variants(upload_data):
        payload = {"data": [audio_value, "", prompt]}
        try:
            event = _http_post_json(call_url, payload, min(timeout_s, 60.0), headers=headers, opener=opener)
        except Exception as exc:
            errors.append(str(exc))
            continue
        if not (isinstance(event, dict) and event.get("event_id")):
            errors.append(f"unexpected call response: {event}")
            continue
        try:
            result = _read_sse_call_result(call_url + "/" + str(event["event_id"]), timeout_s, headers=headers, opener=opener)
            if result not in (None, "", []):
                return result
            errors.append("empty SSE result")
        except Exception as exc:
            errors.append(str(exc))
            continue
    raise RuntimeError("Music Flamingo /call failed: " + " | ".join(errors[-6:]))


def _upload_gradio_file(src_prefixed: str, audio_path: str, timeout_s: float, headers: Optional[Dict[str, str]] = None, opener=None) -> Dict[str, Any]:
    upload_url = src_prefixed.rstrip("/") + "/upload"
    payload = _http_post_multipart(upload_url, "files", audio_path, {}, min(timeout_s, 120.0), headers=headers, opener=opener)
    first: Any = None
    if isinstance(payload, list) and payload:
        first = payload[0]
    elif isinstance(payload, dict):
        first = payload
    if not first:
        raise RuntimeError(f"Unexpected Music Flamingo upload response: {payload}")

    if isinstance(first, str) and first.strip():
        first = {"path": first.strip()}
    if not isinstance(first, dict) or not str(first.get("path") or "").strip():
        raise RuntimeError(f"Unexpected Music Flamingo upload response: {payload}")

    out = dict(first)
    path = str(out.get("path") or "").strip()
    file_name = Path(audio_path).name
    suffix = Path(audio_path).suffix.lower()
    mime = {
        ".wav": "audio/wav",
        ".mp3": "audio/mpeg",
        ".flac": "audio/flac",
        ".ogg": "audio/ogg",
        ".m4a": "audio/mp4",
    }.get(suffix, "application/octet-stream")
    out.setdefault("orig_name", file_name)
    out.setdefault("size", os.path.getsize(audio_path) if os.path.exists(audio_path) else None)
    out.setdefault("mime_type", mime)
    out.setdefault("is_stream", False)
    out.setdefault("meta", {"_type": "gradio.FileData"})
    if not out.get("url") and path:
        out["url"] = src_prefixed.rstrip("/") + "/file=" + path
    return out


def _read_queue_stream(data_url: str, session_hash: str, event_id: str, timeout_s: float, headers: Optional[Dict[str, str]] = None, opener=None) -> Any:
    stream_url = f"{data_url}?session_hash={session_hash}"
    req = Request(stream_url, method="GET", headers=headers or {})
    with _open_request(req, timeout_s, opener=opener) as resp:
        current_data: Optional[str] = None
        for raw_line in resp:
            line = raw_line.decode("utf-8", errors="replace").strip()
            if not line or not line.startswith("data:"):
                continue
            data = line[5:].strip()
            try:
                msg = json.loads(data)
            except Exception:
                continue
            if msg.get("msg") == "heartbeat":
                continue
            if msg.get("event_id") != event_id:
                continue
            if msg.get("msg") == "process_completed":
                output = msg.get("output") or {}
                if not msg.get("success", True):
                    raise RuntimeError(str(output.get("error") or output))
                return output.get("data", [])
            if msg.get("msg") == "server_stopped":
                raise RuntimeError("Music Flamingo server stopped")
            current_data = data
    raise RuntimeError(f"No Music Flamingo result returned (last_event={current_data})")


def _parse_sse_payload(payload_lines: list[str]) -> Any:
    payload = "\n".join(payload_lines).strip()
    if not payload:
        return ""
    if payload == "null":
        return None
    try:
        return json.loads(payload)
    except Exception:
        return payload


def _extract_sse_result_value(parsed: Any) -> Any:
    if parsed in (None, "", [], {}):
        return None
    if isinstance(parsed, dict):
        for key in ("output", "data", "value", "result"):
            if key in parsed:
                value = parsed.get(key)
                if value not in (None, "", [], {}):
                    return value
        if parsed.get("success") is False:
            err = parsed.get("error") or parsed
            raise RuntimeError(str(err))
        return parsed
    return parsed


def _read_sse_call_result(url: str, timeout_s: float, headers: Optional[Dict[str, str]] = None, opener=None) -> Any:
    req_headers = dict(headers or {})
    req_headers.setdefault("Accept", "text/event-stream")
    req = Request(url, method="GET", headers=req_headers)
    last_payload: Any = None
    last_event_name = ""
    event_name = "message"
    data_lines: list[str] = []

    def flush_event() -> tuple[str, Any] | None:
        nonlocal event_name, data_lines, last_event_name
        if not data_lines and not event_name:
            return None
        parsed = _parse_sse_payload(data_lines)
        name = event_name or "message"
        last_event_name = name
        event_name = "message"
        data_lines = []
        return name, parsed

    with _open_request(req, timeout_s, opener=opener) as resp:
        for raw_line in resp:
            line = raw_line.decode("utf-8", errors="replace").rstrip("\r\n")
            if line == "":
                flushed = flush_event()
                if not flushed:
                    continue
                name, parsed = flushed
                value = _extract_sse_result_value(parsed)
                if name in {"generating", "complete", "message"} and value not in (None, "", []):
                    last_payload = value
                if name == "complete":
                    return value if value not in (None, "", []) else last_payload
                if name == "error":
                    if value not in (None, "", []):
                        raise RuntimeError(str(value))
                    if last_payload not in (None, "", []):
                        return last_payload
                    raise RuntimeError(f"Music Flamingo SSE call failed without details (last_event={last_event_name or 'error'})")
                continue
            if line.startswith(":"):
                continue
            if line.startswith("event:"):
                event_name = line.split(":", 1)[1].strip() or "message"
                continue
            if line.startswith("data:"):
                data_lines.append(line.split(":", 1)[1].lstrip())
                continue
        flushed = flush_event()
        if flushed:
            name, parsed = flushed
            value = _extract_sse_result_value(parsed)
            if name == "complete" and value not in (None, "", []):
                return value
            if value not in (None, "", []):
                last_payload = value
    if last_payload not in (None, "", []):
        return last_payload
    raise RuntimeError(f"No result returned from Music Flamingo /call (last_event={last_event_name or 'none'})")


def _call_gradio_http(root_url: str, audio_path: str, prompt: str, timeout_s: float, hf_token: str | None = None) -> Any:
    opener = _build_cookie_opener()
    initial_headers = _auth_headers(hf_token, root_url)
    resolved_root, config = _load_gradio_config(root_url, timeout_s, headers=initial_headers, opener=opener)
    src_prefixed = _src_prefixed(resolved_root, config)
    headers = _auth_headers(hf_token, src_prefixed)
    _fn_index, api_name = _resolve_gradio_endpoint(config)
    upload_data = _upload_gradio_file(src_prefixed, audio_path, timeout_s, headers=headers, opener=opener)
    # Only use the verified hf.space + /gradio_api/call/<api> flow.
    return _call_gradio_call(src_prefixed, api_name, upload_data, prompt, timeout_s, headers=headers, opener=opener)


def _normalize_text_payload(out: Any) -> str:
    if out is None:
        return ""
    if isinstance(out, str):
        return out
    if isinstance(out, (list, tuple)):
        strings = [x.strip() for x in out if isinstance(x, str) and x.strip()]
        if strings:
            return max(strings, key=len)
        nested = [
            _normalize_text_payload(x)
            for x in out
            if isinstance(x, (dict, list, tuple))
        ]
        nested = [x for x in nested if x.strip()]
        if nested:
            return max(nested, key=len)
        return ""
    if isinstance(out, dict):
        for key in ("text", "output", "response", "result", "value"):
            val = out.get(key)
            if isinstance(val, str) and val.strip():
                return val
        for key in ("output", "data", "result", "value"):
            val = out.get(key)
            if isinstance(val, (list, tuple, dict)):
                nested = _normalize_text_payload(val)
                if nested.strip():
                    return nested
    return str(out)




def _strip_ui_noise(text: str) -> str:
    lines = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            lines.append("")
            continue
        low = line.lower()
        if low.startswith("✅ using audio file") or low.startswith("using audio file"):
            continue
        if low.startswith("uploaded file:") or low.startswith("audio file:"):
            continue
        lines.append(line)
    text = "\n".join(lines)
    text = re.sub(r"\n{3,}", "\n\n", text).strip()
    return text


def _extract_keyed_string_loose(text: str, keys: list[str]) -> str:
    s = str(text or "")
    if not s:
        return ""
    for key in keys:
        patterns = [
            rf'"{re.escape(key)}"\s*:\s*"(.*?)(?="\s*,\s*"[A-Za-z_][A-Za-z0-9_ -]*"\s*:|"\s*[}}\]]|$)',
            rf'{re.escape(key)}\s*:\s*"(.*?)(?="\s*,\s*[A-Za-z_][A-Za-z0-9_ -]*\s*:|"\s*[}}\]]|$)',
        ]
        for pattern in patterns:
            m = re.search(pattern, s, re.I | re.S)
            if not m:
                continue
            val = m.group(1).strip()
            if "\\" in val:
                try:
                    val = json.loads(f'"{val}"').strip()
                except Exception:
                    pass
            val = val.rstrip('"').strip()
            if val:
                return val
    return ""


def _looks_generic_caption(text: str) -> bool:
    s = _sentenceish_caption(text).strip().lower()
    if not s:
        return True
    generic_patterns = (
        r'^[a-z&/+ -]+ track\.?$',
        r'^music track\.?$',
        r'^audio track\.?$',
        r'^[a-z&/+ -]+ song\.?$',
        r'^this track is a [a-z0-9 ,\-]+ piece that blends [^.]+\.?$',
        r'^this track is a [a-z0-9 ,\-]+ track that blends [^.]+\.?$',
        r'^this track blends [^.]+\.?$',
    )
    if any(re.fullmatch(p, s) for p in generic_patterns):
        return True
    boilerplate_prefixes = (
        'this track is a high-energy',
        'this track is a vibrant',
        'this track is an energetic',
        'this track blends',
    )
    return s.startswith(boilerplate_prefixes)




def _track_identity_from_path(audio_path: str) -> tuple[str, str]:
    stem = Path(str(audio_path or '')).stem.strip()
    if not stem:
        return '', ''
    parts = [x.strip() for x in stem.split(' - ', 1)]
    if len(parts) == 2:
        return parts[0], parts[1]
    return stem, ''


def _specific_caption_from_identity(fields: Dict[str, str], audio_path: str) -> str:
    title, artist = _track_identity_from_path(audio_path)
    genre = str(fields.get('genre') or '').strip()
    bpm = str(fields.get('bpm') or '').strip()
    key = str(fields.get('key') or '').strip()
    signature = str(fields.get('signature') or '').strip()
    language = str(fields.get('language') or '').strip()

    subject = f'"{title}"' if title else 'This track'
    if artist:
        subject += f' by {artist}'

    details = []
    if genre:
        details.append(f'is a {genre} song')
    else:
        details.append('is a song')
    if bpm:
        details.append(f'at around {bpm} BPM')
    if key:
        details.append(f'in {key}')
    if signature:
        details.append(f'with a {signature} feel')
    if language:
        details.append(f'and {language} vocals')

    caption = subject + ' ' + ' '.join(details)
    return _sentenceish_caption(caption.strip().rstrip('.') + '.')
def _sentenceish_caption(text: str, max_words: int = 30) -> str:
    cleaned = _strip_ui_noise(text)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    if not cleaned:
        return ""
    cleaned = re.sub(r"^(caption|description)\s*[:=-]\s*", "", cleaned, flags=re.I).strip()
    first = re.split(r"(?<=[.!?])\s+|\n+", cleaned, maxsplit=1)[0].strip()
    words = first.split()
    if len(words) > max_words:
        first = " ".join(words[:max_words]).rstrip(",;:-") + "."
    return first

def _extract_json(text: str) -> Optional[Dict[str, Any]]:
    if not text:
        return None
    match = _JSON_RE.search(text)
    if not match:
        return None
    block = match.group(0).strip()
    for parser in (json.loads, ast.literal_eval):
        try:
            obj = parser(block)
            if isinstance(obj, dict):
                return obj
        except Exception:
            pass
    normalized = re.sub(r"([\{,]\s*)([A-Za-z_][A-Za-z0-9_]*)(\s*:)", r'\1"\2"\3', block)
    normalized = normalized.replace("None", "null").replace("True", "true").replace("False", "false")
    try:
        obj = json.loads(normalized)
        return obj if isinstance(obj, dict) else None
    except Exception:
        return None


def _looks_like_json_text(text: str) -> bool:
    s = (text or "").strip()
    if not s:
        return False
    return (s.startswith("{") and s.endswith("}")) or (s.startswith('"{') and s.endswith('}"'))


def _normalize_json_keys(obj: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    aliases = {
        "genre": "genres",
        "genres": "genres",
        "bpm": "bpm",
        "tempo": "bpm",
        "key": "key_scale",
        "keyscale": "key_scale",
        "key_scale": "key_scale",
        "timesignature": "timesignature",
        "time_signature": "timesignature",
        "time-signature": "timesignature",
        "signature": "timesignature",
        "language": "vocal_language",
        "lang": "vocal_language",
        "vocal_language": "vocal_language",
        "detected_language": "vocal_language",
        "caption": "caption",
        "description": "caption",
        "summary": "caption",
        "is_instrumental": "is_instrumental",
    }
    for key, value in (obj or {}).items():
        canon = aliases.get(str(key).strip().lower().replace(" ", "_").replace("-", "_"), key)
        out[canon] = value
    return out


def _synthesize_caption(fields: Dict[str, str], audio_path: str = "") -> str:
    genre = str(fields.get("genre") or "").strip()
    key = str(fields.get("key") or "").strip()
    bpm = str(fields.get("bpm") or "").strip()
    signature = str(fields.get("signature") or "").strip()
    language = str(fields.get("language") or "").strip()
    parts = []
    if genre:
        parts.append(f"{genre} track")
    else:
        parts.append("Music track")
    details = []
    if bpm:
        details.append(f"around {bpm} BPM")
    if key:
        details.append(f"in {key}")
    if signature:
        details.append(f"with a {signature} feel")
    if language:
        details.append(f"and {language} vocals")
    if details:
        return _sentenceish_caption(parts[0].capitalize() + " " + ", ".join(details) + ".")
    return _sentenceish_caption(parts[0].capitalize() + ".")

def _extract_metas_from_text(text: str) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    patterns = {
        "genres": [r"genres?\s*[:=-]\s*([^\n]+)"],
        "bpm": [
            r"\bbpm\s*[:=-]\s*([0-9]{2,3})",
            r"(?:around|about|at|tempo\s*(?:of)?|moves\s*at|sits\s*at)?\s*[≈~]?\s*([0-9]{2,3}(?:\.[0-9]+)?)\s*BPM\b",
        ],
        "key_scale": [
            r"key(?:[_\s-]*scale)?\s*[:=-]\s*([^\n,;]+)",
            r"\bkey\s+of\s+([A-G][#b]?\s*(?:major|minor))\b",
            r"\bcentered\s+in\s+([A-G][#b]?\s*(?:major|minor))\b",
            r"\bin\s+([A-G][#b]?\s*(?:major|minor))\b",
        ],
        "timesignature": [
            r"time\s*signature\s*[:=-]\s*([^\n,;]+)",
            r"timesignature\s*[:=-]\s*([^\n,;]+)",
            r"\b(2/4|3/4|4/4|5/4|6/8|7/8|9/8|12/8)\s*time\b",
            r"\b(2/4|3/4|4/4|5/4|6/8|7/8|9/8|12/8)\b",
        ],
        "caption": [r"caption\s*[:=-]\s*([^\n]+)", r"description\s*[:=-]\s*([^\n]+)"],
        "vocal_language": [
            r"(?:vocal_)?language\s*[:=-]\s*([^\n,;]+)",
            r"\b(Spanish|English|Italian|Portuguese|French|German)\s+vocals\b",
        ],
    }
    for key, regs in patterns.items():
        for rgx in regs:
            match = re.search(rgx, text, re.I)
            if match:
                out[key] = match.group(1).strip()
                break
    return out


def _as_int(x: Any) -> Optional[int]:
    if x is None or isinstance(x, bool):
        return None
    try:
        return int(float(str(x).strip()))
    except Exception:
        return None


def _clean_genres(v: Any) -> str:
    if v is None:
        return ""
    if isinstance(v, (list, tuple)):
        return ", ".join(str(x).strip() for x in v if str(x).strip())
    s = str(v).strip().strip("[](){}")
    s = s.replace('"', '').replace("'", "")
    return re.sub(r"\s*,\s*", ", ", s)




def _infer_genres_from_text(*values: Any) -> str:
    text = " ".join(str(v or "") for v in values)
    s = text.lower()
    if not s.strip():
        return ""

    checks: list[tuple[str, tuple[str, ...]]] = [
        ("Salsa", ("salsa",)),
        ("Timba", ("timba",)),
        ("Salsa Dura", ("salsa dura",)),
        ("Salsa Romántica", ("salsa romantica", "salsa romántica")),
        ("Son Cubano", ("son cubano",)),
        ("Afro-Cuban", ("afro-cuban", "afrocuban", "afro cuban")),
        ("Latin Jazz", ("latin jazz",)),
        ("Jazz Fusion", ("jazz fusion", "fusion jazz")),
        ("Latin Pop", ("latin pop",)),
        ("Big Band", ("big band",)),
        ("Jazz", (" jazz ",)),
        ("Latin", (" latin ",)),
        ("Pop", (" pop ",)),
        ("Rock", (" rock ",)),
        ("Hip-Hop", ("hip hop", "hip-hop", "rap")),
        ("R&B", ("r&b", "soul")),
        ("Electronic", ("electronic", "edm", "dance music")),
        ("House", (" house ", "deep house", "tech house")),
        ("Techno", (" techno ",)),
        ("Reggaeton", ("reggaeton",)),
        ("Bachata", ("bachata",)),
        ("Merengue", ("merengue",)),
        ("Cumbia", ("cumbia",)),
        ("Flamenco", ("flamenco",)),
        ("Folk", (" folk ",)),
        ("Classical", ("classical", "orchestral", "symphonic")),
    ]

    found: list[str] = []
    padded = f" {s} "
    for label, needles in checks:
        for needle in needles:
            if needle in padded or needle in s:
                if label not in found:
                    found.append(label)
                break

    percussion_markers = ("conga", "bongo", "timbales", "guiro", "güiro", "clave", "claves", "montuno")
    if "Salsa" not in found and sum(1 for tok in percussion_markers if tok in s) >= 2:
        found.insert(0, "Salsa")
    if "Latin" not in found and ("Salsa" in found or "Timba" in found or "Son Cubano" in found or "Latin Jazz" in found):
        insert_at = 1 if found and found[0] == "Salsa" else len(found)
        found.insert(insert_at, "Latin")

    return ", ".join(found[:4])

def _clean_timesig(x: Any) -> str:
    if x is None:
        return ""
    s = str(x).strip()
    if not s or s.lower() in {"na", "n/a", "none", "null", "unknown"}:
        return ""
    return s


def _clean_language(x: Any) -> str:
    if x is None:
        return ""
    s = str(x).strip()
    if not s or s.lower() in {"na", "n/a", "none", "null", "unknown"}:
        return ""
    return s


def _normalize_fields(obj: Dict[str, Any]) -> Dict[str, str]:
    obj = _normalize_json_keys(obj)
    result: Dict[str, str] = {}
    caption = obj.get("caption") or obj.get("description") or obj.get("summary")
    if isinstance(caption, str) and _looks_like_json_text(caption):
        nested = _extract_json(caption)
        if nested:
            obj = _normalize_json_keys({**obj, **nested})
            caption = obj.get("caption") or obj.get("description") or obj.get("summary")
    genre = _clean_genres(obj.get("genres") or obj.get("genre"))
    if not genre:
        genre = _infer_genres_from_text(caption, obj.get("lyrics"), obj.get("raw_text"))
    bpm = _as_int(obj.get("bpm") or obj.get("tempo"))
    key_scale = obj.get("key_scale") or obj.get("keyscale") or obj.get("key")
    signature = _clean_timesig(obj.get("timesignature") or obj.get("time_signature") or obj.get("signature"))
    language = _clean_language(obj.get("vocal_language") or obj.get("detected_language") or obj.get("language") or obj.get("lang"))
    if caption and str(caption).strip() and not _looks_like_json_text(str(caption).strip()):
        clean_caption = _sentenceish_caption(str(caption).strip())
        if clean_caption and not _looks_generic_caption(clean_caption):
            result["caption"] = clean_caption
    if genre:
        result["genre"] = genre
    if bpm is not None:
        result["bpm"] = str(bpm)
    if key_scale and str(key_scale).strip():
        result["key"] = str(key_scale).strip()
    if signature:
        result["signature"] = signature
    if language:
        result["language"] = language
    return {k: v for k, v in result.items() if str(v).strip()}


def _call_local_music_flamingo(root_url: str, audio_path: str, prompt: str, timeout_s: float, hf_token: str | None = None) -> Any:
    endpoint = _resolve_local_caption_endpoint(root_url)
    headers = _auth_headers(hf_token, endpoint)
    fields = {"prompt": prompt}
    return _http_post_multipart(endpoint, "file", audio_path, fields, timeout_s, headers=headers)


def _call_music_flamingo_transport(
    root_url: str,
    audio_path: str,
    prompt: str,
    timeout_s: float,
    hf_token: str | None = None,
) -> Any:
    if _looks_like_local_server(root_url):
        return _call_local_music_flamingo(root_url, audio_path, prompt, timeout_s, hf_token=hf_token)
    if _looks_like_gradio_space(root_url):
        return _call_gradio_http(root_url, audio_path, prompt, timeout_s, hf_token=hf_token)
    endpoint = validate_http_url(_resolve_generic_endpoint(root_url))
    headers = _auth_headers(hf_token, endpoint)
    fields = {"prompt": prompt, "mode": "metadata", "return_json": "true"}
    return _http_post_multipart(endpoint, "file", audio_path, fields, timeout_s, headers=headers)


def _parse_music_flamingo_payload(raw: Any, *, audio_path: str = "") -> tuple[Dict[str, str], str]:
    if isinstance(raw, dict):
        payload = _normalize_json_keys(raw)
        raw_text = _normalize_text_payload(raw)
    else:
        raw_text = _normalize_text_payload(raw)
        cleaned_text = _strip_ui_noise(raw_text)
        parsed_json = _extract_json(cleaned_text)
        if parsed_json:
            payload = _normalize_json_keys(parsed_json)
        else:
            payload = _extract_metas_from_text(cleaned_text) or {"caption": cleaned_text.strip()}

    fields = _normalize_fields(payload)
    cleaned_raw = _strip_ui_noise(raw_text)
    if raw_text and (
        not fields.get("genre")
        or not fields.get("bpm")
        or not fields.get("key")
        or not fields.get("signature")
        or not fields.get("language")
        or not fields.get("caption")
        or _looks_generic_caption(fields.get("caption", ""))
    ):
        text_meta = _normalize_fields(_extract_metas_from_text(cleaned_raw))
        for k, v in text_meta.items():
            if v and (not fields.get(k) or (k == "caption" and _looks_generic_caption(fields.get("caption", "")))):
                fields[k] = v

        loose_caption = _extract_keyed_string_loose(cleaned_raw, ["caption", "description", "summary"])
        if loose_caption and (not fields.get("caption") or _looks_generic_caption(fields.get("caption", ""))):
            fields["caption"] = _sentenceish_caption(loose_caption)
    if not fields.get("caption") or _looks_generic_caption(fields.get("caption", "")):
        if cleaned_raw:
            nested_from_raw = _extract_json(cleaned_raw)
            if nested_from_raw:
                nested_fields = _normalize_fields(nested_from_raw)
                for k, v in nested_fields.items():
                    if v and (not fields.get(k) or (k == "caption" and _looks_generic_caption(fields.get("caption", "")))):
                        fields[k] = v
        if not fields.get("caption") or _looks_generic_caption(fields.get("caption", "")):
            fallback_caption = _extract_keyed_string_loose(cleaned_raw, ["caption", "description", "summary"]) if cleaned_raw else ""
            if fallback_caption:
                fields["caption"] = _sentenceish_caption(fallback_caption)
        if not fields.get("caption") or _looks_generic_caption(fields.get("caption", "")):
            if cleaned_raw:
                fallback_caption = _sentenceish_caption(cleaned_raw)
                if fallback_caption and not _looks_like_json_text(fallback_caption):
                    fields["caption"] = fallback_caption
        if not fields.get("caption") or _looks_generic_caption(fields.get("caption", "")):
            rich_enough = any(fields.get(k) for k in ("bpm", "key", "signature", "language", "genre"))
            if fields and rich_enough:
                synth = _synthesize_caption(fields, audio_path=audio_path)
                if synth:
                    fields["caption"] = synth
    return fields, raw_text


def _structured_field_count(fields: Dict[str, str]) -> int:
    return sum(1 for key in ("bpm", "key", "signature", "language") if str((fields or {}).get(key) or "").strip())


def _merge_structured_fields(primary: Dict[str, str], secondary: Dict[str, str]) -> Dict[str, str]:
    merged = dict(primary or {})
    for key in ("bpm", "key", "signature", "language", "genre"):
        cand = str((secondary or {}).get(key) or "").strip()
        if cand and not str(merged.get(key) or "").strip():
            merged[key] = cand
    cur_caption = str(merged.get("caption") or "").strip()
    new_caption = str((secondary or {}).get("caption") or "").strip()
    if new_caption and ((not cur_caption) or _looks_generic_caption(cur_caption)):
        merged["caption"] = new_caption
    return {k: v for k, v in merged.items() if str(v).strip()}


def _merge_music_flamingo_fields(primary: Dict[str, str], secondary: Dict[str, str], *, audio_path: str = "") -> Dict[str, str]:
    merged = dict(primary or {})
    secondary = dict(secondary or {})

    # Keep stable structured fields from pass 1, but fill holes from pass 2.
    for key in ("bpm", "key", "signature", "language"):
        if not str(merged.get(key) or "").strip() and str(secondary.get(key) or "").strip():
            merged[key] = str(secondary.get(key)).strip()

    for key in ("caption", "genre"):
        cand = str(secondary.get(key) or "").strip()
        if not cand:
            continue
        if key == "caption":
            cur = str(merged.get("caption") or "").strip()
            if (not cur) or _looks_generic_caption(cur):
                merged[key] = cand
            elif len(cand) > len(cur) + 8 and not _looks_generic_caption(cand):
                merged[key] = cand
        else:
            cur = str(merged.get("genre") or "").strip()
            if not cur:
                merged[key] = cand
            else:
                cur_parts = [x.strip() for x in cur.split(',') if x.strip()]
                new_parts = [x.strip() for x in cand.split(',') if x.strip()]
                merged_parts = []
                for part in cur_parts + new_parts:
                    if part and part not in merged_parts:
                        merged_parts.append(part)
                merged[key] = ', '.join(merged_parts[:4])

    if (not merged.get("caption") or _looks_generic_caption(merged.get("caption", ""))) and merged:
        synth = _synthesize_caption(merged, audio_path=audio_path)
        if synth:
            merged["caption"] = synth
    return {k: v for k, v in merged.items() if str(v).strip()}


def fetch_music_flamingo_metadata(
    audio_path: str,
    *,
    server_url: str,
    prompt: str = _DEFAULT_PROMPT,
    timeout_s: float = _DEFAULT_TIMEOUT_S,
    hf_token: str | None = None,
) -> Dict[str, str]:
    root_url = _normalize_root_url(server_url)
    if not root_url:
        raise ValueError("Music Flamingo URL is not configured")
    try:
        raw = _call_music_flamingo_transport(root_url, audio_path, prompt, timeout_s, hf_token=hf_token)
    except HTTPError as exc:
        details = exc.read().decode("utf-8", errors="replace") if hasattr(exc, "read") else str(exc)
        hint = ""
        low = details.lower()
        if "quota" in low and "gpu" in low:
            hint = " (Music Flamingo reported Hugging Face/Space quota limits; auth may not have been accepted by the Space runtime)"
        raise RuntimeError(f"HTTP {getattr(exc, 'code', '?')}: {details}{hint}") from exc
    except URLError as exc:
        raise RuntimeError(str(exc.reason or exc)) from exc
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Malformed JSON response: {exc}") from exc

    fields, raw_text = _parse_music_flamingo_payload(raw, audio_path=audio_path)

    # Local server only: retry structured extraction when needed, then run a richer caption pass.
    # Remote/HF behavior stays unchanged.
    if _looks_like_local_server(root_url):
        primary_fields = dict(fields or {})
        best_structured = dict(primary_fields)
        best_raw_text = raw_text

        if str(prompt or "").strip() == _DEFAULT_PROMPT and _structured_field_count(best_structured) < 4:
            retry_prompt = _DEFAULT_PROMPT_STRUCTURED_RETRY
            try:
                retry_raw = _call_music_flamingo_transport(root_url, audio_path, retry_prompt, timeout_s, hf_token=hf_token)
                retry_fields, retry_raw_text = _parse_music_flamingo_payload(retry_raw, audio_path=audio_path)
                merged_retry = _merge_structured_fields(best_structured, retry_fields)
                if _structured_field_count(merged_retry) >= _structured_field_count(best_structured):
                    best_structured = merged_retry
                    if len(str(retry_raw_text or '').strip()) > len(str(best_raw_text or '').strip()):
                        best_raw_text = retry_raw_text
            except Exception as exc:
                logger.warning("Music Flamingo structured local retry failed; keeping primary metadata: %s", exc)

        fields = dict(best_structured)
        raw_text = best_raw_text

        rich_prompt = _DEFAULT_PROMPT_RICH if str(prompt or "").strip() == _DEFAULT_PROMPT else prompt
        if rich_prompt != prompt:
            try:
                rich_raw = _call_music_flamingo_transport(root_url, audio_path, rich_prompt, timeout_s, hf_token=hf_token)
                rich_fields, rich_raw_text = _parse_music_flamingo_payload(rich_raw, audio_path=audio_path)
                merged_fields = _merge_music_flamingo_fields(best_structured, rich_fields, audio_path=audio_path)
                if merged_fields:
                    fields = merged_fields
                if len(str(rich_raw_text or '').strip()) > len(str(raw_text or '').strip()):
                    raw_text = rich_raw_text
            except Exception as exc:
                fields = best_structured
                logger.warning("Music Flamingo secondary local caption pass failed; keeping structured metadata: %s", exc)
        else:
            fields = best_structured

    if not fields:
        logger.warning("Music Flamingo raw payload produced no usable fields: %r", raw)
        raise RuntimeError("Music Flamingo returned no usable metadata")
    return fields
