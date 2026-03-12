"""Helpers for sanitizing structured provider payloads."""

from __future__ import annotations

import ast
import json
from typing import Any

from sidestep_engine.data.caption_config import parse_structured_response


def looks_like_mapping_blob(value: Any) -> bool:
    s = str(value or "").strip()
    if not s:
        return False
    low = s.lower()
    return (
        (s.startswith("{") and ("'caption'" in s or '"caption"' in s or "'ok'" in s or '"ok"' in s))
        or low.startswith("caption: {")
    )


def extract_caption_from_blob(value: Any) -> str:
    if isinstance(value, dict):
        payload = value
    else:
        s = str(value or "").strip()
        if not s:
            return ""
        if s.lower().startswith("caption:"):
            s = s.split(":", 1)[1].strip()
        try:
            payload = json.loads(s)
        except Exception:
            try:
                payload = ast.literal_eval(s)
            except Exception:
                payload = None
        if not isinstance(payload, dict):
            structured = parse_structured_response(s)
            caption = str(structured.get("caption") or "").strip()
            return "" if looks_like_mapping_blob(caption) else caption
    if not isinstance(payload, dict):
        return ""
    structured = parse_structured_response(payload)
    caption = str(structured.get("caption") or "").strip()
    return "" if looks_like_mapping_blob(caption) else caption
