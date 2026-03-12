"""Shared HTTP helpers for provider integrations."""

from __future__ import annotations

import uuid
from pathlib import Path
from urllib.parse import urlparse


def validate_http_url(url: str) -> str:
    value = str(url or "").strip()
    parsed = urlparse(value)
    if parsed.scheme not in {"http", "https"}:
        raise ValueError(f"Unsupported URL scheme: {parsed.scheme or '<missing>'}")
    if not parsed.netloc:
        raise ValueError("URL must include a host")
    return value


def build_multipart(fields: dict[str, str], file_field_name: str, file_path: str) -> tuple[bytes, str]:
    boundary = "----sidestep-" + uuid.uuid4().hex
    filename = Path(file_path).name
    with open(file_path, "rb") as f:
        data = f.read()
    parts: list[bytes] = []
    for k, v in (fields or {}).items():
        parts.append(f"--{boundary}\r\n".encode())
        parts.append(f'Content-Disposition: form-data; name="{k}"\r\n\r\n'.encode())
        parts.append(str(v).encode())
        parts.append(b"\r\n")
    parts.append(f"--{boundary}\r\n".encode())
    parts.append((
        f'Content-Disposition: form-data; name="{file_field_name}"; filename="{filename}"\r\n'
        "Content-Type: application/octet-stream\r\n\r\n"
    ).encode())
    parts.append(data)
    parts.append(b"\r\n")
    parts.append(f"--{boundary}--\r\n".encode())
    return b"".join(parts), boundary
