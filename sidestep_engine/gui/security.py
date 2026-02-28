"""
Security middleware and helpers for the Side-Step GUI server.

Defenses:
    1. **Bearer token** — random 32-byte hex token generated per session.
       Required on all ``/api/`` and ``/ws/`` requests via ``Authorization``
       header or ``?token=`` query param.
    2. **Host header validation** — rejects requests where the ``Host``
       header isn't ``127.0.0.1:<port>`` or ``localhost:<port>``.
       Defeats DNS rebinding attacks.
    3. **API key masking** — ``mask_keys()`` replaces sensitive values
       in settings dicts with ``"••••<last4>"`` for safe display.
"""

from __future__ import annotations

import logging
import secrets
from typing import Any, Dict, Set

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse, Response
from starlette.websockets import WebSocket


def generate_token() -> str:
    """Generate a cryptographically random session token."""
    return secrets.token_hex(32)


# ---------------------------------------------------------------------------
# Bearer-token auth middleware
# ---------------------------------------------------------------------------

# Paths that do NOT require auth (static assets, index page)
_PUBLIC_PREFIXES = ("/css/", "/js/", "/favicon")


def _is_protected(path: str) -> bool:
    """Return True if the path requires token auth."""
    if path == "/":
        return False
    for prefix in _PUBLIC_PREFIXES:
        if path.startswith(prefix):
            return False
    return True


def _extract_token(scope: dict) -> str | None:
    """Extract token from Authorization header or ?token= query param."""
    # Check headers first
    headers = dict(scope.get("headers", []))
    auth = headers.get(b"authorization", b"").decode("latin-1", errors="ignore")
    if auth.startswith("Bearer "):
        return auth[7:]

    # Fall back to query param
    qs = scope.get("query_string", b"").decode("latin-1", errors="ignore")
    for part in qs.split("&"):
        if part.startswith("token="):
            return part[6:]
    return None


class TokenAuthMiddleware(BaseHTTPMiddleware):
    """Reject /api/ and /ws/ requests without a valid bearer token."""

    def __init__(self, app, token: str) -> None:
        super().__init__(app)
        self._token = token

    async def dispatch(self, request: Request, call_next) -> Response:
        if _is_protected(request.url.path):
            provided = _extract_token(request.scope)
            if provided != self._token:
                logging.getLogger(__name__).warning(
                    "[TokenAuth] 401 path=%s provided=%s",
                    request.url.path,
                    "yes" if provided else "no",
                )
                return JSONResponse(
                    {"error": "Unauthorized"},
                    status_code=401,
                )
        return await call_next(request)


class TokenAuthWSMiddleware:
    """ASGI middleware that also validates WebSocket connections."""

    def __init__(self, app, token: str) -> None:
        self.app = app
        self._token = token

    async def __call__(self, scope, receive, send) -> None:
        if scope["type"] == "websocket" and _is_protected(scope.get("path", "")):
            provided = _extract_token(scope)
            if provided != self._token:
                ws = WebSocket(scope, receive, send)
                await ws.close(code=4401)
                return
        await self.app(scope, receive, send)


# ---------------------------------------------------------------------------
# Host header validation middleware
# ---------------------------------------------------------------------------

class HostValidationMiddleware(BaseHTTPMiddleware):
    """Reject requests where Host header isn't a known localhost variant."""

    def __init__(self, app, port: int = 8770) -> None:
        super().__init__(app)
        self._allowed: Set[str] = {
            f"127.0.0.1:{port}",
            f"localhost:{port}",
            f"[::1]:{port}",
            "127.0.0.1",
            "localhost",
            "[::1]",
        }

    async def dispatch(self, request: Request, call_next) -> Response:
        host = request.headers.get("host", "")
        if host not in self._allowed:
            logging.getLogger(__name__).warning(
                "[HostValidation] 403 path=%s host=%r",
                request.url.path,
                host,
            )
            return JSONResponse(
                {"error": "Forbidden: invalid Host header"},
                status_code=403,
            )
        return await call_next(request)


# ---------------------------------------------------------------------------
# API key masking
# ---------------------------------------------------------------------------

_SENSITIVE_KEYS = {"gemini_api_key", "openai_api_key", "genius_api_token"}
_SENSITIVE_PATTERNS = ("_api_key", "_api_token", "_secret")


def _is_sensitive(key: str) -> bool:
    """Check if a settings key should be masked (explicit set + pattern match)."""
    if key in _SENSITIVE_KEYS:
        return True
    kl = key.lower()
    return any(kl.endswith(p) for p in _SENSITIVE_PATTERNS)


def mask_keys(settings: Dict[str, Any]) -> Dict[str, Any]:
    """Return a copy of settings with sensitive keys masked for display."""
    out = dict(settings)
    for key in list(out.keys()):
        if not _is_sensitive(key):
            continue
        val = out[key]
        if val and isinstance(val, str) and len(val) > 4:
            out[key] = "••••" + val[-4:]
        elif val:
            out[key] = "••••"
    return out
