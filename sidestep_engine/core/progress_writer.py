"""
JSONL progress file for GUI telemetry (``{output_dir}/.progress.jsonl``).

**Per-step loss:** When ``maybe_write(..., step=...)`` is called with an
integer *step*, each **new** optimizer step writes a line immediately so
loss charts stay aligned with training steps. Fast runs (many steps per
second) are not collapsed to 1 Hz.

Calls **without** a numeric ``step`` still use time-only throttling (see
``interval``) for backward compatibility.
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Any, Optional


def sanitize_floats(obj: Any) -> Any:
    """Replace non-finite floats (NaN, Inf, -Inf) with JSON-safe values.

    This is critical because the JSONL file is consumed by the GUI via
    WebSocket → JavaScript ``JSON.parse()``, which rejects bare ``Infinity``
    and ``NaN`` tokens.  Python's ``json.dumps`` emits them by default
    (``allow_nan=True``), but they are **not valid JSON** per RFC 7159.
    """
    if isinstance(obj, float):
        if math.isinf(obj) or math.isnan(obj):
            return None
        return obj
    if isinstance(obj, dict):
        return {k: sanitize_floats(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [sanitize_floats(v) for v in obj]
    return obj


# Backward-compat alias
_sanitize_floats = sanitize_floats


class ProgressWriter:
    """JSONL writer for training progress telemetry."""

    def __init__(self, output_dir: str | Path, interval: float = 1.0) -> None:
        self._path = Path(output_dir) / ".progress.jsonl"
        self._interval = interval
        self._last_write = 0.0
        self._last_step: Optional[int] = None
        self._fh: Optional[Any] = None

    def _ensure_open(self) -> None:
        if self._fh is None:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            self._fh = open(self._path, "a", encoding="utf-8")

    def maybe_write(self, **kwargs: Any) -> None:
        """Write a training step line: once per new *step*, else time-gated."""
        now = time.monotonic()
        raw = kwargs.get("step")
        step: Optional[int] = None
        if isinstance(raw, int):
            step = raw
        elif isinstance(raw, float) and raw == int(raw):
            step = int(raw)

        if step is not None:
            if step == self._last_step:
                return
            self._last_step = step
            self._last_write = now
            self._write_line(kind="step", **kwargs)
            return

        if now - self._last_write < self._interval:
            return
        self._last_write = now
        self._write_line(kind="step", **kwargs)

    def write_event(self, **kwargs: Any) -> None:
        """Write an event line unconditionally (epoch, checkpoint, complete, fail)."""
        self._write_line(**kwargs)

    def _write_line(self, **kwargs: Any) -> None:
        self._ensure_open()
        kwargs["ts"] = time.time()
        assert self._fh is not None
        clean = _sanitize_floats(kwargs)
        self._fh.write(json.dumps(clean, default=str, allow_nan=False) + "\n")
        self._fh.flush()

    def close(self) -> None:
        if self._fh is not None:
            try:
                self._fh.close()
            except OSError:
                pass
            self._fh = None

    def __enter__(self) -> "ProgressWriter":
        self._ensure_open()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()
