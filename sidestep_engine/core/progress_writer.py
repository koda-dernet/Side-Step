"""
Time-gated JSONL progress file for GUI telemetry.

Writes one JSON line per call, throttled to at most once per second
(configurable).  The file lives at ``{output_dir}/.progress.jsonl``
and is consumed by the GUI server via tail + WebSocket.

Usage in the training loop::

    pw = ProgressWriter(output_dir)
    # ... every optimizer step:
    pw.maybe_write(step=global_step, epoch=epoch+1, max_epochs=...,
                   loss=avg_loss, lr=_lr, best_loss=best_loss,
                   best_epoch=best_epoch, steps_per_epoch=steps_per_epoch)
    # ... at end:
    pw.write_event(kind="complete", step=global_step, epoch=epoch+1,
                   max_epochs=max_epochs, loss=final_loss)
    pw.close()
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
    """Time-gated JSONL writer for training progress telemetry."""

    def __init__(self, output_dir: str | Path, interval: float = 1.0) -> None:
        self._path = Path(output_dir) / ".progress.jsonl"
        self._interval = interval
        self._last_write = 0.0
        self._fh: Optional[Any] = None
        self._step_count = 0

    def _ensure_open(self) -> None:
        if self._fh is None:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            self._fh = open(self._path, "a", encoding="utf-8")

    def maybe_write(self, **kwargs: Any) -> None:
        """Write a progress line if >= interval seconds since last write."""
        now = time.monotonic()
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
