"""Path normalization for cross-platform (Linux/Windows/macOS) compatibility."""

from __future__ import annotations

from pathlib import Path
from typing import Optional


def normalize_path(path: Optional[str]) -> str:
    """Strip whitespace, expand ~, and resolve to absolute path. Cross-platform safe.

    Works with any drive (C:, D:, etc.), relative paths, paths with spaces,
    and ~ for home directory. Mitigates whitespace, cwd mismatch, and path format
    differences across Linux, Windows, and macOS.
    """
    if path is None:
        return ""
    s = str(path).strip()
    if not s:
        return ""
    try:
        p = Path(s).expanduser()
        return str(p.resolve())
    except (OSError, RuntimeError):
        return s
