"""
Persistent user settings for Side-Step.

Stored as JSON at a platform-aware location:

    Linux/macOS:  ``~/.config/sidestep/settings.json``
    Windows:      ``%APPDATA%\\sidestep\\settings.json``

Settings hold the checkpoint directory path, API keys for AI-assisted
dataset building, and first-run state.  They are *not* training
hyperparameters -- those live in presets.
"""

from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)

# In-process cache: (mtime, data) — invalidated when file changes.
_cache: tuple[float, Dict[str, Any]] | None = None

# Current schema version -- bump when adding/renaming keys.
_SCHEMA_VERSION = 7

# Environment variable names for API key resolution
_ENV_GEMINI_KEY = "GEMINI_API_KEY"
_ENV_OPENAI_KEY = "OPENAI_API_KEY"
_ENV_OPENAI_BASE = "OPENAI_BASE_URL"
_ENV_OPENAI_MODEL = "OPENAI_MODEL"
_ENV_GEMINI_MODEL = "GEMINI_MODEL"
_ENV_GENIUS_TOKEN = "GENIUS_API_TOKEN"


# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

def settings_dir() -> Path:
    """Platform-aware root config directory for Side-Step."""
    if sys.platform == "win32":
        base = Path(os.environ.get("APPDATA", Path.home() / "AppData" / "Roaming"))
    else:
        base = Path(os.environ.get("XDG_CONFIG_HOME", Path.home() / ".config"))
    return base / "sidestep"


def settings_path() -> Path:
    """Full path to the settings JSON file."""
    return settings_dir() / "settings.json"


# ---------------------------------------------------------------------------
# Load / Save
# ---------------------------------------------------------------------------

def _default_settings() -> Dict[str, Any]:
    """Return a blank settings dict with the current schema version."""
    return {
        "version": _SCHEMA_VERSION,
        "checkpoint_dir": None,
        "first_run_complete": False,
        "caption_provider": "gemini",
        "gemini_api_key": None,
        "gemini_model": None,
        "openai_api_key": None,
        "openai_base_url": None,
        "openai_model": None,
        "genius_api_token": None,
        "trained_adapters_dir": None,
        "preprocessed_tensors_dir": None,
        "audio_dir": None,
        "history_output_roots": [],
    }


def load_settings() -> Optional[Dict[str, Any]]:
    """Load settings from disk with mtime-based caching.

    Returns ``None`` if the file does not exist or cannot be parsed.
    Performs lightweight schema migration when the on-disk version is
    older than ``_SCHEMA_VERSION``.  Subsequent calls within the same
    process return a cached copy unless the file's mtime has changed.
    """
    global _cache
    p = settings_path()
    if not p.is_file():
        return None

    try:
        mtime = p.stat().st_mtime
    except OSError:
        mtime = 0.0

    if _cache is not None:
        cached_mtime, cached_data = _cache
        if cached_mtime == mtime:
            return cached_data.copy()

    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        logger.warning("Failed to read settings: %s", exc)
        return None

    # Schema migration: fill missing keys from defaults
    defaults = _default_settings()
    for key, val in defaults.items():
        data.setdefault(key, val)
    data["version"] = _SCHEMA_VERSION

    _cache = (mtime, data.copy())
    return data


def save_settings(data: Dict[str, Any]) -> None:
    """Write settings to disk atomically, creating parent dirs as needed.

    Uses a temporary file + ``os.replace`` so a crash mid-write cannot
    corrupt the settings file.  Invalidates the in-process cache so the
    next :func:`load_settings` call returns the freshly written data.
    """
    import tempfile

    global _cache
    data["version"] = _SCHEMA_VERSION
    p = settings_path()
    p.parent.mkdir(parents=True, exist_ok=True)
    content = json.dumps(data, indent=2) + "\n"
    fd, tmp = tempfile.mkstemp(dir=str(p.parent), suffix=".tmp")
    closed = False
    try:
        os.write(fd, content.encode("utf-8"))
        os.close(fd)
        closed = True
        os.replace(tmp, str(p))
    except BaseException:
        if not closed:
            os.close(fd)
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise
    _cache = None  # Invalidate cache
    logger.debug("Settings saved to %s", p)


# ---------------------------------------------------------------------------
# Convenience helpers
# ---------------------------------------------------------------------------

def is_first_run() -> bool:
    """Return ``True`` if settings do not exist or setup was never completed."""
    data = load_settings()
    if data is None:
        return True
    return not data.get("first_run_complete", False)


def get_checkpoint_dir() -> Optional[str]:
    """Return the stored checkpoint directory, or ``None``."""
    data = load_settings()
    if data is None:
        return None
    return data.get("checkpoint_dir")


def get_caption_provider() -> str:
    """Return the preferred caption provider name (``'gemini'`` or ``'openai'``)."""
    data = load_settings()
    if data is None:
        return "gemini"
    return data.get("caption_provider", "gemini")


def _resolve_key(env_var: str, settings_key: str) -> Optional[str]:
    """Resolve an API key: env var takes precedence over settings file."""
    env_val = os.environ.get(env_var)
    if env_val:
        return env_val
    data = load_settings()
    if data is None:
        return None
    return data.get(settings_key)


def get_gemini_api_key() -> Optional[str]:
    """Return the Gemini API key (env var → settings file)."""
    return _resolve_key(_ENV_GEMINI_KEY, "gemini_api_key")


def get_gemini_model() -> Optional[str]:
    """Return the Gemini model name (env var → settings file)."""
    return _resolve_key(_ENV_GEMINI_MODEL, "gemini_model")


def get_openai_api_key() -> Optional[str]:
    """Return the OpenAI API key (env var → settings file)."""
    return _resolve_key(_ENV_OPENAI_KEY, "openai_api_key")


def get_openai_base_url() -> Optional[str]:
    """Return the OpenAI-compatible base URL (env var → settings file)."""
    return _resolve_key(_ENV_OPENAI_BASE, "openai_base_url")


def get_openai_model() -> Optional[str]:
    """Return the OpenAI model name (env var → settings file)."""
    return _resolve_key(_ENV_OPENAI_MODEL, "openai_model")


def get_genius_api_token() -> Optional[str]:
    """Return the Genius API token (env var → settings file)."""
    return _resolve_key(_ENV_GENIUS_TOKEN, "genius_api_token")


def get_trained_adapters_dir() -> str:
    """Return the configured adapter weights directory.

    Falls back to ``./trained_adapters`` (relative to CWD) when unset.
    """
    data = load_settings()
    val = data.get("trained_adapters_dir") if data else None
    return val or "./trained_adapters"


def get_preprocessed_tensors_dir() -> str:
    """Return the configured preprocessed tensors directory.

    Falls back to ``./preprocessed_tensors`` (relative to CWD) when unset.
    """
    data = load_settings()
    val = data.get("preprocessed_tensors_dir") if data else None
    return val or "./preprocessed_tensors"


def get_history_output_roots() -> list[str]:
    """Return remembered extra history roots for output-dir overrides.

    These are directories outside ``trained_adapters_dir`` that should also be
    scanned for run history.
    """
    data = load_settings() or {}
    roots = data.get("history_output_roots")
    if not isinstance(roots, list):
        return []
    out: list[str] = []
    for root in roots:
        if isinstance(root, str) and root.strip():
            out.append(root)
    return out


_MAX_HISTORY_ROOTS = 20


def remember_history_output_root(path: str) -> None:
    """Persist an additional run-history root path if it is new."""
    if not path:
        return
    p = Path(os.path.expandvars(path)).expanduser()
    try:
        normalized = str(p.resolve(strict=False))
    except (OSError, ValueError):
        normalized = str(p)

    data = load_settings() or _default_settings()
    roots = data.get("history_output_roots")
    if not isinstance(roots, list):
        roots = []

    # Prune roots that no longer exist on disk
    roots = [r for r in roots if Path(r).is_dir()]

    if normalized not in roots:
        roots.append(normalized)

    # Cap at _MAX_HISTORY_ROOTS (keep most recent)
    if len(roots) > _MAX_HISTORY_ROOTS:
        roots = roots[-_MAX_HISTORY_ROOTS:]

    data["history_output_roots"] = roots
    save_settings(data)

