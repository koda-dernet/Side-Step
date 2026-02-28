"""
Filesystem helpers for the Side-Step GUI REST endpoints.

Thin wrappers around existing engine functions and standard library
calls.  Each function returns a JSON-serializable dict or list.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

from sidestep_engine.core.constants import (
    AUDIO_EXTENSIONS as _AUDIO_EXTS,
    ADAPTER_TYPES as _ADAPTER_TYPES,
    LATENT_FPS as _LATENT_FPS,
)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def _resolve_gui_path(path: str | Path, *, base: Optional[Path] = None) -> Path:
    """Resolve user/config paths consistently for GUI server operations.

    - Expands env vars and ``~``.
    - Anchors relative paths to project root unless *base* is provided.
    - Returns a non-strict resolved path to tolerate missing targets.
    """
    p = Path(os.path.expandvars(str(path))).expanduser()
    if not p.is_absolute():
        p = (base or _PROJECT_ROOT) / p
    try:
        return p.resolve(strict=False)
    except (OSError, ValueError):
        return p


def _coerce_sidecar_path(path: str) -> Path:
    """Return a safe ``.txt`` sidecar path for user-provided paths.

    GUI clients may accidentally send an audio path instead of the matching
    sidecar path. Coercing to ``.txt`` prevents destructive overwrites of
    source audio files.
    """
    p = Path(path)
    return p if p.suffix.lower() == ".txt" else p.with_suffix(".txt")


# ======================================================================
# Path scoping
# ======================================================================

# Extensions the file browser shows (besides directories)
_BROWSE_EXTS = _AUDIO_EXTS | {".pt", ".safetensors", ".json", ".txt", ".yaml", ".yml"}
_MAX_BROWSE_ENTRIES = 200

# Cached allowed roots (rebuilt when settings change)
_allowed_cache: Optional[Tuple[List[Path], List[Path]]] = None
_allowed_cache_mtime: float = 0.0


def _get_fs_roots() -> List[str]:
    """Return filesystem root paths for the current OS."""
    if os.name == "nt":
        import string
        return [f"{d}:\\" for d in string.ascii_uppercase
                if Path(f"{d}:\\").exists()]
    return ["/"]


def _build_allowed_roots() -> Tuple[List[Path], List[Path]]:
    """Build the resolved and logical allowed root lists."""
    resolved = [_PROJECT_ROOT.resolve(), Path.home().resolve()]
    logical = [Path(os.path.normpath(str(_PROJECT_ROOT))),
               Path(os.path.normpath(str(Path.home())))]
    try:
        from sidestep_engine.settings import load_settings
        s = load_settings()
        for key in ("checkpoint_dir", "trained_adapters_dir",
                     "preprocessed_tensors_dir", "audio_dir"):
            val = s.get(key)
            if val:
                try:
                    resolved.append(_resolve_gui_path(val))
                    logical.append(Path(os.path.normpath(
                        str(Path(os.path.expandvars(str(val))).expanduser()))))
                except (OSError, ValueError):
                    pass
    except Exception:
        pass
    return resolved, logical


def _get_allowed_roots() -> Tuple[List[Path], List[Path]]:
    """Return cached allowed roots, refreshing if settings file changed."""
    global _allowed_cache, _allowed_cache_mtime
    try:
        from sidestep_engine.settings import settings_path
        sp = settings_path()
        mt = sp.stat().st_mtime if sp.exists() else 0.0
    except Exception:
        mt = 0.0
    if _allowed_cache is None or mt != _allowed_cache_mtime:
        _allowed_cache = _build_allowed_roots()
        _allowed_cache_mtime = mt
    return _allowed_cache


def _is_under(child: Path, roots: List[Path]) -> bool:
    for root in roots:
        try:
            child.relative_to(root)
            return True
        except ValueError:
            continue
    return False


def _is_parent_of(parent: Path, roots: List[Path]) -> bool:
    for root in roots:
        try:
            root.relative_to(parent)
            return True
        except ValueError:
            continue
    return False


def is_path_allowed(path: str) -> bool:
    """Check if a path is within allowed scope for browsing.

    Allowed roots (OS-independent):
    - The project root
    - The user's home directory (logical, before symlink resolution)
    - Any directory configured in settings (checkpoint_dir, etc.)
    - OS filesystem roots (/, C:\\, D:\\, etc. — auto-detected)
    """
    try:
        resolved = _resolve_gui_path(path)
    except (OSError, ValueError):
        return False

    # Logical (non-symlink-resolved) path for symlink-friendly checks
    logical = Path(os.path.expandvars(str(path))).expanduser()
    if not logical.is_absolute():
        logical = _PROJECT_ROOT / logical
    try:
        logical = Path(os.path.normpath(str(logical)))
    except (OSError, ValueError):
        logical = resolved

    # Always allow OS filesystem roots
    if str(resolved) in _get_fs_roots():
        return True

    allowed_resolved, allowed_logical = _get_allowed_roots()

    if (_is_under(resolved, allowed_resolved) or _is_under(logical, allowed_logical)
            or _is_under(resolved, allowed_logical) or _is_under(logical, allowed_resolved)):
        return True
    if _is_parent_of(resolved, allowed_resolved) or _is_parent_of(logical, allowed_logical):
        return True

    return False


# ======================================================================
# Directory browsing
# ======================================================================

def browse_dir(path: str, dirs_only: bool = False) -> Dict[str, Any]:
    """List entries in a directory.

    Uses os.scandir for speed (no extra stat calls).
    Only shows directories and files with relevant extensions.
    Capped at _MAX_BROWSE_ENTRIES to prevent huge listings.
    """
    p = Path(path)
    if not p.is_dir():
        return {"error": f"Not a directory: {path}", "entries": []}

    dirs: List[Dict[str, Any]] = []
    files: List[Dict[str, Any]] = []
    try:
        with os.scandir(str(p)) as it:
            for entry in it:
                if entry.name.startswith("."):
                    continue
                is_dir = entry.is_dir(follow_symlinks=False) or entry.is_dir(follow_symlinks=True)
                if is_dir:
                    dirs.append({"name": entry.name, "path": entry.path, "is_dir": True})
                elif not dirs_only:
                    ext = os.path.splitext(entry.name)[1].lower()
                    if ext in _BROWSE_EXTS:
                        files.append({"name": entry.name, "path": entry.path, "is_dir": False})
                if len(dirs) + len(files) >= _MAX_BROWSE_ENTRIES:
                    break
    except PermissionError:
        return {"error": f"Permission denied: {path}", "entries": []}

    dirs.sort(key=lambda e: e["name"].lower())
    files.sort(key=lambda e: e["name"].lower())
    return {"path": str(p), "entries": dirs + files}


# ======================================================================
# Model scanning
# ======================================================================

def scan_models(checkpoint_dir: str) -> Dict[str, Any]:
    """Scan for model subdirectories under checkpoint_dir."""
    if not checkpoint_dir:
        return {"models": [], "error": "No checkpoint_dir provided"}
    try:
        from sidestep_engine.models.discovery import scan_models as _scan
        models = _scan(checkpoint_dir)
        return {"models": [
            {
                "name": m.name,
                "path": str(m.path),
                "is_official": m.is_official,
                "base_model": m.base_model,
            }
            for m in models
        ]}
    except Exception as exc:
        return {"models": [], "error": str(exc)}


# ======================================================================
# Tensor / audio dataset scanning
# ======================================================================

from sidestep_engine.core.dataset_scanner import (  # noqa: E402
    fmt_duration as _fmt_duration,
    scan_tensors_dir as _scan_tensors_core,
)


def scan_tensors(tensors_dir: str) -> Dict[str, Any]:
    """Scan for preprocessed tensor folders."""
    if not tensors_dir:
        return {"datasets": []}
    p = _resolve_gui_path(tensors_dir)
    if not p.is_dir():
        return {"datasets": [], "error": f"Not a directory: {p}"}
    return {"datasets": _scan_tensors_core(p)}


def link_source_audio(tensor_name: str, audio_path: str) -> Dict[str, Any]:
    """Persist a source-audio link into a tensor dataset's preprocess_meta.json."""
    settings_data: Dict[str, Any] = {}
    try:
        from sidestep_engine.settings import load_settings
        settings_data = load_settings() or {}
    except Exception:
        settings_data = {}

    tensors_dir = settings_data.get("preprocessed_tensors_dir", "")
    if not tensors_dir:
        return {"ok": False, "error": "No preprocessed_tensors_dir configured in Settings"}
    base = _resolve_gui_path(tensors_dir)
    target = (base / tensor_name).resolve()
    if not target.is_relative_to(base.resolve()):
        return {"ok": False, "error": "Invalid tensor dataset name"}
    if not target.is_dir():
        return {"ok": False, "error": f"Tensor dataset not found: {tensor_name}"}

    audio_p = _resolve_gui_path(audio_path)
    if not audio_p.is_dir():
        return {"ok": False, "error": f"Audio path is not a directory: {audio_path}"}

    meta_path = target / "preprocess_meta.json"
    meta: Dict[str, Any] = {}
    if meta_path.is_file():
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except Exception:
            pass

    meta["audio_dir"] = str(audio_p)
    meta["source_audio_dir"] = str(audio_p)
    try:
        meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    except OSError as exc:
        return {"ok": False, "error": f"Failed to write metadata: {exc}"}

    return {"ok": True, "audio_dir": str(audio_p)}


def scan_all_datasets() -> Dict[str, Any]:
    """Scan both tensor dirs and audio dirs for the datasets manager."""
    from sidestep_engine.core.dataset_scanner import scan_audio_folder

    datasets: List[Dict[str, Any]] = []

    settings_data: Dict[str, Any] = {}
    try:
        from sidestep_engine.settings import load_settings
        settings_data = load_settings() or {}
    except Exception:
        settings_data = {}

    tensors_dir = settings_data.get("preprocessed_tensors_dir", "")
    if tensors_dir:
        result = scan_tensors(tensors_dir)
        datasets.extend(result.get("datasets", []))

    audio_dir = settings_data.get("audio_dir", "")
    if audio_dir:
        p = _resolve_gui_path(audio_dir)
        if p.is_dir():
            child_dirs = [d for d in sorted(p.iterdir()) if d.is_dir()]
            found_subfolder = False
            for child in child_dirs:
                info = scan_audio_folder(child)
                if info is not None:
                    found_subfolder = True
                    datasets.append(info)

            if not found_subfolder:
                info = scan_audio_folder(p)
                if info is not None:
                    datasets.append(info)

    return {"datasets": datasets}


def scan_audio_dir(path: str) -> Dict[str, Any]:
    """Scan for audio files + sidecars and return flat + folder-tree metadata."""
    if not path:
        return {"files": [], "error": "No path provided"}
    p = _resolve_gui_path(path)
    if not p.is_dir():
        return {"files": [], "error": f"Not a directory: {p}"}

    # Try to import duration helper; fall back gracefully
    _get_dur = None
    try:
        from sidestep_engine.data.audio_duration import get_audio_duration
        _get_dur = get_audio_duration
    except Exception:
        pass

    # Try to import sidecar reader for genre/tags/trigger
    _read_sc = None
    try:
        from sidestep_engine.data.sidecar_io import read_sidecar as _rsc
        _read_sc = _rsc
    except Exception:
        pass

    files = []
    folders: Dict[str, Dict[str, Any]] = {}

    def _ensure_folder(rel_folder: str) -> Dict[str, Any]:
        key = rel_folder or "."
        if key not in folders:
            parent = str(Path(key).parent).replace("\\", "/")
            if parent == ".":
                parent = "" if key == "." else "."
            name = p.name if key == "." else Path(key).name
            depth = 0 if key == "." else key.count("/") + 1
            folders[key] = {
                "path": key,
                "name": name,
                "parent_path": parent,
                "depth": depth,
                "file_count": 0,
                "sidecar_count": 0,
                "total_duration": 0,
            }
        return folders[key]

    _ensure_folder(".")
    total_duration = 0
    longest = 0
    sidecar_count = 0
    for child in sorted(p.rglob("*")):
        if child.suffix.lower() in _AUDIO_EXTS:
            sidecar_path = child.with_suffix(".txt")
            rel_path = str(child.relative_to(p)).replace("\\", "/")
            rel_folder = str(Path(rel_path).parent).replace("\\", "/")
            if rel_folder == ".":
                rel_folder = "."
            dur = 0
            if _get_dur:
                try:
                    dur = int(_get_dur(child))
                except Exception:
                    pass
            total_duration += dur
            if dur > longest:
                longest = dur
            has_sc = sidecar_path.is_file()
            if has_sc:
                sidecar_count += 1
            folder_entry = _ensure_folder(rel_folder)
            folder_entry["file_count"] += 1
            folder_entry["total_duration"] += dur
            if has_sc:
                folder_entry["sidecar_count"] += 1
            # Read sidecar metadata if available
            genre = ""
            tags = ""
            trigger = ""
            if has_sc and _read_sc:
                try:
                    sc_data = _read_sc(sidecar_path)
                    genre = sc_data.get("genre", "") if isinstance(sc_data, dict) else ""
                    tags = sc_data.get("tags", "") if isinstance(sc_data, dict) else ""
                    trigger = (
                        (sc_data.get("custom_tag") or sc_data.get("trigger") or "").strip()
                        if isinstance(sc_data, dict)
                        else ""
                    )
                except Exception:
                    pass
            files.append({
                "name": child.name,
                "path": str(child),
                "relative_path": rel_path,
                "folder_path": rel_folder,
                "sidecar_path": str(sidecar_path),
                "has_sidecar": has_sc,
                "size": child.stat().st_size,
                "duration": dur,
                "genre": genre,
                "tags": tags,
                "trigger": trigger,
            })

    # Compute common trigger tag per folder (4.3)
    for f in files:
        fk = f["folder_path"]
        if fk in folders:
            tag = f.get("trigger", "").strip()
            entry = folders[fk]
            if "trigger_set" not in entry:
                entry["trigger_set"] = set()
            if tag:
                entry["trigger_set"].add(tag)
    for entry in folders.values():
        ts = entry.pop("trigger_set", set())
        entry["common_trigger"] = ts.pop() if len(ts) == 1 else ""

    folder_rows = []
    for rel in sorted(
        folders.keys(),
        key=lambda x: (folders[x]["depth"], x.lower()),
    ):
        row = dict(folders[rel])
        row["duration_label"] = _fmt_duration(int(row.get("total_duration", 0) or 0))
        folder_rows.append(row)

    return {
        "files": files, "path": str(p),
        "total_duration": total_duration, "longest": longest,
        "sidecar_count": sidecar_count,
        "folders": folder_rows,
    }


def create_mix_dataset(
    source_root: str,
    destination_root: str,
    mix_name: str,
    files: List[str],
) -> Dict[str, Any]:
    """Create a storage-light mixed dataset by symlinking selected audio files."""
    src_root = _resolve_gui_path(source_root)
    dest_root = _resolve_gui_path(destination_root)
    name = str(mix_name or "").strip()
    if not name:
        return {"ok": False, "error": "Missing mix_name"}
    if not src_root.is_dir():
        return {"ok": False, "error": f"Invalid source root: {src_root}"}
    if not dest_root.exists():
        try:
            dest_root.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            return {"ok": False, "error": f"Could not create destination root: {exc}"}
    if not dest_root.is_dir():
        return {"ok": False, "error": f"Invalid destination root: {dest_root}"}

    safe_name = "".join(ch if (ch.isalnum() or ch in "-_.") else "_" for ch in name).strip("._")
    if not safe_name:
        safe_name = "mix_dataset"
    out_dir = _resolve_gui_path(safe_name, base=dest_root)

    try:
        out_dir.relative_to(dest_root)
    except ValueError:
        return {"ok": False, "error": "mix_name resolves outside destination root"}

    if out_dir.exists() and any(out_dir.iterdir()):
        return {"ok": False, "error": f"Destination already exists and is not empty: {out_dir}"}

    out_dir.mkdir(parents=True, exist_ok=True)

    created = 0
    skipped = 0
    errors: List[str] = []

    for raw in files or []:
        src = _resolve_gui_path(raw)
        if src.suffix.lower() not in _AUDIO_EXTS:
            skipped += 1
            continue
        if not src.is_file():
            skipped += 1
            continue
        try:
            rel = src.relative_to(src_root)
        except ValueError:
            skipped += 1
            continue

        dest_audio = out_dir / rel
        dest_audio.parent.mkdir(parents=True, exist_ok=True)
        if dest_audio.exists():
            skipped += 1
            continue

        try:
            os.symlink(str(src), str(dest_audio))
            created += 1
        except OSError as exc:
            errors.append(f"{src.name}: {exc}")
            continue

        src_sidecar = src.with_suffix(".txt")
        if src_sidecar.is_file():
            dest_sidecar = dest_audio.with_suffix(".txt")
            if not dest_sidecar.exists():
                try:
                    os.symlink(str(src_sidecar), str(dest_sidecar))
                except OSError:
                    # Sidecar symlink failures should not fail the whole mix.
                    pass

    return {
        "ok": True,
        "created": created,
        "skipped": skipped,
        "errors": errors,
        "path": str(out_dir),
    }


def fisher_map_status(dataset_dir: str, model_variant: str = "") -> Dict[str, Any]:
    """Return fisher_map.json status for an explicit dataset directory.

    Args:
        dataset_dir: Directory expected to contain preprocessed ``.pt`` files.
        model_variant: Optional selected model variant for stale checks.
    """
    base = Path(dataset_dir)
    fm_path = base / "fisher_map.json"
    status: Dict[str, Any] = {
        "exists": False,
        "path": str(fm_path),
        "modules": 0,
        "rank_min": 0,
        "rank_max": 0,
        "stale": False,
    }

    if not fm_path.is_file():
        return status

    status["exists"] = True
    try:
        data = json.loads(fm_path.read_text(encoding="utf-8"))
        modules = data.get("modules")
        if isinstance(modules, list):
            status["modules"] = len(modules)
        else:
            rank_pattern = data.get("rank_pattern")
            if isinstance(rank_pattern, dict):
                status["modules"] = len(rank_pattern)

        budget = data.get("rank_budget")
        if isinstance(budget, dict):
            status["rank_min"] = int(budget.get("min", 0) or 0)
            status["rank_max"] = int(budget.get("max", 0) or 0)

        selected_variant = str(model_variant or "").strip().lower()
        map_variant = str(data.get("model_variant") or "").strip().lower()
        status["stale"] = bool(selected_variant and map_variant and selected_variant != map_variant)
    except Exception as exc:
        status["error"] = str(exc)

    return status


# ======================================================================
# Sidecar I/O
# ======================================================================

def read_sidecar(path: str) -> Dict[str, Any]:
    """Read a .txt sidecar file."""
    try:
        from sidestep_engine.data.sidecar_io import read_sidecar as _read
        data = _read(_coerce_sidecar_path(path))
        return {"ok": True, "data": data}
    except Exception as exc:
        return {"ok": False, "error": str(exc)}


def write_sidecar(path: str, data: Dict[str, Any]) -> None:
    """Write a .txt sidecar file (merge-on-write).

    Reads the existing sidecar first and merges *data* on top so that
    fields the caller doesn't send (e.g. ``repeat``, ``prompt_override``)
    are preserved.  The caller's values always win — including empty
    strings, which intentionally clear a field.
    """
    if not is_path_allowed(path):
        raise PermissionError(f"Path not in allowed scope: {path}")
    from sidestep_engine.data.sidecar_io import (
        read_sidecar as _read,
        write_sidecar as _write,
    )
    safe_path = _coerce_sidecar_path(path)

    existing: Dict[str, Any] = {}
    if safe_path.is_file():
        try:
            existing = _read(safe_path)
        except Exception:
            pass

    existing.update({k: v for k, v in (data or {}).items()})

    if "trigger" in existing and "custom_tag" not in existing:
        existing["custom_tag"] = existing.pop("trigger")

    _write(safe_path, existing)


def bulk_apply_trigger_tag(paths: List[str], tag: str, position: str) -> Dict[str, Any]:
    """Apply a trigger tag to multiple sidecar files."""
    from sidestep_engine.data.sidecar_io import read_sidecar as _read, write_sidecar as _write

    updated = 0
    errors = []
    for p in paths:
        if not is_path_allowed(p):
            errors.append({"path": p, "error": "Path not in allowed scope"})
            continue
        try:
            pp = _coerce_sidecar_path(p)
            data = _read(pp)
            existing = str(data.get("custom_tag") or data.get("trigger") or "").strip()
            if position == "prepend":
                data["custom_tag"] = f"{tag} {existing}".strip()
            elif position == "append":
                data["custom_tag"] = f"{existing} {tag}".strip()
            elif position == "replace":
                data["custom_tag"] = tag
            data.pop("trigger", None)
            _write(pp, data)
            updated += 1
        except Exception as exc:
            errors.append({"path": p, "error": str(exc)})

    return {"updated": updated, "errors": errors}


# ======================================================================
# Presets -- delegates to the canonical multi-directory system in ui.presets
# ======================================================================

def list_presets() -> List[Dict[str, Any]]:
    from sidestep_engine.ui.presets import list_presets as _list
    return _list()


def load_preset(name: str) -> Optional[Dict[str, Any]]:
    from sidestep_engine.ui.presets import load_preset as _load
    return _load(name)


def save_preset(name: str, data: Dict[str, Any]) -> None:
    from sidestep_engine.ui.presets import save_preset as _save
    _save(name, data.get("description", ""), data)


def delete_preset(name: str) -> dict:
    from sidestep_engine.ui.presets import delete_preset as _delete
    ok = _delete(name)
    if ok:
        return {"ok": True}
    return {"ok": False, "error": f"Preset '{name}' not found or is built-in"}


# ======================================================================
# Run history -- delegates to core.run_discovery
# ======================================================================

from sidestep_engine.core.run_discovery import (  # noqa: E402
    has_adapter_artifacts as _has_adapter_artifacts,
    looks_like_run_dir as _looks_like_run_dir,
    resolve_run_artifact as _resolve_run_artifact,
)
from sidestep_engine.core.progress_writer import sanitize_floats as _sanitize_floats


def _adapters_dir() -> "Path | None":
    """Resolve the trained_adapters directory from settings."""
    try:
        from sidestep_engine.settings import get_trained_adapters_dir
        val = get_trained_adapters_dir()
        if not val:
            return None
        return _resolve_gui_path(val)
    except Exception:
        return _PROJECT_ROOT / "trained_adapters"


def _history_override_roots() -> List[Path]:
    """Return additional history roots remembered from output overrides."""
    try:
        from sidestep_engine.settings import get_history_output_roots
        roots = get_history_output_roots()
    except Exception:
        roots = []
    out: List[Path] = []
    for root in roots:
        if isinstance(root, str) and root.strip():
            out.append(_resolve_gui_path(root))
    return out


def _run_roots_kwargs() -> dict:
    """Build kwargs for core.run_discovery functions."""
    return dict(
        adapters_root=_adapters_dir(),
        extra_roots=_history_override_roots(),
    )


def build_history() -> List[Dict[str, Any]]:
    from sidestep_engine.core.run_discovery import build_history as _build
    return _build(**_run_roots_kwargs(), sanitize=_sanitize_floats)


def delete_detected_history_folder(path: str) -> Dict[str, Any]:
    from sidestep_engine.core.run_discovery import (
        delete_detected_folder, history_roots,
    )
    target = _resolve_gui_path(path)
    if not path:
        return {"ok": False, "error": "Missing path"}
    roots = history_roots(**_run_roots_kwargs())
    return delete_detected_folder(target, roots)


def load_run_config(run_name: str) -> Optional[Dict[str, Any]]:
    from sidestep_engine.core.run_discovery import load_run_config as _load
    return _load(run_name, **_run_roots_kwargs())


def load_run_curve(run_name: str) -> List[Dict[str, Any]]:
    from sidestep_engine.core.run_discovery import load_run_curve as _load
    return _load(run_name, **_run_roots_kwargs(), sanitize=_sanitize_floats)


def list_checkpoints(run_name: str) -> List[Dict[str, Any]]:
    from sidestep_engine.core.run_discovery import list_checkpoints as _list
    return _list(run_name, **_run_roots_kwargs())


# ======================================================================
# API key validation
# ======================================================================

def validate_key(provider: str, key: str, model: Optional[str] = None,
                 base_url: Optional[str] = None) -> Dict[str, Any]:
    """Validate an API key for a caption provider."""
    try:
        if provider == "gemini":
            from sidestep_engine.data.caption_provider_gemini import validate_key as _val
            ok = _val(key)
        elif provider == "openai":
            from sidestep_engine.data.caption_provider_openai import validate_key as _val
            ok = _val(key, base_url=base_url, model=model)
        elif provider == "genius":
            from sidestep_engine.data.lyrics_provider_genius import validate_token as _val
            ok = _val(key)
        else:
            return {"valid": False, "error": f"Unknown provider: {provider}"}
        return {"valid": ok}
    except Exception as exc:
        return {"valid": False, "error": str(exc)}
