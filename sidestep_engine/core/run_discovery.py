"""Run history discovery and checkpoint management.

Pure filesystem logic shared by GUI and Wizard for discovering
past training runs, reading run metadata, and listing checkpoints.
No UI or GUI imports -- callers supply root directories.
"""

from __future__ import annotations

import json
import logging
import math
import shutil
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple

from sidestep_engine.core.constants import ADAPTER_TYPES

logger = logging.getLogger(__name__)

_MAX_CURVE_POINTS = 10_000

# ---------------------------------------------------------------------------
# Structural checks
# ---------------------------------------------------------------------------

_ADAPTER_ARTIFACTS = (
    "adapter_model.safetensors",
    "adapter_model.bin",
    "lokr_weights.safetensors",
    "loha_weights.safetensors",
)


def has_adapter_artifacts(folder: Path) -> bool:
    """True when *folder* contains inference-loadable adapter weights."""
    return any((folder / fname).is_file() for fname in _ADAPTER_ARTIFACTS)


def parse_epoch_num(name: str) -> int:
    """Extract integer epoch from a checkpoint directory name like ``epoch_42``."""
    if not name.startswith("epoch_"):
        return -1
    try:
        return int(name.split("_", 1)[1])
    except (TypeError, ValueError):
        return -1


def latest_checkpoint_with_artifacts(run_dir: Path) -> Optional[Path]:
    """Return the highest-numbered checkpoint that has adapter weights."""
    ckpt_root = run_dir / "checkpoints"
    if not ckpt_root.is_dir():
        return None
    best: Tuple[int, Path] | None = None
    for child in ckpt_root.iterdir():
        if not child.is_dir():
            continue
        if not has_adapter_artifacts(child):
            continue
        ep = parse_epoch_num(child.name)
        if best is None or ep > best[0]:
            best = (ep, child)
    return best[1] if best is not None else None


def resolve_run_artifact(run_dir: Path) -> Optional[Tuple[str, Path]]:
    """Return ``(artifact_source, artifact_path)`` for a valid run, else ``None``."""
    final_dir = run_dir / "final"
    if final_dir.is_dir() and has_adapter_artifacts(final_dir):
        return ("final", final_dir)

    ckpt_dir = latest_checkpoint_with_artifacts(run_dir)
    if ckpt_dir is not None:
        return ("checkpoint", ckpt_dir)

    return None


def looks_like_run_dir(path: Path) -> bool:
    """Heuristic: does *path* appear to be a Side-Step training run output?"""
    if not path.is_dir():
        return False
    markers = (
        ".progress.jsonl",
        "training_config.json",
        "sidestep_training_config.json",
    )
    if any((path / m).exists() for m in markers):
        return True
    structural = sum(1 for d in ("final", "checkpoints", "best") if (path / d).is_dir())
    if structural >= 2:
        return True
    if structural == 1:
        for d in ("final", "checkpoints", "best"):
            sub = path / d
            if not sub.is_dir():
                continue
            if has_adapter_artifacts(sub):
                return True
            try:
                if any(has_adapter_artifacts(c) for c in sub.iterdir() if c.is_dir()):
                    return True
            except OSError:
                pass
    return False


# ---------------------------------------------------------------------------
# Root-based discovery
# ---------------------------------------------------------------------------

def iter_run_dirs(
    adapters_root: Optional[Path] = None,
    extra_roots: Optional[List[Path]] = None,
) -> List[Tuple[Path, str]]:
    """Collect candidate run directories from canonical + extra roots.

    Args:
        adapters_root: The ``trained_adapters/`` directory (layout: adapter_type/run_name).
        extra_roots: Additional directories to scan (e.g. output_dir overrides).
    """
    seen: set[str] = set()
    runs: List[Tuple[Path, str]] = []

    if adapters_root is not None and adapters_root.is_dir():
        for adapter_dir in sorted(adapters_root.iterdir()):
            if not adapter_dir.is_dir():
                continue
            for run_dir in sorted(adapter_dir.iterdir()):
                if not looks_like_run_dir(run_dir):
                    continue
                key = str(run_dir.resolve(strict=False))
                if key in seen:
                    continue
                seen.add(key)
                runs.append((run_dir, adapter_dir.name))

    for ext_root in (extra_roots or []):
        if not ext_root.is_dir():
            continue

        candidates: List[Path] = [ext_root]
        try:
            children = [c for c in sorted(ext_root.iterdir()) if c.is_dir()]
        except OSError:
            children = []
        candidates.extend(children)
        for child in children:
            try:
                candidates.extend([g for g in sorted(child.iterdir()) if g.is_dir()])
            except OSError:
                continue

        for cand in candidates:
            if not looks_like_run_dir(cand):
                continue
            key = str(cand.resolve(strict=False))
            if key in seen:
                continue
            seen.add(key)
            parent_name = cand.parent.name.lower()
            adapter_hint = parent_name if parent_name in ADAPTER_TYPES else ""
            runs.append((cand, adapter_hint))

    return runs


def history_roots(
    adapters_root: Optional[Path],
    extra_roots: Optional[List[Path]] = None,
) -> List[Path]:
    """Return deduplicated roots used for history discovery / deletion checks."""
    raw = ([adapters_root] if adapters_root is not None else []) + (extra_roots or [])
    out: List[Path] = []
    seen: set[str] = set()
    for root in raw:
        resolved = root.resolve(strict=False)
        key = str(resolved)
        if key not in seen:
            seen.add(key)
            out.append(resolved)
    return out


def find_run_dir(
    run_name: str,
    adapters_root: Optional[Path] = None,
    extra_roots: Optional[List[Path]] = None,
) -> Optional[Path]:
    """Locate a run directory by name across canonical + override roots."""
    matches = [
        rd for rd, _ad in iter_run_dirs(adapters_root, extra_roots)
        if rd.name == run_name
    ]
    if not matches:
        return None

    canonical = adapters_root.resolve(strict=False) if adapters_root is not None else None
    for run_dir in matches:
        if canonical is not None:
            try:
                run_dir.resolve(strict=False).relative_to(canonical)
                return run_dir
            except ValueError:
                pass

    matches.sort(key=lambda p: p.stat().st_mtime if p.exists() else 0.0, reverse=True)
    return matches[0]


# ---------------------------------------------------------------------------
# Run metadata
# ---------------------------------------------------------------------------

def read_last_jsonl(path: Path) -> Optional[Dict[str, Any]]:
    """Read the last non-empty line of a JSONL file."""
    try:
        with open(path, "rb") as f:
            f.seek(0, 2)
            size = f.tell()
            if size == 0:
                return None
            pos = size - 1
            while pos > 0:
                f.seek(pos)
                if f.read(1) == b"\n" and pos < size - 1:
                    break
                pos -= 1
            if pos == 0:
                f.seek(0)
            line = f.readline().decode("utf-8").strip()
            if line:
                return json.loads(line)
    except (OSError, json.JSONDecodeError):
        pass
    return None


def read_run_meta(run_dir: Path, has_final: Optional[bool] = None) -> Dict[str, Any]:
    """Extract metadata from a run directory."""
    meta: Dict[str, Any] = {
        "model": "", "epochs": 0, "best_loss": 0.0,
        "duration": "", "status": "unknown", "adapter_type": "",
    }

    tc = None
    for candidate in [
        run_dir / "final" / "sidestep_training_config.json",
        run_dir / "final" / "training_config.json",
        run_dir / "sidestep_training_config.json",
        run_dir / "training_config.json",
    ]:
        if candidate.is_file():
            tc = candidate
            break
    if tc is not None:
        try:
            data = json.loads(tc.read_text(encoding="utf-8"))
            meta["model"] = data.get("model_variant", "")
            meta["epochs"] = data.get("max_epochs", 0)
            meta["adapter_type"] = str(data.get("adapter_type") or "").lower()
        except (json.JSONDecodeError, OSError):
            pass

    progress = run_dir / ".progress.jsonl"
    if progress.is_file():
        last_line = read_last_jsonl(progress)
        if last_line:
            bl = last_line.get("best_loss", 0.0)
            if isinstance(bl, float) and (math.isinf(bl) or math.isnan(bl)):
                bl = 0.0
            meta["best_loss"] = bl
            raw_kind = last_line.get("kind", "unknown")
            _STATUS_MAP = {
                "step": "stopped", "epoch": "stopped",
                "complete": "complete", "fail": "failed",
            }
            meta["status"] = _STATUS_MAP.get(raw_kind, raw_kind)

    if has_final is None:
        has_final = (run_dir / "final").is_dir()
    if has_final:
        meta["status"] = "complete"

    return meta


# ---------------------------------------------------------------------------
# High-level queries
# ---------------------------------------------------------------------------

def build_history(
    adapters_root: Optional[Path] = None,
    extra_roots: Optional[List[Path]] = None,
    sanitize: Optional[Callable] = None,
) -> List[Dict[str, Any]]:
    """Scan known run roots and return both saved runs and detected-only folders."""
    runs: List[Dict[str, Any]] = []
    for run_dir, adapter_hint in iter_run_dirs(adapters_root, extra_roots):
        artifact = resolve_run_artifact(run_dir)
        detected_only = artifact is None
        artifact_source = ""
        artifact_path = ""
        if artifact is not None:
            artifact_source, artifact_dir = artifact
            artifact_path = str(artifact_dir)

        meta = read_run_meta(run_dir, has_final=(artifact_source == "final"))
        if detected_only:
            meta["status"] = "detected"
            meta["best_loss"] = None

        adapter = "--" if detected_only else (
            str(adapter_hint or "").strip()
            or str(meta.get("adapter_type") or "").strip()
            or "unknown"
        )
        try:
            mtime = run_dir.stat().st_mtime
        except OSError:
            mtime = 0.0

        runs.append({
            "run_name": run_dir.name,
            "adapter": adapter,
            "path": str(run_dir),
            "has_final": artifact_source == "final",
            "has_best": (run_dir / "best").is_dir(),
            "artifact_source": artifact_source,
            "artifact_path": artifact_path,
            "detected_only": detected_only,
            "updated_at_ts": mtime,
            **meta,
        })

    runs.sort(key=lambda r: (r.get("updated_at_ts", 0.0), r.get("run_name", "")), reverse=True)
    for r in runs:
        r.pop("updated_at_ts", None)

    if sanitize is not None:
        runs = sanitize(runs)
    return runs


def load_run_config(
    run_name: str,
    adapters_root: Optional[Path] = None,
    extra_roots: Optional[List[Path]] = None,
) -> Optional[Dict[str, Any]]:
    """Load the training config for a run."""
    run_dir = find_run_dir(run_name, adapters_root, extra_roots)
    if run_dir is None:
        return None

    for name in ("sidestep_training_config.json", "training_config.json"):
        for tc in (run_dir / "final" / name, run_dir / name):
            if tc.is_file():
                try:
                    return json.loads(tc.read_text(encoding="utf-8"))
                except (json.JSONDecodeError, OSError):
                    pass
    return None


def load_run_curve(
    run_name: str,
    adapters_root: Optional[Path] = None,
    extra_roots: Optional[List[Path]] = None,
    sanitize: Optional[Callable] = None,
) -> List[Dict[str, Any]]:
    """Load the progress JSONL for a run's loss curve."""
    run_dir = find_run_dir(run_name, adapters_root, extra_roots)
    if run_dir is None:
        return []

    progress = run_dir / ".progress.jsonl"
    if not progress.is_file():
        return []

    points: List[Dict[str, Any]] = []
    try:
        with open(progress, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        points.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
    except OSError:
        pass

    if len(points) > _MAX_CURVE_POINTS:
        step = len(points) / _MAX_CURVE_POINTS
        points = [points[int(i * step)] for i in range(_MAX_CURVE_POINTS)]

    if sanitize is not None:
        points = sanitize(points)
    return points


def list_checkpoints(
    run_name: str,
    adapters_root: Optional[Path] = None,
    extra_roots: Optional[List[Path]] = None,
) -> List[Dict[str, Any]]:
    """List checkpoint directories for a run."""
    run_dir = find_run_dir(run_name, adapters_root, extra_roots)
    if run_dir is None:
        return []

    ckpt_dir = run_dir / "checkpoints"
    if not ckpt_dir.is_dir():
        return []

    checkpoints = []
    for child in sorted(ckpt_dir.iterdir()):
        if child.is_dir() and child.name.startswith("epoch_"):
            checkpoints.append({
                "name": child.name,
                "path": str(child),
                "epoch": int(child.name.split("_")[1]) if "_" in child.name else 0,
            })
    return checkpoints


def delete_detected_folder(
    target: Path,
    allowed_roots: List[Path],
) -> Dict[str, Any]:
    """Delete a detected-only history folder via recursive remove.

    Only deletes if the target is within *allowed_roots* and has no
    adapter artifacts (safety net against deleting trained weights).
    """
    if not target.exists() or not target.is_dir():
        return {"ok": False, "error": f"Not a directory: {target}"}
    if not looks_like_run_dir(target):
        return {"ok": False, "error": "Refusing to delete non-run directory"}

    in_scope = False
    for root in allowed_roots:
        try:
            target.resolve(strict=False).relative_to(root)
            in_scope = True
            break
        except ValueError:
            continue
    if not in_scope:
        return {"ok": False, "error": "Path outside history roots"}

    if resolve_run_artifact(target) is not None:
        return {"ok": False, "error": "Refusing to delete run that has saved adapter artifacts"}

    try:
        shutil.rmtree(target)
        return {"ok": True, "path": str(target)}
    except OSError as exc:
        return {"ok": False, "error": str(exc), "path": str(target)}
