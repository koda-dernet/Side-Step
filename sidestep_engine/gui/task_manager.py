"""
Task lifecycle manager for the Side-Step GUI.

Manages long-running operations:
- **Training**: spawned as a subprocess (``sidestep train --config ...``)
  for GPU isolation and crash safety.
- **Preprocessing / PP++ / AI captions**: run as in-process threads using
  existing ``progress_callback`` + ``cancel_check`` patterns.
"""

from __future__ import annotations

import asyncio
import json
import logging
import math
import os
import signal
import subprocess
import sys
import tempfile
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional


from sidestep_engine.core.progress_writer import sanitize_floats as _sanitize_floats

logger = logging.getLogger(__name__)

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent


def _new_task_id(kind: str) -> str:
    """Return a collision-resistant task identifier for websocket routing."""
    return f"{kind}_{time.time_ns()}_{uuid.uuid4().hex[:8]}"


def _remember_history_root_for_output(config: Dict[str, Any]) -> None:
    """Persist non-canonical output roots for history discovery.

    When users override ``output_dir`` outside ``trained_adapters_dir``,
    the GUI should keep scanning those roots so completed runs remain visible.
    """
    output_dir = str(config.get("output_dir") or "").strip()
    if not output_dir:
        return

    out_path = Path(os.path.expandvars(output_dir)).expanduser()
    if not out_path.is_absolute():
        out_path = _PROJECT_ROOT / out_path
    resolved_output = out_path.resolve(strict=False)
    run_name = str(config.get("run_name") or "").strip()
    candidate_root = resolved_output.parent if run_name and resolved_output.name == run_name else resolved_output

    try:
        from sidestep_engine.settings import (
            get_trained_adapters_dir,
            remember_history_output_root,
        )

        adapters_path = Path(os.path.expandvars(get_trained_adapters_dir())).expanduser()
        if not adapters_path.is_absolute():
            adapters_path = _PROJECT_ROOT / adapters_path
        canonical_adapters_root = adapters_path.resolve(strict=False)
        if candidate_root == canonical_adapters_root or canonical_adapters_root in candidate_root.parents:
            return

        remember_history_output_root(str(candidate_root))
    except Exception as exc:
        logger.debug("Could not persist history output root for %s: %s", output_dir, exc)


_OOM_PATTERNS = (
    "CUDA out of memory",
    "torch.OutOfMemoryError",
    "torch.cuda.OutOfMemoryError",
    "RuntimeError: CUDA error: out of memory",
)

_EXIT_CODE_LABELS = {
    -9: "Killed by system (SIGKILL / OOM killer)",
    -11: "Segmentation fault (SIGSEGV)",
    137: "Killed by system (SIGKILL / OOM killer)",
    139: "Segmentation fault",
}


@dataclass
class Task:
    """Metadata for a running or completed task."""
    task_id: str
    kind: str  # "training", "preprocess", "ppplus", "captions"
    process: Optional[subprocess.Popen] = None
    thread: Optional[threading.Thread] = None
    cancel_flag: threading.Event = field(default_factory=threading.Event)
    progress_queue: asyncio.Queue = field(default_factory=lambda: asyncio.Queue(maxsize=500))
    started_at: float = field(default_factory=time.time)
    status: str = "running"  # "running", "done", "failed", "cancelled"
    progress_file: Optional[Path] = None
    config_file: Optional[str] = None  # temp config JSON for training subprocess
    terminal_event_sent: bool = False
    oom_detected: bool = False
    failure_reason: str = ""


_MUTEX_LABELS = {
    "training": "Training",
    "preprocess": "Preprocessing",
    "ppplus": "Preprocessing++",
    "captions": "Caption generation",
}


class TaskManager:
    """Manages subprocess and thread lifecycles for long-running operations."""

    _MAX_COMPLETED_TASKS = 10
    _TASK_TTL_SECONDS = 3600  # 1 hour

    def __init__(self) -> None:
        self._tasks: Dict[str, Task] = {}
        self._training_task: Optional[Task] = None
        self._training_queue: asyncio.Queue = asyncio.Queue(maxsize=1000)
        self._lock = threading.Lock()

    def active_operation(self) -> Optional[str]:
        """Return the kind of the currently running operation, or None."""
        with self._lock:
            if self._training_task and self._training_task.status == "running":
                return "training"
            for t in self._tasks.values():
                if t.status == "running":
                    return t.kind
        return None

    def _check_mutex(self, requested_kind: str) -> Optional[Dict[str, Any]]:
        """Return an error dict if another operation blocks *requested_kind*."""
        active = self.active_operation()
        if active is None:
            return None
        if active == requested_kind == "training":
            return {"error": "Training already running"}
        active_label = _MUTEX_LABELS.get(active, active)
        requested_label = _MUTEX_LABELS.get(requested_kind, requested_kind)
        return {
            "error": f"Cannot start {requested_label} while {active_label} is running. "
                     f"Stop the current operation first.",
        }

    def _cleanup_old_tasks(self) -> None:
        """Remove finished tasks older than TTL, keeping at most _MAX_COMPLETED_TASKS."""
        now = time.time()
        with self._lock:
            finished = [(tid, t) for tid, t in self._tasks.items()
                        if t.status != "running"]
            finished.sort(key=lambda x: x[1].started_at, reverse=True)
            keep_ids = set()
            for tid, t in finished[:self._MAX_COMPLETED_TASKS]:
                if now - t.started_at < self._TASK_TTL_SECONDS:
                    keep_ids.add(tid)
            for tid, t in finished:
                if tid not in keep_ids:
                    del self._tasks[tid]

    # ==================================================================
    # Training (subprocess)
    # ==================================================================

    def start_training(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Spawn a training subprocess from a config dict."""
        blocked = self._check_mutex("training")
        if blocked:
            return blocked

        _remember_history_root_for_output(config)

        task_id = _new_task_id("train")
        output_dir = config.get("output_dir", "")

        # Write config to temp file
        fd, config_path = tempfile.mkstemp(suffix=".json", prefix="sidestep_config_")
        with os.fdopen(fd, "w") as f:
            json.dump(config, f)

        # Build command
        cmd = [
            sys.executable, str(_PROJECT_ROOT / "train.py"),
            "-y", "train", "--config", config_path,
        ]

        logger.info("[TaskManager] Starting training: %s", " ".join(cmd))

        # Force unbuffered output so log lines stream immediately
        env = os.environ.copy()
        env["PYTHONUNBUFFERED"] = "1"

        try:
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                cwd=str(_PROJECT_ROOT),
                env=env,
                text=True,
                bufsize=1,
            )
        except Exception as exc:
            return {"error": str(exc)}

        # Resolve output_dir relative to PROJECT_ROOT (subprocess CWD)
        # so _tail_progress can find .progress.jsonl regardless of GUI server CWD
        if output_dir:
            pf = (_PROJECT_ROOT / output_dir).resolve() / ".progress.jsonl"
        else:
            pf = None

        task = Task(
            task_id=task_id,
            kind="training",
            process=proc,
            config_file=config_path,
            progress_file=pf,
        )

        with self._lock:
            self._tasks[task_id] = task
            self._training_task = task

        # Start reader threads
        threading.Thread(target=self._tail_stdout, args=(task,), daemon=True).start()
        if task.progress_file:
            threading.Thread(target=self._tail_progress, args=(task,), daemon=True).start()

        return {"ok": True, "task_id": task_id}

    def stop_training(self) -> Dict[str, Any]:
        """Send SIGTERM to the training subprocess, escalate to SIGKILL after 5s."""
        with self._lock:
            task = self._training_task
        if not task or task.status != "running" or not task.process:
            return {"error": "No training running"}

        try:
            if sys.platform == "win32":
                task.process.terminate()
            else:
                task.process.send_signal(signal.SIGTERM)
        except OSError:
            pass

        # Escalate to SIGKILL if process doesn't exit within 5 seconds
        def _escalate():
            try:
                task.process.wait(timeout=5)
            except Exception:
                try:
                    task.process.kill()
                except OSError:
                    pass
        threading.Thread(target=_escalate, daemon=True).start()
        return {"ok": True, "task_id": task.task_id}

    async def get_training_update(self) -> Optional[Dict[str, Any]]:
        """Non-blocking fetch of the next training update."""
        try:
            return self._training_queue.get_nowait()
        except asyncio.QueueEmpty:
            return None

    def _tail_stdout(self, task: Task) -> None:
        """Read subprocess stdout line by line, detect OOM, push to queue."""
        assert task.process and task.process.stdout
        try:
            for line in task.process.stdout:
                line = line.rstrip()
                if line:
                    if not task.oom_detected and any(p in line for p in _OOM_PATTERNS):
                        task.oom_detected = True
                        task.failure_reason = "CUDA out of memory"
                    msg = {"type": "log", "msg": line, "ts": time.time()}
                    try:
                        self._training_queue.put_nowait(msg)
                    except asyncio.QueueFull:
                        pass
                    if task.progress_file is None and "Session config:" in line:
                        try:
                            config_path = line.split("Session config:", 1)[1].strip()
                            out_dir = (_PROJECT_ROOT / config_path).resolve().parent.parent
                            pf = out_dir / ".progress.jsonl"
                            task.progress_file = pf
                            threading.Thread(
                                target=self._tail_progress, args=(task,), daemon=True
                            ).start()
                        except (IndexError, ValueError):
                            pass
        except (ValueError, OSError):
            pass  # pipe closed
        finally:
            rc = task.process.wait()
            task.status = "done" if rc == 0 else "failed"

            reason = task.failure_reason
            if not reason and rc != 0:
                reason = _EXIT_CODE_LABELS.get(rc, "")
            if task.oom_detected:
                task.status = "failed"
                reason = reason or "CUDA out of memory"

            try:
                self._training_queue.put_nowait({
                    "type": "status", "status": task.status,
                    "exit_code": rc, "ts": time.time(),
                    "oom": task.oom_detected,
                    "reason": reason,
                })
            except asyncio.QueueFull:
                pass
            if task.config_file:
                try:
                    os.unlink(task.config_file)
                except OSError:
                    pass

    def _tail_progress(self, task: Task) -> None:
        """Tail the .progress.jsonl file and push parsed lines to queue."""
        if not task.progress_file:
            return

        # Wait for the file to appear
        for _ in range(60):
            if task.progress_file.exists() or task.status != "running":
                break
            time.sleep(1)

        if not task.progress_file.exists():
            try:
                self._training_queue.put_nowait({
                    "type": "log", "kind": "warn",
                    "msg": "[warn] Progress file not found — only log output is available",
                    "ts": time.time(),
                })
            except asyncio.QueueFull:
                pass
            return

        try:
            with open(task.progress_file, "r", encoding="utf-8") as f:
                while task.status == "running":
                    line = f.readline()
                    if line:
                        try:
                            data = _sanitize_floats(json.loads(line))
                            data["type"] = "progress"
                            self._training_queue.put_nowait(data)
                        except (json.JSONDecodeError, asyncio.QueueFull):
                            pass
                    else:
                        time.sleep(0.5)
                # Drain remaining lines after completion — sleep briefly
                # so the subprocess can flush final writes before we read.
                time.sleep(0.3)
                for _drain_pass in range(2):
                    for line in f:
                        line = line.strip()
                        if line:
                            try:
                                data = _sanitize_floats(json.loads(line))
                                data["type"] = "progress"
                                self._training_queue.put_nowait(data)
                            except (json.JSONDecodeError, asyncio.QueueFull):
                                pass
                    if _drain_pass == 0:
                        time.sleep(0.2)  # second pass catches late flushes
        except OSError:
            pass

    # ==================================================================
    # In-process tasks (preprocess, PP++, captions)
    # ==================================================================

    def start_preprocess(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Start preprocessing in a background thread."""
        blocked = self._check_mutex("preprocess")
        if blocked:
            return blocked
        self._cleanup_old_tasks()
        task_id = _new_task_id("preprocess")
        task = Task(task_id=task_id, kind="preprocess")

        def _run():
            try:
                from sidestep_engine.data.preprocess import preprocess_audio_files
                result = preprocess_audio_files(
                    audio_dir=config.get("audio_dir"),
                    output_dir=config.get("output_dir") or config.get("tensor_output", ""),
                    checkpoint_dir=config.get("checkpoint_dir", ""),
                    variant=config.get("model_variant", "turbo"),
                    max_duration=config.get("max_duration", 0),
                    dataset_json=config.get("dataset_json"),
                    device=config.get("device", "auto"),
                    precision=config.get("precision", "auto"),
                    normalize=config.get("normalize", "none"),
                    target_db=float(config.get("target_db", -1.0)),
                    target_lufs=float(config.get("target_lufs", -14.0)),
                    progress_callback=lambda cur, tot, msg: _push(task, cur, tot, msg),
                    cancel_check=lambda: task.cancel_flag.is_set(),
                    custom_tag=config.get("trigger_tag") or config.get("custom_tag", ""),
                    tag_position=config.get("tag_position", ""),
                    genre_ratio=int(config.get("genre_ratio", 0)),
                )
                if task.cancel_flag.is_set():
                    task.status = "cancelled"
                    _push_event(task, "cancelled",
                                processed=result.get("processed", 0),
                                failed=result.get("failed", 0),
                                total=result.get("total", 0),
                                output_dir=result.get("output_dir", ""))
                    return
                task.status = "done"
                _push_event(task, "complete",
                            processed=result.get("processed", 0),
                            failed=result.get("failed", 0),
                            total=result.get("total", 0),
                            output_dir=result.get("output_dir", ""))
            except Exception as exc:
                logger.exception("Preprocessing failed")
                task.status = "failed"
                _push_event(task, "fail", str(exc))

        task.thread = threading.Thread(target=_run, daemon=True)
        with self._lock:
            self._tasks[task_id] = task
        task.thread.start()
        return {"ok": True, "task_id": task_id}

    def start_ppplus(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Start Fisher analysis in a background thread."""
        blocked = self._check_mutex("ppplus")
        if blocked:
            return blocked
        self._cleanup_old_tasks()
        task_id = _new_task_id("ppplus")
        task = Task(task_id=task_id, kind="ppplus")

        def _run():
            try:
                from sidestep_engine.analysis.fisher.analysis import run_fisher_analysis
                result = run_fisher_analysis(
                    checkpoint_dir=config.get("checkpoint_dir", ""),
                    dataset_dir=config.get("dataset_dir", ""),
                    variant=config.get("model_variant", "turbo"),
                    base_rank=int(config.get("base_rank", config.get("rank", 64))),
                    rank_min=int(config.get("rank_min", 16)),
                    rank_max=int(config.get("rank_max", 128)),
                    timestep_focus=config.get("timestep_focus", "balanced"),
                    num_runs=int(config.get("num_runs", 3)),
                    batches_per_run=int(config.get("batches_per_run", 20)),
                    convergence_patience=int(config.get("convergence_patience", 5)),
                    progress_callback=lambda cur, tot, msg: _push(task, cur, tot, msg),
                    cancel_check=lambda: task.cancel_flag.is_set(),
                    auto_confirm=True,
                )
                if task.cancel_flag.is_set():
                    task.status = "cancelled"
                    _push_event(task, "cancelled")
                    return
                if result is None:
                    task.status = "failed"
                    _push_event(task, "fail", "PP++ ended without result")
                    return
                task.status = "done"
                _push_event(task, "complete", result=result)
            except Exception as exc:
                logger.exception("PP++ failed")
                task.status = "failed"
                _push_event(task, "fail", str(exc))

        task.thread = threading.Thread(target=_run, daemon=True)
        with self._lock:
            self._tasks[task_id] = task
        task.thread.start()
        return {"ok": True, "task_id": task_id}

    def start_captions(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Start AI caption generation in a background thread."""
        blocked = self._check_mutex("captions")
        if blocked:
            return blocked
        self._cleanup_old_tasks()
        task_id = _new_task_id("captions")
        task = Task(task_id=task_id, kind="captions")

        def _run():
            try:
                from sidestep_engine.data.enrich_song import enrich_one
                from sidestep_engine.data.preprocess_discovery import AUDIO_EXTENSIONS

                def _resolve_audio_files() -> List[Path]:
                    explicit = config.get("audio_files") or []
                    if explicit:
                        return [Path(p) for p in explicit]
                    dataset_dir = str(config.get("dataset_dir") or "").strip()
                    if not dataset_dir:
                        return []
                    base = Path(dataset_dir)
                    if not base.is_dir():
                        return []
                    return sorted(
                        p for p in base.rglob("*")
                        if p.is_file() and p.suffix.lower() in AUDIO_EXTENSIONS
                    )

                def _build_caption_fn() -> Optional[Callable[..., Optional[str]]]:
                    provider = str(config.get("provider") or "skip").lower()
                    if provider == "skip":
                        return None

                    if provider == "gemini":
                        from sidestep_engine.data.caption_provider_gemini import (
                            generate_caption as _generate,
                        )

                        key = str(config.get("gemini_key") or config.get("api_key") or "")
                        model = config.get("gemini_model") or config.get("model")
                        if not key:
                            return None

                        def _run_caption(
                            title: str,
                            artist: str,
                            excerpt: str,
                            audio_path: Path,
                        ) -> Optional[str]:
                            kwargs: Dict[str, Any] = {
                                "audio_path": audio_path,
                                "lyrics_excerpt": excerpt,
                            }
                            if model:
                                kwargs["model"] = model
                            return _generate(title, artist, key, **kwargs)

                        return _run_caption

                    if provider == "openai":
                        from sidestep_engine.data.caption_provider_openai import (
                            generate_caption as _generate,
                        )

                        key = str(config.get("openai_key") or config.get("api_key") or "")
                        model = config.get("openai_model") or config.get("model")
                        base_url = config.get("openai_base") or config.get("base_url")
                        if not key:
                            return None

                        def _run_caption(
                            title: str,
                            artist: str,
                            excerpt: str,
                            audio_path: Path,
                        ) -> Optional[str]:
                            kwargs: Dict[str, Any] = {
                                "audio_path": audio_path,
                                "lyrics_excerpt": excerpt,
                            }
                            if model:
                                kwargs["model"] = model
                            if base_url:
                                kwargs["base_url"] = base_url
                            return _generate(title, artist, key, **kwargs)

                        return _run_caption

                    raise ValueError(f"Unknown caption provider: {provider}")

                def _build_lyrics_fn() -> Optional[Callable[[str, str], Optional[str]]]:
                    token = str(config.get("genius_token") or "").strip()
                    if not token:
                        return None

                    from sidestep_engine.data.lyrics_provider_genius import fetch_lyrics

                    def _run_lyrics(artist: str, title: str) -> Optional[str]:
                        return fetch_lyrics(artist, title, token)

                    return _run_lyrics

                audio_files = _resolve_audio_files()
                total = len(audio_files)
                stats = {"written": 0, "skipped": 0, "failed": 0}
                if total == 0:
                    task.status = "done"
                    _push_event(task, "complete", result={**stats, "total": 0})
                    return

                caption_fn = _build_caption_fn()
                lyrics_fn = _build_lyrics_fn()
                default_artist = str(config.get("default_artist") or "")
                policy = str(config.get("overwrite") or "fill_missing")

                for i, af in enumerate(audio_files, 1):
                    if task.cancel_flag.is_set():
                        task.status = "cancelled"
                        _push_event(task, "cancelled", result={**stats, "total": total})
                        return

                    result = enrich_one(
                        af,
                        default_artist=default_artist,
                        caption_fn=caption_fn,
                        lyrics_fn=lyrics_fn,
                        policy=policy,
                    )
                    status = str(result.get("status") or "failed")
                    if status not in stats:
                        status = "failed"
                    stats[status] += 1

                    msg = f"{af.name}: {status}"
                    if status == "failed" and result.get("error"):
                        msg = f"{af.name}: failed ({result.get('error')})"

                    _push(
                        task,
                        i,
                        total,
                        msg,
                        written=stats["written"],
                        skipped=stats["skipped"],
                        failed=stats["failed"],
                    )

                if task.cancel_flag.is_set():
                    task.status = "cancelled"
                    _push_event(task, "cancelled", result={**stats, "total": total})
                    return

                task.status = "done"
                _push_event(task, "complete", result={**stats, "total": total})
            except Exception as exc:
                logger.exception("Caption generation failed")
                task.status = "failed"
                _push_event(task, "fail", str(exc))

        task.thread = threading.Thread(target=_run, daemon=True)
        with self._lock:
            self._tasks[task_id] = task
        task.thread.start()
        return {"ok": True, "task_id": task_id}

    def stop_task(self, task_id: str) -> Dict[str, Any]:
        """Cancel an in-process task by setting its cancel flag."""
        with self._lock:
            task = self._tasks.get(task_id)
        if not task:
            return {"error": "Task not found"}
        if task.cancel_flag.is_set():
            return {"ok": True}
        task.cancel_flag.set()
        if task.status == "running":
            task.status = "cancelled"
            _push_event(task, "cancelled")
        return {"ok": True}

    async def get_task_update(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Non-blocking fetch of the next task update."""
        with self._lock:
            task = self._tasks.get(task_id)
        if not task:
            return None
        try:
            return task.progress_queue.get_nowait()
        except asyncio.QueueEmpty:
            return None


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _push(task: Task, current: int, total: int, msg: str, **extra: Any) -> None:
    """Push a progress update to a task's queue."""
    pct = round(current / max(total, 1) * 100)
    data: Dict[str, Any] = {
        "type": "progress", "current": current, "total": total,
        "percent": pct, "msg": msg, "log": msg, "ts": time.time(),
    }
    data.update(extra)
    try:
        task.progress_queue.put_nowait(data)
    except asyncio.QueueFull:
        pass


def _push_event(task: Task, kind: str, msg: str = "", **extra: Any) -> None:
    """Push a status event to a task's queue.

    Maps internal kinds to frontend-expected types:
        "complete" -> "done", "fail" -> "error".
    Terminal events (done/error/cancelled) are only sent once per task.
    """
    _TERMINAL = {"complete", "fail", "cancelled"}
    if kind in _TERMINAL:
        if task.terminal_event_sent:
            return
        task.terminal_event_sent = True

    _TYPE_MAP = {"complete": "done", "fail": "error"}
    wire_type = _TYPE_MAP.get(kind, kind)
    data: Dict[str, Any] = {"type": wire_type, "msg": msg, "ts": time.time()}
    data.update(extra)
    try:
        task.progress_queue.put_nowait(data)
    except asyncio.QueueFull:
        pass
