"""
Task lifecycle manager for the Side-Step GUI.

Manages long-running operations:
- **Training**: spawned as a subprocess (``sidestep train --config ...``)
  for GPU isolation and crash safety.
- **Preprocessing / PP++ / AI captions**: run as in-process threads using
  existing ``progress_callback`` + ``cancel_check`` patterns.
"""

from __future__ import annotations

import json
import logging
import math
import os
import queue
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
_MASK_CHAR = "•"


def _is_masked_secret(value: Any) -> bool:
    return isinstance(value, str) and _MASK_CHAR in value


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
    kind: str  # "training", "preprocess", "ppplus", "captions", "audio_analyze"
    process: Optional[subprocess.Popen] = None
    thread: Optional[threading.Thread] = None
    cancel_flag: threading.Event = field(default_factory=threading.Event)
    progress_queue: queue.Queue = field(default_factory=lambda: queue.Queue(maxsize=500))
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
    "audio_analyze": "Audio analysis",
}


class TaskManager:
    """Manages subprocess and thread lifecycles for long-running operations."""

    _MAX_COMPLETED_TASKS = 10
    _TASK_TTL_SECONDS = 3600  # 1 hour

    def __init__(self) -> None:
        self._tasks: Dict[str, Task] = {}
        self._training_task: Optional[Task] = None
        self._training_queue: queue.Queue = queue.Queue(maxsize=1000)
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

        # Flush stale messages from previous run so the new WS doesn't consume them
        while not self._training_queue.empty():
            try:
                self._training_queue.get_nowait()
            except queue.Empty:
                break

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
            # Hide the console window on Windows
            creationflags = subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                cwd=str(_PROJECT_ROOT),
                env=env,
                text=True,
                bufsize=1,
                creationflags=creationflags,
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
        except queue.Empty:
            return None

    def drain_training_updates(self, limit: int = 50) -> List[Dict[str, Any]]:
        """Drain up to *limit* messages from the training queue.

        Returns a list so the WebSocket handler can send them in one
        burst, reducing backpressure and ensuring progress messages
        are not starved by verbose stdout logging.
        """
        batch: List[Dict[str, Any]] = []
        for _ in range(limit):
            try:
                batch.append(self._training_queue.get_nowait())
            except queue.Empty:
                break
        return batch

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
                    except queue.Full:
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

            status_msg = {
                "type": "status", "status": task.status,
                "exit_code": rc, "ts": time.time(),
                "oom": task.oom_detected,
                "reason": reason,
            }
            # Status is critical — force it into the queue even if full
            for _attempt in range(20):
                try:
                    self._training_queue.put_nowait(status_msg)
                    break
                except queue.Full:
                    try:
                        self._training_queue.get_nowait()
                    except queue.Empty:
                        break
                    time.sleep(0.05)
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
            except queue.Full:
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
                            try:
                                self._training_queue.put_nowait(data)
                            except queue.Full:
                                # Progress is critical — drop an old message to make room
                                try:
                                    self._training_queue.get_nowait()
                                except queue.Empty:
                                    pass
                                try:
                                    self._training_queue.put_nowait(data)
                                except queue.Full:
                                    pass
                        except json.JSONDecodeError:
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
                                try:
                                    self._training_queue.put_nowait(data)
                                except queue.Full:
                                    try:
                                        self._training_queue.get_nowait()
                                    except queue.Empty:
                                        pass
                                    try:
                                        self._training_queue.put_nowait(data)
                                    except queue.Full:
                                        pass
                            except json.JSONDecodeError:
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
                    variant=config.get("model_variant", "base"),
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
                    variant=config.get("model_variant", "base"),
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
                    if provider in ("skip", "lyrics_only", "none", "music_flamingo"):
                        return None

                    generation_keys = (
                        ("caption_temperature", "temperature"),
                        ("caption_max_tokens", "max_tokens"),
                        ("caption_top_p", "top_p"),
                        ("caption_presence_penalty", "presence_penalty"),
                        ("caption_frequency_penalty", "frequency_penalty"),
                        ("caption_repetition_penalty", "repetition_penalty"),
                    )
                    generation_kwargs = {
                        target: config.get(source)
                        for source, target in generation_keys
                        if config.get(source) is not None
                    }

                    if provider == "gemini":
                        from sidestep_engine.data.caption_provider_gemini import (
                            generate_caption as _generate,
                        )

                        key = str(config.get("gemini_key") or config.get("api_key") or "").strip()
                        model = config.get("gemini_model") or config.get("model")
                        if not key:
                            return None

                        use_google_search = bool(config.get("gemini_google_search"))

                        def _run_caption(
                            title: str,
                            artist: str,
                            excerpt: str,
                            audio_path: Path,
                        ) -> Optional[str]:
                            kwargs: Dict[str, Any] = {
                                "audio_path": audio_path,
                                "lyrics_excerpt": excerpt,
                                "google_search": use_google_search,
                                **generation_kwargs,
                            }
                            if model:
                                kwargs["model"] = model
                            return _generate(title, artist, key, **kwargs)

                        return _run_caption

                    if provider == "openai":
                        from sidestep_engine.data.caption_provider_openai import (
                            generate_caption as _generate,
                        )

                        key = str(config.get("openai_key") or config.get("api_key") or "").strip()
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
                                **generation_kwargs,
                            }
                            if model:
                                kwargs["model"] = model
                            if base_url:
                                kwargs["base_url"] = base_url
                            return _generate(title, artist, key, **kwargs)

                        return _run_caption

                    if provider in ("local_8-10gb", "local_16gb"):
                        from sidestep_engine.data.caption_provider_local import (
                            generate_caption as _generate_local,
                        )

                        tier = "8-10gb" if provider == "local_8-10gb" else "16gb"
                        allow_cpu_offload = bool(config.get("caption_local_cpu_offload"))

                        _cancel = task.cancel_flag

                        def _run_caption(
                            title: str,
                            artist: str,
                            excerpt: str,
                            audio_path: Path,
                        ) -> Optional[str]:
                            return _generate_local(
                                title, artist,
                                audio_path=audio_path,
                                lyrics_excerpt=excerpt,
                                tier=tier,
                                max_new_tokens=generation_kwargs.get("max_tokens"),
                                temperature=generation_kwargs.get("temperature"),
                                top_p=generation_kwargs.get("top_p"),
                                repetition_penalty=generation_kwargs.get("repetition_penalty"),
                                allow_cpu_offload=allow_cpu_offload,
                                stop_event=_cancel,
                            )

                        return _run_caption

                    raise ValueError(f"Unknown caption provider: {provider}")

                def _build_metadata_fn() -> Optional[Callable[[Path], Optional[Dict[str, str]]]]:
                    provider = str(config.get("metadata_provider") or config.get("provider") or "skip").lower()
                    if provider != "music_flamingo":
                        return None

                    server_url = str(config.get("music_flamingo_url") or "").strip()
                    hf_token = config.get("hf_token")
                    if _is_masked_secret(hf_token) or not str(hf_token or "").strip():
                        from sidestep_engine.settings import get_hf_token
                        hf_token = get_hf_token()
                    hf_token = str(hf_token or "").strip()
                    if not server_url:
                        return None

                    from sidestep_engine.data.metadata_provider_music_flamingo import (
                        fetch_music_flamingo_metadata as _generate_metadata,
                    )

                    def _run_metadata(audio_path: Path) -> Optional[Dict[str, str]]:
                        return _generate_metadata(
                            str(audio_path),
                            server_url=server_url,
                            hf_token=hf_token or None,
                        )

                    return _run_metadata

                def _build_lyrics_fn() -> Optional[Callable[[str, str, Path], Optional[str]]]:
                    lyrics_provider = str(config.get("lyrics_provider") or "").strip().lower()
                    provider = str(config.get("provider") or "").strip().lower()
                    if not lyrics_provider:
                        lyrics_provider = "genius" if provider == "lyrics_only" else "none"

                    if lyrics_provider == "none":
                        return None

                    if lyrics_provider == "genius":
                        token = config.get("genius_token")
                        if _is_masked_secret(token) or not str(token or "").strip():
                            from sidestep_engine.settings import get_genius_api_token
                            token = get_genius_api_token()
                        token = str(token or "").strip()
                        if not token:
                            return None
                        from sidestep_engine.data.lyrics_provider_genius import fetch_lyrics

                        def _run_lyrics(artist: str, title: str, _audio_path: Path) -> Optional[str]:
                            return fetch_lyrics(artist, title, token)

                        return _run_lyrics

                    if lyrics_provider == "transcriber_server":
                        server_url = str(config.get("transcriber_server_url") or "").strip()
                        if not server_url:
                            return None
                        from sidestep_engine.data.lyrics_provider_server import fetch_lyrics_from_server

                        def _run_lyrics(artist: str, title: str, audio_path: Path) -> Optional[str]:
                            return fetch_lyrics_from_server(
                                str(audio_path),
                                server_url=server_url,
                                artist=artist,
                                title=title,
                            )

                        return _run_lyrics

                    if lyrics_provider == "music_flamingo":
                        server_url = str(config.get("music_flamingo_url") or "").strip()
                        hf_token = config.get("hf_token")
                        if _is_masked_secret(hf_token) or not str(hf_token or "").strip():
                            from sidestep_engine.settings import get_hf_token
                            hf_token = get_hf_token()
                        hf_token = str(hf_token or "").strip()
                        if not server_url:
                            return None
                        from sidestep_engine.data.lyrics_provider_music_flamingo import fetch_lyrics_from_music_flamingo

                        def _run_lyrics(artist: str, title: str, audio_path: Path) -> Optional[str]:
                            return fetch_lyrics_from_music_flamingo(
                                str(audio_path),
                                server_url=server_url,
                                artist=artist,
                                title=title,
                                hf_token=hf_token or None,
                            )

                        return _run_lyrics

                    raise ValueError(f"Unknown lyrics provider: {lyrics_provider}")

                audio_files = _resolve_audio_files()
                total = len(audio_files)
                stats = {"written": 0, "skipped": 0, "failed": 0}
                if total == 0:
                    task.status = "done"
                    _push_event(task, "complete", result={**stats, "total": 0})
                    return

                caption_fn = _build_caption_fn()
                metadata_fn = _build_metadata_fn()
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
                        lyrics_fn=(lambda artist, title, af=af: lyrics_fn(artist, title, af)) if lyrics_fn else None,
                        metadata_fn=metadata_fn,
                        policy=policy,
                    )
                    status = str(result.get("status") or "failed")
                    if status not in stats:
                        status = "failed"
                    stats[status] += 1

                    msg = f"{af.name}: {status}"
                    if status == "failed" and result.get("error"):
                        msg = f"{af.name}: failed ({result.get('error')})"
                    elif result.get("warnings"):
                        msg = f"{af.name}: {status} ({'; '.join(result['warnings'])})"

                    if result.get("error_code") == "local_caption_oom":
                        task.oom_detected = True
                        task.failure_reason = str(result.get("error") or "Local caption OOM")
                        _push(
                            task,
                            i,
                            total,
                            msg,
                            written=stats["written"],
                            skipped=stats["skipped"],
                            failed=stats["failed"],
                            error_code="local_caption_oom",
                        )
                        task.cancel_flag.set()
                        task.status = "failed"
                        _push_event(
                            task,
                            "fail",
                            task.failure_reason,
                            result={**stats, "total": total},
                            error_code="local_caption_oom",
                            path=str(af),
                            fatal=True,
                        )
                        return

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
            finally:
                # Free VRAM if a local model was loaded
                provider = str(config.get("provider") or "").lower()
                if provider in ("local_8-10gb", "local_16gb"):
                    try:
                        from sidestep_engine.data.caption_provider_local import (
                            unload_model,
                        )
                        unload_model()
                    except Exception:
                        pass

        task.thread = threading.Thread(target=_run, daemon=True)
        with self._lock:
            self._tasks[task_id] = task
        task.thread.start()
        return {"ok": True, "task_id": task_id}

    def start_audio_analyze(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Start local audio analysis in a background thread."""
        blocked = self._check_mutex("audio_analyze")
        if blocked:
            return blocked
        self._cleanup_old_tasks()
        task_id = _new_task_id("audio_analyze")
        task = Task(task_id=task_id, kind="audio_analyze")

        def _run():
            try:
                from sidestep_engine.analysis.audio_analysis import analyze_audio
                from sidestep_engine.data.preprocess_discovery import AUDIO_EXTENSIONS
                from sidestep_engine.data.sidecar_io import (
                    merge_fields, read_sidecar, sidecar_path_for, write_sidecar,
                )

                # Support explicit file list (from selection) or full directory scan
                explicit_paths = config.get("audio_files") or []
                if explicit_paths:
                    audio_files = [Path(p) for p in explicit_paths if Path(p).is_file()]
                else:
                    dataset_dir = str(config.get("dataset_dir") or "").strip()
                    if not dataset_dir:
                        task.status = "failed"
                        _push_event(task, "fail", "No dataset directory specified")
                        return

                    base = Path(dataset_dir)
                    if not base.is_dir():
                        task.status = "failed"
                        _push_event(task, "fail", f"Not a directory: {dataset_dir}")
                        return

                    audio_files = sorted(
                        p for p in base.rglob("*")
                        if p.is_file() and p.suffix.lower() in AUDIO_EXTENSIONS
                    )
                total = len(audio_files)
                if total == 0:
                    task.status = "done"
                    _push_event(task, "complete", result={
                        "written": 0, "skipped": 0, "failed": 0, "total": 0,
                    })
                    return

                device = str(config.get("device") or "auto")
                policy = str(config.get("policy") or "fill_missing")
                mode = str(config.get("mode") or "mid")
                n_chunks = int(config.get("chunks") or 5)
                stats = {"written": 0, "skipped": 0, "failed": 0}

                for i, af in enumerate(audio_files, 1):
                    if task.cancel_flag.is_set():
                        task.status = "cancelled"
                        _push_event(task, "cancelled", result={**stats, "total": total})
                        return

                    try:
                        result = analyze_audio(af, device=device, mode=mode, n_chunks=n_chunks)
                        # Strip confidence (GUI-only, not for sidecars)
                        sidecar_fields = {
                            k: v for k, v in result.items()
                            if k != "confidence"
                        }
                        if not sidecar_fields:
                            stats["skipped"] += 1
                            _push(task, i, total, f"{af.name}: skipped (no results)",
                                  **stats)
                            continue

                        sc_path = sidecar_path_for(af)
                        existing = read_sidecar(sc_path)

                        if policy == "fill_missing":
                            if all(existing.get(k, "").strip()
                                   for k in ("bpm", "key", "signature")):
                                stats["skipped"] += 1
                                _push(task, i, total,
                                      f"{af.name}: skipped (already populated)",
                                      **stats)
                                continue

                        merged = merge_fields(existing, sidecar_fields, policy=policy)
                        write_sidecar(sc_path, merged)
                        stats["written"] += 1
                        parts = ", ".join(f"{k}={v}" for k, v in sidecar_fields.items())
                        _push(task, i, total, f"{af.name}: written ({parts})",
                              **stats)

                    except Exception as exc:
                        stats["failed"] += 1
                        _push(task, i, total,
                              f"{af.name}: failed ({exc})", **stats)
                        logger.exception("Audio analysis failed for %s", af)

                if task.cancel_flag.is_set():
                    task.status = "cancelled"
                    _push_event(task, "cancelled", result={**stats, "total": total})
                    return

                task.status = "done"
                _push_event(task, "complete", result={**stats, "total": total})
            except Exception as exc:
                logger.exception("Audio analysis failed")
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
        except queue.Empty:
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
    except queue.Full:
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
    except queue.Full:
        pass
