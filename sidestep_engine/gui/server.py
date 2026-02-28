"""
FastAPI application for the Side-Step GUI.

Serves the frontend static files and provides REST + WebSocket endpoints
for training control, file browsing, settings management, and real-time
telemetry.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import sys
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

logger = logging.getLogger(__name__)

# Resolve frontend directory (Side-Step_BETA1/frontend/)
_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_FRONTEND_DIR = _PROJECT_ROOT / "frontend"


def _resolve_server_path(path: str) -> Path:
    """Resolve frontend-provided paths consistently on the backend.

    Delegates to :func:`file_ops._resolve_gui_path` for a single
    implementation.
    """
    from sidestep_engine.gui.file_ops import _resolve_gui_path
    return _resolve_gui_path(path or "")


# ---------------------------------------------------------------------------
# Pydantic request/response models
# ---------------------------------------------------------------------------

class SettingsUpdate(BaseModel):
    """Partial settings update."""
    data: Dict[str, Any]


class BrowseRequest(BaseModel):
    path: str
    dirs_only: bool = False


class TrainStartRequest(BaseModel):
    config: Dict[str, Any]


class SidecarUpdate(BaseModel):
    path: str
    data: Dict[str, Any]


class PresetSave(BaseModel):
    name: str
    data: Dict[str, Any]


class TriggerTagBulk(BaseModel):
    paths: List[str]
    tag: str
    position: str = "prepend"


class ValidateKeyRequest(BaseModel):
    provider: str
    key: str
    model: Optional[str] = None
    base_url: Optional[str] = None


class MixDatasetRequest(BaseModel):
    source_root: str
    destination_root: str
    mix_name: str
    files: List[str]


class LinkSourceAudioRequest(BaseModel):
    tensor_name: str
    audio_path: str


class OpenFolderRequest(BaseModel):
    path: str


class VRAMEstimateRequest(BaseModel):
    batch_size: int = 1
    chunk_duration: Optional[int] = None
    rank: int = 64
    gradient_checkpointing_ratio: float = 1.0
    adapter_type: str = "lora"
    optimizer_type: str = "adamw8bit"
    target_mlp: bool = True
    offload_encoder: bool = True


class TrainValidateRequest(BaseModel):
    """Pre-flight validation before training starts."""
    checkpoint_dir: str
    model_variant: str
    dataset_dir: str
    adapter_type: str = "lora"
    batch_size: int = 1
    gradient_checkpointing_ratio: float = 1.0
    learning_rate: float = 3e-4
    output_dir: str = ""


class PreprocessStartRequest(BaseModel):
    """Typed model for preprocessing start."""
    config: Dict[str, Any]


class PPPlusStartRequest(BaseModel):
    """Typed model for PP++ start."""
    config: Dict[str, Any]


# ---------------------------------------------------------------------------
# Module-level shared state (survives ASGI middleware wrapping)
# ---------------------------------------------------------------------------
_state: Dict[str, Any] = {}
_shutdown_timer: threading.Timer | None = None

# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------

def create_app(token: str | None = None, port: int = 8770) -> FastAPI:
    """Build and return the FastAPI application.

    Args:
        token: Bearer token for API auth.  If ``None``, a random token is
               generated (and printed to the console so the browser
               fallback can use it).
        port:  Server port (used for Host header validation).
    """
    from sidestep_engine.gui.security import (
        HostValidationMiddleware,
        TokenAuthMiddleware,
        TokenAuthWSMiddleware,
        generate_token,
    )

    if token is None:
        token = generate_token()
    app = FastAPI(title="Side-Step GUI", docs_url=None, redoc_url=None)
    app.state.auth_token = token

    # -- No-cache for static assets (local dev tool, not production) --------
    @app.middleware("http")
    async def no_cache_static(request: Request, call_next):
        response = await call_next(request)
        path = request.url.path
        if path.startswith("/css/") or path.startswith("/js/") or path == "/":
            response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
        return response

    # -- Security middleware (order matters: outermost runs first) ----------
    app.add_middleware(
        CORSMiddleware,
        allow_origins=[
            f"http://127.0.0.1:{port}",
            f"http://localhost:{port}",
        ],
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.add_middleware(TokenAuthMiddleware, token=token)
    app.add_middleware(HostValidationMiddleware, port=port)

    # -- Task manager (shared state) ----------------------------------------
    from sidestep_engine.gui.task_manager import TaskManager
    tm = TaskManager()
    _state["task_manager"] = tm
    _state["auth_token"] = token

    # ======================================================================
    # Static files — serve frontend
    # ======================================================================

    @app.get("/")
    async def index():
        global _shutdown_timer
        # Cancel pending shutdown (user refreshed, not closed)
        if _shutdown_timer is not None:
            _shutdown_timer.cancel()
            _shutdown_timer = None
        from starlette.responses import HTMLResponse
        from sidestep_engine.ui.banner import _MOTTOS
        html = (_FRONTEND_DIR / "index.html").read_text(encoding="utf-8")
        # Inject at start of body so token runs before any <script src="...">
        # NOTE: some pywebview backends may not preserve URL query params,
        # so expose auth token in-page as a fallback for frontend API auth.
        inj_token = _state.get("auth_token") or token
        runtime_globals = (
            f"<script>window.__MOTTOS__={json.dumps(_MOTTOS)};"
            f"window.__SIDESTEP_TOKEN__={json.dumps(inj_token)};"
            f"window.__SIDESTEP_PLATFORM__={json.dumps(sys.platform)};"
            f"</script>\n"
        )
        html = html.replace("<body>", "<body>\n" + runtime_globals, 1)
        if inj_token:
            logger.debug("[Side-Step GUI] Token injected into index.html")
        return HTMLResponse(html)

    @app.post("/api/shutdown")
    async def shutdown():
        """Graceful shutdown with 3s delay (cancelled if page reloads)."""
        global _shutdown_timer
        if _shutdown_timer is not None:
            _shutdown_timer.cancel()
        logger.info("[Side-Step GUI] Shutdown requested, exiting in 3s...")
        _shutdown_timer = threading.Timer(3.0, lambda: os._exit(0))
        _shutdown_timer.daemon = True
        _shutdown_timer.start()
        return JSONResponse({"ok": True})

    # Mount static assets (css/, js/) under /css and /js
    if (_FRONTEND_DIR / "css").is_dir():
        app.mount("/css", StaticFiles(directory=str(_FRONTEND_DIR / "css")), name="css")
    if (_FRONTEND_DIR / "js").is_dir():
        app.mount("/js", StaticFiles(directory=str(_FRONTEND_DIR / "js")), name="js")

    # ======================================================================
    # Settings
    # ======================================================================

    @app.get("/api/settings")
    async def get_settings():
        from sidestep_engine.gui.security import mask_keys
        from sidestep_engine.settings import load_settings
        data = load_settings()
        return JSONResponse(mask_keys(data) if data else {})

    @app.post("/api/settings")
    async def update_settings(body: SettingsUpdate):
        from sidestep_engine.settings import load_settings, save_settings, _default_settings
        current = load_settings() or _default_settings()
        # Skip masked values (e.g. "••••xxxx") so round-trip doesn't corrupt real keys
        _MASK_CHAR = "\u2022"  # bullet used by mask_keys()
        _PATH_KEYS = {"checkpoint_dir", "trained_adapters_dir", "preprocessed_tensors_dir", "audio_dir"}
        filtered = {}
        for k, v in body.data.items():
            if isinstance(v, str) and _MASK_CHAR in v:
                continue
            if isinstance(v, str) and k in _PATH_KEYS:
                v = v.strip()
            filtered[k] = v
        current.update(filtered)
        save_settings(current)
        return JSONResponse({"ok": True})

    @app.get("/api/path-exists")
    async def path_exists(path: str = ""):
        """Lightweight check: does the given directory path exist?"""
        if not path:
            return JSONResponse({"exists": False})
        try:
            from sidestep_engine.gui.file_ops import is_path_allowed
            if not is_path_allowed(path):
                return JSONResponse({"exists": False})
            resolved = _resolve_server_path(path)
            return JSONResponse({"exists": resolved.is_dir()})
        except Exception:
            return JSONResponse({"exists": False})

    # ======================================================================
    # File browsing
    # ======================================================================

    @app.post("/api/browse")
    async def browse_directory(body: BrowseRequest):
        from sidestep_engine.gui.file_ops import browse_dir, is_path_allowed
        if not is_path_allowed(body.path):
            return JSONResponse(
                {"error": "Path outside allowed scope", "entries": []},
                status_code=403,
            )
        return JSONResponse(browse_dir(body.path, dirs_only=body.dirs_only))

    # ======================================================================
    # Models & datasets
    # ======================================================================

    @app.get("/api/models")
    async def list_models(checkpoint_dir: str = ""):
        from sidestep_engine.gui.file_ops import scan_models
        return JSONResponse(scan_models(checkpoint_dir))

    @app.get("/api/datasets")
    async def list_datasets(tensors_dir: str = ""):
        from sidestep_engine.gui.file_ops import scan_tensors
        return JSONResponse(scan_tensors(tensors_dir))

    @app.get("/api/datasets/all")
    async def list_all_datasets():
        from sidestep_engine.gui.file_ops import scan_all_datasets
        return JSONResponse(scan_all_datasets())

    @app.post("/api/datasets/link-audio")
    async def link_source_audio(body: LinkSourceAudioRequest):
        from sidestep_engine.gui.file_ops import link_source_audio
        return JSONResponse(link_source_audio(body.tensor_name, body.audio_path))

    @app.get("/api/fisher-map/status")
    async def get_fisher_map_status(dataset_dir: str = "", model_variant: str = ""):
        from sidestep_engine.gui.file_ops import fisher_map_status
        return JSONResponse(fisher_map_status(dataset_dir, model_variant=model_variant))

    @app.get("/api/dataset/scan")
    async def scan_audio(path: str = ""):
        from sidestep_engine.gui.file_ops import scan_audio_dir
        return JSONResponse(scan_audio_dir(path))

    @app.post("/api/dataset/mix")
    async def create_mix_dataset(body: MixDatasetRequest):
        from sidestep_engine.gui.file_ops import create_mix_dataset
        return JSONResponse(create_mix_dataset(
            source_root=body.source_root,
            destination_root=body.destination_root,
            mix_name=body.mix_name,
            files=body.files,
        ))

    @app.post("/api/open-folder")
    async def open_folder(body: OpenFolderRequest):
        import subprocess
        import sys
        from sidestep_engine.gui.file_ops import is_path_allowed
        folder_raw = body.path
        folder = _resolve_server_path(folder_raw)
        if not folder_raw or not folder.is_dir():
            return JSONResponse({"ok": False, "error": "Invalid path"})
        if not is_path_allowed(folder_raw):
            return JSONResponse(
                {"ok": False, "error": "Path outside allowed scope"},
                status_code=403,
            )
        try:
            if sys.platform == "linux":
                subprocess.Popen(["xdg-open", str(folder)])
            elif sys.platform == "darwin":
                subprocess.Popen(["open", str(folder)])
            elif sys.platform == "win32":
                subprocess.Popen(["explorer", str(folder)])
            else:
                return JSONResponse({"ok": False, "error": f"Unsupported platform: {sys.platform}"})
            return JSONResponse({"ok": True})
        except Exception as exc:
            return JSONResponse({"ok": False, "error": str(exc)})

    # ======================================================================
    # Sidecars
    # ======================================================================

    @app.get("/api/sidecar")
    async def get_sidecar(path: str):
        from sidestep_engine.gui.file_ops import read_sidecar
        return JSONResponse(read_sidecar(path))

    @app.post("/api/sidecar")
    async def save_sidecar(body: SidecarUpdate):
        from sidestep_engine.gui.file_ops import write_sidecar
        write_sidecar(body.path, body.data)
        return JSONResponse({"ok": True})

    @app.post("/api/trigger-tag/bulk")
    async def bulk_trigger_tag(body: TriggerTagBulk):
        from sidestep_engine.gui.file_ops import bulk_apply_trigger_tag
        result = bulk_apply_trigger_tag(body.paths, body.tag, body.position)
        return JSONResponse(result)

    # ======================================================================
    # Presets
    # ======================================================================

    @app.get("/api/presets")
    async def list_presets():
        from sidestep_engine.gui.file_ops import list_presets
        return JSONResponse(list_presets())

    @app.get("/api/presets/{name}")
    async def get_preset(name: str):
        from sidestep_engine.gui.file_ops import load_preset
        data = load_preset(name)
        if data is None:
            return JSONResponse({"error": "not found"}, status_code=404)
        return JSONResponse(data)

    @app.post("/api/presets")
    async def save_preset(body: PresetSave):
        from sidestep_engine.gui.file_ops import save_preset
        save_preset(body.name, body.data)
        return JSONResponse({"ok": True})

    @app.delete("/api/presets/{name}")
    async def delete_preset(name: str):
        from sidestep_engine.gui.file_ops import delete_preset
        result = delete_preset(name)
        return JSONResponse(result)

    # ======================================================================
    # History
    # ======================================================================

    @app.get("/api/history")
    async def list_history():
        from sidestep_engine.gui.file_ops import build_history
        return JSONResponse(build_history())

    @app.get("/api/history/{run_name}/config")
    async def get_run_config(run_name: str):
        from sidestep_engine.gui.file_ops import load_run_config
        data = load_run_config(run_name)
        if data is None:
            return JSONResponse({"error": "not found"}, status_code=404)
        return JSONResponse(data)

    @app.get("/api/history/{run_name}/curve")
    async def get_run_curve(run_name: str):
        from sidestep_engine.gui.file_ops import load_run_curve
        return JSONResponse(load_run_curve(run_name))

    @app.delete("/api/history/folder")
    async def delete_history_folder(path: str = ""):
        from sidestep_engine.gui.file_ops import delete_detected_history_folder
        return JSONResponse(delete_detected_history_folder(path))

    # ======================================================================
    # Resume checkpoints
    # ======================================================================

    @app.get("/api/checkpoints/{run_name}")
    async def list_checkpoints(run_name: str):
        from sidestep_engine.gui.file_ops import list_checkpoints
        return JSONResponse({"checkpoints": list_checkpoints(run_name)})

    # ======================================================================
    # API key validation
    # ======================================================================

    @app.post("/api/validate-key")
    async def validate_api_key(body: ValidateKeyRequest):
        from sidestep_engine.gui.file_ops import validate_key
        result = await asyncio.get_event_loop().run_in_executor(
            None, validate_key, body.provider, body.key, body.model, body.base_url
        )
        return JSONResponse(result)

    # ======================================================================
    # GPU info (one-shot)
    # ======================================================================

    @app.get("/api/gpu")
    async def get_gpu_info():
        from sidestep_engine.gui.gpu_monitor import get_gpu_snapshot
        return JSONResponse(get_gpu_snapshot())

    # ======================================================================
    # VRAM estimation
    # ======================================================================

    @app.post("/api/vram/estimate")
    async def estimate_vram(body: VRAMEstimateRequest):
        from sidestep_engine.core.vram_estimation import (
            estimate_peak_vram_mb, system_vram_used_mb, vram_verdict,
        )
        from sidestep_engine.gui.gpu_monitor import get_gpu_snapshot

        chunk_s = body.chunk_duration if body.chunk_duration is not None else None

        peak, breakdown = estimate_peak_vram_mb(
            checkpointing_ratio=body.gradient_checkpointing_ratio,
            batch_size=body.batch_size,
            chunk_duration_s=chunk_s,
            attn_backend="sdpa",
            offload_encoder=body.offload_encoder,
            adapter_type=body.adapter_type,
            rank=body.rank,
            target_mlp=body.target_mlp,
            optimizer_type=body.optimizer_type,
        )
        gpu = get_gpu_snapshot()
        gpu_total = gpu.get("vram_total_mb", 0)
        gpu_free = gpu.get("vram_free_mb", 0)
        gpu_used = gpu.get("vram_used_mb", 0)
        sys_used = system_vram_used_mb(gpu_total, gpu_free)

        breakdown["gpu_total_mb"] = gpu_total
        breakdown["gpu_free_mb"] = gpu_free
        breakdown["gpu_used_mb"] = gpu_used
        breakdown["system_used_mb"] = sys_used
        breakdown["verdict"] = vram_verdict(peak, gpu_total, system_used_mb=sys_used)
        return JSONResponse(breakdown)

    # ======================================================================
    # Training control
    # ======================================================================

    @app.post("/api/train/validate")
    async def validate_training(body: TrainValidateRequest):
        """Pre-flight validation: checks paths, dataset, VRAM before training."""
        from sidestep_engine.core.dataset_validator import validate_dataset
        from sidestep_engine.gui.file_ops import is_path_allowed

        errors = []
        warnings = []

        if not body.checkpoint_dir or not Path(body.checkpoint_dir).is_dir():
            errors.append("Checkpoint directory not found")
        elif not is_path_allowed(body.checkpoint_dir):
            errors.append("Checkpoint directory outside allowed scope")

        if not body.dataset_dir or not Path(body.dataset_dir).is_dir():
            errors.append("Dataset directory not found")
        elif not is_path_allowed(body.dataset_dir):
            errors.append("Dataset directory outside allowed scope")
        else:
            ds = validate_dataset(
                body.dataset_dir,
                expected_model_variant=body.model_variant,
            )
            if ds.kind in ("empty", "invalid"):
                errors.append(ds.issues[0] if ds.issues else "Dataset is empty or invalid")
            for issue in ds.issues:
                warnings.append(issue)
            if ds.is_stale:
                warnings.append(
                    f"Tensors preprocessed with '{ds.model_hash}' but training "
                    f"with '{body.model_variant}'"
                )

        model_dir = Path(body.checkpoint_dir) / body.model_variant
        if body.checkpoint_dir and Path(body.checkpoint_dir).is_dir():
            from sidestep_engine.core.constants import VARIANT_DIR_MAP
            mapped = VARIANT_DIR_MAP.get(body.model_variant)
            if mapped and (Path(body.checkpoint_dir) / mapped).is_dir():
                model_dir = Path(body.checkpoint_dir) / mapped
            if not model_dir.is_dir():
                errors.append(f"Model variant directory not found: {body.model_variant}")

        if body.output_dir:
            out = Path(body.output_dir)
            if out.exists() and not os.access(str(out), os.W_OK):
                errors.append(f"Output directory is not writable: {body.output_dir}")

        return JSONResponse({
            "valid": len(errors) == 0,
            "errors": errors,
            "warnings": warnings,
        })

    @app.post("/api/train/start")
    async def start_training(body: TrainStartRequest):
        result = tm.start_training(body.config)
        return JSONResponse(result)

    @app.post("/api/train/stop")
    async def stop_training():
        result = tm.stop_training()
        return JSONResponse(result)

    # ======================================================================
    # Task control (preprocess, PP++, captions)
    # ======================================================================

    @app.post("/api/preprocess/start")
    async def start_preprocess(body: PreprocessStartRequest):
        result = tm.start_preprocess(body.config)
        return JSONResponse(result)

    @app.post("/api/ppplus/start")
    async def start_ppplus(body: PPPlusStartRequest):
        result = tm.start_ppplus(body.config)
        return JSONResponse(result)

    @app.post("/api/captions/start")
    async def start_captions(body: Dict[str, Any]):
        result = tm.start_captions(body)
        return JSONResponse(result)

    @app.post("/api/task/{task_id}/stop")
    async def stop_task(task_id: str):
        result = tm.stop_task(task_id)
        return JSONResponse(result)

    # ======================================================================
    # TensorBoard
    # ======================================================================

    @app.post("/api/tensorboard/start")
    async def start_tensorboard(body: Dict[str, Any]):
        import shutil
        import socket
        import subprocess

        log_dir = str(body.get("log_dir", "") or "").strip()
        output_dir = str(body.get("output_dir", "") or "").strip()
        tb_port = int(body.get("port", 6006))

        # Resolve log_dir relative to project root unless absolute.
        # If no explicit log_dir is provided, default to run-local output_dir/runs.
        if log_dir:
            resolved = str(_resolve_server_path(log_dir))
        elif output_dir:
            resolved = str((_resolve_server_path(output_dir) / "runs").resolve())
        else:
            return JSONResponse({
                "ok": False,
                "error": "No log_dir or output_dir provided",
            })

        # Kill existing TB if we own it, so it relaunches with the new logdir
        tb_proc = _state.get("tb_proc")
        if tb_proc is not None:
            try:
                tb_proc.terminate()
                tb_proc.wait(timeout=3)
            except Exception:
                try:
                    tb_proc.kill()
                except Exception:
                    pass
            _state["tb_proc"] = None

        # Try bare tensorboard first, then uvx (matches subprocess launcher)
        _SETUPTOOLS_CONSTRAINT = "setuptools<70"
        if shutil.which("tensorboard"):
            cmd = ["tensorboard", "--logdir", resolved, "--port", str(tb_port), "--host", "127.0.0.1"]
        elif shutil.which("uvx"):
            cmd = ["uvx", "--with", _SETUPTOOLS_CONSTRAINT, "tensorboard", "--logdir", resolved, "--port", str(tb_port), "--host", "127.0.0.1"]
        else:
            return JSONResponse({"ok": False, "error": "Neither tensorboard nor uvx found on PATH"})

        try:
            proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, start_new_session=True)
            _state["tb_proc"] = proc
            restarted = tb_proc is not None
            return JSONResponse({"ok": True, "url": f"http://localhost:{tb_port}", "restarted": restarted, "log_dir": resolved})
        except Exception as exc:
            return JSONResponse({"ok": False, "error": str(exc)})

    # ======================================================================
    # WebSockets — real-time streaming
    # ======================================================================

    @app.websocket("/ws/training")
    async def ws_training(websocket: WebSocket):
        await websocket.accept()
        try:
            while True:
                msg = await tm.get_training_update()
                if msg is not None:
                    await websocket.send_json(msg)
                else:
                    await asyncio.sleep(0.5)
        except WebSocketDisconnect:
            pass

    @app.websocket("/ws/gpu")
    async def ws_gpu(websocket: WebSocket):
        from sidestep_engine.gui.gpu_monitor import get_gpu_snapshot
        await websocket.accept()
        try:
            while True:
                await websocket.send_json(get_gpu_snapshot())
                await asyncio.sleep(2)
        except WebSocketDisconnect:
            pass

    @app.websocket("/ws/task/{task_id}")
    async def ws_task(websocket: WebSocket, task_id: str):
        await websocket.accept()
        try:
            while True:
                msg = await tm.get_task_update(task_id)
                if msg is not None:
                    await websocket.send_json(msg)
                else:
                    await asyncio.sleep(0.5)
        except WebSocketDisconnect:
            pass

    # WebSocket auth runs as raw ASGI middleware — must wrap AFTER all
    # routes are registered so @app.get / @app.websocket decorators work.
    wrapped = TokenAuthWSMiddleware(app, token=token)
    return wrapped
