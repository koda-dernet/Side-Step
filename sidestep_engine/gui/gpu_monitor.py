"""
GPU monitoring for the Side-Step GUI.

Provides a one-shot snapshot and is polled every 2s by the ``/ws/gpu``
WebSocket endpoint.  Falls back gracefully when ``pynvml`` is unavailable
or no GPU is detected.
"""

from __future__ import annotations

import logging
import torch

from typing import Any, Dict, List

logger = logging.getLogger(__name__)

_nvml_initialized = False
_nvml_available = False


def _ensure_nvml() -> bool:
    """Initialize pynvml once. Returns True if usable."""
    global _nvml_initialized, _nvml_available
    if _nvml_initialized:
        return _nvml_available
    _nvml_initialized = True
    try:
        import pynvml
        pynvml.nvmlInit()
        _nvml_available = True
    except Exception:
        _nvml_available = False
    return _nvml_available


def _is_mps_available() -> bool:
    """Check if Apple Silicon MPS is available."""
    return torch.backends.mps.is_available()


def _get_mps_snapshot(gpu_index) -> Dict[str, Any]:
    """Return stats for Apple Silicon MPS using torch.mps."""
    used = torch.mps.driver_allocated_memory()
    limit = torch.mps.recommended_max_memory()
    stats = {
        "available": True,
        "index": gpu_index,
        "name": "Apple Silicon MPS",
        "vram_used_mb": used // (1024 * 1024),
        "vram_total_mb": limit // (1024 * 1024),
        "vram_free_mb": max(0, (limit - used) // (1024 * 1024)),
        "utilization": "0",
        "temperature": "0",
        "power_draw_w": "0",
    }
    return stats


def _snapshot_one(gpu_index: int) -> Dict[str, Any]:
    """Return stats for a single GPU by index."""
    import pynvml
    handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_index)
    mem = pynvml.nvmlDeviceGetMemoryInfo(handle)
    util = pynvml.nvmlDeviceGetUtilizationRates(handle)
    temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
    try:
        power = pynvml.nvmlDeviceGetPowerUsage(handle) // 1000
    except pynvml.NVMLError:
        power = 0
    name = pynvml.nvmlDeviceGetName(handle)
    if isinstance(name, bytes):
        name = name.decode("utf-8")
    return {
        "available": True,
        "index": gpu_index,
        "name": name,
        "vram_used_mb": mem.used // (1024 * 1024),
        "vram_total_mb": mem.total // (1024 * 1024),
        "vram_free_mb": mem.free // (1024 * 1024),
        "utilization": util.gpu,
        "temperature": temp,
        "power_draw_w": power,
    }


def get_all_gpus() -> List[Dict[str, Any]]:
    """Return a list of stats dicts, one per GPU."""
    available_gpus = []

    if _ensure_nvml():
        import pynvml
        try:
            count = pynvml.nvmlDeviceGetCount()
            for i in range(count):
                try:
                    available_gpus.append(_snapshot_one(i))
                except pynvml.NVMLError as exc:
                    logger.debug("GPU %d enumeration failed: %s", i, exc)
        except pynvml.NVMLError as exc:
            logger.debug("GPU enumeration failed: %s", exc)
    
    if _is_mps_available():
        gpu_index = 0 if not len(available_gpus) else available_gpus[-1].get("index", 0) + 1
        available_gpus.append(_get_mps_snapshot(gpu_index))

    return available_gpus


def get_gpu_snapshot(gpu_index: int = 0) -> Dict[str, Any]:
    """Return a dict with current GPU stats, or a fallback stub.

    When multiple GPUs are present, *gpu_index* selects which one.
    The response also includes a ``gpus`` list with all devices so
    the frontend can display a multi-GPU overview.
    """
    try:
        all_gpus = get_all_gpus()
        if not all_gpus:
            return {"available": False, "name": "No GPU detected", "gpus": []}
        # Clamp index to valid range
        idx = gpu_index if gpu_index < len(all_gpus) else 0
        primary = dict(all_gpus[idx])
        primary["gpus"] = all_gpus
        return primary
    except Exception as exc:
        logger.info("GPU snapshot failed: %s", exc)
        return {"available": False, "name": "GPU read error", "error": str(exc), "gpus": []}
