"""
GPU monitoring for the Side-Step GUI.

Provides a one-shot snapshot and is polled every 2s by the ``/ws/gpu``
WebSocket endpoint.  Falls back gracefully when ``pynvml`` is unavailable
or no GPU is detected.
"""

from __future__ import annotations

import logging
from typing import Any, Dict

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


def get_gpu_snapshot(gpu_index: int = 0) -> Dict[str, Any]:
    """Return a dict with current GPU stats, or a fallback stub."""
    if not _ensure_nvml():
        return {"available": False, "name": "No GPU detected"}

    try:
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
            "name": name,
            "vram_used_mb": mem.used // (1024 * 1024),
            "vram_total_mb": mem.total // (1024 * 1024),
            "vram_free_mb": mem.free // (1024 * 1024),
            "utilization": util.gpu,
            "temperature": temp,
            "power_draw_w": power,
        }
    except Exception as exc:
        logger.debug("GPU snapshot failed: %s", exc)
        return {"available": False, "name": "GPU read error", "error": str(exc)}
