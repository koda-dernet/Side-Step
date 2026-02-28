"""GPU detection, VRAM query, and device selection utilities."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import torch

logger = logging.getLogger(__name__)

@dataclass
class GPUInfo:
    """Describes the detected accelerator."""
    device: str
    device_type: str
    name: str
    vram_total_mb: Optional[float] = None
    vram_free_mb: Optional[float] = None
    precision: str = "fp32"


def _best_cuda_device() -> int:
    """Return the CUDA device index with the most total VRAM."""
    try:
        count = torch.cuda.device_count()
        if count <= 1:
            return 0
        best_idx, best_mem = 0, 0
        for i in range(count):
            mem = torch.cuda.get_device_properties(i).total_memory
            if mem > best_mem:
                best_idx, best_mem = i, mem
        if best_idx != 0:
            name_0 = torch.cuda.get_device_name(0)
            name_best = torch.cuda.get_device_name(best_idx)
            logger.info(
                "[INFO] Multiple CUDA devices found (%d). "
                "Selected cuda:%d (%s, %.0f MiB) over cuda:0 (%s).",
                count, best_idx, name_best, best_mem / (1024 ** 2), name_0,
            )
        return best_idx
    except Exception:
        return 0


def detect_gpu(requested_device: str = "auto", requested_precision: str = "auto") -> GPUInfo:
    """Detect the best available accelerator (CUDA > MPS > XPU > CPU)."""
    device_type: str
    device_str: str
    name: str
    vram_total: Optional[float] = None
    vram_free: Optional[float] = None

    if requested_device == "auto":
        if torch.cuda.is_available():
            device_str = f"cuda:{_best_cuda_device()}"
            device_type = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device_str = "mps"
            device_type = "mps"
        elif hasattr(torch, "xpu") and torch.xpu.is_available():
            device_str = "xpu:0"
            device_type = "xpu"
        else:
            device_str = "cpu"
            device_type = "cpu"
        logger.info("[INFO] Auto-detected device: %s", device_str)
    else:
        device_type = requested_device.split(":")[0]
        # Bare "cuda" or "xpu" without index → append ":0" so
        # torch.device() / Fabric always gets an explicit ordinal.
        if device_type in ("cuda", "xpu") and ":" not in requested_device:
            device_str = f"{device_type}:0"
        else:
            device_str = requested_device

    # Resolve name + VRAM
    if device_type == "cuda":
        idx = 0
        if ":" in device_str:
            idx = int(device_str.split(":")[1])
        name = torch.cuda.get_device_name(idx)
        vram_total = torch.cuda.get_device_properties(idx).total_memory / (1024 ** 2)
        vram_free = _cuda_free_mb(idx)
    elif device_type == "mps":
        name = "Apple MPS"
    elif device_type == "xpu":
        idx = 0
        if ":" in device_str:
            idx = int(device_str.split(":")[1])
        name = f"Intel XPU:{idx}"
        if hasattr(torch.xpu, "get_device_properties"):
            props = torch.xpu.get_device_properties(idx)
            vram_total = getattr(props, "total_memory", 0) / (1024 ** 2)
    else:
        name = "CPU"

    # Resolve precision
    if requested_precision == "auto":
        if device_type in ("cuda", "xpu"):
            precision = "bf16"
        elif device_type == "mps":
            precision = "fp16"
        else:
            precision = "fp32"
    else:
        precision = requested_precision

    return GPUInfo(
        device=device_str,
        device_type=device_type,
        name=name,
        vram_total_mb=vram_total,
        vram_free_mb=vram_free,
        precision=precision,
    )


def _cuda_free_mb(idx: int = 0) -> Optional[float]:
    """Return free CUDA memory in MiB, or None on failure."""
    try:
        torch.cuda.synchronize(idx)
        free, _total = torch.cuda.mem_get_info(idx)
        return free / (1024 ** 2)
    except (RuntimeError, AssertionError):
        return None


def get_available_vram_mb(device: str = "auto") -> Optional[float]:
    """Query free VRAM in MiB on *device*. Returns None if unavailable."""
    info = detect_gpu(requested_device=device)
    return info.vram_free_mb


_BYTES_PER_BATCH_ESTIMATE_BF16: float = 1200.0  # ~1.2 GiB per sample


def get_gpu_info(device: str = "auto") -> dict:
    """Return GPU info dict for TUI widgets (deprecated)."""
    try:
        info = detect_gpu(requested_device=device)
        total_mb = info.vram_total_mb or 0
        free_mb = info.vram_free_mb or 0
        used_mb = max(0, total_mb - free_mb)
        return {
            "name": info.name,
            "vram_used_gb": used_mb / 1024,
            "vram_total_gb": total_mb / 1024,
            "utilization": 0,  # nvidia-smi would be needed for live util
            "temperature": 0,
            "power": 0,
        }
    except (RuntimeError, OSError, ValueError, AssertionError):
        return {
            "name": "Unknown",
            "vram_used_gb": 0,
            "vram_total_gb": 0,
            "utilization": 0,
            "temperature": 0,
            "power": 0,
        }


def estimate_batch_budget(
    device: str = "auto",
    safety_factor: float = 0.8,
    min_batches: int = 4,
    max_batches: int = 64,
) -> int:
    """Estimate how many batches fit in available VRAM."""
    free_mb = get_available_vram_mb(device)
    if free_mb is None:
        logger.info("[INFO] VRAM unknown -- using minimum batch budget of %d", min_batches)
        return min_batches

    usable_mb = free_mb * safety_factor
    # Subtract ~4 GiB for the model weights themselves
    usable_mb = max(0.0, usable_mb - 4096.0)
    n_batches = int(usable_mb / _BYTES_PER_BATCH_ESTIMATE_BF16)
    n_batches = max(min_batches, min(n_batches, max_batches))

    logger.info(
        "[INFO] Estimation budget: %d batches (%.0f MiB free, %.0f MiB usable)",
        n_batches,
        free_mb,
        usable_mb,
    )
    return n_batches
