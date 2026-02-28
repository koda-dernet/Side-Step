"""
Cached GPU / VRAM monitor for the live training display.

Wraps ``gpu_utils`` with a time-based cache to avoid hammering the GPU
driver on every render tick.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Optional


@dataclass
class VRAMSnapshot:
    """A point-in-time GPU memory reading."""

    used_mb: float = 0.0
    total_mb: float = 0.0
    global_used_mb: float = 0.0
    """Device-wide used VRAM (all processes), when available."""
    global_free_mb: float = 0.0
    """Device-wide free VRAM, when available."""
    name: str = "unknown"
    timestamp: float = 0.0

    @property
    def used_gb(self) -> float:
        return self.used_mb / 1024.0

    @property
    def total_gb(self) -> float:
        return self.total_mb / 1024.0

    @property
    def global_used_gb(self) -> float:
        return self.global_used_mb / 1024.0

    @property
    def global_free_gb(self) -> float:
        return self.global_free_mb / 1024.0

    @property
    def percent(self) -> float:
        if self.total_mb <= 0:
            return 0.0
        return (self.used_mb / self.total_mb) * 100.0

    @property
    def free_mb(self) -> float:
        return max(0.0, self.total_mb - self.used_mb)


class GPUMonitor:
    """Cached GPU VRAM monitor.

    Queries the device at most every ``interval`` seconds.  Returns the
    last snapshot otherwise.

    Args:
        device: Torch device string (``cuda:0``, ``mps``, ``cpu``, ...).
        interval: Minimum seconds between actual GPU queries.
    """

    def __init__(self, device: str = "cuda:0", interval: float = 5.0) -> None:
        self._device = device
        self._device_type = device.split(":")[0]
        self._interval = interval
        self._last: Optional[VRAMSnapshot] = None
        self._available = self._device_type == "cuda"
        self._name: str = ""
        self._total_mb: float = 0.0
        self._init_static()

    # ---- static (one-time) queries -----------------------------------------

    def _init_static(self) -> None:
        """Cache device name and total VRAM (these don't change)."""
        if not self._available:
            return
        try:
            import torch

            idx = self._cuda_idx()
            self._name = torch.cuda.get_device_name(idx)
            self._total_mb = (
                torch.cuda.get_device_properties(idx).total_memory / (1024 ** 2)
            )
        except Exception:
            self._available = False

    def _cuda_idx(self) -> int:
        if ":" in self._device:
            return int(self._device.split(":")[1])
        return 0

    # ---- public API ---------------------------------------------------------

    @property
    def available(self) -> bool:
        """``True`` when VRAM monitoring is possible."""
        return self._available

    def snapshot(self) -> VRAMSnapshot:
        """Return the latest VRAM snapshot, potentially cached."""
        now = time.monotonic()

        if self._last is not None and (now - self._last.timestamp) < self._interval:
            return self._last

        if not self._available:
            snap = VRAMSnapshot(timestamp=now)
            self._last = snap
            return snap

        try:
            import torch

            idx = self._cuda_idx()
            torch.cuda.synchronize(idx)
            reserved = torch.cuda.memory_reserved(idx) / (1024 ** 2)
            free_bytes, total_bytes = torch.cuda.mem_get_info(idx)
            global_free_mb = free_bytes / (1024 ** 2)
            global_used_mb = max(0.0, (total_bytes - free_bytes) / (1024 ** 2))
            snap = VRAMSnapshot(
                used_mb=reserved,
                total_mb=self._total_mb,
                global_used_mb=global_used_mb,
                global_free_mb=global_free_mb,
                name=self._name,
                timestamp=now,
            )
        except Exception:
            snap = VRAMSnapshot(
                total_mb=self._total_mb,
                name=self._name,
                timestamp=now,
            )

        self._last = snap
        return snap

    def peak_mb(self) -> float:
        """Return peak allocated VRAM in MiB (CUDA only)."""
        if not self._available:
            return 0.0
        try:
            import torch
            return torch.cuda.max_memory_allocated(self._cuda_idx()) / (1024 ** 2)
        except Exception:
            return 0.0
