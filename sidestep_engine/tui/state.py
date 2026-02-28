"""
Side-Step TUI State Management

Centralized reactive state for the application, including:
- Current training run information
- Recent runs history
- User preferences
- GPU status
"""

from __future__ import annotations

import json
import sys
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional, List, Dict, Any, Callable
from datetime import datetime


@dataclass
class RunInfo:
    """Information about a training run."""
    name: str
    trainer_type: str  # "train"
    status: str  # "running", "completed", "failed", "paused"
    started_at: Optional[str] = None
    finished_at: Optional[str] = None
    current_epoch: int = 0
    total_epochs: int = 100
    current_loss: float = 0.0
    best_loss: float = float("inf")
    best_epoch: int = 0
    output_dir: str = ""
    checkpoint_dir: str = ""
    dataset_dir: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RunInfo":
        return cls(**data)


@dataclass
class GPUStatus:
    """Current GPU status information."""
    name: str = "Unknown"
    vram_used_gb: float = 0.0
    vram_total_gb: float = 0.0
    utilization_percent: float = 0.0
    temperature_c: float = 0.0
    power_w: float = 0.0
    
    @property
    def vram_percent(self) -> float:
        if self.vram_total_gb == 0:
            return 0.0
        return (self.vram_used_gb / self.vram_total_gb) * 100


@dataclass
class UserPreferences:
    """User preferences and settings."""
    default_checkpoint_dir: str = "./checkpoints"
    default_output_dir: str = "./lora_output"
    default_dataset_dir: str = "./datasets"
    theme: str = "dark"
    show_gpu_in_header: bool = True
    auto_save_config: bool = True
    confirm_on_quit: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "UserPreferences":
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


class AppState:
    """
    Centralized application state with reactive updates.
    
    Manages:
    - Current training run
    - Recent runs history
    - GPU status
    - User preferences
    """
    
    @staticmethod
    def _resolve_config_dir() -> Path:
        """Platform-aware config directory."""
        import os
        if sys.platform == "win32":
            base = Path(os.environ.get("APPDATA", Path.home() / "AppData" / "Roaming"))
        else:
            base = Path.home() / ".config"
        return base / "sidestep"
    
    def __init__(self) -> None:
        self.CONFIG_DIR = self._resolve_config_dir()
        self.CONFIG_FILE = self.CONFIG_DIR / "config.json"
        self.HISTORY_FILE = self.CONFIG_DIR / "history.json"
        
        self._current_run: Optional[RunInfo] = None
        self._recent_runs: List[RunInfo] = []
        self._gpu_status = GPUStatus()
        self._preferences = UserPreferences()
        self._listeners: Dict[str, List[Callable]] = {}
        self._last_estimation: Optional[str] = None  # Path to last estimation JSON
        
        # Load saved state
        self._load_config()
        self._load_history()
    
    # =========================================================================
    # Properties
    # =========================================================================
    
    @property
    def current_run(self) -> Optional[RunInfo]:
        return self._current_run
    
    @current_run.setter
    def current_run(self, value: Optional[RunInfo]) -> None:
        self._current_run = value
        self._notify("current_run", value)
    
    @property
    def recent_runs(self) -> List[RunInfo]:
        return self._recent_runs
    
    @property
    def gpu_status(self) -> GPUStatus:
        return self._gpu_status
    
    @gpu_status.setter
    def gpu_status(self, value: GPUStatus) -> None:
        self._gpu_status = value
        self._notify("gpu_status", value)
    
    @property
    def preferences(self) -> UserPreferences:
        return self._preferences
    
    # =========================================================================
    # Run Management
    # =========================================================================
    
    def start_run(self, run: RunInfo) -> None:
        """Start a new training run."""
        run.status = "running"
        run.started_at = datetime.now().isoformat()
        self._current_run = run
        self._notify("run_started", run)
    
    def update_run_progress(
        self,
        epoch: int,
        loss: float,
        best_loss: Optional[float] = None,
        best_epoch: Optional[int] = None,
    ) -> None:
        """Update the current run's progress."""
        if self._current_run is None:
            return
        
        self._current_run.current_epoch = epoch
        self._current_run.current_loss = loss
        
        if best_loss is not None:
            self._current_run.best_loss = best_loss
        if best_epoch is not None:
            self._current_run.best_epoch = best_epoch
        
        # Track best automatically
        if loss < self._current_run.best_loss:
            self._current_run.best_loss = loss
            self._current_run.best_epoch = epoch
        
        self._notify("run_progress", self._current_run)
    
    def complete_run(self, success: bool = True) -> None:
        """Mark the current run as completed."""
        if self._current_run is None:
            return
        
        self._current_run.status = "completed" if success else "failed"
        self._current_run.finished_at = datetime.now().isoformat()
        
        # Add to history
        self._recent_runs.insert(0, self._current_run)
        self._recent_runs = self._recent_runs[:20]  # Keep last 20
        self._save_history()
        
        self._notify("run_completed", self._current_run)
        self._current_run = None
    
    def pause_run(self) -> None:
        """Pause the current run."""
        if self._current_run is None:
            return
        self._current_run.status = "paused"
        self._notify("run_paused", self._current_run)
    
    def resume_run(self) -> None:
        """Resume a paused run."""
        if self._current_run is None:
            return
        self._current_run.status = "running"
        self._notify("run_resumed", self._current_run)
    
    # =========================================================================
    # GPU Status
    # =========================================================================
    
    def update_gpu_status(
        self,
        vram_used_gb: float,
        vram_total_gb: float,
        utilization: float = 0.0,
        temperature: float = 0.0,
        power: float = 0.0,
        name: str = "",
    ) -> None:
        """Update GPU status information."""
        if name:
            self._gpu_status.name = name
        self._gpu_status.vram_used_gb = vram_used_gb
        self._gpu_status.vram_total_gb = vram_total_gb
        self._gpu_status.utilization_percent = utilization
        self._gpu_status.temperature_c = temperature
        self._gpu_status.power_w = power
        self._notify("gpu_status", self._gpu_status)
    
    # =========================================================================
    # Preferences
    # =========================================================================
    
    def update_preferences(self, **kwargs) -> None:
        """Update user preferences."""
        for key, value in kwargs.items():
            if hasattr(self._preferences, key):
                setattr(self._preferences, key, value)
        self._save_config()
        self._notify("preferences", self._preferences)
    
    # =========================================================================
    # Event System
    # =========================================================================
    
    def subscribe(self, event: str, callback: Callable) -> None:
        """Subscribe to state changes."""
        if event not in self._listeners:
            self._listeners[event] = []
        self._listeners[event].append(callback)
    
    def unsubscribe(self, event: str, callback: Callable) -> None:
        """Unsubscribe from state changes."""
        if event in self._listeners:
            try:
                self._listeners[event].remove(callback)
            except ValueError:
                pass
    
    def _notify(self, event: str, data: Any) -> None:
        """Notify all listeners of a state change."""
        for callback in self._listeners.get(event, []):
            try:
                callback(data)
            except Exception:
                pass  # Don't let listener errors crash the app
    
    # =========================================================================
    # Persistence
    # =========================================================================
    
    def _ensure_config_dir(self) -> None:
        """Ensure the config directory exists."""
        self.CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    
    def _load_config(self) -> None:
        """Load configuration from disk."""
        if not self.CONFIG_FILE.exists():
            return
        
        try:
            with open(self.CONFIG_FILE) as f:
                data = json.load(f)
            self._preferences = UserPreferences.from_dict(data.get("preferences", {}))
        except Exception:
            pass  # Use defaults on error
    
    def _save_config(self) -> None:
        """Save configuration to disk."""
        self._ensure_config_dir()
        try:
            data = {"preferences": self._preferences.to_dict()}
            with open(self.CONFIG_FILE, "w") as f:
                json.dump(data, f, indent=2)
        except Exception:
            pass  # Silently fail
    
    def _load_history(self) -> None:
        """Load run history from disk."""
        if not self.HISTORY_FILE.exists():
            return
        
        try:
            with open(self.HISTORY_FILE) as f:
                data = json.load(f)
            self._recent_runs = [
                RunInfo.from_dict(run) for run in data.get("runs", [])
            ]
        except Exception:
            pass  # Use empty history on error
    
    def _save_history(self) -> None:
        """Save run history to disk."""
        self._ensure_config_dir()
        try:
            data = {"runs": [run.to_dict() for run in self._recent_runs]}
            with open(self.HISTORY_FILE, "w") as f:
                json.dump(data, f, indent=2)
        except Exception:
            pass  # Silently fail
    
    # =========================================================================
    # Utility
    # =========================================================================
    
    def get_stats(self) -> Dict[str, Any]:
        """Get quick statistics for the dashboard."""
        completed = sum(1 for r in self._recent_runs if r.status == "completed")
        failed = sum(1 for r in self._recent_runs if r.status == "failed")

        return {
            "total_runs": len(self._recent_runs),
            "completed_runs": completed,
            "failed_runs": failed,
            "has_active_run": self._current_run is not None,
        }

    # =========================================================================
    # Estimation Results
    # =========================================================================

    def get_last_estimation_modules(self) -> Optional[List[str]]:
        """Load top module names from the last estimation JSON, if available."""
        if not self._last_estimation:
            return None
        path = Path(self._last_estimation)
        if not path.is_file():
            return None
        try:
            with open(path) as f:
                data = json.load(f)
            return [
                item.get("module", item.get("name", ""))
                for item in data
                if item.get("module") or item.get("name")
            ]
        except Exception:
            return None

    def save_last_paths(
        self,
        checkpoint_dir: str = "",
        dataset_dir: str = "",
        output_dir: str = "",
    ) -> None:
        """Persist commonly used paths so they auto-populate next session."""
        updates = {}
        if checkpoint_dir:
            updates["default_checkpoint_dir"] = checkpoint_dir
        if dataset_dir:
            updates["default_dataset_dir"] = dataset_dir
        if output_dir:
            updates["default_output_dir"] = output_dir
        if updates:
            self.update_preferences(**updates)
