"""
Side-Step TUI Widgets

Reusable UI components for the TUI application:
- GPUGauge: Live GPU/VRAM utilization gauge
- LossSparkline: Mini loss graph widget
- ConfigForm: Reusable configuration input forms
- FilePicker: Directory/file browser widget
- LogViewer: Scrolling log panel
- StatusBar: Bottom status bar
"""

from __future__ import annotations

__all__ = [
    "GPUGauge",
    "LossSparkline",
    "FilePicker",
    "LogViewer",
]

# Lazy imports
def __getattr__(name: str):
    if name == "GPUGauge":
        from sidestep_engine.tui.widgets.gpu_gauge import GPUGauge
        return GPUGauge
    elif name == "LossSparkline":
        from sidestep_engine.tui.widgets.loss_sparkline import LossSparkline
        return LossSparkline
    elif name == "FilePicker":
        from sidestep_engine.tui.widgets.file_picker import FilePicker
        return FilePicker
    elif name == "LogViewer":
        from sidestep_engine.tui.widgets.log_viewer import LogViewer
        return LogViewer
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
