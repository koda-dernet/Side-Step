"""
Log Viewer Widget

Scrolling log panel for training output with:
- Auto-scroll to bottom
- Timestamp formatting
- Log level coloring
- Search capability
"""

from __future__ import annotations

from datetime import datetime
from typing import Optional, List

from textual.app import ComposeResult
from textual.containers import Container
from textual.widgets import Static, RichLog
from textual.widget import Widget
from textual.binding import Binding

from rich.text import Text


class LogViewer(Widget):
    """
    Scrolling log viewer widget for training output.
    """
    
    DEFAULT_CSS = """
    LogViewer {
        height: 100%;
        border: round $primary 30%;
    }
    
    #log-header {
        dock: top;
        height: 1;
        background: $panel;
        padding: 0 1;
    }
    
    #log-content {
        height: 1fr;
    }
    
    RichLog {
        height: 100%;
    }
    """
    
    BINDINGS = [
        Binding("g", "scroll_home", "Top"),
        Binding("G", "scroll_end", "Bottom"),
        Binding("c", "clear", "Clear"),
    ]
    
    def __init__(
        self,
        title: str = "Log",
        max_lines: int = 1000,
        auto_scroll: bool = True,
        show_timestamps: bool = True,
        name: Optional[str] = None,
        id: Optional[str] = None,
        classes: Optional[str] = None,
    ) -> None:
        """
        Initialize the log viewer.
        
        Args:
            title: Title for the log panel
            max_lines: Maximum lines to keep in buffer
            auto_scroll: Auto-scroll to bottom on new content
            show_timestamps: Prepend timestamps to log entries
            name: Widget name
            id: Widget ID
            classes: CSS classes
        """
        super().__init__(name=name, id=id, classes=classes)
        self.title = title
        self.max_lines = max_lines
        self.auto_scroll = auto_scroll
        self.show_timestamps = show_timestamps
        self._line_count = 0
    
    def compose(self) -> ComposeResult:
        """Compose the log viewer layout."""
        yield Static(self.title, id="log-header")
        with Container(id="log-content"):
            yield RichLog(
                highlight=True,
                markup=True,
                max_lines=self.max_lines,
                id="rich-log",
            )
    
    def write(
        self,
        message: str,
        level: str = "info",
        timestamp: Optional[datetime] = None,
    ) -> None:
        """
        Write a message to the log.
        
        Args:
            message: The message to write
            level: Log level (info, warning, error, debug, success)
            timestamp: Optional timestamp (defaults to now)
        """
        if timestamp is None:
            timestamp = datetime.now()
        
        # Format with timestamp
        if self.show_timestamps:
            time_str = timestamp.strftime("%H:%M:%S")
            prefix = f"[dim][{time_str}][/dim] "
        else:
            prefix = ""
        
        # Apply level coloring
        level_styles = {
            "info": "",
            "warning": "[yellow]",
            "error": "[red bold]",
            "debug": "[dim]",
            "success": "[green]",
        }
        
        style = level_styles.get(level, "")
        end_style = "[/]" if style else ""
        
        formatted = f"{prefix}{style}{message}{end_style}"
        
        try:
            log = self.query_one("#rich-log", RichLog)
            log.write(formatted)
            self._line_count += 1
        except Exception:
            pass
    
    def write_line(self, message: str) -> None:
        """Write a simple line (alias for write with default level)."""
        self.write(message)
    
    def info(self, message: str) -> None:
        """Write an info message."""
        self.write(message, level="info")
    
    def warning(self, message: str) -> None:
        """Write a warning message."""
        self.write(message, level="warning")
    
    def error(self, message: str) -> None:
        """Write an error message."""
        self.write(message, level="error")
    
    def debug(self, message: str) -> None:
        """Write a debug message."""
        self.write(message, level="debug")
    
    def success(self, message: str) -> None:
        """Write a success message."""
        self.write(message, level="success")
    
    def write_separator(self, char: str = "─") -> None:
        """Write a separator line."""
        try:
            log = self.query_one("#rich-log", RichLog)
            # Get approximate width
            width = log.size.width - 4
            if width < 10:
                width = 40
            log.write(f"[dim]{char * width}[/dim]")
        except Exception:
            pass
    
    def write_header(self, title: str) -> None:
        """Write a section header."""
        self.write_separator()
        self.write(f"[bold]{title}[/bold]", level="info")
        self.write_separator()
    
    def clear(self) -> None:
        """Clear all log content."""
        try:
            log = self.query_one("#rich-log", RichLog)
            log.clear()
            self._line_count = 0
        except Exception:
            pass
    
    def action_clear(self) -> None:
        """Action to clear the log."""
        self.clear()
    
    def action_scroll_home(self) -> None:
        """Scroll to top of log."""
        try:
            log = self.query_one("#rich-log", RichLog)
            log.scroll_home()
        except Exception:
            pass
    
    def action_scroll_end(self) -> None:
        """Scroll to bottom of log."""
        try:
            log = self.query_one("#rich-log", RichLog)
            log.scroll_end()
        except Exception:
            pass


class TrainingLogViewer(LogViewer):
    """
    Specialized log viewer for training output.
    
    Adds training-specific formatting and parsing.
    """
    
    def __init__(self, **kwargs) -> None:
        kwargs.setdefault("title", "Training Log")
        super().__init__(**kwargs)
        self._current_epoch = 0
    
    def log_epoch_start(self, epoch: int, total: int) -> None:
        """Log the start of an epoch."""
        self._current_epoch = epoch
        self.write_separator("═")
        self.write(f"Epoch {epoch}/{total} started", level="info")
    
    def log_epoch_end(self, epoch: int, loss: float, lr: float) -> None:
        """Log the end of an epoch."""
        self.success(f"Epoch {epoch} completed | Loss: {loss:.4f} | LR: {lr:.2e}")
    
    def log_step(self, step: int, total_steps: int, loss: float) -> None:
        """Log a training step."""
        self.debug(f"Step {step}/{total_steps} | Loss: {loss:.4f}")
    
    def log_checkpoint_saved(self, path: str) -> None:
        """Log checkpoint save."""
        self.success(f"Checkpoint saved: {path}")
    
    def log_best_model(self, epoch: int, loss: float) -> None:
        """Log new best model."""
        self.success(f"★ New best model @ epoch {epoch} | Loss: {loss:.4f}")
    
    def log_training_complete(self, epochs: int, best_loss: float, time_str: str) -> None:
        """Log training completion."""
        self.write_separator("═")
        self.success(f"Training complete!")
        self.info(f"Total epochs: {epochs}")
        self.info(f"Best loss: {best_loss:.4f}")
        self.info(f"Total time: {time_str}")
        self.write_separator("═")
