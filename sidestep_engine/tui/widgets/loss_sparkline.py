"""
Loss Sparkline Widget

Mini loss graph showing training progress with:
- ASCII-based sparkline visualization
- Rolling window of recent values
- Best/current/average annotations
"""

from __future__ import annotations

from typing import Optional, List
from collections import deque

from textual.app import ComposeResult
from textual.containers import Container, Horizontal, Vertical
from textual.widgets import Static, Sparkline, Label
from textual.widget import Widget
from textual.reactive import reactive


# Sparkline characters for different resolutions
SPARK_CHARS_8 = "▁▂▃▄▅▆▇█"
SPARK_CHARS_4 = "▁▃▅█"


class LossSparkline(Widget):
    """
    Mini loss graph widget showing training loss over time.
    
    Uses ASCII sparkline characters for a compact visualization.
    """
    
    DEFAULT_CSS = """
    LossSparkline {
        height: auto;
        min-height: 6;
        border: round $primary 30%;
        padding: 1;
    }
    
    #sparkline-title {
        text-style: bold;
        margin-bottom: 1;
    }
    
    #sparkline-graph {
        height: 3;
        width: 100%;
    }
    
    #sparkline-axis {
        height: 1;
        color: $text-muted;
    }
    
    #sparkline-stats {
        height: auto;
        margin-top: 1;
        layout: horizontal;
    }
    
    .stat-item {
        width: 1fr;
        text-align: center;
    }
    
    .stat-label {
        color: $text-muted;
    }
    
    .stat-value {
        text-style: bold;
    }
    
    .current-loss {
        color: $primary;
    }
    
    .best-loss {
        color: $success;
    }
    """
    
    # Reactive properties
    current_loss: reactive[float] = reactive(0.0)
    best_loss: reactive[float] = reactive(float("inf"))
    best_epoch: reactive[int] = reactive(0)
    
    def __init__(
        self,
        max_points: int = 100,
        title: str = "Loss",
        show_stats: bool = True,
        name: Optional[str] = None,
        id: Optional[str] = None,
        classes: Optional[str] = None,
    ) -> None:
        """
        Initialize the loss sparkline.
        
        Args:
            max_points: Maximum number of data points to display
            title: Title for the widget
            show_stats: Whether to show current/best/avg stats
            name: Widget name
            id: Widget ID
            classes: CSS classes
        """
        super().__init__(name=name, id=id, classes=classes)
        self.max_points = max_points
        self.title = title
        self.show_stats = show_stats
        self._data: deque[float] = deque(maxlen=max_points)
    
    def compose(self) -> ComposeResult:
        """Compose the sparkline layout."""
        yield Static(self.title, id="sparkline-title")
        yield Static("", id="sparkline-graph")
        yield Static("", id="sparkline-axis")
        
        if self.show_stats:
            with Horizontal(id="sparkline-stats"):
                with Vertical(classes="stat-item"):
                    yield Static("Current", classes="stat-label")
                    yield Static("--", id="stat-current", classes="stat-value current-loss")
                with Vertical(classes="stat-item"):
                    yield Static("Best", classes="stat-label")
                    yield Static("--", id="stat-best", classes="stat-value best-loss")
                with Vertical(classes="stat-item"):
                    yield Static("Average", classes="stat-label")
                    yield Static("--", id="stat-avg", classes="stat-value")
    
    def add_value(self, value: float, epoch: Optional[int] = None) -> None:
        """
        Add a new loss value to the sparkline.
        
        Args:
            value: The loss value to add
            epoch: Optional epoch number for tracking best
        """
        self._data.append(value)
        self.current_loss = value
        
        if value < self.best_loss:
            self.best_loss = value
            if epoch is not None:
                self.best_epoch = epoch
        
        self._update_display()
    
    def add_values(self, values: List[float]) -> None:
        """Add multiple values at once."""
        for v in values:
            self._data.append(v)
        
        if values:
            self.current_loss = values[-1]
            min_val = min(values)
            if min_val < self.best_loss:
                self.best_loss = min_val
        
        self._update_display()
    
    def clear(self) -> None:
        """Clear all data."""
        self._data.clear()
        self.current_loss = 0.0
        self.best_loss = float("inf")
        self.best_epoch = 0
        self._update_display()
    
    def _update_display(self) -> None:
        """Update the sparkline and stats display."""
        if not self._data:
            self.query_one("#sparkline-graph", Static).update("")
            return
        
        # Generate sparkline
        sparkline = self._generate_sparkline()
        self.query_one("#sparkline-graph", Static).update(sparkline)
        
        # Generate axis
        axis = self._generate_axis()
        self.query_one("#sparkline-axis", Static).update(axis)
        
        # Update stats
        if self.show_stats:
            self._update_stats()
    
    def _generate_sparkline(self) -> str:
        """Generate ASCII sparkline from data."""
        if not self._data:
            return ""
        
        data = list(self._data)
        
        # Get terminal width (approximate)
        try:
            graph = self.query_one("#sparkline-graph", Static)
            width = graph.size.width - 2  # Account for padding
        except Exception:
            width = 60
        
        # Sample data to fit width
        if len(data) > width:
            step = len(data) / width
            sampled = [data[int(i * step)] for i in range(width)]
        else:
            sampled = data
        
        if not sampled:
            return ""
        
        # Normalize to sparkline range
        min_val = min(sampled)
        max_val = max(sampled)
        
        if max_val == min_val:
            # All values the same
            return SPARK_CHARS_8[4] * len(sampled)
        
        # Map to character indices
        chars = SPARK_CHARS_8
        range_val = max_val - min_val
        
        sparkline = ""
        for v in sampled:
            normalized = (v - min_val) / range_val
            idx = int(normalized * (len(chars) - 1))
            idx = max(0, min(len(chars) - 1, idx))
            sparkline += chars[idx]
        
        return sparkline
    
    def _generate_axis(self) -> str:
        """Generate the axis labels."""
        if not self._data:
            return ""
        
        min_val = min(self._data)
        max_val = max(self._data)
        
        try:
            graph = self.query_one("#sparkline-graph", Static)
            width = graph.size.width - 2
        except Exception:
            width = 60
        
        # Create axis with min/max labels
        min_str = f"{min_val:.3f}"
        max_str = f"{max_val:.3f}"
        
        # Calculate spacing
        padding = width - len(min_str) - len(max_str)
        if padding < 0:
            padding = 0
        
        return f"{min_str}{' ' * padding}{max_str}"
    
    def _update_stats(self) -> None:
        """Update the statistics display."""
        try:
            # Current
            current_text = f"{self.current_loss:.4f}"
            self.query_one("#stat-current", Static).update(current_text)
            
            # Best
            if self.best_loss < float("inf"):
                best_text = f"{self.best_loss:.4f}"
                if self.best_epoch > 0:
                    best_text += f" @ {self.best_epoch}"
            else:
                best_text = "--"
            self.query_one("#stat-best", Static).update(best_text)
            
            # Average
            if self._data:
                avg = sum(self._data) / len(self._data)
                avg_text = f"{avg:.4f}"
            else:
                avg_text = "--"
            self.query_one("#stat-avg", Static).update(avg_text)
        except Exception:
            pass


class MiniSparkline(Widget):
    """
    Very compact sparkline for embedding in tables or lists.
    Just shows the graph, no labels.
    """
    
    DEFAULT_CSS = """
    MiniSparkline {
        height: 1;
        width: auto;
        min-width: 20;
    }
    """
    
    def __init__(
        self,
        max_points: int = 20,
        name: Optional[str] = None,
        id: Optional[str] = None,
        classes: Optional[str] = None,
    ) -> None:
        super().__init__(name=name, id=id, classes=classes)
        self.max_points = max_points
        self._data: deque[float] = deque(maxlen=max_points)
    
    def compose(self) -> ComposeResult:
        yield Static("", id="mini-spark")
    
    def add_value(self, value: float) -> None:
        """Add a value to the sparkline."""
        self._data.append(value)
        self._update()
    
    def _update(self) -> None:
        """Update the display."""
        if not self._data:
            return
        
        data = list(self._data)
        min_val = min(data)
        max_val = max(data)
        
        if max_val == min_val:
            sparkline = SPARK_CHARS_4[2] * len(data)
        else:
            chars = SPARK_CHARS_4
            range_val = max_val - min_val
            sparkline = ""
            for v in data:
                normalized = (v - min_val) / range_val
                idx = int(normalized * (len(chars) - 1))
                idx = max(0, min(len(chars) - 1, idx))
                sparkline += chars[idx]
        
        try:
            self.query_one("#mini-spark", Static).update(sparkline)
        except Exception:
            pass
