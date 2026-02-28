"""
Dashboard Screen - Main menu and quick overview

Displays:
- Quick action buttons for training/preprocessing
- Recent runs list
- Quick stats (total runs, datasets, etc.)
- Navigation to other screens
"""

from __future__ import annotations

from textual.app import ComposeResult
from textual.containers import Container, Horizontal, Vertical, Grid
from textual.screen import Screen
from textual.widgets import (
    Static,
    Button,
    DataTable,
    Label,
    Rule,
)
from textual.binding import Binding

from rich.text import Text
from rich.panel import Panel

# ASCII art banner for Side-Step
BANNER = r"""
 ███████╗██╗██████╗ ███████╗    ███████╗████████╗███████╗██████╗ 
 ██╔════╝██║██╔══██╗██╔════╝    ██╔════╝╚══██╔══╝██╔════╝██╔══██╗
 ███████╗██║██║  ██║█████╗█████╗███████╗   ██║   █████╗  ██████╔╝
 ╚════██║██║██║  ██║██╔══╝╚════╝╚════██║   ██║   ██╔══╝  ██╔═══╝ 
 ███████║██║██████╔╝███████╗    ███████║   ██║   ███████╗██║     
 ╚══════╝╚═╝╚═════╝ ╚══════╝    ╚══════╝   ╚═╝   ╚══════╝╚═╝     
"""

MOTTOS = [
    "LoRA Training Made Visual",
    "Train Like You Mean It",
    "Your Music, Your Model",
]


class DashboardScreen(Screen):
    """Main dashboard screen with quick actions and overview."""
    
    BINDINGS = [
        Binding("f", "fixed_training", "Fixed Training", show=True),
        Binding("v", "vanilla_training", "Vanilla Training", show=True),
        Binding("p", "preprocess", "Preprocess", show=True),
        Binding("e", "estimate", "Estimate", show=True),
        Binding("h", "history", "History"),
        Binding("s", "settings", "Settings"),
        Binding("r", "refresh", "Refresh"),
    ]
    
    CSS = """
    DashboardScreen {
        layout: vertical;
    }

    #banner-container {
        height: auto;
        align: center middle;
        padding: 1 0 0 0;
    }

    #banner {
        text-align: center;
        color: $primary;
    }

    #version-tag {
        text-align: center;
        color: $text-muted;
        text-style: italic;
    }

    #motto {
        text-align: center;
        color: $secondary;
        margin-top: 0;
        text-style: italic;
    }

    #quick-actions {
        height: 5;
        align: center middle;
        padding: 0 2;
    }

    #quick-actions Button {
        margin: 0 1;
        min-width: 24;
    }

    #main-content {
        layout: horizontal;
        height: 1fr;
        padding: 1 2;
    }

    #recent-runs-container {
        width: 60%;
        height: 100%;
        border: round $primary 35%;
        padding: 1;
    }

    #stats-container {
        width: 40%;
        height: 100%;
        border: round $primary 35%;
        padding: 1;
        margin-left: 1;
    }

    .section-title {
        text-style: bold;
        color: $primary;
        margin-bottom: 1;
    }

    #nav-footer {
        height: 3;
        align: center middle;
        padding: 0 2;
    }

    #nav-footer Button {
        margin: 0 1;
        min-width: 16;
    }

    DataTable {
        height: 1fr;
    }

    #stats-grid {
        height: auto;
    }

    .stat-row {
        height: 2;
        layout: horizontal;
    }

    .stat-label {
        width: 50%;
        color: $text-muted;
    }

    .stat-value {
        width: 50%;
        text-align: right;
        text-style: bold;
        color: $primary;
    }

    #workflow-panel {
        border: round $secondary 25%;
        background: $panel;
        padding: 1;
        margin-top: 1;
    }

    .workflow-title {
        text-style: bold;
        color: $secondary;
    }

    .workflow-step {
        color: $text-muted;
        margin-left: 1;
    }
    """
    
    def __init__(self) -> None:
        super().__init__()
        import random
        self._motto = random.choice(MOTTOS)
    
    def compose(self) -> ComposeResult:
        """Compose the dashboard layout."""
        # Banner
        with Container(id="banner-container"):
            yield Static(BANNER, id="banner")
            yield Static("by dernet | v0.2.0", id="version-tag")
            yield Static(self._motto, id="motto")
        
        yield Rule()
        
        # Quick action buttons
        with Horizontal(id="quick-actions"):
            yield Button("[F] Fixed Training", id="btn-fixed", variant="primary")
            yield Button("[V] Vanilla Training", id="btn-vanilla", variant="default")
            yield Button("[P] Preprocess", id="btn-preprocess", variant="success")
            yield Button("[E] Estimate", id="btn-estimate", variant="warning")
        
        yield Rule()
        
        # Main content area
        with Horizontal(id="main-content"):
            # Recent runs
            with Container(id="recent-runs-container"):
                yield Label("Recent Runs", classes="section-title")
                yield DataTable(id="recent-runs-table")
            
            # Quick stats
            with Container(id="stats-container"):
                yield Label("Quick Stats", classes="section-title")
                with Vertical(id="stats-grid"):
                    with Horizontal(classes="stat-row"):
                        yield Static("Total Runs:", classes="stat-label")
                        yield Static("0", id="stat-total-runs", classes="stat-value")
                    with Horizontal(classes="stat-row"):
                        yield Static("Completed:", classes="stat-label")
                        yield Static("0", id="stat-completed", classes="stat-value")
                    with Horizontal(classes="stat-row"):
                        yield Static("Failed:", classes="stat-label")
                        yield Static("0", id="stat-failed", classes="stat-value")
                    with Horizontal(classes="stat-row"):
                        yield Static("Dataset:", classes="stat-label")
                        yield Static("None", id="stat-dataset", classes="stat-value")
                    yield Rule()
                    yield Label("GPU Status", classes="section-title")
                    with Horizontal(classes="stat-row"):
                        yield Static("VRAM:", classes="stat-label")
                        yield Static("-- / -- GB", id="stat-vram", classes="stat-value")
                    with Horizontal(classes="stat-row"):
                        yield Static("Utilization:", classes="stat-label")
                        yield Static("--%", id="stat-util", classes="stat-value")
                
                # Workflow guidance panel
                with Container(id="workflow-panel"):
                    yield Static("First Time?", classes="workflow-title")
                    yield Static("1. [P] Preprocess your audio files", classes="workflow-step")
                    yield Static("2. [E] Estimate to find best layers (optional)", classes="workflow-step")
                    yield Static("3. [F] Fixed Training (recommended)", classes="workflow-step")
                    yield Static("", classes="workflow-step")
                    yield Static("[?] Press ? for help anytime", classes="workflow-step")
        
        # Navigation footer
        with Horizontal(id="nav-footer"):
            yield Button("[D] Datasets", id="btn-datasets")
            yield Button("[H] History", id="btn-history")
            yield Button("[S] Settings", id="btn-settings")
            yield Button("[Q] Quit", id="btn-quit", variant="error")
    
    def on_mount(self) -> None:
        """Initialize the dashboard on mount."""
        # Set up the recent runs table
        table = self.query_one("#recent-runs-table", DataTable)
        table.add_columns("Name", "Type", "Status", "Epochs", "Loss")
        table.cursor_type = "row"
        
        # Load data
        self._refresh_data()
        
        # Start GPU monitoring
        self._start_gpu_monitor()
    
    def _refresh_data(self) -> None:
        """Refresh dashboard data from app state."""
        state = self.app.app_state
        
        # Update stats
        stats = state.get_stats()
        self.query_one("#stat-total-runs", Static).update(str(stats["total_runs"]))
        self.query_one("#stat-completed", Static).update(str(stats["completed_runs"]))
        self.query_one("#stat-failed", Static).update(str(stats["failed_runs"]))
        
        # Show last-used dataset
        try:
            ds = getattr(state.preferences, "default_dataset_dir", "")
            if ds:
                from pathlib import Path
                label = Path(ds).name
                self.query_one("#stat-dataset", Static).update(label)
        except Exception:
            pass
        
        # Update recent runs table
        table = self.query_one("#recent-runs-table", DataTable)
        table.clear()
        
        # Add current run if exists
        if state.current_run:
            run = state.current_run
            status_text = Text(f"● {run.status}", style="green bold")
            table.add_row(
                run.name,
                run.trainer_type,
                status_text,
                f"{run.current_epoch}/{run.total_epochs}",
                f"{run.current_loss:.4f}",
            )
        
        # Add recent runs
        for run in state.recent_runs[:10]:
            status_style = {
                "completed": "green",
                "failed": "red",
                "paused": "yellow",
            }.get(run.status, "white")
            status_text = Text(run.status, style=status_style)
            
            table.add_row(
                run.name,
                run.trainer_type,
                status_text,
                f"{run.current_epoch}/{run.total_epochs}",
                f"{run.best_loss:.4f}" if run.best_loss < float("inf") else "--",
            )
    
    def _start_gpu_monitor(self) -> None:
        """Start periodic GPU status updates."""
        self._update_gpu_status()
        self.set_interval(2.0, self._update_gpu_status)
    
    def _update_gpu_status(self) -> None:
        """Update GPU status display."""
        try:
            from sidestep_engine.models.gpu_utils import get_gpu_info
            info = get_gpu_info()
            
            vram_used = info.get("vram_used_gb", 0)
            vram_total = info.get("vram_total_gb", 0)
            util = info.get("utilization", 0)
            
            self.query_one("#stat-vram", Static).update(
                f"{vram_used:.1f} / {vram_total:.1f} GB"
            )
            self.query_one("#stat-util", Static).update(f"{util:.0f}%")
            
            # Update app state
            self.app.app_state.update_gpu_status(
                vram_used_gb=vram_used,
                vram_total_gb=vram_total,
                utilization=util,
                name=info.get("name", "GPU"),
            )
        except Exception:
            self.query_one("#stat-vram", Static).update("N/A")
            self.query_one("#stat-util", Static).update("N/A")
    
    # =========================================================================
    # Button Handlers
    # =========================================================================
    
    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button presses."""
        button_id = event.button.id
        
        if button_id == "btn-fixed":
            self.action_fixed_training()
        elif button_id == "btn-vanilla":
            self.action_vanilla_training()
        elif button_id == "btn-preprocess":
            self.action_preprocess()
        elif button_id == "btn-estimate":
            self.action_estimate()
        elif button_id == "btn-datasets":
            self.action_preprocess()  # Opens browse mode
        elif button_id == "btn-history":
            self.action_history()
        elif button_id == "btn-settings":
            self.action_settings()
        elif button_id == "btn-quit":
            self.app.exit()
    
    # =========================================================================
    # Actions
    # =========================================================================
    
    def action_fixed_training(self) -> None:
        """Start fixed training configuration."""
        self.app.action_new_fixed_training()
    
    def action_vanilla_training(self) -> None:
        """Start vanilla training configuration."""
        self.app.action_new_vanilla_training()
    
    def action_preprocess(self) -> None:
        """Go to preprocessing/dataset browser."""
        self.app.action_goto_preprocess()
    
    def action_estimate(self) -> None:
        """Go to gradient estimation."""
        self.app.action_goto_estimate()
    
    def action_history(self) -> None:
        """Go to run history."""
        self.app.action_goto_history()
    
    def action_settings(self) -> None:
        """Go to settings."""
        self.app.action_goto_settings()
    
    def action_refresh(self) -> None:
        """Refresh dashboard data."""
        self._refresh_data()
        self._update_gpu_status()
        self.notify("Dashboard refreshed", timeout=2)
