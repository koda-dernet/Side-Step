"""
Run History Screen

View past training runs with:
- Run list with status, epochs, loss
- Run details view
- Checkpoint management
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from textual.app import ComposeResult
from textual.containers import Container, Horizontal, Vertical, ScrollableContainer
from textual.screen import Screen
from textual.widgets import (
    Static,
    Button,
    DataTable,
    Input,
    Label,
    Rule,
)
from textual.binding import Binding

from rich.text import Text

from sidestep_engine.tui.state import RunInfo


class RunHistoryScreen(Screen):
    """View and manage past training runs."""
    
    BINDINGS = [
        Binding("escape", "back", "Back"),
        Binding("r", "refresh", "Refresh"),
        Binding("d", "delete_run", "Delete"),
        Binding("enter", "view_details", "View Details"),
    ]
    
    CSS = """
    RunHistoryScreen {
        layout: vertical;
    }
    
    #history-header {
        height: 3;
        background: $panel;
        padding: 0 2;
        layout: horizontal;
        align: left middle;
    }
    
    #history-title {
        width: 1fr;
        text-style: bold;
        color: $primary;
    }
    
    #history-controls {
        width: auto;
    }
    
    #history-main {
        layout: horizontal;
        height: 1fr;
        padding: 1;
    }
    
    #runs-panel {
        width: 60%;
        height: 100%;
        border: round $primary 35%;
        padding: 1;
    }
    
    #details-panel {
        width: 40%;
        height: 100%;
        border: round $primary 35%;
        padding: 1;
        margin-left: 1;
    }
    
    .panel-title {
        text-style: bold;
        color: $primary;
        margin-bottom: 1;
    }
    
    DataTable {
        height: 1fr;
    }
    
    #run-details {
        height: auto;
    }
    
    .detail-row {
        height: 2;
        layout: horizontal;
    }
    
    .detail-label {
        width: 40%;
        color: $text-muted;
    }
    
    .detail-value {
        width: 60%;
        color: $primary;
    }
    
    #detail-actions {
        height: auto;
        margin-top: 2;
        layout: horizontal;
    }
    
    #detail-actions Button {
        margin-right: 1;
    }
    """
    
    def __init__(self, name: Optional[str] = None) -> None:
        super().__init__(name=name)
        self._selected_run: Optional[RunInfo] = None
    
    def compose(self) -> ComposeResult:
        """Compose the history layout."""
        # Header
        with Container(id="history-header"):
            yield Static("Run History", id="history-title")
            with Horizontal(id="history-controls"):
                yield Button("Refresh", id="btn-refresh", variant="default")
                yield Button("Back", id="btn-back", variant="default")
        
        # Main content
        with Horizontal(id="history-main"):
            # Runs list
            with Container(id="runs-panel"):
                yield Label("Past Runs", classes="panel-title")
                yield DataTable(id="runs-table")
            
            # Details panel
            with ScrollableContainer(id="details-panel"):
                yield Label("Run Details", classes="panel-title")
                
                with Vertical(id="run-details"):
                    yield Static("Select a run to view details", id="detail-placeholder")
                    
                    with Horizontal(classes="detail-row", id="detail-name-row"):
                        yield Static("Name:", classes="detail-label")
                        yield Static("--", id="detail-name", classes="detail-value")
                    
                    with Horizontal(classes="detail-row", id="detail-type-row"):
                        yield Static("Trainer:", classes="detail-label")
                        yield Static("--", id="detail-type", classes="detail-value")
                    
                    with Horizontal(classes="detail-row", id="detail-status-row"):
                        yield Static("Status:", classes="detail-label")
                        yield Static("--", id="detail-status", classes="detail-value")
                    
                    with Horizontal(classes="detail-row", id="detail-epochs-row"):
                        yield Static("Epochs:", classes="detail-label")
                        yield Static("--", id="detail-epochs", classes="detail-value")
                    
                    with Horizontal(classes="detail-row", id="detail-loss-row"):
                        yield Static("Best Loss:", classes="detail-label")
                        yield Static("--", id="detail-loss", classes="detail-value")
                    
                    with Horizontal(classes="detail-row", id="detail-started-row"):
                        yield Static("Started:", classes="detail-label")
                        yield Static("--", id="detail-started", classes="detail-value")
                    
                    with Horizontal(classes="detail-row", id="detail-finished-row"):
                        yield Static("Finished:", classes="detail-label")
                        yield Static("--", id="detail-finished", classes="detail-value")
                    
                    with Horizontal(classes="detail-row", id="detail-output-row"):
                        yield Static("Output:", classes="detail-label")
                        yield Static("--", id="detail-output", classes="detail-value")
                    
                    yield Rule()
                    
                    with Horizontal(id="detail-actions"):
                        yield Button("Open Folder", id="btn-open-folder", variant="default")
                        yield Button("Resume", id="btn-resume", variant="primary")
                        yield Button("Delete", id="btn-delete", variant="error")
    
    def on_mount(self) -> None:
        """Initialize on mount."""
        # Set up table
        table = self.query_one("#runs-table", DataTable)
        table.add_columns("Name", "Type", "Status", "Epochs", "Best Loss", "Date")
        table.cursor_type = "row"
        
        # Hide detail rows initially
        self._hide_detail_rows()
        
        # Load data
        self._refresh_data()
    
    def _hide_detail_rows(self) -> None:
        """Hide detail rows initially."""
        row_ids = [
            "detail-name-row", "detail-type-row", "detail-status-row",
            "detail-epochs-row", "detail-loss-row", "detail-started-row",
            "detail-finished-row", "detail-output-row",
        ]
        for row_id in row_ids:
            try:
                self.query_one(f"#{row_id}").display = False
            except Exception:
                pass
        try:
            self.query_one("#detail-actions").display = False
        except Exception:
            pass
    
    def _show_detail_rows(self) -> None:
        """Show detail rows."""
        try:
            self.query_one("#detail-placeholder").display = False
        except Exception:
            pass
        
        row_ids = [
            "detail-name-row", "detail-type-row", "detail-status-row",
            "detail-epochs-row", "detail-loss-row", "detail-started-row",
            "detail-finished-row", "detail-output-row",
        ]
        for row_id in row_ids:
            try:
                self.query_one(f"#{row_id}").display = True
            except Exception:
                pass
        try:
            self.query_one("#detail-actions").display = True
        except Exception:
            pass
    
    def _refresh_data(self) -> None:
        """Refresh the runs list."""
        table = self.query_one("#runs-table", DataTable)
        table.clear()
        
        runs = self.app.app_state.recent_runs
        
        if not runs:
            table.add_row("No runs yet", "", "", "", "", "")
            return
        
        for run in runs:
            # Format status with color
            status_style = {
                "completed": "green",
                "failed": "red",
                "paused": "yellow",
                "running": "cyan",
            }.get(run.status, "white")
            status_text = Text(run.status, style=status_style)
            
            # Format date
            if run.started_at:
                date_str = run.started_at[:10]  # Just the date part
            else:
                date_str = "--"
            
            # Format loss
            if run.best_loss < float("inf"):
                loss_str = f"{run.best_loss:.4f}"
            else:
                loss_str = "--"
            
            table.add_row(
                run.name,
                run.trainer_type,
                status_text,
                f"{run.current_epoch}/{run.total_epochs}",
                loss_str,
                date_str,
            )
    
    def on_data_table_row_selected(self, event: DataTable.RowSelected) -> None:
        """Handle row selection."""
        runs = self.app.app_state.recent_runs
        
        if event.cursor_row < len(runs):
            self._selected_run = runs[event.cursor_row]
            self._update_details(self._selected_run)
    
    def _update_details(self, run: RunInfo) -> None:
        """Update the details panel."""
        self._show_detail_rows()
        
        self.query_one("#detail-name", Static).update(run.name)
        self.query_one("#detail-type", Static).update(run.trainer_type)
        
        # Status with style
        status_style = {
            "completed": "green",
            "failed": "red",
            "paused": "yellow",
        }.get(run.status, "")
        self.query_one("#detail-status", Static).update(
            Text(run.status, style=status_style) if status_style else run.status
        )
        
        self.query_one("#detail-epochs", Static).update(
            f"{run.current_epoch}/{run.total_epochs}"
        )
        
        if run.best_loss < float("inf"):
            self.query_one("#detail-loss", Static).update(
                f"{run.best_loss:.4f} @ epoch {run.best_epoch}"
            )
        else:
            self.query_one("#detail-loss", Static).update("--")
        
        self.query_one("#detail-started", Static).update(run.started_at or "--")
        self.query_one("#detail-finished", Static).update(run.finished_at or "--")
        self.query_one("#detail-output", Static).update(run.output_dir or "--")
        
        # Enable/disable resume button based on status
        resume_btn = self.query_one("#btn-resume", Button)
        resume_btn.disabled = run.status not in ("paused", "failed")
    
    # =========================================================================
    # Button Handlers
    # =========================================================================
    
    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button presses."""
        if event.button.id == "btn-back":
            self.action_back()
        elif event.button.id == "btn-refresh":
            self.action_refresh()
        elif event.button.id == "btn-delete":
            self.action_delete_run()
        elif event.button.id == "btn-open-folder":
            self._open_folder()
        elif event.button.id == "btn-resume":
            self._resume_run()
    
    # =========================================================================
    # Actions
    # =========================================================================
    
    def action_back(self) -> None:
        """Go back."""
        self.app.pop_screen()
    
    def action_refresh(self) -> None:
        """Refresh the list."""
        self._refresh_data()
        self.notify("Refreshed", timeout=2)
    
    def action_delete_run(self) -> None:
        """Delete the selected run from history."""
        if not self._selected_run:
            self.notify("No run selected", severity="warning")
            return
        
        # Remove from history
        runs = self.app.app_state.recent_runs
        if self._selected_run in runs:
            runs.remove(self._selected_run)
            self.app.app_state._save_history()
            self._refresh_data()
            self._hide_detail_rows()
            try:
                self.query_one("#detail-placeholder").display = True
            except Exception:
                pass
            self._selected_run = None
            self.notify("Run removed from history", timeout=3)
    
    def action_view_details(self) -> None:
        """View details of selected run."""
        # Already handled by row selection
        pass
    
    def _open_folder(self) -> None:
        """Open the output folder in file manager."""
        if not self._selected_run or not self._selected_run.output_dir:
            self.notify("No output directory", severity="warning")
            return
        
        path = Path(self._selected_run.output_dir)
        if path.exists():
            import subprocess
            import sys
            
            if sys.platform == "darwin":
                subprocess.run(["open", str(path)])
            elif sys.platform == "linux":
                subprocess.run(["xdg-open", str(path)])
            else:
                subprocess.run(["explorer", str(path)])
            
            self.notify(f"Opened {path}", timeout=3)
        else:
            self.notify(f"Directory not found: {path}", severity="error")
    
    def _resume_run(self) -> None:
        """Resume a paused/failed run by opening training config pre-filled with resume path."""
        if not self._selected_run:
            return

        run = self._selected_run
        output_dir = run.output_dir

        if not output_dir:
            self.notify("No output directory recorded for this run", severity="error", timeout=5)
            return

        # Find the latest checkpoint
        from pathlib import Path
        ckpt_dir = Path(output_dir) / "checkpoints"
        resume_path = None
        if ckpt_dir.exists():
            epoch_dirs = sorted(
                [d for d in ckpt_dir.iterdir() if d.is_dir() and d.name.startswith("epoch_")],
                key=lambda d: int(d.name.split("_")[1]) if d.name.split("_")[1].isdigit() else 0,
                reverse=True,
            )
            if epoch_dirs:
                resume_path = str(epoch_dirs[0])

        if not resume_path:
            self.notify("No checkpoints found in output directory", severity="warning", timeout=5)
            return

        # Open training config screen pre-filled with resume info.
        # We pass the resume values via a dict attached to the screen so its
        # on_mount can pick them up without needing a timer hack.
        from sidestep_engine.tui.screens.training_config import TrainingConfigScreen
        screen = TrainingConfigScreen(trainer_type=run.trainer_type)

        # Stash resume hints on the screen instance before it's mounted
        screen._resume_prefill = {
            "output_dir": output_dir,
            "resume_from": resume_path,
            "checkpoint_dir": getattr(run, "checkpoint_dir", ""),
            "dataset_dir": getattr(run, "dataset_dir", ""),
        }

        self.app.push_screen(screen)
        self.notify(f"Resuming from {Path(resume_path).name}", timeout=4)
