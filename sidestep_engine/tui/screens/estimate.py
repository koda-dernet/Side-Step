"""
Estimate Screen

Gradient sensitivity analysis for selecting optimal LoRA modules:
- Configuration form for estimation parameters
- Live progress monitoring
- Results visualization with module ranking
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Any, Optional, List
import threading
import queue
import json

from textual.app import ComposeResult
from textual.containers import Container, Horizontal, Vertical, ScrollableContainer
from textual.screen import Screen
from textual.widgets import (
    Static,
    Button,
    Input,
    Select,
    ProgressBar,
    DataTable,
    Rule,
    Label,
    RadioButton,
    RadioSet,
)
from textual.binding import Binding
from textual.worker import Worker, get_current_worker
from textual.validation import Number

from sidestep_engine.tui.widgets.log_viewer import LogViewer


class EstimateConfigScreen(Screen):
    """Configuration screen for gradient estimation."""
    
    BINDINGS = [
        Binding("escape", "back", "Back"),
        Binding("ctrl+s", "start", "Start Estimation"),
    ]
    
    CSS = """
    EstimateConfigScreen {
        layout: vertical;
    }
    
    #estimate-header {
        height: 3;
        background: $panel;
        padding: 0 2;
        layout: horizontal;
        align: left middle;
    }
    
    #estimate-title {
        width: 1fr;
        text-style: bold;
        color: $primary;
    }
    
    #header-buttons {
        width: auto;
    }
    
    #estimate-content {
        height: 1fr;
        padding: 1 2;
    }
    
    .section-header {
        text-style: bold;
        color: $primary;
        margin-bottom: 1;
        margin-top: 1;
    }
    
    .form-row {
        height: auto;
        layout: horizontal;
        margin-bottom: 1;
        align: left middle;
    }
    
    .form-label {
        width: 25;
        padding-right: 1;
        color: $text-muted;
    }
    
    .form-input {
        width: 1fr;
    }
    
    .form-hint {
        color: $text-muted;
        margin-left: 25;
        margin-top: 0;
    }
    
    .impact-hint {
        color: $text-muted;
        margin-left: 25;
        margin-top: 0;
        text-style: italic;
    }
    
    #info-panel {
        border: round $secondary 25%;
        padding: 1;
        margin-top: 1;
        background: $panel;
    }
    
    RadioSet {
        layout: horizontal;
        height: auto;
        width: 1fr;
    }
    
    RadioButton {
        margin-right: 2;
    }
    """
    
    def __init__(self, name: Optional[str] = None) -> None:
        super().__init__(name=name)
    
    def compose(self) -> ComposeResult:
        """Compose the estimation config form."""
        # Header
        with Container(id="estimate-header"):
            yield Static("[Side-Step] Gradient Sensitivity Estimation", id="estimate-title")
            with Horizontal(id="header-buttons"):
                yield Button("Start", id="btn-start", variant="success")
                yield Button("Back", id="btn-back", variant="default")
        
        # Main content
        with ScrollableContainer(id="estimate-content"):
            # Info panel
            with Container(id="info-panel"):
                yield Static(
                    "[bold]What is this?[/bold]\n\n"
                    "Gradient sensitivity analysis identifies which LoRA modules "
                    "have the most impact on your specific dataset. This helps you:\n\n"
                    "• Train faster by targeting only important modules\n"
                    "• Reduce VRAM usage with selective training\n"
                    "• Get better results by focusing on relevant layers\n\n"
                    "The process runs a few forward/backward passes on your data "
                    "and measures gradient magnitudes per module."
                )
            
            yield Rule()
            yield Static("Required Settings", classes="section-header")
            
            # Checkpoint directory
            with Horizontal(classes="form-row"):
                yield Static("Checkpoint Directory:", classes="form-label")
                with Horizontal(classes="form-input"):
                    yield Input(
                        placeholder="./checkpoints/acestep-v15-turbo",
                        id="input-checkpoint-dir",
                    )
                    yield Button("...", id="btn-browse-checkpoint", variant="default")
            yield Static("Path to model checkpoints", classes="form-hint")
            
            # Model variant
            with Horizontal(classes="form-row"):
                yield Static("Model Variant:", classes="form-label")
                with RadioSet(id="radio-variant"):
                    yield RadioButton("Turbo", value=True, id="variant-turbo")
                    yield RadioButton("Base", id="variant-base")
                    yield RadioButton("SFT", id="variant-sft")
            
            # Dataset directory
            with Horizontal(classes="form-row"):
                yield Static("Dataset Directory:", classes="form-label")
                with Horizontal(classes="form-input"):
                    yield Input(
                        placeholder="./datasets/my_preprocessed",
                        id="input-dataset-dir",
                    )
                    yield Button("...", id="btn-browse-dataset", variant="default")
            yield Static("Directory containing preprocessed .pt files", classes="form-hint")
            
            # Output file
            with Horizontal(classes="form-row"):
                yield Static("Output JSON:", classes="form-label")
                with Horizontal(classes="form-input"):
                    yield Input(
                        placeholder="./module_config.json",
                        id="input-output",
                    )
                    yield Button("...", id="btn-browse-output", variant="default")
            yield Static("Where to save the module ranking results", classes="form-hint")
            
            yield Rule()
            yield Static("Estimation Parameters", classes="section-header")
            
            # Number of batches
            with Horizontal(classes="form-row"):
                yield Static("Batches:", classes="form-label")
                yield Input(
                    value="10",
                    id="input-batches",
                    validators=[Number(minimum=1, maximum=100)],
                    classes="form-input",
                )
            yield Static("How many data samples to analyze", classes="form-hint")
            yield Static("↓ Fewer = faster, less accurate  |  ↑ More = slower, more reliable ranking", classes="impact-hint")
            
            # Top K modules
            with Horizontal(classes="form-row"):
                yield Static("Top K Modules:", classes="form-label")
                yield Input(
                    value="16",
                    id="input-top-k",
                    validators=[Number(minimum=1, maximum=192)],
                    classes="form-input",
                )
            yield Static("How many modules to select for training", classes="form-hint")
            yield Static("↓ Fewer = faster training, less capacity  |  ↑ More = slower, more capacity", classes="impact-hint")
            
            # Granularity
            with Horizontal(classes="form-row"):
                yield Static("Granularity:", classes="form-label")
                yield Select(
                    [
                        ("Module (recommended)", "module"),
                        ("Layer", "layer"),
                    ],
                    value="module",
                    id="select-granularity",
                )
            yield Static("Analysis level", classes="form-hint")
            yield Static("Module = individual attention layers  |  Layer = broader grouping", classes="impact-hint")
            
            # Batch size
            with Horizontal(classes="form-row"):
                yield Static("Batch Size:", classes="form-label")
                yield Input(
                    value="1",
                    id="input-batch-size",
                    validators=[Number(minimum=1, maximum=8)],
                    classes="form-input",
                )
            yield Static("Samples per estimation batch", classes="form-hint")
            yield Static("↓ Smaller = less VRAM  |  ↑ Larger = faster but more VRAM", classes="impact-hint")
    
    def _get_config(self) -> Dict[str, Any]:
        """Build config dict from form values."""
        return {
            "checkpoint_dir": self.query_one("#input-checkpoint-dir", Input).value,
            "dataset_dir": self.query_one("#input-dataset-dir", Input).value,
            "output_file": self.query_one("#input-output", Input).value,
            "variant": self._get_selected_variant(),
            "num_batches": int(self.query_one("#input-batches", Input).value or 10),
            "top_k": int(self.query_one("#input-top-k", Input).value or 16),
            "granularity": self.query_one("#select-granularity", Select).value,
            "batch_size": int(self.query_one("#input-batch-size", Input).value or 1),
        }
    
    def _get_selected_variant(self) -> str:
        """Get the selected model variant."""
        try:
            if self.query_one("#variant-turbo", RadioButton).value:
                return "turbo"
            elif self.query_one("#variant-base", RadioButton).value:
                return "base"
            elif self.query_one("#variant-sft", RadioButton).value:
                return "sft"
        except Exception:
            pass
        return "turbo"
    
    def _validate(self) -> Optional[str]:
        """Validate form inputs. Returns error message or None."""
        config = self._get_config()
        
        if not config["checkpoint_dir"]:
            return "Checkpoint directory is required"
        if not Path(config["checkpoint_dir"]).exists():
            return f"Checkpoint directory not found: {config['checkpoint_dir']}"
        if not config["dataset_dir"]:
            return "Dataset directory is required"
        if not config["output_file"]:
            return "Output file path is required"
        
        return None
    
    # =========================================================================
    # Button Handlers
    # =========================================================================
    
    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button presses."""
        if event.button.id == "btn-start":
            self.action_start()
        elif event.button.id == "btn-back":
            self.action_back()
        elif event.button.id == "btn-browse-checkpoint":
            self._open_file_picker("Select Checkpoint Directory", "#input-checkpoint-dir")
        elif event.button.id == "btn-browse-dataset":
            self._open_file_picker("Select Dataset Directory", "#input-dataset-dir")
        elif event.button.id == "btn-browse-output":
            self._open_file_picker("Select Output Path", "#input-output")
    
    # =========================================================================
    # Actions
    # =========================================================================
    
    def _open_file_picker(self, title: str, target_input_id: str) -> None:
        """Open a file picker modal and populate the target input."""
        from sidestep_engine.tui.widgets.file_picker import FilePickerModal

        current_val = ""
        try:
            current_val = self.query_one(target_input_id, Input).value
        except Exception:
            pass
        start = Path(current_val) if current_val and Path(current_val).exists() else Path.cwd()

        async def _handle_result(path) -> None:
            if path is not None:
                try:
                    self.query_one(target_input_id, Input).value = str(path)
                except Exception:
                    pass

        self.app.push_screen(
            FilePickerModal(title=title, start_path=start, select_directory=True),
            _handle_result,
        )

    def action_start(self) -> None:
        """Start the estimation."""
        error = self._validate()
        if error:
            self.notify(error, severity="error", timeout=5)
            return
        
        config = self._get_config()
        self.app.push_screen(EstimateMonitorScreen(config=config))
    
    def action_back(self) -> None:
        """Go back."""
        self.app.pop_screen()


class EstimateMonitorScreen(Screen):
    """Monitor estimation progress and show results."""
    
    BINDINGS = [
        Binding("escape", "back", "Back"),
        Binding("c", "cancel", "Cancel"),
        Binding("u", "use_results", "Use Results"),
    ]
    
    CSS = """
    EstimateMonitorScreen {
        layout: vertical;
    }
    
    #monitor-header {
        height: 3;
        background: $panel;
        padding: 0 2;
        layout: horizontal;
        align: left middle;
    }
    
    #monitor-title {
        width: 1fr;
        text-style: bold;
        color: $primary;
    }
    
    #header-controls {
        width: auto;
    }
    
    #progress-section {
        height: 5;
        padding: 1 2;
    }
    
    #main-content {
        layout: horizontal;
        height: 1fr;
        padding: 0 2;
    }
    
    #results-panel {
        width: 50%;
        height: 100%;
        border: round $primary 35%;
        padding: 1;
    }
    
    #log-panel {
        width: 50%;
        height: 100%;
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
    
    ProgressBar {
        width: 100%;
    }
    """
    
    def __init__(self, config: Dict[str, Any], name: Optional[str] = None) -> None:
        super().__init__(name=name)
        self.config = config
        self._is_cancelling = False
        self._worker: Optional[Worker] = None
        self._log_queue: queue.Queue = queue.Queue()
        self._results: List[Dict[str, Any]] = []
    
    def compose(self) -> ComposeResult:
        """Compose the monitor layout."""
        # Header
        with Container(id="monitor-header"):
            yield Static("Gradient Estimation", id="monitor-title")
            with Horizontal(id="header-controls"):
                yield Button("Cancel", id="btn-cancel", variant="error")
                yield Button("Use Results", id="btn-use", variant="success", disabled=True)
                yield Button("Back", id="btn-back", variant="default", disabled=True)
        
        # Progress section
        with Container(id="progress-section"):
            yield Static("Initializing...", id="progress-label")
            yield ProgressBar(total=100, show_eta=False, id="estimate-progress")
        
        # Main content
        with Horizontal(id="main-content"):
            # Results table
            with Container(id="results-panel"):
                yield Label("Top Modules", classes="panel-title")
                yield DataTable(id="results-table")
            
            # Log panel
            with Container(id="log-panel"):
                yield LogViewer(title="Estimation Log", id="estimate-log")
    
    def on_mount(self) -> None:
        """Start estimation on mount."""
        # Set up results table
        table = self.query_one("#results-table", DataTable)
        table.add_columns("Rank", "Module", "Sensitivity")
        
        # Start estimation
        log = self.query_one("#estimate-log", LogViewer)
        log.info("Starting gradient sensitivity estimation...")
        log.info(f"Dataset: {self.config['dataset_dir']}")
        log.info(f"Batches: {self.config['num_batches']}")
        log.info(f"Top K: {self.config['top_k']}")
        log.write_separator()
        
        self._worker = self.run_worker(
            self._run_estimation,
            name="estimation",
            thread=True,
        )
        
        self.set_interval(0.1, self._process_log_queue)
    
    def _run_estimation(self) -> None:  # sync, runs in a thread
        """Run estimation in a background thread."""
        worker = get_current_worker()

        try:
            try:
                from sidestep_engine.analysis.estimate import run_estimation

                results = run_estimation(
                    checkpoint_dir=self.config["checkpoint_dir"],
                    variant=self.config["variant"],
                    dataset_dir=self.config["dataset_dir"],
                    num_batches=self.config["num_batches"],
                    batch_size=self.config["batch_size"],
                    top_k=self.config["top_k"],
                    granularity=self.config["granularity"],
                    progress_callback=self._progress_callback,
                    cancel_check=lambda: self._is_cancelling or worker.is_cancelled,
                )

                if not worker.is_cancelled and not self._is_cancelling:
                    self._results = results

                    # Save results
                    output_file = Path(self.config["output_file"])
                    output_file.parent.mkdir(parents=True, exist_ok=True)
                    with open(output_file, "w") as f:
                        json.dump(results, f, indent=2)

                    self._log_queue.put(("success", f"Results saved to {output_file}"))
                    self.app.call_from_thread(self._on_complete, True)
                else:
                    self._log_queue.put(("warning", "Estimation cancelled"))
                    self.app.call_from_thread(self._on_complete, False)

            except ImportError as e:
                self._log_queue.put(("error", f"Estimation module not available: {e}"))
                self._log_queue.put(("info", "Use CLI: python train.py estimate --help"))
                self.app.call_from_thread(self._on_complete, False)

        except Exception as e:
            self._log_queue.put(("error", f"Estimation failed: {e}"))
            self.app.call_from_thread(self._on_complete, False)
    
    def _progress_callback(self, batch: int, total: int, module_name: str = "") -> None:
        """Callback for estimation progress (called from worker thread)."""
        percent = (batch / total * 100) if total > 0 else 0
        self.app.call_from_thread(self._update_progress, batch, total, percent)

        if module_name:
            self._log_queue.put(("info", f"Processing: {module_name}"))
    
    def _update_progress(self, batch: int, total: int, percent: float) -> None:
        """Update progress display."""
        self.query_one("#estimate-progress", ProgressBar).progress = percent
        self.query_one("#progress-label", Static).update(
            f"Batch {batch}/{total} ({percent:.0f}%)"
        )
    
    def _process_log_queue(self) -> None:
        """Process queued log messages."""
        log = self.query_one("#estimate-log", LogViewer)
        
        while True:
            try:
                level, message = self._log_queue.get_nowait()
                if level == "info":
                    log.info(message)
                elif level == "warning":
                    log.warning(message)
                elif level == "error":
                    log.error(message)
                elif level == "success":
                    log.success(message)
            except queue.Empty:
                break
    
    def _on_complete(self, success: bool) -> None:
        """Handle estimation completion."""
        self.query_one("#btn-cancel", Button).disabled = True
        self.query_one("#btn-back", Button).disabled = False
        
        if success and self._results:
            self.query_one("#btn-use", Button).disabled = False
            self._populate_results()
            self.notify("Estimation complete!", timeout=5)
        else:
            self.notify("Estimation stopped or failed", severity="warning", timeout=5)
    
    def _populate_results(self) -> None:
        """Populate the results table."""
        table = self.query_one("#results-table", DataTable)
        table.clear()
        
        for i, result in enumerate(self._results[:20], 1):  # Show top 20
            module_name = result.get("module", result.get("name", "unknown"))
            sensitivity = result.get("sensitivity", result.get("score", 0))
            table.add_row(str(i), module_name, f"{sensitivity:.6f}")
    
    # =========================================================================
    # Button Handlers
    # =========================================================================
    
    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button presses."""
        if event.button.id == "btn-cancel":
            self.action_cancel()
        elif event.button.id == "btn-back":
            self.action_back()
        elif event.button.id == "btn-use":
            self.action_use_results()
    
    # =========================================================================
    # Actions
    # =========================================================================
    
    def action_cancel(self) -> None:
        """Cancel estimation."""
        if self._is_cancelling:
            return
        
        self._is_cancelling = True
        self._log_queue.put(("warning", "Cancelling..."))
        
        if self._worker:
            self._worker.cancel()
    
    def action_back(self) -> None:
        """Go back."""
        self.app.pop_screen()
    
    def action_use_results(self) -> None:
        """Use these results for training."""
        if not self._results:
            self.notify("No results available", severity="warning")
            return
        
        # Store results in app state for use in training config
        self.app.app_state._last_estimation = self.config["output_file"]
        
        self.notify(
            f"Module config saved. Use in selective training:\n"
            f"--module-config {self.config['output_file']}",
            timeout=10,
        )
        
        # Go back to dashboard
        self.app.action_goto_dashboard()
