"""
Preprocess Monitor Screen

Live view of preprocessing progress with:
- File progress bar
- Current file being processed
- Log output
- Cancel button
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Any, Optional
import threading
import queue

from textual.app import ComposeResult
from textual.containers import Container, Horizontal, Vertical
from textual.screen import Screen
from textual.widgets import (
    Static,
    Button,
    ProgressBar,
    Rule,
)
from textual.binding import Binding
from textual.worker import Worker, get_current_worker

from sidestep_engine.tui.widgets.log_viewer import LogViewer


class PreprocessMonitorScreen(Screen):
    """Monitor preprocessing progress."""
    
    BINDINGS = [
        Binding("escape", "back", "Back"),
        Binding("c", "cancel", "Cancel"),
    ]
    
    CSS = """
    PreprocessMonitorScreen {
        layout: vertical;
    }
    
    #preprocess-header {
        height: 3;
        background: $panel;
        padding: 0 2;
        layout: horizontal;
        align: left middle;
    }
    
    #preprocess-title {
        width: 1fr;
        text-style: bold;
        color: $primary;
    }
    
    #header-controls {
        width: auto;
    }
    
    #progress-section {
        height: auto;
        padding: 1 2;
        border-bottom: solid $primary 25%;
    }
    
    #progress-label {
        margin-bottom: 1;
    }
    
    #current-file {
        color: $text-muted;
        margin-top: 1;
    }
    
    #stats-section {
        height: auto;
        padding: 1 2;
        layout: horizontal;
    }
    
    .stat-box {
        width: 1fr;
        text-align: center;
        padding: 1;
        border: round $primary 30%;
        margin-right: 1;
    }
    
    .stat-label {
        color: $text-muted;
    }
    
    .stat-value {
        text-style: bold;
        color: $primary;
        margin-top: 0;
    }
    
    #log-section {
        height: 1fr;
        padding: 1 2;
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
        self._processed_count = 0
        self._total_count = 0
        self._failed_count = 0
    
    def compose(self) -> ComposeResult:
        """Compose the monitor layout."""
        source_name = Path(self.config["source_dir"]).name
        
        # Header
        with Container(id="preprocess-header"):
            yield Static(f"[Side-Step] Preprocessing: {source_name}", id="preprocess-title")
            with Horizontal(id="header-controls"):
                yield Button("Cancel", id="btn-cancel", variant="error")
                yield Button("Back", id="btn-back", variant="default", disabled=True)
        
        # Progress section
        with Container(id="progress-section"):
            yield Static("Preparing...", id="progress-label")
            yield ProgressBar(total=100, show_eta=False, id="file-progress")
            yield Static("", id="current-file")
        
        # Stats section
        with Horizontal(id="stats-section"):
            with Vertical(classes="stat-box"):
                yield Static("Processed", classes="stat-label")
                yield Static("0", id="stat-processed", classes="stat-value")
            with Vertical(classes="stat-box"):
                yield Static("Failed", classes="stat-label")
                yield Static("0", id="stat-failed", classes="stat-value")
            with Vertical(classes="stat-box"):
                yield Static("Total", classes="stat-label")
                yield Static("0", id="stat-total", classes="stat-value")
        
        # Log section
        with Container(id="log-section"):
            yield LogViewer(title="Preprocessing Log", id="preprocess-log")
    
    def on_mount(self) -> None:
        """Start preprocessing on mount."""
        log = self.query_one("#preprocess-log", LogViewer)
        log.info("Starting preprocessing...")
        log.info(f"Source: {self.config['source_dir']}")
        log.info(f"Output: {self.config['output_dir']}")
        log.info(f"Model: {self.config['variant']}")
        log.write_separator()
        
        # Start preprocessing worker (sync function → runs in a real thread)
        self._worker = self.run_worker(
            self._run_preprocessing,
            name="preprocessing",
            thread=True,
        )
        
        # Start log processor
        self.set_interval(0.1, self._process_log_queue)
    
    def _run_preprocessing(self) -> None:  # sync, runs in a thread
        """Run preprocessing in a background thread."""
        worker = get_current_worker()
        
        try:
            source_dir = Path(self.config["source_dir"])
            output_dir = Path(self.config["output_dir"])
            checkpoint_dir = Path(self.config["checkpoint_dir"])
            variant = self.config["variant"]
            max_duration = self.config.get("max_duration", 240.0)
            
            # Find audio files
            audio_extensions = (".mp3", ".wav", ".flac", ".ogg", ".m4a")
            audio_files = []
            for ext in audio_extensions:
                audio_files.extend(source_dir.glob(f"*{ext}"))
                audio_files.extend(source_dir.glob(f"**/*{ext}"))
            
            # Also check for JSON manifest
            json_files = list(source_dir.glob("*.json"))
            
            if not audio_files and not json_files:
                self._log_queue.put(("error", "No audio files or JSON manifests found"))
                self.app.call_from_thread(self._on_complete, False)
                return
            
            self._total_count = len(audio_files) if audio_files else 1
            self.app.call_from_thread(self._update_total, self._total_count)
            
            # Create output directory
            output_dir.mkdir(parents=True, exist_ok=True)
            
            # Import preprocessing modules
            try:
                from sidestep_engine.data.preprocess import preprocess_audio_files
                
                self._log_queue.put(("info", f"Found {len(audio_files)} audio files"))
                
                # Run preprocessing with callback
                preprocess_audio_files(
                    source_dir=str(source_dir),
                    output_dir=str(output_dir),
                    checkpoint_dir=str(checkpoint_dir),
                    variant=variant,
                    max_duration=max_duration,
                    progress_callback=self._progress_callback,
                    cancel_check=lambda: self._is_cancelling or worker.is_cancelled,
                )
                
                if not worker.is_cancelled and not self._is_cancelling:
                    self._log_queue.put(("success", "Preprocessing completed!"))
                    self.app.call_from_thread(self._on_complete, True)
                else:
                    self._log_queue.put(("warning", "Preprocessing cancelled"))
                    self.app.call_from_thread(self._on_complete, False)
                    
            except ImportError:
                # Fallback: try using CLI
                self._log_queue.put(("warning", "Using CLI fallback for preprocessing..."))
                self._run_cli_preprocessing()
                
        except Exception as e:
            self._log_queue.put(("error", f"Preprocessing failed: {e}"))
            self.app.call_from_thread(self._on_complete, False)
    
    def _run_cli_preprocessing(self) -> None:
        """Fallback: run preprocessing via CLI subprocess."""
        import subprocess
        import sys
        
        source_dir = self.config["source_dir"]
        output_dir = self.config["output_dir"]
        checkpoint_dir = self.config["checkpoint_dir"]
        variant = self.config["variant"]
        max_duration = self.config.get("max_duration", 240.0)
        
        # Resolve train.py from the project root (platform-agnostic)
        train_script = str(Path(__file__).resolve().parents[4] / "train.py")
        cmd = [
            sys.executable, train_script, "preprocess",
            "--checkpoint-dir", checkpoint_dir,
            "--model-variant", variant,
            "--audio-dir", source_dir,
            "--tensor-output", output_dir,
            "--max-duration", str(max_duration),
        ]
        
        self._log_queue.put(("info", f"Running: {' '.join(cmd)}"))
        
        try:
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )
            
            for line in process.stdout:
                line = line.strip()
                if line:
                    self._log_queue.put(("info", line))
                    
                    # Try to parse progress
                    if "Processing" in line and "/" in line:
                        try:
                            parts = line.split("/")
                            if len(parts) >= 2:
                                current = int(parts[0].split()[-1])
                                total = int(parts[1].split()[0])
                                self.app.call_from_thread(self._update_progress, current, total)
                        except Exception:
                            pass
                
                if self._is_cancelling:
                    process.terminate()
                    break
            
            process.wait()
            
            if process.returncode == 0:
                self._log_queue.put(("success", "Preprocessing completed!"))
                self.app.call_from_thread(self._on_complete, True)
            else:
                self._log_queue.put(("error", f"Process exited with code {process.returncode}"))
                self.app.call_from_thread(self._on_complete, False)
                
        except Exception as e:
            self._log_queue.put(("error", f"Failed to run CLI: {e}"))
            self.app.call_from_thread(self._on_complete, False)
    
    def _progress_callback(
        self,
        current: int,
        total: int,
        filename: str,
        success: bool = True,
    ) -> None:
        """Callback for preprocessing progress."""
        self._processed_count = current
        
        if success:
            self._log_queue.put(("info", f"Processed: {filename}"))
        else:
            self._failed_count += 1
            self._log_queue.put(("warning", f"Failed: {filename}"))
        
        self.app.call_from_thread(self._update_progress, current, total, filename)
    
    def _update_progress(self, current: int, total: int, filename: str = "") -> None:
        """Update progress display."""
        percent = (current / total * 100) if total > 0 else 0
        
        self.query_one("#file-progress", ProgressBar).progress = percent
        self.query_one("#progress-label", Static).update(f"Progress: {current}/{total} files ({percent:.0f}%)")
        
        if filename:
            self.query_one("#current-file", Static).update(f"Current: {Path(filename).name}")
        
        self.query_one("#stat-processed", Static).update(str(current))
        self.query_one("#stat-failed", Static).update(str(self._failed_count))
    
    def _update_total(self, total: int) -> None:
        """Update total count display."""
        self._total_count = total
        self.query_one("#stat-total", Static).update(str(total))
    
    def _process_log_queue(self) -> None:
        """Process queued log messages."""
        log = self.query_one("#preprocess-log", LogViewer)
        
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
        """Handle preprocessing completion."""
        self.query_one("#btn-cancel", Button).disabled = True
        self.query_one("#btn-back", Button).disabled = False
        
        if success:
            self.notify(
                f"Preprocessing complete! {self._processed_count} files processed.",
                timeout=5,
            )
        else:
            self.notify(
                "Preprocessing stopped or failed",
                severity="warning",
                timeout=5,
            )
    
    # =========================================================================
    # Button Handlers
    # =========================================================================
    
    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button presses."""
        if event.button.id == "btn-cancel":
            self.action_cancel()
        elif event.button.id == "btn-back":
            self.action_back()
    
    # =========================================================================
    # Actions
    # =========================================================================
    
    def action_cancel(self) -> None:
        """Cancel preprocessing."""
        if self._is_cancelling:
            return
        
        self._is_cancelling = True
        log = self.query_one("#preprocess-log", LogViewer)
        log.warning("Cancelling...")
        self.notify("Cancelling preprocessing...", timeout=3)
        
        if self._worker:
            self._worker.cancel()
    
    def action_back(self) -> None:
        """Go back."""
        self.app.pop_screen()
