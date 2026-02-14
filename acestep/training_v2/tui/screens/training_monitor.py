"""
Training Monitor Screen

Live training view with:
- Progress bar for epochs
- Loss sparkline graph
- GPU utilization gauge
- Scrolling log output
- Pause/Stop controls
"""

from __future__ import annotations

import time
from datetime import datetime
from pathlib import Path
from typing import Optional, Dict, Any
import queue

from textual.app import ComposeResult
from textual.containers import Container, Horizontal, Vertical
from textual.screen import Screen
from textual.widgets import (
    Static,
    Button,
    ProgressBar,
)
from textual.binding import Binding
from textual.worker import Worker, get_current_worker

from acestep.training_v2.tui.state import RunInfo
from acestep.training_v2.tui.widgets.gpu_gauge import GPUGauge
from acestep.training_v2.tui.widgets.loss_sparkline import LossSparkline
from acestep.training_v2.tui.widgets.log_viewer import TrainingLogViewer


class TrainingMonitorScreen(Screen):
    """Live training monitoring screen."""

    BINDINGS = [
        Binding("escape", "back", "Back", show=True),
        Binding("p", "toggle_pause", "Pause/Resume", show=True),
        Binding("s", "stop_training", "Stop", show=True),
        Binding("l", "toggle_log", "Toggle Log"),
    ]

    CSS = """
    TrainingMonitorScreen {
        layout: vertical;
    }

    #monitor-header {
        height: 3;
        background: $panel;
        layout: horizontal;
        padding: 0 2;
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

    #header-controls Button {
        margin-left: 1;
    }

    #progress-section {
        height: 6;
        padding: 1 2;
        border-bottom: solid $primary 25%;
    }

    #progress-header {
        height: auto;
        layout: horizontal;
        margin-bottom: 1;
    }

    #epoch-label {
        width: 1fr;
    }

    #eta-label {
        width: auto;
    }

    #main-metrics {
        layout: horizontal;
        height: 1fr;
        padding: 1 2;
    }

    #loss-section {
        width: 60%;
        height: 100%;
        padding-right: 1;
    }

    #gpu-section {
        width: 40%;
        height: 100%;
    }

    #log-section {
        height: 12;
        padding: 0 2 1 2;
    }

    #current-stats {
        height: 3;
        layout: horizontal;
        padding: 0 2;
        background: $panel;
        align: left middle;
    }

    .stat-display {
        width: 1fr;
        text-align: center;
    }

    .stat-name {
        color: $text-muted;
    }

    .stat-val {
        text-style: bold;
        color: $primary;
    }

    ProgressBar {
        width: 100%;
    }
    """

    def __init__(
        self,
        run: RunInfo,
        config: Dict[str, Any],
        name: Optional[str] = None,
    ) -> None:
        super().__init__(name=name)
        self.run = run
        self.config = config
        self._is_paused = False
        self._is_stopping = False
        self._start_time: Optional[datetime] = None
        self._training_worker: Optional[Worker] = None
        self._log_queue: queue.Queue = queue.Queue()

    def compose(self) -> ComposeResult:
        """Compose the monitor layout."""
        with Container(id="monitor-header"):
            yield Static(
                f"[Side-Step] Training: {self.run.name}",
                id="monitor-title",
            )
            with Horizontal(id="header-controls"):
                yield Button("Pause", id="btn-pause", variant="warning")
                yield Button("Stop", id="btn-stop", variant="error")
                yield Button("Back", id="btn-back", variant="default")

        with Container(id="progress-section"):
            with Horizontal(id="progress-header"):
                yield Static(
                    f"Epoch 0/{self.run.total_epochs}",
                    id="epoch-label",
                )
                yield Static("ETA: calculating...", id="eta-label")
            yield ProgressBar(
                total=self.run.total_epochs, show_eta=False, id="epoch-progress"
            )

        with Horizontal(id="main-metrics"):
            with Container(id="loss-section"):
                yield LossSparkline(max_points=100, title="Loss", id="loss-graph")
            with Container(id="gpu-section"):
                yield GPUGauge(auto_refresh=True, refresh_interval=2.0, id="gpu-gauge")

        with Horizontal(id="current-stats"):
            with Vertical(classes="stat-display"):
                yield Static("Current Loss", classes="stat-name")
                yield Static("--", id="stat-current-loss", classes="stat-val")
            with Vertical(classes="stat-display"):
                yield Static("Best Loss", classes="stat-name")
                yield Static("--", id="stat-best-loss", classes="stat-val")
            with Vertical(classes="stat-display"):
                yield Static("Best Epoch", classes="stat-name")
                yield Static("--", id="stat-best-epoch", classes="stat-val")
            with Vertical(classes="stat-display"):
                yield Static("Learning Rate", classes="stat-name")
                yield Static("--", id="stat-lr", classes="stat-val")

        with Container(id="log-section"):
            yield TrainingLogViewer(max_lines=500, id="training-log")

    # =========================================================================
    # Lifecycle
    # =========================================================================

    def on_mount(self) -> None:
        """Start training when mounted."""
        self._start_time = datetime.now()
        self.app.app_state.start_run(self.run)

        log = self.query_one("#training-log", TrainingLogViewer)
        log.write_header("Training Started")
        log.info(f"Trainer: {self.run.trainer_type}")
        log.info(f"Output: {self.config.get('output_dir', 'N/A')}")
        log.info(f"Epochs: {self.config.get('epochs', 100)}")
        log.info(
            f"Batch size: {self.config.get('batch_size', 1)}"
            f" x {self.config.get('gradient_accumulation_steps', 4)}"
        )
        log.write_separator()

        # run_worker with a *sync* function runs it in a real thread,
        # keeping the UI responsive while the trainer blocks.
        self._training_worker = self.run_worker(
            self._run_training,
            name="training",
            thread=True,
        )

        # Poll the log queue from the UI thread
        self.set_interval(0.1, self._process_log_queue)

    # =========================================================================
    # Training Worker  (runs in a THREAD, not the event loop)
    # =========================================================================

    def _run_training(self) -> None:  # NOTE: sync, not async
        """Execute training in a background thread."""
        worker = get_current_worker()

        try:
            from acestep.training_v2.configs import LoRAConfigV2, TrainingConfigV2
            from acestep.training_v2.cli.common import resolve_target_modules

            # ---- build LoRA config ----------------------------------------
            target_mods_raw = self.config.get(
                "target_modules", ["q_proj", "k_proj", "v_proj", "o_proj"]
            )
            if isinstance(target_mods_raw, str):
                target_mods_raw = target_mods_raw.replace(",", " ").split()

            lora_config = LoRAConfigV2(
                r=self.config.get("rank", 64),
                alpha=self.config.get("alpha", 128),
                dropout=self.config.get("dropout", 0.0),
                target_modules=resolve_target_modules(
                    target_mods_raw,
                    self.config.get("attention_type", "both"),
                ),
                attention_type=self.config.get("attention_type", "both"),
            )

            # ---- build training config ------------------------------------
            training_config = TrainingConfigV2(
                checkpoint_dir=self.config.get("checkpoint_dir", "./checkpoints"),
                model_variant=self.config.get("variant", "turbo"),
                dataset_dir=self.config.get("dataset_dir", ""),
                output_dir=self.config.get("output_dir", ""),
                max_epochs=self.config.get("epochs", 100),
                batch_size=self.config.get("batch_size", 1),
                gradient_accumulation_steps=self.config.get(
                    "gradient_accumulation_steps", 4
                ),
                learning_rate=self.config.get("learning_rate", 1e-4),
                warmup_steps=self.config.get("warmup_steps", 500),
                log_every=self.config.get("log_every_n_steps", 10),
                save_every_n_epochs=self.config.get("save_every_n_epochs", 10),
                device=self.config.get("device", "auto"),
                precision=self.config.get("precision", "auto"),
                seed=self.config.get("seed", 42),
                weight_decay=self.config.get("weight_decay", 0.01),
                max_grad_norm=self.config.get("max_grad_norm", 1.0),
                num_workers=self.config.get("num_workers", 4),
                pin_memory=self.config.get("pin_memory", True),
                cfg_ratio=self.config.get("cfg_ratio", 0.15),
                resume_from=self.config.get("resume_from", None),
                log_dir=self.config.get("log_dir", None),
                log_heavy_every=self.config.get("log_heavy_every", 50),
                sample_every_n_epochs=self.config.get("sample_every_n_epochs", 0),
                optimizer_type=self.config.get("optimizer_type", "adamw"),
                scheduler_type=self.config.get("scheduler_type", "cosine"),
                gradient_checkpointing=self.config.get(
                    "gradient_checkpointing", True
                ),
                offload_encoder=self.config.get("offload_encoder", False),
            )

            # ---- dispatch to the right trainer ----------------------------
            if self.run.trainer_type == "vanilla":
                self._run_vanilla(lora_config, training_config, worker)
            else:
                self._run_fixed(lora_config, training_config, worker)

        except Exception as exc:
            self._log_queue.put(("error", f"Training failed: {exc}"))
            self.app.call_from_thread(self._on_training_complete, False)

    # ------------------------------------------------------------------
    # Fixed trainer (generator pattern)
    # ------------------------------------------------------------------

    def _run_fixed(
        self,
        lora_config,
        training_config,
        worker,
    ) -> None:
        """Run the corrected FixedLoRATrainer."""
        from acestep.training_v2.trainer_fixed import FixedLoRATrainer
        from acestep.training_v2.model_loader import load_decoder_for_training
        from acestep.training_v2.gpu_utils import detect_gpu

        self._log_queue.put(("info", "Loading model..."))

        gpu = detect_gpu(
            requested_device=training_config.device,
            requested_precision=training_config.precision,
        )

        model = load_decoder_for_training(
            checkpoint_dir=training_config.checkpoint_dir,
            variant=training_config.model_variant,
            device=gpu.device,
            precision=gpu.precision,
        )

        # Patch resolved device/precision back into the config
        training_config.device = gpu.device
        training_config.precision = gpu.precision

        self._log_queue.put(("info", "Model loaded, starting training..."))

        trainer = FixedLoRATrainer(
            model=model,
            lora_config=lora_config,
            training_config=training_config,
        )

        training_state: Dict[str, Any] = {"should_stop": False}
        current_epoch = 0

        for update in trainer.train(training_state):
            if worker.is_cancelled or self._is_stopping:
                training_state["should_stop"] = True
                break

            # TrainingUpdate objects support tuple unpacking (step, loss, msg)
            step = update.step if hasattr(update, "step") else update[0]
            loss = update.loss if hasattr(update, "loss") else update[1]
            msg = update.msg if hasattr(update, "msg") else update[2]
            kind = getattr(update, "kind", "info")
            epoch = getattr(update, "epoch", current_epoch)
            lr = getattr(update, "lr", 0.0)

            if epoch > current_epoch:
                current_epoch = epoch

            # Log the message
            if msg:
                log_level = "info"
                if kind == "fail":
                    log_level = "error"
                elif kind == "warn":
                    log_level = "warning"
                elif kind == "checkpoint":
                    log_level = "success"
                elif kind == "complete":
                    log_level = "success"
                self._log_queue.put((log_level, msg))

            # Update the UI on the main thread
            self.app.call_from_thread(
                self._update_progress, epoch, step, loss, lr, kind == "epoch"
            )

            if kind == "fail":
                self.app.call_from_thread(self._on_training_complete, False)
                return

            # Handle pause (we're in a thread, so time.sleep is fine)
            while self._is_paused and not self._is_stopping:
                time.sleep(0.1)

        # Finished normally
        if not self._is_stopping:
            self._log_queue.put(("success", "Training completed successfully!"))
            self.app.call_from_thread(self._on_training_complete, True)
        else:
            self._log_queue.put(("warning", "Training stopped by user"))
            self.app.call_from_thread(self._on_training_complete, False)

    # ------------------------------------------------------------------
    # Vanilla trainer (callback pattern)
    # ------------------------------------------------------------------

    def _run_vanilla(
        self,
        lora_config,
        training_config,
        worker,
    ) -> None:
        """Run the VanillaTrainer (upstream LoRATrainer wrapper)."""
        from acestep.training_v2.trainer_vanilla import VanillaTrainer

        self._log_queue.put(("info", "Loading model (vanilla)..."))

        def _callback(
            epoch: int = 0,
            step: int = 0,
            loss: float = 0.0,
            lr: float = 0.0,
            is_epoch_end: bool = False,
            **kwargs,
        ) -> bool:
            """Thread-safe progress callback."""
            self.app.call_from_thread(
                self._update_progress, epoch, step, loss, lr, is_epoch_end
            )
            if is_epoch_end:
                self._log_queue.put(
                    ("info", f"Epoch {epoch} -- loss {loss:.4f}")
                )
            # Handle pause
            while self._is_paused and not self._is_stopping:
                time.sleep(0.1)
            return not self._is_stopping

        trainer = VanillaTrainer(
            lora_config=lora_config,
            training_config=training_config,
            progress_callback=_callback,
        )

        self._log_queue.put(("info", "Starting vanilla training..."))
        trainer.train()

        if not self._is_stopping:
            self._log_queue.put(("success", "Training completed successfully!"))
            self.app.call_from_thread(self._on_training_complete, True)
        else:
            self._log_queue.put(("warning", "Training stopped by user"))
            self.app.call_from_thread(self._on_training_complete, False)

    # =========================================================================
    # UI Updates (called on the main thread)
    # =========================================================================

    def _update_progress(
        self,
        epoch: int,
        step: int,
        loss: float,
        lr: float,
        is_epoch_end: bool,
    ) -> None:
        """Update UI with training progress."""
        try:
            self.query_one("#epoch-progress", ProgressBar).progress = epoch
            self.query_one("#epoch-label", Static).update(
                f"Epoch {epoch}/{self.run.total_epochs}"
            )
        except Exception:
            pass

        # ETA
        if self._start_time and epoch > 0:
            elapsed = datetime.now() - self._start_time
            per_epoch = elapsed / epoch
            remaining = per_epoch * (self.run.total_epochs - epoch)
            eta_str = str(remaining).split(".")[0]
            try:
                self.query_one("#eta-label", Static).update(f"ETA: {eta_str}")
            except Exception:
                pass

        # Loss graph
        try:
            self.query_one("#loss-graph", LossSparkline).add_value(loss, epoch)
        except Exception:
            pass

        # Stat bar
        try:
            self.query_one("#stat-current-loss", Static).update(f"{loss:.4f}")
            if lr > 0:
                self.query_one("#stat-lr", Static).update(f"{lr:.2e}")
        except Exception:
            pass

        if loss > 0 and loss < self.run.best_loss:
            self.run.best_loss = loss
            self.run.best_epoch = epoch

        if self.run.best_loss < float("inf"):
            try:
                self.query_one("#stat-best-loss", Static).update(
                    f"{self.run.best_loss:.4f}"
                )
                self.query_one("#stat-best-epoch", Static).update(
                    str(self.run.best_epoch)
                )
            except Exception:
                pass

        self.run.current_epoch = epoch
        self.run.current_loss = loss
        try:
            self.app.app_state.update_run_progress(epoch, loss)
        except Exception:
            pass

        if is_epoch_end:
            try:
                log = self.query_one("#training-log", TrainingLogViewer)
                log.log_epoch_end(epoch, loss, lr)
            except Exception:
                pass

    def _process_log_queue(self) -> None:
        """Drain queued log messages into the log viewer."""
        try:
            log = self.query_one("#training-log", TrainingLogViewer)
        except Exception:
            return
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
                elif level == "debug":
                    log.debug(message)
            except queue.Empty:
                break

    def _on_training_complete(self, success: bool) -> None:
        """Handle training completion."""
        self.app.app_state.complete_run(success)
        try:
            self.query_one("#btn-pause", Button).disabled = True
            self.query_one("#btn-stop", Button).disabled = True
        except Exception:
            pass
        if success:
            self.notify("Training completed successfully!", timeout=5)
        else:
            self.notify("Training stopped or failed", severity="warning", timeout=5)

    # =========================================================================
    # Button Handlers
    # =========================================================================

    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button presses."""
        if event.button.id == "btn-pause":
            self.action_toggle_pause()
        elif event.button.id == "btn-stop":
            self.action_stop_training()
        elif event.button.id == "btn-back":
            self.action_back()

    # =========================================================================
    # Actions
    # =========================================================================

    def action_toggle_pause(self) -> None:
        """Toggle training pause state."""
        self._is_paused = not self._is_paused

        btn = self.query_one("#btn-pause", Button)
        log = self.query_one("#training-log", TrainingLogViewer)

        if self._is_paused:
            btn.label = "Resume"
            log.warning("Training paused")
            self.notify("Training paused", timeout=3)
            self.app.app_state.pause_run()
        else:
            btn.label = "Pause"
            log.info("Training resumed")
            self.notify("Training resumed", timeout=3)
            self.app.app_state.resume_run()

    def action_stop_training(self) -> None:
        """Stop the training."""
        if self._is_stopping:
            return

        self._is_stopping = True
        self._is_paused = False  # Unpause so it can stop

        log = self.query_one("#training-log", TrainingLogViewer)
        log.warning("Stopping training...")
        self.notify("Stopping training...", timeout=3)

        if self._training_worker:
            self._training_worker.cancel()

    def action_back(self) -> None:
        """Go back (with confirmation if running)."""
        if self.run.status == "running" and not self._is_stopping:
            self.notify(
                "Training is still running. Press Stop first.",
                severity="warning",
                timeout=3,
            )
            return
        self.app.pop_screen()

    def action_toggle_log(self) -> None:
        """Toggle log section visibility."""
        log_section = self.query_one("#log-section", Container)
        log_section.display = not log_section.display
