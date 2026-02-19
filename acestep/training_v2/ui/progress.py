"""
Live training progress display using Rich.

Renders a live-updating dashboard that shows:
    - Epoch progress bar with ETA
    - Step-level progress bar within the current epoch
    - Current metrics (loss, learning rate, speed)
    - GPU VRAM usage bar
    - Scrolling log of recent messages

Falls back to plain ``print(msg)`` when Rich is unavailable or stdout
is not a TTY.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple, Union

from acestep.training_v2.ui import TrainingUpdate, console, is_rich_active
from acestep.training_v2.ui.gpu_monitor import GPUMonitor


# ---- Logging capture (prevents ghost panels in tmux / web terminals) --------

class _LiveLogCapture(logging.Handler):
    """Redirects log messages into the panel's scrolling log area.

    During a Rich Live session, any direct writes to stderr (including
    from Python's ``logging.StreamHandler``) break the ANSI cursor
    positioning that Live uses to overwrite the panel in-place.  This
    causes "ghost panels" — stale copies of the display that get pushed
    up.  The problem is especially visible in tmux and web terminals.

    This handler captures log messages into the ``recent_msgs`` list
    that the panel display reads from, and a ``live`` reference so the
    display refreshes immediately after the log message is captured.
    The file handler (``sidestep.log``) is unaffected and keeps logging
    normally.
    """

    def __init__(
        self,
        messages: list,
        live: object = None,
        session_logger: "_SessionTextLogger | None" = None,
    ) -> None:
        super().__init__(level=logging.INFO)
        self._messages = messages
        self._live = live
        self._session_logger = session_logger

    def emit(self, record: logging.LogRecord) -> None:
        try:
            msg = record.getMessage()
            _append_recent_message(self._messages, msg)
            if self._session_logger is not None:
                self._session_logger.log(msg)
        except Exception:
            pass


# ---- Session log sink --------------------------------------------------------

class _SessionTextLogger:
    """Line-based session logger for UI/training events."""

    def __init__(self, path: str | Path, session_name: str = "") -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = self.path.open("a", encoding="utf-8", buffering=1)
        self._closed = False
        self._session_name = session_name.strip() or "session"
        if self.path.stat().st_size == 0:
            self._fh.write(f"# Side-Step session UI log\n")
            self._fh.write(f"# session: {self._session_name}\n")
            self._fh.write(f"# started: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")

    def log(self, msg: object) -> None:
        if self._closed:
            return
        line = _normalize_live_log_message(msg)
        if not line:
            return
        self._fh.write(f"{time.strftime('%Y-%m-%d %H:%M:%S')} {line}\n")

    def close(self) -> None:
        if self._closed:
            return
        try:
            self._fh.flush()
            self._fh.close()
        finally:
            self._closed = True


# ---- Training statistics tracker --------------------------------------------

@dataclass
class TrainingStats:
    """Accumulates statistics during training for the live display and
    the post-training summary.
    """

    start_time: float = 0.0
    first_loss: float = 0.0
    best_loss: float = float("inf")
    last_loss: float = 0.0
    last_lr: float = 0.0
    _lr_seen: bool = False
    current_epoch: int = 0
    session_start_epoch: int = -1
    """Completed-epoch baseline for this process session.

    Prefer checkpoint metadata (``resume_start_epoch`` from ``TrainingUpdate``)
    when available. Otherwise infer from the first seen epoch as ``epoch - 1``.
    """
    max_epochs: int = 0
    current_step: int = 0
    total_steps_estimate: int = 0
    steps_this_session: int = 0
    peak_vram_mb: float = 0.0
    last_epoch_time: float = 0.0
    steps_per_epoch: int = 0
    """Total optimizer steps per epoch (for step-level progress bar)."""
    step_in_epoch: int = 0
    """Current step index within the epoch (resets each epoch)."""
    _step_times: list = field(default_factory=list)
    checkpoints: List[Dict[str, object]] = field(default_factory=list)
    """Saved checkpoints: ``[{"epoch": int, "loss": float, "path": str}, ...]``."""
    attention_backend: str = "unknown"
    """Selected attention backend (flash_attention_2/sdpa/eager)."""
    device_label: str = ""
    """Device label shown in the live panel."""

    @property
    def elapsed(self) -> float:
        if self.start_time <= 0:
            return 0.0
        return time.time() - self.start_time

    @property
    def elapsed_str(self) -> str:
        return _fmt_duration(self.elapsed)

    @property
    def samples_per_sec(self) -> float:
        if not self._step_times or len(self._step_times) < 2:
            return 0.0
        dt = self._step_times[-1] - self._step_times[0]
        if dt <= 0:
            return 0.0
        return (len(self._step_times) - 1) / dt

    @property
    def eta_seconds(self) -> float:
        if self.max_epochs <= 0 or self.current_epoch <= 0:
            return 0.0
        remaining_epochs = self.max_epochs - self.current_epoch
        if remaining_epochs <= 0:
            return 0.0

        # Prefer measured epoch duration once available (stable on resume).
        if self.last_epoch_time > 0:
            return self.last_epoch_time * remaining_epochs

        elapsed = self.elapsed
        if elapsed <= 0:
            return 0.0

        if self.session_start_epoch < 0:
            return 0.0
        session_epochs = self.current_epoch - self.session_start_epoch
        if session_epochs <= 0:
            return 0.0
        return (elapsed / session_epochs) * remaining_epochs

    @property
    def eta_str(self) -> str:
        eta = self.eta_seconds
        if eta <= 0:
            return "--"
        return _fmt_duration(eta)

    def record_step(self) -> None:
        now = time.time()
        self._step_times.append(now)
        if len(self._step_times) > 50:
            self._step_times = self._step_times[-50:]

    def note_epoch(self, epoch: int) -> None:
        """Capture first seen epoch as this session's ETA baseline."""
        if epoch <= 0:
            return
        if self.session_start_epoch < 0:
            self.session_start_epoch = max(epoch - 1, 0)

    def note_resume_start_epoch(self, start_epoch: int) -> None:
        """Use explicit checkpoint epoch as ETA baseline."""
        if start_epoch < 0:
            return
        self.session_start_epoch = max(start_epoch, 0)


def _fmt_duration(seconds: float) -> str:
    """Format seconds to ``1h 23m 45s`` or ``12m 34s`` or ``45s``."""
    if seconds < 0:
        return "--"
    s = int(seconds)
    h, s = divmod(s, 3600)
    m, s = divmod(s, 60)
    if h > 0:
        return f"{h}h {m:02d}m {s:02d}s"
    if m > 0:
        return f"{m}m {s:02d}s"
    return f"{s}s"


# ---- Rich live display builder ----------------------------------------------

_LOG_LINES = 5  # Fixed number of log lines for stable panel height
_LOG_HISTORY = 20
_MIN_LOG_WIDTH = 48
_LOG_WIDTH_FALLBACK = 110


def _normalize_live_log_message(msg: object) -> str:
    """Normalize one update to a single display line.

    Rich Live rendering needs stable frame height. Embedded newlines in
    updates (or weird whitespace) can make the panel grow/shrink and create
    visual stacking artifacts in some terminals.
    """
    if msg is None:
        return ""
    text = str(msg).replace("\r\n", "\n").replace("\r", "\n")
    parts = [segment.strip() for segment in text.split("\n") if segment.strip()]
    if not parts:
        return ""
    return " | ".join(" ".join(part.split()) for part in parts)


def _log_char_limit() -> int:
    """Estimate max chars per log row to avoid soft wrapping."""
    width = _LOG_WIDTH_FALLBACK
    if console is not None:
        try:
            width = int(console.size.width)
        except Exception:
            pass
    # Keep headroom for panel border/padding and row indent.
    return max(_MIN_LOG_WIDTH, width - 16)


def _truncate_for_log_panel(msg: str, max_chars: int) -> str:
    """Truncate one-line messages for the fixed-height panel log."""
    if max_chars <= 0 or len(msg) <= max_chars:
        return msg
    if max_chars <= 3:
        return "." * max_chars
    return msg[: max_chars - 3].rstrip() + "..."


def _append_recent_message(recent_msgs: list[str], msg: object) -> None:
    """Append normalized message while capping and de-duplicating history."""
    normalized = _normalize_live_log_message(msg)
    if not normalized:
        return
    if recent_msgs and recent_msgs[-1] == normalized:
        return
    recent_msgs.append(normalized)
    if len(recent_msgs) > _LOG_HISTORY:
        recent_msgs.pop(0)


def _memory_snapshot_mb() -> tuple[float, float, float, float]:
    """Return process/system RAM in MiB.

    Returns ``(proc_mb, sys_used_mb, sys_total_mb, sys_pct)`` and caches
    values for 2 seconds to keep refresh overhead low.
    """
    now = time.monotonic()
    cache = getattr(_memory_snapshot_mb, "_cache", None)
    if cache is not None:
        ts, value = cache
        if (now - ts) < 2.0:
            return value

    value = (0.0, 0.0, 0.0, 0.0)
    try:
        import psutil

        proc = psutil.Process()
        proc_mb = proc.memory_info().rss / (1024 ** 2)
        vm = psutil.virtual_memory()
        value = (
            proc_mb,
            vm.used / (1024 ** 2),
            vm.total / (1024 ** 2),
            float(vm.percent),
        )
    except Exception:
        pass

    setattr(_memory_snapshot_mb, "_cache", (now, value))
    return value


def _build_display(
    stats: TrainingStats,
    gpu: GPUMonitor,
    recent_msgs: list,
) -> Any:
    """Build the composite Rich renderable for one Live refresh."""
    from rich.console import Group
    from rich.panel import Panel
    from rich.progress_bar import ProgressBar
    from rich.table import Table
    from rich.text import Text

    # -- Epoch progress -------------------------------------------------------
    epoch_pct = 0.0
    if stats.max_epochs > 0:
        epoch_pct = stats.current_epoch / stats.max_epochs
    epoch_bar = ProgressBar(total=100, completed=int(epoch_pct * 100), width=40)

    epoch_line = Text()
    epoch_line.append("  Epoch ", style="dim")
    epoch_line.append(f"{stats.current_epoch}", style="bold")
    epoch_line.append(f" / {stats.max_epochs}  ", style="dim")
    epoch_line.append_text(Text.from_markup(f"  Step {stats.current_step}"))
    epoch_line.append(f"  |  ETA {stats.eta_str}", style="dim")

    # -- Step-level progress (within epoch) -----------------------------------
    # Only show the inner step bar when there are enough steps per epoch
    # to make it meaningful.  For small datasets (e.g. 3 steps/epoch) the
    # bar just flickers up/down and is more confusing than helpful.
    _MIN_STEPS_FOR_BAR = 10
    step_parts: list = []
    if stats.steps_per_epoch >= _MIN_STEPS_FOR_BAR:
        shown_step = max(stats.step_in_epoch, 0)
        step_pct = min(shown_step / stats.steps_per_epoch, 1.0)
        step_bar = ProgressBar(
            total=100, completed=int(step_pct * 100), width=30,
        )
        step_line = Text()
        step_line.append(f"  Step {shown_step}", style="dim")
        step_line.append(f" / {stats.steps_per_epoch}  ", style="dim")
        step_bar_row = Table.grid(padding=0)
        step_bar_row.add_column(width=2)
        step_bar_row.add_column()
        step_bar_row.add_column(width=6)
        step_bar_row.add_row(
            Text("  "),
            step_bar,
            Text.from_markup(f"[dim]{step_pct * 100:.0f}%[/]"),
        )
        step_parts = [step_line, step_bar_row]

    # -- Metrics table --------------------------------------------------------
    metrics = Table(
        show_header=False, show_edge=False, pad_edge=False,
        box=None, expand=True,
    )
    metrics.add_column("key", style="dim", ratio=1)
    metrics.add_column("val", ratio=1)
    metrics.add_column("key2", style="dim", ratio=1)
    metrics.add_column("val2", ratio=1)

    loss_str = f"{stats.last_loss:.4f}" if stats.last_loss > 0 else "--"
    best_str = f"{stats.best_loss:.4f}" if stats.best_loss < float("inf") else "--"
    lr_str = f"{stats.last_lr:.2e}" if stats._lr_seen else "--"
    speed_str = (
        f"{stats.samples_per_sec:.1f} steps/s"
        if stats.samples_per_sec > 0 else "--"
    )
    attn_str = stats.attention_backend or "--"
    device_str = stats.device_label or "--"
    proc_ram_mb, sys_used_mb, sys_total_mb, sys_pct = _memory_snapshot_mb()
    proc_ram_str = f"{proc_ram_mb / 1024:.2f} GiB" if proc_ram_mb > 0 else "--"
    if sys_total_mb > 0:
        sys_ram_str = f"{sys_used_mb / 1024:.1f}/{sys_total_mb / 1024:.1f} GiB ({sys_pct:.0f}%)"
    else:
        sys_ram_str = "--"

    metrics.add_row("Loss", f"[bold]{loss_str}[/]", "Best", f"[green]{best_str}[/]")
    metrics.add_row("LR", lr_str, "Speed", speed_str)
    metrics.add_row(
        "Elapsed", stats.elapsed_str,
        "Epoch time",
        f"{stats.last_epoch_time:.1f}s" if stats.last_epoch_time > 0 else "--",
    )
    metrics.add_row("Attn", attn_str, "Device", device_str)
    metrics.add_row("Proc RAM", proc_ram_str, "System RAM", sys_ram_str)

    # -- VRAM bar -------------------------------------------------------------
    vram_global_line = ""
    if gpu.available:
        snap = gpu.snapshot()
        pct = snap.percent
        bar_width = 30
        filled = int(bar_width * pct / 100)
        bar_color = "green" if pct < 70 else ("yellow" if pct < 90 else "red")
        bar = f"[{bar_color}]{'#' * filled}[/][dim]{'-' * (bar_width - filled)}[/]"
        vram_line = (
            f"  VRAM {bar}  "
            f"{snap.used_gb:.1f} / {snap.total_gb:.1f} GiB  "
            f"[dim]({pct:.0f}%)[/]"
        )
        global_used_mb = float(getattr(snap, "global_used_mb", 0.0) or 0.0)
        total_mb = float(getattr(snap, "total_mb", 0.0) or 0.0)
        global_used_gb = float(
            getattr(snap, "global_used_gb", global_used_mb / 1024.0)
            or 0.0
        )
        if global_used_mb > 0 and total_mb > 0:
            global_pct = (global_used_mb / total_mb) * 100.0
            vram_global_line = (
                f"  [dim]VRAM global (all processes): "
                f"{global_used_gb:.1f} / {snap.total_gb:.1f} GiB "
                f"({global_pct:.0f}%)[/]"
            )
        else:
            vram_global_line = "  [dim]VRAM global (all processes): --[/]"
    else:
        vram_line = "  [dim]VRAM monitoring not available[/]"

    # -- Recent log (fixed height for stable panel) ---------------------------
    log_text = Text(no_wrap=True, overflow="ellipsis")
    log_limit = _log_char_limit()
    padded = recent_msgs[-_LOG_LINES:]
    while len(padded) < _LOG_LINES:
        padded.insert(0, "")
    for msg in padded:
        if not msg:
            log_text.append("  \n")
        else:
            line = _truncate_for_log_panel(msg, log_limit)
            if line.startswith("[OK]"):
                log_text.append(f"  {line}\n", style="green")
            elif line.startswith("[WARN]"):
                log_text.append(f"  {line}\n", style="yellow")
            elif line.startswith("[FAIL]"):
                log_text.append(f"  {line}\n", style="red")
            elif line.startswith("[INFO]"):
                log_text.append(f"  {line}\n", style="blue")
            else:
                log_text.append(f"  {line}\n", style="dim")

    # -- Assemble panel -------------------------------------------------------
    epoch_bar_row = Table.grid(padding=0)
    epoch_bar_row.add_column(width=2)
    epoch_bar_row.add_column()
    epoch_bar_row.add_column(width=6)
    epoch_bar_row.add_row(
        Text("  "),
        epoch_bar,
        Text.from_markup(f"[dim]{epoch_pct * 100:.0f}%[/]"),
    )

    parts: list = [
        epoch_line,
        Text(""),
        epoch_bar_row,
    ]
    parts.extend(step_parts)
    parts.extend([
        Text(""),
        metrics,
        Text(""),
        Text.from_markup(vram_line),
    ])
    if vram_global_line:
        parts.append(Text.from_markup(vram_global_line))
    parts.extend([
        Text(""),
        log_text,
    ])

    return Panel(
        Group(*parts),
        title="[bold]Side-Step Training Progress[/]",
        border_style="green",
        padding=(0, 1),
    )


# ---- Main entry point -------------------------------------------------------

def track_training(
    training_iter: Iterator[Union[Tuple[int, float, str], TrainingUpdate]],
    max_epochs: int,
    device: str = "cuda:0",
    refresh_per_second: int = 2,
    attention_backend: Optional[str] = None,
    session_log_path: Optional[str] = None,
    session_name: str = "",
) -> TrainingStats:
    """Consume training yields and display live progress.

    Args:
        training_iter: Generator yielding ``(step, loss, msg)`` or
            ``TrainingUpdate`` objects.
        max_epochs: Total number of epochs (for progress bar).
        device: Device string for GPU monitoring.
        refresh_per_second: Rich Live refresh rate.
        attention_backend: Selected attention backend label.
        session_log_path: Optional per-session UI log file path.
        session_name: Friendly session name used in the UI log header.

    Returns:
        Final ``TrainingStats`` for the summary display.
    """
    stats = TrainingStats(
        start_time=time.time(),
        max_epochs=max_epochs,
        attention_backend=attention_backend or "unknown",
        device_label=device,
    )
    gpu = GPUMonitor(device=device, interval=3.0)
    recent_msgs: list[str] = []
    session_logger: _SessionTextLogger | None = None
    if session_log_path:
        try:
            session_logger = _SessionTextLogger(session_log_path, session_name=session_name)
            session_logger.log(f"[INFO] Progress tracking started (device={device})")
        except Exception:
            session_logger = None

    try:
        if is_rich_active() and console is not None:
            return _track_rich(
                training_iter,
                stats,
                gpu,
                recent_msgs,
                refresh_per_second,
                session_logger=session_logger,
            )
        return _track_plain(training_iter, stats, gpu, recent_msgs, session_logger=session_logger)
    finally:
        if session_logger is not None:
            try:
                session_logger.log("[INFO] Progress tracking ended")
            except Exception:
                pass
            session_logger.close()


def _track_rich(
    training_iter: Iterator,
    stats: TrainingStats,
    gpu: GPUMonitor,
    recent_msgs: list,
    refresh_per_second: int,
    session_logger: "_SessionTextLogger | None" = None,
) -> TrainingStats:
    """Rich Live display loop.

    During the Live session, Python logging is redirected from stderr into
    the panel's scrolling log area.  This prevents log output (e.g. from
    checkpoint saves) from breaking Rich's ANSI cursor positioning, which
    caused "ghost panels" — especially in tmux and web terminals.  The file
    handler (sidestep.log) is unaffected.

    Rendering uses manual refresh to avoid background redraw races.
    """
    import warnings
    from rich.live import Live

    assert console is not None

    def _all_loggers() -> list[logging.Logger]:
        out: list[logging.Logger] = [logging.getLogger()]
        for obj in logging.Logger.manager.loggerDict.values():
            if isinstance(obj, logging.Logger):
                out.append(obj)
        # De-duplicate by identity while preserving order.
        seen: set[int] = set()
        uniq: list[logging.Logger] = []
        for lg in out:
            ident = id(lg)
            if ident in seen:
                continue
            seen.add(ident)
            uniq.append(lg)
        return uniq

    # Suppress warnings that print to stderr during Live and disrupt
    # Rich's cursor positioning (e.g. Lightning SLURM, fork safety).
    warnings.filterwarnings(
        "ignore", message=".*srun.*command is available.*",
    )
    warnings.filterwarnings(
        "ignore", message=".*fork.*",
    )

    # -- Redirect logging to prevent ghost panels ---------------------------
    # Mute stream handlers from *all* configured loggers, not just root.
    # Some libraries install non-root handlers (or disable propagation),
    # which can still write directly to stderr and break Live cursor math.
    root_logger = logging.getLogger()
    all_loggers = _all_loggers()
    muted_handlers: list[tuple[logging.Logger, logging.Handler]] = []
    for lg in all_loggers:
        for h in list(lg.handlers):
            if isinstance(h, logging.StreamHandler) and not isinstance(h, logging.FileHandler):
                lg.removeHandler(h)
                muted_handlers.append((lg, h))

    capture = _LiveLogCapture(recent_msgs, session_logger=session_logger)
    root_logger.addHandler(capture)
    capture_on_nonprop: list[logging.Logger] = []
    for lg in all_loggers:
        if lg is root_logger:
            continue
        if not lg.propagate:
            lg.addHandler(capture)
            capture_on_nonprop.append(lg)

    try:
        with Live(
            _build_display(stats, gpu, recent_msgs),
            console=console,
            refresh_per_second=refresh_per_second,
            transient=False,
            redirect_stdout=True,
            redirect_stderr=True,
            auto_refresh=False,
            vertical_overflow="crop",
        ) as live:
            for update in training_iter:
                if isinstance(update, TrainingUpdate):
                    step, loss, msg = update.step, update.loss, update.msg
                    _process_structured(update, stats)
                else:
                    step, loss, msg = update
                    _process_tuple(step, loss, msg, stats)

                _append_recent_message(recent_msgs, msg)
                if session_logger is not None:
                    session_logger.log(msg)

                live.update(_build_display(stats, gpu, recent_msgs), refresh=True)
    finally:
        # Restore original console handlers
        root_logger.removeHandler(capture)
        for lg in capture_on_nonprop:
            try:
                lg.removeHandler(capture)
            except Exception:
                pass
        for lg, h in muted_handlers:
            lg.addHandler(h)

    stats.peak_vram_mb = gpu.peak_mb()
    return stats


def _track_plain(
    training_iter: Iterator,
    stats: TrainingStats,
    gpu: GPUMonitor,
    recent_msgs: list,
    session_logger: "_SessionTextLogger | None" = None,
) -> TrainingStats:
    """Plain-text fallback (no Rich)."""
    for update in training_iter:
        if isinstance(update, TrainingUpdate):
            step, loss, msg = update.step, update.loss, update.msg
            _process_structured(update, stats)
        else:
            step, loss, msg = update
            _process_tuple(step, loss, msg, stats)

        print(msg)
        if session_logger is not None:
            session_logger.log(msg)

    stats.peak_vram_mb = gpu.peak_mb()
    return stats


# ---- Update processing helpers ----------------------------------------------

def _process_structured(update: TrainingUpdate, stats: TrainingStats) -> None:
    """Extract stats from a TrainingUpdate."""
    stats.note_resume_start_epoch(update.resume_start_epoch)
    stats.current_step = update.step
    stats.last_loss = update.loss
    stats.current_epoch = update.epoch
    stats.note_epoch(update.epoch)
    if update.max_epochs > 0:
        stats.max_epochs = update.max_epochs
    if update.lr >= 0 and update.kind == "step":
        stats.last_lr = update.lr
        stats._lr_seen = True
    if update.epoch_time > 0:
        stats.last_epoch_time = update.epoch_time
    if update.steps_per_epoch > 0:
        stats.steps_per_epoch = update.steps_per_epoch

    if stats.first_loss == 0.0 and update.loss > 0:
        stats.first_loss = update.loss
    if update.loss > 0 and update.loss < stats.best_loss:
        stats.best_loss = update.loss

    if update.kind == "step":
        stats.record_step()
        stats.steps_this_session += 1

    # Track step position within the current epoch.
    # Derived from global_step so the bar stays correct even when
    # updates are sparse (log_every > 1).  At epoch boundaries
    # (kind="epoch") we show the bar as fully complete rather than
    # resetting to 0, which looked broken ("Step 0 / 3").
    if stats.steps_per_epoch > 0 and stats.current_step > 0:
        if update.kind == "epoch":
            stats.step_in_epoch = stats.steps_per_epoch  # show as complete
        else:
            stats.step_in_epoch = (
                (stats.current_step - 1) % stats.steps_per_epoch
            ) + 1

    if update.kind == "checkpoint":
        stats.checkpoints.append({
            "epoch": update.epoch,
            "loss": update.loss,
            "path": update.checkpoint_path,
        })


def _process_tuple(step: int, loss: float, msg: str, stats: TrainingStats) -> None:
    """Extract stats from a raw ``(step, loss, msg)`` tuple by parsing the msg."""
    stats.current_step = step
    stats.last_loss = loss

    if stats.first_loss == 0.0 and loss > 0:
        stats.first_loss = loss
    if loss > 0 and loss < stats.best_loss:
        stats.best_loss = loss

    msg_lower = msg.lower()
    if "epoch" in msg_lower:
        try:
            idx = msg.lower().index("epoch")
            rest = msg[idx + 5:].strip()
            parts = rest.split("/")
            if len(parts) >= 2:
                epoch_num = int(parts[0].strip())
                max_part = parts[1].split(",")[0].split(" ")[0].strip()
                max_epochs = int(max_part)
                stats.current_epoch = epoch_num
                stats.note_epoch(epoch_num)
                if max_epochs > 0:
                    stats.max_epochs = max_epochs
        except (ValueError, IndexError):
            pass

    if " in " in msg and ("s," in msg or msg.rstrip().endswith("s")):
        try:
            time_part = msg.split(" in ")[1].split("s")[0].strip()
            stats.last_epoch_time = float(time_part)
        except (IndexError, ValueError):
            pass

    if msg.startswith("Epoch") and "Step" in msg and "Loss" in msg:
        stats.record_step()
        stats.steps_this_session += 1
