"""
Dataset Browser & Picker

Two entry points:
  1. DatasetBrowserScreen   -- full-screen standalone (dashboard [P])
     Browse directories, view info, preprocess raw audio.
  2. DatasetPickerModal      -- modal overlay (from training config "Browse Dataset")
     Pick a dataset directory and return the path to the caller.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from textual.app import ComposeResult
from textual.containers import Container, Horizontal, Vertical, ScrollableContainer
from textual.screen import Screen, ModalScreen
from textual.widgets import (
    Static,
    Button,
    DirectoryTree,
    Label,
    Input,
    Rule,
    Select,
)
from textual.binding import Binding


# ═══════════════════════════════════════════════════════════════════════════════
# Shared CSS for both screens
# ═══════════════════════════════════════════════════════════════════════════════

_SHARED_CSS = """
    #browser-header {
        height: 3;
        background: $panel;
        padding: 0 2;
        layout: horizontal;
        align: left middle;
    }

    #browser-title {
        width: 1fr;
        text-style: bold;
        color: $primary;
    }

    #browser-controls {
        width: auto;
    }

    #browser-controls Button {
        margin-left: 1;
    }

    #browser-main {
        layout: horizontal;
        height: 1fr;
        padding: 1;
    }

    #tree-panel {
        width: 40%;
        height: 100%;
        border: round $primary 35%;
        padding: 1;
    }

    #preview-panel {
        width: 60%;
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

    DirectoryTree {
        height: 1fr;
        background: $surface;
    }

    #dataset-info {
        height: auto;
    }

    .info-row {
        height: 2;
        layout: horizontal;
    }

    .info-label {
        width: 40%;
        color: $text-muted;
    }

    .info-value {
        width: 60%;
        text-style: bold;
        color: $primary;
    }

    #dataset-status {
        height: auto;
        margin-top: 1;
        padding: 1;
        text-align: center;
    }

    .status-ready {
        color: $success;
        text-style: bold;
    }

    .status-needs-preprocess {
        color: $warning;
        text-style: bold;
    }

    .status-empty {
        color: $text-muted;
    }

    #preprocess-section {
        height: auto;
        border-top: solid $primary 25%;
        margin-top: 1;
        padding-top: 1;
    }

    #preprocess-form {
        height: auto;
    }

    .form-row {
        height: auto;
        layout: horizontal;
        margin-bottom: 1;
    }

    .form-label {
        width: 30%;
        color: $text-muted;
    }

    .form-input {
        width: 70%;
    }

    .info-panel {
        border: round $secondary 25%;
        padding: 1;
        margin-bottom: 1;
        background: $panel;
    }

    .form-hint {
        color: $text-muted;
        margin-left: 4;
        margin-bottom: 1;
        text-style: italic;
    }
"""


# ═══════════════════════════════════════════════════════════════════════════════
# Helper: detect what a directory contains
# ═══════════════════════════════════════════════════════════════════════════════

def _scan_directory(path: Path) -> dict:
    """Scan a directory and return a summary dict."""
    pt_files = list(path.glob("*.pt"))
    audio_exts = ("*.mp3", "*.wav", "*.flac", "*.ogg", "*.m4a")
    audio_files = []
    for ext in audio_exts:
        audio_files.extend(path.glob(ext))
    json_files = list(path.glob("*.json"))
    total = len(pt_files) + len(audio_files) + len(json_files)

    if pt_files:
        kind = "preprocessed"
        type_label = f"Preprocessed ({len(pt_files)} .pt files)"
    elif audio_files:
        kind = "raw_audio"
        type_label = f"Raw audio ({len(audio_files)} files)"
    elif json_files:
        kind = "json"
        type_label = "JSON manifest"
    else:
        kind = "unknown"
        type_label = "Unknown"

    try:
        size = sum(f.stat().st_size for f in path.iterdir() if f.is_file())
    except Exception:
        size = 0

    return {
        "pt": len(pt_files),
        "audio": len(audio_files),
        "json": len(json_files),
        "total": total,
        "kind": kind,
        "type_label": type_label,
        "size": size,
    }


def _format_size(size_bytes: int) -> str:
    """Format byte size to human-readable."""
    for unit in ("B", "KB", "MB", "GB"):
        if size_bytes < 1024:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024
    return f"{size_bytes:.1f} TB"


# ═══════════════════════════════════════════════════════════════════════════════
# Standalone: DatasetBrowserScreen  (full-screen, dashboard [P])
# ═══════════════════════════════════════════════════════════════════════════════

class DatasetBrowserScreen(Screen):
    """Full-screen dataset browser with preprocessing support."""

    BINDINGS = [
        Binding("escape", "back", "Back"),
        Binding("r", "refresh", "Refresh"),
    ]

    CSS = (
        "DatasetBrowserScreen { layout: vertical; }\n"
        + _SHARED_CSS
    )

    def __init__(self, mode: str = "browse", name: Optional[str] = None) -> None:
        super().__init__(name=name)
        self.mode = mode
        self._selected_path: Optional[Path] = None

    def compose(self) -> ComposeResult:
        title = "Datasets & Preprocessing"

        with Container(id="browser-header"):
            yield Static(title, id="browser-title")
            with Horizontal(id="browser-controls"):
                yield Button("Preprocess", id="btn-preprocess", variant="success")
                yield Button("Refresh", id="btn-refresh", variant="default")
                yield Button("Back", id="btn-back", variant="default")

        with Horizontal(id="browser-main"):
            with Container(id="tree-panel"):
                yield Label("Directories", classes="panel-title")
                yield DirectoryTree(Path.cwd(), id="dir-tree")

            with ScrollableContainer(id="preview-panel"):
                yield Label("Dataset Info", classes="panel-title")

                with Vertical(id="dataset-info"):
                    yield Static(
                        "Select a directory to view info",
                        id="info-placeholder",
                    )
                    with Horizontal(classes="info-row", id="info-path-row"):
                        yield Static("Path:", classes="info-label")
                        yield Static("--", id="info-path", classes="info-value")
                    with Horizontal(classes="info-row", id="info-files-row"):
                        yield Static("Files:", classes="info-label")
                        yield Static("--", id="info-files", classes="info-value")
                    with Horizontal(classes="info-row", id="info-type-row"):
                        yield Static("Type:", classes="info-label")
                        yield Static("--", id="info-type", classes="info-value")
                    with Horizontal(classes="info-row", id="info-size-row"):
                        yield Static("Total Size:", classes="info-label")
                        yield Static("--", id="info-size", classes="info-value")

                # Status indicator
                yield Static("", id="dataset-status")

                yield Rule()

                # Preprocessing form
                with Container(id="preprocess-section"):
                    yield Label("Preprocessing Options", classes="panel-title")
                    yield Static(
                        "[bold]What is Preprocessing?[/bold]\n\n"
                        "Converts audio files into tensor format the model can use.\n"
                        "This only needs to be done once per dataset -- training loads\n"
                        "the preprocessed .pt files directly.",
                        classes="info-panel",
                    )
                    with Vertical(id="preprocess-form"):
                        with Horizontal(classes="form-row"):
                            yield Static("Output Directory:", classes="form-label")
                            yield Input(
                                placeholder="./datasets/preprocessed",
                                id="input-output",
                                classes="form-input",
                            )
                        yield Static(
                            "Where to save the .pt tensor files",
                            classes="form-hint",
                        )
                        with Horizontal(classes="form-row"):
                            yield Static("Checkpoint Dir:", classes="form-label")
                            yield Input(
                                placeholder="./checkpoints",
                                id="input-checkpoint",
                                classes="form-input",
                            )
                        yield Static(
                            "Path to ACE-Step model weights",
                            classes="form-hint",
                        )
                        with Horizontal(classes="form-row"):
                            yield Static("Model Variant:", classes="form-label")
                            yield Select(
                                [
                                    ("Turbo", "turbo"),
                                    ("Base", "base"),
                                    ("SFT", "sft"),
                                ],
                                value="turbo",
                                id="select-variant",
                            )
                        with Horizontal(classes="form-row"):
                            yield Static("Duration (s):", classes="form-label")
                            yield Input(
                                value="60",
                                id="input-duration",
                                classes="form-input",
                            )
                        yield Static(
                            "Max audio length to process",
                            classes="form-hint",
                        )
                        yield Button(
                            "Start Preprocessing",
                            id="btn-start-preprocess",
                            variant="primary",
                        )

    def on_mount(self) -> None:
        self._hide_info_rows()
        # Load defaults from app preferences
        try:
            prefs = self.app.app_state.preferences
            ckpt = getattr(prefs, "default_checkpoint_dir", "")
            if ckpt:
                self.query_one("#input-checkpoint", Input).value = ckpt
        except Exception:
            pass

    # ---- info rows visibility --------------------------------------------

    def _hide_info_rows(self) -> None:
        for rid in ("info-path-row", "info-files-row", "info-type-row", "info-size-row"):
            try:
                self.query_one(f"#{rid}").display = False
            except Exception:
                pass

    def _show_info_rows(self) -> None:
        try:
            self.query_one("#info-placeholder").display = False
        except Exception:
            pass
        for rid in ("info-path-row", "info-files-row", "info-type-row", "info-size-row"):
            try:
                self.query_one(f"#{rid}").display = True
            except Exception:
                pass

    # ---- directory selection handler -------------------------------------

    def on_directory_tree_directory_selected(
        self, event: DirectoryTree.DirectorySelected
    ) -> None:
        self._selected_path = event.path
        self._update_directory_info(event.path)

    def _update_directory_info(self, path: Path) -> None:
        self._show_info_rows()
        info = _scan_directory(path)

        self.query_one("#info-path", Static).update(str(path))
        self.query_one("#info-files", Static).update(f"{info['total']} total")
        self.query_one("#info-type", Static).update(info["type_label"])
        self.query_one("#info-size", Static).update(_format_size(info["size"]))

        # Status indicator
        status = self.query_one("#dataset-status", Static)
        if info["kind"] == "preprocessed":
            status.update(f"Ready to train ({info['pt']} .pt files)")
            status.set_classes("status-ready")
        elif info["kind"] == "raw_audio":
            status.update(f"Needs preprocessing ({info['audio']} audio files)")
            status.set_classes("status-needs-preprocess")
        else:
            status.update("Select a directory with .pt or audio files")
            status.set_classes("status-empty")

        # Auto-fill output path if raw audio
        if info["kind"] == "raw_audio":
            output = path.parent / f"{path.name}_preprocessed"
            try:
                self.query_one("#input-output", Input).value = str(output)
            except Exception:
                pass

    # ---- button handlers -------------------------------------------------

    def on_button_pressed(self, event: Button.Pressed) -> None:
        bid = event.button.id
        if bid == "btn-back":
            self.action_back()
        elif bid == "btn-refresh":
            self.action_refresh()
        elif bid == "btn-preprocess":
            # Just scroll to preprocess section
            try:
                self.query_one("#input-output", Input).focus()
            except Exception:
                pass
        elif bid == "btn-start-preprocess":
            self._start_preprocessing()

    def action_back(self) -> None:
        self.app.pop_screen()

    def action_refresh(self) -> None:
        try:
            self.query_one("#dir-tree", DirectoryTree).reload()
        except Exception:
            pass
        self.notify("Refreshed", timeout=2)

    def _start_preprocessing(self) -> None:
        if not self._selected_path:
            self.notify("Select a dataset directory first", severity="error")
            return
        output_dir = self.query_one("#input-output", Input).value
        if not output_dir:
            self.notify("Specify an output directory", severity="error")
            return
        checkpoint_dir = self.query_one("#input-checkpoint", Input).value
        if not checkpoint_dir:
            self.notify("Specify a checkpoint directory", severity="error")
            return

        variant = self.query_one("#select-variant", Select).value
        duration = self.query_one("#input-duration", Input).value

        from sidestep_engine.tui.screens.preprocess_monitor import (
            PreprocessMonitorScreen,
        )

        config = {
            "source_dir": str(self._selected_path),
            "output_dir": output_dir,
            "checkpoint_dir": checkpoint_dir,
            "variant": variant,
            "max_duration": float(duration) if duration else 240.0,
        }
        self.app.push_screen(PreprocessMonitorScreen(config=config))


# ═══════════════════════════════════════════════════════════════════════════════
# Modal: DatasetPickerModal  (returns Optional[str] path)
# ═══════════════════════════════════════════════════════════════════════════════

class DatasetPickerModal(ModalScreen[Optional[str]]):
    """
    Modal overlay for selecting a dataset directory.

    Usage from any screen::

        path = await self.app.push_screen_wait(DatasetPickerModal())
        if path:
            self.query_one("#input-dataset-dir", Input).value = path
    """

    BINDINGS = [
        Binding("escape", "cancel", "Cancel"),
    ]

    CSS = """
    DatasetPickerModal {
        align: center middle;
    }

    #picker-outer {
        width: 90%;
        height: 85%;
        border: round $primary;
        background: $surface;
    }
    """ + _SHARED_CSS

    def __init__(self, start_path: Optional[Path] = None) -> None:
        super().__init__()
        self._start_path = start_path or Path.cwd()
        self._selected_path: Optional[Path] = None

    def compose(self) -> ComposeResult:
        with Container(id="picker-outer"):
            with Container(id="browser-header"):
                yield Static(
                    "Select Dataset Directory",
                    id="browser-title",
                )
                with Horizontal(id="browser-controls"):
                    yield Button(
                        "Select", id="btn-select", variant="primary"
                    )
                    yield Button("Cancel", id="btn-cancel", variant="default")

            with Horizontal(id="browser-main"):
                with Container(id="tree-panel"):
                    yield Label("Directories", classes="panel-title")
                    yield DirectoryTree(self._start_path, id="dir-tree")

                with ScrollableContainer(id="preview-panel"):
                    yield Label("Dataset Info", classes="panel-title")

                    with Vertical(id="dataset-info"):
                        yield Static(
                            "Select a directory to view info",
                            id="info-placeholder",
                        )
                        with Horizontal(classes="info-row", id="info-path-row"):
                            yield Static("Path:", classes="info-label")
                            yield Static("--", id="info-path", classes="info-value")
                        with Horizontal(classes="info-row", id="info-files-row"):
                            yield Static("Files:", classes="info-label")
                            yield Static("--", id="info-files", classes="info-value")
                        with Horizontal(classes="info-row", id="info-type-row"):
                            yield Static("Type:", classes="info-label")
                            yield Static("--", id="info-type", classes="info-value")
                        with Horizontal(classes="info-row", id="info-size-row"):
                            yield Static("Total Size:", classes="info-label")
                            yield Static(
                                "--", id="info-size", classes="info-value"
                            )

                    yield Static("", id="dataset-status")

    def on_mount(self) -> None:
        for rid in ("info-path-row", "info-files-row", "info-type-row", "info-size-row"):
            try:
                self.query_one(f"#{rid}").display = False
            except Exception:
                pass

    # ---- directory selection ---------------------------------------------

    def on_directory_tree_directory_selected(
        self, event: DirectoryTree.DirectorySelected
    ) -> None:
        self._selected_path = event.path
        self._update_info(event.path)

    def _update_info(self, path: Path) -> None:
        try:
            self.query_one("#info-placeholder").display = False
        except Exception:
            pass
        for rid in ("info-path-row", "info-files-row", "info-type-row", "info-size-row"):
            try:
                self.query_one(f"#{rid}").display = True
            except Exception:
                pass

        info = _scan_directory(path)
        self.query_one("#info-path", Static).update(str(path))
        self.query_one("#info-files", Static).update(f"{info['total']} total")
        self.query_one("#info-type", Static).update(info["type_label"])
        self.query_one("#info-size", Static).update(_format_size(info["size"]))

        status = self.query_one("#dataset-status", Static)
        if info["kind"] == "preprocessed":
            status.update(f"Ready to train ({info['pt']} .pt files)")
            status.set_classes("status-ready")
        elif info["kind"] == "raw_audio":
            status.update(
                f"Contains {info['audio']} audio files -- preprocess first"
            )
            status.set_classes("status-needs-preprocess")
        else:
            status.update("")
            status.set_classes("status-empty")

    # ---- buttons ---------------------------------------------------------

    def on_button_pressed(self, event: Button.Pressed) -> None:
        if event.button.id == "btn-select":
            self._do_select()
        elif event.button.id == "btn-cancel":
            self.action_cancel()

    def _do_select(self) -> None:
        if not self._selected_path:
            self.notify("Select a directory first", severity="warning")
            return
        self.dismiss(str(self._selected_path))

    def action_cancel(self) -> None:
        self.dismiss(None)
