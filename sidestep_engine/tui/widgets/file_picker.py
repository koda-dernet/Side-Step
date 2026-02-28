"""
File/Directory Picker Widget

Interactive file browser with:
- Keyboard navigation (j/k, enter, backspace)
- Directory tree view
- File filtering by extension
- Quick path input
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Callable

from textual.app import ComposeResult
from textual.containers import Container, Horizontal
from textual.widgets import Static, Input, Button, DirectoryTree, Tree
from textual.widget import Widget
from textual.binding import Binding
from textual.message import Message
from textual.screen import ModalScreen


class FilePicker(Widget):
    """
    File/directory picker widget with tree navigation.
    
    Can be used inline or as a modal dialog.
    """
    
    DEFAULT_CSS = """
    FilePicker {
        height: 100%;
        border: round $primary 35%;
        padding: 1;
        layout: vertical;
    }
    
    #picker-header {
        height: 3;
        layout: horizontal;
        padding: 0 1;
    }
    
    #picker-path-input {
        width: 1fr;
    }
    
    #picker-path-input Input {
        width: 100%;
    }
    
    #picker-up-btn {
        width: auto;
        margin-left: 1;
    }
    
    #picker-tree {
        height: 1fr;
        border: round $primary 25%;
    }
    
    #picker-footer {
        height: 3;
        layout: horizontal;
        align: right middle;
    }
    
    #picker-footer Button {
        margin-left: 1;
    }
    
    DirectoryTree {
        height: 100%;
    }
    """
    
    BINDINGS = [
        Binding("enter", "select", "Select", show=True),
        Binding("escape", "cancel", "Cancel", show=True),
        Binding("backspace", "parent", "Parent Dir"),
        Binding("h", "parent", "Parent Dir", show=False),
    ]
    
    class Selected(Message):
        """Message sent when a path is selected."""
        def __init__(self, path: Path) -> None:
            super().__init__()
            self.path = path
    
    class Cancelled(Message):
        """Message sent when selection is cancelled."""
        pass
    
    def __init__(
        self,
        start_path: Optional[Path] = None,
        select_directory: bool = True,
        file_filter: Optional[str] = None,
        name: Optional[str] = None,
        id: Optional[str] = None,
        classes: Optional[str] = None,
    ) -> None:
        """
        Initialize the file picker.
        
        Args:
            start_path: Initial directory to show
            select_directory: If True, select directories; if False, select files
            file_filter: Glob pattern for file filtering (e.g., "*.pt")
            name: Widget name
            id: Widget ID
            classes: CSS classes
        """
        super().__init__(name=name, id=id, classes=classes)
        self.start_path = start_path or Path.cwd()
        self.select_directory = select_directory
        self.file_filter = file_filter
        self._current_path = self.start_path
    
    def compose(self) -> ComposeResult:
        """Compose the file picker layout."""
        # Path input
        with Horizontal(id="picker-header"):
            with Container(id="picker-path-input"):
                yield Input(value=str(self._current_path), id="path-input")
            yield Button("↑", id="picker-up-btn", variant="default")
        
        # Directory tree
        with Container(id="picker-tree"):
            yield DirectoryTree(self._current_path, id="dir-tree")
        
        # Footer with action buttons
        with Horizontal(id="picker-footer"):
            yield Button("Cancel", id="btn-cancel", variant="default")
            yield Button("Select", id="btn-select", variant="primary")
    
    def on_mount(self) -> None:
        """Set up the tree on mount."""
        tree = self.query_one("#dir-tree", DirectoryTree)
        tree.focus()
    
    def on_directory_tree_directory_selected(
        self, event: DirectoryTree.DirectorySelected
    ) -> None:
        """Handle directory selection in tree."""
        self._current_path = event.path
        self.query_one("#path-input", Input).value = str(event.path)
    
    def on_directory_tree_file_selected(
        self, event: DirectoryTree.FileSelected
    ) -> None:
        """Handle file selection in tree."""
        if not self.select_directory:
            self._current_path = event.path
            self.query_one("#path-input", Input).value = str(event.path)
    
    def on_input_submitted(self, event: Input.Submitted) -> None:
        """Handle path input submission."""
        if event.input.id == "path-input":
            path = Path(event.value)
            if path.exists():
                self._current_path = path
                tree = self.query_one("#dir-tree", DirectoryTree)
                tree.path = path
                tree.reload()
    
    def on_button_pressed(self, event: Button.Pressed) -> None:
        """Handle button presses."""
        if event.button.id == "btn-select":
            self.action_select()
        elif event.button.id == "btn-cancel":
            self.action_cancel()
        elif event.button.id == "picker-up-btn":
            self.action_parent()
    
    def action_select(self) -> None:
        """Select the current path."""
        self.post_message(self.Selected(self._current_path))
    
    def action_cancel(self) -> None:
        """Cancel selection."""
        self.post_message(self.Cancelled())
    
    def action_parent(self) -> None:
        """Navigate to parent directory."""
        parent = self._current_path.parent
        if parent.exists():
            self._current_path = parent
            self.query_one("#path-input", Input).value = str(parent)
            tree = self.query_one("#dir-tree", DirectoryTree)
            tree.path = parent
            tree.reload()


class FilePickerModal(ModalScreen[Optional[Path]]):
    """
    Modal dialog for file/directory selection.
    
    Usage:
        path = await self.app.push_screen_wait(FilePickerModal())
    """
    
    DEFAULT_CSS = """
    FilePickerModal {
        align: center middle;
    }
    
    #picker-container {
        width: 80%;
        height: 80%;
        max-width: 100;
        max-height: 40;
        border: round $primary;
        background: $surface;
    }
    
    #picker-title {
        dock: top;
        height: 3;
        padding: 1 2;
        background: $panel;
        text-style: bold;
    }
    
    #picker-content {
        height: 1fr;
    }
    """
    
    BINDINGS = [
        Binding("escape", "cancel", "Cancel"),
    ]
    
    def __init__(
        self,
        title: str = "Select Path",
        start_path: Optional[Path] = None,
        select_directory: bool = True,
        file_filter: Optional[str] = None,
    ) -> None:
        super().__init__()
        self.title_text = title
        self.start_path = start_path
        self.select_directory = select_directory
        self.file_filter = file_filter
    
    def compose(self) -> ComposeResult:
        with Container(id="picker-container"):
            yield Static(self.title_text, id="picker-title")
            with Container(id="picker-content"):
                yield FilePicker(
                    start_path=self.start_path,
                    select_directory=self.select_directory,
                    file_filter=self.file_filter,
                    id="modal-picker",
                )
    
    def on_file_picker_selected(self, event: FilePicker.Selected) -> None:
        """Handle selection from picker."""
        self.dismiss(event.path)
    
    def on_file_picker_cancelled(self, event: FilePicker.Cancelled) -> None:
        """Handle cancellation."""
        self.dismiss(None)
    
    def action_cancel(self) -> None:
        """Cancel and dismiss."""
        self.dismiss(None)
