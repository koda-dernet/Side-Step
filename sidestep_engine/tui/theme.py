"""
Side-Step TUI Theme
Omarchy-inspired deep navy + cyan aesthetic.

Palette reference:
  bg         #0a0e14   very deep navy
  surface    #0d1117   slightly lighter navy
  panel      #161b22   card / panel fill
  primary    #5ccfe6   bright cyan (signature)
  secondary  #c3a6ff   soft lavender
  accent     #73d0ff   light cyan for focus rings
  text       #cbccc6   warm gray-white
  muted      #565b66   dimmed labels
  success    #87d37c   green
  warning    #ffcc66   amber
  error      #f07178   soft coral red
"""

from __future__ import annotations


def register_sidestep_theme(app) -> None:
    """Register the Side-Step custom theme on the Textual App instance."""
    try:
        from textual.theme import Theme

        sidestep = Theme(
            name="sidestep",
            primary="#5ccfe6",
            secondary="#c3a6ff",
            warning="#ffcc66",
            error="#f07178",
            success="#87d37c",
            accent="#73d0ff",
            foreground="#cbccc6",
            background="#0a0e14",
            surface="#0d1117",
            panel="#161b22",
            dark=True,
            variables={
                "block-cursor-foreground": "#0a0e14",
                "input-selection-background": "#5ccfe6 30%",
            },
        )
        app.register_theme(sidestep)
        app.theme = "sidestep"
    except Exception:
        # Fallback: older Textual or missing API -- the CSS below still
        # carries hardcoded colors for the most important elements.
        pass


# ══════════════════════════════════════════════════════════════════════════════
# Application CSS
# ══════════════════════════════════════════════════════════════════════════════

APP_CSS = """
/* ============================================================================
   Global
   ============================================================================ */

Screen {
    background: $surface;
}

/* ---- Header & Footer (top / bottom bars) -------------------------------- */

Header {
    dock: top;
    height: 1;
    background: $panel;
    color: $primary;
}

Footer {
    dock: bottom;
    height: 1;
    background: $panel;
    color: $text-muted;
}

/* ---- Rules / Dividers --------------------------------------------------- */

Rule {
    color: $primary 25%;
    margin: 0 1;
}

/* ============================================================================
   Dashboard
   ============================================================================ */

#dashboard {
    layout: grid;
    grid-size: 2 3;
    grid-gutter: 1;
    padding: 1;
}

#dashboard-title {
    column-span: 2;
    height: 3;
    content-align: center middle;
    text-style: bold;
}

#quick-actions {
    column-span: 2;
    height: auto;
    layout: horizontal;
    align: center middle;
}

.action-button {
    margin: 0 1;
    min-width: 20;
}

#recent-runs {
    height: 100%;
    border: round $primary 40%;
}

#quick-stats {
    height: 100%;
    border: round $primary 40%;
}

#nav-buttons {
    column-span: 2;
    height: 3;
    layout: horizontal;
    align: center middle;
}

.nav-button {
    margin: 0 1;
}

/* ============================================================================
   Panels & Containers
   ============================================================================ */

.panel {
    border: round $primary 40%;
    padding: 1;
}

.panel-title {
    text-style: bold;
    color: $primary;
    margin-bottom: 1;
}

/* ============================================================================
   Forms
   ============================================================================ */

.form-group {
    height: auto;
    margin-bottom: 1;
}

.form-label {
    width: 20;
    text-align: right;
    padding-right: 1;
    color: $text-muted;
}

/* Input fields: subtle border, cyan glow on focus */
Input {
    width: 100%;
    border: tall $panel;
    background: $panel;
}

Input:focus {
    border: tall $accent;
}

/* Select / dropdowns */
Select {
    width: 100%;
    border: tall $panel;
    background: $panel;
}

Select:focus {
    border: tall $accent;
}

/* Switches */
Switch {
    background: $panel;
}

/* ============================================================================
   Buttons
   ============================================================================ */

Button {
    min-width: 10;
    border: none;
    background: $panel;
    color: $text;
}

Button:hover {
    background: $primary 20%;
    color: $primary;
}

Button:focus {
    text-style: bold;
}

Button.-primary {
    background: $primary 18%;
    color: $primary;
    text-style: bold;
}

Button.-primary:hover {
    background: $primary 30%;
}

Button.-success {
    background: $success 18%;
    color: $success;
}

Button.-success:hover {
    background: $success 30%;
}

Button.-warning {
    background: $warning 18%;
    color: $warning;
}

Button.-warning:hover {
    background: $warning 30%;
}

Button.-error {
    background: $error 18%;
    color: $error;
}

Button.-error:hover {
    background: $error 30%;
}

/* ============================================================================
   Training Monitor
   ============================================================================ */

#monitor-header {
    dock: top;
    height: 3;
    background: $panel;
    color: $primary;
}

#progress-section {
    height: 5;
    padding: 1;
}

#metrics-section {
    layout: horizontal;
    height: 1fr;
}

#loss-panel {
    width: 60%;
    border: round $primary 30%;
    padding: 1;
}

#gpu-panel {
    width: 40%;
    border: round $primary 30%;
    padding: 1;
}

#log-section {
    height: 10;
    border: round $primary 30%;
}

/* ============================================================================
   GPU Gauge
   ============================================================================ */

.gpu-gauge {
    height: 3;
    margin-bottom: 1;
}

.gauge-label {
    width: 8;
    color: $text-muted;
}

.gauge-bar {
    width: 1fr;
}

/* ============================================================================
   Status Indicators
   ============================================================================ */

.status-running {
    color: $success;
    text-style: bold;
}

.status-completed {
    color: $primary;
}

.status-failed {
    color: $error;
}

.status-pending {
    color: $warning;
}

/* ============================================================================
   Data Tables
   ============================================================================ */

DataTable {
    height: 100%;
}

DataTable > .datatable--cursor {
    background: $primary 20%;
    color: $text;
}

DataTable > .datatable--header {
    background: $panel;
    color: $primary;
    text-style: bold;
}

/* ============================================================================
   Tabs
   ============================================================================ */

TabbedContent {
    height: 100%;
}

ContentSwitcher {
    height: 100%;
}

TabPane {
    padding: 1;
}

Tabs {
    background: $panel;
}

Tab {
    color: $text-muted;
}

Tab.-active {
    color: $primary;
    text-style: bold;
}

Tab:hover {
    color: $primary;
}

Underline > .underline--bar {
    color: $primary 40%;
}

/* ============================================================================
   File Picker
   ============================================================================ */

#file-picker {
    height: 100%;
    border: round $primary 30%;
}

#file-picker-tree {
    height: 1fr;
}

#file-picker-input {
    dock: bottom;
    height: 3;
}

DirectoryTree {
    background: $surface;
}

/* ============================================================================
   Help Overlay
   ============================================================================ */

#help-overlay {
    align: center middle;
}

#help-panel {
    width: 60;
    height: auto;
    max-height: 30;
    border: round $primary;
    background: $panel;
    padding: 1;
}

/* ============================================================================
   Sparkline / Loss Graph
   ============================================================================ */

.sparkline {
    height: 8;
    border: round $primary 30%;
    padding: 0 1;
}

/* ============================================================================
   Log Viewer
   ============================================================================ */

RichLog {
    height: 100%;
    border: round $primary 30%;
    background: $surface;
}

/* ============================================================================
   Progress Bars
   ============================================================================ */

ProgressBar > .bar--bar {
    color: $primary;
}

ProgressBar > .bar--complete {
    color: $success;
}

/* ============================================================================
   Impact Hints (form guidance text)
   ============================================================================ */

.impact-hint {
    color: $text-muted;
    text-style: italic;
}

/* ============================================================================
   Info Panels (contextual guidance boxes)
   ============================================================================ */

.info-panel {
    border: round $secondary 25%;
    background: $panel;
    padding: 1;
    margin-bottom: 1;
}

/* ============================================================================
   Hidden utility class
   ============================================================================ */

.hidden {
    display: none;
}
"""
