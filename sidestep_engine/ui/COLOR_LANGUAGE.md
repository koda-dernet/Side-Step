# Side-Step Wizard Color Language

Standard color vocabulary for all wizard UI output. Every message goes
through one of two helpers in `prompt_helpers.py` — never call
`console.print()` or `print()` directly.

---

## Output Helpers

| Helper | When to use |
|--------|------------|
| `print_message(text, kind=...)` | Simple single-style messages. Text is Rich-escaped (safe for user input). |
| `print_rich(text)` | Multi-style lines with inline Rich markup (e.g. `[bold]label:[/] value`). **Only for trusted code strings.** |
| `blank_line()` | Emit one empty line with no trailing whitespace. |
| `section(title)` | Section headers (`bold cyan`, auto-spaced). |
| `step_indicator(n, total, label)` | Step progress tags (`[Step 3/8] Label`). |

## Semantic Kinds (`kind=`)

| `kind` | Rich style | Use for | Example |
|--------|-----------|---------|---------|
| `error` / `fail` | `red` | Errors, failures, hard stops | `"Not found: /path"` |
| `warn` / `warning` | `yellow` | Warnings, cautions, soft issues | `"Learning rate is high"` |
| `ok` / `success` | `green` | Confirmations, completions | `"Preset saved"` |
| `info` | `cyan` | Informational highlights | `"Starting preprocessing ..."` |
| `dim` | `dim` | Explanatory text, hints, tips | `"Tip: paste an absolute path"` |
| `heading` | `bold cyan` | Informational banners, summary titles | `"Turbo detected -- ..."` |
| `banner` | `bold` | Emphasized labels, list headers | `"Found 5 audio files:"` |
| `recalled` | `magenta` | Recalled/persisted values shown in prose | `"Using saved checkpoint dir"` |

> **`style=`** is the escape hatch for one-off Rich styles not covered
> by `kind`. Prefer `kind=` whenever a semantic match exists.

## Color Semantics

| Color | Meaning | Where you see it |
|-------|---------|------------------|
| **Magenta** | **Recalled / persistence** — values from settings, defaults, saved state | Prompt defaults `[./checkpoints]`, menu `> (default)`, `Y/n` hints |
| **Cyan** | **Structural / informational** — UI chrome, section headers, info highlights | `--- Section ---`, `info` kind, `heading` kind |
| **Green** | **Success / confirmation** | `ok`/`success` kind |
| **Yellow** | **Warning / caution** | `warn`/`warning` kind |
| **Red** | **Error / failure** | `error`/`fail` kind |
| **Dim** | **Explanatory / secondary** | Tips, hints, background context |
| **Bold** | **Emphasis / labels** | `banner` kind, list headers |

## Rules

1. **Never** use `console.print()` / `print()` directly for display —
   except `review_summary.py`'s Rich `Table`/`Panel` which has no
   `print_message` equivalent.
2. **Always** use `kind=` over `style=` when a semantic kind fits.
3. **Leading `\n`** is fine in text passed to `print_message` /
   `print_rich` — the helper strips it and emits bare blank lines
   (no trailing whitespace artefact).
4. **User-provided content** (paths, names, values) goes through
   `print_message` (auto-escaped) or `_esc()` when embedded in
   `print_rich` markup.
5. Multi-line continuation: indent continuation lines with `"  "` inside
   the string (the helper adds its own 2-space prefix for the first line).

## Quick Reference

```python
from sidestep_engine.ui.prompt_helpers import (
    print_message,
    print_rich,
    blank_line,
    section,
    step_indicator,
    _esc,
)

# Simple messages
print_message("Dataset built!", kind="ok")
print_message("LR too high", kind="warn")
print_message("\nHint text here", kind="dim")  # leading \n = blank line before

# Multi-style
print_rich(f"[bold]Selected:[/] {_esc(path)}")
print_rich("[dim]Checkpoint:[/] some_value")

# Spacing
blank_line()
```
