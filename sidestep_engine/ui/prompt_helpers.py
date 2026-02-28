"""
Reusable Rich/fallback prompt helpers for the interactive wizard.

Provides menu selection, typed value prompts, path prompts, boolean prompts,
section headers, go-back navigation, and step indicators -- with automatic
Rich fallback to plain ``input()``.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import Any, Optional

from sidestep_engine.ui import console, is_rich_active

# Windows uses spawn-based multiprocessing which breaks DataLoader workers
IS_WINDOWS = sys.platform == "win32"
DEFAULT_NUM_WORKERS = 0 if IS_WINDOWS else 4

# Back-navigation keyword recognised by all prompts
_BACK_KEYWORDS = {"b", "back"}


def _esc(text: object) -> str:
    """Escape Rich markup characters in user-provided text for safe display.

    Replaces ``[`` with ``\\[`` so that paths and other user input containing
    square brackets (e.g. ``/media/user/[volume]/path``) are not interpreted
    as Rich markup tags.
    """
    return str(text).replace("[", "\\[")


def native_path(path: str) -> str:
    """Convert a path string to use the native OS separator for display.

    On Windows, replaces forward slashes with backslashes so that path
    defaults shown in wizard prompts look natural to the user
    (e.g. ``.\\checkpoints`` instead of ``./checkpoints``).

    On Linux/macOS, returns the path unchanged.
    """
    if IS_WINDOWS:
        return path.replace("/", "\\")
    return path


# ---- Go-back exception -----------------------------------------------------

class GoBack(Exception):
    """Raised when the user types 'b' or 'back' at any prompt."""


def _is_back(raw: str) -> bool:
    """Return True if the raw input string is a back-navigation request."""
    return raw.strip().lower() in _BACK_KEYWORDS


# ---- Step indicator ---------------------------------------------------------

def step_indicator(current: int, total: int, label: str) -> None:
    """Print a step progress indicator, e.g. ``[Step 3/8] LoRA Settings``."""
    tag = f"Step {current}/{total}"
    if is_rich_active() and console is not None:
        # Use \[ to escape the bracket so Rich doesn't parse it as markup
        console.print(f"\n  [bold green]\\[{tag}][/] [bold]{label}[/]")
    else:
        print(f"\n  [{tag}] {label}")


# ---- Helpers ----------------------------------------------------------------

_KIND_STYLES: dict[str, str] = {
    "warn": "yellow",
    "warning": "yellow",
    "fail": "red",
    "error": "red",
    "ok": "green",
    "success": "green",
    "info": "cyan",
    "dim": "dim",
    "heading": "bold cyan",
    "banner": "bold",
    "recalled": "magenta",
}


def print_message(
    text: object,
    style: str | None = None,
    *,
    kind: str | None = None,
) -> None:
    """Print one consistently formatted message line.

    Leading newlines in *text* are emitted as bare blank lines (without
    trailing whitespace) so callers can space paragraphs without the
    ``"  \\n"`` artefact that a 2-space indent would produce.

    Args:
        text: Message text (Rich markup characters are escaped).
        style: Explicit Rich style string (e.g. ``"bold red"``).
        kind: Convenience alias mapped to a style. One of
            ``warn``, ``error``, ``ok``, ``info``, ``dim``,
            ``heading``, ``banner``.
            Ignored when *style* is provided.
    """
    raw = str(text)
    # Emit leading newlines as bare blank lines (no trailing spaces).
    while raw.startswith("\n"):
        if is_rich_active() and console is not None:
            console.print()
        else:
            print()
        raw = raw[1:]

    resolved = style or _KIND_STYLES.get(kind or "", "")
    if is_rich_active() and console is not None:
        body = _esc(raw)
        if resolved:
            console.print(f"  [{resolved}]{body}[/]")
        else:
            console.print(f"  {body}")
    else:
        print(f"  {raw}")


def print_rich(text: str) -> None:
    """Print a message that contains Rich markup (trusted internal text).

    Unlike ``print_message``, Rich markup tags in *text* are **not**
    escaped — use this only for messages authored in code, never for
    user-provided content.  Adds a 2-space indent for visual alignment.

    In plain mode, Rich tags are stripped via regex before printing.
    """
    raw = str(text)
    # Emit leading newlines as bare blank lines (no trailing spaces).
    while raw.startswith("\n"):
        if is_rich_active() and console is not None:
            console.print()
        else:
            print()
        raw = raw[1:]

    if is_rich_active() and console is not None:
        console.print(f"  {raw}")
    else:
        import re
        # Strip Rich tags but preserve escaped brackets (\[)
        plain = re.sub(r"(?<!\\)\[/?[^\]]*\]", "", raw)
        plain = plain.replace("\\[", "[")
        print(f"  {plain}")


def blank_line() -> None:
    """Print a single empty line with no trailing whitespace."""
    if is_rich_active() and console is not None:
        console.print()
    else:
        print()


def menu(
    title: str,
    options: list[tuple[str, str]],
    default: int = 1,
    allow_back: bool = True,
) -> str:
    """Display a numbered menu and return the chosen key.

    Args:
        title: Prompt text.
        options: List of ``(key, label)`` tuples.
        default: 1-based default index.
        allow_back: If True, typing 'b'/'back' raises ``GoBack``.

    Returns:
        The ``key`` of the chosen option.

    Raises:
        GoBack: When ``allow_back`` is True and user types 'b'/'back'.
    """
    back_hint = "  [dim]Type 'b' to go back[/]" if allow_back else ""
    back_hint_plain = "  Type 'b' to go back" if allow_back else ""

    if is_rich_active() and console is not None:
        console.print()
        console.print(f"  [bold]{title}[/]\n")
        for i, (key, label) in enumerate(options, 1):
            marker = "[bold magenta]>[/]" if i == default else " "
            tag = "  [magenta](default)[/]" if i == default else ""
            console.print(f"    {marker} [bold]{i}[/]. {label}{tag}")
        if back_hint:
            console.print(back_hint)
        console.print()

        from rich.prompt import IntPrompt
        while True:
            raw = console.input("  Choice: ") if allow_back else None
            if allow_back and raw is not None:
                if _is_back(raw):
                    raise GoBack()
                try:
                    choice = int(raw) if raw.strip() else default
                except ValueError:
                    console.print(f"  [red]Please enter a number between 1 and {len(options)}[/]")
                    continue
            else:
                choice = IntPrompt.ask(
                    "  Choice",
                    default=default,
                    console=console,
                )
            if 1 <= choice <= len(options):
                return options[choice - 1][0]
            console.print(f"  [red]Please enter a number between 1 and {len(options)}[/]")
    else:
        print(f"\n  {title}\n")
        for i, (key, label) in enumerate(options, 1):
            tag = " (default)" if i == default else ""
            print(f"    {i}. {label}{tag}")
        if back_hint_plain:
            print(back_hint_plain)
        print()
        while True:
            try:
                raw = input(f"  Choice [{default}]: ").strip()
                if allow_back and _is_back(raw):
                    raise GoBack()
                choice = int(raw) if raw else default
                if 1 <= choice <= len(options):
                    return options[choice - 1][0]
                print(f"  Please enter a number between 1 and {len(options)}")
            except ValueError:
                print(f"  Please enter a number between 1 and {len(options)}")


def ask(
    label: str,
    default: Any = None,
    required: bool = False,
    type_fn: type = str,
    choices: Optional[list] = None,
    allow_back: bool = True,
    validate_fn: Optional[Any] = None,
) -> Any:
    """Ask for a single value with an optional default.

    Args:
        label: Prompt text.
        default: Default value (None = required).
        required: If True, empty input is rejected.
        type_fn: Cast function (str, int, float).
        choices: Optional list of valid string values.
        allow_back: If True, typing 'b'/'back' raises ``GoBack``.
        validate_fn: Optional ``(value) -> str | None``.  When provided,
            called after type casting.  Return an error message string to
            reject the value (shown to the user, then re-prompt), or
            ``None`` to accept.

    Returns:
        The user's input, cast to ``type_fn``.

    Raises:
        GoBack: When ``allow_back`` is True and user types 'b'/'back'.
    """
    if choices:
        choice_str = f" ({'/'.join(str(c) for c in choices)})"
    else:
        choice_str = ""

    if is_rich_active() and console is not None:
        from rich.prompt import Prompt, IntPrompt, FloatPrompt

        prompt_cls = Prompt
        if type_fn is int:
            prompt_cls = IntPrompt
        elif type_fn is float:
            prompt_cls = FloatPrompt

        while True:
            if allow_back:
                # Use raw console.input so we can intercept 'b'/'back'
                # Escape default value so paths with brackets aren't
                # misinterpreted as Rich markup tags.
                default_str = f" [magenta]\\[{_esc(default)}][/]" if default is not None else ""
                raw = console.input(f"  {label}{choice_str}{default_str}: ").strip()
                if _is_back(raw):
                    raise GoBack()
                if not raw and default is not None:
                    return default
                if not raw and required:
                    console.print("  [red]This field is required[/]")
                    continue
                try:
                    val = type_fn(raw)
                except (ValueError, TypeError):
                    console.print(f"  [red]Invalid input, expected {type_fn.__name__}[/]")
                    continue
                if choices and str(val) not in [str(c) for c in choices]:
                    console.print(f"  [red]Must be one of: {', '.join(str(c) for c in choices)}[/]")
                    continue
                if validate_fn is not None:
                    err = validate_fn(val)
                    if err:
                        console.print(f"  [red]{_esc(err)}[/]")
                        continue
                return val
            else:
                result = prompt_cls.ask(
                    f"  {label}{choice_str}",
                    default=default if default is not None else ...,
                    console=console,
                )
                if result is ...:
                    if required:
                        console.print("  [red]This field is required[/]")
                        continue
                    return None
                if required and not str(result).strip():
                    console.print("  [red]This field is required[/]")
                    continue
                if choices and str(result) not in [str(c) for c in choices]:
                    console.print(f"  [red]Must be one of: {', '.join(str(c) for c in choices)}[/]")
                    continue
                final = type_fn(result) if not isinstance(result, type_fn) else result
                if validate_fn is not None:
                    err = validate_fn(final)
                    if err:
                        console.print(f"  [red]{_esc(err)}[/]")
                        continue
                return final
    else:
        default_str = f" [{default}]" if default is not None else ""
        while True:
            raw = input(f"  {label}{choice_str}{default_str}: ").strip()
            if allow_back and _is_back(raw):
                raise GoBack()
            if not raw and default is not None:
                return default
            if not raw and required:
                print("  This field is required")
                continue
            try:
                val = type_fn(raw)
                if choices and str(val) not in [str(c) for c in choices]:
                    print(f"  Must be one of: {', '.join(str(c) for c in choices)}")
                    continue
                if validate_fn is not None:
                    err = validate_fn(val)
                    if err:
                        print(f"  {err}")
                        continue
                return val
            except (ValueError, TypeError):
                print(f"  Invalid input, expected {type_fn.__name__}")


def _check_path_writable(p: Path, *, for_directory: bool = True) -> Optional[str]:
    """Check that a path is writable. Returns None if OK, error message otherwise.

    For directory paths: ensure dir (or parent if creating) exists and is writable.
    Probes by creating a temp file then deleting it.
    Returns error string for PermissionError/OSError. Returns None for other
    exceptions (assume OK) to avoid blocking users on quirky filesystems.
    """
    target = p if for_directory else p.parent
    target = target.resolve()
    try:
        target.mkdir(parents=True, exist_ok=True)
        probe = target / ".sidestep_probe"
        probe.write_text("", encoding="utf-8")
        try:
            probe.unlink(missing_ok=True)
        except OSError:
            # Write succeeded; unlink may fail on Windows (antivirus lock).
            # Writability is proven, so treat as OK.
            logging.getLogger(__name__).debug(
                "Could not remove probe file %s (antivirus lock?); assuming writable",
                probe,
            )
        return None
    except PermissionError:
        return f"Permission denied: cannot write to {target}"
    except OSError as e:
        return f"Cannot write to {target}: {e}"
    except Exception as e:
        logging.getLogger(__name__).debug(
            "Path writability check failed (assuming OK): %s", e
        )
        return None  # Assume OK on unknown errors (e.g. network drives)


def ask_output_path(
    label: str,
    default: Optional[str] = None,
    required: bool = True,
    allow_back: bool = True,
    *,
    for_file: bool = False,
) -> str:
    """Ask for an output path (directory or file), validating writability.

    Ensures parent directories can be created and the target location is writable.
    For file paths (e.g. estimate JSON), checks the parent directory.

    Args:
        label: Prompt text.
        default: Default value.
        required: If True, empty input is rejected.
        allow_back: If True, typing 'b'/'back' raises GoBack.
        for_file: If True, treat path as a file (validate parent dir writability).

    Raises:
        GoBack: When allow_back is True and user types 'b'/'back'.
    """
    while True:
        val = ask(label, default=default, required=required, allow_back=allow_back)
        if val is None or (isinstance(val, str) and not val.strip()):
            return "" if not required else (val if val is not None else "")
        p = Path(val).expanduser()
        err = _check_path_writable(p, for_directory=not for_file)
        if err is None:
            try:
                return str(p.resolve(strict=False))
            except Exception:
                return str(p)
        if is_rich_active() and console is not None:
            console.print(f"  [red]{_esc(err)}[/]")
            console.print("  [dim]Choose a different path or ensure you have write permissions.[/]")
        else:
            print(f"  {err}")
            print("  Choose a different path or ensure you have write permissions.")


def ask_path(
    label: str,
    default: Optional[str] = None,
    must_exist: bool = False,
    allow_back: bool = True,
) -> str:
    """Ask for a filesystem path, optionally validating existence.

    Raises:
        GoBack: When ``allow_back`` is True and user types 'b'/'back'.
    """
    while True:
        val = ask(label, default=default, required=True, allow_back=allow_back)
        p = Path(val).expanduser()
        if must_exist:
            try:
                exists = p.exists()
            except PermissionError:
                if is_rich_active() and console is not None:
                    console.print(f"  [red]Permission denied accessing: {_esc(p)}[/]")
                    console.print("  [dim]Check permissions or choose a different path.[/]")
                else:
                    print(f"  Permission denied accessing: {p}")
                    print("  Check permissions or choose a different path.")
                continue
            if not exists:
                if is_rich_active() and console is not None:
                    console.print(f"  [red]Path not found: {_esc(p)}[/]")
                    console.print("  [dim]Use an absolute path or verify the folder exists.[/]")
                else:
                    print(f"  Path not found: {p}")
                    print("  Use an absolute path or verify the folder exists.")
                continue
            try:
                p.stat()
            except PermissionError:
                if is_rich_active() and console is not None:
                    console.print(f"  [red]Permission denied: {_esc(p)}[/]")
                    console.print("  [dim]Check permissions or choose a different path.[/]")
                else:
                    print(f"  Permission denied: {p}")
                    print("  Check permissions or choose a different path.")
                continue
        try:
            return str(p.resolve(strict=False))
        except Exception:
            # Keep user input if resolution fails for any reason.
            return str(p)


def ask_bool(label: str, default: bool = True, allow_back: bool = True) -> bool:
    """Ask for a yes/no boolean value.

    Accepts ``y``, ``yes``, ``n``, ``no`` (case-insensitive).
    Empty input returns *default*.

    Raises:
        GoBack: When ``allow_back`` is True and user types 'b'/'back'.
    """
    default_hint = "Y/n" if default else "y/N"

    if is_rich_active() and console is not None:
        prompt_label = f"{label} [magenta]\\[{default_hint}][/]"
        while True:
            raw = console.input(f"  {prompt_label}: ").strip()
            if allow_back and _is_back(raw):
                raise GoBack()
            if not raw:
                return default
            low = raw.lower()
            if low in ("y", "yes"):
                return True
            if low in ("n", "no"):
                return False
            console.print("  [red]Please answer y or n[/]")
    else:
        prompt_label = f"{label} [{default_hint}]"
        while True:
            raw = input(f"  {prompt_label}: ").strip()
            if allow_back and _is_back(raw):
                raise GoBack()
            if not raw:
                return default
            low = raw.lower()
            if low in ("y", "yes"):
                return True
            if low in ("n", "no"):
                return False
            print("  Please answer y or n")


def section(title: str) -> None:
    """Print a section header."""
    if is_rich_active() and console is not None:
        console.print(f"\n  [bold cyan]--- {title} ---[/]\n")
    else:
        print(f"\n  --- {title} ---\n")
