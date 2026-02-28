"""
TensorBoard launch helpers for interactive training runs.

Provides:
    - A user-facing command hint that uses the uvx setuptools workaround.
    - A confirmation prompt (default yes) for auto-launch.
    - Best-effort background launch that never blocks training startup.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

from sidestep_engine.ui import console, is_rich_active
from sidestep_engine.ui.prompt_helpers import _esc

_SETUPTOOLS_CONSTRAINT = "setuptools<70"


def _normalize_log_dir(log_dir: str | Path) -> Path:
    """Return an expanded, absolute log directory path when possible."""
    p = Path(log_dir).expanduser()
    try:
        return p.resolve(strict=False)
    except Exception:
        return p


def tensorboard_manual_command(log_dir: str | Path) -> str:
    """Return the recommended manual TensorBoard command string."""
    resolved = _normalize_log_dir(log_dir)
    safe = str(resolved).replace('"', '\\"')
    return f'uvx --with "{_SETUPTOOLS_CONSTRAINT}" tensorboard --logdir "{safe}"'


def should_launch_tensorboard(
    log_dir: str | Path,
    *,
    default: bool = True,
    skip_prompt: bool = False,
    interactive: bool | None = None,
) -> bool:
    """Ask whether to auto-launch TensorBoard before training.

    Returns ``False`` in non-interactive contexts to avoid background
    processes in automation/CI environments.
    """
    if skip_prompt:
        return default
    if interactive is None:
        interactive = sys.stdin.isatty()
    if not interactive:
        return False

    display_dir = _normalize_log_dir(log_dir)
    prompt = f"Launch TensorBoard now? (logdir: {display_dir})"

    if is_rich_active() and console is not None:
        from rich.prompt import Confirm

        try:
            return Confirm.ask(
                f"[bold]{_esc(prompt)}[/]",
                default=default,
                console=console,
            )
        except (EOFError, KeyboardInterrupt):
            from sidestep_engine.ui.prompt_helpers import print_message
            print_message("Skipping TensorBoard auto-launch.", kind="dim")
            return False

    yn = "Y/n" if default else "y/N"
    try:
        answer = input(f"{prompt} [{yn}] ").strip().lower()
    except (EOFError, KeyboardInterrupt):
        print("\nSkipping TensorBoard auto-launch.", file=sys.stderr)
        return False

    if not answer:
        return default
    return answer in ("y", "yes")


def launch_tensorboard_background(
    log_dir: str | Path,
    *,
    host: str = "127.0.0.1",
    port: int = 6006,
) -> tuple[bool, str]:
    """Launch TensorBoard in a detached/background subprocess.

    Returns:
        (success, user_facing_message)
    """
    resolved = _normalize_log_dir(log_dir)
    try:
        resolved.mkdir(parents=True, exist_ok=True)
    except Exception:
        # TensorBoard can still create the directory itself in many cases.
        pass

    cmd = [
        "uvx",
        "--with",
        _SETUPTOOLS_CONSTRAINT,
        "tensorboard",
        "--logdir",
        str(resolved),
        "--host",
        host,
        "--port",
        str(port),
    ]

    kwargs: dict = {
        "stdin": subprocess.DEVNULL,
        "stdout": subprocess.DEVNULL,
        "stderr": subprocess.DEVNULL,
    }
    if sys.platform == "win32":
        flags = int(getattr(subprocess, "CREATE_NEW_PROCESS_GROUP", 0))
        flags |= int(getattr(subprocess, "DETACHED_PROCESS", 0))
        if flags:
            kwargs["creationflags"] = flags
    else:
        kwargs["start_new_session"] = True

    try:
        subprocess.Popen(cmd, **kwargs)
    except FileNotFoundError:
        return (
            False,
            "Could not auto-launch TensorBoard because 'uvx' was not found. "
            f"Run manually: {tensorboard_manual_command(resolved)}",
        )
    except Exception as exc:
        return (
            False,
            "Could not auto-launch TensorBoard "
            f"({exc}). Run manually: {tensorboard_manual_command(resolved)}",
        )

    return (
        True,
        f"TensorBoard launched in background at http://{host}:{port} (logdir: {resolved})",
    )
