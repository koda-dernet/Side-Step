"""
Optional runtime dependency checks with prompt-before-install UX.

This module keeps optional features novice-friendly by detecting missing
packages early, explaining impact, and offering an explicit install prompt.
"""

from __future__ import annotations

import importlib.util
import logging
import shutil
import subprocess
import sys
from dataclasses import dataclass
from typing import Iterable

from sidestep_engine.ui.prompt_helpers import print_message

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class OptionalDependency:
    """Descriptor for one optional package requirement."""

    key: str
    module: str
    install_spec: str
    reason: str
    impact_if_missing: str


_BITSANDBYTES = OptionalDependency(
    key="bitsandbytes",
    module="bitsandbytes",
    install_spec="bitsandbytes>=0.45.0",
    reason="Optimizer `adamw8bit` was selected",
    impact_if_missing="Training falls back to AdamW (higher VRAM use).",
)

_PRODIGYOPT = OptionalDependency(
    key="prodigyopt",
    module="prodigyopt",
    install_spec="prodigyopt>=1.1.2",
    reason="Optimizer `prodigy` was selected",
    impact_if_missing="Training falls back to AdamW (no Prodigy adaptation).",
)

_TENSORBOARD = OptionalDependency(
    key="tensorboard",
    module="torch.utils.tensorboard",
    install_spec="tensorboard",
    reason="Training logging is enabled",
    impact_if_missing="TensorBoard logs are disabled (training still runs).",
)

_PYLOUDNORM = OptionalDependency(
    key="pyloudnorm",
    module="pyloudnorm",
    install_spec="pyloudnorm>=0.1.0",
    reason="LUFS normalization was selected",
    impact_if_missing="Preprocessing fails if LUFS is kept.",
)


def _has_module(module_name: str) -> bool:
    """Return True when a module import target exists."""
    return importlib.util.find_spec(module_name) is not None



def _ask_yes_no(prompt: str, default: bool = True) -> bool:
    """Simple y/n prompt without raising GoBack.

    Delegates to :func:`~sidestep_engine.ui.prompt_helpers.ask_bool`
    with ``allow_back=False``.
    """
    from sidestep_engine.ui.prompt_helpers import ask_bool
    return ask_bool(prompt, default=default, allow_back=False)


def _install_dep(dep: OptionalDependency) -> bool:
    """Attempt to install one dependency with uv preferred."""
    install_commands: list[list[str]] = []
    if shutil.which("uv"):
        install_commands.append(["uv", "pip", "install", dep.install_spec])
    install_commands.append([sys.executable, "-m", "pip", "install", dep.install_spec])

    for cmd in install_commands:
        try:
            print_message(f"Installing {dep.install_spec} via: {' '.join(cmd)}")
            proc = subprocess.run(cmd, check=False, capture_output=True, text=True)
        except Exception as exc:
            logger.warning("Dependency install command failed to start (%s): %s", cmd, exc)
            continue

        if proc.returncode == 0:
            print_message(f"Installed {dep.install_spec}.", kind="ok")
            return True

        logger.warning(
            "Dependency install failed (%s): exit=%s stderr=%s",
            " ".join(cmd),
            proc.returncode,
            (proc.stderr or "").strip()[:500],
        )

    print_message(f"Could not install {dep.install_spec} automatically.", kind="fail")
    print_message(f"Install manually with: pip install {dep.install_spec}", kind="warn")
    return False


def required_training_optionals(train_cfg) -> list[OptionalDependency]:
    """Return optional deps implied by the chosen training config."""
    deps: list[OptionalDependency] = []
    optimizer_type = str(getattr(train_cfg, "optimizer_type", "adamw")).lower()
    if optimizer_type == "adamw8bit":
        deps.append(_BITSANDBYTES)
    elif optimizer_type == "prodigy":
        deps.append(_PRODIGYOPT)

    log_every = int(getattr(train_cfg, "log_every", 0) or 0)
    log_heavy_every = int(getattr(train_cfg, "log_heavy_every", 0) or 0)
    if log_every > 0 or log_heavy_every > 0:
        deps.append(_TENSORBOARD)
    return deps


def required_preprocess_optionals(normalize: str) -> list[OptionalDependency]:
    """Return optional deps implied by preprocessing options."""
    if str(normalize).lower() == "lufs":
        return [_PYLOUDNORM]
    return []


def ensure_optional_dependencies(
    deps: Iterable[OptionalDependency],
    *,
    interactive: bool,
    allow_install_prompt: bool,
) -> list[OptionalDependency]:
    """Ensure optional deps are present; prompt-install when allowed.

    Returns missing deps that remain unresolved after optional install attempts.
    """
    unresolved: list[OptionalDependency] = []
    for dep in deps:
        if _has_module(dep.module):
            continue

        print_message(f"Missing optional dependency: {dep.key}", kind="warn")
        print_message(f"Reason: {dep.reason}", kind="warn")
        print_message(f"Impact: {dep.impact_if_missing}", kind="warn")

        if not allow_install_prompt:
            print_message("Auto-install prompt skipped in non-interactive mode.", kind="warn")
            unresolved.append(dep)
            continue

        if not interactive:
            print_message("Interactive prompt unavailable; continuing without install.", kind="warn")
            unresolved.append(dep)
            continue

        if _ask_yes_no(f"Install {dep.install_spec} now?", default=True):
            if _install_dep(dep) and _has_module(dep.module):
                continue
            unresolved.append(dep)
        else:
            unresolved.append(dep)
    return unresolved

