"""HuggingFace ``trust_remote_code`` import preparation for ACE-Step checkpoints.

Checkpoint directories ship thin Python stubs that import ``acestep.models.common``
(and sometimes ``acestep.models.flow_matching_solvers``). Side-Step prepends a
suitable ``sys.path`` entry so those imports resolve:

1. ``ACESTEP_SRC`` (full ACE-Step checkout)
2. ``../ACE-Step-1.5`` next to the Side-Step repo
3. ``<Side-Step>/vendor/ACE-Step-1.5``
4. A minimal bundled tree under ``bundled_acestep/`` (snapshot of upstream
   ``acestep/models/common``; see ``bundled_acestep/BUNDLED_ACESTEP_SOURCE.txt``)

A candidate is accepted only if
``acestep/models/common/configuration_acestep_v15.py`` exists, so a hollow
``acestep`` package cannot shadow the bundled fallback.

When refreshing bundled files, copy from ACE-Step 1.5 ``acestep/models/common/``
and update ``BUNDLED_ACESTEP_SOURCE.txt``.
"""

from __future__ import annotations

import importlib.machinery
import importlib.util
import logging
import os
import sys
import types
from pathlib import Path

logger = logging.getLogger(__name__)

_COMMON_CONFIG_REL = Path("acestep") / "models" / "common" / "configuration_acestep_v15.py"


def _root_provides_acestep_common(root: Path) -> bool:
    """True when *root* is a sys.path prefix that can satisfy HF common imports."""
    return (root / _COMMON_CONFIG_REL).is_file()


def bundled_acestep_root() -> Path:
    """Directory prepended for the bundled minimal ``acestep`` tree (for tests)."""
    return Path(__file__).resolve().parent / "bundled_acestep"


def _prepend_acestep_src_paths() -> None:
    """Prepend the first matching ACE-Step tree so ``import acestep.models.common`` works."""
    candidates: list[Path] = []
    env = os.environ.get("ACESTEP_SRC")
    if env:
        candidates.append(Path(env).expanduser().resolve())
    # sidestep_engine/models/acestep_remote_imports.py -> parents[2] == Side-Step repo root
    side_step_root = Path(__file__).resolve().parents[2]
    candidates.append((side_step_root.parent / "ACE-Step-1.5").resolve())
    candidates.append((side_step_root / "vendor" / "ACE-Step-1.5").resolve())
    candidates.append(bundled_acestep_root().resolve())

    for root in candidates:
        if _root_provides_acestep_common(root):
            s = str(root)
            if s not in sys.path:
                sys.path.insert(0, s)
            logger.debug("Prepended ACE-Step import path: %s", root)
            return


def _drop_hollow_acestep_modules() -> None:
    """Remove placeholder ``acestep`` entries so a prepended path can load real packages."""
    for name in ("acestep.models", "acestep"):
        mod = sys.modules.get(name)
        if mod is not None and getattr(mod, "__path__", None) is None:
            del sys.modules[name]


def _safe_find_spec(name: str):
    """Like :func:`importlib.util.find_spec` but never raises on broken stubs.

    A module sitting in ``sys.modules`` with ``__spec__ is None`` (e.g. an older
    manual shim) makes :func:`importlib.util.find_spec` raise ``ValueError``.
    """
    try:
        return importlib.util.find_spec(name)
    except (ValueError, ImportError, ModuleNotFoundError):
        return None


def _ensure_acestep_remote_imports() -> None:
    """Prepare ``acestep.models.*`` for HuggingFace ``trust_remote_code`` snapshots.

    Some checkpoints import ``acestep.models.flow_matching_solvers``.  An older
    Side-Step workaround registered ``acestep.models`` as a plain
    :class:`types.ModuleType` **without** :attr:`__path__`, which makes
    ``acestep.models`` a non-package and breaks ``from acestep.models.X``.

    We first try to load the real upstream package via :func:`_prepend_acestep_src_paths`,
    then register namespace-package stubs only when imports are still missing, and
    finally add a minimal ``flow_matching_solvers`` shim if that submodule is absent.
    """
    _prepend_acestep_src_paths()
    _drop_hollow_acestep_modules()

    if _safe_find_spec("acestep") is None:
        if "acestep" not in sys.modules:
            stub = types.ModuleType("acestep")
            stub.__path__ = []
            sys.modules["acestep"] = stub
    else:
        ace = sys.modules.get("acestep")
        if ace is not None and not getattr(ace, "__path__", None):
            ace.__path__ = []

    if _safe_find_spec("acestep.models") is None:
        if "acestep.models" not in sys.modules:
            mp = types.ModuleType("acestep.models")
            mp.__path__ = []
            sys.modules["acestep.models"] = mp
    else:
        mp = sys.modules.get("acestep.models")
        if mp is not None and not getattr(mp, "__path__", None):
            mp.__path__ = []

    fms_name = "acestep.models.flow_matching_solvers"
    # Older shims registered this name without __spec__, which breaks find_spec.
    _fms_existing = sys.modules.get(fms_name)
    if _fms_existing is not None and getattr(_fms_existing, "__spec__", None) is None:
        _fms_existing.__spec__ = importlib.machinery.ModuleSpec(
            fms_name, loader=None, origin="side-step-shim"
        )
        _fms_existing.__loader__ = None
        if not hasattr(_fms_existing, "SOLVER_REGISTRY"):
            _fms_existing.SOLVER_REGISTRY = {}
        if not hasattr(_fms_existing, "VALID_INFER_METHODS"):
            _fms_existing.VALID_INFER_METHODS = frozenset()

    if _safe_find_spec(fms_name) is None:
        fms = types.ModuleType(fms_name)
        fms.__spec__ = importlib.machinery.ModuleSpec(
            fms_name, loader=None, origin="side-step-shim"
        )
        fms.__loader__ = None
        fms.SOLVER_REGISTRY = {}
        fms.VALID_INFER_METHODS = frozenset()
        sys.modules[fms_name] = fms
        logger.debug(
            "Using minimal acestep.models.flow_matching_solvers shim (no upstream "
            "module on sys.path). Set ACESTEP_SRC to a full ACE-Step tree if you "
            "need non-default flow solvers."
        )
