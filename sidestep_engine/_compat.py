"""
ACE-Step Compatibility Check for Side-Step.

Side-Step bundles vendored copies of ACE-Step utilities for its corrected
(fixed) training loop, so a full ACE-Step installation is no longer required.

This module checks that critical vendored modules are importable.
"""

from __future__ import annotations

import logging
import os

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Version pin
# ---------------------------------------------------------------------------

TESTED_ACESTEP_COMMIT = "46116a6"
"""Short SHA of the upstream ``ace-step/ACE-Step-1.5`` commit that the
vendored files were last synced from."""

SIDESTEP_VERSION = "1.0.0-beta"
"""Current Side-Step release string."""


# ---------------------------------------------------------------------------
# Compatibility check
# ---------------------------------------------------------------------------

def check_compatibility() -> None:
    """Verify that critical symbols exist.

    Checks vendored modules (required). Non-fatal: prints warnings and
    continues.
    """
    warnings: list[str] = []

    # 1. Vendored modules (required for fixed training)
    try:
        from sidestep_engine.vendor.data_module import PreprocessedDataModule  # noqa: F401
    except Exception as e:
        warnings.append(
            f"Cannot import vendored data_module.PreprocessedDataModule: {e}"
        )

    try:
        from sidestep_engine.vendor.lora_utils import inject_lora_into_dit  # noqa: F401
    except Exception as e:
        warnings.append(
            f"Cannot import vendored lora_utils.inject_lora_into_dit: {e}"
        )

    try:
        from sidestep_engine.vendor.configs import TrainingConfig  # noqa: F401
    except Exception as e:
        warnings.append(
            f"Cannot import vendored configs.TrainingConfig: {e}"
        )

    if warnings:
        msg = (
            f"[Side-Step] Compatibility warning (vendored from ACE-Step "
            f"commit {TESTED_ACESTEP_COMMIT}):\n"
        )
        for w in warnings:
            msg += f"  - {w}\n"
        msg += (
            "  Side-Step's corrected training may not work.\n"
            "  Try reinstalling Side-Step or check for missing files."
        )
        logger.warning(msg)
        print(f"\n{msg}\n")
    else:
        logger.debug(
            "[Side-Step] Compatibility check passed (pin: %s)",
            TESTED_ACESTEP_COMMIT,
        )


# ---------------------------------------------------------------------------
# Warning filters
# ---------------------------------------------------------------------------

_TORCHAO_CPP_WARN_SNIPPET = "Skipping import of cpp extensions due to incompatible torch version"


_torchao_filter_installed = False


def install_torchao_warning_filter() -> None:
    """Suppress one known non-fatal torchao compatibility warning.

    Call early in every entry point (``train.py``, ``sidestep_tui.py``)
    so the user never sees irrelevant torchao noise.  Idempotent.
    """
    global _torchao_filter_installed
    if _torchao_filter_installed:
        return
    if os.getenv("SIDESTEP_DISABLE_TORCHAO_WARN_FILTER", "").strip().lower() in {
        "1", "true", "yes", "on",
    }:
        return

    def _drop_only_known_torchao_warning(record: logging.LogRecord) -> bool:
        if not record.name.startswith("torchao"):
            return True
        try:
            msg = record.getMessage()
        except Exception:
            msg = str(record.msg)
        return _TORCHAO_CPP_WARN_SNIPPET not in msg

    root_logger = logging.getLogger()
    root_logger.addFilter(_drop_only_known_torchao_warning)
    logging.getLogger("torchao").addFilter(_drop_only_known_torchao_warning)
    _torchao_filter_installed = True
