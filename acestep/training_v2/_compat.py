"""
ACE-Step Compatibility Check for Side-Step.

Side-Step bundles vendored copies of ACE-Step utilities for its corrected
(fixed) training loop, so a full ACE-Step installation is no longer required.

This module checks that critical vendored modules are importable.
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Version pin
# ---------------------------------------------------------------------------

TESTED_ACESTEP_COMMIT = "46116a6"
"""Short SHA of the upstream ``ace-step/ACE-Step-1.5`` commit that the
vendored files were last synced from."""

SIDESTEP_VERSION = "0.9.0-beta"
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
        from acestep.training_v2._vendor.data_module import PreprocessedDataModule  # noqa: F401
    except ImportError:
        warnings.append(
            "Cannot import vendored data_module.PreprocessedDataModule"
        )

    try:
        from acestep.training_v2._vendor.lora_utils import inject_lora_into_dit  # noqa: F401
    except ImportError:
        warnings.append(
            "Cannot import vendored lora_utils.inject_lora_into_dit"
        )

    try:
        from acestep.training_v2._vendor.configs import TrainingConfig  # noqa: F401
    except ImportError:
        warnings.append(
            "Cannot import vendored configs.TrainingConfig"
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
