"""
ACE-Step Compatibility Check for Side-Step.

Side-Step bundles vendored copies of ACE-Step utilities for its corrected
(fixed) training loop, so a full ACE-Step installation is no longer required
for that path.  Vanilla training mode still needs base ACE-Step.

This module checks that the vendored modules are importable and, optionally,
whether base ACE-Step is available for vanilla mode.
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

SIDESTEP_VERSION = "0.8.0-beta"
"""Current Side-Step release string."""


# ---------------------------------------------------------------------------
# Compatibility check
# ---------------------------------------------------------------------------

def check_compatibility() -> None:
    """Verify that critical symbols exist.

    Checks vendored modules (required) and base ACE-Step (optional, for
    vanilla mode).  Non-fatal: prints warnings and continues.
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

    # 2. Base ACE-Step (optional -- only needed for vanilla mode)
    vanilla_ok = True
    try:
        from acestep.training.trainer import LoRATrainer  # noqa: F401
    except ImportError:
        vanilla_ok = False

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
            "[Side-Step] Compatibility check passed (pin: %s, vanilla: %s)",
            TESTED_ACESTEP_COMMIT,
            "available" if vanilla_ok else "not available",
        )
        if not vanilla_ok:
            logger.info(
                "[Side-Step] Base ACE-Step not detected. Vanilla training mode "
                "will not be available. Corrected (fixed) mode works standalone."
            )
