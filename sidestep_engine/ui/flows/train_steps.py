"""
Individual wizard steps for the training flow — facade module.

Re-exports all step functions from the split sub-modules so existing
``from .train_steps import step_required, ...`` imports continue to work.
"""

from __future__ import annotations

# Re-export from sub-modules (preserve stable import surface)
from sidestep_engine.ui.flows.train_steps_required import (  # noqa: F401
    _has_fisher_map,
    step_config_mode,
    step_required,
)
from sidestep_engine.ui.flows.train_steps_adapter import (  # noqa: F401
    step_lora,
    step_dora,
    step_lokr,
    step_loha,
    step_oft,
    ADAPTER_STEP_MAP,
    ADAPTER_LABEL_MAP,
)
from sidestep_engine.ui.flows.train_steps_helpers import (  # noqa: F401
    smart_save_best_default as _smart_save_best_default,
    warn_warmup_ratio as _warn_warmup_ratio,
)
from sidestep_engine.ui.flows.train_steps_training import (  # noqa: F401
    step_training,
    step_cfg,
)
from sidestep_engine.ui.flows.train_steps_logging import (  # noqa: F401
    step_logging,
    step_chunk_duration,
)
from sidestep_engine.ui.flows.train_steps_advanced import (  # noqa: F401
    step_advanced_device,
    step_advanced_optimizer,
    step_advanced_vram,
    step_advanced_training,
    step_advanced_dataloader,
    step_advanced_logging,
)
from sidestep_engine.ui.flows.train_steps_levers import (  # noqa: F401
    step_all_the_levers,
)

__all__ = [
    "_has_fisher_map",
    "step_config_mode",
    "step_required",
    "step_lora",
    "step_lokr",
    "step_training",
    "step_cfg",
    "step_logging",
    "step_chunk_duration",
    "step_advanced_device",
    "step_advanced_optimizer",
    "step_advanced_vram",
    "step_advanced_training",
    "step_advanced_dataloader",
    "step_advanced_logging",
    "step_all_the_levers",
]
