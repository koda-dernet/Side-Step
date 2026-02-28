"""
Adapter wizard steps — facade module.

Re-exports all per-adapter step functions and shared helpers so existing
``from .train_steps_adapter import step_lora, ...`` imports continue to work.
"""

from __future__ import annotations

from sidestep_engine.ui.flows.common import _DEFAULT_PROJECTIONS
from sidestep_engine.ui.prompt_helpers import ask, menu


def _ask_attention_type(a: dict) -> None:
    """Prompt for attention layer targeting."""
    a["attention_type"] = menu(
        "Which attention layers to target?",
        [
            ("both", "Both self-attention and cross-attention"),
            ("self", "Self-attention only (audio patterns)"),
            ("cross", "Cross-attention only (text conditioning)"),
        ],
        default=1,
        allow_back=True,
    )


def _ask_projections(a: dict) -> None:
    """Prompt for target projections, splitting by attention type when 'both'.

    When the user selects "both", asks separately for self-attention and
    cross-attention projections so they can be configured independently.
    When "self" or "cross", asks once as a single set.
    """
    if a.get("attention_type") == "both":
        a["self_target_modules_str"] = ask(
            "Self-attention projections",
            default=a.get("self_target_modules_str", _DEFAULT_PROJECTIONS),
            allow_back=True,
        )
        a["cross_target_modules_str"] = ask(
            "Cross-attention projections",
            default=a.get("cross_target_modules_str", _DEFAULT_PROJECTIONS),
            allow_back=True,
        )
    else:
        a["target_modules_str"] = ask(
            "Target projections",
            default=a.get("target_modules_str", _DEFAULT_PROJECTIONS),
            allow_back=True,
        )


# Re-export per-adapter steps (preserve stable import surface)
from sidestep_engine.ui.flows.train_steps_adapter_lora import step_lora  # noqa: F401,E402
from sidestep_engine.ui.flows.train_steps_adapter_dora import step_dora  # noqa: F401,E402
from sidestep_engine.ui.flows.train_steps_adapter_lokr import step_lokr  # noqa: F401,E402
from sidestep_engine.ui.flows.train_steps_adapter_loha import step_loha  # noqa: F401,E402
from sidestep_engine.ui.flows.train_steps_adapter_oft import step_oft  # noqa: F401,E402

# Map adapter_type string → step function for dispatch
ADAPTER_STEP_MAP = {
    "lora": step_lora,
    "dora": step_dora,
    "lokr": step_lokr,
    "loha": step_loha,
    "oft": step_oft,
}

ADAPTER_LABEL_MAP = {
    "lora": "LoRA Settings",
    "dora": "DoRA Settings",
    "lokr": "LoKR Settings",
    "loha": "LoHA Settings",
    "oft": "OFT Settings [Experimental]",
}
