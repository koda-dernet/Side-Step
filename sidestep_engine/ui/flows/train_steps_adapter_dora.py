"""DoRA adapter wizard step.

Collects DoRA hyperparameters — same as LoRA but with ``use_dora=True``.
DoRA decomposes weight updates into magnitude and direction components.
PP++ compatible.
"""

from __future__ import annotations

from sidestep_engine.ui.flows.train_steps_required import _has_fisher_map
from sidestep_engine.ui.prompt_helpers import ask, ask_bool, print_message, section


def step_dora(a: dict) -> None:
    """DoRA hyperparameters.

    Mirrors LoRA step but always sets ``use_dora=True``.  When a
    Preprocessing++ map is detected, rank/alpha/targets are locked.
    """
    from sidestep_engine.ui.flows.train_steps_adapter import (
        _ask_attention_type,
        _ask_projections,
    )

    a["use_dora"] = True
    _is_basic = a.get("config_mode") == "basic"

    if _has_fisher_map(a):
        section("DoRA Settings (Preprocessing++ guided -- rank & targets locked)")
        print_message(
            "LoRA with learned magnitude scaling. Better fine-detail learning, "
            "minimal overhead. PP++ compatible.",
            kind="dim",
        )
        if not _is_basic:
            a["dropout"] = ask(
                "Dropout", default=a.get("dropout", 0.1),
                type_fn=float, allow_back=True,
            )
        else:
            a.setdefault("dropout", 0.1)
        return

    section("DoRA Settings (press Enter for defaults)")
    print_message(
        "LoRA with learned magnitude scaling. Better fine-detail learning, "
        "minimal overhead. PP++ compatible.",
        kind="dim",
    )
    a["rank"] = ask("Rank", default=a.get("rank", 64), type_fn=int, allow_back=True)
    a["alpha"] = ask("Alpha", default=a.get("alpha", 128), type_fn=int, allow_back=True)

    if _is_basic:
        a.setdefault("dropout", 0.1)
        a.setdefault("attention_type", "both")
        a.setdefault("target_mlp", True)
        return

    a["dropout"] = ask("Dropout", default=a.get("dropout", 0.1), type_fn=float, allow_back=True)
    _ask_attention_type(a)
    _ask_projections(a)
    a["target_mlp"] = ask_bool(
        "Also target MLP/FFN layers (gate_proj, up_proj, down_proj)?",
        default=a.get("target_mlp", True),
        allow_back=True,
    )
