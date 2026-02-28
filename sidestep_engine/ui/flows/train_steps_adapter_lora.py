"""LoRA adapter wizard step.

Collects LoRA hyperparameters (rank, alpha, dropout, targeting).
When a Preprocessing++ map is detected, rank/alpha/targets are locked.
"""

from __future__ import annotations

from sidestep_engine.ui.flows.train_steps_required import _has_fisher_map
from sidestep_engine.ui.prompt_helpers import ask, ask_bool, print_message, section


def step_lora(a: dict) -> None:
    """LoRA hyperparameters.

    When a Preprocessing++ map is detected in the dataset directory, rank / alpha /
    target-module questions are skipped because the map will
    override them in ``build_configs``.

    In basic mode, dropout, attention targeting, and projection selection
    are auto-defaulted (both attention types, all projections + MLP).
    """
    from sidestep_engine.ui.flows.train_steps_adapter import (
        _ask_attention_type,
        _ask_projections,
    )

    _is_basic = a.get("config_mode") == "basic"

    if _has_fisher_map(a):
        section("LoRA Settings (Preprocessing++ guided -- rank & targets locked)")
        print_message(
            "Standard low-rank adapter. Balanced quality and efficiency. PP++ compatible.",
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

    section("LoRA Settings (press Enter for defaults)")
    print_message(
        "Standard low-rank adapter. Balanced quality and efficiency. PP++ compatible.",
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
