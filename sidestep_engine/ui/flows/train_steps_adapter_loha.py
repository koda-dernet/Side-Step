"""LoHA adapter wizard step.

Collects LoHA hyperparameters (linear dim, alpha, factor, Tucker, scalar).
LoHA uses Hadamard products for richer parameter interactions per rank.
"""

from __future__ import annotations

from sidestep_engine.ui.prompt_helpers import ask, ask_bool, print_message, section


def step_loha(a: dict) -> None:
    """LoHA hyperparameters (LyCORIS Hadamard adapter).

    In basic mode, Tucker/scalar toggles, attention targeting, and
    projection selection are skipped (sensible defaults applied).
    """
    from sidestep_engine.ui.flows.train_steps_adapter import (
        _ask_attention_type,
        _ask_projections,
    )

    section("LoHA Settings (press Enter for defaults)")
    print_message(
        "Hadamard-product adapter. Richer parameter interactions per rank "
        "-- best for large, diverse datasets.",
        kind="dim",
    )
    a["loha_linear_dim"] = ask("Linear dimension", default=a.get("loha_linear_dim", 64), type_fn=int, allow_back=True)
    a["loha_linear_alpha"] = ask("Linear alpha", default=a.get("loha_linear_alpha", 128), type_fn=int, allow_back=True)
    a["loha_factor"] = ask("Factor (-1 = auto)", default=a.get("loha_factor", -1), type_fn=int, allow_back=True)

    if a.get("config_mode") == "basic":
        a.setdefault("loha_use_tucker", False)
        a.setdefault("loha_use_scalar", False)
        a.setdefault("attention_type", "both")
        a.setdefault("target_mlp", True)
        return

    a["loha_use_tucker"] = ask_bool(
        "Use Tucker decomposition?",
        default=a.get("loha_use_tucker", False),
        allow_back=True,
    )
    a["loha_use_scalar"] = ask_bool(
        "Use scalar scaling?",
        default=a.get("loha_use_scalar", False),
        allow_back=True,
    )

    _ask_attention_type(a)
    _ask_projections(a)
    a["target_mlp"] = ask_bool(
        "Also target MLP/FFN layers (gate_proj, up_proj, down_proj)?",
        default=a.get("target_mlp", True),
        allow_back=True,
    )
