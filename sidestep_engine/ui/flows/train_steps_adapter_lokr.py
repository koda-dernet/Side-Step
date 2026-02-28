"""LoKR adapter wizard step.

Collects LoKR hyperparameters (linear dim, alpha, factor, decomposition toggles).
"""

from __future__ import annotations

from sidestep_engine.ui.prompt_helpers import ask, ask_bool, print_message, section


def step_lokr(a: dict) -> None:
    """LoKR hyperparameters (LyCORIS Kronecker adapter).

    In basic mode, decomposition toggles, attention targeting, and
    projection selection are skipped (sensible defaults applied).
    """
    from sidestep_engine.ui.flows.train_steps_adapter import (
        _ask_attention_type,
        _ask_projections,
    )

    section("LoKR Settings (press Enter for defaults)")
    print_message(
        "Kronecker-product adapter. More parameter-efficient than LoRA at same rank.",
        kind="dim",
    )
    a["lokr_linear_dim"] = ask("Linear dimension", default=a.get("lokr_linear_dim", 64), type_fn=int, allow_back=True)
    a["lokr_linear_alpha"] = ask("Linear alpha", default=a.get("lokr_linear_alpha", 128), type_fn=int, allow_back=True)
    if not a.get("lokr_decompose_both", False):
        print_message(
            "Note: alpha only affects scaling when 'decompose both' is enabled. "
            "With default settings, LyCORIS sets scale=1 (alpha=dim).",
            kind="dim",
        )
    a["lokr_factor"] = ask("Factor (-1 = auto)", default=a.get("lokr_factor", -1), type_fn=int, allow_back=True)

    if a.get("config_mode") == "basic":
        a.setdefault("lokr_decompose_both", False)
        a.setdefault("lokr_use_tucker", False)
        a.setdefault("lokr_use_scalar", False)
        a.setdefault("lokr_weight_decompose", False)
        a.setdefault("attention_type", "both")
        a.setdefault("target_mlp", True)
        return

    a["lokr_decompose_both"] = ask_bool(
        "Decompose both Kronecker factors?",
        default=a.get("lokr_decompose_both", False),
        allow_back=True,
    )
    a["lokr_use_tucker"] = ask_bool(
        "Use Tucker decomposition?",
        default=a.get("lokr_use_tucker", False),
        allow_back=True,
    )
    a["lokr_use_scalar"] = ask_bool(
        "Use scalar scaling?",
        default=a.get("lokr_use_scalar", False),
        allow_back=True,
    )
    a["lokr_weight_decompose"] = ask_bool(
        "Enable DoRA-style weight decomposition?",
        default=a.get("lokr_weight_decompose", False),
        allow_back=True,
    )

    _ask_attention_type(a)
    _ask_projections(a)
    a["target_mlp"] = ask_bool(
        "Also target MLP/FFN layers (gate_proj, up_proj, down_proj)?",
        default=a.get("target_mlp", True),
        allow_back=True,
    )
