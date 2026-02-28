"""OFT adapter wizard step (experimental).

Collects OFT hyperparameters (block size, constrained OFT, epsilon).
OFT constrains weight updates to orthogonal transformations.
"""

from __future__ import annotations

from sidestep_engine.ui.prompt_helpers import ask, ask_bool, print_message, section


def step_oft(a: dict) -> None:
    """OFT hyperparameters (PEFT Orthogonal Fine-Tuning).

    Marked experimental.  In basic mode, coft/eps are auto-defaulted.
    """
    from sidestep_engine.ui.flows.train_steps_adapter import (
        _ask_attention_type,
        _ask_projections,
    )

    section("OFT Settings [Experimental] (press Enter for defaults)")
    print_message(
        "EXPERIMENTAL -- Orthogonal fine-tuning. Preserves pretrained knowledge "
        "by constraining updates to rotations. Untested on audio DiT.",
        kind="warn",
    )
    print_message(
        "Block size controls the granularity of orthogonal blocks. "
        "Smaller = more expressive but more parameters. "
        "Constrained OFT (COFT) projects updates back to the Cayley manifold "
        "for stricter orthogonality.",
        kind="dim",
    )

    a["oft_block_size"] = ask(
        "Block size",
        default=a.get("oft_block_size", 64),
        type_fn=int,
        allow_back=True,
    )

    if a.get("config_mode") == "basic":
        a.setdefault("oft_coft", False)
        a.setdefault("oft_eps", 6e-5)
        a.setdefault("attention_type", "both")
        a.setdefault("target_mlp", True)
        return

    a["oft_coft"] = ask_bool(
        "Enable constrained OFT (Cayley projection)?",
        default=a.get("oft_coft", False),
        allow_back=True,
    )
    a["oft_eps"] = ask(
        "Epsilon (numerical stability)",
        default=a.get("oft_eps", 6e-5),
        type_fn=float,
        allow_back=True,
    )

    _ask_attention_type(a)
    _ask_projections(a)
    a["target_mlp"] = ask_bool(
        "Also target MLP/FFN layers (gate_proj, up_proj, down_proj)?",
        default=a.get("target_mlp", True),
        allow_back=True,
    )
