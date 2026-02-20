"""
Path validation and target-module resolution for ACE-Step Training V2 CLI.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from acestep.training_v2.cli.args import VARIANT_DIR_MAP


def validate_paths(args: argparse.Namespace) -> bool:
    """Validate that required paths exist and attach the resolved model dir.

    On success, sets ``args.model_dir`` to the resolved ``Path`` so that
    callers can consume it directly.  Returns ``True`` if all OK.
    Prints ``[FAIL]`` messages and returns False on the first error.
    """
    sub = args.subcommand

    if sub == "compare-configs":
        for p in args.configs:
            if not Path(p).is_file():
                print(f"[FAIL] Config file not found: {p}", file=sys.stderr)
                return False
        return True

    # All other subcommands need checkpoint-dir
    ckpt_root = Path(args.checkpoint_dir)
    if not ckpt_root.is_dir():
        print(f"[FAIL] Checkpoint directory not found: {ckpt_root}", file=sys.stderr)
        return False

    # Resolve model directory: try known alias first, then literal folder name
    variant_dir = VARIANT_DIR_MAP.get(args.model_variant)
    if variant_dir and (ckpt_root / variant_dir).is_dir():
        model_dir = ckpt_root / variant_dir
    elif (ckpt_root / args.model_variant).is_dir():
        model_dir = ckpt_root / args.model_variant
    else:
        tried = variant_dir or args.model_variant
        available = sorted([p.name for p in ckpt_root.iterdir() if p.is_dir()]) if ckpt_root.is_dir() else []
        sample = ", ".join(available[:8]) if available else "(none found)"
        print(
            f"[FAIL] Model directory not found: {ckpt_root / tried}\n"
            f"       Looked for '{tried}' under {ckpt_root}\n"
            f"       Available folders: {sample}\n"
            f"       Tip: use --model-variant with the exact folder name (e.g. turbo/base/sft or your fine-tune folder).",
            file=sys.stderr,
        )
        return False

    # Attach resolved path so callers can use it directly
    args.model_dir = model_dir

    # Dataset dir
    ds_dir = getattr(args, "dataset_dir", None)
    if ds_dir is not None and not Path(ds_dir).is_dir():
        print(
            f"[FAIL] Dataset directory not found: {ds_dir}\n"
            f"       Tip: it must contain preprocessed .pt files (or a valid manifest.json).",
            file=sys.stderr,
        )
        return False

    # Resume path: fail if explicitly set but missing
    resume = getattr(args, "resume_from", None)
    if resume is not None and str(resume).strip() and not Path(str(resume)).exists():
        print(
            f"[FAIL] Resume path not found: {resume}\n"
            f"       Fix the path or leave empty to train from scratch.",
            file=sys.stderr,
        )
        return False

    return True


def _prefix_modules(modules: list, prefix: str) -> list:
    """Add *prefix* to each module name that is not already fully qualified."""
    return [m if "." in m else f"{prefix}.{m}" for m in modules]


def resolve_target_modules(
    target_modules: list,
    attention_type: str,
    *,
    self_target_modules: list | None = None,
    cross_target_modules: list | None = None,
    target_mlp: bool = False,
) -> list:
    """Resolve target modules based on attention type selection.

    Args:
        target_modules: Fallback list of module patterns (e.g. ["q_proj"]).
        attention_type: One of "self", "cross", or "both".
        self_target_modules: Per-type projections for self-attention
            (only used when *attention_type* is "both").
        cross_target_modules: Per-type projections for cross-attention
            (only used when *attention_type* is "both").
        target_mlp: When True, append MLP/FFN module names
            (gate_proj, up_proj, down_proj) to the resolved list.

    Returns:
        Resolved list of module patterns with appropriate prefixes.

    When *attention_type* is "both" and per-type lists are provided, each
    set is prefixed independently and merged.  If neither per-type list
    is provided, *target_modules* is returned unchanged (PEFT matches all
    occurrences).

    Examples:
        resolve_target_modules(["q_proj", "v_proj"], "both")
        -> ["q_proj", "v_proj"]  # unchanged, PEFT matches all

        resolve_target_modules(["q_proj"], "both",
            self_target_modules=["q_proj", "v_proj"],
            cross_target_modules=["q_proj"])
        -> ["self_attn.q_proj", "self_attn.v_proj", "cross_attn.q_proj"]

        resolve_target_modules(["q_proj", "v_proj"], "self")
        -> ["self_attn.q_proj", "self_attn.v_proj"]

        resolve_target_modules(["q_proj", "v_proj"], "cross")
        -> ["cross_attn.q_proj", "cross_attn.v_proj"]

        resolve_target_modules(["q_proj"], "self", target_mlp=True)
        -> ["self_attn.q_proj", "gate_proj", "up_proj", "down_proj"]
    """
    if attention_type == "both":
        if self_target_modules is not None or cross_target_modules is not None:
            s_mods = self_target_modules if self_target_modules is not None else target_modules
            c_mods = cross_target_modules if cross_target_modules is not None else target_modules
            resolved = _prefix_modules(s_mods, "self_attn") + _prefix_modules(c_mods, "cross_attn")
        else:
            resolved = list(target_modules)
    else:
        prefix_map = {
            "self": "self_attn",
            "cross": "cross_attn",
        }
        prefix = prefix_map.get(attention_type)
        if prefix is None:
            resolved = list(target_modules)
        else:
            resolved = _prefix_modules(target_modules, prefix)

    if target_mlp:
        mlp_modules = ["gate_proj", "up_proj", "down_proj"]
        existing = set(resolved)
        for m in mlp_modules:
            if m not in existing:
                resolved.append(m)

    return resolved

