"""Module discovery for Fisher analysis.

Finds all LoRA-targetable projections in the ACE-Step decoder and
optionally groups them for chunked backward passes on low-VRAM cards.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Tuple

import torch.nn as nn

logger = logging.getLogger(__name__)

# Projection suffixes that are LoRA-targetable in the Audio DiT decoder.
_ATTN_PROJECTIONS = ("q_proj", "k_proj", "v_proj", "o_proj")
_MLP_PROJECTIONS = ("gate_proj", "up_proj", "down_proj")


def find_all_targetable_modules(
    model: nn.Module,
) -> List[Tuple[str, nn.Module]]:
    """Discover every LoRA-targetable projection in the decoder.

    Searches ``model.decoder`` (or *model* itself when no ``.decoder``
    attribute exists) for ``nn.Linear`` layers whose name ends with one
    of the known projection suffixes.

    Args:
        model: The full ACE-Step model (expects a ``.decoder`` attribute).

    Returns:
        Sorted list of ``(fully_qualified_name, module)`` pairs.
    """
    decoder = getattr(model, "decoder", model)
    prefix = "decoder." if hasattr(model, "decoder") else ""

    targets: List[Tuple[str, nn.Module]] = []
    for name, mod in decoder.named_modules():
        if not isinstance(mod, nn.Linear):
            continue
        short = name.rsplit(".", 1)[-1] if "." in name else name
        if short in _ATTN_PROJECTIONS or short in _MLP_PROJECTIONS:
            targets.append((f"{prefix}{name}", mod))

    targets.sort(key=lambda t: t[0])
    logger.debug("Found %d targetable modules", len(targets))
    return targets


def build_param_to_module_map(
    model: nn.Module,
    target_modules: List[Tuple[str, nn.Module]],
) -> Dict[str, str]:
    """Map parameter names to their parent module name.

    Uses identity comparison on the parameter tensor objects for O(n)
    matching instead of nested string comparisons.

    Args:
        model: The full model (used only for ``named_parameters``).
        target_modules: Output of :func:`find_all_targetable_modules`.

    Returns:
        ``{param_name: module_name}`` for every parameter belonging to
        a targetable module.
    """
    # Build id(tensor) -> module_name for all target module parameters
    id_to_mod: Dict[int, str] = {}
    for mod_name, mod in target_modules:
        for _pname, param in mod.named_parameters():
            id_to_mod[id(param)] = mod_name

    mapping: Dict[str, str] = {}
    for pname, param in model.named_parameters():
        mod_name = id_to_mod.get(id(param))
        if mod_name is not None:
            mapping[pname] = mod_name
    return mapping


def group_modules_for_chunking(
    target_modules: List[Tuple[str, nn.Module]],
) -> List[Tuple[str, List[str]]]:
    """Split targetable modules into three groups for chunked backward.

    Groups: ``self_attn``, ``cross_attn``, ``mlp``.  Each group can be
    processed in a separate backward pass to reduce peak gradient VRAM.

    Args:
        target_modules: Output of :func:`find_all_targetable_modules`.

    Returns:
        List of ``(group_name, [module_names])``.
    """
    self_attn: List[str] = []
    cross_attn: List[str] = []
    mlp: List[str] = []

    for name, _mod in target_modules:
        short = name.rsplit(".", 1)[-1]
        if short in _MLP_PROJECTIONS:
            mlp.append(name)
        elif ".cross_attn." in name:
            cross_attn.append(name)
        else:
            self_attn.append(name)

    groups = []
    if self_attn:
        groups.append(("self_attn", self_attn))
    if cross_attn:
        groups.append(("cross_attn", cross_attn))
    if mlp:
        groups.append(("mlp", mlp))
    return groups
