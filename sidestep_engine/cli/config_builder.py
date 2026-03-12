"""
Config-object construction for ACE-Step Training V2 CLI.

Reads model ``config.json`` for timestep parameters, auto-detects GPU,
and builds adapter config (LoRA, DoRA, LoKR, LoHA, or OFT) +
``TrainingConfigV2`` from CLI args.
"""

from __future__ import annotations

import argparse
import json as _json_mod
import logging
import warnings
from pathlib import Path
from typing import Any, Dict, Tuple, Union

from sidestep_engine.core.configs import (
    LoRAConfigV2, LoKRConfigV2, LoHAConfigV2, OFTConfigV2, TrainingConfigV2,
)
from sidestep_engine.core.constants import VARIANT_DIR_MAP

logger = logging.getLogger(__name__)

AdapterConfig = Union[LoRAConfigV2, LoKRConfigV2, LoHAConfigV2, OFTConfigV2]

# JSON key → argparse dest mapping for keys that differ.
# Canonical source: sidestep_engine.training_defaults.GUI_KEY_MAP
from sidestep_engine.training_defaults import GUI_KEY_MAP as _JSON_KEY_MAP  # noqa: E402


def _coerce_type(value: Any, reference: Any) -> Any:
    """Cast *value* to match the type of *reference* (argparse default).

    HTML form inputs are always strings; this ensures numeric and boolean
    values survive the GUI → JSON → argparse round-trip.
    """
    if value is None:
        return value
    if reference is None:
        # Default is None — try to auto-coerce strings that look numeric
        if isinstance(value, str):
            # Don't coerce obvious path/name strings
            if value == "" or "/" in value or "\\" in value:
                return value
            try:
                f = float(value)
                return int(f) if f == int(f) else f
            except (ValueError, OverflowError):
                pass
        return value
    ref_type = type(reference)
    if isinstance(value, ref_type):
        return value  # already correct type
    try:
        if ref_type is bool:
            if isinstance(value, str):
                return value.lower() in ("true", "1", "yes")
            return bool(value)
        if ref_type is int:
            return int(float(value))  # handles "64" and "64.0"
        if ref_type is float:
            return float(value)  # handles "3e-4"
    except (ValueError, TypeError):
        pass
    return value  # can't coerce, pass through


def _apply_config_file(args: argparse.Namespace) -> None:
    """Merge a JSON config file into the argparse namespace.

    Values from the JSON are applied only when the corresponding CLI arg
    was not explicitly provided (i.e. still has its default value).
    CLI args always take priority over JSON values.
    """
    config_path = getattr(args, "config", None)
    if not config_path:
        return

    p = Path(config_path)
    if not p.is_file():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    data: Dict[str, Any] = _json_mod.loads(p.read_text(encoding="utf-8"))
    logger.info("[Side-Step] Loading config from %s (%d keys)", config_path, len(data))

    # Internal metadata keys the GUI/wizard embed in config JSON
    _INTERNAL_KEYS = {
        "config", "subcommand", "yes", "_from_wizard", "_from_gui",
        "plain", "preprocess", "preprocess_only",
        # use_dora is derived from adapter_type=="dora" in config_factory;
        # the GUI sends it redundantly — suppress the unknown-key warning.
        "use_dora",
    }
    unknown_keys = []
    for key, value in data.items():
        dest = _JSON_KEY_MAP.get(key, key.replace("-", "_"))
        current = getattr(args, dest, _SENTINEL)
        if current is _SENTINEL:
            if dest not in _INTERNAL_KEYS and not dest.startswith("_"):
                unknown_keys.append(key)
            continue
        # Only apply if CLI didn't explicitly set this arg (value == default)
        default = _DEFAULTS_CACHE.get(dest, _SENTINEL)
        if current == default or current is None:
            coerced = _coerce_type(value, default)
            # GUI sends target_modules as space-separated string; argparse expects list
            if isinstance(coerced, str) and dest in _LIST_DEST_KEYS:
                coerced = coerced.split()
            setattr(args, dest, coerced)

    if unknown_keys:
        import sys
        preview = ", ".join(unknown_keys[:5])
        extra = f" (and {len(unknown_keys) - 5} more)" if len(unknown_keys) > 5 else ""
        print(
            f"[WARN] Config file contains {len(unknown_keys)} unrecognized key(s): "
            f"{preview}{extra}. Check for typos.",
            file=sys.stderr,
        )


# Sentinel for missing attrs
_SENTINEL = object()

# Argparse dests whose values are lists (GUI sends as space-separated strings)
_LIST_DEST_KEYS = {"target_modules", "self_target_modules", "cross_target_modules"}

# Cache of argparse defaults for the fixed subparser (populated lazily)
_DEFAULTS_CACHE: Dict[str, Any] = {}


def _populate_defaults_cache() -> None:
    """Build the defaults cache from the train subparser (called once)."""
    if _DEFAULTS_CACHE:
        return
    from sidestep_engine.cli.args import build_root_parser
    parser = build_root_parser()
    for action in parser._subparsers._actions:
        if hasattr(action, "_parser_class"):
            for name, sub in action.choices.items():
                if name == "train":
                    for a in sub._actions:
                        if a.dest and a.dest != "help":
                            _DEFAULTS_CACHE[a.dest] = a.default
                    return



def _resolve_model_config_path(ckpt_root: Path, variant: str) -> Path:
    """Find config.json for *variant*, supporting custom folder names.

    Checks the ``VARIANT_DIR_MAP`` alias first, then tries *variant* as
    a literal subdirectory name.
    """
    # 1. Known alias
    mapped = VARIANT_DIR_MAP.get(variant)
    if mapped:
        p = ckpt_root / mapped / "config.json"
        if p.is_file():
            return p

    # 2. Literal folder name (fine-tunes, custom models)
    p = ckpt_root / variant / "config.json"
    if p.is_file():
        return p

    # 3. Fallback: return the mapped path (even if missing) so the caller
    #    gets a meaningful "not found" message.
    return ckpt_root / (mapped or variant) / "config.json"


def _resolve_scheduler_formula(args: argparse.Namespace) -> str:
    """Resolve scheduler_formula with cross-field validation.

    Warns when a formula is provided but scheduler_type is not 'custom'
    (the formula would be silently ignored).  Returns empty string when
    scheduler_type is not 'custom'.
    """
    sched_type = getattr(args, "scheduler_type", "cosine")
    formula = getattr(args, "scheduler_formula", "")
    if sched_type != "custom" and formula:
        logger.warning(
            "[Side-Step] --scheduler-formula was provided but --scheduler-type "
            "is '%s' (not 'custom') -- the formula will be ignored. "
            "Set --scheduler-type custom to use a custom formula.",
            sched_type,
        )
        return ""
    return formula


def build_configs(args: argparse.Namespace) -> Tuple[AdapterConfig, TrainingConfigV2]:
    """Construct adapter config and TrainingConfigV2 from parsed CLI args.

    Converts the Namespace to a flat parameter dict and delegates to
    :func:`~sidestep_engine.core.config_factory.build_training_config`,
    the single source of truth for config construction.
    """
    # Pre-resolve scheduler formula (requires cross-field Namespace access)
    resolved_formula = _resolve_scheduler_formula(args)

    # Convert Namespace → flat dict for config_factory
    from sidestep_engine.core.config_factory import (
        build_training_config,
        namespace_to_params,
    )
    params = namespace_to_params(args)
    # Inject the pre-resolved formula so config_factory uses it
    params["scheduler_formula"] = resolved_formula

    return build_training_config(params)


def build_configs_from_dict(params: Dict[str, Any]) -> Tuple[AdapterConfig, TrainingConfigV2]:
    """Build configs from a flat parameter dict (used by GUI and Wizard).

    Delegates to ``core.config_factory.build_training_config``.
    """
    from sidestep_engine.core.config_factory import build_training_config
    return build_training_config(params)
