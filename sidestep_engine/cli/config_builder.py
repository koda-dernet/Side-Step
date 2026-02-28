"""
Config-object construction for ACE-Step Training V2 CLI.

Reads model ``config.json`` for timestep parameters, auto-detects GPU,
and builds adapter config (LoRA or LoKR) + ``TrainingConfigV2`` from CLI args.
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
from sidestep_engine.models.gpu_utils import detect_gpu
from sidestep_engine.core.constants import (
    VARIANT_DIR_MAP,
    DEFAULT_TIMESTEP_MU,
    DEFAULT_TIMESTEP_SIGMA,
    DEFAULT_DATA_PROPORTION,
    PP_LR_WARN_THRESHOLD,
    TURBO_SHIFT,
    BASE_SHIFT,
    TURBO_INFERENCE_STEPS,
    BASE_INFERENCE_STEPS,
    is_pp_compatible,
)
from sidestep_engine.cli.validation import resolve_target_modules

logger = logging.getLogger(__name__)

AdapterConfig = Union[LoRAConfigV2, LoKRConfigV2, LoHAConfigV2, OFTConfigV2]

# JSON key → argparse dest mapping for keys that differ
_JSON_KEY_MAP: Dict[str, str] = {
    "lr": "learning_rate",
    "learning-rate": "learning_rate",
    "epochs": "epochs",
    "batch-size": "batch_size",
    "gradient-accumulation": "gradient_accumulation",
    "save-every": "save_every",
    # GUI frontend uses different key names than argparse dests
    "grad_accum": "gradient_accumulation",
    "scheduler": "scheduler_type",
    "early_stop": "early_stop_patience",
    "projections": "target_modules",
    "self_projections": "self_target_modules",
    "cross_projections": "cross_target_modules",
}


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


def _warn_deprecated_base_model(args: argparse.Namespace) -> None:
    """No-op: --base-model flag was removed in Beta 1.

    Kept as a stub so callers don't need updating.
    """


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

    Returns LoRAConfigV2 when ``args.adapter_type == "lora"`` and
    LoKRConfigV2 when ``args.adapter_type == "lokr"``.

    Also patches in ``timestep_mu``, ``timestep_sigma``, and
    ``data_proportion`` from the model's ``config.json`` so the user
    does not need to pass them manually.
    """
    _warn_deprecated_base_model(args)

    adapter_type = getattr(args, "adapter_type", "lora")

    # -- Resolve model config path ------------------------------------------
    ckpt_root = Path(args.checkpoint_dir)
    model_config_path = _resolve_model_config_path(ckpt_root, args.model_variant)

    timestep_mu = DEFAULT_TIMESTEP_MU
    timestep_sigma = DEFAULT_TIMESTEP_SIGMA
    data_proportion = DEFAULT_DATA_PROPORTION
    num_hidden_layers: int | None = None

    if model_config_path.is_file():
        try:
            mcfg = _json_mod.loads(model_config_path.read_text(encoding="utf-8"))
            timestep_mu = mcfg.get("timestep_mu", timestep_mu)
            timestep_sigma = mcfg.get("timestep_sigma", timestep_sigma)
            data_proportion = mcfg.get("data_proportion", data_proportion)
            num_hidden_layers = mcfg.get("num_hidden_layers")
        except (_json_mod.JSONDecodeError, OSError) as exc:
            logger.warning(
                "[Side-Step] Failed to parse %s: %s -- using default timestep parameters",
                model_config_path, exc,
            )

    # -- Override from --base-model if provided and config lacked params ----
    base_model = getattr(args, "base_model", None)
    if base_model and not model_config_path.is_file():
        from sidestep_engine.models.discovery import get_base_defaults
        defaults = get_base_defaults(base_model)
        timestep_mu = defaults.get("timestep_mu", timestep_mu)
        timestep_sigma = defaults.get("timestep_sigma", timestep_sigma)

    # -- CLI / wizard explicit overrides (take priority over model config) --
    _user_mu = getattr(args, "timestep_mu", None)
    if _user_mu is not None:
        timestep_mu = float(_user_mu)
    _user_sigma = getattr(args, "timestep_sigma", None)
    if _user_sigma is not None:
        timestep_sigma = float(_user_sigma)

    # -- GPU info -----------------------------------------------------------
    gpu_info = detect_gpu(
        requested_device=args.device,
        requested_precision=args.precision,
    )

    # -- Adapter config -------------------------------------------------------
    attention_type = getattr(args, "attention_type", "both")
    target_mlp = getattr(args, "target_mlp", False)
    resolved_modules = resolve_target_modules(
        args.target_modules,
        attention_type,
        self_target_modules=getattr(args, "self_target_modules", None),
        cross_target_modules=getattr(args, "cross_target_modules", None),
        target_mlp=target_mlp,
    )

    adapter_cfg: AdapterConfig
    if adapter_type == "lokr":
        adapter_cfg = LoKRConfigV2(
            linear_dim=getattr(args, "lokr_linear_dim", 64),
            linear_alpha=getattr(args, "lokr_linear_alpha", 128),
            factor=getattr(args, "lokr_factor", -1),
            decompose_both=getattr(args, "lokr_decompose_both", False),
            use_tucker=getattr(args, "lokr_use_tucker", False),
            use_scalar=getattr(args, "lokr_use_scalar", False),
            weight_decompose=getattr(args, "lokr_weight_decompose", False),
            target_modules=resolved_modules,
            attention_type=attention_type,
            target_mlp=target_mlp,
        )
    elif adapter_type == "loha":
        adapter_cfg = LoHAConfigV2(
            linear_dim=getattr(args, "loha_linear_dim", 64),
            linear_alpha=getattr(args, "loha_linear_alpha", 128),
            factor=getattr(args, "loha_factor", -1),
            use_tucker=getattr(args, "loha_use_tucker", False),
            use_scalar=getattr(args, "loha_use_scalar", False),
            target_modules=resolved_modules,
            attention_type=attention_type,
            target_mlp=target_mlp,
        )
    elif adapter_type == "oft":
        adapter_cfg = OFTConfigV2(
            block_size=getattr(args, "oft_block_size", 64),
            coft=getattr(args, "oft_coft", False),
            eps=getattr(args, "oft_eps", 6e-5),
            target_modules=resolved_modules,
            attention_type=attention_type,
            target_mlp=target_mlp,
        )
    else:
        # lora and dora share the same config; dora sets use_dora=True
        adapter_cfg = LoRAConfigV2(
            r=args.rank,
            alpha=args.alpha,
            dropout=args.dropout,
            target_modules=resolved_modules,
            bias=args.bias,
            attention_type=attention_type,
            target_mlp=target_mlp,
            use_dora=(adapter_type == "dora"),
        )

    # -- Fisher map auto-detection (LoRA only) --------------------------------
    ignore_fisher = getattr(args, "ignore_fisher_map", False)
    fisher_map_path = Path(args.dataset_dir) / "fisher_map.json"

    if is_pp_compatible(adapter_type) and not ignore_fisher and fisher_map_path.is_file():
        from sidestep_engine.analysis.fisher.io import load_fisher_map
        fisher_data = load_fisher_map(
            fisher_map_path,
            expected_variant=args.model_variant,
            dataset_dir=args.dataset_dir,
            expected_num_layers=num_hidden_layers,
        )
        if fisher_data:
            adapter_cfg.target_modules = fisher_data["target_modules"]
            adapter_cfg.rank_pattern = fisher_data["rank_pattern"]
            adapter_cfg.alpha_pattern = fisher_data["alpha_pattern"]
            budget = fisher_data.get("rank_budget", {})
            adapter_cfg.rank_min = budget.get("min", 16)
            adapter_cfg.r = adapter_cfg.rank_min
            adapter_cfg.alpha = adapter_cfg.rank_min * 2
            logger.info(
                "[Side-Step] Fisher map loaded: %d modules, adaptive ranks %d-%d",
                len(fisher_data["rank_pattern"]),
                budget.get("min", 16),
                budget.get("max", 128),
            )

            if args.learning_rate > PP_LR_WARN_THRESHOLD:
                logger.warning(
                    "[Side-Step] Preprocessing++ is active with lr=%.1e. "
                    "Adaptive ranks concentrate capacity on fewer modules, "
                    "which overfits faster. Consider lowering to ~5e-5.",
                    args.learning_rate,
                )

    # -- Clamp DataLoader flags when num_workers <= 0 -----------------------
    num_workers = args.num_workers
    prefetch_factor = args.prefetch_factor
    persistent_workers = args.persistent_workers

    if num_workers <= 0:
        if persistent_workers:
            logger.info("[Side-Step] num_workers=0 -- forcing persistent_workers=False")
            persistent_workers = False
        if prefetch_factor and prefetch_factor > 0:
            logger.info("[Side-Step] num_workers=0 -- forcing prefetch_factor=0")
            prefetch_factor = 0

    # -- Turbo auto-detection ------------------------------------------------
    infer_steps = getattr(args, "num_inference_steps", TURBO_INFERENCE_STEPS)
    shift = getattr(args, "shift", TURBO_SHIFT)
    base_model_label = getattr(args, "base_model", None) or args.model_variant
    label_lower = base_model_label.lower() if isinstance(base_model_label, str) else ""

    if "turbo" in label_lower:
        is_turbo = True
    elif "base" in label_lower or "sft" in label_lower:
        is_turbo = False
    else:
        is_turbo = int(infer_steps) == TURBO_INFERENCE_STEPS
        logger.info(
            "[Side-Step] Could not determine variant from '%s' -- "
            "inferring %s from num_inference_steps=%s. Use --base-model to override.",
            base_model_label,
            "turbo" if is_turbo else "base/sft",
            infer_steps,
        )

    if not is_turbo:
        if infer_steps == TURBO_INFERENCE_STEPS:
            infer_steps = BASE_INFERENCE_STEPS
        if shift == TURBO_SHIFT:
            shift = BASE_SHIFT

    try:
        gc_ratio = float(getattr(args, "gradient_checkpointing_ratio", 1.0))
    except (TypeError, ValueError):
        gc_ratio = 1.0
    gc_enabled = bool(getattr(args, "gradient_checkpointing", True))
    if gc_ratio <= 0:
        gc_enabled = False

    # -- Training config ----------------------------------------------------
    train_cfg = TrainingConfigV2(
        is_turbo=is_turbo,
        shift=shift,
        num_inference_steps=infer_steps,
        learning_rate=args.learning_rate,
        batch_size=args.batch_size,
        gradient_accumulation_steps=args.gradient_accumulation,
        max_epochs=args.epochs,
        warmup_steps=args.warmup_steps,
        weight_decay=args.weight_decay,
        max_grad_norm=args.max_grad_norm,
        seed=args.seed,
        output_dir=args.output_dir,
        save_every_n_epochs=args.save_every,
        num_workers=num_workers,
        pin_memory=args.pin_memory,
        prefetch_factor=prefetch_factor,
        persistent_workers=persistent_workers,
        # V2 extensions
        adapter_type=adapter_type,
        optimizer_type=getattr(args, "optimizer_type", "adamw"),
        scheduler_type=getattr(args, "scheduler_type", "cosine"),
        scheduler_formula=_resolve_scheduler_formula(args),
        gradient_checkpointing=gc_enabled,
        gradient_checkpointing_ratio=gc_ratio,
        offload_encoder=getattr(args, "offload_encoder", True),
        save_best=getattr(args, "save_best", True),
        save_best_after=getattr(args, "save_best_after", 200),
        early_stop_patience=getattr(args, "early_stop_patience", 0),
        cfg_ratio=getattr(args, "cfg_ratio", 0.15),
        loss_weighting=getattr(args, "loss_weighting", "none"),
        snr_gamma=getattr(args, "snr_gamma", 5.0),
        timestep_mu=timestep_mu,
        timestep_sigma=timestep_sigma,
        data_proportion=data_proportion,
        model_variant=args.model_variant,
        checkpoint_dir=args.checkpoint_dir,
        dataset_dir=args.dataset_dir,
        device=gpu_info.device,
        precision=gpu_info.precision,
        resume_from=args.resume_from,
        strict_resume=getattr(args, "strict_resume", True),
        run_name=getattr(args, "run_name", None),
        log_dir=args.log_dir,
        log_every=args.log_every,
        log_heavy_every=max(0, getattr(args, "log_heavy_every", 50)),
        # Preprocessing
        preprocess=args.preprocess,
        audio_dir=args.audio_dir,
        dataset_json=args.dataset_json,
        tensor_output=args.tensor_output,
        max_duration=args.max_duration,
        normalize=getattr(args, "normalize", "none"),
        chunk_duration=getattr(args, "chunk_duration", None) or None,
        chunk_decay_every=getattr(args, "chunk_decay_every", 10),
        dataset_repeats=getattr(args, "dataset_repeats", 1),
        max_steps=getattr(args, "max_steps", 0),
        # "All the Levers" enhancements
        ema_decay=getattr(args, "ema_decay", 0.0),
        val_split=getattr(args, "val_split", 0.0),
        adaptive_timestep_ratio=getattr(args, "adaptive_timestep_ratio", 0.0),
        warmup_start_factor=getattr(args, "warmup_start_factor", 0.1),
        cosine_eta_min_ratio=getattr(args, "cosine_eta_min_ratio", 0.01),
        cosine_restarts_count=getattr(args, "cosine_restarts_count", 4),
        save_best_every_n_steps=getattr(args, "save_best_every_n_steps", 0),
    )

    return adapter_cfg, train_cfg


def build_configs_from_dict(params: Dict[str, Any]) -> Tuple[AdapterConfig, TrainingConfigV2]:
    """Build configs from a flat parameter dict (used by GUI and Wizard).

    Delegates to ``core.config_factory.build_training_config``.
    """
    from sidestep_engine.core.config_factory import build_training_config
    return build_training_config(params)
