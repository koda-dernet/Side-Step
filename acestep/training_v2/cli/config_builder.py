"""
Config-object construction for ACE-Step Training V2 CLI.

Reads model ``config.json`` for timestep parameters, auto-detects GPU,
and builds adapter config (LoRA or LoKR) + ``TrainingConfigV2`` from CLI args.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Tuple, Union

from acestep.training_v2.configs import LoRAConfigV2, LoKRConfigV2, TrainingConfigV2
from acestep.training_v2.gpu_utils import detect_gpu
from acestep.training_v2.path_utils import normalize_path
from acestep.training_v2.cli.args import VARIANT_DIR_MAP
from acestep.training_v2.cli.validation import resolve_target_modules

logger = logging.getLogger(__name__)

AdapterConfig = Union[LoRAConfigV2, LoKRConfigV2]


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


def build_configs(args: argparse.Namespace) -> Tuple[AdapterConfig, TrainingConfigV2]:
    """Construct adapter config and TrainingConfigV2 from parsed CLI args.

    Returns LoRAConfigV2 when ``args.adapter_type == "lora"`` and
    LoKRConfigV2 when ``args.adapter_type == "lokr"``.

    Also patches in ``timestep_mu``, ``timestep_sigma``, and
    ``data_proportion`` from the model's ``config.json`` so the user
    does not need to pass them manually.
    """
    import json as _json

    adapter_type = getattr(args, "adapter_type", "lora")

    # -- Resolve model config path ------------------------------------------
    ckpt_root = Path(args.checkpoint_dir)
    model_config_path = _resolve_model_config_path(ckpt_root, args.model_variant)

    timestep_mu = -0.4
    timestep_sigma = 1.0
    data_proportion = 0.5
    num_hidden_layers: int | None = None

    if model_config_path.is_file():
        try:
            mcfg = _json.loads(model_config_path.read_text(encoding="utf-8"))
            timestep_mu = mcfg.get("timestep_mu", timestep_mu)
            timestep_sigma = mcfg.get("timestep_sigma", timestep_sigma)
            data_proportion = mcfg.get("data_proportion", data_proportion)
            num_hidden_layers = mcfg.get("num_hidden_layers")
        except (_json.JSONDecodeError, OSError) as exc:
            logger.warning(
                "[Side-Step] Failed to parse %s: %s -- using default timestep parameters",
                model_config_path, exc,
            )

    # -- Override from --base-model if provided and config lacked params ----
    base_model = getattr(args, "base_model", None)
    if base_model and not model_config_path.is_file():
        from acestep.training_v2.model_discovery import get_base_defaults
        defaults = get_base_defaults(base_model)
        timestep_mu = defaults.get("timestep_mu", timestep_mu)
        timestep_sigma = defaults.get("timestep_sigma", timestep_sigma)

    # -- GPU info -----------------------------------------------------------
    gpu_info = detect_gpu(
        requested_device=args.device,
        requested_precision=args.precision,
    )

    # -- Adapter config (LoRA or LoKR) --------------------------------------
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
    else:
        adapter_cfg = LoRAConfigV2(
            r=args.rank,
            alpha=args.alpha,
            dropout=args.dropout,
            target_modules=resolved_modules,
            bias=args.bias,
            attention_type=attention_type,
            target_mlp=target_mlp,
        )

    # -- Fisher map auto-detection (LoRA only) --------------------------------
    ignore_fisher = getattr(args, "ignore_fisher_map", False)
    fisher_map_path = Path(args.dataset_dir) / "fisher_map.json"

    if adapter_type == "lora" and not ignore_fisher and fisher_map_path.is_file():
        from acestep.training_v2.fisher.io import load_fisher_map
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

            _PP_LR_WARN_THRESHOLD = 1e-4
            if args.learning_rate > _PP_LR_WARN_THRESHOLD:
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
    infer_steps = getattr(args, "num_inference_steps", 8)
    shift = getattr(args, "shift", 3.0)
    base_model_label = getattr(args, "base_model", None) or args.model_variant
    label_lower = base_model_label.lower() if isinstance(base_model_label, str) else ""

    if "turbo" in label_lower:
        is_turbo = True
    elif "base" in label_lower or "sft" in label_lower:
        is_turbo = False
    else:
        # Unknown custom model names: infer from inference-step metadata.
        # 8-step schedules are turbo-style; anything else defaults to base/SFT.
        is_turbo = int(infer_steps) == 8
        logger.info(
            "[Side-Step] Could not determine variant from '%s' -- "
            "inferring %s from num_inference_steps=%s. Use --base-model to override.",
            base_model_label,
            "turbo" if is_turbo else "base/sft",
            infer_steps,
        )

    # Auto-correct shift / inference-steps metadata when the CLI default
    # (turbo-oriented) doesn't match the detected variant.
    if not is_turbo:
        if infer_steps == 8:
            infer_steps = 50
        if shift == 3.0:
            shift = 1.0

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
        gradient_checkpointing=getattr(args, "gradient_checkpointing", True),
        offload_encoder=getattr(args, "offload_encoder", False),
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
        dataset_dir=normalize_path(args.dataset_dir) or (args.dataset_dir or "").strip(),
        device=gpu_info.device,
        precision=gpu_info.precision,
        resume_from=args.resume_from,
        strict_resume=getattr(args, "strict_resume", True),
        run_name=getattr(args, "run_name", None),
        log_dir=args.log_dir,
        log_every=args.log_every,
        log_heavy_every=max(0, getattr(args, "log_heavy_every", 50)),
        # Estimation / selective (may not exist on all subcommands)
        estimate_batches=getattr(args, "estimate_batches", None),
        top_k=getattr(args, "top_k", 16),
        granularity=getattr(args, "granularity", "module"),
        module_config=getattr(args, "module_config", None),
        auto_estimate=getattr(args, "auto_estimate", False),
        estimate_output=getattr(args, "estimate_output", None),
        # Preprocessing
        preprocess=args.preprocess,
        audio_dir=args.audio_dir,
        dataset_json=args.dataset_json,
        tensor_output=args.tensor_output,
        max_duration=args.max_duration,
        normalize=getattr(args, "normalize", "none"),
        chunk_duration=getattr(args, "chunk_duration", None),
    )

    return adapter_cfg, train_cfg
