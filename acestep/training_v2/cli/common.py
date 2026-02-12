"""
Shared argparse setup, path validation, device auto-detection, and
config-object construction for every ACE-Step Training V2 subcommand.

Usage from an entry-point script::

    from acestep.training_v2.cli.common import build_root_parser, build_configs

    parser = build_root_parser()
    args = parser.parse_args()
    lora_cfg, train_cfg = build_configs(args)
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Optional, Tuple

from acestep.training_v2.configs import LoRAConfigV2, TrainingConfigV2
from acestep.training_v2.gpu_utils import detect_gpu

logger = logging.getLogger(__name__)

# Windows uses spawn-based multiprocessing which breaks DataLoader workers
_DEFAULT_NUM_WORKERS = 0 if sys.platform == "win32" else 4

# ---- Model variant -> checkpoint subdirectory mapping ---------------------
VARIANT_DIR_MAP = {
    "turbo": "acestep-v15-turbo",
    "base": "acestep-v15-base",
    "sft": "acestep-v15-sft",
}


# ===========================================================================
# Root parser
# ===========================================================================

def build_root_parser() -> argparse.ArgumentParser:
    """Build the top-level argparse parser with all subcommands."""

    # -- Try to use Rich help formatter if available -------------------------
    formatter_class = argparse.HelpFormatter
    try:
        from acestep.training_v2.ui.help_formatter import RichHelpFormatter
        formatter_class = RichHelpFormatter
    except ImportError:
        pass

    root = argparse.ArgumentParser(
        prog="train.py",
        description="ACE-Step Training V2 -- corrected LoRA fine-tuning CLI",
        formatter_class=formatter_class,
    )

    # -- Global flags (before subcommand) ------------------------------------
    root.add_argument(
        "--plain",
        action="store_true",
        default=False,
        help="Disable Rich output; use plain text (also set automatically when stdout is not a TTY)",
    )
    root.add_argument(
        "--yes",
        "-y",
        action="store_true",
        default=False,
        help="Skip the confirmation prompt and start training immediately",
    )

    subparsers = root.add_subparsers(dest="subcommand", required=True)

    # -- vanilla -------------------------------------------------------------
    p_vanilla = subparsers.add_parser(
        "vanilla",
        help="Reproduce existing (bugged) training for backward compatibility",
        formatter_class=formatter_class,
    )
    _add_common_training_args(p_vanilla)

    # -- fixed ---------------------------------------------------------------
    p_fixed = subparsers.add_parser(
        "fixed",
        help="Corrected training: continuous timesteps + CFG dropout",
        formatter_class=formatter_class,
    )
    _add_common_training_args(p_fixed)
    _add_fixed_args(p_fixed)

    # -- selective -----------------------------------------------------------
    p_selective = subparsers.add_parser(
        "selective",
        help="Corrected training with dataset-specific module selection",
        formatter_class=formatter_class,
    )
    _add_common_training_args(p_selective)
    _add_fixed_args(p_selective)
    _add_selective_args(p_selective)

    # -- estimate ------------------------------------------------------------
    p_estimate = subparsers.add_parser(
        "estimate",
        help="Gradient sensitivity analysis (no training)",
        formatter_class=formatter_class,
    )
    _add_model_args(p_estimate)
    _add_device_args(p_estimate)
    _add_estimation_args(p_estimate)
    p_estimate.add_argument(
        "--dataset-dir",
        type=str,
        required=True,
        help="Directory containing preprocessed .pt files",
    )
    p_estimate.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Batch size for estimation forward passes (default: 1)",
    )
    p_estimate.add_argument(
        "--num-workers",
        type=int,
        default=_DEFAULT_NUM_WORKERS,
        help=f"DataLoader workers (default: {_DEFAULT_NUM_WORKERS}; 0 on Windows)",
    )
    p_estimate.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )

    # -- compare-configs -----------------------------------------------------
    p_compare = subparsers.add_parser(
        "compare-configs",
        help="Compare module config JSON files",
        formatter_class=formatter_class,
    )
    p_compare.add_argument(
        "--configs",
        nargs="+",
        required=True,
        metavar="JSON",
        help="Paths to module config JSON files to compare",
    )

    return root


# ===========================================================================
# Argument groups
# ===========================================================================

def _add_model_args(parser: argparse.ArgumentParser) -> None:
    """Add --checkpoint-dir and --model-variant."""
    g = parser.add_argument_group("Model / paths")
    g.add_argument(
        "--checkpoint-dir",
        type=str,
        required=True,
        help="Path to checkpoints root directory",
    )
    g.add_argument(
        "--model-variant",
        type=str,
        default="turbo",
        choices=["turbo", "base", "sft"],
        help="Model variant (default: turbo)",
    )


def _add_device_args(parser: argparse.ArgumentParser) -> None:
    """Add --device and --precision."""
    g = parser.add_argument_group("Device / platform")
    g.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Device: auto, cuda, cuda:0, mps, xpu, cpu (default: auto)",
    )
    g.add_argument(
        "--precision",
        type=str,
        default="auto",
        choices=["auto", "bf16", "fp16", "fp32"],
        help="Precision: auto, bf16, fp16, fp32 (default: auto)",
    )


def _add_common_training_args(parser: argparse.ArgumentParser) -> None:
    """Add arguments shared by vanilla / fixed / selective subcommands."""
    _add_model_args(parser)
    _add_device_args(parser)

    # -- Data ----------------------------------------------------------------
    g_data = parser.add_argument_group("Data")
    g_data.add_argument(
        "--dataset-dir",
        type=str,
        required=True,
        help="Directory containing preprocessed .pt files",
    )
    g_data.add_argument(
        "--num-workers",
        type=int,
        default=_DEFAULT_NUM_WORKERS,
        help=f"DataLoader workers (default: {_DEFAULT_NUM_WORKERS}; 0 on Windows)",
    )
    g_data.add_argument(
        "--pin-memory",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Pin memory for GPU transfer (default: True)",
    )
    g_data.add_argument(
        "--prefetch-factor",
        type=int,
        default=2,
        help="DataLoader prefetch factor (default: 2)",
    )
    g_data.add_argument(
        "--persistent-workers",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Keep workers alive between epochs (default: True)",
    )

    # -- Training hyperparams ------------------------------------------------
    g_train = parser.add_argument_group("Training")
    g_train.add_argument(
        "--lr",
        "--learning-rate",
        type=float,
        default=1e-4,
        dest="learning_rate",
        help="Initial learning rate (default: 1e-4)",
    )
    g_train.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Training batch size (default: 1)",
    )
    g_train.add_argument(
        "--gradient-accumulation",
        type=int,
        default=4,
        help="Gradient accumulation steps (default: 4)",
    )
    g_train.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Maximum training epochs (default: 100)",
    )
    g_train.add_argument(
        "--warmup-steps",
        type=int,
        default=100,
        help="LR warmup steps (default: 100)",
    )
    g_train.add_argument(
        "--weight-decay",
        type=float,
        default=0.01,
        help="AdamW weight decay (default: 0.01)",
    )
    g_train.add_argument(
        "--max-grad-norm",
        type=float,
        default=1.0,
        help="Gradient clipping norm (default: 1.0)",
    )
    g_train.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )
    g_train.add_argument(
        "--optimizer-type",
        type=str,
        default="adamw",
        choices=["adamw", "adamw8bit", "adafactor", "prodigy"],
        help="Optimizer: adamw, adamw8bit (low VRAM), adafactor, prodigy (auto-LR) (default: adamw)",
    )
    g_train.add_argument(
        "--scheduler-type",
        type=str,
        default="cosine",
        choices=["cosine", "linear", "constant", "constant_with_warmup"],
        help="LR scheduler: cosine, linear, constant, constant_with_warmup (default: cosine)",
    )
    g_train.add_argument(
        "--gradient-checkpointing",
        action="store_true",
        default=False,
        help="Recompute activations to save VRAM (~40-60%% less, ~30%% slower)",
    )
    g_train.add_argument(
        "--offload-encoder",
        action="store_true",
        default=False,
        help="Move encoder/VAE to CPU after setup (saves ~2-4GB VRAM)",
    )

    # -- LoRA hyperparams ---------------------------------------------------
    g_lora = parser.add_argument_group("LoRA")
    g_lora.add_argument(
        "--rank",
        "-r",
        type=int,
        default=64,
        help="LoRA rank (default: 64)",
    )
    g_lora.add_argument(
        "--alpha",
        type=int,
        default=128,
        help="LoRA alpha (default: 128)",
    )
    g_lora.add_argument(
        "--dropout",
        type=float,
        default=0.1,
        help="LoRA dropout (default: 0.1)",
    )
    g_lora.add_argument(
        "--target-modules",
        nargs="+",
        default=["q_proj", "k_proj", "v_proj", "o_proj"],
        help="Modules to apply LoRA to (default: q_proj k_proj v_proj o_proj)",
    )
    g_lora.add_argument(
        "--bias",
        type=str,
        default="none",
        choices=["none", "all", "lora_only"],
        help="Bias training mode (default: none)",
    )
    g_lora.add_argument(
        "--attention-type",
        type=str,
        default="both",
        choices=["self", "cross", "both"],
        help="Which attention layers to target: self, cross, or both (default: both)",
    )

    # -- Checkpointing -------------------------------------------------------
    g_ckpt = parser.add_argument_group("Checkpointing")
    g_ckpt.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Output directory for LoRA weights",
    )
    g_ckpt.add_argument(
        "--save-every",
        type=int,
        default=10,
        help="Save checkpoint every N epochs (default: 10)",
    )
    g_ckpt.add_argument(
        "--resume-from",
        type=str,
        default=None,
        help="Path to checkpoint dir to resume from",
    )

    # -- Logging / TensorBoard -----------------------------------------------
    g_log = parser.add_argument_group("Logging / TensorBoard")
    g_log.add_argument(
        "--log-dir",
        type=str,
        default=None,
        help="TensorBoard log directory (default: {output-dir}/runs)",
    )
    g_log.add_argument(
        "--log-every",
        type=int,
        default=10,
        help="Log basic metrics every N steps (default: 10)",
    )
    g_log.add_argument(
        "--log-heavy-every",
        type=int,
        default=50,
        help="Log per-layer gradient norms every N steps (default: 50)",
    )
    g_log.add_argument(
        "--sample-every-n-epochs",
        type=int,
        default=0,
        help="Generate audio sample every N epochs; 0=disabled (default: 0)",
    )

    # -- Preprocessing -------------------------------------------------------
    g_pre = parser.add_argument_group("Preprocessing")
    g_pre.add_argument(
        "--preprocess",
        action="store_true",
        default=False,
        help="Run preprocessing before training",
    )
    g_pre.add_argument(
        "--audio-dir",
        type=str,
        default=None,
        help="Source audio directory (preprocessing)",
    )
    g_pre.add_argument(
        "--dataset-json",
        type=str,
        default=None,
        help="Labeled dataset JSON file (preprocessing)",
    )
    g_pre.add_argument(
        "--tensor-output",
        type=str,
        default=None,
        help="Output directory for .pt tensor files (preprocessing)",
    )
    g_pre.add_argument(
        "--max-duration",
        type=float,
        default=240.0,
        help="Max audio duration in seconds (default: 240)",
    )


def _add_fixed_args(parser: argparse.ArgumentParser) -> None:
    """Add arguments specific to the fixed/selective subcommands."""
    g = parser.add_argument_group("Corrected training")
    g.add_argument(
        "--cfg-ratio",
        type=float,
        default=0.15,
        help="CFG dropout probability (default: 0.15)",
    )


def _add_selective_args(parser: argparse.ArgumentParser) -> None:
    """Add arguments specific to the selective subcommand."""
    g = parser.add_argument_group("Selective / estimation")
    g.add_argument(
        "--module-config",
        type=str,
        default=None,
        help="Path to JSON module config from estimation",
    )
    g.add_argument(
        "--auto-estimate",
        action="store_true",
        default=False,
        help="Run estimation inline before training",
    )
    _add_estimation_args(parser)


def _add_estimation_args(parser: argparse.ArgumentParser) -> None:
    """Add arguments shared by estimate and selective subcommands."""
    g = parser.add_argument_group("Estimation")
    g.add_argument(
        "--estimate-batches",
        type=int,
        default=None,
        help="Number of batches for estimation (default: auto from GPU)",
    )
    g.add_argument(
        "--top-k",
        type=int,
        default=16,
        help="Number of top modules to select (default: 16)",
    )
    g.add_argument(
        "--granularity",
        type=str,
        default="module",
        choices=["layer", "module"],
        help="Estimation granularity (default: module)",
    )
    g.add_argument(
        "--output",
        type=str,
        default=None,
        dest="estimate_output",
        help="Path to write module config JSON (estimate only)",
    )


# ===========================================================================
# Path validation
# ===========================================================================

def validate_paths(args: argparse.Namespace) -> bool:
    """Validate that required paths exist.  Returns True if all OK.

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

    variant_dir = VARIANT_DIR_MAP.get(args.model_variant)
    if variant_dir is None:
        print(f"[FAIL] Unknown model variant: {args.model_variant}", file=sys.stderr)
        return False

    model_dir = ckpt_root / variant_dir
    if not model_dir.is_dir():
        print(
            f"[FAIL] Model directory not found: {model_dir}\n"
            f"       Expected subdirectory '{variant_dir}' under {ckpt_root}",
            file=sys.stderr,
        )
        return False

    # Dataset dir
    ds_dir = getattr(args, "dataset_dir", None)
    if ds_dir is not None and not Path(ds_dir).is_dir():
        print(f"[FAIL] Dataset directory not found: {ds_dir}", file=sys.stderr)
        return False

    # Resume path
    resume = getattr(args, "resume_from", None)
    if resume is not None and not Path(resume).exists():
        print(f"[WARN] Resume path not found (will train from scratch): {resume}", file=sys.stderr)

    return True


# ===========================================================================
# Target module resolution
# ===========================================================================

def resolve_target_modules(target_modules: list, attention_type: str) -> list:
    """Resolve target modules based on attention type selection.

    Args:
        target_modules: List of module patterns (e.g. ["q_proj", "v_proj"])
        attention_type: One of "self", "cross", or "both"

    Returns:
        Resolved list of module patterns with appropriate prefixes.

    Examples:
        resolve_target_modules(["q_proj", "v_proj"], "both")
        -> ["q_proj", "v_proj"]  # unchanged, PEFT matches all

        resolve_target_modules(["q_proj", "v_proj"], "self")
        -> ["self_attn.q_proj", "self_attn.v_proj"]

        resolve_target_modules(["q_proj", "v_proj"], "cross")
        -> ["cross_attn.q_proj", "cross_attn.v_proj"]
    """
    if attention_type == "both":
        return target_modules

    # Map attention_type to prefix
    prefix_map = {
        "self": "self_attn",
        "cross": "cross_attn",
    }
    prefix = prefix_map.get(attention_type)
    if prefix is None:
        return target_modules

    resolved = []
    for mod in target_modules:
        # Skip if user already specified a full path (contains a dot)
        if "." in mod:
            resolved.append(mod)
        else:
            resolved.append(f"{prefix}.{mod}")

    return resolved


# ===========================================================================
# Config construction
# ===========================================================================

def build_configs(args: argparse.Namespace) -> Tuple[LoRAConfigV2, TrainingConfigV2]:
    """Construct LoRAConfigV2 and TrainingConfigV2 from parsed CLI args.

    Also patches in ``timestep_mu``, ``timestep_sigma``, and
    ``data_proportion`` from the model's ``config.json`` so the user
    does not need to pass them manually.
    """
    import json as _json

    # -- Resolve model config path ------------------------------------------
    ckpt_root = Path(args.checkpoint_dir)
    variant_dir = VARIANT_DIR_MAP[args.model_variant]
    model_config_path = ckpt_root / variant_dir / "config.json"

    timestep_mu = -0.4
    timestep_sigma = 1.0
    data_proportion = 0.5

    if model_config_path.is_file():
        mcfg = _json.loads(model_config_path.read_text())
        timestep_mu = mcfg.get("timestep_mu", timestep_mu)
        timestep_sigma = mcfg.get("timestep_sigma", timestep_sigma)
        data_proportion = mcfg.get("data_proportion", data_proportion)

    # -- GPU info -----------------------------------------------------------
    gpu_info = detect_gpu(
        requested_device=args.device,
        requested_precision=args.precision,
    )

    # -- LoRA config --------------------------------------------------------
    attention_type = getattr(args, "attention_type", "both")
    resolved_modules = resolve_target_modules(args.target_modules, attention_type)

    lora_cfg = LoRAConfigV2(
        r=args.rank,
        alpha=args.alpha,
        dropout=args.dropout,
        target_modules=resolved_modules,
        bias=args.bias,
        attention_type=attention_type,
    )

    # -- Training config ----------------------------------------------------
    train_cfg = TrainingConfigV2(
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
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        prefetch_factor=args.prefetch_factor,
        persistent_workers=args.persistent_workers,
        # V2 extensions
        optimizer_type=getattr(args, "optimizer_type", "adamw"),
        scheduler_type=getattr(args, "scheduler_type", "cosine"),
        gradient_checkpointing=getattr(args, "gradient_checkpointing", False),
        offload_encoder=getattr(args, "offload_encoder", False),
        cfg_ratio=getattr(args, "cfg_ratio", 0.15),
        timestep_mu=timestep_mu,
        timestep_sigma=timestep_sigma,
        data_proportion=data_proportion,
        model_variant=args.model_variant,
        checkpoint_dir=args.checkpoint_dir,
        dataset_dir=args.dataset_dir,
        device=gpu_info.device,
        precision=gpu_info.precision,
        resume_from=args.resume_from,
        log_dir=args.log_dir,
        log_every=args.log_every,
        log_heavy_every=args.log_heavy_every,
        sample_every_n_epochs=args.sample_every_n_epochs,
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
    )

    return lora_cfg, train_cfg
