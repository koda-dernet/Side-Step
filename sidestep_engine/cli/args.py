"""
Argparse construction for Side-Step CLI.

Contains ``build_root_parser`` and all ``_add_*`` argument-group helpers,
plus shared constants (``_DEFAULT_NUM_WORKERS``, ``VARIANT_DIR_MAP``).
"""

from __future__ import annotations

import argparse
import sys

from sidestep_engine.training_defaults import (
    DEFAULT_ALPHA,
    DEFAULT_ATTENTION_TYPE,
    DEFAULT_BATCH_SIZE,
    DEFAULT_BIAS,
    DEFAULT_CFG_RATIO,
    DEFAULT_CHUNK_DECAY_EVERY,
    DEFAULT_MAX_LATENT_LENGTH,
    DEFAULT_COSINE_ETA_MIN_RATIO,
    DEFAULT_COSINE_RESTARTS_COUNT,
    DEFAULT_DATASET_REPEATS,
    DEFAULT_DROPOUT,
    DEFAULT_EARLY_STOP_PATIENCE,
    DEFAULT_EMA_DECAY,
    DEFAULT_TARGET_LOSS,
    DEFAULT_TARGET_LOSS_FLOOR,
    DEFAULT_TARGET_LOSS_WARMUP,
    DEFAULT_TARGET_LOSS_SMOOTHING,
    DEFAULT_EPOCHS,
    DEFAULT_GRADIENT_ACCUMULATION,
    DEFAULT_GRADIENT_CHECKPOINTING_RATIO,
    DEFAULT_LEARNING_RATE,
    DEFAULT_LOG_EVERY,
    DEFAULT_LOG_HEAVY_EVERY,
    DEFAULT_LOHA_FACTOR,
    DEFAULT_LOHA_LINEAR_ALPHA,
    DEFAULT_LOHA_LINEAR_DIM,
    DEFAULT_LOKR_DECOMPOSE_BOTH,
    DEFAULT_LOKR_FACTOR,
    DEFAULT_LOKR_LINEAR_ALPHA,
    DEFAULT_LOKR_LINEAR_DIM,
    DEFAULT_LOSS_WEIGHTING,
    DEFAULT_MAX_GRAD_NORM,
    DEFAULT_MAX_STEPS,
    DEFAULT_MODEL_VARIANT,
    DEFAULT_NUM_WORKERS as _DEFAULT_NUM_WORKERS,
    DEFAULT_OFT_BLOCK_SIZE,
    DEFAULT_OFT_EPS,
    DEFAULT_OPTIMIZER_TYPE,
    DEFAULT_PREFETCH_FACTOR,
    DEFAULT_RANK,
    DEFAULT_SAVE_BEST_AFTER,
    DEFAULT_SAVE_BEST_EVERY_N_STEPS,
    DEFAULT_SAVE_EVERY,
    DEFAULT_SCHEDULER_TYPE,
    DEFAULT_SEED,
    DEFAULT_SNR_GAMMA,
    DEFAULT_WARMUP_START_FACTOR,
    DEFAULT_WARMUP_STEPS,
    DEFAULT_WEIGHT_DECAY,
    DEFAULT_ADAPTIVE_TIMESTEP_RATIO,
    DEFAULT_VAL_SPLIT,
)

from sidestep_engine.core.constants import VARIANT_DIR_MAP


# ===========================================================================
# Root parser
# ===========================================================================

def build_root_parser() -> argparse.ArgumentParser:
    """Build the top-level argparse parser with all subcommands."""

    formatter_class = argparse.HelpFormatter
    try:
        from sidestep_engine.ui.help_formatter import RichHelpFormatter
        formatter_class = RichHelpFormatter
    except ImportError:
        pass

    root = argparse.ArgumentParser(
        prog="sidestep",
        description="Side-Step -- LoRA/LoKR fine-tuning CLI",
        formatter_class=formatter_class,
    )

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
        help="Skip the confirmation prompt and start immediately",
    )
    # --gui kept as hidden root flag for backward compat (translated by deprecation shim)
    root.add_argument("--gui", action="store_true", default=False, help=argparse.SUPPRESS)
    root.add_argument("--port", type=int, default=8770, help=argparse.SUPPRESS)

    subparsers = root.add_subparsers(dest="subcommand")

    # -- train (was: fixed) --------------------------------------------------
    p_train = subparsers.add_parser(
        "train",
        help="Train an adapter (LoRA, DoRA, LoKR, LoHA, OFT)",
        formatter_class=formatter_class,
    )
    _add_common_training_args(p_train)
    _add_train_args(p_train)

    # -- preprocess (promoted to top-level) ----------------------------------
    p_preprocess = subparsers.add_parser(
        "preprocess",
        help="Preprocess audio into tensors (two-pass pipeline)",
        formatter_class=formatter_class,
    )
    _add_preprocess_subcommand_args(p_preprocess)

    # -- analyze (was: fisher) -----------------------------------------------
    p_analyze = subparsers.add_parser(
        "analyze",
        help="PP++ / Fisher analysis for adaptive LoRA rank assignment",
        formatter_class=formatter_class,
    )
    _add_model_args(p_analyze)
    _add_device_args(p_analyze)
    _add_fisher_args(p_analyze)

    # -- audio-analyze --------------------------------------------------------
    p_aa = subparsers.add_parser(
        "audio-analyze",
        help="Local offline audio analysis: extract BPM, key, and time signature",
        formatter_class=formatter_class,
    )
    _add_audio_analyze_args(p_aa)

    # -- captions ------------------------------------------------------------
    p_cap = subparsers.add_parser(
        "captions",
        help="Generate AI captions + fetch lyrics for audio sidecar files",
        formatter_class=formatter_class,
    )
    _add_captions_args(p_cap)

    # -- tags ----------------------------------------------------------------
    p_tags = subparsers.add_parser(
        "tags",
        help="Bulk sidecar tag operations (add, remove, list, clear trigger tags)",
        formatter_class=formatter_class,
    )
    _add_tags_args(p_tags)

    # -- dataset (was: build-dataset) ----------------------------------------
    p_dataset = subparsers.add_parser(
        "dataset",
        help="Build dataset.json from a folder of audio + sidecar metadata files",
        formatter_class=formatter_class,
    )
    p_dataset.add_argument(
        "--input", "-i",
        type=str,
        required=True,
        help="Root directory containing audio files (scanned recursively)",
    )
    p_dataset.add_argument(
        "--tag",
        type=str,
        default="",
        help="Custom trigger tag applied to all samples (default: none)",
    )
    p_dataset.add_argument(
        "--tag-position",
        type=str,
        default="prepend",
        choices=["prepend", "append", "replace"],
        help="Tag placement in prompts (default: prepend)",
    )
    p_dataset.add_argument(
        "--genre-ratio",
        type=int,
        default=0,
        help="Percentage of samples that use genre instead of caption (0-100, default: 0)",
    )
    p_dataset.add_argument(
        "--name",
        type=str,
        default="local_dataset",
        help="Dataset name in metadata block (default: local_dataset)",
    )
    p_dataset.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output JSON path (default: <input>/dataset.json)",
    )

    # -- convert-sidecars ----------------------------------------------------
    p_convert = subparsers.add_parser(
        "convert-sidecars",
        help="Convert per-file JSON or dataset.json metadata to TXT sidecars",
        formatter_class=formatter_class,
    )
    p_convert.add_argument(
        "--input", "-i",
        type=str,
        required=True,
        help="Directory with per-file .json sidecars, or path to a dataset.json",
    )
    p_convert.add_argument(
        "--audio-dir",
        type=str,
        default=None,
        help="Audio directory (for dataset.json mode; defaults to JSON parent dir)",
    )
    p_convert.add_argument(
        "--overwrite",
        action="store_true",
        default=False,
        help="Overwrite existing .txt sidecars (default: skip files that already have one)",
    )
    p_convert.add_argument(
        "--yes", "-y",
        action="store_true",
        default=False,
        help="Skip confirmation prompt",
    )

    # -- settings ------------------------------------------------------------
    p_settings = subparsers.add_parser(
        "settings",
        help="View or modify Side-Step persistent settings",
        formatter_class=formatter_class,
    )
    _add_settings_args(p_settings)

    # -- history -------------------------------------------------------------
    p_history = subparsers.add_parser(
        "history",
        help="List past training runs",
        formatter_class=formatter_class,
    )
    p_history.add_argument(
        "--limit", type=int, default=20,
        help="Maximum number of runs to show (default: 20)",
    )
    p_history.add_argument(
        "--json", action="store_true", default=False, dest="json_output",
        help="Output raw JSON instead of a table",
    )

    # -- export --------------------------------------------------------------
    p_export = subparsers.add_parser(
        "export",
        help="Export adapter to ComfyUI format",
        formatter_class=formatter_class,
    )
    p_export.add_argument(
        "adapter_dir",
        type=str,
        help="Path to adapter directory (e.g. output/my_lora/final)",
    )
    p_export.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="Output .safetensors file name (default: <adapter_dir_name>_comfyui.safetensors)",
    )
    p_export.add_argument(
        "--format", "-f",
        type=str,
        default="comfyui",
        choices=["comfyui"],
        dest="export_format",
        help="Export format (default: comfyui)",
    )
    p_export.add_argument(
        "--target", "-t",
        type=str,
        default="native",
        choices=["native", "generic"],
        help=(
            "ComfyUI target format (default: native). "
            "native = built-in ACE-Step 1.5 support (base_model.model prefix); "
            "generic = diffusion_model.decoder prefix (try if native fails)"
        ),
    )
    p_export.add_argument(
        "--prefix",
        type=str,
        default=None,
        help="Advanced: explicit key prefix override (ignores --target)",
    )
    p_export.add_argument(
        "--normalize-alpha",
        action="store_true",
        default=False,
        dest="normalize_alpha",
        help="Set alpha=rank so ComfyUI strength 1.0 = natural LoRA magnitude",
    )

    # -- gui -----------------------------------------------------------------
    p_gui = subparsers.add_parser(
        "gui",
        help="Launch the web GUI",
        formatter_class=formatter_class,
    )
    p_gui.add_argument(
        "--port", type=int, default=8770,
        help="GUI server port (default: 8770)",
    )

    return root


# ===========================================================================
# Argument groups
# ===========================================================================

def _add_model_args(parser: argparse.ArgumentParser) -> None:
    """Add --model (was --model-variant) and --checkpoint-dir."""
    g = parser.add_argument_group("Model / paths")
    g.add_argument(
        "--checkpoint-dir", "-c",
        type=str,
        default=None,
        help="Path to checkpoints root directory (auto-resolves from settings if omitted)",
    )
    g.add_argument(
        "--model", "-M", "--model-variant",
        type=str,
        default=DEFAULT_MODEL_VARIANT,
        dest="model_variant",
        metavar="MODEL",
        help=(
            f"Model variant or subfolder name (default: {DEFAULT_MODEL_VARIANT}). "
            "Official: base, sft, turbo. "
            "For fine-tunes: use the exact folder name under checkpoint-dir."
        ),
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
    """Add arguments shared by training subcommands."""
    _add_model_args(parser)
    _add_device_args(parser)

    # -- Data ----------------------------------------------------------------
    g_data = parser.add_argument_group("Data")
    g_data.add_argument(
        "--dataset-dir", "-d",
        type=str,
        default=None,
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
        default=DEFAULT_PREFETCH_FACTOR,
        help=f"DataLoader prefetch factor (default: {DEFAULT_PREFETCH_FACTOR}; 0 on Windows)",
    )
    g_data.add_argument(
        "--persistent-workers",
        action=argparse.BooleanOptionalAction,
        default=_DEFAULT_NUM_WORKERS > 0,
        help="Keep workers alive between epochs (default: True; False on Windows)",
    )

    # -- Training hyperparams ------------------------------------------------
    g_train = parser.add_argument_group("Training")
    g_train.add_argument("--lr", "--learning-rate", "-l", type=float, default=DEFAULT_LEARNING_RATE, dest="learning_rate", help=f"Initial learning rate (default: {DEFAULT_LEARNING_RATE:g})")
    g_train.add_argument("--batch-size", "-b", type=int, default=DEFAULT_BATCH_SIZE, help=f"Training batch size (default: {DEFAULT_BATCH_SIZE})")
    g_train.add_argument("--gradient-accumulation", "-g", type=int, default=DEFAULT_GRADIENT_ACCUMULATION, help=f"Gradient accumulation steps (default: {DEFAULT_GRADIENT_ACCUMULATION})")
    g_train.add_argument("--epochs", "-e", type=int, default=DEFAULT_EPOCHS, help=f"Maximum training epochs (default: {DEFAULT_EPOCHS})")
    g_train.add_argument("--warmup-steps", type=int, default=DEFAULT_WARMUP_STEPS, help=f"LR warmup steps (default: {DEFAULT_WARMUP_STEPS})")
    g_train.add_argument("--weight-decay", type=float, default=DEFAULT_WEIGHT_DECAY, help=f"AdamW weight decay (default: {DEFAULT_WEIGHT_DECAY})")
    g_train.add_argument("--max-grad-norm", type=float, default=DEFAULT_MAX_GRAD_NORM, help=f"Gradient clipping norm (default: {DEFAULT_MAX_GRAD_NORM})")
    g_train.add_argument("--seed", "-s", type=int, default=DEFAULT_SEED, help=f"Random seed (default: {DEFAULT_SEED})")
    g_train.add_argument("--chunk-duration", type=int, default=None,
                         help="Random latent chunk duration in seconds (default: disabled). "
                              "Recommended: 60. Extracts a random window each iteration for data "
                              "augmentation and VRAM savings. WARNING: values below 60s (e.g. 30) "
                              "may reduce training quality for full-length inference")
    g_train.add_argument("--chunk-decay-every", type=int, default=DEFAULT_CHUNK_DECAY_EVERY,
                         help=f"Epoch interval for halving chunk coverage histogram; 0 disables decay (default: {DEFAULT_CHUNK_DECAY_EVERY})")
    g_train.add_argument("--max-latent-length", type=int, default=None,
                         help="Random crop length in latent frames (0 = disabled). Takes precedence over --chunk-duration when > 0")
    g_train.add_argument("--max-steps", "-m", type=int, default=DEFAULT_MAX_STEPS,
                         help=f"Maximum optimizer steps; 0 = use epochs only (default: {DEFAULT_MAX_STEPS})")
    g_train.add_argument("--shift", type=float, default=None, help=argparse.SUPPRESS)
    g_train.add_argument("--num-inference-steps", type=int, default=None, help=argparse.SUPPRESS)
    g_train.add_argument("--optimizer-type", type=str, default=DEFAULT_OPTIMIZER_TYPE, choices=["adamw", "adamw8bit", "adafactor", "prodigy"], help=f"Optimizer (default: {DEFAULT_OPTIMIZER_TYPE})")
    g_train.add_argument("--scheduler-type", type=str, default=DEFAULT_SCHEDULER_TYPE, choices=["cosine", "cosine_restarts", "linear", "constant", "constant_with_warmup", "custom"], help=f"LR scheduler (default: {DEFAULT_SCHEDULER_TYPE})")
    g_train.add_argument("--scheduler-formula", type=str, default="", help="Custom LR formula (Python math expression). Only used with --scheduler-type custom")
    g_train.add_argument("--gradient-checkpointing", action=argparse.BooleanOptionalAction, default=True, help="Recompute activations to save VRAM (~40-60%% less, ~10-30%% slower). On by default; use --no-gradient-checkpointing to disable")
    g_train.add_argument("--gradient-checkpointing-ratio", type=float, default=DEFAULT_GRADIENT_CHECKPOINTING_RATIO, help=f"Fraction of decoder layers to checkpoint (0.0=none, 0.5=half, 1.0=all). Only applies when --gradient-checkpointing is on (default: {DEFAULT_GRADIENT_CHECKPOINTING_RATIO})")
    g_train.add_argument("--offload-encoder", action=argparse.BooleanOptionalAction, default=True, help="Move encoder/VAE to CPU after setup (saves ~2-4GB VRAM). On by default; use --no-offload-encoder to disable")

    # -- All the Levers (experimental enhancements) -------------------------
    g_levers = parser.add_argument_group("All the Levers (experimental)")
    g_levers.add_argument("--ema-decay", type=float, default=DEFAULT_EMA_DECAY, help=f"EMA decay for adapter weights (0=off, 0.9999=typical, default: {DEFAULT_EMA_DECAY})")
    g_levers.add_argument("--val-split", type=float, default=DEFAULT_VAL_SPLIT, help=f"Validation holdout fraction (0=off, 0.1=10%%, default: {DEFAULT_VAL_SPLIT})")
    g_levers.add_argument("--adaptive-timestep-ratio", type=float, default=DEFAULT_ADAPTIVE_TIMESTEP_RATIO, help=f"Adaptive timestep sampling ratio (0=off, 0.3=recommended, default: {DEFAULT_ADAPTIVE_TIMESTEP_RATIO}). Base/SFT only")
    g_levers.add_argument("--warmup-start-factor", type=float, default=DEFAULT_WARMUP_START_FACTOR, help=f"LR warmup starts at base_lr * this (default: {DEFAULT_WARMUP_START_FACTOR})")
    g_levers.add_argument("--cosine-eta-min-ratio", type=float, default=DEFAULT_COSINE_ETA_MIN_RATIO, help=f"Cosine scheduler decays LR to base_lr * this (default: {DEFAULT_COSINE_ETA_MIN_RATIO})")
    g_levers.add_argument("--cosine-restarts-count", type=int, default=DEFAULT_COSINE_RESTARTS_COUNT, help=f"Number of cosine restart cycles (default: {DEFAULT_COSINE_RESTARTS_COUNT})")
    g_levers.add_argument("--save-best-every-n-steps", type=int, default=DEFAULT_SAVE_BEST_EVERY_N_STEPS, help=f"Step-level best-model check interval (0=epoch only, default: {DEFAULT_SAVE_BEST_EVERY_N_STEPS})")
    g_levers.add_argument("--timestep-mu", type=float, default=None, help="Override logit-normal timestep mean (default: from model config, typically -0.4)")
    g_levers.add_argument("--timestep-sigma", type=float, default=None, help="Override logit-normal timestep sigma (default: from model config, typically 1.0)")

    # -- Adapter selection ---------------------------------------------------
    g_adapter = parser.add_argument_group("Adapter")
    g_adapter.add_argument("--adapter", "-a", "--adapter-type", type=str, default="lora", dest="adapter_type", choices=["lora", "dora", "lokr", "loha", "oft"], help="Adapter type: lora, dora, lokr, loha, or oft (default: lora)")

    # -- LoRA hyperparams ---------------------------------------------------
    g_lora = parser.add_argument_group("LoRA (used when --adapter=lora)")
    g_lora.add_argument("--rank", "-r", type=int, default=DEFAULT_RANK, help=f"LoRA rank (default: {DEFAULT_RANK})")
    g_lora.add_argument("--alpha", type=int, default=DEFAULT_ALPHA, help=f"LoRA alpha (default: {DEFAULT_ALPHA})")
    g_lora.add_argument("--dropout", type=float, default=DEFAULT_DROPOUT, help=f"LoRA dropout (default: {DEFAULT_DROPOUT})")
    g_lora.add_argument("--target-modules", nargs="+", default=["q_proj", "k_proj", "v_proj", "o_proj"], help="Modules to apply adapter to")
    g_lora.add_argument("--bias", type=str, default=DEFAULT_BIAS, choices=["none", "all", "lora_only"], help=f"Bias training mode (default: {DEFAULT_BIAS})")
    g_lora.add_argument("--attention-type", type=str, default=DEFAULT_ATTENTION_TYPE, choices=["self", "cross", "both"], help=f"Attention layers to target (default: {DEFAULT_ATTENTION_TYPE})")
    g_lora.add_argument("--self-target-modules", nargs="+", default=None, help="Projections for self-attention only (used when --attention-type=both)")
    g_lora.add_argument("--cross-target-modules", nargs="+", default=None, help="Projections for cross-attention only (used when --attention-type=both)")
    g_lora.add_argument("--target-mlp", action=argparse.BooleanOptionalAction, default=True, help="Target MLP/FFN layers (gate_proj, up_proj, down_proj). On by default; use --no-target-mlp to disable")

    # -- LoKR hyperparams ---------------------------------------------------
    g_lokr = parser.add_argument_group("LoKR (used when --adapter=lokr)")
    g_lokr.add_argument("--lokr-linear-dim", type=int, default=DEFAULT_LOKR_LINEAR_DIM, help=f"LoKR linear dimension (default: {DEFAULT_LOKR_LINEAR_DIM})")
    g_lokr.add_argument("--lokr-linear-alpha", type=int, default=DEFAULT_LOKR_LINEAR_ALPHA, help=f"LoKR linear alpha (default: {DEFAULT_LOKR_LINEAR_ALPHA})")
    g_lokr.add_argument("--lokr-factor", type=int, default=DEFAULT_LOKR_FACTOR, help=f"LoKR factor; -1 for auto (default: {DEFAULT_LOKR_FACTOR})")
    g_lokr.add_argument("--lokr-decompose-both", action="store_true", default=DEFAULT_LOKR_DECOMPOSE_BOTH, help="Decompose both Kronecker factors")
    g_lokr.add_argument("--lokr-use-tucker", action="store_true", default=False, help="Use Tucker decomposition")
    g_lokr.add_argument("--lokr-use-scalar", action="store_true", default=False, help="Use scalar scaling")
    g_lokr.add_argument("--lokr-weight-decompose", action="store_true", default=False, help="Enable DoRA-style weight decomposition")

    # -- LoHA hyperparams ---------------------------------------------------
    g_loha = parser.add_argument_group("LoHA (used when --adapter=loha)")
    g_loha.add_argument("--loha-linear-dim", type=int, default=DEFAULT_LOHA_LINEAR_DIM, help=f"LoHA linear dimension (default: {DEFAULT_LOHA_LINEAR_DIM})")
    g_loha.add_argument("--loha-linear-alpha", type=int, default=DEFAULT_LOHA_LINEAR_ALPHA, help=f"LoHA linear alpha (default: {DEFAULT_LOHA_LINEAR_ALPHA})")
    g_loha.add_argument("--loha-factor", type=int, default=DEFAULT_LOHA_FACTOR, help=f"LoHA factor; -1 for auto (default: {DEFAULT_LOHA_FACTOR})")
    g_loha.add_argument("--loha-use-tucker", action="store_true", default=False, help="Use Tucker decomposition")
    g_loha.add_argument("--loha-use-scalar", action="store_true", default=False, help="Use scalar scaling")

    # -- OFT hyperparams (experimental) -------------------------------------
    g_oft = parser.add_argument_group("OFT [Experimental] (used when --adapter=oft)")
    g_oft.add_argument("--oft-block-size", type=int, default=DEFAULT_OFT_BLOCK_SIZE, help=f"OFT block size (default: {DEFAULT_OFT_BLOCK_SIZE})")
    g_oft.add_argument("--oft-coft", action="store_true", default=False, help="Enable constrained OFT (Cayley projection)")
    g_oft.add_argument("--oft-eps", type=float, default=DEFAULT_OFT_EPS, help=f"OFT epsilon for numerical stability (default: {DEFAULT_OFT_EPS})")

    # -- Config file ---------------------------------------------------------
    g_cfg = parser.add_argument_group("Config file")
    g_cfg.add_argument("--config", type=str, default=None,
                       help="Load training config from JSON file. CLI args override JSON values.")

    # -- Checkpointing -------------------------------------------------------
    g_ckpt = parser.add_argument_group("Checkpointing")
    g_ckpt.add_argument("--output-dir", "-o", type=str, default=None, help="Output directory for adapter weights")
    g_ckpt.add_argument("--save-every", type=int, default=DEFAULT_SAVE_EVERY, help=f"Save checkpoint every N epochs (default: {DEFAULT_SAVE_EVERY})")
    g_ckpt.add_argument("--resume-from", type=str, default=None, help="Path to checkpoint dir to resume from")
    g_ckpt.add_argument("--strict-resume", action=argparse.BooleanOptionalAction, default=True,
                         help="Abort on config mismatch or failed state restore during resume (default: True)")
    g_ckpt.add_argument("--run-name", "-n", type=str, default=None,
                         help="Name for this training run (used for output dir, TB logs). Auto-generated if omitted")
    g_ckpt.add_argument("--save-best", action=argparse.BooleanOptionalAction, default=True,
                         help="Auto-save best model by smoothed loss (default: True)")
    g_ckpt.add_argument("--save-best-after", type=int, default=DEFAULT_SAVE_BEST_AFTER,
                         help=f"Epoch to start best-model tracking (default: {DEFAULT_SAVE_BEST_AFTER})")
    g_ckpt.add_argument("--early-stop-patience", type=int, default=DEFAULT_EARLY_STOP_PATIENCE,
                         help=f"Stop if no improvement for N epochs; 0=disabled (default: {DEFAULT_EARLY_STOP_PATIENCE})")
    g_ckpt.add_argument("--target-loss", type=float, default=DEFAULT_TARGET_LOSS,
                         dest="target_loss",
                         help=f"Target loss cruise control (0=disabled). Damps LR as smoothed loss "
                              f"approaches this value so training holds steady. (default: {DEFAULT_TARGET_LOSS})")
    g_ckpt.add_argument("--target-loss-floor", type=float, default=DEFAULT_TARGET_LOSS_FLOOR,
                         dest="target_loss_floor",
                         help=f"Min LR multiplier at target loss; 0.01=1%% of scheduled LR "
                              f"(default: {DEFAULT_TARGET_LOSS_FLOOR})")
    g_ckpt.add_argument("--target-loss-warmup", type=int, default=DEFAULT_TARGET_LOSS_WARMUP,
                         dest="target_loss_warmup",
                         help=f"Min steps before cruise control engages "
                              f"(default: {DEFAULT_TARGET_LOSS_WARMUP})")
    g_ckpt.add_argument("--target-loss-smoothing", type=float, default=DEFAULT_TARGET_LOSS_SMOOTHING,
                         dest="target_loss_smoothing",
                         help=f"EMA beta for loss smoothing in cruise control; higher=smoother "
                              f"(default: {DEFAULT_TARGET_LOSS_SMOOTHING})")

    # -- Logging / TensorBoard -----------------------------------------------
    g_log = parser.add_argument_group("Logging / TensorBoard")
    g_log.add_argument("--log-dir", type=str, default=None, help="TensorBoard log directory (default: {output-dir}/runs)")
    g_log.add_argument("--log-every", type=int, default=DEFAULT_LOG_EVERY, help=f"Log basic metrics every N steps (default: {DEFAULT_LOG_EVERY})")
    g_log.add_argument("--log-heavy-every", type=int, default=DEFAULT_LOG_HEAVY_EVERY, help=f"Log per-layer gradient norms every N steps; 0 disables heavy logging (default: {DEFAULT_LOG_HEAVY_EVERY})")

    # -- Inline preprocessing (chained: preprocess then train) ---------------
    g_pre = parser.add_argument_group("Inline preprocessing")
    g_pre.add_argument("--preprocess", action="store_true", default=False,
                       help="Preprocess audio into tensors, then continue to training")
    g_pre.add_argument("--preprocess-only", action="store_true", default=False,
                       help="Run preprocessing and exit (do not train)")
    g_pre.add_argument("--audio-dir", type=str, default=None, help="Source audio directory (preprocessing)")
    g_pre.add_argument("--dataset-json", type=str, default=None, help="Labeled dataset JSON file (preprocessing)")
    g_pre.add_argument("--tensor-output", type=str, default=None, help="Output directory for .pt tensor files (preprocessing)")
    g_pre.add_argument("--max-duration", type=float, default=0, help="Max audio duration in seconds (0 = auto-detect from dataset, default: 0)")
    g_pre.add_argument("--normalize", type=str, default="none", choices=["none", "peak", "lufs"],
                        help="Audio normalization: none, peak (-1.0 dBFS), lufs (-14 LUFS). LUFS requires pyloudnorm (default: none)")
    g_pre.add_argument("--target-db", type=float, default=-1.0,
                        help="Peak normalization target in dBFS (used with --normalize peak, default: -1.0)")
    g_pre.add_argument("--target-lufs", type=float, default=-14.0,
                        help="LUFS normalization target (used with --normalize lufs, default: -14.0)")


def _add_train_args(parser: argparse.ArgumentParser) -> None:
    """Add arguments specific to the train subcommand."""
    g = parser.add_argument_group("Training (advanced)")
    g.add_argument("--cfg-ratio", type=float, default=DEFAULT_CFG_RATIO, help=f"CFG dropout probability (default: {DEFAULT_CFG_RATIO})")
    g.add_argument("--loss-weighting", type=str, default=DEFAULT_LOSS_WEIGHTING, choices=["none", "min_snr"],
                   help=f"Loss weighting: 'none' (flat MSE) or 'min_snr' (can yield better results on SFT/base, default: {DEFAULT_LOSS_WEIGHTING})")
    g.add_argument("--snr-gamma", type=float, default=DEFAULT_SNR_GAMMA,
                   help=f"Gamma for min-SNR weighting (default: {DEFAULT_SNR_GAMMA})")
    g.add_argument("--ignore-fisher-map", action="store_true", default=False,
                   help="Bypass auto-detection of fisher_map.json in --dataset-dir")
    g.add_argument("--dataset-repeats", "-R", type=int, default=DEFAULT_DATASET_REPEATS,
                   help=f"Global dataset repetition multiplier (1 = no repetition, default: {DEFAULT_DATASET_REPEATS})")


def _add_preprocess_subcommand_args(parser: argparse.ArgumentParser) -> None:
    """Add arguments for the top-level ``preprocess`` subcommand."""
    _add_model_args(parser)
    _add_device_args(parser)
    g = parser.add_argument_group("Preprocessing")
    g.add_argument("--audio-dir", "-i", "--input",
                   type=str, default=None, dest="audio_dir",
                   help="Source audio directory (scanned recursively)")
    g.add_argument("--dataset-json", type=str, default=None,
                   help="Labeled dataset JSON file (alternative to --audio-dir)")
    g.add_argument("--output", "-o", "--tensor-output",
                   type=str, default=None, dest="tensor_output",
                   help="Output directory for .pt tensor files")
    g.add_argument("--max-duration", type=float, default=0,
                   help="Max audio duration in seconds (0 = auto-detect, default: 0)")
    g.add_argument("--normalize", type=str, default="none",
                   choices=["none", "peak", "lufs"],
                   help="Audio normalization: none, peak, lufs (default: none)")
    g.add_argument("--target-db", type=float, default=-1.0,
                   help="Peak normalization target in dBFS (default: -1.0)")
    g.add_argument("--target-lufs", type=float, default=-14.0,
                   help="LUFS normalization target (default: -14.0)")


def _add_fisher_args(parser: argparse.ArgumentParser) -> None:
    """Add arguments for the ``analyze`` subcommand (Fisher + Spectral)."""
    g = parser.add_argument_group("PP++ / Fisher analysis")
    g.add_argument("--dataset-dir", "-d", type=str, required=True,
                   help="Directory containing preprocessed .pt files")
    g.add_argument("--rank", "-r", type=int, default=64,
                   help="Base LoRA rank (median target, default: 64)")
    g.add_argument("--rank-min", type=int, default=16,
                   help="Minimum adaptive rank (default: 16)")
    g.add_argument("--rank-max", type=int, default=128,
                   help="Maximum adaptive rank (default: 128)")
    g.add_argument("--timestep-focus", type=str, default="balanced",
                   help="Timestep focus: balanced (default), texture, structure, or low,high")
    g.add_argument("--runs", "--fisher-runs", type=int, default=None, dest="runs",
                   help="Number of estimation runs (default: auto from dataset size)")
    g.add_argument("--batches", "--fisher-batches", type=int, default=None, dest="batches",
                   help="Batches per run (default: auto from dataset size)")
    g.add_argument("--convergence-patience", type=int, default=5,
                   help="Early stop when ranking stable for N batches (default: 5)")
    g.add_argument("--output", type=str, default=None, dest="fisher_output",
                   help="Override output path for fisher_map.json")


def _add_audio_analyze_args(parser: argparse.ArgumentParser) -> None:
    """Add arguments for the ``audio-analyze`` subcommand."""
    g = parser.add_argument_group("Audio analysis")
    g.add_argument(
        "--input", "-i", type=str, required=True,
        help="Directory containing audio files to analyze (scanned recursively)",
    )
    g.add_argument(
        "--device", type=str, default="auto",
        help="Device: auto, cuda, cpu (default: auto)",
    )
    g.add_argument(
        "--policy", type=str, default="fill_missing",
        choices=["fill_missing", "overwrite_all"],
        help="Merge policy for existing sidecar fields (default: fill_missing)",
    )
    g.add_argument(
        "--mode", type=str, default="mid",
        choices=["faf", "mid", "sas"],
        help="Analysis quality: faf (fast, no Demucs), mid (default, ensemble), sas (deep multi-technique)",
    )
    g.add_argument(
        "--chunks", type=int, default=5,
        help="Number of analysis chunks for sas mode (default: 5)",
    )


def _add_captions_args(parser: argparse.ArgumentParser) -> None:
    """Add arguments for the ``captions`` subcommand."""
    g = parser.add_argument_group("Captions")
    g.add_argument(
        "--input", "-i", type=str, required=True,
        help="Directory containing audio files to caption (scanned recursively)",
    )
    g.add_argument(
        "--provider", type=str, default=None,
        choices=["gemini", "openai", "local_8-10gb", "local_16gb"],
        help="Caption provider: gemini, openai, local_8-10gb, local_16gb (default: from settings, or gemini)",
    )
    g.add_argument(
        "--ai-model", "--model", type=str, default=None, dest="ai_model",
        help="Override the AI model name (e.g. gemini-2.5-flash, gpt-4o)",
    )
    g.add_argument(
        "--policy", type=str, default="fill_missing",
        choices=["fill_missing", "overwrite_caption", "overwrite_all"],
        help="Merge policy for existing sidecars (default: fill_missing)",
    )
    g.add_argument(
        "--lyrics", action=argparse.BooleanOptionalAction, default=True,
        help="Fetch lyrics from Genius (requires GENIUS_API_TOKEN or settings). "
             "On by default; use --no-lyrics to skip",
    )
    g.add_argument(
        "--default-artist", type=str, default="",
        help="Default artist name for Genius lookups when filename has no artist",
    )
    g.add_argument(
        "--gemini-api-key", type=str, default=None,
        help="Gemini API key (overrides env/settings)",
    )
    g.add_argument(
        "--openai-api-key", type=str, default=None,
        help="OpenAI API key (overrides env/settings)",
    )
    g.add_argument(
        "--openai-base-url", type=str, default=None,
        help="Custom OpenAI-compatible base URL",
    )
    g.add_argument(
        "--genius-token", type=str, default=None,
        help="Genius API token (overrides env/settings)",
    )
    g.add_argument(
        "--google-search", action="store_true", default=False,
        help="Enable Grounding with Google Search for Gemini captions "
             "(lets the model look up track info online; adds per-query cost)",
    )


def _add_tags_args(parser: argparse.ArgumentParser) -> None:
    """Add arguments for the ``tags`` subcommand."""
    _dir_parent = argparse.ArgumentParser(add_help=False)
    _dir_parent.add_argument(
        "directory", type=str, nargs="?", default=None,
        help="Directory of audio/sidecar files",
    )
    _dir_parent.add_argument("--input", "-i", type=str, default=None,
                             dest="input_compat", help=argparse.SUPPRESS)

    sub = parser.add_subparsers(dest="tags_action")

    p_add = sub.add_parser("add", parents=[_dir_parent],
                           help="Add a trigger tag to sidecar files")
    p_add.add_argument("--tag", "-t", type=str, required=True,
                       help="Trigger tag to add")
    p_add.add_argument("--position", type=str, default="prepend",
                       choices=["prepend", "append", "replace"],
                       help="Tag placement (default: prepend)")

    p_rm = sub.add_parser("remove", parents=[_dir_parent],
                          help="Remove a trigger tag from sidecar files")
    p_rm.add_argument("--tag", "-t", type=str, required=True,
                      help="Trigger tag to remove")

    sub.add_parser("clear", parents=[_dir_parent],
                   help="Clear all trigger tags from sidecar files")

    sub.add_parser("list", parents=[_dir_parent],
                   help="List trigger tags from sidecar files")


def _add_settings_args(parser: argparse.ArgumentParser) -> None:
    """Add arguments for the ``settings`` subcommand."""
    sub = parser.add_subparsers(dest="settings_action")

    sub.add_parser("show", help="Display current settings")

    p_set = sub.add_parser("set", help="Set a setting value")
    p_set.add_argument("key", type=str, help="Setting key (e.g. checkpoint_dir, gemini_api_key)")
    p_set.add_argument("value", type=str, help="New value")

    p_clear = sub.add_parser("clear", help="Clear a setting (reset to default)")
    p_clear.add_argument("key", type=str, help="Setting key to clear")

    sub.add_parser("path", help="Print the settings file path")

    p_defaults = sub.add_parser("defaults", help="View or modify training defaults")
    p_defaults.add_argument("--set", nargs=2, metavar=("KEY", "VALUE"), action="append",
                            dest="default_overrides",
                            help="Override a training default (e.g. --set lr 1e-4)")
