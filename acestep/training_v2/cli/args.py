"""
Argparse construction for ACE-Step Training V2 CLI.

Contains ``build_root_parser`` and all ``_add_*`` argument-group helpers,
plus shared constants (``_DEFAULT_NUM_WORKERS``, ``VARIANT_DIR_MAP``).
"""

from __future__ import annotations

import argparse
import sys

# Windows uses spawn-based multiprocessing which breaks DataLoader workers
_DEFAULT_NUM_WORKERS = 0 if sys.platform == "win32" else 4

# Model variant -> checkpoint subdirectory mapping
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

    # -- fixed ---------------------------------------------------------------
    p_fixed = subparsers.add_parser(
        "fixed",
        help="Train a LoRA/LoKR (auto-detects turbo vs base/sft)",
        formatter_class=formatter_class,
    )
    _add_common_training_args(p_fixed)
    _add_fixed_args(p_fixed)

    # -- selective (not yet implemented -- hidden from CLI) -------------------

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

    # -- fisher --------------------------------------------------------------
    p_fisher = subparsers.add_parser(
        "fisher",
        help="Fisher + Spectral analysis for adaptive LoRA rank assignment",
        formatter_class=formatter_class,
    )
    _add_model_args(p_fisher)
    _add_device_args(p_fisher)
    _add_fisher_args(p_fisher)

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

    # -- build-dataset -------------------------------------------------------
    p_build = subparsers.add_parser(
        "build-dataset",
        help="Build dataset.json from a folder of audio + sidecar metadata files",
        formatter_class=formatter_class,
    )
    p_build.add_argument(
        "--input", "-i",
        type=str,
        required=True,
        help="Root directory containing audio files (scanned recursively)",
    )
    p_build.add_argument(
        "--tag",
        type=str,
        default="",
        help="Custom trigger tag applied to all samples (default: none)",
    )
    p_build.add_argument(
        "--tag-position",
        type=str,
        default="prepend",
        choices=["prepend", "append", "replace"],
        help="Tag placement in prompts (default: prepend)",
    )
    p_build.add_argument(
        "--genre-ratio",
        type=int,
        default=0,
        help="Percentage of samples that use genre instead of caption (0-100, default: 0)",
    )
    p_build.add_argument(
        "--name",
        type=str,
        default="local_dataset",
        help="Dataset name in metadata block (default: local_dataset)",
    )
    p_build.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output JSON path (default: <input>/dataset.json)",
    )

    return root


# ===========================================================================
# Argument groups
# ===========================================================================

def _add_model_args(parser: argparse.ArgumentParser) -> None:
    """Add --checkpoint-dir, --model-variant, and --base-model."""
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
        help=(
            "Model variant or subfolder name (default: turbo). "
            "Official: turbo, base, sft. "
            "For fine-tunes: use the exact folder name under checkpoint-dir."
        ),
    )
    g.add_argument(
        "--base-model",
        type=str,
        default=None,
        choices=["turbo", "base", "sft"],
        help=(
            "Base model a fine-tune was trained from (turbo/base/sft). "
            "Used to condition timestep sampling. Auto-detected for official models."
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
    """Add arguments shared by fixed / selective subcommands."""
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
        default=2 if _DEFAULT_NUM_WORKERS > 0 else 0,
        help="DataLoader prefetch factor (default: 2; 0 on Windows)",
    )
    g_data.add_argument(
        "--persistent-workers",
        action=argparse.BooleanOptionalAction,
        default=_DEFAULT_NUM_WORKERS > 0,
        help="Keep workers alive between epochs (default: True; False on Windows)",
    )

    # -- Training hyperparams ------------------------------------------------
    g_train = parser.add_argument_group("Training")
    g_train.add_argument("--lr", "--learning-rate", type=float, default=1e-4, dest="learning_rate", help="Initial learning rate (default: 1e-4)")
    g_train.add_argument("--batch-size", type=int, default=1, help="Training batch size (default: 1)")
    g_train.add_argument("--gradient-accumulation", type=int, default=4, help="Gradient accumulation steps (default: 4)")
    g_train.add_argument("--epochs", type=int, default=100, help="Maximum training epochs (default: 100)")
    g_train.add_argument("--warmup-steps", type=int, default=100, help="LR warmup steps (default: 100)")
    g_train.add_argument("--weight-decay", type=float, default=0.01, help="AdamW weight decay (default: 0.01)")
    g_train.add_argument("--max-grad-norm", type=float, default=1.0, help="Gradient clipping norm (default: 1.0)")
    g_train.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    g_train.add_argument("--chunk-duration", type=int, default=None,
                         help="Random latent chunk duration in seconds (default: disabled). "
                              "Recommended: 60. Extracts a random window each iteration for data "
                              "augmentation and VRAM savings. WARNING: values below 60s (e.g. 30) "
                              "may reduce training quality for full-length inference")
    g_train.add_argument("--shift", type=float, default=3.0, help="Noise schedule shift (turbo=3.0, base/sft=1.0)")
    g_train.add_argument("--num-inference-steps", type=int, default=8, help="Inference steps for timestep schedule (turbo=8, base/sft=50)")
    g_train.add_argument("--optimizer-type", type=str, default="adamw", choices=["adamw", "adamw8bit", "adafactor", "prodigy"], help="Optimizer (default: adamw)")
    g_train.add_argument("--scheduler-type", type=str, default="cosine", choices=["cosine", "cosine_restarts", "linear", "constant", "constant_with_warmup"], help="LR scheduler (default: cosine)")
    g_train.add_argument("--gradient-checkpointing", action=argparse.BooleanOptionalAction, default=True, help="Recompute activations to save VRAM (~40-60%% less, ~10-30%% slower). On by default; use --no-gradient-checkpointing to disable")
    g_train.add_argument("--offload-encoder", action=argparse.BooleanOptionalAction, default=False, help="Move encoder/VAE to CPU after setup (saves ~2-4GB VRAM)")

    # -- Adapter selection ---------------------------------------------------
    g_adapter = parser.add_argument_group("Adapter")
    g_adapter.add_argument("--adapter-type", type=str, default="lora", choices=["lora", "lokr"], help="Adapter type: lora (PEFT) or lokr (LyCORIS) (default: lora)")

    # -- LoRA hyperparams ---------------------------------------------------
    g_lora = parser.add_argument_group("LoRA (used when --adapter-type=lora)")
    g_lora.add_argument("--rank", "-r", type=int, default=64, help="LoRA rank (default: 64)")
    g_lora.add_argument("--alpha", type=int, default=128, help="LoRA alpha (default: 128)")
    g_lora.add_argument("--dropout", type=float, default=0.1, help="LoRA dropout (default: 0.1)")
    g_lora.add_argument("--target-modules", nargs="+", default=["q_proj", "k_proj", "v_proj", "o_proj"], help="Modules to apply adapter to")
    g_lora.add_argument("--bias", type=str, default="none", choices=["none", "all", "lora_only"], help="Bias training mode (default: none)")
    g_lora.add_argument("--attention-type", type=str, default="both", choices=["self", "cross", "both"], help="Attention layers to target (default: both)")
    g_lora.add_argument("--self-target-modules", nargs="+", default=None, help="Projections for self-attention only (used when --attention-type=both)")
    g_lora.add_argument("--cross-target-modules", nargs="+", default=None, help="Projections for cross-attention only (used when --attention-type=both)")
    g_lora.add_argument("--target-mlp", action=argparse.BooleanOptionalAction, default=False, help="Also target MLP/FFN layers (gate_proj, up_proj, down_proj)")

    # -- LoKR hyperparams ---------------------------------------------------
    g_lokr = parser.add_argument_group("LoKR (used when --adapter-type=lokr)")
    g_lokr.add_argument("--lokr-linear-dim", type=int, default=64, help="LoKR linear dimension (default: 64)")
    g_lokr.add_argument("--lokr-linear-alpha", type=int, default=128, help="LoKR linear alpha (default: 128)")
    g_lokr.add_argument("--lokr-factor", type=int, default=-1, help="LoKR factor; -1 for auto (default: -1)")
    g_lokr.add_argument("--lokr-decompose-both", action="store_true", default=False, help="Decompose both Kronecker factors")
    g_lokr.add_argument("--lokr-use-tucker", action="store_true", default=False, help="Use Tucker decomposition")
    g_lokr.add_argument("--lokr-use-scalar", action="store_true", default=False, help="Use scalar scaling")
    g_lokr.add_argument("--lokr-weight-decompose", action="store_true", default=False, help="Enable DoRA-style weight decomposition")

    # -- Checkpointing -------------------------------------------------------
    g_ckpt = parser.add_argument_group("Checkpointing")
    g_ckpt.add_argument("--output-dir", type=str, required=True, help="Output directory for LoRA weights")
    g_ckpt.add_argument("--save-every", type=int, default=10, help="Save checkpoint every N epochs (default: 10)")
    g_ckpt.add_argument("--resume-from", type=str, default=None, help="Path to checkpoint dir to resume from")
    g_ckpt.add_argument("--strict-resume", action=argparse.BooleanOptionalAction, default=True,
                         help="Abort on config mismatch or failed state restore during resume (default: True)")
    g_ckpt.add_argument("--run-name", type=str, default=None,
                         help="Name for this training run (used for output dir, TB logs). Auto-generated if omitted")
    g_ckpt.add_argument("--save-best", action=argparse.BooleanOptionalAction, default=True,
                         help="Auto-save best model by smoothed loss (default: True)")
    g_ckpt.add_argument("--save-best-after", type=int, default=200,
                         help="Epoch to start best-model tracking (default: 200)")
    g_ckpt.add_argument("--early-stop-patience", type=int, default=0,
                         help="Stop if no improvement for N epochs; 0=disabled (default: 0)")

    # -- Logging / TensorBoard -----------------------------------------------
    g_log = parser.add_argument_group("Logging / TensorBoard")
    g_log.add_argument("--log-dir", type=str, default=None, help="TensorBoard log directory (default: {output-dir}/runs)")
    g_log.add_argument("--log-every", type=int, default=10, help="Log basic metrics every N steps (default: 10)")
    g_log.add_argument("--log-heavy-every", type=int, default=50, help="Log per-layer gradient norms every N steps; 0 disables heavy logging (default: 50)")

    # -- Preprocessing -------------------------------------------------------
    g_pre = parser.add_argument_group("Preprocessing")
    g_pre.add_argument("--preprocess", action="store_true", default=False, help="Run preprocessing before training")
    g_pre.add_argument("--audio-dir", type=str, default=None, help="Source audio directory (preprocessing)")
    g_pre.add_argument("--dataset-json", type=str, default=None, help="Labeled dataset JSON file (preprocessing)")
    g_pre.add_argument("--tensor-output", type=str, default=None, help="Output directory for .pt tensor files (preprocessing)")
    g_pre.add_argument("--max-duration", type=float, default=0, help="Max audio duration in seconds (0 = auto-detect from dataset, default: 0)")
    g_pre.add_argument("--normalize", type=str, default="none", choices=["none", "peak", "lufs"],
                        help="Audio normalization: none, peak (-1.0 dBFS), lufs (-14 LUFS). LUFS requires pyloudnorm (default: none)")


def _add_fixed_args(parser: argparse.ArgumentParser) -> None:
    """Add arguments specific to the fixed/selective subcommands."""
    g = parser.add_argument_group("Corrected training")
    g.add_argument("--cfg-ratio", type=float, default=0.15, help="CFG dropout probability (default: 0.15)")
    g.add_argument("--loss-weighting", type=str, default="none", choices=["none", "min_snr"],
                   help="Loss weighting: 'none' (flat MSE) or 'min_snr' (can yield better results on SFT/base)")
    g.add_argument("--snr-gamma", type=float, default=5.0,
                   help="Gamma for min-SNR weighting (default: 5.0)")
    g.add_argument("--ignore-fisher-map", action="store_true", default=False,
                   help="Bypass auto-detection of fisher_map.json in --dataset-dir")


def _add_selective_args(parser: argparse.ArgumentParser) -> None:
    """Add arguments specific to the selective subcommand."""
    g = parser.add_argument_group("Selective / estimation")
    g.add_argument("--module-config", type=str, default=None, help="Path to JSON module config from estimation")
    g.add_argument("--auto-estimate", action="store_true", default=False, help="Run estimation inline before training")
    _add_estimation_args(parser)


def _add_estimation_args(parser: argparse.ArgumentParser) -> None:
    """Add arguments shared by estimate and selective subcommands."""
    g = parser.add_argument_group("Estimation")
    g.add_argument("--estimate-batches", type=int, default=None, help="Number of batches for estimation (default: auto from GPU)")
    g.add_argument("--top-k", type=int, default=16, help="Number of top modules to select (default: 16)")
    g.add_argument("--granularity", type=str, default="module", choices=["layer", "module"], help="Estimation granularity (default: module)")
    g.add_argument("--output", type=str, default=None, dest="estimate_output", help="Path to write module config JSON (estimate only)")


def _add_fisher_args(parser: argparse.ArgumentParser) -> None:
    """Add arguments for the ``fisher`` subcommand."""
    g = parser.add_argument_group("Fisher analysis")
    g.add_argument("--dataset-dir", type=str, required=True,
                   help="Directory containing preprocessed .pt files")
    g.add_argument("--rank", "-r", type=int, default=64,
                   help="Base LoRA rank (median target, default: 64)")
    g.add_argument("--rank-min", type=int, default=16,
                   help="Minimum adaptive rank (default: 16)")
    g.add_argument("--rank-max", type=int, default=128,
                   help="Maximum adaptive rank (default: 128)")
    g.add_argument("--timestep-focus", type=str, default="balanced",
                   help="Timestep focus: balanced (default), texture, structure, or low,high")
    g.add_argument("--fisher-runs", type=int, default=None,
                   help="Number of estimation runs (default: auto from dataset size)")
    g.add_argument("--fisher-batches", type=int, default=None,
                   help="Batches per run (default: auto from dataset size)")
    g.add_argument("--convergence-patience", type=int, default=5,
                   help="Early stop when ranking stable for N batches (default: 5)")
    g.add_argument("--output", type=str, default=None, dest="fisher_output",
                   help="Override output path for fisher_map.json")
