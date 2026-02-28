"""
Argparse construction for Side-Step CLI.

Contains ``build_root_parser`` and all ``_add_*`` argument-group helpers,
plus shared constants (``_DEFAULT_NUM_WORKERS``, ``VARIANT_DIR_MAP``).
"""

from __future__ import annotations

import argparse
import sys

from sidestep_engine.training_defaults import (
    DEFAULT_EPOCHS,
    DEFAULT_LEARNING_RATE,
    DEFAULT_OPTIMIZER_TYPE,
    DEFAULT_SAVE_EVERY,
)

from sidestep_engine.core.constants import (
    DEFAULT_NUM_WORKERS as _DEFAULT_NUM_WORKERS,
    VARIANT_DIR_MAP,
)


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
        default="turbo",
        dest="model_variant",
        metavar="MODEL",
        help=(
            "Model variant or subfolder name (default: turbo). "
            "Official: turbo, base, sft. "
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
    g_train.add_argument("--lr", "--learning-rate", "-l", type=float, default=DEFAULT_LEARNING_RATE, dest="learning_rate", help=f"Initial learning rate (default: {DEFAULT_LEARNING_RATE:g})")
    g_train.add_argument("--batch-size", "-b", type=int, default=1, help="Training batch size (default: 1)")
    g_train.add_argument("--gradient-accumulation", "-g", type=int, default=4, help="Gradient accumulation steps (default: 4)")
    g_train.add_argument("--epochs", "-e", type=int, default=DEFAULT_EPOCHS, help=f"Maximum training epochs (default: {DEFAULT_EPOCHS})")
    g_train.add_argument("--warmup-steps", type=int, default=100, help="LR warmup steps (default: 100)")
    g_train.add_argument("--weight-decay", type=float, default=0.01, help="AdamW weight decay (default: 0.01)")
    g_train.add_argument("--max-grad-norm", type=float, default=1.0, help="Gradient clipping norm (default: 1.0)")
    g_train.add_argument("--seed", "-s", type=int, default=42, help="Random seed (default: 42)")
    g_train.add_argument("--chunk-duration", type=int, default=None,
                         help="Random latent chunk duration in seconds (default: disabled). "
                              "Recommended: 60. Extracts a random window each iteration for data "
                              "augmentation and VRAM savings. WARNING: values below 60s (e.g. 30) "
                              "may reduce training quality for full-length inference")
    g_train.add_argument("--chunk-decay-every", type=int, default=10,
                         help="Epoch interval for halving chunk coverage histogram; 0 disables decay (default: 10)")
    g_train.add_argument("--max-steps", "-m", type=int, default=0,
                         help="Maximum optimizer steps; 0 = use epochs only (default: 0)")
    g_train.add_argument("--shift", type=float, default=None, help=argparse.SUPPRESS)
    g_train.add_argument("--num-inference-steps", type=int, default=None, help=argparse.SUPPRESS)
    g_train.add_argument("--optimizer-type", type=str, default=DEFAULT_OPTIMIZER_TYPE, choices=["adamw", "adamw8bit", "adafactor", "prodigy"], help=f"Optimizer (default: {DEFAULT_OPTIMIZER_TYPE})")
    g_train.add_argument("--scheduler-type", type=str, default="cosine", choices=["cosine", "cosine_restarts", "linear", "constant", "constant_with_warmup", "custom"], help="LR scheduler (default: cosine)")
    g_train.add_argument("--scheduler-formula", type=str, default="", help="Custom LR formula (Python math expression). Only used with --scheduler-type custom")
    g_train.add_argument("--gradient-checkpointing", action=argparse.BooleanOptionalAction, default=True, help="Recompute activations to save VRAM (~40-60%% less, ~10-30%% slower). On by default; use --no-gradient-checkpointing to disable")
    g_train.add_argument("--gradient-checkpointing-ratio", type=float, default=1.0, help="Fraction of decoder layers to checkpoint (0.0=none, 0.5=half, 1.0=all). Only applies when --gradient-checkpointing is on (default: 1.0)")
    g_train.add_argument("--offload-encoder", action=argparse.BooleanOptionalAction, default=True, help="Move encoder/VAE to CPU after setup (saves ~2-4GB VRAM). On by default; use --no-offload-encoder to disable")

    # -- All the Levers (experimental enhancements) -------------------------
    g_levers = parser.add_argument_group("All the Levers (experimental)")
    g_levers.add_argument("--ema-decay", type=float, default=0.0, help="EMA decay for adapter weights (0=off, 0.9999=typical)")
    g_levers.add_argument("--val-split", type=float, default=0.0, help="Validation holdout fraction (0=off, 0.1=10%%)")
    g_levers.add_argument("--adaptive-timestep-ratio", type=float, default=0.0, help="Adaptive timestep sampling ratio (0=off, 0.3=recommended). Base/SFT only")
    g_levers.add_argument("--warmup-start-factor", type=float, default=0.1, help="LR warmup starts at base_lr * this (default: 0.1)")
    g_levers.add_argument("--cosine-eta-min-ratio", type=float, default=0.01, help="Cosine scheduler decays LR to base_lr * this (default: 0.01)")
    g_levers.add_argument("--cosine-restarts-count", type=int, default=4, help="Number of cosine restart cycles (default: 4)")
    g_levers.add_argument("--save-best-every-n-steps", type=int, default=0, help="Step-level best-model check interval (0=epoch only)")
    g_levers.add_argument("--timestep-mu", type=float, default=None, help="Override logit-normal timestep mean (default: from model config, typically -0.4)")
    g_levers.add_argument("--timestep-sigma", type=float, default=None, help="Override logit-normal timestep sigma (default: from model config, typically 1.0)")

    # -- Adapter selection ---------------------------------------------------
    g_adapter = parser.add_argument_group("Adapter")
    g_adapter.add_argument("--adapter", "-a", "--adapter-type", type=str, default="lora", dest="adapter_type", choices=["lora", "dora", "lokr", "loha", "oft"], help="Adapter type: lora, dora, lokr, loha, or oft (default: lora)")

    # -- LoRA hyperparams ---------------------------------------------------
    g_lora = parser.add_argument_group("LoRA (used when --adapter=lora)")
    g_lora.add_argument("--rank", "-r", type=int, default=64, help="LoRA rank (default: 64)")
    g_lora.add_argument("--alpha", type=int, default=128, help="LoRA alpha (default: 128)")
    g_lora.add_argument("--dropout", type=float, default=0.1, help="LoRA dropout (default: 0.1)")
    g_lora.add_argument("--target-modules", nargs="+", default=["q_proj", "k_proj", "v_proj", "o_proj"], help="Modules to apply adapter to")
    g_lora.add_argument("--bias", type=str, default="none", choices=["none", "all", "lora_only"], help="Bias training mode (default: none)")
    g_lora.add_argument("--attention-type", type=str, default="both", choices=["self", "cross", "both"], help="Attention layers to target (default: both)")
    g_lora.add_argument("--self-target-modules", nargs="+", default=None, help="Projections for self-attention only (used when --attention-type=both)")
    g_lora.add_argument("--cross-target-modules", nargs="+", default=None, help="Projections for cross-attention only (used when --attention-type=both)")
    g_lora.add_argument("--target-mlp", action=argparse.BooleanOptionalAction, default=True, help="Target MLP/FFN layers (gate_proj, up_proj, down_proj). On by default; use --no-target-mlp to disable")

    # -- LoKR hyperparams ---------------------------------------------------
    g_lokr = parser.add_argument_group("LoKR (used when --adapter=lokr)")
    g_lokr.add_argument("--lokr-linear-dim", type=int, default=64, help="LoKR linear dimension (default: 64)")
    g_lokr.add_argument("--lokr-linear-alpha", type=int, default=128, help="LoKR linear alpha (default: 128)")
    g_lokr.add_argument("--lokr-factor", type=int, default=-1, help="LoKR factor; -1 for auto (default: -1)")
    g_lokr.add_argument("--lokr-decompose-both", action="store_true", default=False, help="Decompose both Kronecker factors")
    g_lokr.add_argument("--lokr-use-tucker", action="store_true", default=False, help="Use Tucker decomposition")
    g_lokr.add_argument("--lokr-use-scalar", action="store_true", default=False, help="Use scalar scaling")
    g_lokr.add_argument("--lokr-weight-decompose", action="store_true", default=False, help="Enable DoRA-style weight decomposition")

    # -- LoHA hyperparams ---------------------------------------------------
    g_loha = parser.add_argument_group("LoHA (used when --adapter=loha)")
    g_loha.add_argument("--loha-linear-dim", type=int, default=64, help="LoHA linear dimension (default: 64)")
    g_loha.add_argument("--loha-linear-alpha", type=int, default=128, help="LoHA linear alpha (default: 128)")
    g_loha.add_argument("--loha-factor", type=int, default=-1, help="LoHA factor; -1 for auto (default: -1)")
    g_loha.add_argument("--loha-use-tucker", action="store_true", default=False, help="Use Tucker decomposition")
    g_loha.add_argument("--loha-use-scalar", action="store_true", default=False, help="Use scalar scaling")

    # -- OFT hyperparams (experimental) -------------------------------------
    g_oft = parser.add_argument_group("OFT [Experimental] (used when --adapter=oft)")
    g_oft.add_argument("--oft-block-size", type=int, default=64, help="OFT block size (default: 64)")
    g_oft.add_argument("--oft-coft", action="store_true", default=False, help="Enable constrained OFT (Cayley projection)")
    g_oft.add_argument("--oft-eps", type=float, default=6e-5, help="OFT epsilon for numerical stability (default: 6e-5)")

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
    g_ckpt.add_argument("--save-best-after", type=int, default=200,
                         help="Epoch to start best-model tracking (default: 200)")
    g_ckpt.add_argument("--early-stop-patience", type=int, default=0,
                         help="Stop if no improvement for N epochs; 0=disabled (default: 0)")

    # -- Logging / TensorBoard -----------------------------------------------
    g_log = parser.add_argument_group("Logging / TensorBoard")
    g_log.add_argument("--log-dir", type=str, default=None, help="TensorBoard log directory (default: {output-dir}/runs)")
    g_log.add_argument("--log-every", type=int, default=10, help="Log basic metrics every N steps (default: 10)")
    g_log.add_argument("--log-heavy-every", type=int, default=50, help="Log per-layer gradient norms every N steps; 0 disables heavy logging (default: 50)")

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
    g.add_argument("--cfg-ratio", type=float, default=0.15, help="CFG dropout probability (default: 0.15)")
    g.add_argument("--loss-weighting", type=str, default="none", choices=["none", "min_snr"],
                   help="Loss weighting: 'none' (flat MSE) or 'min_snr' (can yield better results on SFT/base)")
    g.add_argument("--snr-gamma", type=float, default=5.0,
                   help="Gamma for min-SNR weighting (default: 5.0)")
    g.add_argument("--ignore-fisher-map", action="store_true", default=False,
                   help="Bypass auto-detection of fisher_map.json in --dataset-dir")
    g.add_argument("--dataset-repeats", "-R", type=int, default=1,
                   help="Global dataset repetition multiplier (1 = no repetition, default: 1)")


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


def _add_captions_args(parser: argparse.ArgumentParser) -> None:
    """Add arguments for the ``captions`` subcommand."""
    g = parser.add_argument_group("Captions")
    g.add_argument(
        "--input", "-i", type=str, required=True,
        help="Directory containing audio files to caption (scanned recursively)",
    )
    g.add_argument(
        "--provider", type=str, default=None,
        choices=["gemini", "openai"],
        help="Caption provider (default: from settings, or gemini)",
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
