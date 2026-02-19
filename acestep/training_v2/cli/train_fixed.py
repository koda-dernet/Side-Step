"""
fixed subcommand -- Variant-aware adapter training.

Auto-detects the model variant and selects the appropriate strategy:

  **Turbo**: discrete 8-step timestep sampling, no CFG dropout.
  **Base / SFT**: continuous logit-normal timestep sampling via
  ``sample_timesteps()`` + CFG dropout (``cfg_ratio=0.15``).

Both paths share the same data pipeline (``PreprocessedDataModule``)
and adapter utilities (``inject_lora_into_dit``, ``save_lora_weights``).
"""

from __future__ import annotations

import argparse
import gc
import json
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

from acestep.training_v2.cli.common import build_configs
from acestep.training_v2.model_loader import load_decoder_for_training
from acestep.training_v2.trainer_fixed import FixedLoRATrainer


def _cleanup_gpu() -> None:
    """Release GPU memory so the process can safely reuse it."""
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


def _safe_slug(text: str) -> str:
    """Convert arbitrary text to a filesystem-safe slug."""
    clean = re.sub(r"[^A-Za-z0-9._-]+", "-", str(text).strip())
    clean = clean.strip("-._")
    return clean or "session"


def _build_session_name(train_cfg: Any) -> str:
    """Create a stable session name for artifacts/logs."""
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    variant = _safe_slug(getattr(train_cfg, "model_variant", "model"))
    adapter = _safe_slug(getattr(train_cfg, "adapter_type", "adapter"))
    return f"{stamp}_{variant}_{adapter}"


def _session_artifact_paths(train_cfg: Any, session_name: str) -> tuple[Path, Path]:
    """Return ``(config_txt_path, ui_log_path)`` for this session."""
    output_dir = Path(getattr(train_cfg, "output_dir", "") or ".").expanduser()
    session_dir = output_dir / "session_logs"
    return (
        session_dir / f"{session_name}_config.txt",
        session_dir / f"{session_name}_ui.log",
    )


def _append_session_log(path: Path, msg: str) -> None:
    """Append one timestamped line to the session UI log."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} {msg}\n")


def _write_session_config_dump(
    adapter_cfg: Any,
    train_cfg: Any,
    path: Path,
    session_name: str,
    attention_backend: str,
) -> None:
    """Write a human-readable session config snapshot to a TXT file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    adapter_payload = adapter_cfg.to_dict() if hasattr(adapter_cfg, "to_dict") else vars(adapter_cfg)
    train_payload = train_cfg.to_dict() if hasattr(train_cfg, "to_dict") else vars(train_cfg)
    body = (
        f"Session: {session_name}\n"
        f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        f"Attention backend: {attention_backend}\n\n"
        "[Adapter Configuration]\n"
        f"{json.dumps(adapter_payload, indent=2, sort_keys=True, default=str)}\n\n"
        "[Training Configuration]\n"
        f"{json.dumps(train_payload, indent=2, sort_keys=True, default=str)}\n"
    )
    path.write_text(body, encoding="utf-8")


def run_fixed(args: argparse.Namespace) -> int:
    """Execute the fixed (corrected) training subcommand.

    Returns 0 on success, non-zero on failure.
    """
    import torch

    # -- UI setup -------------------------------------------------------------
    from acestep.training_v2.ui import set_plain_mode
    from acestep.training_v2.ui.banner import show_banner
    from acestep.training_v2.ui.config_panel import show_config, confirm_start
    from acestep.training_v2.ui.errors import handle_error, show_info
    from acestep.training_v2.ui.progress import track_training
    from acestep.training_v2.ui.summary import show_summary
    from acestep.training_v2.ui.tensorboard_launcher import (
        launch_tensorboard_background,
        should_launch_tensorboard,
    )

    if getattr(args, "plain", False):
        set_plain_mode(True)

    # -- Matmul precision (matches handler.initialize_service behaviour) ------
    torch.set_float32_matmul_precision("medium")

    # -- Build V2 config objects from CLI args --------------------------------
    adapter_cfg, train_cfg = build_configs(args)

    # -- Optional dependency preflight ---------------------------------------
    from acestep.training_v2.ui.dependency_check import (
        ensure_optional_dependencies,
        required_training_optionals,
    )
    ensure_optional_dependencies(
        required_training_optionals(train_cfg),
        interactive=sys.stdin.isatty(),
        allow_install_prompt=not getattr(args, "yes", False),
    )

    # -- Banner (skip if wizard already showed one) ---------------------------
    if not getattr(args, "_from_wizard", False):
        show_banner(
            subcommand="fixed",
            device=train_cfg.device,
            precision=train_cfg.precision,
        )

    # -- Config summary & confirmation (always shown) -----------------------
    show_config(adapter_cfg, train_cfg, subcommand="fixed")
    skip_confirm = getattr(args, "yes", False)
    if not confirm_start(skip=skip_confirm):
        return 0
    auto_launch_tensorboard = should_launch_tensorboard(
        train_cfg.effective_log_dir,
        default=True,
        skip_prompt=skip_confirm,
        interactive=sys.stdin.isatty(),
    )

    session_name = _build_session_name(train_cfg)
    session_cfg_path: Path | None = None
    session_ui_log_path: Path | None = None
    attention_backend = "unknown"
    try:
        cfg_path, ui_path = _session_artifact_paths(train_cfg, session_name)
        _write_session_config_dump(
            adapter_cfg,
            train_cfg,
            cfg_path,
            session_name=session_name,
            attention_backend="pending",
        )
        _append_session_log(ui_path, f"[INFO] Session created: {session_name}")
        session_cfg_path = cfg_path
        session_ui_log_path = ui_path
        show_info(f"Session: {session_name}")
        show_info(f"Session config: {session_cfg_path}")
        show_info(f"Session UI log: {session_ui_log_path}")
    except Exception as exc:
        show_info(f"Session artifact setup skipped: {exc}")

    if auto_launch_tensorboard:
        launched, launch_msg = launch_tensorboard_background(train_cfg.effective_log_dir)
        show_info(launch_msg)
        if session_ui_log_path is not None:
            level = "[OK]" if launched else "[WARN]"
            _append_session_log(session_ui_log_path, f"{level} {launch_msg}")

    model = None
    trainer = None
    try:
        # -- Load model -------------------------------------------------------
        try:
            show_info(f"Loading model (variant={train_cfg.model_variant}, device={train_cfg.device})")
            if session_ui_log_path is not None:
                _append_session_log(
                    session_ui_log_path,
                    f"[INFO] Loading model (variant={train_cfg.model_variant}, device={train_cfg.device})",
                )
            model = load_decoder_for_training(
                checkpoint_dir=train_cfg.checkpoint_dir,
                variant=train_cfg.model_variant,
                device=train_cfg.device,
                precision=train_cfg.precision,
            )
            attention_backend = str(getattr(model, "_side_step_attn_backend", "unknown"))
            if session_cfg_path is not None:
                _write_session_config_dump(
                    adapter_cfg,
                    train_cfg,
                    session_cfg_path,
                    session_name=session_name,
                    attention_backend=attention_backend,
                )
            if session_ui_log_path is not None:
                _append_session_log(
                    session_ui_log_path,
                    f"[INFO] Attention backend selected: {attention_backend}",
                )
        except Exception as exc:
            if session_ui_log_path is not None:
                _append_session_log(session_ui_log_path, f"[FAIL] Model loading failed: {exc}")
            handle_error(exc, context="Model loading", show_traceback=True)
            return 1

        # -- Train ------------------------------------------------------------
        try:
            trainer = FixedLoRATrainer(model, adapter_cfg, train_cfg)

            stats = track_training(
                training_iter=trainer.train(),
                max_epochs=train_cfg.max_epochs,
                device=train_cfg.device,
                attention_backend=attention_backend,
                session_log_path=str(session_ui_log_path) if session_ui_log_path is not None else None,
                session_name=session_name,
            )

            # -- Summary ------------------------------------------------------
            show_summary(
                stats=stats,
                output_dir=train_cfg.output_dir,
                log_dir=str(train_cfg.effective_log_dir),
            )
            if session_ui_log_path is not None:
                _append_session_log(
                    session_ui_log_path,
                    f"[OK] Training summary complete (steps={stats.current_step})",
                )
        except KeyboardInterrupt:
            if session_ui_log_path is not None:
                _append_session_log(session_ui_log_path, "[INFO] Training interrupted by user")
            show_info("Training interrupted by user (Ctrl+C)")
            return 130
        except Exception as exc:
            if session_ui_log_path is not None:
                _append_session_log(session_ui_log_path, f"[FAIL] Training failed: {exc}")
            handle_error(exc, context="Training", show_traceback=True)
            return 1

        return 0
    finally:
        # Explicitly release GPU memory so the session loop can reuse it.
        del trainer
        del model
        _cleanup_gpu()
