#!/usr/bin/env python3
"""Side-Step -- CLI Entry Point

Usage:
    sidestep                       # Interactive wizard
    sidestep gui                   # Launch GUI
    sidestep <subcommand> [args]   # Direct CLI

Subcommands:
    train            Train an adapter (LoRA, DoRA, LoKR, LoHA, OFT)
    preprocess       Preprocess audio into tensors (two-pass pipeline)
    analyze          PP++ / Fisher analysis for adaptive LoRA rank
    audio-analyze    Local offline audio analysis (BPM, key, time signature)
    dataset          Build dataset.json from audio + sidecar metadata
    captions         Generate AI captions + fetch lyrics for sidecars
    tags             Bulk sidecar tag operations
    settings         View or modify persistent settings
    history          List past training runs
    gui              Launch the web GUI

Examples:
    sidestep train -c ./checkpoints -M turbo \\
        -d ./preprocessed_tensors/jazz -o ./lora_output/jazz

    sidestep captions -i ./my_audio --provider gemini --policy fill_missing

    sidestep --help
"""

from __future__ import annotations

import gc
import logging
import os
import sys

# ---------------------------------------------------------------------------
# Logging setup (before any library imports that might configure logging)
# ---------------------------------------------------------------------------

_log_formatter = logging.Formatter(
    "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

# Console (same as before)
_console_handler = logging.StreamHandler()
_console_handler.setLevel(logging.INFO)
_console_handler.setFormatter(_log_formatter)

# File (captures DEBUG+ including tracebacks)
# Guard against read-only working directories (e.g. some Windows setups)
try:
    _file_handler = logging.FileHandler("sidestep.log", mode="a", encoding="utf-8")
    _file_handler.setLevel(logging.DEBUG)
    _file_handler.setFormatter(_log_formatter)
    _log_handlers = [_console_handler, _file_handler]
except OSError:
    _log_handlers = [_console_handler]

logging.basicConfig(level=logging.DEBUG, handlers=_log_handlers)
logger = logging.getLogger("train")

from sidestep_engine._compat import install_torchao_warning_filter
install_torchao_warning_filter()


# ---------------------------------------------------------------------------
# Deprecation shim -- translates legacy CLI syntax to Beta 1 equivalents
# ---------------------------------------------------------------------------

_DEPRECATED_SUBCOMMANDS: dict[str, str] = {
    "fixed": "train",
    "build-dataset": "dataset",
    "fisher": "analyze",
}


def _apply_deprecation_shim() -> None:
    """Rewrite sys.argv in-place when legacy command names are detected.

    Handles:
    - Old subcommand names (fixed -> train, build-dataset -> dataset, fisher -> analyze)
    - Root-level --gui flag -> gui subcommand
    """
    args = sys.argv[1:]
    if not args:
        return

    # --gui as root flag -> gui subcommand
    if "--gui" in args:
        new_args = ["gui"]
        for i, a in enumerate(args):
            if a == "--gui":
                continue
            # --port VALUE after --gui -> becomes gui --port VALUE
            new_args.append(a)
        print(
            "[DEPRECATED] 'sidestep --gui' was replaced by 'sidestep gui' in Beta 1. "
            "Translating automatically.",
            file=sys.stderr,
        )
        sys.argv[1:] = new_args
        args = new_args

    # Deprecated subcommand names
    for i, a in enumerate(args):
        if a in _DEPRECATED_SUBCOMMANDS:
            new_name = _DEPRECATED_SUBCOMMANDS[a]
            print(
                f"[DEPRECATED] 'sidestep {a}' was renamed to 'sidestep {new_name}' "
                f"in Beta 1. Translating automatically.",
                file=sys.stderr,
            )
            sys.argv[i + 1] = new_name
            break
        if a.startswith("-"):
            continue
        break  # first positional that isn't deprecated -> stop scanning


_KNOWN_SUBCOMMANDS = {
    "train", "preprocess", "analyze", "audio-analyze", "dataset",
    "captions", "tags", "settings", "history", "export", "gui",
}


def _has_subcommand() -> bool:
    """Check if sys.argv contains a recognized subcommand or --help."""
    args = sys.argv[1:]
    if "--help" in args or "-h" in args:
        return True
    return bool(_KNOWN_SUBCOMMANDS & set(args))


def _cleanup_gpu() -> None:
    """Release GPU memory between session-loop iterations."""
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        elif hasattr(torch, 'mps') and torch.mps.is_available():
            torch.mps.empty_cache()
        elif hasattr(torch, 'xpu') and torch.xpu.is_available():
            torch.xpu.empty_cache()
    except ImportError:
        pass


def _validate_required_args(args, fields) -> bool:
    """Check that required fields are set (not None/empty). Returns True if OK."""
    for dest, flag in fields:
        val = getattr(args, dest, None)
        if not val:
            print(f"[FAIL] {flag} is required (provide via CLI or --config)", file=sys.stderr)
            return False
    return True


def _auto_resolve_checkpoint_dir(args) -> None:
    """Fill in checkpoint_dir from settings when not provided on CLI."""
    if getattr(args, "checkpoint_dir", None):
        return
    try:
        from sidestep_engine.settings import load_settings
        settings = load_settings()
        if settings:
            ckpt = settings.get("checkpoint_dir")
            if ckpt:
                args.checkpoint_dir = ckpt
                print(f"[INFO] Using checkpoint_dir from settings: {ckpt}")
    except Exception:
        pass


def _auto_resolve_shift_steps(args) -> None:
    """Set --shift and --num-inference-steps from model variant when not explicit.

    Uses the canonical ``is_turbo()`` detection from ``core.constants`` so
    that CLI, Wizard, and GUI all resolve unknown variant names identically.
    """
    from sidestep_engine.core.constants import is_turbo as _is_turbo_check

    # Build a lightweight params dict for is_turbo().
    # Only include num_inference_steps if the user explicitly provided it
    # (not None) so the name-based detection takes priority for unknowns.
    params: dict = {}
    for key in ("model_variant", "base_model", "num_inference_steps"):
        val = getattr(args, key, None)
        if val is not None:
            params[key] = val

    turbo = _is_turbo_check(params)
    variant = getattr(args, "model_variant", "turbo") or "turbo"

    shift = getattr(args, "shift", None)
    steps = getattr(args, "num_inference_steps", None)

    if shift is None:
        args.shift = 3.0 if turbo else 1.0
    if steps is None:
        args.num_inference_steps = 8 if turbo else 50

    if shift is None or steps is None:
        print(f"[INFO] Using shift={args.shift}, steps={args.num_inference_steps} "
              f"for {variant} model")


def _dispatch(args) -> int:
    """Route a parsed argparse.Namespace to the correct subcommand runner.

    Returns an int exit code (0 = success).
    """
    from sidestep_engine.cli.common import validate_paths

    # -- Config file merge (JSON values fill unset CLI args) ----------------
    if getattr(args, "config", None):
        from sidestep_engine.cli.config_builder import (
            _apply_config_file,
            _populate_defaults_cache,
        )
        try:
            _populate_defaults_cache()
            _apply_config_file(args)
        except Exception as exc:
            print(f"[FAIL] Could not load --config: {exc}", file=sys.stderr)
            return 1

    # -- Preprocessing (chains into training unless --preprocess-only) ------
    if getattr(args, "preprocess", False) or getattr(args, "preprocess_only", False):
        # Auto-default dataset-dir to tensor-output for chained preprocess->train
        if not getattr(args, "dataset_dir", None) and getattr(args, "tensor_output", None):
            args.dataset_dir = args.tensor_output
        rc = _run_preprocess(args)
        if rc != 0:
            return rc
        if getattr(args, "preprocess_only", False):
            return 0
        # Fall through to training

    sub = args.subcommand

    # Subcommands with their own validation (no path check needed)
    if sub == "dataset":
        return _run_build_dataset(args)
    if sub == "audio-analyze":
        return _run_audio_analyze(args)
    if sub == "captions":
        return _run_captions(args)
    if sub == "tags":
        return _run_tags(args)
    if sub == "convert-sidecars":
        return _run_convert_sidecars(args)
    if sub == "export":
        return _run_export(args)
    if sub == "settings":
        return _run_settings(args)
    if sub == "history":
        return _run_history(args)
    if sub == "preprocess":
        return _run_preprocess_subcommand(args)

    # Auto-resolve checkpoint_dir from settings for train/analyze
    _auto_resolve_checkpoint_dir(args)

    # Auto-detect --shift and --num-inference-steps from model variant
    _auto_resolve_shift_steps(args)

    # -- Validate required fields (relaxed from argparse for --config/--preprocess)
    if sub in ("train", "analyze"):
        required = [("checkpoint_dir", "--checkpoint-dir / -c"), ("dataset_dir", "--dataset-dir / -d")]
        if sub == "train":
            required.append(("output_dir", "--output-dir / -o"))
        if not _validate_required_args(args, required):
            return 1

    # All other subcommands need path validation
    if not validate_paths(args):
        return 1

    if sub == "train":
        from sidestep_engine.cli.train_fixed import run_fixed
        return run_fixed(args)

    elif sub == "analyze":
        return _run_fisher(args)

    else:
        print(f"[FAIL] Unknown subcommand: {sub}", file=sys.stderr)
        return 1


def main() -> int:
    """Entry point for Side-Step training CLI.

    When invoked with a subcommand (``sidestep train ...``), runs
    that subcommand once and exits.  When invoked without arguments,
    launches the interactive wizard in a session loop so the user can
    preprocess, train, and manage presets without restarting.
    """
    # -- Deprecation shim (rewrite legacy CLI syntax) -----------------------
    _apply_deprecation_shim()

    # -- Compatibility check (non-fatal) ------------------------------------
    try:
        from sidestep_engine._compat import check_compatibility
        check_compatibility()
    except Exception:
        pass  # never let the compat check itself crash the CLI

    # -- GUI subcommand (explicit or translated from --gui) -----------------
    if len(sys.argv) > 1 and sys.argv[1] == "gui":
        try:
            from sidestep_engine.gui import launch
        except ImportError:
            print(
                "[FAIL] GUI dependencies not installed.\n"
                "       Install with: pip install side-step[gui]",
                file=sys.stderr,
            )
            return 1
        port = 8770
        for i, arg in enumerate(sys.argv):
            if arg == "--port" and i + 1 < len(sys.argv):
                try:
                    port = int(sys.argv[i + 1])
                except ValueError:
                    pass
        launch(port=port)
        return 0

    # -- Direct CLI mode (subcommand given) ---------------------------------
    if _has_subcommand():
        from sidestep_engine.settings import is_first_run
        if is_first_run():
            print(
                "[INFO] First-time setup not complete. "
                "Run 'sidestep' without arguments for the interactive setup wizard."
            )
        from sidestep_engine.cli.common import build_root_parser
        parser = build_root_parser()
        args = parser.parse_args()
        return _dispatch(args)

    # -- Interactive wizard session loop ------------------------------------
    from sidestep_engine.ui.wizard import run_wizard_session

    last_code = 0
    for args in run_wizard_session():
        try:
            last_code = _dispatch(args)
        except Exception as exc:
            logger.exception("Unhandled error in session loop")
            print(f"[FAIL] {exc}", file=sys.stderr)
            last_code = 1
        finally:
            _cleanup_gpu()

    return last_code


# ===========================================================================
# Subcommand implementations
# ===========================================================================

def _run_preprocess(args) -> int:
    """Run the two-pass preprocessing pipeline (inline from train --preprocess)."""
    from sidestep_engine.data.preprocess import preprocess_audio_files
    from sidestep_engine.ui.dependency_check import (
        ensure_optional_dependencies,
        required_preprocess_optionals,
    )

    audio_dir = getattr(args, "audio_dir", None)
    dataset_json = getattr(args, "dataset_json", None)
    tensor_output = getattr(args, "tensor_output", None)

    if not audio_dir and not dataset_json:
        print("[FAIL] --audio-dir or --dataset-json is required for preprocessing.", file=sys.stderr)
        return 1
    if not tensor_output:
        print("[FAIL] --tensor-output is required for preprocessing.", file=sys.stderr)
        return 1

    _auto_resolve_checkpoint_dir(args)

    source_label = dataset_json if dataset_json else audio_dir

    ensure_optional_dependencies(
        required_preprocess_optionals(getattr(args, "normalize", "none")),
        interactive=sys.stdin.isatty(),
        allow_install_prompt=not getattr(args, "yes", False),
    )

    # Show summary and confirm before starting
    print("\n" + "=" * 60)
    print("  Preprocessing Summary")
    print("=" * 60)
    print(f"  Source:        {source_label}")
    print(f"  Output:        {tensor_output}")
    print(f"  Checkpoint:    {args.checkpoint_dir}")
    print(f"  Model variant: {args.model_variant}")
    _md = getattr(args, "max_duration", 0)
    _norm = getattr(args, "normalize", "none")
    _tdb = getattr(args, "target_db", -1.0)
    _tlufs = getattr(args, "target_lufs", -14.0)
    print(f"  Max duration:  {'auto-detect' if _md <= 0 else f'{_md}s'}")
    if _norm != "none":
        _norm_str = f"{_norm} ({_tdb} dBFS)" if _norm == "peak" else f"{_norm} ({_tlufs} LUFS)"
        print(f"  Normalize:     {_norm_str}")
    else:
        print(f"  Normalize:     {_norm}")
    print("=" * 60)
    print("[INFO] Two-pass pipeline (sequential model loading for low VRAM)")

    try:
        result = preprocess_audio_files(
            audio_dir=audio_dir,
            output_dir=tensor_output,
            checkpoint_dir=args.checkpoint_dir,
            variant=args.model_variant,
            max_duration=getattr(args, "max_duration", 0),
            dataset_json=dataset_json,
            device=getattr(args, "device", "auto"),
            precision=getattr(args, "precision", "auto"),
            normalize=getattr(args, "normalize", "none"),
            target_db=getattr(args, "target_db", -1.0),
            target_lufs=getattr(args, "target_lufs", -14.0),
        )
    except Exception as exc:
        print(f"[FAIL] Preprocessing failed: {exc}", file=sys.stderr)
        logger.exception("Preprocessing error")
        return 1
    finally:
        _cleanup_gpu()

    print(f"\n[OK] Preprocessing complete:")
    print(f"     Processed: {result['processed']}/{result['total']}")
    if result["failed"]:
        print(f"     Failed:    {result['failed']}")
    print(f"     Output:    {result['output_dir']}")
    print(f"\n[INFO] You can now train with:")
    print(f"       sidestep train -d {result['output_dir']} ...")
    return 0


def _run_preprocess_subcommand(args) -> int:
    """Run preprocessing as a top-level subcommand (``sidestep preprocess``)."""
    from sidestep_engine.data.preprocess import preprocess_audio_files
    from sidestep_engine.ui.dependency_check import (
        ensure_optional_dependencies,
        required_preprocess_optionals,
    )

    audio_dir = getattr(args, "audio_dir", None)
    dataset_json = getattr(args, "dataset_json", None)
    tensor_output = getattr(args, "tensor_output", None)

    if not audio_dir and not dataset_json:
        print("[FAIL] --audio-dir / -i or --dataset-json is required.", file=sys.stderr)
        return 1
    if not tensor_output:
        print("[FAIL] --output / -o is required.", file=sys.stderr)
        return 1

    _auto_resolve_checkpoint_dir(args)
    if not getattr(args, "checkpoint_dir", None):
        print("[FAIL] --checkpoint-dir / -c is required (or set via 'sidestep settings set checkpoint_dir <path>').", file=sys.stderr)
        return 1

    source_label = dataset_json if dataset_json else audio_dir

    ensure_optional_dependencies(
        required_preprocess_optionals(getattr(args, "normalize", "none")),
        interactive=sys.stdin.isatty(),
        allow_install_prompt=not getattr(args, "yes", False),
    )

    print("\n" + "=" * 60)
    print("  Preprocessing Summary")
    print("=" * 60)
    print(f"  Source:        {source_label}")
    print(f"  Output:        {tensor_output}")
    print(f"  Checkpoint:    {args.checkpoint_dir}")
    print(f"  Model variant: {args.model_variant}")
    _md = getattr(args, "max_duration", 0)
    _norm = getattr(args, "normalize", "none")
    _tdb = getattr(args, "target_db", -1.0)
    _tlufs = getattr(args, "target_lufs", -14.0)
    print(f"  Max duration:  {'auto-detect' if _md <= 0 else f'{_md}s'}")
    if _norm != "none":
        _norm_str = f"{_norm} ({_tdb} dBFS)" if _norm == "peak" else f"{_norm} ({_tlufs} LUFS)"
        print(f"  Normalize:     {_norm_str}")
    else:
        print(f"  Normalize:     {_norm}")
    print("=" * 60)
    print("[INFO] Two-pass pipeline (sequential model loading for low VRAM)")

    try:
        result = preprocess_audio_files(
            audio_dir=audio_dir,
            output_dir=tensor_output,
            checkpoint_dir=args.checkpoint_dir,
            variant=args.model_variant,
            max_duration=getattr(args, "max_duration", 0),
            dataset_json=dataset_json,
            device=getattr(args, "device", "auto"),
            precision=getattr(args, "precision", "auto"),
            normalize=getattr(args, "normalize", "none"),
            target_db=getattr(args, "target_db", -1.0),
            target_lufs=getattr(args, "target_lufs", -14.0),
        )
    except Exception as exc:
        print(f"[FAIL] Preprocessing failed: {exc}", file=sys.stderr)
        logger.exception("Preprocessing error")
        return 1
    finally:
        _cleanup_gpu()

    print(f"\n[OK] Preprocessing complete:")
    print(f"     Processed: {result['processed']}/{result['total']}")
    if result["failed"]:
        print(f"     Failed:    {result['failed']}")
    print(f"     Output:    {result['output_dir']}")
    print(f"\n[INFO] You can now train with:")
    print(f"       sidestep train -d {result['output_dir']} ...")
    return 0


def _run_fisher(args) -> int:
    """Run Fisher + Spectral analysis for adaptive LoRA rank assignment."""
    from sidestep_engine.analysis.fisher import run_fisher_analysis

    print("\n" + "=" * 60)
    print("  PP++ / Fisher + Spectral Analysis")
    print("=" * 60)
    print(f"  Checkpoint:      {args.checkpoint_dir}")
    print(f"  Model variant:   {args.model_variant}")
    print(f"  Dataset:         {args.dataset_dir}")
    print(f"  Timestep focus:  {getattr(args, 'timestep_focus', 'balanced')}")
    print(f"  Rank budget:     {args.rank} (base), {args.rank_min}-{args.rank_max}")
    print("=" * 60)

    try:
        result = run_fisher_analysis(
            checkpoint_dir=args.checkpoint_dir,
            variant=args.model_variant,
            dataset_dir=args.dataset_dir,
            base_rank=args.rank,
            rank_min=getattr(args, "rank_min", 16),
            rank_max=getattr(args, "rank_max", 128),
            timestep_focus=getattr(args, "timestep_focus", "balanced"),
            num_runs=getattr(args, "runs", None),
            batches_per_run=getattr(args, "batches", None),
            convergence_patience=getattr(args, "convergence_patience", 5),
            auto_confirm=getattr(args, "yes", False),
        )
    except Exception as exc:
        print(f"[FAIL] Analysis failed: {exc}", file=sys.stderr)
        logger.exception("Fisher analysis error")
        return 1
    finally:
        _cleanup_gpu()

    if result is None:
        print("[INFO] Analysis cancelled or produced no results.")
        return 1

    n_modules = len(result.get("rank_pattern", {}))
    budget = result.get("rank_budget", {})
    print(f"\n[OK] Analysis complete: {n_modules} modules, "
          f"ranks {budget.get('min', '?')}-{budget.get('max', '?')}")
    return 0


def _run_build_dataset(args) -> int:
    """Build a dataset.json from a folder of audio + sidecar metadata."""
    from sidestep_engine.data.dataset_builder import build_dataset

    input_dir = args.input
    tag = getattr(args, "tag", "")
    tag_position = getattr(args, "tag_position", "prepend")
    genre_ratio = getattr(args, "genre_ratio", 0)
    name = getattr(args, "name", "local_dataset")
    output = getattr(args, "output", None)

    print("\n" + "=" * 60)
    print("  Build Dataset")
    print("=" * 60)
    print(f"  Input:         {input_dir}")
    print(f"  Tag:           {tag or '(none)'}")
    print(f"  Tag position:  {tag_position}")
    print(f"  Genre ratio:   {genre_ratio}%")
    print(f"  Dataset name:  {name}")
    print(f"  Output:        {output or '<input>/dataset.json'}")
    print("=" * 60)

    try:
        out_path, stats = build_dataset(
            input_dir=input_dir,
            tag=tag,
            tag_position=tag_position,
            name=name,
            output=output,
            genre_ratio=genre_ratio,
        )
    except (FileNotFoundError, OSError) as exc:
        print(f"[FAIL] {exc}", file=sys.stderr)
        return 1
    except Exception as exc:
        print(f"[FAIL] Dataset build failed: {exc}", file=sys.stderr)
        logger.exception("Dataset build error")
        return 1

    print(f"\n[OK] Dataset built: {stats['total']} samples")
    print(f"     With metadata: {stats['with_metadata']}")
    print(f"     Output:        {out_path}")
    print(f"\n[INFO] You can now preprocess with:")
    print(f"       sidestep preprocess --dataset-json {out_path} -o ./tensors -c ./checkpoints")
    return 0


def _run_audio_analyze(args) -> int:
    """Run local offline audio analysis (BPM, key, time signature) on audio files."""
    from pathlib import Path
    from sidestep_engine.analysis.audio_analysis import analyze_audio
    from sidestep_engine.data.sidecar_io import (
        merge_fields, read_sidecar, sidecar_path_for, write_sidecar,
    )

    input_dir = Path(args.input)
    if not input_dir.is_dir():
        print(f"[FAIL] Not a directory: {input_dir}", file=sys.stderr)
        return 1

    audio_exts = {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac"}
    audio_files = sorted(
        f for f in input_dir.rglob("*") if f.suffix.lower() in audio_exts and f.is_file()
    )
    if not audio_files:
        print(f"[FAIL] No audio files found in {input_dir}", file=sys.stderr)
        return 1

    device = getattr(args, "device", "auto")
    policy = getattr(args, "policy", "fill_missing")
    mode = getattr(args, "mode", "mid")
    n_chunks = getattr(args, "chunks", 5)

    print("\n" + "=" * 60)
    print("  Audio Analysis (local offline)")
    print("=" * 60)
    print(f"  Input:       {input_dir}")
    print(f"  Audio files: {len(audio_files)}")
    print(f"  Device:      {device}")
    print(f"  Mode:        {mode}")
    print(f"  Policy:      {policy}")
    if mode == "sas":
        print(f"  Chunks:      {n_chunks}")
    print("=" * 60)
    _pipelines = {"faf": "librosa direct (no Demucs)", "mid": "Demucs + librosa ensemble", "sas": "Demucs + deep multi-technique"}
    print(f"[INFO] Pipeline: {_pipelines.get(mode, mode)}")

    written = 0
    skipped = 0
    failed = 0

    for i, af in enumerate(audio_files, 1):
        label = af.relative_to(input_dir) if af.is_relative_to(input_dir) else af.name
        try:
            result = analyze_audio(af, device=device, mode=mode, n_chunks=n_chunks)
            # Strip confidence (GUI-only, not for sidecars)
            confidence = result.pop("confidence", {})
            if not result:
                skipped += 1
                print(f"  [{i}/{len(audio_files)}] {label} -- skipped (no results)")
                continue

            sc_path = sidecar_path_for(af)
            existing = read_sidecar(sc_path)

            if policy == "fill_missing":
                # Skip if all analysis fields already populated
                if all(existing.get(k, "").strip() for k in ("bpm", "key", "signature")):
                    skipped += 1
                    print(f"  [{i}/{len(audio_files)}] {label} -- skipped (already populated)")
                    continue

            merged = merge_fields(existing, result, policy=policy)
            write_sidecar(sc_path, merged)
            written += 1
            parts = ", ".join(f"{k}={v}" for k, v in result.items())
            conf_parts = ", ".join(f"{k}={v}" for k, v in confidence.items())
            print(f"  [{i}/{len(audio_files)}] {label} -- written ({parts}) [{conf_parts}]")

        except Exception as exc:
            failed += 1
            print(f"  [{i}/{len(audio_files)}] {label} -- FAILED: {exc}")
            logger.exception("Audio analysis failed for %s", af)
        finally:
            _cleanup_gpu()

    print(f"\n[OK] Audio analysis complete: {written} written, {skipped} skipped, {failed} failed")
    return 0


def _run_captions(args) -> int:
    """Generate AI captions and/or fetch lyrics for audio sidecar files."""
    from pathlib import Path
    from sidestep_engine.settings import (
        get_caption_provider,
        get_gemini_api_key,
        get_gemini_model,
        get_genius_api_token,
        get_hf_token,
        get_music_flamingo_url,
        get_openai_api_key,
        get_openai_base_url,
        get_openai_model,
        get_transcriber_server_url,
    )
    from sidestep_engine.data.enrich_song import enrich_one, parse_filename

    input_dir = Path(args.input)
    if not input_dir.is_dir():
        print(f"[FAIL] Not a directory: {input_dir}", file=sys.stderr)
        return 1

    audio_exts = {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac"}
    audio_files = sorted(
        f for f in input_dir.rglob("*") if f.suffix.lower() in audio_exts and f.is_file()
    )
    if not audio_files:
        print(f"[FAIL] No audio files found in {input_dir}", file=sys.stderr)
        return 1

    provider = args.provider or get_caption_provider() or "gemini"
    policy = args.policy
    default_artist = args.default_artist or ""

    # Resolve API keys (CLI flag -> env -> settings)
    caption_fn = None
    if provider == "gemini":
        api_key = args.gemini_api_key or get_gemini_api_key()
        if not api_key:
            print("[FAIL] No Gemini API key. Set GEMINI_API_KEY env var, "
                  "use --gemini-api-key, or run 'sidestep settings set gemini_api_key <key>'.",
                  file=sys.stderr)
            return 1
        model = args.ai_model or get_gemini_model() or "gemini-2.5-flash"
        use_google_search = getattr(args, "google_search", False)
        from sidestep_engine.data.caption_provider_gemini import generate_caption as _gem_cap

        def caption_fn(title, artist, lyrics_excerpt, audio_path):
            return _gem_cap(title, artist, api_key, audio_path=audio_path,
                            lyrics_excerpt=lyrics_excerpt, model=model,
                            google_search=use_google_search)
    elif provider == "openai":
        api_key = args.openai_api_key or get_openai_api_key()
        if not api_key:
            print("[FAIL] No OpenAI API key. Set OPENAI_API_KEY env var, "
                  "use --openai-api-key, or run 'sidestep settings set openai_api_key <key>'.",
                  file=sys.stderr)
            return 1
        model = args.ai_model or get_openai_model() or "gpt-4o"
        base_url = args.openai_base_url or get_openai_base_url()
        from sidestep_engine.data.caption_provider_openai import generate_caption as _oai_cap

        def caption_fn(title, artist, lyrics_excerpt, audio_path):
            return _oai_cap(title, artist, api_key, audio_path=audio_path,
                            lyrics_excerpt=lyrics_excerpt, model=model,
                            base_url=base_url)
    elif provider in ("local_8-10gb", "local_16gb"):
        from sidestep_engine.data.caption_provider_local import generate_caption as _local_cap
        tier = "8-10gb" if provider == "local_8-10gb" else "16gb"

        def caption_fn(title, artist, lyrics_excerpt, audio_path):
            return _local_cap(title, artist, audio_path=audio_path,
                              lyrics_excerpt=lyrics_excerpt, tier=tier)
    elif provider in ("music_flamingo", "lyrics_only", "none"):
        caption_fn = None

    # -- Metadata provider (Music Flamingo) ----------------------------------
    metadata_fn = None
    meta_provider = getattr(args, "metadata_provider", None) or "none"
    if meta_provider == "music_flamingo" or provider == "music_flamingo":
        flamingo_url = getattr(args, "music_flamingo_url", None) or get_music_flamingo_url() or ""
        hf_token = getattr(args, "hf_token", None) or get_hf_token() or ""
        if flamingo_url:
            from sidestep_engine.data.metadata_provider_music_flamingo import (
                fetch_music_flamingo_metadata as _gen_meta,
            )

            def metadata_fn(audio_path):
                return _gen_meta(str(audio_path), server_url=flamingo_url,
                                 hf_token=hf_token or None)
        else:
            print("[INFO] No Music Flamingo URL configured — metadata provider disabled.",
                  file=sys.stderr)

    # -- Lyrics provider -----------------------------------------------------
    lyrics_fn = None
    lyrics_provider = getattr(args, "lyrics_provider", None)
    if lyrics_provider is None:
        if not args.lyrics:
            lyrics_provider = "none"
        elif provider == "lyrics_only":
            lyrics_provider = "genius"
        else:
            lyrics_provider = "genius"

    if lyrics_provider == "genius":
        token = args.genius_token or get_genius_api_token()
        if token:
            from sidestep_engine.data.lyrics_provider_genius import fetch_lyrics as _genius

            def lyrics_fn(artist, title):
                return _genius(artist, title, token)
        else:
            print("[INFO] No Genius API token — lyrics fetching disabled. "
                  "Set GENIUS_API_TOKEN or use --genius-token.", file=sys.stderr)
    elif lyrics_provider == "transcriber_server":
        server_url = getattr(args, "transcriber_server_url", None) or get_transcriber_server_url() or ""
        if server_url:
            from sidestep_engine.data.lyrics_provider_server import fetch_lyrics_from_server as _srv_lyrics

            def lyrics_fn(artist, title):
                return _srv_lyrics("", server_url=server_url, artist=artist, title=title)
        else:
            print("[FAIL] --transcriber-server-url required for transcriber_server lyrics provider.",
                  file=sys.stderr)
            return 1
    elif lyrics_provider == "music_flamingo":
        flamingo_url = getattr(args, "music_flamingo_url", None) or get_music_flamingo_url() or ""
        hf_token = getattr(args, "hf_token", None) or get_hf_token() or ""
        if flamingo_url:
            from sidestep_engine.data.lyrics_provider_music_flamingo import fetch_lyrics_from_music_flamingo as _flam_lyrics

            def lyrics_fn(artist, title):
                return _flam_lyrics("", server_url=flamingo_url, artist=artist,
                                    title=title, hf_token=hf_token or None)
        else:
            print("[FAIL] --music-flamingo-url required for music_flamingo lyrics provider.",
                  file=sys.stderr)
            return 1

    print("\n" + "=" * 60)
    print("  AI Captions")
    print("=" * 60)
    print(f"  Input:          {input_dir}")
    print(f"  Audio files:    {len(audio_files)}")
    print(f"  Provider:       {provider}")
    if metadata_fn:
        print(f"  Metadata:       {meta_provider}")
    print(f"  Lyrics:         {lyrics_provider}")
    print(f"  Policy:         {policy}")
    if default_artist:
        print(f"  Default artist: {default_artist}")
    print("=" * 60)

    written = 0
    skipped = 0
    failed = 0

    for i, af in enumerate(audio_files, 1):
        result = enrich_one(
            af,
            default_artist=default_artist,
            caption_fn=caption_fn,
            lyrics_fn=lyrics_fn,
            metadata_fn=metadata_fn,
            policy=policy,
        )
        status = result.get("status", "unknown")
        label = af.relative_to(input_dir) if af.is_relative_to(input_dir) else af.name
        if status == "written":
            written += 1
            print(f"  [{i}/{len(audio_files)}] {label} — written")
        elif status == "skipped":
            skipped += 1
            print(f"  [{i}/{len(audio_files)}] {label} — skipped")
        else:
            failed += 1
            err = result.get("error", "unknown error")
            print(f"  [{i}/{len(audio_files)}] {label} — FAILED: {err}")
        for w in result.get("warnings", []):
            print(f"           warning: {w}")

    # Free VRAM if a local model was loaded
    if provider in ("local_8-10gb", "local_16gb"):
        try:
            from sidestep_engine.data.caption_provider_local import unload_model
            unload_model()
        except Exception:
            pass

    print(f"\n[OK] Captions complete: {written} written, {skipped} skipped, {failed} failed")
    return 0


def _run_tags(args) -> int:
    """Bulk sidecar trigger-tag operations."""
    from pathlib import Path
    from sidestep_engine.data.sidecar_io import read_sidecar, write_sidecar, sidecar_path_for

    action = args.tags_action
    if not action:
        print("[FAIL] Specify a tags action: add, remove, clear, or list", file=sys.stderr)
        print("       Example: sidestep tags add ./audio -t 'my_style'", file=sys.stderr)
        return 1

    raw_dir = getattr(args, "directory", None) or getattr(args, "input_compat", None)
    if not raw_dir:
        print("[FAIL] A directory is required.", file=sys.stderr)
        print("       Example: sidestep tags list ./my_audio", file=sys.stderr)
        return 1

    input_dir = Path(raw_dir)
    if not input_dir.is_dir():
        print(f"[FAIL] Not a directory: {input_dir}", file=sys.stderr)
        return 1

    audio_exts = {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac"}
    audio_files = sorted(
        f for f in input_dir.rglob("*") if f.suffix.lower() in audio_exts and f.is_file()
    )
    if not audio_files:
        print(f"[FAIL] No audio files found in {input_dir}", file=sys.stderr)
        return 1

    count = 0

    if action == "list":
        for af in audio_files:
            sc = sidecar_path_for(af)
            data = read_sidecar(sc)
            tag_val = (data.get("custom_tag") or data.get("trigger") or "").strip()
            label = af.relative_to(input_dir) if af.is_relative_to(input_dir) else af.name
            if tag_val:
                print(f"  {label}: {tag_val}")
                count += 1
            else:
                print(f"  {label}: (none)")
        print(f"\n[OK] {count}/{len(audio_files)} files have trigger tags")
        return 0

    tag = getattr(args, "tag", "")

    if action == "add":
        position = args.position
        for af in audio_files:
            sc = sidecar_path_for(af)
            data = read_sidecar(sc)
            existing = (data.get("custom_tag") or data.get("trigger") or "").strip()
            if position == "prepend":
                data["custom_tag"] = f"{tag} {existing}".strip()
            elif position == "append":
                data["custom_tag"] = f"{existing} {tag}".strip()
            elif position == "replace":
                data["custom_tag"] = tag
            data.pop("trigger", None)
            write_sidecar(sc, data)
            count += 1
        print(f"[OK] Added tag '{tag}' ({args.position}) to {count} sidecar files")

    elif action == "remove":
        for af in audio_files:
            sc = sidecar_path_for(af)
            data = read_sidecar(sc)
            existing = (data.get("custom_tag") or data.get("trigger") or "").strip()
            if tag in existing:
                cleaned = existing.replace(tag, "").strip()
                while "  " in cleaned:
                    cleaned = cleaned.replace("  ", " ")
                data["custom_tag"] = cleaned
                data.pop("trigger", None)
                write_sidecar(sc, data)
                count += 1
        print(f"[OK] Removed tag '{tag}' from {count} sidecar files")

    elif action == "clear":
        for af in audio_files:
            sc = sidecar_path_for(af)
            data = read_sidecar(sc)
            if data.get("custom_tag") or data.get("trigger"):
                data["custom_tag"] = ""
                data.pop("trigger", None)
                write_sidecar(sc, data)
                count += 1
        print(f"[OK] Cleared trigger tags from {count} sidecar files")

    return 0


def _run_convert_sidecars(args) -> int:
    """Convert JSON sidecars to TXT format."""
    from pathlib import Path
    from sidestep_engine.data.convert_sidecars import (
        convert_per_file_jsons,
        convert_dataset_json,
        detect_json_sidecars,
    )

    input_path = Path(args.input)
    overwrite = getattr(args, "overwrite", False)
    yes = getattr(args, "yes", False)

    if input_path.is_file() and input_path.suffix.lower() == ".json":
        # Dataset JSON mode
        audio_dir = getattr(args, "audio_dir", None)
        print(f"[INFO] Converting dataset JSON: {input_path}")
        print(f"[INFO] Audio directory: {audio_dir or input_path.parent}")

        if not yes:
            confirm = input("Proceed? [Y/n] ").strip().lower()
            if confirm and confirm != "y":
                print("[INFO] Cancelled.")
                return 0

        results = convert_dataset_json(
            str(input_path), audio_dir=audio_dir, overwrite=overwrite,
        )
        print(f"\n[OK] Converted {len(results)} samples to TXT sidecars")
        for fname, txt_path in results:
            print(f"  {fname} -> {txt_path.name}")
        return 0

    elif input_path.is_dir():
        # Per-file JSON mode
        json_files = detect_json_sidecars(str(input_path))
        if not json_files:
            print(f"[INFO] No per-file .json sidecars found in {input_path}")
            return 0

        print(f"[INFO] Found {len(json_files)} per-file JSON sidecars in {input_path}")
        for jf in json_files[:10]:
            print(f"  {jf.name}")
        if len(json_files) > 10:
            print(f"  ... and {len(json_files) - 10} more")

        if not yes:
            confirm = input("Convert to TXT? [Y/n] ").strip().lower()
            if confirm and confirm != "y":
                print("[INFO] Cancelled.")
                return 0

        results = convert_per_file_jsons(str(input_path), overwrite=overwrite)
        print(f"\n[OK] Converted {len(results)} JSON sidecars to TXT")
        return 0

    else:
        print(f"[FAIL] Input must be a directory or a .json file: {input_path}", file=sys.stderr)
        return 1


def _run_export(args) -> int:
    """Export adapter to ComfyUI format."""
    from sidestep_engine.core.comfyui_export import export_for_comfyui, resolve_target, get_scaling_info

    adapter_dir = args.adapter_dir
    output = getattr(args, "output", None)
    target = getattr(args, "target", "native")
    prefix = getattr(args, "prefix", None)
    normalize_alpha = getattr(args, "normalize_alpha", False)

    resolved_prefix = prefix if prefix is not None else resolve_target(target)

    print("\n" + "=" * 60)
    print("  Export Adapter to ComfyUI")
    print("=" * 60)
    print(f"  Adapter dir:  {adapter_dir}")
    print(f"  Output:       {output or '(auto)'}")
    print(f"  Target:       {target}")
    if prefix is not None:
        print(f"  Prefix:       {prefix} (manual override)")
    else:
        print(f"  Prefix:       {resolved_prefix}")

    scaling = get_scaling_info(adapter_dir)
    if scaling["needs_normalization"]:
        print(f"  Alpha/Rank:   {scaling['alpha']}/{scaling['rank']} = {scaling['ratio']}x")
        if normalize_alpha:
            print(f"  Normalize:    yes (alpha will be set to rank)")
        else:
            print(f"  Strength:     ~{scaling['recommended_strength']} recommended in ComfyUI")
    if normalize_alpha:
        print(f"  Normalize:    yes")
    print("=" * 60)

    result = export_for_comfyui(
        adapter_dir,
        output_path=output,
        model_prefix=prefix,
        target=target,
        normalize_alpha=normalize_alpha,
    )

    if result["ok"]:
        if result.get("already_compatible"):
            print(f"\n[INFO] {result['message']}")
        else:
            print(f"\n[OK] {result['message']}")
    else:
        print(f"\n[FAIL] {result['message']}", file=sys.stderr)
        return 1

    return 0


def _run_settings(args) -> int:
    """View or modify persistent settings."""
    from sidestep_engine.settings import (
        load_settings,
        save_settings,
        settings_path,
        _default_settings,
    )

    action = args.settings_action

    if action == "path":
        print(settings_path())
        return 0

    if action == "show" or not action:
        data = load_settings()
        if data is None:
            print("[INFO] No settings file found. Run 'sidestep' for first-time setup,")
            print(f"       or create one at: {settings_path()}")
            return 0
        print(f"Settings file: {settings_path()}\n")
        _SENSITIVE = {"gemini_api_key", "openai_api_key", "genius_api_token"}
        for key, value in sorted(data.items()):
            if key == "version":
                continue
            display = value
            if key in _SENSITIVE and value:
                display = value[:4] + "***" + value[-4:] if len(str(value)) > 8 else "***"
            print(f"  {key}: {display}")
        return 0

    if action == "set":
        data = load_settings() or _default_settings()
        key = args.key
        value = args.value
        defaults = _default_settings()
        if key not in defaults and key != "first_run_complete":
            known = ", ".join(k for k in sorted(defaults) if k != "version")
            print(f"[FAIL] Unknown setting '{key}'. Valid keys: {known}", file=sys.stderr)
            return 1
        # Type coercion for boolean/list fields
        if key == "first_run_complete":
            value = value.lower() in ("true", "1", "yes")
        elif key == "history_output_roots":
            import json as _json
            try:
                value = _json.loads(value)
            except _json.JSONDecodeError:
                value = [v.strip() for v in value.split(",") if v.strip()]
        data[key] = value
        save_settings(data)
        print(f"[OK] {key} = {value}")
        return 0

    if action == "clear":
        data = load_settings()
        if data is None:
            print("[INFO] No settings file to modify.")
            return 0
        key = args.key
        defaults = _default_settings()
        if key not in defaults:
            known = ", ".join(k for k in sorted(defaults) if k != "version")
            print(f"[FAIL] Unknown setting '{key}'. Valid keys: {known}", file=sys.stderr)
            return 1
        data[key] = defaults.get(key)
        save_settings(data)
        print(f"[OK] {key} reset to default ({defaults.get(key)})")
        return 0

    if action == "defaults":
        from sidestep_engine.training_defaults import (
            DEFAULT_LEARNING_RATE,
            DEFAULT_EPOCHS,
            DEFAULT_SAVE_EVERY,
            DEFAULT_OPTIMIZER_TYPE,
        )
        current = {
            "lr": DEFAULT_LEARNING_RATE,
            "epochs": DEFAULT_EPOCHS,
            "save_every": DEFAULT_SAVE_EVERY,
            "optimizer_type": DEFAULT_OPTIMIZER_TYPE,
        }
        overrides = getattr(args, "default_overrides", None)
        if overrides:
            print("[INFO] Training defaults are compile-time constants in "
                  "sidestep_engine/training_defaults.py.")
            print("       To override them per-run, use CLI flags (--lr, --epochs, etc.)")
            print("       or save a preset via the wizard.\n")
            for k, v in overrides:
                if k in current:
                    print(f"  CLI equivalent: --{k.replace('_', '-')} {v}")
                else:
                    print(f"  Unknown default: {k}")
        else:
            print("Training defaults:")
            for k, v in current.items():
                print(f"  {k}: {v}")
            print(f"\nOverride per-run with CLI flags (--lr, --epochs, etc.) or presets.")
        return 0

    print(f"[FAIL] Unknown settings action: {action}", file=sys.stderr)
    return 1


def _run_history(args) -> int:
    """List past training runs."""
    from sidestep_engine.gui.file_ops import build_history

    runs = build_history()
    limit = getattr(args, "limit", 20)
    runs = runs[:limit]

    if getattr(args, "json_output", False):
        import json as _json
        print(_json.dumps(runs, indent=2))
        return 0

    if not runs:
        print("[INFO] No training runs found.")
        print("       Runs are discovered from trained_adapters_dir in settings")
        print("       and any additional history_output_roots.")
        return 0

    # Table output
    hdr = f"{'Run Name':<35} {'Adapter':<8} {'Epochs':>6} {'Best Loss':>10} {'Status':<10}"
    print(hdr)
    print("-" * len(hdr))
    for r in runs:
        name = r.get("run_name", "?")[:34]
        adapter = r.get("adapter", "?")[:7]
        epochs = r.get("epochs", 0)
        best = r.get("best_loss")
        best_str = f"{best:.6f}" if best and isinstance(best, (int, float)) else "--"
        status = r.get("status", "?")[:9]
        print(f"  {name:<33} {adapter:<8} {epochs:>6} {best_str:>10} {status:<10}")

    print(f"\n  Showing {len(runs)} run(s). Use --limit N or --json for more.")
    return 0


# ===========================================================================
# Entry
# ===========================================================================

if __name__ == "__main__":
    sys.exit(main())
