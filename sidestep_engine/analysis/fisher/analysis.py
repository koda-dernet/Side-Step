"""Fisher + Spectral analysis orchestrator.

Coordinates the full Preprocessing++ pipeline:
1. Load model, discover modules
2. Multi-run Fisher diagonal estimation (timestep-focused)
3. Spectral SVD analysis
4. Rank assignment
5. Preview + confirmation
6. Save fisher_map.json
"""

from __future__ import annotations

import gc
import logging
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import torch
from torch.utils.data import DataLoader, Subset

from sidestep_engine.analysis.fisher.engine import single_fisher_run, _clear_cache
from sidestep_engine.analysis.fisher.io import compute_dataset_hash, save_fisher_map
from sidestep_engine.analysis.fisher.modules import (
    build_param_to_module_map,
    find_all_targetable_modules,
    group_modules_for_chunking,
)
from sidestep_engine.analysis.fisher.preview import ask_confirmation, print_preview
from sidestep_engine.analysis.fisher.ranks import assign_ranks
from sidestep_engine.analysis.fisher.spectral import compute_spectral_complexity

logger = logging.getLogger(__name__)

_FISHER_BASE_SEED = 1729

_REQUIRED_TENSOR_KEYS = frozenset({
    "target_latents",
    "attention_mask",
    "encoder_hidden_states",
    "encoder_attention_mask",
})


def _preflight_tensor_check(pt_files: list[Path]) -> str | None:
    """Verify the first .pt file is loadable and has the expected keys.

    Returns an error message string on failure, or None on success.
    """
    if not pt_files:
        return "No .pt files to check"
    sample_path = pt_files[0]
    try:
        data = torch.load(sample_path, map_location="cpu", weights_only=True)
    except Exception as exc:
        return f"Cannot load {sample_path.name}: {exc}"

    if not isinstance(data, dict):
        return f"{sample_path.name} is not a dict (got {type(data).__name__})"

    missing = _REQUIRED_TENSOR_KEYS - set(data.keys())
    if missing:
        return (
            f"{sample_path.name} is missing required keys: {', '.join(sorted(missing))}. "
            "This file may not be a valid preprocessed tensor."
        )

    for key in _REQUIRED_TENSOR_KEYS:
        val = data[key]
        if not isinstance(val, torch.Tensor):
            return f"{sample_path.name}['{key}'] is {type(val).__name__}, expected Tensor"
        if val.numel() == 0:
            return f"{sample_path.name}['{key}'] is empty (0 elements)"

    return None


def _resolve_model_timestep_params(checkpoint_dir: str, variant: str) -> tuple[float, float, float]:
    """Resolve ``(timestep_mu, timestep_sigma, data_proportion)`` for Fisher.

    Uses the same model ``config.json`` policy as CLI training config build:
    load variant config when available, otherwise keep conservative defaults.
    """
    import json as _json
    from sidestep_engine.core.constants import (
        VARIANT_DIR_MAP,
        DEFAULT_TIMESTEP_MU,
        DEFAULT_TIMESTEP_SIGMA,
        DEFAULT_DATA_PROPORTION,
    )

    ckpt_root = Path(checkpoint_dir)
    mapped = VARIANT_DIR_MAP.get(variant)
    candidate_paths = []
    if mapped:
        candidate_paths.append(ckpt_root / mapped / "config.json")
    candidate_paths.append(ckpt_root / variant / "config.json")

    config_path = next((p for p in candidate_paths if p.is_file()), None)
    timestep_mu, timestep_sigma, data_proportion = (
        DEFAULT_TIMESTEP_MU, DEFAULT_TIMESTEP_SIGMA, DEFAULT_DATA_PROPORTION,
    )
    if config_path is None:
        return timestep_mu, timestep_sigma, data_proportion

    try:
        raw = _json.loads(config_path.read_text(encoding="utf-8"))
        timestep_mu = float(raw.get("timestep_mu", timestep_mu))
        timestep_sigma = float(raw.get("timestep_sigma", timestep_sigma))
        data_proportion = float(raw.get("data_proportion", data_proportion))
    except Exception as exc:
        logger.warning("Failed to parse timestep params from %s: %s", config_path, exc)

    return timestep_mu, timestep_sigma, data_proportion


def _build_run_subset(
    dataset_size: int,
    max_batches: int,
    run_idx: int,
    base_seed: int = _FISHER_BASE_SEED,
) -> List[int]:
    """Build deterministic per-run sample indices (without replacement)."""
    import random

    if dataset_size <= 0:
        return []
    take = min(dataset_size, max_batches)
    indices = list(range(dataset_size))
    rng = random.Random(base_seed + run_idx)
    rng.shuffle(indices)
    return indices[:take]


def _make_subset_loader_factory(
    dataset,
    subset_indices: List[int],
) -> Callable[[], DataLoader]:
    """Return a loader factory that always yields same subset/order."""
    from sidestep_engine.vendor.data_module import collate_preprocessed_batch

    subset = Subset(dataset, subset_indices)

    def _factory() -> DataLoader:
        return DataLoader(
            subset,
            batch_size=1,
            shuffle=False,
            num_workers=0,
            pin_memory=False,
            collate_fn=collate_preprocessed_batch,
            drop_last=False,
        )

    return _factory


def run_fisher_analysis(
    checkpoint_dir: str,
    variant: str,
    dataset_dir: str,
    base_rank: int = 64,
    rank_min: int = 16,
    rank_max: int = 128,
    timestep_focus: str = "balanced",
    num_runs: Optional[int] = None,
    batches_per_run: Optional[int] = None,
    convergence_patience: int = 5,
    progress_callback: Optional[Callable] = None,
    cancel_check: Optional[Callable] = None,
    auto_confirm: bool = False,
) -> Optional[Dict[str, Any]]:
    """Run the full Fisher + Spectral analysis pipeline.

    Args:
        checkpoint_dir: Path to model checkpoints root.
        variant: Model variant (turbo, base, sft).
        dataset_dir: Directory with preprocessed .pt tensor files.
        base_rank: User's chosen base LoRA rank (median target).
        rank_min: Minimum adaptive rank.
        rank_max: Maximum adaptive rank.
        timestep_focus: Timestep focus mode.
        num_runs: Override auto-scaled run count.
        batches_per_run: Override auto-scaled batch count.
        convergence_patience: Early-stop patience per run.
        progress_callback: ``(batch, total, msg) -> None``.
        cancel_check: ``() -> bool``.
        auto_confirm: Skip interactive confirmation (for CLI --yes).

    Returns:
        The fisher_map dict if saved, or None if cancelled/failed.
    """
    from sidestep_engine.models.loader import (
        load_decoder_for_training,
        unload_models,
    )
    from sidestep_engine.models.gpu_utils import detect_gpu, get_available_vram_mb
    from sidestep_engine.vendor.data_module import PreprocessedDataModule

    # Count dataset
    dataset_path = Path(dataset_dir)
    pt_files = sorted(dataset_path.glob("*.pt"))
    dataset_size = len(pt_files)
    if dataset_size == 0:
        logger.error("No .pt files found in %s", dataset_dir)
        return None

    if dataset_size < 5:
        logger.warning(
            "Small dataset (%d samples) -- Fisher module rankings may be unreliable. "
            "Consider adding more training data for better Preprocessing++ results.",
            dataset_size,
        )

    # Tensor integrity check -- fail fast before loading the model
    preflight_error = _preflight_tensor_check(pt_files)
    if preflight_error is not None:
        logger.error("Tensor preflight failed: %s", preflight_error)
        return None

    # Auto-scale
    runs = num_runs or (2 if dataset_size <= 10 else 3)
    bpr = batches_per_run or min(dataset_size, 50)

    logger.info(
        "Fisher analysis: %d songs, %d runs x %d batches, focus=%s",
        dataset_size, runs, bpr, timestep_focus,
    )
    timestep_mu, timestep_sigma, data_proportion = _resolve_model_timestep_params(
        checkpoint_dir,
        variant,
    )
    logger.info(
        "Fisher timestep params: mu=%.3f sigma=%.3f data_proportion=%.3f",
        timestep_mu, timestep_sigma, data_proportion,
    )

    # GPU setup
    gpu = detect_gpu()
    device = torch.device(gpu.device)
    dtype_map = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}
    dtype = dtype_map.get(gpu.precision, torch.bfloat16)

    # VRAM check → decide full vs chunked mode
    device_str = gpu.device  # string like "cuda:0", not torch.device
    free_mb = get_available_vram_mb(device_str)
    if free_mb is not None and free_mb < 7000:
        logger.warning("Low VRAM (%.0f MB). Fisher may fail.", free_mb)

    # Load model
    logger.info("Loading model for Fisher analysis (variant=%s)", variant)
    model = load_decoder_for_training(
        checkpoint_dir=checkpoint_dir, variant=variant,
        device=gpu.device, precision=gpu.precision,
    )

    # Offload encoder/VAE/non-decoder components to CPU -- Fisher only
    # needs the decoder.  Without this the full model eats ~13 GB on a
    # 16 GB card, leaving nothing for gradients.
    try:
        from sidestep_engine.core.trainer_helpers import offload_non_decoder
        n_offloaded = offload_non_decoder(model)
        if n_offloaded:
            logger.info("Offloaded %d non-decoder components to CPU", n_offloaded)
    except Exception as exc:
        logger.warning("Could not offload non-decoder components: %s", exc)

    # Free GPU memory released by offloading
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Enable gradient checkpointing -- critical for VRAM.
    # Try multiple approaches since the bare model (no PEFT/Fabric wrappers)
    # may not respond to the wrapper-aware configure_memory_features().
    ckpt_ok = False
    decoder = model.decoder

    # 1. Direct HuggingFace method (works on bare PreTrainedModel)
    if hasattr(decoder, "gradient_checkpointing_enable"):
        try:
            decoder.gradient_checkpointing_enable()
            ckpt_ok = True
        except Exception:
            pass

    # 2. Manual flag (some models use this instead)
    if not ckpt_ok and hasattr(decoder, "gradient_checkpointing"):
        decoder.gradient_checkpointing = True
        ckpt_ok = True

    # 3. Fall back to the wrapper-aware helper
    if not ckpt_ok:
        try:
            from sidestep_engine.core.trainer_helpers import configure_memory_features
            result = configure_memory_features(decoder)
            ckpt_ok = result[0] if result else False
        except Exception:
            pass

    # 4. Disable KV cache regardless
    cfg = getattr(decoder, "config", None)
    if cfg is not None and hasattr(cfg, "use_cache"):
        cfg.use_cache = False

    # 5. Enable input_require_grads for checkpointing to work
    if hasattr(decoder, "enable_input_require_grads"):
        try:
            decoder.enable_input_require_grads()
        except Exception:
            pass

    # Decoder MUST be in train mode for gradient checkpointing to activate.
    # HF silently skips checkpointing in eval mode.
    decoder.train()

    if ckpt_ok:
        logger.info("Gradient checkpointing enabled for Fisher analysis")
    else:
        logger.warning(
            "Could not enable gradient checkpointing -- VRAM usage will be high"
        )

    # Discover modules
    target_modules = find_all_targetable_modules(model)
    if not target_modules:
        logger.error("No targetable modules found")
        unload_models(model)
        return None

    target_names = [n for n, _ in target_modules]
    param_to_module = build_param_to_module_map(model, target_modules)
    num_analyzed = len(target_modules)
    logger.info(
        "Module discovery: %d modules, %d params mapped",
        num_analyzed, len(param_to_module),
    )
    if not param_to_module:
        logger.error("param_to_module is EMPTY -- id-based matching failed")
        unload_models(model)
        return None

    # Always use chunked mode: each module group (self_attn, cross_attn,
    # mlp) gets its own full data pass so only one group's gradients
    # exist on GPU at a time.  3x forward passes but rock-solid VRAM.
    free_after = get_available_vram_mb(device_str)
    chunk_groups = group_modules_for_chunking(target_modules)
    logger.info(
        "Using chunked mode (%d passes, %.0f MB free VRAM)",
        len(chunk_groups), free_after or 0,
    )

    # Data
    data_module = PreprocessedDataModule(
        tensor_dir=dataset_dir, batch_size=1, num_workers=0, pin_memory=False,
    )
    data_module.setup("fit")
    train_dataset = data_module.train_dataset
    if train_dataset is None:
        logger.error("Fisher dataloader dataset is unavailable")
        unload_models(model)
        return None

    # Multi-run Fisher estimation
    all_run_scores: List[Dict[str, float]] = []
    total_batches = 0
    run_coverages: List[Dict[str, Any]] = []

    for run_idx in range(runs):
        if cancel_check and cancel_check():
            break
        logger.info("Fisher run %d/%d", run_idx + 1, runs)
        subset_indices = _build_run_subset(len(train_dataset), bpr, run_idx)
        if not subset_indices:
            logger.warning("Run %d has empty subset; skipping", run_idx + 1)
            continue
        loader_factory = _make_subset_loader_factory(train_dataset, subset_indices)
        selected_count = len(subset_indices)
        coverage_ratio = selected_count / max(len(train_dataset), 1)
        selected_files: List[str] = []
        valid_paths = getattr(train_dataset, "valid_paths", None)
        if isinstance(valid_paths, list):
            for idx in subset_indices:
                if 0 <= idx < len(valid_paths):
                    selected_files.append(Path(valid_paths[idx]).name)
        run_coverages.append(
            {
                "run": run_idx + 1,
                "selected_count": selected_count,
                "total_count": len(train_dataset),
                "coverage_ratio": round(coverage_ratio, 6),
                "selected_files": selected_files,
            }
        )

        scores = single_fisher_run(
            model=model,
            loader_factory=loader_factory,
            param_to_module=param_to_module,
            target_names=target_names,
            max_batches=bpr,
            timestep_focus=timestep_focus,
            device=device, dtype=dtype,
            chunk_groups=chunk_groups,
            patience=convergence_patience,
            progress_callback=progress_callback,
            cancel_check=cancel_check,
            timestep_mu=timestep_mu,
            timestep_sigma=timestep_sigma,
            data_proportion=data_proportion,
        )
        all_run_scores.append(scores)
        total_batches += selected_count

        # Between-run cleanup
        gc.collect()
        _clear_cache(device)

    if not all_run_scores:
        unload_models(model)
        return None

    # Average and std across runs
    fisher_scores, fisher_stds = _aggregate_runs(all_run_scores, target_names)

    # Spectral analysis
    logger.info("Running spectral analysis on %d modules", num_analyzed)
    spectral = compute_spectral_complexity(target_modules, device)

    # Rank assignment
    target_mods, rank_pattern, alpha_pattern = assign_ranks(
        fisher_scores, spectral,
        base_rank=base_rank, rank_min=rank_min, rank_max=rank_max,
    )

    excluded = [n for n in target_names if not any(
        n.endswith(k) or k in n for k in rank_pattern
    )]

    # Preview
    print_preview(
        fisher_scores=fisher_scores, fisher_stds=fisher_stds,
        spectral_ranks=spectral, rank_pattern=rank_pattern,
        alpha_pattern=alpha_pattern, target_modules=target_mods,
        excluded=excluded, base_rank=base_rank,
        rank_min=rank_min, rank_max=rank_max,
        total_batches=total_batches, num_runs=runs,
        variant=variant, timestep_focus=timestep_focus,
        num_analyzed=num_analyzed,
        sample_coverage=run_coverages,
    )

    # Confirm
    if not auto_confirm and not ask_confirmation():
        logger.info("Fisher map save cancelled by user")
        unload_models(model)
        return None

    # Resolve architecture fingerprint for staleness detection
    _decoder_cfg = getattr(model.decoder, "config", None)
    _num_hidden_layers: int | None = None
    if _decoder_cfg is not None:
        _num_hidden_layers = getattr(_decoder_cfg, "num_hidden_layers", None)

    # Build output
    fisher_map = _build_fisher_map(
        variant=variant, timestep_focus=timestep_focus,
        runs=runs, bpr=bpr, total_batches=total_batches,
        base_rank=base_rank, rank_min=rank_min, rank_max=rank_max,
        target_mods=target_mods, rank_pattern=rank_pattern,
        alpha_pattern=alpha_pattern, fisher_scores=fisher_scores,
        fisher_stds=fisher_stds, spectral=spectral, excluded=excluded,
        dataset_dir=dataset_dir,
        sample_coverage=run_coverages,
        timestep_mu=timestep_mu,
        timestep_sigma=timestep_sigma,
        data_proportion=data_proportion,
        num_hidden_layers=_num_hidden_layers,
    )

    out_path = dataset_path / "fisher_map.json"
    save_fisher_map(fisher_map, out_path)

    unload_models(model)
    return fisher_map


def _aggregate_runs(
    all_scores: List[Dict[str, float]], names: List[str],
) -> tuple:
    """Compute mean and std of Fisher scores across runs."""
    import math
    means: Dict[str, float] = {}
    stds: Dict[str, float] = {}
    for name in names:
        vals = [s.get(name, 0.0) for s in all_scores]
        m = sum(vals) / len(vals) if vals else 0.0
        means[name] = m
        if len(vals) > 1:
            var = sum((v - m) ** 2 for v in vals) / (len(vals) - 1)
            stds[name] = math.sqrt(var)
        else:
            stds[name] = 0.0
    return means, stds


def _build_fisher_map(
    variant, timestep_focus, runs, bpr, total_batches,
    base_rank, rank_min, rank_max, target_mods, rank_pattern,
    alpha_pattern, fisher_scores, fisher_stds, spectral,
    excluded, dataset_dir,
    sample_coverage=None,
    timestep_mu=-0.4,
    timestep_sigma=1.0,
    data_proportion=0.5,
    num_hidden_layers: int | None = None,
) -> Dict[str, Any]:
    """Assemble the full fisher_map dict for JSON serialisation."""
    focus_range = _focus_to_range(timestep_focus)
    modules_detail = []
    for name, score in sorted(fisher_scores.items(), key=lambda x: -x[1]):
        peft_key = name[len("decoder."):] if name.startswith("decoder.") else name
        if peft_key in rank_pattern:
            modules_detail.append({
                "name": name,
                "fisher_score": round(score, 8),
                "fisher_cross_run_std": round(fisher_stds.get(name, 0.0), 8),
                "effective_rank": spectral.get(name, -1),
                "assigned_rank": rank_pattern[peft_key],
            })

    result: Dict[str, Any] = {
        "version": 2,
        "model_variant": variant,
        "timestep_focus": timestep_focus,
        "timestep_range": focus_range,
        "timestep_params": {
            "mu": timestep_mu,
            "sigma": timestep_sigma,
            "data_proportion": data_proportion,
        },
        "num_runs": runs,
        "batches_per_run": bpr,
        "total_batches": total_batches,
        "base_rank": base_rank,
        "rank_budget": {"min": rank_min, "max": rank_max},
        "dataset_hash": compute_dataset_hash(dataset_dir),
        "target_modules": target_mods,
        "rank_pattern": rank_pattern,
        "alpha_pattern": alpha_pattern,
        "modules": modules_detail,
        "excluded_modules": excluded,
        "sample_coverage": sample_coverage or [],
    }
    if num_hidden_layers is not None:
        result["num_hidden_layers"] = num_hidden_layers
    return result


def _focus_to_range(focus: str) -> list:
    """Convert a focus string to a [low, high] pair."""
    if focus == "texture":
        return [0.0, 0.4]
    if focus == "structure":
        return [0.6, 1.0]
    if focus == "balanced":
        return [0.0, 1.0]
    parts = focus.split(",")
    return [float(parts[0]), float(parts[1])]
