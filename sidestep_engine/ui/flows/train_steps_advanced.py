"""
Advanced wizard steps: device, optimizer, VRAM, dataloader, and logging.

Extracted from ``train_steps.py`` to meet the module LOC policy.
"""

from __future__ import annotations

from sidestep_engine.ui.prompt_helpers import (
    IS_WINDOWS,
    DEFAULT_NUM_WORKERS,
    ask,
    ask_bool,
    ask_output_path,
    menu,
    print_message,
    section,
)


_DEFAULT_NUM_LAYERS = 24


def _detect_num_layers(a: dict) -> int:
    """Read num_hidden_layers from model config, falling back to 24."""
    from pathlib import Path
    ckpt = a.get("checkpoint_dir", "")
    variant = a.get("model_variant", "")
    if ckpt and variant:
        cfg_path = Path(ckpt) / variant / "config.json"
        if cfg_path.is_file():
            try:
                import json
                data = json.loads(cfg_path.read_text(encoding="utf-8"))
                n = data.get("num_hidden_layers")
                if isinstance(n, int) and n > 0:
                    return n
            except Exception:
                pass
    return _DEFAULT_NUM_LAYERS


def step_advanced_device(a: dict) -> None:
    """Advanced: device and precision."""
    section("Device & Precision (Advanced, press Enter for defaults)")
    a["device"] = ask("Device", default=a.get("device", "auto"), choices=["auto", "cuda", "cuda:0", "cuda:1", "mps", "xpu", "cpu"], allow_back=True)
    a["precision"] = ask("Precision", default=a.get("precision", "auto"), choices=["auto", "bf16", "fp16", "fp32"], allow_back=True)


def step_advanced_optimizer(a: dict) -> None:
    """Advanced: optimizer and scheduler."""
    section("Optimizer & Scheduler (press Enter for defaults)")
    a["optimizer_type"] = menu(
        "Which optimizer to use?",
        [
            ("adamw", "AdamW (default, reliable)"),
            ("adamw8bit", "AdamW 8-bit (saves ~30% optimizer VRAM, needs bitsandbytes)"),
            ("adafactor", "Adafactor (minimal state memory)"),
            ("prodigy", "Prodigy (auto-tunes LR -- start around 0.1, needs prodigyopt)"),
        ],
        default=1,
        allow_back=True,
    )
    if a["optimizer_type"] == "prodigy":
        a["learning_rate"] = ask(
            "Learning rate (Prodigy: start around 0.1, lower if unstable)",
            default=0.1,
            type_fn=float,
            allow_back=True,
        )

    sched_options = [
        ("cosine", "Cosine Annealing (smooth decay to near-zero, most popular)"),
        ("cosine_restarts", "Cosine with Restarts (cyclical decay, LR resets periodically)"),
        ("linear", "Linear (steady decay to near-zero)"),
        ("constant", "Constant (flat LR after warmup)"),
        ("constant_with_warmup", "Constant with Warmup (explicit warmup then flat)"),
    ]
    if a["optimizer_type"] != "prodigy":
        sched_options.append(("custom", "Custom formula (define your own post-warmup LR curve)"))

    a["scheduler_type"] = menu(
        "LR scheduler?",
        sched_options,
        default=1,
        allow_back=True,
    )

    if a["scheduler_type"] == "custom":
        from sidestep_engine.core.formula_scheduler import (
            FORMULA_PRESETS,
            check_formula_warnings,
            formula_help_text,
            preview_formula,
            validate_formula,
        )

        print_message(formula_help_text(), kind="dim")

        template_options = [
            (key, f"{label}: {formula[:50]}{'...' if len(formula) > 50 else ''}")
            for key, label, formula in FORMULA_PRESETS
        ]
        template_options.append(("scratch", "Write from scratch"))

        choice = menu("Start from a template?", template_options, default=1, allow_back=True)

        default_formula = a.get("scheduler_formula", "")
        if choice != "scratch":
            for key, _label, formula in FORMULA_PRESETS:
                if key == choice:
                    default_formula = formula
                    break

        def _validate(val: str) -> str | None:
            return validate_formula(val)

        a["scheduler_formula"] = ask(
            "LR formula (post-warmup)",
            default=default_formula or None,
            required=True,
            validate_fn=_validate,
            allow_back=True,
        )

        for w in check_formula_warnings(a["scheduler_formula"]):
            print_message(w, kind="warn")

        start, mid, end = preview_formula(a["scheduler_formula"])
        print_message(
            f"Valid -- post-warmup LR: start={start:.2e}  mid={mid:.2e}  end={end:.2e}",
            kind="ok",
        )
    else:
        a["scheduler_formula"] = ""


def _detect_vram() -> tuple[float | None, float | None]:
    """Query GPU VRAM in MB. Returns ``(total_mb, free_mb)``."""
    try:
        from sidestep_engine.models.gpu_utils import detect_gpu
        info = detect_gpu()
        return info.vram_total_mb, info.vram_free_mb
    except Exception:
        return None, None


def _get_adapter_rank(a: dict) -> int:
    """Extract adapter rank from wizard answers."""
    atype = a.get("adapter_type", "lora")
    if atype in ("lokr", "loha"):
        return a.get("linear_dim", 64)
    if atype == "oft":
        return a.get("block_size", 64)
    return a.get("r", 8)


def _show_vram_breakdown(
    a: dict,
    vram_total_mb: float,
    vram_free_mb: float | None,
    suggested_ratio: float,
    attn_backend: str,
    est_kwargs: dict,
    suggestion_reason: str,
    estimate_peak_vram_mb,
) -> None:
    """Print a detailed VRAM estimation breakdown to the user."""
    from sidestep_engine.core.vram_estimation import system_vram_used_mb

    total_gb = vram_total_mb / 1024
    sys_used = system_vram_used_mb(vram_total_mb, vram_free_mb)
    free_gb = (vram_free_mb / 1024) if vram_free_mb is not None else None

    raw_chunk = a.get("chunk_duration")
    chunk_s = raw_chunk if raw_chunk and raw_chunk > 0 else 0
    chunk_label = f"{raw_chunk}s chunks" if raw_chunk else "full length (up to 240s)"
    adapter_type = a.get("adapter_type", "lora")
    rank = _get_adapter_rank(a)
    mlp_str = " + MLP" if a.get("target_mlp", False) else ""
    optim = a.get("optimizer_type", "adamw")

    _, breakdown = estimate_peak_vram_mb(
        suggested_ratio, offload_encoder=a.get("offload_encoder", True),
        **est_kwargs,
    )

    sys_line = ""
    if sys_used > 0:
        sys_line = f"  System/other:   ~{sys_used / 1024:.1f} GB (already in use by other processes)\n"

    avail_line = ""
    if free_gb is not None:
        avail_line = f"  Available:      ~{free_gb:.1f} GB\n"

    print_message(
        f"  Detected {total_gb:.0f} GB GPU ({attn_backend})\n"
        f"{sys_line}"
        f"{avail_line}"
        f"  Estimating: batch={est_kwargs['batch_size']}, {chunk_label}, "
        f"{adapter_type} r={rank}{mlp_str}, {optim}\n"
        f"  Model weights:  ~{breakdown['model_mb'] / 1024:.1f} GB\n"
        f"  Activations:    ~{breakdown['activation_mb'] / 1024:.1f} GB\n"
        f"  Optimizer:      ~{breakdown['optimizer_mb'] / 1024:.1f} GB\n"
        f"  Peak estimate:  ~{breakdown['peak_mb'] / 1024:.1f} GB\n"
        f"  Suggestion: {suggestion_reason}",
        kind="dim",
    )


def step_advanced_vram(a: dict) -> None:
    """Advanced: VRAM savings with selective checkpointing and suggestions."""
    section("VRAM Savings (Advanced, press Enter for defaults)")

    # Offload encoder first — affects VRAM budget for checkpointing suggestion
    a["offload_encoder"] = ask_bool(
        "Offload encoder/VAE to CPU? (saves ~2-4 GB VRAM after setup)",
        default=a.get("offload_encoder", True),
        allow_back=True,
    )

    print_message(
        "Gradient checkpointing recomputes layer activations during\n"
        "  backpropagation instead of storing them. Without it, VRAM\n"
        "  scales with sample length × layers — long songs can easily\n"
        "  need 30-40+ GB even at batch size 1.",
        kind="dim",
    )

    # Gather all context available at this wizard step
    vram_total_mb, vram_free_mb = _detect_vram()
    batch_size = a.get("batch_size", 1)
    # None = chunking disabled -> pass 0 so estimator uses 240s worst-case.
    # An explicit int > 0 is the actual chunk length.
    raw_chunk = a.get("chunk_duration")
    chunk_s = raw_chunk if raw_chunk and raw_chunk > 0 else 0
    adapter_type = a.get("adapter_type", "lora")
    rank = _get_adapter_rank(a)
    target_mlp = a.get("target_mlp", False)
    optimizer_type = a.get("optimizer_type", "adamw")
    offload = a.get("offload_encoder", True)
    num_layers = _detect_num_layers(a)

    from sidestep_engine.core.vram_estimation import (
        build_checkpointing_options,
        detect_attn_backend,
        estimate_peak_vram_mb,
        suggest_checkpointing,
    )

    attn_backend = detect_attn_backend(
        a.get("device", "auto"), a.get("precision", "auto"),
    )

    est_kwargs = dict(
        batch_size=batch_size, chunk_duration_s=chunk_s,
        attn_backend=attn_backend, adapter_type=adapter_type,
        rank=rank, target_mlp=target_mlp, optimizer_type=optimizer_type,
        num_layers=num_layers,
    )

    options = build_checkpointing_options(
        vram_total_mb, offload_encoder=offload, **est_kwargs,
    )

    menu_items = [(f"{r:.2f}", label) for r, label, _est in options]
    menu_items.append(("custom", "Custom ratio (0.0 \u2013 1.0)"))

    # Determine suggested default
    suggested_ratio = 1.0
    suggestion_reason = ""
    default_idx = 1  # Full
    if vram_total_mb is not None:
        suggested_ratio, suggestion_reason = suggest_checkpointing(
            vram_total_mb, offload_encoder=offload,
            vram_free_mb=vram_free_mb, **est_kwargs,
        )
        for idx, (r, _label, _est) in enumerate(options):
            if abs(r - suggested_ratio) < 0.01:
                default_idx = idx + 1
                break

        # Show VRAM breakdown
        _show_vram_breakdown(
            a, vram_total_mb, vram_free_mb, suggested_ratio, attn_backend,
            est_kwargs, suggestion_reason, estimate_peak_vram_mb,
        )

    choice = menu(
        "Gradient checkpointing mode",
        menu_items,
        default=default_idx,
        allow_back=True,
    )

    if choice == "custom":
        ratio = ask(
            "Checkpointing ratio (0.0=off, 0.5=half, 1.0=all)",
            default=a.get("gradient_checkpointing_ratio", 1.0),
            type_fn=float,
            allow_back=True,
        )
        ratio = max(0.0, min(1.0, ratio))
    else:
        ratio = float(choice)

    a["gradient_checkpointing"] = ratio > 0.0
    a["gradient_checkpointing_ratio"] = ratio

    # Soft-gate VRAM warning: never blocks, always lets the user continue.
    if vram_total_mb is not None:
        from sidestep_engine.core.vram_estimation import system_vram_used_mb, vram_verdict
        peak, bd = estimate_peak_vram_mb(
            ratio, offload_encoder=offload, **est_kwargs,
        )
        sys_used = system_vram_used_mb(vram_total_mb, vram_free_mb)
        verdict = vram_verdict(peak, vram_total_mb, system_used_mb=sys_used)

        sys_note = ""
        if sys_used > 100:
            sys_note = (
                f"\n  Your system is already using ~{sys_used / 1024:.1f} GB "
                f"of your {vram_total_mb / 1024:.0f} GB GPU."
            )
        if verdict == "red":
            print_message(
                f"\033[1;31m*** WARNING: Estimated peak VRAM (~{peak / 1024:.1f} GB) "
                f"exceeds available GPU memory. ***{sys_note}\n"
                "  You WILL very likely run out of memory (OOM).\n"
                "  To reduce VRAM: raise checkpointing ratio, lower batch size,\n"
                "  enable latent chunking, or use an 8-bit optimizer.\033[0m",
                kind="warn",
            )
            if not ask_bool(
                "Continue anyway? (you have been warned)",
                default=False,
                allow_back=True,
            ):
                a["gradient_checkpointing_ratio"] = 1.0
                a["gradient_checkpointing"] = True
                print_message(
                    "Falling back to full checkpointing for safety.",
                    kind="dim",
                )
                return
        elif verdict == "yellow":
            print_message(
                f"Estimated peak VRAM (~{peak / 1024:.1f} GB) is tight for your GPU."
                f"{sys_note}\n"
                "  Training may work but could OOM on longer samples. "
                "Consider chunking or a higher checkpointing ratio.",
                kind="warn",
            )
        if not raw_chunk and ratio < 0.5:
            print_message(
                "Chunking is disabled — samples use their full length.\n"
                "  Without chunking, a 4-minute song needs ~4× the VRAM of\n"
                "  a 60-second chunk. Consider enabling chunking if you see\n"
                "  VRAM issues.",
                kind="dim",
            )


def step_advanced_training(a: dict) -> None:
    """Advanced: weight decay, grad norm, bias."""
    section("Advanced Training Settings (press Enter for defaults)")
    a["weight_decay"] = ask("Weight decay", default=a.get("weight_decay", 0.01), type_fn=float, allow_back=True)
    a["max_grad_norm"] = ask("Max gradient norm", default=a.get("max_grad_norm", 1.0), type_fn=float, allow_back=True)
    a["bias"] = ask("Bias training mode", default=a.get("bias", "none"), choices=["none", "all", "lora_only"], allow_back=True)


def step_advanced_dataloader(a: dict) -> None:
    """Advanced: DataLoader tuning."""
    section("Data Loading (Advanced, press Enter for defaults)")
    a["num_workers"] = ask("DataLoader workers", default=a.get("num_workers", DEFAULT_NUM_WORKERS), type_fn=int, allow_back=True)
    if IS_WINDOWS and a["num_workers"] > 0:
        print_message("Warning: Windows detected -- forcing num_workers=0", kind="warn")
        a["num_workers"] = 0
    a["pin_memory"] = ask_bool("Pin memory for GPU transfer?", default=a.get("pin_memory", True), allow_back=True)
    a["prefetch_factor"] = ask("Prefetch factor", default=a.get("prefetch_factor", 2 if a["num_workers"] > 0 else 0), type_fn=int, allow_back=True)
    a["persistent_workers"] = ask_bool("Keep workers alive between epochs?", default=a.get("persistent_workers", a["num_workers"] > 0), allow_back=True)


def step_advanced_logging(a: dict) -> None:
    """Advanced: TensorBoard logging."""
    section("Advanced Logging (press Enter for defaults)")
    print_message(
        "Leave empty to use the default ({output_dir}/runs). "
        "Only change this if you want TensorBoard logs in a shared or custom location.",
        kind="dim",
    )
    log_dir_raw = ask("TensorBoard log directory (leave empty for default)", default=a.get("log_dir"), allow_back=True)
    if log_dir_raw in (None, "None", "") or not str(log_dir_raw).strip():
        a["log_dir"] = None
    else:
        a["log_dir"] = ask_output_path(
            "TensorBoard log directory",
            default=str(log_dir_raw).strip(),
            required=True,
            allow_back=True,
        )
