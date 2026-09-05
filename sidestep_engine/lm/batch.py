"""Batch driver: planner-LM adapters for every preprocessed dataset.

Sweeps a tensors root (e.g. ``M:/preprocessed_tensors/thirds``) and for each
complete dataset runs extract -> train -> export. Engineered for a 185-dataset
overnight campaign:

- **Burst extraction**: the full DiT model is loaded ONCE per sweep and shared
  across every pending dataset's code extraction (saves ~1-2 min/dataset),
  then freed before training.
- **Burst training**: the base 4B LM is loaded ONCE per sweep; each dataset
  wraps it in a fresh PEFT LoRA, trains, saves, and unloads the wrapper to
  restore the shared base (saves ~1 min/dataset).
- **Resumable**: datasets whose adapter is already deployed in
  ``<models root>/adapters/lm/`` are skipped; failures are marked with a
  ``lm_batch_failed.marker`` in the dataset's output dir and skipped on later
  sweeps (delete the marker to retry). Safe to kill and relaunch anytime.
- **Watch mode**: keeps polling for datasets as a concurrently-running
  preprocessing job completes them (a dataset is "complete" when
  ``preprocess_meta.json`` exists and no ``*.tmp.pt`` remain). Exits after
  ``idle_exit_mins`` with nothing new to do.

"""

from __future__ import annotations

import gc
import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

POLL_SECONDS = 300
FAIL_MARKER = "lm_batch_failed.marker"


def _dataset_complete(ds: Path) -> bool:
    """A dataset is safe to process when preprocessing has finished with it."""
    if not (ds / "preprocess_meta.json").is_file():
        return False
    if any(ds.glob("*.tmp.pt")):
        return False
    return any(p for p in ds.glob("*.pt"))


def _dataset_variant(ds: Path, fallback: str) -> str:
    try:
        meta = json.loads((ds / "preprocess_meta.json").read_text(encoding="utf-8"))
        return meta.get("model_variant") or fallback
    except Exception:
        return fallback


def _deployed(deploy_root: Path, name: str, lm_size: str) -> bool:
    from sidestep_engine.lm.export_lm import sanitize_export_name
    return (deploy_root / f"{sanitize_export_name(name)}-{lm_size}" / "adapter_model.safetensors").is_file()


def find_pending(
    tensors_root: Path,
    output_root: Path,
    deploy_root: Path,
    lm_size: str,
) -> List[Path]:
    """Complete datasets that have no deployed adapter and no failure marker."""
    pending = []
    if not tensors_root.is_dir():
        return pending
    for ds in sorted(p for p in tensors_root.iterdir() if p.is_dir()):
        if _deployed(deploy_root, ds.name, lm_size):
            continue
        if (output_root / ds.name / FAIL_MARKER).is_file():
            continue
        if not _dataset_complete(ds):
            continue
        pending.append(ds)
    return pending


def _free_cuda() -> None:
    import torch
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def run_lm_batch(
    tensors_root: str,
    checkpoint_dir: str,
    output_root: str,
    model_variant: str = "acestep-v15-merge-base-sft-turbo-xl-thirds",
    lm_size: str = "4B",
    rank: int = 16,
    alpha: int = 32,
    dropout: float = 0.05,
    learning_rate: float = 1e-4,
    epochs: int = 50,
    grad_accum: int = 2,
    max_len: int = 8192,
    loss_on_cot: bool = True,
    target_loss: float = 0.0,
    seed: int = 42,
    device: str = "cuda",
    precision: str = "bf16",
    deploy_root: Optional[str] = None,
    watch: bool = False,
    idle_exit_mins: int = 120,
) -> Dict[str, Any]:
    """Run the batch campaign. Returns a summary dict."""
    from sidestep_engine.lm.export_lm import export_lm_adapter
    from sidestep_engine.lm.extract import extract_codes
    from sidestep_engine.lm.train_lm import load_lm_base, train_lm_adapter

    tensors = Path(tensors_root)
    out_root = Path(output_root)
    out_root.mkdir(parents=True, exist_ok=True)
    deploy = Path(deploy_root) if deploy_root else Path(checkpoint_dir).parent / "adapters" / "lm"
    batch_log = out_root / "lm_batch_log.jsonl"

    def log_result(name: str, status: str, **extra: Any) -> None:
        row = {"dataset": name, "status": status, "ts": time.time(), **extra}
        with batch_log.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(row) + "\n")

    done = 0
    failed = 0
    idle_since = time.time()
    t_start = time.time()

    while True:
        pending = find_pending(tensors, out_root, deploy, lm_size)
        if not pending:
            if not watch:
                break
            if (time.time() - idle_since) / 60.0 > idle_exit_mins:
                logger.info("[LM-Batch] Idle for %d min — exiting watch mode", idle_exit_mins)
                break
            time.sleep(POLL_SECONDS)
            continue
        idle_since = time.time()
        logger.info("[LM-Batch] Sweep: %d dataset(s) pending", len(pending))

        # ── Burst 1: extraction, ONE shared DiT model ────────────────────────
        need_extract = [
            ds for ds in pending
            if not (out_root / ds.name / "lm_codes.jsonl").is_file()
        ]
        if need_extract:
            from sidestep_engine.models.loader import load_decoder_for_training
            variant = _dataset_variant(need_extract[0], model_variant)
            logger.info("[LM-Batch] Extraction burst: %d dataset(s), loading DiT once (%s)",
                        len(need_extract), variant)
            dit = load_decoder_for_training(checkpoint_dir, variant, device, precision)
            try:
                for ds in need_extract:
                    out_dir = out_root / ds.name
                    out_dir.mkdir(parents=True, exist_ok=True)
                    try:
                        r = extract_codes(
                            ds, checkpoint_dir, _dataset_variant(ds, variant),
                            out_dir / "lm_codes.jsonl",
                            device=device, precision=precision, model=dit,
                        )
                        logger.info("[LM-Batch] Extracted %s: %d songs", ds.name, r["rows"])
                    except Exception as exc:
                        failed += 1
                        (out_dir / FAIL_MARKER).write_text(f"extract: {exc}", encoding="utf-8")
                        log_result(ds.name, "extract_failed", error=str(exc))
                        logger.exception("[LM-Batch] EXTRACT FAILED %s", ds.name)
            finally:
                del dit
                _free_cuda()

        # ── Burst 2: training + export, ONE shared base LM ───────────────────
        trainable = [
            ds for ds in pending
            if (out_root / ds.name / "lm_codes.jsonl").is_file()
        ]
        if trainable:
            base, tokenizer = load_lm_base(checkpoint_dir, lm_size, device)
            try:
                for i, ds in enumerate(trainable):
                    out_dir = out_root / ds.name
                    t0 = time.time()
                    try:
                        result = train_lm_adapter(
                            out_dir / "lm_codes.jsonl", checkpoint_dir, out_dir,
                            lm_size=lm_size, rank=rank, alpha=alpha, dropout=dropout,
                            learning_rate=learning_rate, epochs=epochs,
                            grad_accum=grad_accum, max_len=max_len,
                            loss_on_cot=loss_on_cot, seed=seed, device=device,
                            preloaded_model=base, preloaded_tokenizer=tokenizer,
                            keep_base=True, target_loss=target_loss,
                        )
                        exp = export_lm_adapter(
                            out_dir / "lm_adapter", checkpoint_dir,
                            lm_size=lm_size, export_name=ds.name, deploy_root=deploy,
                        )
                        done += 1
                        mins = (time.time() - t0) / 60.0
                        log_result(ds.name, "ok", final_loss=result["final_loss"],
                                   songs=result["samples"], minutes=round(mins, 1),
                                   adapter=exp["adapter_name"])
                        logger.info("[LM-Batch] [%d/%d] %s DONE: loss %.3f, %.1f min -> %s",
                                    i + 1, len(trainable), ds.name,
                                    result["final_loss"], mins, exp["adapter_name"])
                    except Exception as exc:
                        failed += 1
                        (out_dir / FAIL_MARKER).write_text(f"train: {exc}", encoding="utf-8")
                        log_result(ds.name, "train_failed", error=str(exc))
                        logger.exception("[LM-Batch] TRAIN FAILED %s", ds.name)
                        # A mid-training failure loses the PEFT wrapper while the
                        # base's modules are still LoRA-mutated in place — the only
                        # safe recovery is a fresh base load (~1 min, failure-only).
                        del base
                        _free_cuda()
                        base, tokenizer = load_lm_base(checkpoint_dir, lm_size, device)
            finally:
                del base, tokenizer
                _free_cuda()

        if not watch:
            break

    hours = (time.time() - t_start) / 3600.0
    summary = {"done": done, "failed": failed, "hours": round(hours, 2), "log": str(batch_log)}
    logger.info("[LM-Batch] Campaign summary: %s", summary)
    return summary
