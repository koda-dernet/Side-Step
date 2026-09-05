"""LoRA SFT trainer for the ACE-Step 5Hz planner LM.

Standard causal-LM supervised fine-tuning with PEFT LoRA:

- Base: ``acestep-5Hz-lm-{size}`` (Qwen3ForCausalLM, HF checkpoint dir
  inside the models root, which doubles as Side-Step's
  checkpoint_dir).
- Samples: prompt (masked) + optional CoT block + audio-code tokens +
  ``<|im_end|>`` (trained), built byte-exactly like the engine's prompt
  builder (see prompts.py).
- Loss on completion only.  ``loss_on_cot=True`` (default) also trains
  the CoT metas block so the adapter learns artist-typical bpm/keys for
  Phase-1 planning.

"""

from __future__ import annotations

import json
import logging
import math
import random
import time
from pathlib import Path
from typing import Any, Dict, List

import torch

from sidestep_engine.lm.prompts import (
    TOKEN_IM_END,
    build_training_texts,
    codes_to_token_ids,
    verify_tokenizer_atomicity,
)

logger = logging.getLogger(__name__)

LM_DIR_TEMPLATE = "acestep-5Hz-lm-{size}"
DEFAULT_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]


def resolve_lm_dir(checkpoint_dir: str | Path, lm_size: str) -> Path:
    lm_dir = Path(checkpoint_dir) / LM_DIR_TEMPLATE.format(size=lm_size)
    if not (lm_dir / "config.json").is_file():
        raise FileNotFoundError(
            f"LM checkpoint not found: {lm_dir} (need the HF safetensors dir, "
            f"e.g. <models root>/acestep-5Hz-lm-{lm_size})"
        )
    return lm_dir


def load_lm_base(checkpoint_dir: str | Path, lm_size: str = "4B", device: str = "cuda"):
    """Load the base LM + tokenizer once for reuse across batch trainings.

    Pass the result to :func:`train_lm_adapter` via ``preloaded_model``/
    ``preloaded_tokenizer`` with ``keep_base=True``.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    lm_dir = resolve_lm_dir(checkpoint_dir, lm_size)
    logger.info("[LM-Train] Loading shared base LM: %s", lm_dir)
    tokenizer = AutoTokenizer.from_pretrained(str(lm_dir))
    model = AutoModelForCausalLM.from_pretrained(
        str(lm_dir), torch_dtype=torch.bfloat16, attn_implementation="sdpa",
    )
    model.to(device)
    return model, tokenizer


def _build_samples(
    jsonl_path: Path,
    tokenizer,
    loss_on_cot: bool,
    max_len: int,
) -> List[Dict[str, Any]]:
    samples = []
    skipped_long = 0
    with jsonl_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            prompt_text, think_text = build_training_texts(row, row["caption"], row["lyrics"])
            prompt_ids = tokenizer.encode(prompt_text, add_special_tokens=False)
            think_ids = tokenizer.encode(think_text, add_special_tokens=False)
            code_ids = codes_to_token_ids(row["codes"])

            if loss_on_cot:
                masked = prompt_ids
                trained = think_ids + code_ids + [TOKEN_IM_END]
            else:
                masked = prompt_ids + think_ids
                trained = code_ids + [TOKEN_IM_END]

            total = len(masked) + len(trained)
            if total > max_len:
                skipped_long += 1
                logger.warning(
                    "[LM-Train] SKIP %s: %d tokens > max_len %d (raise --lm-max-len "
                    "or shorten the song)", row.get("file"), total, max_len,
                )
                continue
            samples.append({
                "file": row.get("file", "?"),
                "input_ids": masked + trained,
                "n_masked": len(masked),
            })
    if not samples:
        raise RuntimeError(
            f"No usable training samples from {jsonl_path} "
            f"({skipped_long} skipped as over-length)"
        )
    logger.info(
        "[LM-Train] %d samples (%d skipped over-length), lengths %d..%d tokens",
        len(samples), skipped_long,
        min(len(s["input_ids"]) for s in samples),
        max(len(s["input_ids"]) for s in samples),
    )
    return samples


def _forward_backward_chunked(
    model: Any,
    input_ids: torch.Tensor,
    labels: torch.Tensor,
    grad_accum: int,
    chunk_size: int = 512,
) -> float:
    """Memory-efficient causal-LM loss + backward for a huge vocabulary.

    The 5Hz LM's vocab is ~217k tokens, so full-sequence fp32 logits are
    multiple GB and the stock ``labels=`` loss path materializes several
    copies — that blows past VRAM on whole-song sequences.  Instead:
    run the trunk once, then per ~chunk_size positions compute
    lm_head + cross-entropy against a DETACHED hidden copy and backward
    immediately (peak extra memory = one chunk of logits), finally
    backward the trunk once with the accumulated hidden-state gradient.
    Mathematically identical to the stock loss; lm_head is frozen so the
    per-chunk backwards only produce hidden grads.

    Returns the mean CE over trained tokens (unscaled by grad_accum).
    """
    import torch.nn.functional as F

    base = model.get_base_model()  # PEFT -> Qwen3ForCausalLM (.model + .lm_head)
    h = base.model(input_ids=input_ids).last_hidden_state  # [1, S, H]
    hd = h.detach().requires_grad_(True)

    S = input_ids.shape[1]
    n_tok = int((labels[:, 1:] != -100).sum().item())
    if n_tok == 0:
        return 0.0

    total = 0.0
    for i in range(0, S - 1, chunk_size):
        j = min(i + chunk_size, S - 1)  # positions [i, j) predict labels [i+1, j+1)
        tgt = labels[:, i + 1 : j + 1]
        if not bool((tgt != -100).any()):
            continue  # fully-masked (prompt) chunk: no loss, no grad
        logits = base.lm_head(hd[:, i:j]).float()
        loss = F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]), tgt.reshape(-1),
            ignore_index=-100, reduction="sum",
        )
        (loss / (n_tok * grad_accum)).backward()
        total += float(loss.item())
        del logits, loss

    if hd.grad is not None:
        h.backward(gradient=hd.grad)
    return total / n_tok


def train_lm_adapter(
    jsonl_path: str | Path,
    checkpoint_dir: str | Path,
    output_dir: str | Path,
    lm_size: str = "4B",
    rank: int = 16,
    alpha: int = 32,
    dropout: float = 0.05,
    learning_rate: float = 1e-4,
    epochs: int = 4,
    grad_accum: int = 4,
    max_len: int = 8192,
    loss_on_cot: bool = True,
    seed: int = 42,
    device: str = "cuda",
    warmup_ratio: float = 0.05,
    progress: Any = None,
    preloaded_model: Any = None,
    preloaded_tokenizer: Any = None,
    keep_base: bool = False,
    target_loss: float = 0.0,
) -> Dict[str, Any]:
    """Train the LoRA adapter.  Returns ``{"adapter_dir", "final_loss", ...}``.

    Batch size is 1 sequence with ``grad_accum`` accumulation — sequence
    lengths vary per song, and at 4B with gradient checkpointing this is
    the VRAM-safe shape.

    Batch mode: pass ``preloaded_model``/``preloaded_tokenizer`` (from
    :func:`load_lm_base`) and ``keep_base=True`` to reuse one resident base
    LM across many adapter trainings — the PEFT wrapper is unloaded at the
    end, restoring the caller's base model in place.
    """
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    jsonl_path = Path(jsonl_path)
    output_dir = Path(output_dir)
    adapter_dir = output_dir / "lm_adapter"
    adapter_dir.mkdir(parents=True, exist_ok=True)

    if preloaded_tokenizer is not None:
        tokenizer = preloaded_tokenizer
    else:
        lm_dir = resolve_lm_dir(checkpoint_dir, lm_size)
        logger.info("[LM-Train] Base LM: %s", lm_dir)
        tokenizer = AutoTokenizer.from_pretrained(str(lm_dir))
    atomicity_err = verify_tokenizer_atomicity(tokenizer)
    if atomicity_err:
        raise RuntimeError(f"Tokenizer/prompt mismatch vs engine: {atomicity_err}")

    samples = _build_samples(jsonl_path, tokenizer, loss_on_cot, max_len)

    # Defensive: reclaim anything a prior pipeline stage (code extraction's
    # full DiT model) left behind before loading the 4B LM.
    import gc
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    if preloaded_model is not None:
        model = preloaded_model
    else:
        model = AutoModelForCausalLM.from_pretrained(
            str(resolve_lm_dir(checkpoint_dir, lm_size)),
            torch_dtype=torch.bfloat16, attn_implementation="sdpa",
        )
        model.to(device)
    model.config.use_cache = False
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()

    lora_cfg = LoraConfig(
        r=rank, lora_alpha=alpha, lora_dropout=dropout,
        target_modules=DEFAULT_TARGET_MODULES, bias="none", task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_cfg)
    # fp32 adapter weights for optimizer stability (PEFT casts activations)
    for p in model.parameters():
        if p.requires_grad:
            p.data = p.data.to(torch.float32)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info("[LM-Train] LoRA r=%d alpha=%d dropout=%.2f -> %.1fM trainable params",
                rank, alpha, dropout, trainable / 1e6)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(params, lr=learning_rate, weight_decay=0.01)

    steps_per_epoch = math.ceil(len(samples) / grad_accum)
    total_steps = max(1, steps_per_epoch * epochs)
    warmup_steps = max(1, int(total_steps * warmup_ratio))

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return step / warmup_steps
        p = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * p))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    rng = random.Random(seed)
    torch.manual_seed(seed)

    model.train()
    global_step = 0
    epoch_losses: List[float] = []
    t0 = time.time()
    log: Dict[str, Any] = {"epochs": [], "config": {
        "lm_size": lm_size, "rank": rank, "alpha": alpha, "dropout": dropout,
        "lr": learning_rate, "epochs": epochs, "grad_accum": grad_accum,
        "loss_on_cot": loss_on_cot, "seed": seed, "samples": len(samples),
        "target_loss": target_loss,
    }}

    for epoch in range(epochs):
        order = list(range(len(samples)))
        rng.shuffle(order)
        running = 0.0
        n_micro = 0
        optimizer.zero_grad(set_to_none=True)
        for j, idx in enumerate(order):
            s = samples[idx]
            input_ids = torch.tensor([s["input_ids"]], dtype=torch.long, device=device)
            labels = input_ids.clone()
            labels[0, : s["n_masked"]] = -100

            # Chunked loss+backward: never materializes full [S, 217k] logits
            sample_loss = _forward_backward_chunked(model, input_ids, labels, grad_accum)
            running += sample_loss
            n_micro += 1

            if (j + 1) % grad_accum == 0 or (j + 1) == len(order):
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1

            del input_ids, labels

        avg = running / max(1, n_micro)
        epoch_losses.append(avg)
        lr_now = scheduler.get_last_lr()[0]
        logger.info("[LM-Train] Epoch %d/%d  loss=%.4f  lr=%.2e  (%.0fs elapsed)",
                    epoch + 1, epochs, avg, lr_now, time.time() - t0)
        if progress:
            progress(epoch + 1, epochs, avg)
        log["epochs"].append({"epoch": epoch + 1, "loss": avg, "lr": lr_now})

        model.save_pretrained(str(adapter_dir))
        if device.startswith("cuda") and torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Loss-target stop: fixed epoch counts overtrain large datasets
        # (steps/epoch scales with song count; measured r=-0.74 between song
        # count and final CE). The target normalizes FIT level across dataset
        # sizes — `epochs` is just the safety cap.
        if target_loss > 0 and avg <= target_loss:
            logger.info("[LM-Train] Target loss reached: %.4f <= %.2f at epoch %d/%d — stopping",
                        avg, target_loss, epoch + 1, epochs)
            log["target_loss_stop"] = {"epoch": epoch + 1, "loss": avg}
            break

    (output_dir / "lm_train_log.json").write_text(
        json.dumps(log, indent=2), encoding="utf-8"
    )
    if keep_base:
        # Restore the caller's resident base LM in place: unload() strips the
        # LoRA modules and returns the original (mutated-back) base model, so
        # the next dataset in a batch can re-wrap it without a 4B reload.
        model.unload()
        del model, optimizer, params
    else:
        del model, optimizer, params
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    logger.info("[LM-Train] Done: adapter -> %s (final loss %.4f)", adapter_dir, epoch_losses[-1])
    return {
        "adapter_dir": str(adapter_dir),
        "final_loss": epoch_losses[-1],
        "epoch_losses": epoch_losses,
        "samples": len(samples),
    }
