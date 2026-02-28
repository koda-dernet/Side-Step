## Overview

Every `--flag` that `train.py` accepts, organized by subcommand and group. Arguments marked **(required)** must always be provided; everything else has a default.

Side-Step uses four subcommands:

| Subcommand | Purpose |
| --- | --- |
| `fixed` | Train a LoRA or LoKR (auto-detects turbo vs base/sft) |
| `fisher` | Fisher + Spectral analysis for adaptive LoRA rank assignment |
| `compare-configs` | Compare module config JSON files |
| `build-dataset` | Build `dataset.json` from audio + sidecar metadata |

> `vanilla` also exists but is deprecated -- it prints a message and exits. Use `fixed` instead.

---

## Global Flags

Available on every subcommand. Place these **before** the subcommand name.

```bash
uv run python train.py --plain --yes fixed ...
```

| Flag | Short | Type | Default | Description |
| --- | --- | --- | --- | --- |
| `--plain` | | bool | `False` | Disable Rich output; use plain text. Auto-set when stdout is not a TTY |
| `--yes` | `-y` | bool | `False` | Skip the confirmation prompt and start training immediately |

---

## `fixed` -- Train a LoRA/LoKR

This is the main training subcommand. It includes all argument groups below.

### Model / Paths

| Flag | Short | Type | Default | Description |
| --- | --- | --- | --- | --- |
| `--checkpoint-dir` | | str | **(required)** | Path to checkpoints root directory |
| `--model-variant` | | str | `turbo` | Model variant or subfolder name. Official: `turbo`, `base`, `sft`. For fine-tunes: use the exact folder name under checkpoint-dir |
| `--base-model` | | str | `None` | Base model a fine-tune was trained from (`turbo`/`base`/`sft`). Used to condition timestep sampling. Auto-detected for official models |

### Device / Platform

| Flag | Short | Type | Default | Description |
| --- | --- | --- | --- | --- |
| `--device` | | str | `auto` | Device selection: `auto`, `cuda`, `cuda:0`, `mps`, `xpu`, `cpu` |
| `--precision` | | str | `auto` | Precision: `auto`, `bf16`, `fp16`, `fp32` |

### Data

| Flag | Short | Type | Default | Description |
| --- | --- | --- | --- | --- |
| `--dataset-dir` | `-d` | str | **(required)** | Directory containing preprocessed `.pt` files |
| `--num-workers` | | int | `4` (Linux), `0` (Windows) | DataLoader workers |
| `--pin-memory` / `--no-pin-memory` | | bool | `True` | Pin memory for GPU transfer |
| `--prefetch-factor` | | int | `2` (Linux), `0` (Windows) | DataLoader prefetch factor |
| `--persistent-workers` / `--no-persistent-workers` | | bool | `True` (Linux), `False` (Windows) | Keep workers alive between epochs |

### Training

| Flag | Short | Type | Default | Description |
| --- | --- | --- | --- | --- |
| `--lr` / `--learning-rate` | `-l` | float | `1e-4` | Initial learning rate. **Set to 1.0 for Prodigy** |
| `--batch-size` | `-b` | int | `1` | Training batch size |
| `--gradient-accumulation` | `-g` | int | `4` | Gradient accumulation steps. Effective batch = `batch-size` x this |
| `--epochs` | `-e` | int | `100` | Maximum training epochs |
| `--max-steps` | `-m` | int | `0` | Maximum optimizer steps; 0 = use epochs only |
| `--warmup-steps` | | int | `100` | LR warmup steps (ramps from 10% to 100%) |
| `--weight-decay` | | float | `0.01` | AdamW weight decay |
| `--max-grad-norm` | | float | `1.0` | Gradient clipping norm |
| `--seed` | `-s` | int | `42` | Random seed |
| `--chunk-duration` | | int | `None` (disabled) | Random latent chunk duration in seconds. Recommended: `60`. Extracts a random window each iteration for augmentation and VRAM savings. Values below 60s may hurt quality |
| `--chunk-decay-every` | | int | `10` | Epoch interval for halving chunk coverage histogram; 0 disables decay |
| `--shift` | | float | `3.0` | Noise schedule shift. Auto-set: turbo=3.0, base/sft=1.0. See [[Shift and Timestep Sampling]] |
| `--num-inference-steps` | | int | `8` | Inference steps for timestep schedule. Auto-set: turbo=8, base/sft=50 |
| `--optimizer-type` | | str | `adamw` | Optimizer. Choices: `adamw`, `adamw8bit`, `adafactor`, `prodigy` |
| `--scheduler-type` | | str | `cosine` | LR scheduler. Choices: `cosine`, `cosine_restarts`, `linear`, `constant`, `constant_with_warmup` |
| `--gradient-checkpointing` / `--no-gradient-checkpointing` | | bool | `True` | Recompute activations to save VRAM (~40-60% less, ~10-30% slower). See [[VRAM Optimization Guide]] |
| `--offload-encoder` / `--no-offload-encoder` | | bool | `False` | Move encoder/VAE to CPU after setup (saves ~2-4 GB VRAM) |

### Adapter

| Flag | Short | Type | Default | Description |
| --- | --- | --- | --- | --- |
| `--adapter-type` | | str | `lora` | Adapter type: `lora` (PEFT) or `lokr` (LyCORIS) |

### LoRA (used when `--adapter-type=lora`)

| Flag | Short | Type | Default | Description |
| --- | --- | --- | --- | --- |
| `--rank` | `-r` | int | `64` | LoRA rank. Higher = more capacity, more VRAM |
| `--alpha` | | int | `128` | LoRA alpha (scaling factor). Usually 2x rank |
| `--dropout` | | float | `0.1` | LoRA dropout. Increase for small datasets, decrease for large |
| `--target-modules` | | list | `q_proj k_proj v_proj o_proj` | Modules to apply adapter to. Space-separated |
| `--target-mlp` / `--no-target-mlp` | | bool | `True` | Target MLP/FFN layers (`gate_proj`, `up_proj`, `down_proj`) |
| `--bias` | | str | `none` | Bias training mode: `none`, `all`, `lora_only` |
| `--attention-type` | | str | `both` | Attention layers to target: `self`, `cross`, `both`. |
| `--self-target-modules` | | list | `None` | Projections for self-attention only (when `--attention-type=both`) |
| `--cross-target-modules` | | list | `None` | Projections for cross-attention only (when `--attention-type=both`) |

### LoKR (used when `--adapter-type=lokr`)

| Flag | Short | Type | Default | Description |
| --- | --- | --- | --- | --- |
| `--lokr-linear-dim` | | int | `64` | LoKR linear dimension (analogous to LoRA rank) |
| `--lokr-linear-alpha` | | int | `128` | LoKR linear alpha (keep at 2x linear dim) |
| `--lokr-factor` | | int | `-1` | Kronecker factorization factor. `-1` = auto |
| `--lokr-decompose-both` | | bool | `False` | Decompose both Kronecker factors |
| `--lokr-use-tucker` | | bool | `False` | Use Tucker decomposition |
| `--lokr-use-scalar` | | bool | `False` | Use scalar scaling |
| `--lokr-weight-decompose` | | bool | `False` | Enable DoRA-style weight decomposition |
| `--target-mlp` / `--no-target-mlp` | | bool | `True` | Target MLP/FFN layers (`gate_proj`, `up_proj`, `down_proj`) |

### Checkpointing

| Flag | Short | Type | Default | Description |
| --- | --- | --- | --- | --- |
| `--output-dir` | | str | **(required)** | Output directory for adapter weights and logs |
| `--save-every` | | int | `10` | Save checkpoint every N epochs |
| `--resume-from` | | str | `None` | Path to a `checkpoint-epoch-N` directory to resume from |

### Logging / TensorBoard

| Flag | Short | Type | Default | Description |
| --- | --- | --- | --- | --- |
| `--log-dir` | | str | `None` | TensorBoard log directory. Default: `{output-dir}/runs` |
| `--log-every` | | int | `10` | Log basic metrics (loss, LR) every N steps |
| `--log-heavy-every` | | int | `50` | Log per-layer gradient norms every N steps |
| `--sample-every-n-epochs` | | int | `0` | Generate an audio sample every N epochs. `0` = disabled |

### Preprocessing

These flags trigger the preprocessing pipeline. Add `--preprocess` to activate.

| Flag | Short | Type | Default | Description |
| --- | --- | --- | --- | --- |
| `--preprocess` | | bool | `False` | Run preprocessing before training |
| `--audio-dir` | | str | `None` | Source audio directory |
| `--dataset-json` | | str | `None` | Labeled dataset JSON file (lyrics, genre, BPM) |
| `--tensor-output` | | str | `None` | Output directory for `.pt` tensor files |
| `--max-duration` | | float | `0` | Max audio duration in seconds. `0` = auto-detect from dataset |
| `--normalize` | | str | `none` | Audio normalization: `none`, `peak` (-1.0 dBFS), `lufs` (-14 LUFS). LUFS requires `pyloudnorm` |

### Corrected Training

These control the corrected training behavior for base/sft models. Auto-disabled for turbo. See [[Loss Weighting and CFG Dropout]].

| Flag | Short | Type | Default | Description |
| --- | --- | --- | --- | --- |
| `--cfg-ratio` | | float | `0.15` | CFG dropout probability. The base model was trained with 0.15 -- match it |
| `--loss-weighting` | | str | `none` | Loss weighting strategy: `none` (flat MSE) or `min_snr` |
| `--snr-gamma` | | float | `5.0` | Gamma for min-SNR weighting (only with `min_snr`) |

---

## `fisher` -- Adaptive LoRA Rank Assignment

Analyzes the dataset using Fisher Information and Spectral metrics to generate a Preprocessing++ map for variable-rank LoRA targeting.

### Model / Paths

| Flag | Short | Type | Default | Description |
| --- | --- | --- | --- | --- |
| `--checkpoint-dir` | `-c` | str | **(required)** | Path to checkpoints root directory |
| `--model-variant` | | str | `turbo` | Model variant or subfolder name |
| `--base-model` | | str | `None` | Base model a fine-tune was trained from |

### Device / Platform

| Flag | Short | Type | Default | Description |
| --- | --- | --- | --- | --- |
| `--device` | | str | `auto` | Device selection |
| `--precision` | | str | `auto` | Precision: `auto`, `bf16`, `fp16`, `fp32` |

### Analysis

| Flag | Short | Type | Default | Description |
| --- | --- | --- | --- | --- |
| `--dataset-dir` | `-d` | str | **(required)** | Directory containing preprocessed `.pt` files |
| `--batch-size` | | int | `1` | Batch size for forward passes |
| `--num-workers` | | int | `4` (Linux), `0` (Windows) | DataLoader workers |
| `--seed` | | int | `42` | Random seed |
| `--fisher-batches` | | int | `32` | Number of batches for Fisher estimation |
| `--spectral-batches` | | int | `32` | Number of batches for Spectral estimation |
| `--target-average-rank` | | float | `64.0` | Target average rank across all chosen layers |
| `--max-rank` | | int | `256` | Maximum rank allowed for any single layer |
| `--min-rank` | | int | `16` | Minimum rank allowed for any selected layer |
| `--top-k-layers` | | int | `32` | Number of top layers to select for training |
| `--output-dir` | | str | `None` | Path to write the output JSON (default: inside dataset dir) |

---

## `compare-configs` -- Compare Module Configs

Side-by-side comparison of module config JSON files (e.g. from Preprocessing++).

| Flag | Short | Type | Default | Description |
| --- | --- | --- | --- | --- |
| `--configs` | | list | **(required)** | Paths to module config JSON files. Space-separated, at least two |

```bash
uv run python train.py compare-configs --configs pp_turbo.json pp_base.json
```

---

## `build-dataset` -- Build Dataset JSON

Scans a folder of audio files and builds a `dataset.json` with metadata from sidecar files.

| Flag | Short | Type | Default | Description |
| --- | --- | --- | --- | --- |
| `--input` | `-i` | str | **(required)** | Root directory containing audio files (scanned recursively) |
| `--tag` | | str | `""` (none) | Custom trigger tag applied to all samples |
| `--tag-position` | | str | `prepend` | Tag placement: `prepend`, `append`, `replace` |
| `--genre-ratio` | | int | `0` | Percentage of samples that use genre instead of caption (0-100) |
| `--name` | | str | `local_dataset` | Dataset name in metadata block |
| `--output` | | str | `None` | Output JSON path. Default: `<input>/dataset.json` |

See [[Dataset Preparation]] for the JSON schema and sidecar file format.

---

## Quick Reference: Defaults by Model Variant

Some arguments are automatically adjusted based on the detected model. You can override them, but the defaults are:

| Argument | Turbo | Base / SFT |
| --- | --- | --- |
| `--shift` | `3.0` | `1.0` |
| `--num-inference-steps` | `8` | `50` |
| `--cfg-ratio` | disabled | `0.15` |

See [[Shift and Timestep Sampling]] and [[Loss Weighting and CFG Dropout]] for details.

---

## See Also

- [[Training Guide]] -- LoRA vs LoKR walkthrough with CLI examples
- [[The Settings Wizard]] -- Every wizard setting and what it maps to
- [[VRAM Optimization Guide]] -- VRAM profiles and optimization flags
- [[Dataset Preparation]] -- Preparing audio and metadata for training
