# [BETA] Side-Step for ACE-Step 1.5

```bash
  ███████ ██ ██████  ███████       ███████ ████████ ███████ ██████
  ██      ██ ██   ██ ██            ██         ██    ██      ██   ██
  ███████ ██ ██   ██ █████   █████ ███████    ██    █████   ██████
       ██ ██ ██   ██ ██                 ██    ██    ██      ██
  ███████ ██ ██████  ███████       ███████    ██    ███████ ██
  by dernet     ((BETA TESTING))
```

**Side-Step** is a high-performance training "sidecar" for [ACE-Step 1.5](https://github.com/TencentGameMate/ACE-Step). It provides a corrected LoRA fine-tuning implementation that fixes fundamental bugs in the original trainer while adding low-VRAM support for local GPUs.

## 🚀 Why Side-Step?

The original ACE-Step trainer has two critical discrepancies from how the base models were actually trained. Side-Step was built to bridge this gap:

1.  **Continuous Timestep Sampling:** The original trainer uses a discrete 8-step schedule. This is fine for turbo, which the original training script is hardcoded for. Side-Step implements **Logit-Normal continuous sampling**, ensuring the model learns the full range of the denoising process.
2.  **CFG Dropout (Classifier-Free Guidance):** The original trainer lacks condition dropout. Side-Step implements a **15% null-condition dropout**, teaching the model how to handle both prompted and unprompted generation. Without this, inference quality suffers.
3.  **Non-Destructive Architecture:** It lives *alongside* your ACE-Step installation. It imports what it needs without touching a single line of the original source code.
4.  **Built for the cloud** The original Gradio breaks when you try to use it for training. Use this instead :)

---

## ⚠️ Beta Status & Support
**Current Version:** 0.2.0-beta

| Feature | Status | Note |
| :--- | :--- | :--- |
| **Fixed Training** | ✅ Working | Recommended for all users. |
| **Vanilla Training** | ✅ Working | For reproduction of old results. |
| **Interactive Wizard** | ✅ Working | `python train.py` with no args. |
| **TUI (Textual UI)** | ❌ **BROKEN** | Do not use `sidestep_tui.py` yet. |
| **CLI Preprocessing** | ❌ Planned | Use the Gradio UI for preprocessing for now. |
| **Gradient Estimation** | ❌ Planned | Coming in future update. |

---

## 📦 Installation

Side-Step is designed to be placed **inside** your existing ACE-Step 1.5 folder.

1. **Requirements:** Ensure you have ACE-Step 1.5 installed and working.
2. **Clone into ACE-Step:**
   ```bash
   cd path/to/ACE-Step
   git clone https://github.com/koda-dernet/Side-Step.git temp_side
   mv temp_side/* .
   rm -rf temp_side
   ```
3. **Install Dependencies:**
   (We recommend [uv](https://github.com/astral-sh/uv) for 10x faster installation and syncronization with the actual ACE-Step 1.5 project)
   ```bash
   # Using uv
   uv pip install -r requirements-sidestep.txt
   
   # Using standard pip
   pip install -r requirements-sidestep.txt
   ```
4. **Optional (Low VRAM):**
   ```bash
   uv pip install bitsandbytes>=0.45.0
   ```
5. **Optional (For bragging rights by using the Prodigy optimizer)**
   ```bash
   uv pip intall prodigyopt>=1.1.2
    ```
---
## Side-Note (ba dum tss) / Platform compatibility

While this is coded to be platform agnostic, being built for cloud computing and developed in a Linux system means this is mostly optmized for Linux or UNIX-Like systems. Windows and MacOS should in theory work as well. 

---

## 🛠️ Usage

### Option A: The Interactive Wizard (Recommended)
Simply run the script with no arguments. It will ask you everything it needs to know.
```bash
python train.py
```

### Option B: The Quick Start One-Liner
If you have your preprocessed tensors ready in `./my_data`, run:
```bash
python train.py fixed \
    --checkpoint-dir ./checkpoints \
    --model-variant turbo \
    --dataset-dir ./my_data \
    --output-dir ./output/my_lora \
    --epochs 100
```

---

## 📉 Optimization & VRAM Profiles
Side-Step is optimized for both heavy Cloud GPUs (H100/A100) and local "underpowered" gear (RTX 3060/4070).

| Profile | VRAM | Key Settings |
| :--- | :--- | :--- |
| **Comfortable** | 24GB+ | Standard AdamW, Batch 2+ |
| **Standard** | 16-24GB | Standard AdamW, Batch 1 |
| **Tight** | 10-16GB | **AdamW8bit**, Grad Checkpointing, Encoder Offloading |
| **Minimal** | <10GB | **AdaFactor** or **AdamW8bit**, Rank 16, High Grad Accumulation |

### Pro Features:
*   **`--offload-encoder`**: Moves the heavy VAE and Text Encoders to CPU after setup. Frees ~4GB VRAM.
*   **`--gradient-checkpointing`**: Drastically reduces memory usage during the backward pass.
*   **`--optimizer-type prodigy`**: Uses the Prodigy optimizer to automatically find the best learning rate for you.

---

## 📂 Project Structure
```text
.
├── train.py                 <-- Your main entry point
├── requirements-sidestep.txt
└── acestep/
    ├── training/            <-- Original ACE-Step code (Untouched)
    └── training_v2/         <-- Side-Step logic
        ├── trainer_fixed.py <-- The corrected logic
        ├── optim.py         <-- 8-bit and adaptive optimizers
        └── ui/              <-- Wizard and CLI visual logic
```

---
## Complete Argument Reference

Every argument, its default, and what it does.

### Global Flags

Available in: all subcommands (placed **before** the subcommand name)

| Argument | Default | Description |
|----------|---------|-------------|
| `--plain` | `False` | Disable Rich output; use plain text. Also set automatically when stdout is piped |
| `--yes` or `-y` | `False` | Skip the confirmation prompt and start training immediately |

### Model and Paths

Available in: vanilla, fixed

| Argument | Default | Description |
|----------|---------|-------------|
| `--checkpoint-dir` | **(required)** | Path to the root checkpoints directory (contains `acestep-v15-turbo/`, etc.) |
| `--model-variant` | `turbo` | Which model to use: `turbo`, `base`, or `sft` |
| `--dataset-dir` | **(required)** | Directory containing your preprocessed `.pt` tensor files and `manifest.json` |

### Device and Precision

Available in: all subcommands

| Argument | Default | Description |
|----------|---------|-------------|
| `--device` | `auto` | Which device to train on. Options: `auto`, `cuda`, `cuda:0`, `cuda:1`, `mps`, `xpu`, `cpu`. Auto-detection priority: CUDA > MPS (Apple Silicon) > XPU (Intel) > CPU |
| `--precision` | `auto` | Floating point precision. Options: `auto`, `bf16`, `fp16`, `fp32`. Auto picks: bf16 on CUDA/XPU, fp16 on MPS, fp32 on CPU |

### LoRA Settings

Available in: vanilla, fixed

| Argument | Default | Description |
|----------|---------|-------------|
| `--rank` or `-r` | `64` | LoRA rank. Higher = more capacity and more VRAM. Recommended: 64 (ACE-Step dev recommendation) |
| `--alpha` | `128` | LoRA scaling factor. Controls how strongly the adapter affects the model. Usually 2x the rank. Recommended: 128 |
| `--dropout` | `0.1` | Dropout probability on LoRA layers. Helps prevent overfitting. Range: 0.0 to 0.5 |
| `--attention-type` | `both` | Which attention layers to target. Options: `both` (self + cross attention, 192 modules), `self` (self-attention only, audio patterns, 96 modules), `cross` (cross-attention only, text conditioning, 96 modules) |
| `--target-modules` | `q_proj k_proj v_proj o_proj` | Which projection layers get LoRA adapters. Space-separated list. Combined with `--attention-type` to determine final target modules |
| `--bias` | `none` | Whether to train bias parameters. Options: `none` (no bias training), `all` (train all biases), `lora_only` (only biases in LoRA layers) |

### Training Hyperparameters

Available in: vanilla, fixed

| Argument | Default | Description |
|----------|---------|-------------|
| `--lr` or `--learning-rate` | `0.0001` | Initial learning rate. For Prodigy optimizer, set to `1.0` |
| `--batch-size` | `1` | Number of samples per training step. Usually 1 for music generation (audio tensors are large) |
| `--gradient-accumulation` | `4` | Number of steps to accumulate gradients before updating weights. Effective batch size = batch-size x gradient-accumulation |
| `--epochs` | `100` | Maximum number of training epochs (full passes through the dataset) |
| `--warmup-steps` | `100` | Number of optimizer steps where the learning rate ramps up from 10% to 100% |
| `--weight-decay` | `0.01` | Weight decay (L2 regularization). Helps prevent overfitting |
| `--max-grad-norm` | `1.0` | Maximum gradient norm for gradient clipping. Prevents training instability from large gradients |
| `--seed` | `42` | Random seed for reproducibility. Same seed + same data = same results |
| `--optimizer-type` | `adamw` | Optimizer: `adamw`, `adamw8bit` (saves VRAM), `adafactor` (minimal state), `prodigy` (auto-tunes LR) |
| `--scheduler-type` | `cosine` | LR schedule: `cosine`, `linear`, `constant`, `constant_with_warmup`. Prodigy auto-forces `constant` |
| `--gradient-checkpointing` | `False` | Recompute activations during backward to save VRAM (~40-60% less activation memory, ~30% slower) |
| `--offload-encoder` | `False` | Move encoder/VAE to CPU after setup. Frees ~2-4GB VRAM with minimal speed impact |

### Corrected Training (fixed mode only)

Available in: fixed

| Argument | Default | Description |
|----------|---------|-------------|
| `--cfg-ratio` | `0.15` | Classifier-free guidance dropout rate. With this probability, each sample's condition is replaced with a null embedding during training. This teaches the model to work both with and without text prompts. The model was originally trained with 0.15 |

### Data Loading

Available in: vanilla, fixed

| Argument | Default | Description |
|----------|---------|-------------|
| `--num-workers` | `4` | Number of parallel data loading worker processes. Set to 0 on machines with limited RAM |
| `--pin-memory` / `--no-pin-memory` | `True` | Pin loaded tensors in CPU memory for faster GPU transfer. Disable if you're low on RAM |
| `--prefetch-factor` | `2` | Number of batches each worker prefetches in advance |
| `--persistent-workers` / `--no-persistent-workers` | `True` | Keep data loading workers alive between epochs instead of respawning them |

### Checkpointing

Available in: vanilla, fixed

| Argument | Default | Description |
|----------|---------|-------------|
| `--output-dir` | **(required)** | Directory where LoRA weights, checkpoints, and TensorBoard logs are saved |
| `--save-every` | `10` | Save a full checkpoint (LoRA weights + optimizer + scheduler state) every N epochs |
| `--resume-from` | *(none)* | Path to a checkpoint directory to resume training from. Restores LoRA weights, optimizer state, and scheduler state |

### Logging and Monitoring

Available in: vanilla, fixed

| Argument | Default | Description |
|----------|---------|-------------|
| `--log-dir` | `{output-dir}/runs` | Directory for TensorBoard log files. View with `tensorboard --logdir <path>` |
| `--log-every` | `10` | Log loss and learning rate every N optimizer steps |
| `--log-heavy-every` | `50` | Log per-layer gradient norms every N optimizer steps. These are more expensive to compute but useful for debugging |
| `--sample-every-n-epochs` | `0` | Generate an audio sample every N epochs during training. 0 = disabled. (Not yet implemented) |

### Preprocessing (optional)

Available in: vanilla, fixed

| Argument | Default | Description |
|----------|---------|-------------|
| `--preprocess` | `False` (flag) | If set, run audio preprocessing before training |
| `--audio-dir` | *(none)* | Source directory containing audio files (for preprocessing) |
| `--dataset-json` | *(none)* | Path to labeled dataset JSON file (for preprocessing) |
| `--tensor-output` | *(none)* | Output directory where preprocessed .pt tensor files will be saved |
| `--max-duration` | `240` | Maximum audio duration in seconds. Longer files are truncated |

---

## 🤝 Contributing
Contributions are welcome! Specifically looking for help fixing the **Textual TUI** and completing the **CLI Preprocessing** module.

**License:** Follows the original ACE-Step 1.5 licensing
