# [BETA] Side-Step for ACE-Step 1.5

```bash
  ███████ ██ ██████  ███████       ███████ ████████ ███████ ██████
  ██      ██ ██   ██ ██            ██         ██    ██      ██   ██
  ███████ ██ ██   ██ █████   █████ ███████    ██    █████   ██████
       ██ ██ ██   ██ ██                 ██    ██    ██      ██
  ███████ ██ ██████  ███████       ███████    ██    ███████ ██
  by dernet     ((BETA TESTING))
```

**Side-Step** is a **standalone**, battery-included training toolkit for [ACE-Step 1.5](https://github.com/ace-step/ACE-Step-1.5) audio generation models.

It takes you from **raw audio files** to a **working LoRA/LoKR adapter** without the friction. It auto-detects your model variant (turbo, base, or sft), selects the scientifically correct training schedule, and runs on consumer hardware (down to 8GB VRAM with enough optimizations).

## Why Side-Step?

It is a complete toolkit for either fine grained, advanced and up to a point, novel approaches of diffusion transformer fine-tuning. Runs on cloud due to its terminal nature, and fixes some problems the original trainer has.

-   **Auto-Configured Training**: Training Turbo? You should stay on the discrete 8-steps sampling. What about Base or SFT? Continuous Logit Normal Sampling is your friend. Side-Step detects your model and automatically applies the correct math.
-   **Preprocessing++**: The latest implementation of something uncommon yet incredibly powerful on your hands. Gradient estimation and auto-ranking. Don't believe me? Try it!
-   **Low VRAM**: Trains down to **10GB GPUs**. Built-in support for 8-bit optimizers, gradient checkpointing, and encoder offloading. You still need overhead for your system though.
-   **Standalone & Portable**: Installs as its own project with `uv`. No need to mess with your base ACE-Step installation.
-   **Interactive Wizard**: Run `uv run train.py` and follow the prompts. Includes "Go Back", Presets, Flow Chaining, session carry-over defaults, and a final training review screen.
-   **Two-Pass Preprocessing**: Converts raw audio to training tensors in two low-memory passes (~3 GB then ~6 GB).
-   **Dataset Builder**: Point it at a folder of music files + text files, and it generates your `dataset.json` automatically.

## Quick Start

### 1. Install

We recommend [uv](https://docs.astral.sh/uv/) for instant, isolated environments.

**Windows (Easy Mode):**
Double-click `install_windows.bat`. It handles Python, CUDA, and dependencies for you.

**Linux**
```bash
# 1. Clone
git clone https://github.com/koda-dernet/Side-Step.git
cd Side-Step

# 2. Install dependencies (includes PyTorch + Flash Attention)
uv sync
```

### 2. Get Models

You need the ACE-Step 1.5 model checkpoints. If you don't have them, grab them from HuggingFace or the official repo:
```bash
git clone https://github.com/ace-step/ACE-Step-1.5.git
cd ACE-Step-1.5 && uv sync && uv run acestep-download
```

### 3. Run the Wizard

The wizard is the easiest way to train. It handles preprocessing, training configuration, and rank selection interactively.
Recent UX improvements make repeated runs faster by carrying forward key defaults (checkpoint/model/dataset) between actions, and by showing a compact review step before training starts.

```bash
uv run train.py
```

---

## Workflows

### 🎵 Preprocessing
Convert your audio into training tensors. Side-Step uses a two-pass approach to keep VRAM low.
**Wizard**: Main Menu > "Preprocess Audio"
**CLI**:
```bash
uv run train.py fixed \
    --preprocess \
    --audio-dir ./my_songs \
    --tensor-output ./my_tensors \
    --normalize peak
```

### 🧠 Training (LoRA/LoKR)
Train an adapter on your preprocessed tensors.
**Wizard**: Main Menu > "Train a LoRA"
**CLI**:
```bash
uv run train.py fixed \
    --checkpoint-dir ./checkpoints \
    --model-variant turbo \
    --dataset-dir ./my_tensors \
    --output-dir ./output/my_lora \
    --epochs 100
```

### 📂 Dataset Building
Generate a `dataset.json` from a folder of audio files (`.wav`/`.mp3`) and metadata files (`.txt`).
```bash
uv run train.py build-dataset --input ./my_music_folder
```

### 📊 Gradient Estimation
Not sure which layers to train? Use the estimator to find the most responsive modules for your specific audio data.
**Wizard**: Main Menu > "Preprocessing++"
For estimation-style flows, the wizard can also load overlapping rank defaults from presets so you can skip repetitive tuning input.

---

## Optimization & VRAM Profiles

Side-Step runs on everything from an RTX 3060 to an H100.

| Profile | VRAM | Strategy |
| :--- | :--- | :--- |
| **Comfortable** | 24 GB+ | AdamW, Batch 2+, Rank 128 |
| **Standard** | 16-24 GB | AdamW, Batch 1, Rank 64 |
| **Tight** | 12-16 GB | AdamW8bit, Encoder Offloading |
| **Minimal** | 10 GB | AdamW8bit, Offloading, Gradient Accumulation |

*Note: Gradient Checkpointing is **ON** by default, reducing VRAM usage to ~7GB.*

---

## Technical Notes: Timestep Sampling

Side-Step ensures your fine-tuning matches the base model's original training distribution:

1.  **Turbo Models**: Uses **discrete 8-step sampling** (matching inference).
2.  **Base/SFT Models**: Uses **continuous logit-normal sampling** + **CFG Dropout** (matching training).

The upstream trainer often forces the Turbo schedule on all models, which is incorrect for Base/SFT. Side-Step fixes this automatically.

## Support

-   **Changelog**: See [CHANGELOG.md](CHANGELOG.md) for version history.
-   **Logs**: Errors are written to `sidestep.log`.
-   **Beta**: This is beta software. If an update breaks something, `git checkout` the previous commit.

---

<details>
<summary><strong>Click to expand: Complete Argument Reference</strong></summary>

### Global Flags
| Argument | Default | Description |
|----------|---------|-------------|
| `--plain` | `False` | Disable Rich output (plain text). |
| `--yes`, `-y` | `False` | Skip confirmation prompts. |

### Model & Paths (command: `fixed`)
| Argument | Default | Description |
|----------|---------|-------------|
| `--checkpoint-dir` | **Required** | Path to ACE-Step checkpoints. |
| `--model-variant` | `turbo` | `turbo`, `base`, `sft`, or custom folder name. |
| `--dataset-dir` | **Required** | Directory containing preprocessed `.pt` tensors. |
| `--output-dir` | **Required** | Where to save adapters and logs. |

### Training Parameters
| Argument | Default | Description |
|----------|---------|-------------|
| `--adapter-type` | `lora` | `lora` or `lokr`. |
| `--rank`, `-r` | `64` | LoRA rank / LoKR linear dim. |
| `--alpha` | `128` | Scaling factor (usually 2x rank). |
| `--epochs` | `100` | Total training epochs. |
| `--lr` | `1e-4` | Learning rate. |
| `--batch-size` | `1` | Samples per step. |
| `--gradient-accumulation`| `4` | Steps to accumulate before update. |
| `--optimizer-type` | `adamw` | `adamw`, `adamw8bit`, `adafactor`, `prodigy`. |
| `--scheduler-type` | `cosine` | LR scheduler. |

### Advanced & Optimization
| Argument | Default | Description |
|----------|---------|-------------|
| `--gradient-checkpointing`| `True` | Save VRAM by recomputing activations. |
| `--offload-encoder` | `False` | Move VAE/Text Encoder to CPU to save VRAM. |
| `--loss-weighting` | `none` | `none` or `min_snr` (rebalances loss). |
| `--cfg-ratio` | `0.15` | CFG Dropout (auto-disabled for Turbo). |
| `--chunk-duration` | `0` | Slice tensors into random N-second windows (augmentation). |

### Preprocessing (command: `fixed --preprocess`)
| Argument | Default | Description |
|----------|---------|-------------|
| `--preprocess` | `False` | Enable preprocessing mode. |
| `--audio-dir` | *None* | Input audio directory. |
| `--normalize` | `none` | `peak` (-1dB) or `lufs` (-14 LUFS). |

</details>

# Contributions?

Contributions are always welcome, theres a lot we can start talking about, if unsure, here's a list of things that do not work yet:

- The TUI is broken
- No support for Apple Silicon
- No support for AMD ROCm
- The inherent novelty of Audio Transformer-Based Diffusion makes these scripts fresh, but also hindered. You can always help little by little, with ideas or full on implementations. The sky is the limit and your contributions help every one of us!
