# Side-Step for ACE-Step 1.5

```
 ░▒▓███████▓▒░▒▓█▓▒░▒▓███████▓▒░░▒▓████████▓▒░░▒▓███████▓▒░▒▓████████▓▒░▒▓████████▓▒░▒▓███████▓▒░ 
░▒▓█▓▒░      ░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░      ░▒▓█▓▒░         ░▒▓█▓▒░   ░▒▓█▓▒░      ░▒▓█▓▒░░▒▓█▓▒░
░▒▓█▓▒░      ░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░      ░▒▓█▓▒░         ░▒▓█▓▒░   ░▒▓█▓▒░      ░▒▓█▓▒░░▒▓█▓▒░
 ░▒▓██████▓▒░░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░▒▓██████▓▒░  ░▒▓██████▓▒░   ░▒▓█▓▒░   ░▒▓██████▓▒░ ░▒▓███████▓▒░ 
       ░▒▓█▓▒░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░             ░▒▓█▓▒░  ░▒▓█▓▒░   ░▒▓█▓▒░      ░▒▓█▓▒░       
       ░▒▓█▓▒░▒▓█▓▒░▒▓█▓▒░░▒▓█▓▒░▒▓█▓▒░             ░▒▓█▓▒░  ░▒▓█▓▒░   ░▒▓█▓▒░      ░▒▓█▓▒░       
░▒▓███████▓▒░░▒▓█▓▒░▒▓███████▓▒░░▒▓████████▓▒░▒▓███████▓▒░   ░▒▓█▓▒░   ░▒▓████████▓▒░▒▓█▓▒░       
 by dernet -- BETA
```

**Standalone training toolkit for ACE-Step 1.5 audio generation models.**
Takes you from raw audio files to a working adapter without the friction. Variant-aware multi-adapter fine-tuning (LoRA, DoRA, LoKR, LoHA, OFT) with auto-detection, low-VRAM support, and three ways to work.

> **Status:** v1.1.0-beta -- Stable enough for daily use. Some features are still experimental. This is maintained by one person only; if you encounter an issue, please let me know in the issues tab.

## Why Side-Step?

Side-Step auto-detects your model variant (base, sft, or turbo), selects the scientifically correct training schedule, and runs on consumer hardware down to 8 GB VRAM. Version 1.1.0 adds cruise control training, a completely overhauled captioning pipeline, and polished presets — building on the full standalone suite introduced in 1.0.0.

### What was already here

- **Auto-Configured Training** -- Turbo gets discrete 8-step sampling. Base/SFT gets continuous logit-normal + CFG dropout. The upstream trainer forces the Turbo schedule on all models; Side-Step fixes this automatically.
- **LoRA + LoKR Adapters** -- Standard and Kronecker-product low-rank fine-tuning.
- **Preprocessing++ (PP++)** -- Fisher Information analysis assigns adaptive per-module ranks based on how important each layer is to *your specific audio*. Writes a `fisher_map.json` that training auto-detects.
- **Two-Pass Preprocessing** -- Converts raw audio to training tensors in two low-memory passes (~3 GB then ~6 GB).
- **Interactive Wizard** -- Step-by-step prompts with "Go Back" support, presets, flow chaining, and session carry-over defaults.
- **Dataset Builder** -- Point at a folder of audio + sidecar `.txt` files and get a `dataset.json` automatically.
- **Low VRAM** -- 8-bit optimizers, gradient checkpointing, encoder offloading. Trains down to ~10 GB.
- **Standalone & Portable** -- Installs as its own project via `uv`. No need to touch your ACE-Step installation.

### New in 1.0.0

- **Full Electron GUI** -- Desktop application with Ez Mode (3-click training), Advanced Mode (every knob), real-time Monitor (loss charts, GPU stats), and a Lab workspace (datasets, preprocessing, PP++, export). CRT shader with phosphor bloom, scanlines, and chromatic aberration. Themeable (4 built-in themes + full editor).
- **DoRA, LoHA, OFT Adapters** -- Three additional adapter architectures alongside LoRA and LoKR.
- **ComfyUI Export** -- Convert PEFT LoRA/DoRA adapters to the single-file `.safetensors` format ComfyUI expects. LyCORIS adapters (LoKR, LoHA) are already natively compatible.
- **AI Captioning** -- Generate sidecar metadata with **local AI** (Qwen2.5-Omni, no API key, runs on your GPU), Google Gemini, OpenAI, or lyrics scraped from Genius.
- **Offline Audio Analysis** -- BPM, key, and time signature extraction via `demucs` stem separation + `librosa`. No API keys required.
- **Built-in Music Player** -- Play dataset audio directly in the GUI. Marquee display, EQ visualizer, volume control, auto-play, dockable bar.
- **Live VRAM Estimation** -- Segmented bar shows model + activation + optimizer breakdown before you start training. Changes reactively as you adjust settings.
- **VRAM Presets** -- One-click profiles: 8 GB, 12 GB, 16 GB, 24 GB+, Quick Test, High Quality, Recommended.
- **Run History** -- Persistent log of past training runs with best loss, adapter path, and hyperparameters.
- **Tag Management** -- Bulk add/remove trigger tags and convert legacy sidecar formats.
- **Cross-Platform Entry Point** -- `sidestep` (or `uv run sidestep` if not on PATH) works on all platforms.

### New in 1.1.0

- **Cruise Control (Target Loss)** -- Set a target loss value and Side-Step automatically damps the learning rate as training approaches it, holding the model at a sweet spot instead of over-fitting past it. EMA-smoothed loss signal, configurable warmup and floor. Works with all schedulers (conflict guards for Prodigy and cosine restarts). Resumes cleanly from checkpoints.
- **Caption System Overhaul** -- Richer song-focused prompts that emphasize audible content over generic descriptions. Configurable generation parameters (temperature, top_p, penalties). Structured response parsing extracts genre, BPM, key, and time signature alongside the caption. Google Search grounding for Gemini. Lossless audio auto-converted to MP3 before upload to save bandwidth.
- **Local Captioner Rewrite** -- Qwen2.5-Omni local captioner rebuilt with tiered VRAM configs, OOM recovery (retries with reduced token count), CPU offload option, audio transcoding fallback, cancellation support, and timing logs.
- **Default Model Variant: Base** -- Base is now the recommended default everywhere (CLI, GUI, wizard, TUI, presets). Turbo remains available but is no longer the automatic first choice. Model variant dropdown auto-selects base > sft > turbo based on what's available in your checkpoint directory.
- **Preset Revamp** -- All 7 built-in presets fleshed out with complete field coverage (adapter type, cruise control, checkpointing ratio, etc.). Presets now display adapter type, rank, LR, and epochs in the selection card. Type coercion fixes presets saved with numbers as strings.
- **Encoding Error Resilience** -- Genius, Gemini, and OpenAI providers now detect encoding errors (including errors wrapped by SDK exception types) and bail immediately instead of retrying 3× on deterministic failures. Saves ~70 seconds per batch when processing songs with non-ASCII titles.
- **Linux Desktop Integration** -- `.desktop` file and icon installed to XDG standard locations by the Linux installer. Side-Step appears in your application menu with its own icon.
- **Electron Hardening** -- Navigation guard prevents blank-page crashes, renderer crash detection, DevTools shortcut (F12), native desktop notifications for training completion.
- **Prompt Helpers Fix** -- `ask()` now correctly casts default values through `type_fn`, fixing numeric wizard defaults that were silently returned as strings.

---

## The Three Ways to Use Side-Step

> The experimental TUI from 0.9.0 and before has been deprecated. The interactive Wizard is its definitive replacement.

### 1. The Desktop Window (GUI)

Visual training, dataset management, live charts, and CRT-classic aesthetics (if you manage to find it :3).

```bash
uv run sidestep gui
```

**Modes:** Ez Mode | Advanced | Monitor | Lab (History, Tensor Datasets, Audio Library, Preprocess, PP++, Export)

![Ez Mode](assets/Screenshots/Side-Step%20EZ%20Mode.png)
![Advanced Mode](assets/Screenshots/Side-Step%20Advanced%20Mode.png)
![Audio Library](assets/Screenshots/Side-Step%20Audio%20Library.png)

### 2. The Interactive Wizard

Terminal prompts with back-navigation, presets, and flow chaining (preprocess -> train, PP++ -> train, build dataset -> preprocess -> train).

```bash
uv run sidestep
```

![Wizard](assets/Screenshots/Side-Step%20Wizard%20Main.png)

### 3. The Command Line (CLI)

Automate pipelines or bypass menus entirely. Every argument has a `(default: X)` in `--help`.

```bash
uv run sidestep train \
    --checkpoint-dir ./checkpoints \
    --model base \
    --dataset-dir ./my_tensors \
    --output-dir ./output/my_lora \
    --adapter-type dora \
    --rank 64 --alpha 128 \
    --epochs 500
```

---

## Quick Install

### Linux / macOS
```bash
git clone https://github.com/koda-dernet/Side-Step.git
cd Side-Step
chmod +x install_linux.sh && ./install_linux.sh
```

### Windows
```powershell
git clone https://github.com/koda-dernet/Side-Step.git
cd Side-Step
.\install_windows.ps1
```

The installer handles Python 3.11, PyTorch, Electron, and all dependencies via `uv`. Flash Attention is pulled from pre-built wheels -- no 20-minute (or more) local compilation.

### Get Models

You need the ACE-Step 1.5 checkpoints. If you don't have them:
```bash
git clone https://github.com/ace-step/ACE-Step-1.5.git
cd ACE-Step-1.5 && uv sync && uv run acestep-download
```

---

## VRAM Profiles

Side-Step runs on everything from an RTX 3060 to an H100. Built-in presets configure these automatically.

| Profile | VRAM | Strategy |
| :--- | :--- | :--- |
| **Comfortable** | 24 GB+ | AdamW, Batch 2+, Rank 128 |
| **Standard** | 16-24 GB | AdamW, Batch 1, Rank 64 |
| **Tight** | 12-16 GB | AdamW8bit, Encoder offloading |
| **Minimal** | 8-10 GB | AdamW8bit, Offloading, Grad accumulation 8, Rank 16 |

Gradient checkpointing is **on by default**, reducing baseline VRAM to ~7 GB before optimizer state.

---

## Workflows

### Preprocessing

Convert raw audio into training tensors. Two-pass approach keeps peak VRAM low.

```bash
uv run sidestep preprocess \
    --audio-dir ./my_songs \
    --tensor-output ./my_tensors \
    --normalize peak
```

### Training

Train an adapter on preprocessed tensors. Side-Step detects your variant and applies the correct schedule.

```bash
uv run sidestep train \
    --checkpoint-dir ./checkpoints \
    --model base \
    --dataset-dir ./my_tensors \
    --output-dir ./output/my_lora \
    --epochs 500
```

### Preprocessing++ (Adaptive Ranks)

Find which layers matter most for your data, then allocate rank accordingly.

```bash
uv run sidestep analyze \
    --checkpoint-dir ./checkpoints \
    --model base \
    --dataset-dir ./my_tensors
```

Writes `fisher_map.json` into the dataset folder. Training auto-detects it and applies variable ranks.

### AI Captioning

Generate rich sidecar metadata for your audio files.

```bash
uv run sidestep captions \
    --audio-dir ./my_songs \
    --provider local_16gb       # or gemini, openai, lyrics_only
```

### Export to ComfyUI

```bash
uv run sidestep export \
    --adapter-dir ./output/my_lora/final \
    --target native
```

### Dataset Building

```bash
uv run sidestep dataset --input ./my_music_folder
```

---

## Complete Subcommand List

Run `uv run sidestep --help` for full details.

| Subcommand | Description |
| :--- | :--- |
| `train` | Train an adapter (LoRA, DoRA, LoKR, LoHA, OFT) |
| `preprocess` | Convert audio to .pt tensors (two-pass pipeline) |
| `analyze` | PP++ -- Fisher analysis for adaptive rank assignment |
| `audio-analyze` | Offline BPM, key, time signature extraction |
| `captions` | AI caption generation + lyrics scraping |
| `tags` | Bulk sidecar tag operations (add/remove triggers) |
| `dataset` | Build `dataset.json` from audio + sidecar folders |
| `convert-sidecars` | Migrate legacy sidecar formats |
| `history` | List past training runs and best loss values |
| `export` | Export adapter to ComfyUI `.safetensors` |
| `settings` | View/modify persistent configuration |
| `gui` | Launch the Electron desktop application |

---

## Technical Notes: Timestep Sampling

Side-Step ensures your fine-tuning matches the base model's original training distribution:

1. **Turbo models** -- Discrete 8-step sampling (matching inference).
2. **Base/SFT models** -- Continuous logit-normal sampling + CFG dropout (matching training).

The upstream trainer often forces the Turbo schedule on all models, which is incorrect for Base/SFT. Side-Step detects and fixes this automatically.

---

## Documentation

See `sidestep_documentation/` for detailed guides:

- [Getting Started](sidestep_documentation/Getting%20Started.md)
- [End-to-End Tutorial](sidestep_documentation/End-to-End%20Tutorial.md)
- [Dataset Preparation](sidestep_documentation/Dataset%20Preparation.md)
- [Training Guide](sidestep_documentation/Training%20Guide.md)
- [Preprocessing++](sidestep_documentation/Preprocessing++.md)
- [Preset Management](sidestep_documentation/Preset%20Management.md)
- [VRAM Optimization Guide](sidestep_documentation/VRAM%20Optimization%20Guide.md)
- [Shift and Timestep Sampling](sidestep_documentation/Shift%20and%20Timestep%20Sampling.md)
- [Using Your Adapter](sidestep_documentation/Using%20Your%20Adapter.md)
- [CLI Argument Reference](sidestep_documentation/CLI%20Argument%20Reference.md)
- [Windows Notes](sidestep_documentation/Windows%20Notes.md)

## License

[CC BY-NC-SA 4.0](LICENSE) — free for personal and research use with attribution. Commercial use requires written permission from the author.

---

Contributions are always welcome. The inherent novelty of Audio Transformer-Based Diffusion makes these scripts fresh, and your contributions help every one of us. Open an issue, send a PR, or just share your results.

Massive shoutout to [Signorlimone](https://github.com/Signorlimone) for designing and compositing the Side-Step logo.
