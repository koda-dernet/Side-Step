# Changelog

### What's new in 0.9.0-beta

**Wizard and UX upgrades:**
- **Preprocessing++ promoted to main wizard flow** -- "Gradient estimation and spectral complexity" has been moved/renamed to **Preprocessing++** in the main "What would you like to do?" menu with clearer wording and stronger safety messaging.
- **Turbo safeguard for Preprocessing++** -- Selecting turbo now requires explicit opt-in with a warning prompt; choosing "no" loops back to model selection.
- **Checkpoint path defaults unified** -- Preprocessing++, estimate, and preprocess now all default checkpoint discovery from saved settings (`get_checkpoint_dir()`), consistent with the rest of the wizard.
- **Removed stale standalone warning** -- The old base-ACE/vanilla warning text was removed from compatibility checks.
- **Session carry-over defaults** -- The interactive session now reuses key values (checkpoint/model/dataset) across actions, reducing repeated typing on iterative runs.
- **Training review checkpoint** -- Training now shows a concise review/confirm screen with quick section-jump edits before dependency checks and launch.
- **Non-blocking Preprocessing++ recommendation in training** -- Missing `fisher_map.json` now surfaces a clear recommendation instead of launching long inline analysis.
- **Cross-flow validation messaging** -- Training, estimate, and Preprocessing++ now share consistent model fallback and dataset JSON recovery tips.

**Training stability and performance:**
- **Decoder KV cache forced off during training** -- `use_cache=False` is now enforced independent of gradient checkpointing, avoiding avoidable VRAM pressure.
- **Heavy logging is truly optional** -- `log_heavy_every` now supports explicit disable (`0`), appears in basic wizard logging, and no longer incurs expensive per-layer calculations when disabled.
- **Turbo loss-weighting guardrail** -- Wizard namespace construction now forces `loss_weighting=none` for turbo selections, preventing stale `min_snr` carryover.
- **Removed dead `sample_every_n_epochs` surface** -- Fully removed from CLI args, config mapping, presets, and wizard prompts (non-TUI).

**Preprocessing++ analysis quality:**
- **Model-aware balanced timestep sampling** -- Uses model `config.json` timestep parameters (`mu`, `sigma`, `data_proportion`) during balanced analysis.
- **Deterministic per-run subset selection** -- Stabilizes analyzed sample sets across chunk groups for reproducibility.
- **Coverage reporting** -- Preprocessing++ preview and saved map now include sample coverage information and selected sample counts.

**Windows and cross-platform hardening:**
- **Manifest handling improved for Windows paths** -- Better invalid-escape diagnostics, path normalization, and fallback behavior when manifest content is malformed.
- **Attention backend transparency** -- Clearer FA2/SDPA fallback diagnostics for unsupported environments.

### What's new in 0.8.3-beta

**Removed broken feature:**
- **ComfyUI LoRA export removed** -- The PEFT-to-diffusers key conversion tool (added in 0.8.1) was based on incorrect assumptions about how ComfyUI loads ACE-Step LoRAs. ComfyUI nodes use `PeftModel.from_pretrained()` which natively handles PEFT's key format. The converter was stripping prefixes that PEFT's own loader needs, producing adapters that fail to load ("lora key not loaded" errors). **Your PEFT adapters already work in ComfyUI as-is** -- no conversion needed. The `convert` CLI subcommand, wizard menu option, and `export_utils.py` module have been removed.

**New feature:**
- **Genre ratio in dataset builder** -- The `build-dataset` command and wizard now accept a `--genre-ratio` percentage (0-100). During preprocessing, that fraction of samples will be conditioned on the short genre tag instead of the full caption, teaching the LoRA to generalize to genre-only prompts at inference time. Default is 0 (caption always used, same behavior as before).

### What's new in 0.8.2-beta

**Build dataset from folder:**
- **No more hand-editing JSON.** New "Build dataset from folder" option in the wizard main menu (and `python train.py build-dataset` CLI subcommand). Point it at a folder of audio files with matching `.txt` metadata and it generates a ready-to-use `dataset.json`. Supports three metadata conventions:
  - `Song.txt` with `key: value` pairs (caption, genre, bpm, key, lyrics)
  - `Song.caption.txt` + `Song.lyrics.txt` (upstream ACE-Step format)
  - No metadata files (caption derived from filename, marked as instrumental)

**Auto-detect max duration:**
- **No more guessing.** Preprocessing now auto-detects the longest audio clip and sets `max_duration` automatically. The wizard shows per-song duration feedback (name, length) so you can see exactly what's being processed.

**Audio normalization:**
- **Consistent loudness across training data.** Preprocessing now offers optional audio normalization before VAE encoding. Choose **peak** (-1.0 dBFS, matches ACE-Step output, no extra deps) or **LUFS** (-14 LUFS, broadcast standard, perceptually uniform). Available in both the wizard and CLI (`--normalize peak` or `--normalize lufs`).

**Latent chunking for data augmentation:**
- **Random windowing of long songs.** New `--chunk-duration` option (default: disabled, recommended: 60s) slices preprocessed latent tensors into random fixed-length windows on every training iteration. Each epoch sees different parts of each song, improving generalization and reducing VRAM for long audio. Chunks shorter than 60 seconds can hurt training quality -- the wizard warns about this.

**Per-attention-type projection selection:**
- **Independent self/cross projections.** When targeting both attention types, the wizard and CLI now let you choose different projections for self-attention and cross-attention independently. For example, train `q_proj v_proj` on self-attention while training all four projections on cross-attention. Previously, the same projection set was applied to both.
- **Wizard:** When you select "Both self-attention and cross-attention", the wizard now asks "Self-attention projections" and "Cross-attention projections" separately. Applies to both LoRA and LoKR.
- **CLI:** New `--self-target-modules` and `--cross-target-modules` flags. When used with `--attention-type both`, each set is prefixed and merged independently. When not provided, `--target-modules` applies to both (backward compatible).

**Refactoring:**
- Shared `_ask_attention_type` and `_ask_projections` helpers eliminate duplicated prompting logic between LoRA and LoKR wizard steps.

### What's new in 0.8.1-beta

**Variant-aware training (the big one):**
- **Auto-detects turbo vs base/sft.** Side-Step now reads the model variant at startup and selects the correct training strategy automatically. Turbo models use discrete 8-step timestep sampling (matching their inference schedule). Base and sft models use continuous logit-normal sampling + CFG dropout (matching how they were originally trained). No manual mode selection needed.
- **Vanilla mode deprecated.** The `vanilla` subcommand and the "Corrected vs Vanilla" wizard submenu are gone. Turbo-correct behavior is now handled natively inside `fixed` mode. If you were using vanilla mode, `fixed` mode now does the same thing better -- and it's fully standalone.
- **Simplified setup wizard.** First-run setup no longer asks about vanilla mode or ACE-Step installation paths. Just point at your checkpoints and go.

**New features:**
- ~~**ComfyUI LoRA export**~~ -- *Removed in 0.8.3. The conversion was based on incorrect assumptions and produced broken adapters. PEFT adapters load natively in ComfyUI without conversion.*
- **Min-SNR loss weighting** -- Optional `--loss-weighting min_snr` rebalances training loss so fine detail (timbre, transients, mixing) gets as much signal as gross structure. Most useful for base/sft models where the continuous timestep range creates an imbalance. Default is `none` (flat MSE, same as before).
- **TensorBoard timestep histograms** -- The sampled timestep distribution is now logged as a histogram. Verify that your training is sampling the right noise levels under Histograms > `train/timestep_distribution`.

### What's new in 0.8.0-beta

**Bug fixes:**
- **Fixed gradient checkpointing crash** -- Training with gradient checkpointing enabled (the default) would crash with `element 0 of tensors does not require grad`. The autograd graph was disconnecting through checkpointed segments because the `xt` input tensor wasn't carrying gradients. Now forces `xt.requires_grad_(True)` when checkpointing is active, matching ACE-Step's upstream behavior. This was the #1 blocker for new users.
- **Fixed training completing with 0 steps on Windows** -- Lightning Fabric's `setup_dataloaders()` was wrapping the DataLoader with a shim that yielded 0 batches on Windows, causing training to silently "complete" with 0 epochs and 0 steps. Reported by multiple users on RTX 3090 and other GPUs. The Fabric DataLoader wrapping is now skipped entirely (the model/optimizer are still Fabric-managed for mixed precision).
- **Fixed multi-GPU device selection** -- Using `cuda:1` (or any non-default GPU) no longer causes training to silently fail. The Fabric device setup has been rewritten to use `torch.cuda.set_device()` instead of passing device indices as lists.
- **LoRA save path fix** -- Adapter files (`adapter_config.json`, `adapter_model.safetensors`) are now saved directly into the output directory. Previously they were nested in an `adapter/` subdirectory, causing Gradio/ComfyUI to fail to find the weights at the path Side-Step reported.
- **Massive VRAM reduction** -- Gradient checkpointing is now ON by default and actually works (see above fix). Measured at ~7 GiB for batch size 1 on a 48 GiB GPU (15% utilization). Previously Side-Step had checkpointing off or broken, causing 20-42 GiB VRAM usage. This brings Side-Step well below ACE-Step's memory footprint.
- **0-step training detection** -- If training completes with zero steps processed, Side-Step now reports a clear `[FAIL]` error instead of a misleading "Training Complete" screen with 0 epochs.
- **Windows `num_workers` safety** -- Explicitly clamps `num_workers=0` on Windows even if overridden via CLI, preventing spawn-based multiprocessing crashes.

**Features:**
- **Inference-ready checkpoints** -- Intermediate checkpoints (`checkpoints/epoch_N/`) now save adapter files flat alongside `training_state.pt`. Point any inference tool directly at a checkpoint directory -- no more digging into nested subdirectories. Checkpoints are usable for both inference AND resume.
- **Resume support in basic training loop** -- The non-Fabric fallback loop now supports `--resume-from`, matching the Fabric path.
- **VRAM-tier presets** -- Four new built-in presets (`vram_24gb_plus`, `vram_16gb`, `vram_12gb`, `vram_8gb`) with tuned settings for each GPU tier. Rank, optimizer, batch size, and offloading are pre-configured for your VRAM budget.
- **Flash Attention 2 auto-installed** -- Prebuilt wheels are now a default dependency. No compilation, no `--extra flash`. Falls back to SDPA silently on unsupported hardware.
- **Banner shows version** -- The startup banner now displays the Side-Step version for easier bug reporting.

### What's new in 0.7.0-beta

- **Truly standalone packaging** -- Side-Step is now its own project with a real `pyproject.toml` and full dependency list. Install it with `uv sync` -- no ACE-Step overlay required. The installer now creates Side-Step alongside ACE-Step as sibling directories.
- **First-run setup wizard** -- On first launch, Side-Step walks you through configuring your checkpoint directory, ACE-Step path (if you want vanilla mode), and validates your setup. Accessible any time from the main menu under "Settings".
- **Model discovery with fuzzy search** -- Instead of hardcoded `turbo/base/sft` choices, the wizard now scans your checkpoint directory for all model folders, labels official vs custom models, and lets you pick by number or search by name. Fine-tunes with arbitrary folder names are fully supported.
- **Fine-tune training support** -- Train on custom fine-tunes by selecting their folder. Side-Step auto-detects the base model from `config.json`. If it can't, it asks which base the fine-tune descends from to condition timestep sampling correctly.
- **`--base-model` CLI argument** -- New flag for CLI users training on fine-tunes. Overrides timestep parameters when `config.json` doesn't contain them.
- **`--model-variant` accepts any folder name** -- No longer restricted to turbo/base/sft. Pass any subfolder name from your checkpoints directory (e.g., `--model-variant my-custom-finetune`).
- **`acestep.__path__` extension** -- When vanilla mode is configured, Side-Step extends its package path to reach ACE-Step's modules. No overlay, no symlinks, no `sys.path` hacks.
- **Settings persistence** -- Checkpoint dir, ACE-Step path, and vanilla intent are saved to `~/.config/sidestep/settings.json` and reused as defaults in subsequent sessions.

### What's new in 0.6.0-beta

- **Mostly standalone** -- The corrected (fixed) training loop, preprocessing pipeline, and wizard no longer require a base ACE-Step installation. All needed ACE-Step utilities are vendored in `_vendor/`. You only need the model checkpoint files. Vanilla training mode still requires base ACE-Step.
- **Enhanced prompt builder** -- Preprocessing now supports `custom_tag`, `genre`, and `prompt_override` fields from dataset JSON metadata, matching upstream feature parity without the AudioSample dependency.
- **Hardened metadata lookup** -- Dataset JSON entries with `audio_path` but no `filename` field are now handled correctly (basename is extracted as fallback key).

### What's new in 0.5.0-beta

- **LoKR adapter support (experimental)** -- Train LoKR (Low-Rank Kronecker) adapters via [LyCORIS](https://github.com/KohakuBlueleaf/LyCORIS) as an alternative to LoRA. LoKR uses Kronecker product factorization and may capture different patterns than LoRA. **This is experimental and may break.** The underlying LyCORIS + Fabric interaction has not been exhaustively tested across all hardware.
- **Restructured wizard menu** -- The main menu now offers "Train a LoRA" and "Train a LoKR" as distinct top-level choices, each leading to a corrected/vanilla sub-menu
- **Unified preprocessing** -- Preprocessing is adapter-agnostic: the same tensors work for both LoRA and LoKR. The adapter type only affects weight injection during training, not the data pipeline. *(Previously, LoKR had a separate preprocessing mode that incorrectly fed target audio into context latents, giving the decoder the answer during training and producing misleadingly low loss.)*
- **LoKR-aware presets** -- Presets now save and restore adapter type and all LoKR-specific hyperparameters

### What's new in 0.4.0-beta

- **Session loop** -- the wizard no longer exits after each action; preprocess, train, and manage presets in one session
- **Go-back navigation** -- type `b` at any prompt to return to the previous question
- **Step indicators** -- `[Step 3/8] LoRA Settings` shows your progress through each flow
- **Presets system** -- save, load, import, and export named training configurations
- **Flow chaining** -- after preprocessing, the wizard offers to start training immediately
- **Experimental submenu** -- gradient estimation and upcoming features live here
- **GPU cleanup** -- memory is released between session loop iterations to prevent VRAM leaks
- **Config summaries** -- preprocessing and estimation show a summary before starting
- **Basic/Advanced mode** -- choose how many questions the training wizard asks
