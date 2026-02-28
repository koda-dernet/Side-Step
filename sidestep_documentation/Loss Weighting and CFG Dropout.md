# Loss Weighting and CFG Dropout

This page explains the settings available in Side-Step's corrected (fixed) training mode. These control how the training loss is computed and how training timesteps are sampled. All settings have sensible defaults -- you do not need to change any of them for a standard training run.

> **TL;DR:** Leave everything at defaults for your first run. If you're training on a **base** or **sft** model and want to experiment with potentially better quality, try turning on `min_snr` loss weighting. The other settings are for advanced users who want fine-grained control over the training distribution.

---

## Loss Weighting

### What it is

Every training step, the model looks at your audio at a random "noise level" (the timestep `t`) and tries to predict what the clean audio looks like. The loss measures how wrong it was.

The problem: predicting audio from heavy noise (high `t`) is inherently harder and produces bigger loss numbers than predicting from light noise (low `t`). Without weighting, the high-noise samples dominate the gradient and the model spends most of its effort learning big-picture structure rather than fine detail.

**Loss weighting rebalances this** so every noise level contributes more equally to what the adapter learns.

### Settings

| Setting | CLI flag | Default | Values |
|---------|----------|---------|--------|
| Loss weighting | `--loss-weighting` | `none` | `none`, `min_snr` |
| SNR gamma | `--snr-gamma` | `5.0` | Any positive float |

### `none` (default)

Standard flat MSE loss. Every timestep contributes equally by magnitude. This is what the model was originally trained with, and what Side-Step has always used.

Use this when:
- You want the safest, most predictable behavior
- You're training on **turbo** (which only uses 8 discrete timesteps in a narrow range, so the imbalance is small)
- You're not sure what to pick

### `min_snr`

Min-SNR-gamma weighting (from the paper "Efficient Diffusion Training via Min-SNR Weighting Strategy"). This is the same technique used in Stable Diffusion 3 and Flux community fine-tuning.

It works by turning down the loss contribution from very noisy samples (where the model can't predict much anyway) and preserving the contribution from cleaner samples (where fine detail matters). The result is that your adapter spends more training effort on the things that actually affect perceived quality -- timbre, transient sharpness, mixing character -- rather than on gross structure that the base model already handles well.

**This can yield better results on SFT and base models**, where the continuous timestep range means the noise-level imbalance is most pronounced. For turbo models, the effect is smaller but still valid.

Use this when:
- Training on **base** or **sft** models
- Your LoRA output sounds "mushy" or lacks detail compared to the base model
- You want to experiment with quality improvements

### SNR gamma

Only used when `loss_weighting` is `min_snr`. Controls how aggressively the weighting rebalances.

- **Higher gamma** (e.g. 10.0) = less aggressive rebalancing. Behavior approaches flat MSE.
- **Lower gamma** (e.g. 1.0) = more aggressive. Strongly downweights noisy samples.
- **Default 5.0** is a well-tested middle ground from the SD3/Flux community.

You almost certainly don't need to change this. If you do experiment:
- Try `1.0` if your output still sounds mushy even with `min_snr` on
- Try `10.0` if `min_snr` seems to hurt macro structure (unlikely but possible)

---

## CFG Dropout Ratio

| Setting | CLI flag | Default | Range |
|---------|----------|---------|-------|
| CFG dropout ratio | `--cfg-ratio` | `0.15` | 0.0 -- 1.0 |

### What it is

During training, 15% of samples randomly have their text/lyric conditioning replaced with a null embedding. This teaches the model to generate both *with* and *without* text guidance, which is necessary for classifier-free guidance (CFG) to work at inference.

At inference, CFG amplifies the difference between the conditioned and unconditioned predictions:

```
output = unconditioned + guidance_scale * (conditioned - unconditioned)
```

Without CFG dropout during training, the model never learns the unconditioned path, and CFG at inference produces garbage.

### When to change it

Almost never. The base/sft models were pre-trained with `cfg_ratio=0.15`, so matching that is correct.

- **Lower** (e.g. 0.05): The adapter relies more heavily on conditioning. CFG at inference will be weaker. Might work better if your dataset has very precise, detailed captions.
- **Higher** (e.g. 0.25): Stronger unconditional generation, stronger CFG effect. Might help if your captions are sparse or noisy.
- **0.0**: Disables CFG dropout entirely. Only do this for turbo models (which don't use CFG at inference).

> **Note:** For turbo models, Side-Step automatically disables CFG dropout regardless of this setting.

---

## Timestep Distribution

These control the shape of the noise-level distribution used during training. Side-Step reads them from the model's `config.json` by default, so you rarely need to touch them.

| Setting | CLI flag | Default | Source |
|---------|----------|---------|--------|
| Timestep mu | *(config panel only)* | `-0.4` | Model's `config.json` |
| Timestep sigma | *(config panel only)* | `1.0` | Model's `config.json` |
| Data proportion | *(config panel only)* | `0.5` | Model's `config.json` |

### How it works

Training timesteps are drawn from a **logit-normal distribution**: sample from a normal distribution with mean `mu` and standard deviation `sigma`, then pass through a sigmoid to get a value in (0, 1).

With the defaults (`mu=-0.4`, `sigma=1.0`), the distribution looks roughly like:

```
Density
  ^
  |          ***
  |        **   **
  |       *       *
  |      *         **
  |    **            **
  |  **                ***
  +-----------------------------> t
  0    0.2   0.4   0.6   0.8  1.0
          peak ~ 0.4
```

Most samples land in the 0.2--0.7 range. The tails (near 0 and 1) are undersampled.

### When to change them

These are expert settings. The defaults match how the model was pre-trained, which is the correct starting point for LoRA/LoKR fine-tuning.

If you understand the training dynamics and want to experiment:

- **More detail focus** (`mu=-0.8, sigma=0.8`): Shifts sampling toward lower timesteps where fine detail is learned. May improve output crispness at the cost of macro structure.
- **More structure focus** (`mu=0.0, sigma=0.8`): Shifts sampling toward higher timesteps. May help learn song structure or composition patterns.
- **Wider coverage** (`mu=-0.4, sigma=1.4`): Broader distribution. Useful for large domain shifts where you need the adapter to learn everything.
- **Tighter style focus** (`mu=-0.2, sigma=0.7`): Concentrates on mid-range timesteps where style/timbre lives. May be more efficient for pure style transfer with small datasets.

> **Tip:** If you change these, check the timestep distribution histogram in TensorBoard (under `train/timestep_distribution`) to verify the distribution looks like what you intended. See the monitoring section below.

---

## Monitoring in TensorBoard

Side-Step logs a **timestep distribution histogram** to TensorBoard every `log_heavy_every` steps (default: 50). You can find it under the **Distributions** or **Histograms** tab in TensorBoard at:

```
train/timestep_distribution
```

This shows you the actual noise levels your training is sampling. Useful for:

- **Verifying defaults**: Confirm the logit-normal shape looks right
- **Checking custom distributions**: If you changed `timestep_mu` or `timestep_sigma`, the histogram shows the real effect
- **Comparing runs**: See if different distribution settings are actually producing different sampling patterns

To view it:

```bash
tensorboard --logdir ./output/my_lora/runs
# Open http://localhost:6006, go to Histograms tab
```

---

## Quick Reference

### Recommended settings by model variant

| Setting | Turbo | Base | SFT |
|---------|-------|------|-----|
| `loss_weighting` | `none` | `none` or `min_snr` | `none` or `min_snr` |
| `snr_gamma` | `5.0` | `5.0` | `5.0` |
| `cfg_ratio` | `0.15` (auto-disabled) | `0.15` | `0.15` |
| `timestep_mu` | auto | auto | auto |
| `timestep_sigma` | auto | auto | auto |

### CLI examples

```bash
# Standard training (all defaults)
uv run python train.py fixed \
    --checkpoint-dir ../ACE-Step-1.5/checkpoints \
    --model-variant base \
    --dataset-dir ./my_tensors \
    --output-dir ./output/my_lora

# With min-SNR weighting for base/sft
uv run python train.py fixed \
    --checkpoint-dir ../ACE-Step-1.5/checkpoints \
    --model-variant base \
    --dataset-dir ./my_tensors \
    --output-dir ./output/my_lora \
    --loss-weighting min_snr

# With min-SNR and custom gamma
uv run python train.py fixed \
    --checkpoint-dir ../ACE-Step-1.5/checkpoints \
    --model-variant sft \
    --dataset-dir ./my_tensors \
    --output-dir ./output/my_lora \
    --loss-weighting min_snr \
    --snr-gamma 3.0
```

---

## See Also

- [[Training Guide]] -- Training modes, hyperparameters, and monitoring
- [[Shift and Timestep Sampling]] -- How timestep sampling works, what shift does, Side-Step vs upstream
- [[The Settings Wizard]] -- All wizard settings explained
- [[VRAM Optimization Guide]] -- GPU tiers and memory management
