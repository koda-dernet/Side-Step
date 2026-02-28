
This page explains how timestep sampling works during training, what the `shift` parameter actually does, and why Side-Step's approach differs from the upstream community trainer.

> **TL;DR:** `shift` is an **inference-only** parameter. It does not affect the training loop. Side-Step auto-detects your model variant and selects the correct timestep sampling strategy: discrete 8-step for turbo, continuous logit-normal for base/sft. The `--shift` and `--num-inference-steps` settings are stored as metadata so you know what values to use when generating audio with your trained adapter.

---

## What shift does (inference only)

During **inference** (audio generation), `shift` warps the timestep schedule used by the diffusion ODE/SDE solver:

```
t_shifted = shift * t / (1 + (shift - 1) * t)
```

This formula appears in `generate_audio()` inside each model variant. It controls how denoising steps are distributed:

- **shift=1.0** -- Uniform linear schedule. Steps are evenly spaced from 1.0 to 0.0. This is the standard schedule and requires more steps (typically 50) for good quality. Used by **base** and **sft** models.
- **shift=3.0** -- Compressed schedule. More denoising happens at the high end (near t=1.0), less at the low end. This allows fewer steps (typically 8) with minimal quality loss. Used by **turbo** models.

The turbo model also has pre-computed discrete timestep tables for each shift value:

| Shift | 8-step schedule |
|-------|----------------|
| 1.0 | `[1.0, 0.875, 0.75, 0.625, 0.5, 0.375, 0.25, 0.125]` |
| 2.0 | `[1.0, 0.933, 0.857, 0.769, 0.667, 0.545, 0.4, 0.222]` |
| 3.0 | `[1.0, 0.955, 0.9, 0.833, 0.75, 0.643, 0.5, 0.3]` |

---

## What controls training timesteps

Side-Step **auto-detects** your model variant and selects the correct timestep sampling strategy:

### Turbo models: discrete 8-step sampling

Turbo models use a discrete timestep schedule matching the inference pipeline. Each training step uniformly picks one of these 8 values:

```python
TURBO_SHIFT3_TIMESTEPS = [1.0, 0.955, 0.9, 0.833, 0.75, 0.643, 0.5, 0.3]
```

This trains the adapter at exactly the timestep values it will see during 8-step inference. CFG dropout is not applied (turbo inference doesn't use classifier-free guidance).

### Base/SFT models: continuous logit-normal sampling

Base and sft models define the same `sample_t_r()` function in their model code:

```python
def sample_t_r(batch_size, device, dtype, data_proportion, timestep_mu, timestep_sigma, use_meanflow):
    t = torch.sigmoid(torch.randn((batch_size,)) * timestep_sigma + timestep_mu)
    r = torch.sigmoid(torch.randn((batch_size,)) * timestep_sigma + timestep_mu)
    t, r = torch.maximum(t, r), torch.minimum(t, r)
    # use_meanflow=False during training -> r = t for all samples
    ...
    return t, r
```

This is **logit-normal sampling**: draw from a normal distribution, then pass through a sigmoid to get values in (0, 1). The shape of this distribution is controlled by two parameters from the model's `config.json`:

| Parameter | Typical value | Effect |
|-----------|--------------|--------|
| `timestep_mu` | `-0.4` | Shifts the distribution center. Negative values bias toward lower timesteps |
| `timestep_sigma` | `1.0` | Controls spread. Larger values give a wider range of timesteps |

Side-Step reads these automatically from each model's `config.json` at startup. You never need to set them manually. CFG dropout (15% null-condition replacement) is applied to teach the model to handle both prompted and unprompted generation.

---

## Side-Step vs upstream trainer

### Side-Step (variant-aware)

Side-Step auto-detects the model variant and uses the correct strategy:

- **Turbo:** `sample_discrete_timesteps()` -- uniform over the 8-step inference schedule
- **Base/SFT:** `sample_timesteps()` -- continuous logit-normal, a **line-for-line reimplementation** of the model's own `sample_t_r()`:

```python
# Side-Step (timestep_sampling.py) -- used for base/sft
t = torch.sigmoid(
    torch.randn((batch_size,), device=device, dtype=dtype) * timestep_sigma + timestep_mu
)
```

**Result:** Side-Step correctly trains adapters for all three variants because it matches each model's actual training distribution.

### Upstream community trainer

The original ACE-Step community trainer (`acestep/training/trainer.py`) uses the same discrete 8-step schedule for **all** model variants:

```python
# Upstream trainer (trainer.py)
TURBO_SHIFT3_TIMESTEPS = [1.0, 0.955, 0.9, 0.833, 0.75, 0.643, 0.5, 0.3]
```

This approach:

1. **Is correct for turbo.** The discrete schedule matches turbo's 8-step inference pipeline.
2. **Is wrong for base and sft.** Those models were trained with continuous logit-normal sampling, not discrete uniform sampling. The upstream trainer applies turbo's schedule regardless.
3. **Lacks CFG dropout.** Base/sft models need null-condition dropout for classifier-free guidance to work at inference. The upstream trainer doesn't implement this.

---

## What the shift setting does in Side-Step

When you set `--shift` or configure shift in the wizard, Side-Step:

1. **Saves it to the training config JSON** alongside your adapter weights
2. **Does NOT use it during training** -- the training timestep distribution is entirely controlled by `timestep_mu` and `timestep_sigma` from the model config

The value is stored so that you (and anyone you share your adapter with) know what shift value to use at **inference time** when generating audio with the trained LoRA/LoKR.

### Recommended values

| Model variant | `--shift` | `--num-inference-steps` |
|--------------|-----------|------------------------|
| Turbo | `3.0` | `8` |
| Base | `1.0` | `50` |
| SFT | `1.0` | `50` |

The wizard auto-detects these from the model you select.

---

## FAQ

**Q: I changed `--shift` and my training results didn't change. Is this a bug?**
No. Shift does not affect training. The training timestep distribution comes from the model's own `timestep_mu`/`timestep_sigma` parameters, which Side-Step reads automatically.

**Q: If shift doesn't affect training, why is it a setting?**
It's metadata. When you finish training and want to generate audio with your LoRA, you need to know the correct shift value. Storing it with the adapter config prevents guesswork.

**Q: Can I train a turbo model and use it with shift=1.0 at inference?**
Technically yes, but the quality will differ from what the model expects. Also, i don't think it is fully supported by upstream. Use the shift value that matches the model variant you trained on.

**Q: Why does the upstream trainer use discrete timesteps?**
The upstream community trainer was written specifically for the turbo model. It takes the 8 discrete timestep values from the turbo inference schedule and samples uniformly from them. This is actually correct for turbo (and Side-Step now uses the same approach for turbo), but it does not work for base or sft models which need continuous sampling.

**Q: Side-Step used to use continuous sampling for turbo too. What changed?**
In 0.8.1, Side-Step switched turbo to discrete 8-step sampling to match the inference pipeline more closely. Training the adapter at exactly the timestep values it will see during inference is more efficient for turbo, where only 8 denoising steps are used. Base and sft still use continuous sampling because they use many more inference steps across the full timestep range.

---

## Source references

These are the relevant source locations for anyone who wants to verify:

- **Model's native training sampling:** `sample_t_r()` in `modeling_acestep_v15_turbo.py` (lines 169-194), `modeling_acestep_v15_base.py` (same), `sft/modeling_acestep_v15_base.py` (same)
- **Model's inference shift warp:** `generate_audio()` in each model file, applies `t = shift * t / (1 + (shift - 1) * t)`
- **Side-Step's continuous sampling (base/sft):** `sample_timesteps()` in `acestep/training_v2/timestep_sampling.py`
- **Side-Step's discrete sampling (turbo):** `sample_discrete_timesteps()` in `acestep/training_v2/timestep_sampling.py`
- **Upstream discrete sampling:** `sample_discrete_timestep()` in `acestep/training/trainer.py` (lines 302-323)

---

## See Also

- [[Training Guide]] -- Training modes, hyperparameters, and monitoring
- [[The Settings Wizard]] -- All wizard settings explained
- [[VRAM Optimization Guide]] -- GPU tiers and memory management
