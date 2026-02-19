# Side-Step Guide

Welcome to the Side-Step documentation vault. This guide covers installation, training, and model management for Side-Step 0.9.0-beta.

## Pages

- [[Getting Started]] -- Installation, first-run setup, prerequisites
- [[End-to-End Tutorial]] -- Raw audio to generated music walkthrough
- [[Dataset Preparation]] -- JSON schema, audio formats, metadata fields
- [[Training Guide]] -- LoRA vs LoKR, variant-aware training, wizard walkthrough, CLI examples
- [[Loss Weighting and CFG Dropout]] -- Loss weighting strategies, CFG dropout, timestep distribution tuning
- [[Using Your Adapter]] -- Output layout, loading in Gradio, ComfyUI export, LoKR limitations
- [[Model Management]] -- Checkpoint structure, fine-tunes, the "never rename" rule
- [[Preset Management]] -- Built-in presets, save/load/import/export
- [[The Settings Wizard]] -- All wizard settings reference
- [[VRAM Optimization Guide]] -- VRAM optimizations, GPU profiles
- [[Estimation Guide]] -- Gradient sensitivity analysis for targeted training
- [[Preprocessing++]] -- Dataset-aware adaptive rank and auto-targeting workflow
- [[Shift and Timestep Sampling]] -- How training timesteps work, what shift actually does, Side-Step vs upstream
- [[CLI Argument Reference]] -- Every `--flag` that `train.py` accepts, by subcommand
- [[Windows Notes]] -- num_workers, paths, installation, known workarounds

## Quick Links

- [Side-Step on GitHub](https://github.com/koda-dernet/Side-Step)
- [ACE-Step 1.5 on GitHub](https://github.com/ace-step/ACE-Step-1.5)
- [ACE-Step models on HuggingFace](https://huggingface.co/ACE-Step/Ace-Step1.5)

## Version

This guide is for **Side-Step 0.9.0-beta**. Side-Step is fully standalone -- no base ACE-Step installation needed.
