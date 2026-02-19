# Pre-Push Audit (General)

Date: 2026-02-18

Scope reviewed:
- Core training flow (`fixed`, LoRA/LoKR)
- Wizard/menu UX and settings defaults
- Preprocessing and Preprocessing++ flow wiring
- Config build path, logging, and memory behavior
- Runtime optional-dependency prompt/install flow
- Novice-safe input validation and preset diagnostics
- Docs/version consistency

Excluded on purpose (local test assets, not part of release):
- `mostercat_dataset.json`
- `mostercat_tensors.zip`
- `mostercat_tensors/`
- `mstct_lora/`
- `mstct_tensors/`

## Summary

The branch is in a releasable state for the standalone `fixed` path with significant hardening in:
- path handling and dataset preflight
- attention backend diagnostics
- memory behavior (`use_cache` and logging overhead controls)
- prompt-before-install runtime dependency handling
- novice-friendly recovery guidance for common input mistakes
- Preprocessing++ user flow and naming
- Fisher/Preprocessing++ sensitivity pipeline quality

No blocker-level issues were found in static review for the touched paths.

## High-Impact Changes Verified

- **Standalone-only runtime messaging**
  - Removed stale base-ACE/vanilla warning path from compatibility check.
  - `check_compatibility()` now validates vendored modules only.

- **Memory and throughput safeguards**
  - Heavy per-layer logging can be disabled (`log_heavy_every=0`) and no-ops when TensorBoard writer is absent.
  - Timestep histogram logging is decoupled from heavy grad-norm logging and runs only in `min_snr` context.
  - Decoder `use_cache` is forced off for training regardless of gradient checkpointing mode.
  - Reduces risk of startup OOM when checkpointing is disabled.

- **Runtime dependency UX (prompt-before-install)**
  - New optional dependency preflight detects missing `bitsandbytes`, `prodigyopt`, `tensorboard`, and `pyloudnorm` when corresponding features are selected.
  - Wizard/CLI now offer explicit install prompts at runtime; no silent auto-installs.
  - If users decline installation, flow continues with clear degraded-mode impact messages.

- **Novice path and model guidance**
  - Better model-folder fallback messaging with concrete variant examples (`turbo/base/sft` or fine-tune folder name).
  - `ask_path` now distinguishes not-found vs permission-denied and provides actionable guidance.
  - CLI validation now prints available model folders and specific fix suggestions.

- **Preset diagnostics**
  - Import/export/view/delete now surface concrete reasons for failure (invalid JSON, oversized file, invalid name chars, not found) instead of generic failures.

- **Wizard consistency**
  - Checkpoint defaults now consistently use:
    - answers override -> settings checkpoint dir -> `./checkpoints`.
  - Preprocessing++ is exposed directly in the main menu.
  - Turbo usage in Preprocessing++ now requires explicit user opt-in with caution prompt.

- **Preprocessing++ pipeline quality**
  - Balanced timestep sampling now uses model-config-driven params (`mu`, `sigma`, `data_proportion`).
  - Chunk-group passes use deterministic per-run subsets for fair module comparisons.
  - Coverage metadata is recorded and previewed.

- **Dead option cleanup**
  - `sample_every_n_epochs` removed from non-TUI CLI/wizard/config/presets/docs surfaces.

## Test/Validation Evidence

- Focused guardrail tests added and passing:
  - `tests/test_wizard_pp_guardrails.py`
- New targeted tests added and passing:
  - `tests/test_dependency_preflight.py`
  - `tests/test_preset_error_messages.py`
  - `tests/test_prompt_helpers_path_errors.py`
- Syntax checks performed on touched Python modules.
- Lint diagnostics for touched files reported clean.

## Residual Risks (Non-blocking)

- TUI still contains legacy references (expected, TUI marked broken).
- Repository still has broad in-progress changes from multiple tasks; recommend selective staging for release commit.
- Full runtime integration tests remain environment-dependent (PyTorch/loguru availability).

## Pre-Push Checklist

- [ ] Stage only intended release files (exclude local test assets above).
- [ ] Run quick smoke in interactive wizard:
  - [ ] Main menu shows Preprocessing++.
  - [ ] Preprocessing++ checkpoint default comes from settings.
  - [ ] Turbo selection triggers opt-in confirmation.
- [ ] Validate prompt-before-install behavior:
  - [ ] `adamw8bit` missing -> prompt for `bitsandbytes`
  - [ ] `prodigy` missing -> prompt for `prodigyopt`
  - [ ] LUFS selected -> prompt for `pyloudnorm`
- [ ] Run one fixed training smoke with `log_heavy_every=0`.
- [ ] Confirm README and Obsidian docs render as expected.

