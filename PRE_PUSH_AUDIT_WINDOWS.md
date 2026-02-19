# Pre-Push Audit (Windows Compatibility)

Date: 2026-02-18

Scope reviewed:
- Windows path normalization
- Manifest parsing/fallback behavior
- DataLoader safety defaults on Windows
- Wizard path defaults and prompts
- Runtime dependency prompt/install behavior
- Installer/version surface consistency

Excluded local assets:
- `mostercat_dataset.json`
- `mostercat_tensors.zip`
- `mostercat_tensors/`
- `mstct_lora/`
- `mstct_tensors/`

## Findings

## 1) Path and manifest robustness

- Windows-style path issues in tensor manifest handling are covered:
  - Invalid JSON (`Invalid \\escape`) is trapped with actionable messaging.
  - Fallback directory scan engages when manifest is broken/unusable.
  - `".C:\\..."` malformed absolute-path prefix is normalized.
  - Relative manifest entries with backslashes are normalized.

Assessment: Good and release-ready for known user error patterns.

## 2) Wizard defaults and path UX

- Checkpoint path default now consistently resolves from saved settings first in:
  - training flow
  - estimate flow
  - preprocessing flow
  - preprocessing++ flow
- Path prompt UX now distinguishes:
  - path not found
  - permission denied
  with actionable next steps.
- Model fallback prompts now include concrete folder-name examples when discovery fails.

Assessment: Good. Reduces repeated user path-entry friction on Windows.

## 3) DataLoader worker safety

- Windows worker clamping behavior remains in place (`num_workers=0` in wizard advanced path and runtime safeguards).

Assessment: Good. Prevents common `spawn` instability and deadlocks.

## 4) Attention backend transparency

- FA2/SDPA fallback messaging is explicit and user-facing.
- Users receive clearer diagnostics about unsupported FA2 conditions.

Assessment: Good for supportability and expectation setting.

## 5) Installer/version consistency

- Installer scripts now provide clearer preflight visibility and failure recovery:
  - install directory and system Python visibility
  - stricter uv PATH verification after installation
  - explicit next actions when ACE-Step deps/models are skipped or fail
- Version banners align with 0.9.0-beta.

Assessment: Addressed in this release update set.

## 6) Runtime optional dependency flow

- Prompt-before-install flow added for novice safety:
  - `adamw8bit` -> `bitsandbytes`
  - `prodigy` -> `prodigyopt`
  - logging -> `tensorboard`
  - LUFS preprocess -> `pyloudnorm`
- No silent auto-installs; users get explicit prompts and degraded-mode guidance.

Assessment: Good. Improves first-run recovery on Windows without hidden side effects.

## Residual Windows Risks

- Missing runtime dependencies on user machines (torch/loguru) still surface as environment errors.
- Long path edge-cases beyond normalized manifest entries can still depend on host Python/path settings.
- TUI is still not supported and should remain out-of-scope for Windows release promises.

## Recommended Windows Smoke Tests Before Push

- [ ] First-run wizard on clean settings file.
- [ ] Preprocessing with dataset JSON containing Windows-style paths.
- [ ] Fixed training launch on `cuda:0` with `num_workers=0`.
- [ ] Preprocessing++ turbo opt-in prompt branch (No then Yes path).
- [ ] Verify startup has no stale vanilla/base-ACE warning text.
- [ ] Validate runtime dependency prompts:
  - [ ] Select `adamw8bit` with missing bnb -> prompt appears
  - [ ] Select `prodigy` with missing prodigyopt -> prompt appears
  - [ ] Select LUFS normalize with missing pyloudnorm -> prompt + peak fallback path works

