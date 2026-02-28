/* ============================================================
   Side-Step GUI — Behaviors & Reactivity
   Model-derived defaults, loss weighting, scheduler, optimizer,
   resume, ckpt presets, smart behaviors, datasets tab,
   form validation, prodigy note, save/export/import handlers,
   settings refresh, modal UX.
   Extracted from workspace.js for 400 LOC cap.
   ============================================================ */

const WorkspaceBehaviors = (() => {
  "use strict";

  function _updateConsole(msg) {
    const el = $("console-line");
    if (el) { el.textContent = msg; el.className = "console__line"; }
  }

  /* ---- Form Validation ---- */
  function initValidation() {
    const _reasons = (dsId, modelSel) => {
      const list = [];
      if (!$(dsId)?.value) list.push("Select a dataset");
      if (modelSel && !modelSel.options.length) list.push("No models found \u2014 check checkpoint directory in Settings");
      if (typeof Training !== "undefined" && Training.isRunning()) list.push("Training already in progress");
      return list;
    };
    const gateStart = () => {
      const ezReasons = _reasons("ez-dataset-dir", $("ez-model-variant"));
      const fullReasons = _reasons("full-dataset-dir", $("full-model-variant"));
      const ezBtn = $("btn-start-ez"), fullBtn = $("btn-start-full");
      if (ezBtn) { ezBtn.disabled = ezReasons.length > 0; ezBtn.title = ezReasons.join(". ") || ""; }
      if (fullBtn) { fullBtn.disabled = fullReasons.length > 0; fullBtn.title = fullReasons.join(". ") || ""; }
      const ezHint = $("ez-start-hint");
      if (ezHint) { ezHint.textContent = ezReasons.length ? ezReasons.join(" \u00b7 ") : ""; ezHint.style.display = ezReasons.length ? "" : "none"; }
      const fullHint = $("full-start-hint");
      if (fullHint) { fullHint.textContent = fullReasons.length ? fullReasons.join(" \u00b7 ") : ""; fullHint.style.display = fullReasons.length ? "" : "none"; }
    };
    $("ez-dataset-dir")?.addEventListener("change", gateStart);
    $("full-dataset-dir")?.addEventListener("change", gateStart);
    document.addEventListener("appstate:status", gateStart);
    gateStart();
  }

  /* ---- Prodigy Optimizer Note ---- */
  function initProdigyNote() {
    const optimizerSel = $("full-optimizer");
    const lrInput = $("full-lr");
    if (!optimizerSel || !lrInput) return;
    let noteEl = null;
    optimizerSel.addEventListener("change", () => {
      if (optimizerSel.value === "prodigy") {
        if (!noteEl) {
          noteEl = document.createElement("div");
          noteEl.className = "hint u-text-warning";
          noteEl.textContent = "Prodigy auto-tunes LR. The value above is the initial weight, typically set to 1.0.";
          lrInput.parentElement.appendChild(noteEl);
        }
        noteEl.style.display = "block";
      } else if (noteEl) {
        noteEl.style.display = "none";
      }
    });
  }

  /* ---- Save Preset Handler ---- */
  function initSavePreset() {
    const modal = $("save-preset-modal");
    const nameInput = $("save-preset-name");
    const descInput = $("save-preset-desc");
    const _open = () => { if (nameInput) nameInput.value = ""; if (descInput) descInput.value = ""; modal?.classList.add("open"); nameInput?.focus(); };
    const _close = () => { modal?.classList.remove("open"); };
    const _save = async () => {
      const name = nameInput?.value?.trim();
      if (!name) { nameInput?.focus(); return; }
      const desc = descInput?.value?.trim() || "";
      const data = typeof WorkspaceConfig !== "undefined" ? WorkspaceConfig.gatherFullConfig() : {};
      try {
        await API.savePreset(name, desc, data);
        _close();
        showToast("Preset '" + name + "' saved", "ok");
        _updateConsole("Preset saved: " + name);
        if (typeof WorkspaceSetup !== "undefined") WorkspaceSetup.initPresets();
      } catch (e) {
        showToast("Failed to save preset: " + e.message, "error");
      }
    };
    $("btn-save-preset")?.addEventListener("click", _open);
    $("save-preset-close")?.addEventListener("click", _close);
    $("save-preset-cancel")?.addEventListener("click", _close);
    $("save-preset-confirm")?.addEventListener("click", _save);
    nameInput?.addEventListener("keydown", (e) => { if (e.key === "Enter") _save(); });
  }

  /* ---- Export CSV Handler ---- */
  function initExportCSV() {
    $("btn-export-csv")?.addEventListener("click", async () => {
      const allRuns = await API.fetchHistory();
      const runs = allRuns.filter((r) => !r.detected_only);
      if (!runs.length) { showToast("No completed runs to export", "warn"); return; }
      const headers = ["run_name", "adapter", "model", "epochs", "best_loss", "duration", "status", "date"];
      const csv = [headers.join(",")];
      runs.forEach((r) => { csv.push(headers.map((h) => String(r[h] || "").replace(/,/g, ";")).join(",")); });
      const blob = new Blob([csv.join("\n")], { type: "text/csv" });
      const url = URL.createObjectURL(blob);
      const a = document.createElement("a");
      a.href = url; a.download = "sidestep_history.csv"; a.click();
      URL.revokeObjectURL(url);
      showToast("CSV exported", "ok");
      _updateConsole("History exported as CSV");
    });
  }

  /* ---- Open Output Dir Handler ---- */
  function initOpenOutputDir() {
    $("btn-open-output")?.addEventListener("click", async () => {
      const dir = $("monitor-output-dir")?.textContent?.replace("Output: ", "").trim() || "";
      if (!dir) { showToast("No output directory available", "warn"); return; }
      const result = await API.openFolder(dir);
      if (result.ok) showToast("Output directory opened", "ok");
      else showToast("Failed to open output folder: " + (result.error || dir), "error");
    });
  }

  /* ---- Import Preset Handler ---- */
  function initImportPreset() {
    const fileInput = document.createElement("input");
    fileInput.type = "file"; fileInput.accept = ".json"; fileInput.style.display = "none";
    document.body.appendChild(fileInput);

    $("btn-import-preset")?.addEventListener("click", () => fileInput.click());

    fileInput.addEventListener("change", async () => {
      const file = fileInput.files?.[0];
      if (!file) return;
      try {
        const text = await file.text();
        const data = JSON.parse(text);
        const name = data.name || file.name.replace(/\.json$/i, "");
        const desc = data.description || "";
        await API.savePreset(name, desc, data);
        showToast("Preset '" + name + "' imported", "ok");
        if (typeof WorkspaceSetup !== "undefined") WorkspaceSetup.initPresets();
      } catch (e) {
        showToast("Failed to import preset: " + e.message, "error");
      }
      fileInput.value = "";
    });
  }

  /* ---- Reusable Confirmation Modal ---- */
  let _confirmCallback = null;
  function _initConfirmModal() {
    const modal = $("confirm-modal");
    const _close = () => { modal?.classList.remove("open"); _confirmCallback = null; };
    $("confirm-modal-close")?.addEventListener("click", _close);
    $("confirm-modal-no")?.addEventListener("click", _close);
    $("confirm-modal-yes")?.addEventListener("click", () => {
      const cb = _confirmCallback;
      _close();
      if (typeof cb === "function") cb();
    });
  }

  /* ---- Modal UX: Escape + Backdrop + Enter-to-confirm ---- */
  function initModalUX() {
    const _enterTargets = [
      { modal: "confirm-modal",     btn: "confirm-modal-yes" },
      { modal: "resume-modal",      btn: "btn-resume-start" },
      { modal: "trigger-tag-modal", btn: "btn-bulk-trigger-apply" },
    ];

    document.addEventListener("keydown", (e) => {
      if (e.key === "Escape") {
        document.querySelectorAll(".modal.open, .modal.active").forEach((m) => {
          m.classList.remove("open", "active");
        });
        $("settings-panel")?.classList.remove("open");
        $("sidecar-editor")?.classList.remove("open");
        return;
      }

      if (e.key === "Enter") {
        // Don't hijack Enter inside textareas
        if (e.target?.tagName === "TEXTAREA") return;

        // Link-audio inline dialog (not a .modal)
        const linkDialog = $("datasets-link-dialog");
        if (linkDialog && linkDialog.style.display !== "none") {
          e.preventDefault();
          $("btn-datasets-link-confirm")?.click();
          return;
        }

        for (const { modal, btn } of _enterTargets) {
          const el = $(modal);
          if (el?.classList.contains("open")) {
            e.preventDefault();
            $(btn)?.click();
            return;
          }
        }
      }
    });

    document.querySelectorAll(".modal").forEach((modal) => {
      modal.addEventListener("click", (e) => {
        if (e.target === modal) {
          modal.classList.remove("open", "active");
        }
      });
    });
  }

  /* ---- Model-derived defaults ---- */
  function initModelDerivedDefaults() {
    const variantSel = $("full-model-variant");
    if (!variantSel) return;
    const update = () => {
      const v = variantSel.value;
      const vLower = String(v || "").toLowerCase();
      const inferredSteps = parseInt($("full-inference-steps")?.value || "8", 10);
      let isTurbo = false;
      let isBaseFamily = false;
      let inferredFromSteps = false;
      if (vLower.includes("turbo")) {
        isTurbo = true;
      } else if (vLower.includes("base") || vLower.includes("sft")) {
        isBaseFamily = true;
      } else {
        // Keep GUI behavior aligned with backend inference for custom names.
        isTurbo = inferredSteps === 8;
        isBaseFamily = !isTurbo;
        inferredFromSteps = true;
      }
      const shiftEl = $("full-shift");
      const stepsEl = $("full-inference-steps");
      if (shiftEl && shiftEl.value === shiftEl.dataset.default) {
        shiftEl.value = isTurbo ? "3.0" : "1.0";
        shiftEl.dataset.default = shiftEl.value;
      }
      if (stepsEl && stepsEl.value === stepsEl.dataset.default) {
        stepsEl.value = isTurbo ? "8" : "50";
        stepsEl.dataset.default = stepsEl.value;
      }
      const cfgGroup = $("cfg-settings-group");
      if (cfgGroup) cfgGroup.style.display = isTurbo ? "none" : "";
      const adaptiveGroup = $("adaptive-timestep-group");
      if (adaptiveGroup) adaptiveGroup.style.display = isTurbo ? "none" : "";
      const banner = $("full-strategy-banner");
      if (banner) {
        if (isTurbo) {
          banner.textContent = inferredFromSteps
            ? "Custom variant (inferred turbo from 8 steps) \u2014 discrete sampling, no CFG"
            : "Turbo \u2014 discrete 8-step sampling, no CFG";
        } else {
          banner.textContent = inferredFromSteps
            ? "Custom variant (inferred base/SFT from inference steps) \u2014 continuous sampling + CFG dropout"
            : "Base/SFT \u2014 continuous sampling + CFG dropout";
        }
      }
      const warnFull = $("warn-non-base-full");
      if (warnFull) warnFull.style.display = isBaseFamily ? "none" : "";
      const warnEz = $("warn-non-base-ez");
      if (warnEz) warnEz.style.display = isBaseFamily ? "none" : "";
    };
    variantSel.addEventListener("change", update);
    // W8: Also listen to ez-model-variant so warning works even without W3 sync
    $("ez-model-variant")?.addEventListener("change", update);
    update();
  }

  /* ---- Loss weighting ---- */
  function initLossWeightingReactivity() {
    const sel = $("full-loss-weighting");
    const snrGroup = $("snr-gamma-group");
    if (!sel || !snrGroup) return;
    sel.addEventListener("change", () => {
      snrGroup.style.display = sel.value === "min_snr" ? "" : "none";
    });
  }

  /* ---- Scheduler ---- */
  function initSchedulerReactivity() {
    const sel = $("full-scheduler");
    const formulaGroup = $("scheduler-formula-group");
    if (!sel || !formulaGroup) return;
    sel.addEventListener("change", () => {
      formulaGroup.style.display = sel.value === "custom" ? "" : "none";
    });
  }

  /* ---- Optimizer ---- */
  function initOptimizerReactivity() {
    const optSel = $("full-optimizer");
    const lrHint = $("prodigy-lr-hint");
    const schedSel = $("full-scheduler");
    if (!optSel) return;
    const update = () => {
      const isProdigy = optSel.value === "prodigy";
      if (lrHint) lrHint.style.display = isProdigy ? "" : "none";
      if (schedSel) {
        const customOpt = schedSel.querySelector('option[value="custom"]');
        if (customOpt) customOpt.disabled = isProdigy;
        if (isProdigy && schedSel.value === "custom") schedSel.value = "cosine";
      }
    };
    optSel.addEventListener("change", update);
    update();
  }

  /* ---- Resume ---- */
  function initResumeReactivity() {
    const resumeInput = $("full-resume-from");
    const strictGroup = $("strict-resume-group");
    if (!resumeInput || !strictGroup) return;
    const update = () => {
      strictGroup.style.display = resumeInput.value.trim() ? "" : "none";
    };
    resumeInput.addEventListener("input", update);
    resumeInput.addEventListener("change", update);
    update();
  }

  /* ---- Checkpointing ratio preset buttons ---- */
  function initCkptPresets() {
    const buttons = Array.from(document.querySelectorAll(".ckpt-preset"));
    const _setPressed = (btn, isPressed) => {
      if (!btn) return;
      btn.setAttribute("aria-pressed", isPressed ? "true" : "false");
    };
    const _setPresetActive = (ratio) => {
      buttons.forEach((b) => {
        const match = b.dataset.ratio === String(ratio);
        b.classList.toggle("active", match);
        _setPressed(b, match);
      });
    };

    buttons.forEach((btn) => {
      btn.addEventListener("click", () => {
        const ratio = btn.dataset.ratio;
        const input = $("full-grad-ckpt-ratio");
        if (input) { input.value = ratio; input.dispatchEvent(new Event("input")); }
        _setPresetActive(ratio);
        _updateConsole("Checkpointing ratio: " + ratio);
      });
    });
    $("full-grad-ckpt-ratio")?.addEventListener("input", () => {
      const val = $("full-grad-ckpt-ratio")?.value || "";
      _setPresetActive(val);
    });
    _setPresetActive($("full-grad-ckpt-ratio")?.value || "1.0");
  }

  /* ---- Smart Behaviors ---- */
  function initSmartBehaviors() {
    const _updateStepEstimate = () => {
      const dsDir = $("full-dataset-dir")?.value;
      if (!dsDir) { if ($("full-step-estimate")) $("full-step-estimate").style.display = "none"; return; }
      const infoEl = $("full-dataset-info");
      const sampleMatch = infoEl?.textContent?.match(/(\d+)\s*samples/);
      const samples = sampleMatch ? parseInt(sampleMatch[1], 10) : 0;
      if (!samples) { if ($("full-step-estimate")) $("full-step-estimate").style.display = "none"; return; }

      const bs = parseInt($("full-batch")?.value, 10) || 1;
      const ga = parseInt($("full-grad-accum")?.value, 10) || 4;
      const epochs = parseInt($("full-epochs")?.value, 10) || 100;
      const repeats = parseInt($("full-dataset-repeats")?.value, 10) || 1;
      const maxSteps = parseInt($("full-max-steps")?.value, 10) || 0;

      const effectiveSamples = samples * repeats;
      const stepsPerEpoch = Math.ceil(effectiveSamples / (bs * ga));
      const totalSteps = maxSteps > 0 ? maxSteps : stepsPerEpoch * epochs;

      const estEl = $("full-step-estimate");
      const textEl = $("full-step-estimate-text");
      if (estEl && textEl) {
        estEl.style.display = "";
        textEl.textContent = stepsPerEpoch + " steps/epoch x " + epochs + " epochs = " + totalSteps + " total optimizer steps" +
          (repeats > 1 ? " (" + repeats + "x repeats)" : "") +
          (maxSteps > 0 ? " (capped at " + maxSteps + ")" : "");
        textEl.dataset.spe = stepsPerEpoch; textEl.dataset.total = totalSteps; textEl.dataset.epochs = epochs;
      }
      const hintEl = $("step-estimate-hint");
      if (hintEl) hintEl.textContent = "~" + stepsPerEpoch + " steps/epoch, ~" + totalSteps + " total";
    };
    ["full-batch", "full-grad-accum", "full-epochs", "full-dataset-repeats", "full-max-steps", "full-warmup"].forEach((id) => {
      $(id)?.addEventListener("input", _updateStepEstimate);
      $(id)?.addEventListener("change", _updateStepEstimate);
    });
    $("full-dataset-dir")?.addEventListener("change", () => setTimeout(_updateStepEstimate, 200));

    const optimizerSel = $("full-optimizer");
    const lrInput = $("full-lr");
    if (optimizerSel && lrInput) {
      let prevLR = lrInput.value;
      optimizerSel.addEventListener("change", () => {
        if (optimizerSel.value === "prodigy") {
          if (lrInput.value === prevLR || lrInput.value === "1e-4") {
            lrInput.value = "1.0";
            lrInput.dispatchEvent(new Event("input"));
          }
        } else {
          if (lrInput.value === "1.0") {
            lrInput.value = prevLR || "1e-4";
            lrInput.dispatchEvent(new Event("input"));
          }
        }
        prevLR = lrInput.value;
      });
    }

    $("full-dataset-dir")?.addEventListener("change", () => {
      setTimeout(() => {
        const ppStatus = $("full-pp-status");
        if (!ppStatus) return;
        const hasPP = ppStatus.textContent.includes("detected");
        if (hasPP && lrInput && (lrInput.value === "1e-4" || lrInput.value === lrInput.dataset?.default)) {
          lrInput.value = "5e-5";
          lrInput.dispatchEvent(new Event("input"));
          _updateConsole("PP++ detected \u2014 LR auto-lowered to 5e-5");
        }
      }, 300);
    });
  }

  function init() {
    _initConfirmModal();
    initValidation();
    initProdigyNote();
    initSavePreset();
    initExportCSV();
    initOpenOutputDir();
    initImportPreset();
    initModalUX();
    initModelDerivedDefaults();
    initLossWeightingReactivity();
    initSchedulerReactivity();
    initOptimizerReactivity();
    initResumeReactivity();
    initCkptPresets();
    initSmartBehaviors();
    if (typeof WorkspaceDatasets !== "undefined") WorkspaceDatasets.init();
  }

  return { init, showConfirmModal };

  function showConfirmModal(title, message, confirmLabel, onConfirm) {
    const modal = $("confirm-modal");
    if (!modal) { if (typeof onConfirm === "function") onConfirm(); return; }
    const titleEl = $("confirm-modal-title");
    const msgEl = $("confirm-modal-message");
    const yesBtn = $("confirm-modal-yes");
    if (titleEl) titleEl.textContent = title || "Confirm";
    if (msgEl) msgEl.innerHTML = message || "Are you sure?";
    if (yesBtn) yesBtn.textContent = confirmLabel || "Confirm";
    _confirmCallback = onConfirm;
    modal.classList.add("open");
  }
})();
