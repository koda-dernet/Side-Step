/* ============================================================
   Side-Step GUI — Workspace JS
   Mode switching, tooltips, section collapse, console toggle,
   file browser, toasts, Ez→Advanced transitions, start training,
   presets grid, and initialization of all sub-modules.
   ============================================================ */

/* ---- Global mode switching (called by other modules) ---- */
function switchMode(mode) {
  const modes = document.querySelectorAll(".topbar__mode");
  const panels = document.querySelectorAll(".mode-panel");

  // Update mode buttons
  modes.forEach(m => {
    const isActive = m.dataset.mode === mode;
    m.classList.toggle("active", isActive);
    m.setAttribute("aria-selected", isActive);
  });

  // Show target panel, hide others (no innerHTML clear/restore — keeps event handlers)
  panels.forEach(p => {
    p.classList.toggle("active", p.id === `mode-${mode}`);
  });

  // Update console
  const consoleMode = document.getElementById("console-mode");
  if (consoleMode) consoleMode.textContent = `${mode.charAt(0).toUpperCase() + mode.slice(1)} Mode`;

}

function startGlobalGPUFeed() {
  if (typeof API === "undefined" || typeof API.connectGpuWS !== "function") return;
  if (_workspaceGpuWs) return;
  _workspaceGpuWs = API.connectGpuWS((data) => {
    if (data && typeof data === "object" && typeof AppState !== "undefined") {
      AppState.setGPU(data);
    }
  });
}

/* ---- Toast notifications ---- */
const _recentToasts = [];
let _workspaceGpuWs = null;
function showToast(message, kind) {
  const container = document.getElementById("toast-container");
  if (!container) return;

  const now = Date.now();
  if (_recentToasts.some(t => t.msg === message && now - t.ts < 2000)) return;
  _recentToasts.push({ msg: message, ts: now });
  if (_recentToasts.length > 8) _recentToasts.shift();

  const toast = document.createElement("div");
  toast.className = "toast toast--" + (kind || "info");

  const span = document.createElement("span");
  span.className = "toast__msg";
  span.textContent = message;
  toast.appendChild(span);

  const closeBtn = document.createElement("button");
  closeBtn.className = "toast__close";
  closeBtn.textContent = "\u00d7";
  closeBtn.setAttribute("aria-label", "Dismiss");
  toast.appendChild(closeBtn);

  container.appendChild(toast);
  const duration = kind === "error" ? 8000 : kind === "warn" ? 5000 : 3000;
  const dismiss = () => {
    if (toast._dismissed) return;
    toast._dismissed = true;
    toast.classList.add("fade-out");
    setTimeout(() => toast.remove(), 300);
  };
  toast.style.cursor = "pointer";
  toast.addEventListener("click", dismiss);
  closeBtn.addEventListener("click", (e) => { e.stopPropagation(); dismiss(); });
  setTimeout(dismiss, duration);
}

/* ---- Clipboard helper (modern API + legacy fallback) ---- */
async function copyTextToClipboard(text) {
  const value = String(text || "");
  if (!value) return false;

  try {
    if (navigator.clipboard && typeof navigator.clipboard.writeText === "function") {
      await navigator.clipboard.writeText(value);
      return true;
    }
  } catch (_) {
    // Fall through to legacy copy path.
  }

  try {
    const textarea = document.createElement("textarea");
    textarea.value = value;
    textarea.setAttribute("readonly", "");
    textarea.style.position = "fixed";
    textarea.style.top = "-1000px";
    textarea.style.left = "-1000px";
    document.body.appendChild(textarea);
    textarea.focus();
    textarea.select();
    textarea.setSelectionRange(0, value.length);
    const copied = document.execCommand("copy");
    textarea.remove();
    return !!copied;
  } catch (_) {
    return false;
  }
}

/* ---- Smart auto-scroll: only scroll if user is already at/near bottom ---- */
function autoScrollLog(el) {
  if (!el) return;
  const threshold = 40;
  const atBottom = el.scrollHeight - el.scrollTop - el.clientHeight < threshold;
  if (atBottom) el.scrollTop = el.scrollHeight;
}

/* ---- Double-click guard for action buttons ---- */
function guardDoubleClick(btnId, cooldownMs) {
  const btn = document.getElementById(btnId);
  if (!btn) return;
  btn.addEventListener("click", () => {
    if (btn.dataset.guarded === "1") return;
    btn.dataset.guarded = "1";
    btn.disabled = true;
    setTimeout(() => { btn.disabled = false; btn.dataset.guarded = ""; }, cooldownMs || 2000);
  }, true);
}

/* ---- Table Row Selection (global — used by history, workspace-datasets, dataset) ----
   opts.colorSlots  — use A/B/overflow coloring for compare semantics (History only)
*/
let _selectionHintShown = false;
const _shiftClickTables = [];  // track all initialized tbodies for Escape handler

function initShiftClickTable(tbodyId, opts) {
  const tbody = document.getElementById(tbodyId);
  if (!tbody || tbody.dataset.shiftInit) return;
  tbody.dataset.shiftInit = "1";
  const useColorSlots = !!(opts && opts.colorSlots);
  const requireModifier = !!(opts && opts.requireModifier);
  let _anchor = -1;

  const _slotCls = ["selected-a", "selected-b", "selected-overflow"];
  function _applySlots() {
    if (!useColorSlots) return;
    tbody.querySelectorAll("tr.selected").forEach((r, i) => {
      r.classList.remove(..._slotCls);
      r.classList.add(_slotCls[Math.min(i, 2)]);
    });
  }
  function _clearAll(rows) {
    rows.forEach(r => r.classList.remove("selected", ..._slotCls));
  }
  function clearSelection() {
    _clearAll([...tbody.querySelectorAll("tr")]);
    _applySlots();
    _anchor = -1;
  }

  _shiftClickTables.push({ tbody, clearSelection });

  tbody.addEventListener("click", (e) => {
    const tr = e.target.closest("tr");
    if (!tr || e.target.closest("button") || e.target.closest("a")) return;

    // Skip empty-state placeholder rows
    if (tr.querySelector(".data-table-empty")) return;

    if (tr.classList.contains("dataset-folder-row")) {
      const isCtrlFolder = e.ctrlKey || e.metaKey;
      const isShiftFolder = e.shiftKey;
      if (!isCtrlFolder && !isShiftFolder) {
        const toggle = tr.querySelector(".dataset-folder-toggle");
        if (toggle && !e.target.closest(".dataset-folder-toggle")) toggle.click();
        return;
      }
      // Fall through to selection logic for ctrl/shift clicks on folders
    }

    const rows = [...tbody.querySelectorAll("tr")];
    const idx = rows.indexOf(tr);
    if (idx < 0) return;

    const isCtrl = e.ctrlKey || e.metaKey;

    if (e.shiftKey && _anchor >= 0) {
      const lo = Math.min(_anchor, idx), hi = Math.max(_anchor, idx);
      if (!isCtrl) _clearAll(rows);
      for (let i = lo; i <= hi; i++) rows[i].classList.add("selected");
    } else if (isCtrl) {
      tr.classList.toggle("selected");
      if (tr.classList.contains("selected")) _anchor = idx;
    } else if (!requireModifier) {
      // Toggle off if clicking an already-selected row
      if (tr.classList.contains("selected")) {
        _clearAll(rows);
        _anchor = -1;
      } else {
        _clearAll(rows);
        tr.classList.add("selected");
        _anchor = idx;
      }
    }
    _applySlots();

    if (!_selectionHintShown && typeof showToast === "function") {
      _selectionHintShown = true;
      showToast("Tip: Ctrl+click to toggle, Shift+click for range", "info");
    }
  });

  // Click on table wrapper area but not on a row → clear selection
  const table = tbody.closest("table") || tbody.closest(".data-table");
  if (table) {
    table.addEventListener("click", (e) => {
      if (!e.target.closest("tr") || e.target.closest("thead")) {
        clearSelection();
      }
    });
  }
}

// Escape key clears selection in all visible tables
document.addEventListener("keydown", (e) => {
  if (e.key === "Escape") {
    _shiftClickTables.forEach(({ tbody, clearSelection }) => {
      if (tbody.closest(".lab-panel.active") || tbody.closest(".mode-panel.active")) {
        if (tbody.querySelectorAll("tr.selected, tr.selected-a, tr.selected-b").length > 0) {
          clearSelection();
          e.preventDefault();
        }
      }
    });
  }
});

(function () {
  "use strict";

  function _markRecalled(el) {
    if (!el || !el.classList) return;
    el.classList.add("recalled");
    if (el.dataset.recalledBound === "1") return;
    const clear = () => el.classList.remove("recalled");
    el.addEventListener("input", clear);
    el.addEventListener("change", clear);
    el.dataset.recalledBound = "1";
  }

  /* ---- Mode switching ---- */
  function initModeSwitching() {
    // Document-level delegation (works in pywebview/WebKit and standard browsers)
    document.addEventListener("click", (e) => {
      const btn = e.target.closest(".topbar__mode");
      if (btn?.dataset?.mode) switchMode(btn.dataset.mode);
    });

    /* Alt+1/2/3/4 handled by Palette keybind system (palette.js) */
  }

  /* ---- Lab sub-navigation ---- */
  function initLabNav() {
    const items = document.querySelectorAll(".lab-nav__item");
    const panels = document.querySelectorAll(".lab-panel");

    items.forEach((btn) => {
      btn.addEventListener("click", () => {
        const target = btn.dataset.lab;
        items.forEach((i) => i.classList.remove("active"));
        panels.forEach((p) => p.classList.remove("active"));
        btn.classList.add("active");
        const panel = document.getElementById("lab-" + target);
        if (panel) panel.classList.add("active");

        if (target === "history" && typeof History !== "undefined" && typeof History.loadHistory === "function") {
          Promise.resolve(History.loadHistory()).catch(() => {});
        }
        if (target === "datasets" && typeof WorkspaceDatasets !== "undefined" && typeof WorkspaceDatasets.refresh === "function") {
          Promise.resolve(WorkspaceDatasets.refresh()).catch(() => {});
        }
        if (target === "dataset" && typeof Dataset !== "undefined" && typeof Dataset.refreshFromSettings === "function") {
          Promise.resolve(Dataset.refreshFromSettings()).catch(() => {});
          if (typeof Dataset.bootBeep === "function") Dataset.bootBeep();
          if (typeof Dataset.reopen === "function") Dataset.reopen();
        }
        if (target === "export") {
          // Pre-fill adapter dir from selected history row
          const sel = document.querySelector('#history-tbody tr.selected, #history-tbody tr.selected-a');
          if (sel) {
            const idx = sel.dataset.idx;
            const runs = (typeof AppState !== 'undefined' && AppState.runs) ? AppState.runs() : [];
            const run = runs[idx];
            if (run && run.artifact_path) {
              const input = document.getElementById('export-adapter-dir');
              if (input && !input.value.trim()) input.value = run.artifact_path;
            }
          }
        }
      });
    });
  }

  /* ---- Help tooltip [?] toggle ---- */
  function initHelpTooltips() {
    document.addEventListener("click", (e) => {
      const icon = e.target.closest(".help-icon");
      if (!icon) return;
      e.preventDefault();
      e.stopPropagation();
      const helpId = icon.dataset.help;
      if (!helpId) return;
      const panel = document.getElementById(helpId);
      if (panel) panel.classList.toggle("active");
    });
  }

  /* ---- Collapsible section groups ---- */
  function initSectionGroups() {
    document.querySelectorAll(".section-group__toggle").forEach((toggle) => {
      toggle.addEventListener("click", () => {
        toggle.closest(".section-group").classList.toggle("open");
      });
    });
  }

  /* ---- Logo toggle (click to cycle variants) ---- */
  function initLogoToggle() {
    const container = $("logo-container");
    if (!container) return;
    const variants = container.querySelectorAll(".logo-variant");
    if (variants.length < 2) return;
    let idx = 0;
    container.addEventListener("click", () => {
      variants[idx].style.display = "none";
      idx = (idx + 1) % variants.length;
      variants[idx].style.display = "block";
    });
  }

  /* ---- Console strip expand/collapse ---- */
  function initConsole() {
    const c = document.querySelector(".console");
    if (c) c.addEventListener("click", () => c.classList.toggle("expanded"));
  }

  /* ---- Adapter card selection ---- */
  function initAdapterCards() {
    const onCardSelect = (card, group) => {
      if (!card || !group) return;
      group.querySelectorAll(".adapter-card").forEach((c) => c.classList.remove("selected"));
      card.classList.add("selected");
      const radio = card.querySelector("input[type=radio]");
      if (radio) radio.checked = true;
    };
    document.querySelectorAll(".adapter-cards").forEach((group) => {
      group.querySelectorAll(".adapter-card").forEach((card) => {
        card.addEventListener("click", () => onCardSelect(card, group));
      });
    });
  }

  /* ---- Projection chip toggles ---- */
  function initProjChips() {
    const single = $("proj-chips"), split = $("proj-chips-split");
    const hAll = $("full-projections"), hSelf = $("full-self-projections"), hCross = $("full-cross-projections");
    const attnSel = $("full-attention-type");
    if (!single || !hAll) return;

    function syncContainer(container, hidden) {
      const vals = [];
      container.querySelectorAll("input[type=checkbox]").forEach(cb => {
        cb.closest(".proj-chip").classList.toggle("active", cb.checked);
        if (cb.checked) vals.push(cb.value);
      });
      if (hidden) { hidden.value = vals.join(" "); hidden.dispatchEvent(new Event("change")); }
      return vals;
    }
    function syncAll() {
      if (attnSel && attnSel.value === "both" && split) {
        syncContainer($("proj-chips-self"), hSelf);
        syncContainer($("proj-chips-cross"), hCross);
        hAll.value = (hSelf?.value || "") + " " + (hCross?.value || "");
      } else {
        syncContainer(single, hAll);
      }
    }
    function toggleMode() {
      const isBoth = attnSel && attnSel.value === "both";
      single.style.display = isBoth ? "none" : "";
      if (split) split.style.display = isBoth ? "" : "none";
      syncAll();
    }
    [single, $("proj-chips-self"), $("proj-chips-cross")].forEach(c => {
      c?.querySelectorAll("input[type=checkbox]").forEach(cb => cb.addEventListener("change", syncAll));
    });
    attnSel?.addEventListener("change", toggleMode);
    toggleMode();
  }

  /* ---- Non-default value highlighting ---- */
  function initModifiedTracking() {
    document.querySelectorAll(".input[data-default]").forEach((input) => {
      const check = () => input.classList.toggle("modified", input.value !== input.dataset.default);
      input.addEventListener("input", check); input.addEventListener("change", check); check();
    });
  }

  /* ---- Compute steps/epoch from dataset file count ---- */
  function _stepsPerEpoch(batchSize, gradAccum) {
    const sel = $("ez-dataset-dir") || $("full-dataset-dir");
    const opt = sel?.selectedOptions?.[0];
    if (!opt) return 0;
    const m = opt.textContent.match(/(\d+)\s*files/);
    const count = m ? parseInt(m[1]) : 0;
    const bs = parseInt(batchSize) || 1, ga = parseInt(gradAccum) || 1;
    return count > 0 ? Math.ceil(count / (bs * ga)) : 0;
  }

  /* ---- Start Training Buttons ---- */
  function initStartTraining() {
    function _isAudioDataset(datasetDir) {
      return typeof datasetDir === 'string' && datasetDir.startsWith('audio:');
    }

    function _stripAudioPrefix(datasetDir) {
      return datasetDir.replace(/^audio:/, '');
    }

    function _checkVramThenRun(config, run) {
      if (typeof VRAM === 'undefined') { run(); return; }
      const verdict = VRAM.getVerdict();
      if (verdict === 'red') {
        if (typeof WorkspaceBehaviors !== 'undefined' && WorkspaceBehaviors.showConfirmModal) {
          WorkspaceBehaviors.showConfirmModal(
            'VRAM Budget Exceeded',
            'The estimated VRAM usage exceeds your GPU capacity. Training will very likely crash with an out-of-memory error. Continue anyway?',
            'Start Anyway',
            run
          );
          return;
        }
      } else if (verdict === 'yellow') {
        if (typeof WorkspaceBehaviors !== 'undefined' && WorkspaceBehaviors.showConfirmModal) {
          WorkspaceBehaviors.showConfirmModal(
            'VRAM Budget Tight',
            'The estimated VRAM usage is close to your GPU limit. Training may fail under certain conditions. Continue?',
            'Continue',
            run
          );
          return;
        }
      }
      run();
    }

    async function _preprocessThenTrain(config) {
      const audioDir = _stripAudioPrefix(config.dataset_dir);
      const tensorsDir = $("settings-tensors-dir")?.value || Defaults.get("settings-tensors-dir") || "preprocessed_tensors";
      const outputDir = _joinPath(tensorsDir, _pathBasename(audioDir) || "tensors");
      showToast("Preprocessing audio before training...", "info");
      try {
        const ppConfig = {
          audio_dir: audioDir,
          output_dir: outputDir,
          model_variant: config.model_variant || "turbo",
          checkpoint_dir: $("settings-checkpoint-dir")?.value || Defaults.get("settings-checkpoint-dir") || "checkpoints",
          normalize: "peak",
          peak_target: -1.0,
          trigger_tag: "",
          tag_position: "prepend",
          genre_ratio: 0,
        };
        const taskResult = await API.runPreprocess(ppConfig);
        if (!taskResult?.task_id) { showToast("Failed to start preprocessing", "error"); return; }
        await new Promise((resolve, reject) => {
          const close = API.connectTaskWS(taskResult.task_id, (msg) => {
            if (msg.status === "done") { close(); resolve(); }
            else if (msg.status === "error") { close(); reject(new Error(msg.error || "Preprocessing failed")); }
          });
        });
        config.dataset_dir = outputDir;
        showToast("Preprocessing complete, starting training...", "ok");
        Training.enqueue(config);
      } catch (e) {
        showToast("Preprocessing failed: " + (e.message || e), "error");
      }
    }

    $("btn-start-ez")?.addEventListener("click", () => {
      if (typeof Validation !== 'undefined') {
        const result = Validation.validateAll();
        if (result !== true) {
          showToast(result.slice(0, 3).join(' \u00b7 '), "error"); return;
        }
      }
      const config = typeof WorkspaceConfig !== 'undefined' ? WorkspaceConfig.gatherFullConfig() : {};
      const adapter = document.querySelector('input[name="ez-adapter"]:checked')?.value || config.adapter_type || "lora";
      const variant = $("ez-model-variant")?.value || config.model_variant || "turbo";
      const runName = adapter + "_" + variant + "_" + _timestamp();
      config.adapter_type = adapter;
      config.model_variant = variant;
      config.dataset_dir = $("ez-dataset-dir")?.value || config.dataset_dir || "";
      config.run_name = runName;
      config.steps_per_epoch = _stepsPerEpoch(config.batch_size, config.grad_accum);
      config.output_dir = _joinPath($("settings-adapters-dir")?.value || Defaults.get("settings-adapters-dir") || "trained_adapters", adapter, runName);

      if (!config.dataset_dir) {
        showToast("Please select a dataset first", "warn");
        return;
      }

      _checkVramThenRun(config, () => {
        if (_isAudioDataset(config.dataset_dir)) {
          _preprocessThenTrain(config);
        } else {
          Training.enqueue(config);
        }
      });
    });

    $("btn-start-full")?.addEventListener("click", () => {
      if (typeof Validation !== 'undefined') {
        const result = Validation.validateAll();
        if (result !== true) {
          showToast(result.slice(0, 3).join(' \u00b7 '), "error");
          return;
        }
      }
      const config = typeof WorkspaceConfig !== 'undefined' ? WorkspaceConfig.gatherFullConfig() : {};
      config.run_name = (config.run_name || "run") + "_" + _timestamp();
      config.steps_per_epoch = _stepsPerEpoch(config.batch_size, config.grad_accum);
      if (!config.output_dir) {
        config.output_dir = _joinPath($("settings-adapters-dir")?.value || Defaults.get("settings-adapters-dir") || "trained_adapters", config.adapter_type || "lora", config.run_name);
      }

      if (!config.dataset_dir) {
        showToast("Please select a dataset first", "warn");
        return;
      }

      _checkVramThenRun(config, () => {
        if (_isAudioDataset(config.dataset_dir)) {
          _preprocessThenTrain(config);
        } else {
          Training.enqueue(config);
        }
      });
    });

    // Monitor controls
    $("btn-stop")?.addEventListener("click", () => {
      Training.stop();
    });
    $("btn-monitor-to-ez")?.addEventListener("click", () => switchMode("ez"));
  }

  const _timestamp = window._timestamp || (() => new Date().toISOString().replace(/[-:T]/g, '').slice(0, 13));

  /* ---- PP++ quick-run from Advanced ---- */
  function initPPQuickRun() {
    $("btn-full-run-ppplus")?.addEventListener("click", () => {
      // Switch to Lab > PP++ tab with dataset pre-filled
      switchMode("lab");
      setTimeout(() => {
        const items = document.querySelectorAll(".lab-nav__item");
        const panels = document.querySelectorAll(".lab-panel");
        items.forEach((i) => i.classList.remove("active"));
        panels.forEach((p) => p.classList.remove("active"));
        const btn = document.querySelector('.lab-nav__item[data-lab="ppplus"]');
        if (btn) btn.classList.add("active");
        const panel = $("lab-ppplus");
        if (panel) panel.classList.add("active");
        // Sync dataset value
        const ds = $("full-dataset-dir")?.value;
        if (ds && $("ppplus-dataset-dir")) $("ppplus-dataset-dir").value = ds;
      }, 50);
    });
  }

  /* ---- Motto (Minecraft-style splash, sourced from banner.py via server injection) ---- */
  function _applyMotto() {
    const el = $("ez-motto");
    const mottos = window.__MOTTOS__;
    if (!el || !mottos || mottos.length === 0) return;
    const motto = mottos[Math.floor(Math.random() * mottos.length)];
    el.textContent = `"${motto}"`;
    document.title = `Side-Step \u2014 ${motto}`;
  }

  /* ---- GPU init through AppState ---- */
  async function initGPU() {
    try {
      const gpu = await API.fetchGPU();
      if (typeof AppState !== 'undefined') AppState.setGPU(gpu);
    } catch (e) {
      if (typeof AppState !== 'undefined') {
        AppState.setGPU({ name: "unavailable", vram_used_mb: 0, vram_total_mb: 0, utilization: 0, temperature: 0, power_draw_w: 0 });
      }
      console.warn("[GPU] fetch failed:", e.message);
    }
  }

  /* ---- Init all ---- */
  async function init() {
    try {
      if (typeof UiPrefs !== "undefined") {
        await UiPrefs.load();
      }
      if (typeof Theme !== "undefined") {
        await Theme.init();
      }
      document.addEventListener('sidestep:api-auth-failed', () => {
        if (typeof showToast === 'function') showToast('API auth failed \u2014 GPU/models/datasets may show placeholders. Check console.', 'warn');
      });

      // Core UI
      initModeSwitching();
      initLabNav();
      initLogoToggle();
      initHelpTooltips();
      initSectionGroups();
      initConsole();
      initAdapterCards();
      initProjChips();
      initModifiedTracking();
      initStartTraining();
      initPPQuickRun();
      _applyMotto();
      document.addEventListener("visibilitychange", () => { if (!document.hidden) _applyMotto(); });
      document.addEventListener("sidestep:settings-saved", (e) => {
        const saved = e.detail;
        if (!saved || typeof saved !== "object") return;
        Object.entries(saved).forEach(([key, val]) => {
          if (val == null || val === "") return;
          const el = document.getElementById(_settingsDomId(key));
          if (!el) return;
          _markRecalled(el);
        });
      });

      initShiftClickTable("datasets-tbody", { requireModifier: true });
      initShiftClickTable("history-tbody", { colorSlots: true });

      ["btn-start-ez", "btn-start-full", "btn-resume-start",
       "btn-start-preprocess", "btn-run-ppplus", "btn-run-captions"].forEach(id => guardDoubleClick(id, 3000));

      initGPU();
      startGlobalGPUFeed();

      setInterval(() => {
        const monitorPanel = document.getElementById("mode-monitor");
        const isMonitorActive = monitorPanel && monitorPanel.classList.contains("active");
        const isTrainingRunning = typeof Training !== "undefined" && Training.isRunning();
        if (isMonitorActive && !isTrainingRunning) initGPU();
      }, 10000);

      // Sub-modules: skip missing or init-less modules silently
      const modules = [
        ["Validation", typeof Validation !== "undefined" && Validation],
        ["AppState", typeof AppState !== "undefined" && AppState],
        ["Reactivity", typeof Reactivity !== "undefined" && Reactivity],
        ["VRAM", typeof VRAM !== "undefined" && VRAM],
        ["Training", typeof Training !== "undefined" && Training],
        ["Dataset", typeof Dataset !== "undefined" && Dataset],
        ["History", typeof History !== "undefined" && History],
        ["Palette", typeof Palette !== "undefined" && Palette],
        ["WorkspaceConfig", typeof WorkspaceConfig !== "undefined" && WorkspaceConfig],
        ["WorkspaceSetup", typeof WorkspaceSetup !== "undefined" && WorkspaceSetup],
        ["WorkspaceCharts", typeof WorkspaceCharts !== "undefined" && WorkspaceCharts],
        ["WorkspaceDatasets", typeof WorkspaceDatasets !== "undefined" && WorkspaceDatasets],
        ["WorkspaceBehaviors", typeof WorkspaceBehaviors !== "undefined" && WorkspaceBehaviors],
        ["WorkspaceLab", typeof WorkspaceLab !== "undefined" && WorkspaceLab],
        ["ReactivityExt", typeof ReactivityExt !== "undefined" && ReactivityExt],
        ["CustomSelect", typeof CustomSelect !== "undefined" && CustomSelect],
        ["CRT", typeof CRT !== "undefined" && CRT],
      ];
      for (const [name, mod] of modules) {
        if (!mod) continue;
        try {
          if (typeof mod.init === "function") { mod.init(); }
        } catch (e) { console.error(name + " init failed:", e); }
      }

    } catch (error) {
      console.error('Workspace initialization failed:', error);
      console.error('Stack trace:', error.stack);
    }
  }

  function initWelcome(savedSettings) {
    const ov = document.getElementById("welcome-overlay");
    if (!ov) return;
    let firstRunComplete = savedSettings?.first_run_complete === true;
    if (!firstRunComplete && typeof UiPrefs !== "undefined") {
      const v = UiPrefs.get("welcomed");
      if (v === true || v === "done") firstRunComplete = true;
    }
    if (firstRunComplete) {
      ov.classList.add("hidden");
      ov.style.display = "none";
      ov.style.pointerEvents = "none";
      return;
    }
    const _dismiss = () => {
      if (typeof UiPrefs !== "undefined") UiPrefs.set("welcomed", true);
      ov.classList.add("hidden");
      ov.style.display = "none";
      ov.style.pointerEvents = "none";
      if (typeof Tutorial !== "undefined" && !Tutorial.isDone()) {
        setTimeout(() => Tutorial.start(), 400);
      }
    };
    $("welcome-skip")?.addEventListener("click", (e) => {
      e.preventDefault();
      API.saveSettings({ first_run_complete: true }).catch(() => {});
      _dismiss();
    });
    const welcomeSaveBtn = $("welcome-save");
    if (welcomeSaveBtn) {
      welcomeSaveBtn.addEventListener("click", async (e) => { e.preventDefault(); await handleWelcomeSave(); });
    }
    async function handleWelcomeSave() {
      [["welcome-checkpoint-dir","settings-checkpoint-dir"],["welcome-audio-dir","settings-audio-dir"],["welcome-tensors-dir","settings-tensors-dir"],["welcome-adapters-dir","settings-adapters-dir"],["welcome-exported-loras-dir","settings-exported-loras-dir"],["welcome-gemini-key","settings-gemini-key"],["welcome-openai-key","settings-openai-key"]]
        .forEach(([s,d]) => { const sv = $(s), dv = $(d); if (sv?.value && dv) dv.value = sv.value; });
      const raw = {
        first_run_complete: true,
        checkpoint_dir: $("settings-checkpoint-dir")?.value,
        audio_dir: $("settings-audio-dir")?.value,
        preprocessed_tensors_dir: $("settings-tensors-dir")?.value,
        trained_adapters_dir: $("settings-adapters-dir")?.value,
        exported_loras_dir: $("settings-exported-loras-dir")?.value,
        gemini_api_key: $("settings-gemini-key")?.value,
        openai_api_key: $("settings-openai-key")?.value,
      };
      const data = {};
      Object.entries(raw).forEach(([k, v]) => { if (v != null && v !== "") data[k] = v; });
      data.first_run_complete = true;
      try { await API.saveSettings(data); } catch (e) { console.warn('[welcome] save failed:', e); }
      _dismiss();
      if (typeof WorkspaceSetup !== "undefined") WorkspaceSetup.populatePickers();
      document.dispatchEvent(new CustomEvent("sidestep:settings-saved", { detail: data }));
    }
    document.addEventListener("keydown", (e) => {
      if (e.key === "Escape" && !ov.classList.contains("hidden")) {
        API.saveSettings({ first_run_complete: true }).catch(() => {});
        _dismiss();
      }
    });
    // Show overlay for first-run users
    ov.classList.remove("hidden");
    ov.style.display = "";
    ov.style.pointerEvents = "";
  }

  // Backend settings key → DOM element ID (where the generic convention fails)
  const _SETTINGS_DOM_MAP = {
    preprocessed_tensors_dir: "settings-tensors-dir",
    trained_adapters_dir:     "settings-adapters-dir",
    exported_loras_dir:       "settings-exported-loras-dir",
    gemini_api_key:           "settings-gemini-key",
    openai_api_key:           "settings-openai-key",
    openai_base_url:          "settings-openai-base",
    genius_api_token:         "settings-genius-token",
    transcriber_server_url:    "settings-transcriber-server-url",
    music_flamingo_url:        "settings-music-flamingo-url",
    hf_token:                  "settings-hf-token",
  };

  function _settingsDomId(key) {
    return _SETTINGS_DOM_MAP[key] || ("settings-" + key.replace(/_/g, "-"));
  }

  function _splashProgress(pct, msg) {
    const fill = document.getElementById('splash-bar-fill');
    const status = document.getElementById('splash-status');
    if (fill) fill.style.width = pct + '%';
    if (status) status.textContent = msg;
  }
  function _splashDone() {
    const splash = document.getElementById('splash');
    if (splash) {
      splash.classList.add('done');
      setTimeout(() => { splash.remove(); }, 700);
    }
  }

  function _reportRuntimeIssue(kind, detail) {
    const msg = `[${String(kind || 'runtime')}] ${String(detail || '')}`;
    const bridge = window.sidestepElectron || window.pywebview?.api;
    if (bridge?.onBootError) {
      try {
        bridge.onBootError(msg);
      } catch (_) {}
    }
  }

  function _captureUiState(tag) {
    const activeMode = document.querySelector('.mode-panel.active')?.id || 'none';
    const activeLab = document.querySelector('.lab-panel.active')?.id || 'none';
    const main = document.querySelector('.main');
    const scrollTop = main ? Math.round(main.scrollTop || 0) : -1;
    const scrollHeight = main ? Math.round(main.scrollHeight || 0) : -1;
    const clientHeight = main ? Math.round(main.clientHeight || 0) : -1;
    const url = `${location.pathname}${location.search}${location.hash}`;
    return `${tag} mode=${activeMode} lab=${activeLab} scroll=${scrollTop}/${scrollHeight}/${clientHeight} url=${url}`;
  }

  let _runtimeHooksInstalled = false;
  function _installRuntimeHooks() {
    if (_runtimeHooksInstalled) return;
    _runtimeHooksInstalled = true;

    window.addEventListener('error', (e) => {
      const detail = e.error?.stack || e.message || String(e.error || 'unknown error');
      console.error('[runtime] window error:', e.error || e.message);
      _reportRuntimeIssue('window-error', detail);
    }, true);

    window.addEventListener('unhandledrejection', (e) => {
      const detail = e.reason?.stack || e.reason?.message || String(e.reason || 'unknown rejection');
      console.error('[runtime] unhandled rejection:', e.reason);
      _reportRuntimeIssue('unhandledrejection', detail);
    });

    ['beforeunload', 'pagehide', 'popstate', 'hashchange'].forEach((evtName) => {
      window.addEventListener(evtName, () => {
        _reportRuntimeIssue(evtName, _captureUiState(evtName));
      });
    });

    document.addEventListener('click', (e) => {
      const tracked = e.target instanceof Element
        ? e.target.closest([
            '#full-prefetch-factor',
            '#full-pin-memory',
            '[data-help="help-full-prefetch"]',
            '[data-help="help-full-pin"]',
            'label[for="caption-local-cpu-offload"]',
            '#caption-local-cpu-offload',
            '#caption-local-cpu-offload + .toggle__track',
          ].join(', '))
        : null;
      if (!tracked) return;
      const detail = tracked.id
        ? `#${tracked.id}`
        : tracked.getAttribute('data-help')
          ? `[data-help="${tracked.getAttribute('data-help')}"]`
          : tracked.getAttribute('for')
            ? `label[for="${tracked.getAttribute('for')}"]`
            : tracked.tagName.toLowerCase();
      console.log('[runtime] tracked click:', detail, 'raw=', e.target);
      _reportRuntimeIssue('tracked-click', detail);
      _reportRuntimeIssue('ui-state', _captureUiState(`post-click:${detail}`));
      setTimeout(() => {
        _reportRuntimeIssue('ui-state', _captureUiState(`post-click+150:${detail}`));
      }, 150);
      setTimeout(() => {
        _reportRuntimeIssue('ui-state', _captureUiState(`post-click+500:${detail}`));
      }, 500);
    }, true);
  }

  async function boot() {
    try {
      _splashProgress(5, 'Loading configuration...');

      // Welcome overlay starts hidden in CSS; initWelcome() shows it if needed

      // Token fallback: some backends (e.g. GTK/WebKit) may not preserve URL params or injection
      const _bridge = window.sidestepElectron || window.pywebview?.api;
      if (!new URLSearchParams(window.location.search).get('token') &&
          typeof window.__SIDESTEP_TOKEN__ === 'undefined' &&
          _bridge?.getToken) {
        try {
          window.__SIDESTEP_TOKEN__ = await _bridge.getToken();
        } catch (e) {
          console.warn('[boot] getToken fallback failed:', e);
        }
      }

      _splashProgress(15, 'Loading defaults...');
      if (typeof Defaults !== "undefined") {
        await Defaults.load();
        try {
          Defaults.apply();
        } catch (e) {
          console.warn('[Defaults] apply failed:', e);
        }
      } else {
        console.error('Defaults not available');
      }

      // Load saved settings from backend and overlay onto DOM (before initWelcome so it can branch)
      let savedSettings = null;
      try {
        const saved = await API.fetchSettings();
        if (saved && typeof saved === 'object') {
          savedSettings = saved;
          try {
            Object.entries(saved).forEach(([key, val]) => {
              if (val == null || val === '') return;
              const el = document.getElementById(_settingsDomId(key));
              if (!el) return;
              if (el.type === 'checkbox') el.checked = val === true;
              else el.value = String(val);
              _markRecalled(el);
            });
          } catch (e) {
            console.warn('[boot] could not apply saved settings to DOM:', e);
          }
          // Sync settings model values into the caption panel fields
          const _syncCaptionModel = (settingsId, captionId) => {
            const v = document.getElementById(settingsId)?.value;
            const el = document.getElementById(captionId);
            if (v && el) el.value = v;
          };
          _syncCaptionModel("settings-gemini-model", "caption-gemini-model");
        }
      } catch (e) {
        console.warn('[boot] could not load saved settings:', e);
      }

      _splashProgress(35, 'Restoring settings...');
      if (typeof UiPrefs !== "undefined") await UiPrefs.load();
      initWelcome(savedSettings);
      
      _splashProgress(45, 'Initializing UI modules...');
      await init();
      
      // Populate pickers AFTER settings are in DOM (fixes boot race condition)
      _splashProgress(70, 'Loading models and datasets...');
      if (typeof WorkspaceSetup !== "undefined") {
        await WorkspaceSetup.populatePickers();
        if (typeof AppState !== "undefined") AppState.setStatus("idle");
      } else {
        console.error('WorkspaceSetup not available');
      }
      if (typeof API !== 'undefined' && API.startHeartbeat) API.startHeartbeat();
      _splashProgress(100, 'Ready.');

      // Auto-start tutorial for users who completed welcome but never saw the tutorial
      // (e.g. CLI wizard set first_run_complete, or app closed before tutorial finished)
      if (typeof Tutorial !== "undefined" && !Tutorial.isDone()) {
        const _wov = document.getElementById("welcome-overlay");
        if (!_wov || _wov.classList.contains("hidden")) {
          setTimeout(() => { if (typeof Tutorial !== "undefined") Tutorial.start(); }, 1600);
        }
      }

      // Fade out splash after app is fully painted
      setTimeout(_splashDone, 1200);
      
    } catch (error) {
      _splashDone();
      const msg = error?.message || String(error);
      console.error('Workspace boot failed:', error);
      console.error('Stack trace:', error.stack);
      if (typeof showToast === 'function') showToast('Boot failed: ' + msg.slice(0, 60), 'error');
      const errEl = document.getElementById('boot-error') || (() => {
        const el = document.createElement('div');
        el.id = 'boot-error';
        el.style.cssText = 'position:fixed;top:0;left:0;right:0;background:#c00;color:#fff;padding:8px;z-index:99999;font-family:monospace;font-size:12px;';
        document.body.appendChild(el);
        return el;
      })();
      errEl.textContent = 'Boot failed: ' + msg;
      errEl.style.display = 'block';
      _reportRuntimeIssue('boot-failed', msg);
    }
  }

  // ---- Electron exit confirmation modal ----
  // When the main process asks to close, check for unsaved state and show
  // a styled in-app modal instead of the stock OS dialog.
  if (window.sidestepElectron && window.sidestepElectron.onCloseRequested) {
    const _exitModal    = $("exit-confirm-modal");
    const _exitMsg      = $("exit-confirm-message");
    const _exitLeave    = $("exit-confirm-leave");
    const _exitStay     = $("exit-confirm-stay");
    const _exitClose    = $("exit-confirm-close");

    const _showExitModal = (msg) => {
      if (_exitMsg) _exitMsg.textContent = msg;
      if (_exitModal) _exitModal.classList.add("open");
    };
    const _hideExitModal = () => {
      if (_exitModal) _exitModal.classList.remove("open");
    };

    window.sidestepElectron.onCloseRequested(() => {
      // Check if there's a reason to warn
      if (typeof Training !== "undefined" && Training.isRunning()) {
        _showExitModal("Training is currently running. Closing will stop it.");
        return;
      }
      const dirty = document.querySelectorAll('.input[data-default]');
      for (const el of dirty) {
        if (el.value !== el.dataset.default) {
          _showExitModal("You have unsaved changes.");
          return;
        }
      }
      // Nothing to guard — close immediately
      window.sidestepElectron.confirmClose();
    });

    _exitLeave?.addEventListener("click", () => {
      _hideExitModal();
      window.sidestepElectron.confirmClose();
    });
    _exitStay?.addEventListener("click", _hideExitModal);
    _exitClose?.addEventListener("click", _hideExitModal);
  }

  // Signal server shutdown when the tab/window closes (browser fallback only).
  // Server uses a 3s delayed exit — cancelled if the page reloads (refresh).
  window.addEventListener("beforeunload", (e) => {
    // Guard: training in progress
    if (typeof Training !== "undefined" && Training.isRunning()) {
      e.preventDefault();
      e.returnValue = '';
    }
    // Guard: dirty config (any field differs from its default)
    if (!e.defaultPrevented) {
      const dirty = document.querySelectorAll('.input[data-default]');
      for (const el of dirty) {
        if (el.value !== el.dataset.default) { e.preventDefault(); e.returnValue = ''; break; }
      }
    }
    // Only signal server shutdown in browser mode — Electron and pywebview
    // lifecycle is managed by the Python launcher (proc.wait / webview.start).
    if (typeof API !== "undefined" && API.signalShutdown
        && !window.sidestepElectron && !window.pywebview) {
      API.signalShutdown();
    }
  });

  _installRuntimeHooks();

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", boot);
  } else {
    boot();
  }
})();
