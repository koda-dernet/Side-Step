/* ============================================================
   Side-Step GUI — Workspace JS
   Mode switching, tooltips, section collapse, console toggle,
   file browser, toasts, Ez→Advanced transitions, start training,
   presets grid, and initialization of all sub-modules.
   ============================================================ */

/* ---- Global mode switching (called by other modules) ---- */
function switchMode(mode) {
  console.log('Switching to mode:', mode);
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

  console.log('Mode switch completed:', mode);
}

/* ---- Toast notifications ---- */
const _recentToasts = [];
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
function initShiftClickTable(tbodyId, opts) {
  const tbody = document.getElementById(tbodyId);
  if (!tbody || tbody.dataset.shiftInit) return;
  tbody.dataset.shiftInit = "1";
  const useColorSlots = !!(opts && opts.colorSlots);
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

  tbody.addEventListener("click", (e) => {
    const tr = e.target.closest("tr");
    if (!tr || e.target.closest("button") || e.target.closest("a")) return;
    const rows = [...tbody.querySelectorAll("tr")];
    const idx = rows.indexOf(tr);
    if (idx < 0) return;

    const isCtrl = e.ctrlKey || e.metaKey;

    if (e.shiftKey && _anchor >= 0) {
      // Range select from anchor
      const lo = Math.min(_anchor, idx), hi = Math.max(_anchor, idx);
      if (!isCtrl) _clearAll(rows);
      for (let i = lo; i <= hi; i++) rows[i].classList.add("selected");
    } else if (isCtrl) {
      // Toggle individual
      tr.classList.toggle("selected");
      if (tr.classList.contains("selected")) _anchor = idx;
    } else {
      // Plain click = single select (clear others, select this one)
      _clearAll(rows);
      tr.classList.add("selected");
      _anchor = idx;
    }
    _applySlots();

    if (!_selectionHintShown && typeof showToast === "function") {
      _selectionHintShown = true;
      showToast("Tip: Ctrl+click to toggle, Shift+click for range", "info");
    }
  });
}

(function () {
  "use strict";

  const $ = (id) => document.getElementById(id);

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
    console.log('Initializing mode switching...');
    const modes = document.querySelectorAll(".topbar__mode");
    console.log('Found', modes.length, 'mode buttons');

    const onModeSelect = (btn) => { if (btn?.dataset?.mode) switchMode(btn.dataset.mode); };
    modes.forEach((btn) => {
      btn.addEventListener("click", (e) => { onModeSelect(btn); });
      btn.addEventListener("mousedown", (e) => { if (e.button === 0) onModeSelect(btn); }); // WebKit fallback
      btn.onclick = () => { onModeSelect(btn); return true; };
    });

    // Document-level delegation fallback (pywebview/WebKit may not deliver direct handlers)
    document.addEventListener("click", (e) => {
      const btn = e.target.closest(".topbar__mode");
      if (btn) onModeSelect(btn);
    });

    /* Alt+1/2/3/4 handled by Palette keybind system (palette.js) */
    console.log('Mode switching initialized');
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
        }
      });
    });
  }

  /* ---- Help tooltip [?] toggle ---- */
  function initHelpTooltips() {
    document.addEventListener("click", (e) => {
      const icon = e.target.closest(".help-icon");
      if (!icon) return;
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
        card.addEventListener("mousedown", (e) => { if (e.button === 0) onCardSelect(card, group); }); // WebKit fallback
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
    // Ez Mode start — W10: delegate to gatherFullConfig for complete config
    $("btn-start-ez")?.addEventListener("click", () => {
      if (typeof Validation !== 'undefined' && !Validation.validateAll()) {
        showToast("Fix validation errors before starting", "error"); return;
      }
      const config = typeof WorkspaceConfig !== 'undefined' ? WorkspaceConfig.gatherFullConfig() : {};
      // Override with Ez-specific selections
      const adapter = document.querySelector('input[name="ez-adapter"]:checked')?.value || config.adapter_type || "lora";
      const variant = $("ez-model-variant")?.value || config.model_variant || "turbo";
      const runName = adapter + "_" + variant + "_" + _timestamp();
      config.adapter_type = adapter;
      config.model_variant = variant;
      config.dataset_dir = $("ez-dataset-dir")?.value || config.dataset_dir || "";
      config.run_name = runName;
      config.steps_per_epoch = _stepsPerEpoch(config.batch_size, config.grad_accum);
      config.output_dir = _joinPath($("settings-adapters-dir")?.value || "./trained_adapters", adapter, runName);

      if (!config.dataset_dir) {
        showToast("Please select a dataset first", "warn");
        return;
      }

      // Console update delegated to extracted module
      Training.start(config);
    });

    // Advanced start
    $("btn-start-full")?.addEventListener("click", () => {
      // Validate all fields before start
      if (typeof Validation !== 'undefined' && !Validation.validateAll()) {
        showToast("Fix validation errors before starting", "error");
        return;
      }
      const config = typeof WorkspaceConfig !== 'undefined' ? WorkspaceConfig.gatherFullConfig() : {};
      config.run_name = (config.run_name || "run") + "_" + _timestamp();
      config.steps_per_epoch = _stepsPerEpoch(config.batch_size, config.grad_accum);
      if (!config.output_dir) {
        config.output_dir = _joinPath($("settings-adapters-dir")?.value || "./trained_adapters", config.adapter_type || "lora", config.run_name);
      }

      if (!config.dataset_dir) {
        showToast("Please select a dataset first", "warn");
        return;
      }

      // Console update delegated to extracted module
      Training.start(config);
    });

    // Monitor controls
    $("btn-stop")?.addEventListener("click", () => {
      Training.stop();
    });
    $("btn-monitor-to-ez")?.addEventListener("click", () => switchMode("ez"));
  }

  function _timestamp() {
    const d = new Date();
    return d.getFullYear().toString() +
      String(d.getMonth() + 1).padStart(2, "0") +
      String(d.getDate()).padStart(2, "0") + "_" +
      String(d.getHours()).padStart(2, "0") +
      String(d.getMinutes()).padStart(2, "0");
  }

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
    el.textContent = `"${mottos[Math.floor(Math.random() * mottos.length)]}"`;
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
  function init() {
    try {
      console.log('Initializing workspace modules...');

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
      console.log('Core UI initialized');

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

      initShiftClickTable("datasets-tbody");
      initShiftClickTable("history-tbody", { colorSlots: true });

      ["btn-start-ez", "btn-start-full", "btn-resume-start",
       "btn-start-preprocess", "btn-run-ppplus", "btn-run-captions"].forEach(id => guardDoubleClick(id, 3000));

      initGPU();

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
      ];
      for (const [name, mod] of modules) {
        if (!mod) continue;
        try {
          if (typeof mod.init === "function") { mod.init(); console.log(name + " initialized"); }
        } catch (e) { console.error(name + " init failed:", e); }
      }

      console.log('All workspace modules initialized successfully');

    } catch (error) {
      console.error('Workspace initialization failed:', error);
      console.error('Stack trace:', error.stack);
    }
  }

  function initWelcome(savedSettings) {
    const ov = document.getElementById("welcome-overlay");
    if (!ov) return;
    // Branch on first_run_complete from backend or localStorage
    let firstRunComplete = savedSettings?.first_run_complete === true;
    try {
      if (localStorage.getItem("sidestep_welcomed") === "done") firstRunComplete = true;
    } catch (e) { /* Storage unavailable in webview — non-fatal */ }
    if (firstRunComplete) {
      ov.classList.add("hidden");
      ov.style.display = "none";
      ov.style.pointerEvents = "none";
      return;
    }
    const _dismiss = () => {
      try { localStorage.setItem("sidestep_welcomed", "done"); } catch (e) {}
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
      welcomeSaveBtn.onclick = async function(e) { e.preventDefault(); await handleWelcomeSave(); return true; };
    }
    async function handleWelcomeSave() {
      [["welcome-checkpoint-dir","settings-checkpoint-dir"],["welcome-audio-dir","settings-audio-dir"],["welcome-tensors-dir","settings-tensors-dir"],["welcome-gemini-key","settings-gemini-key"],["welcome-openai-key","settings-openai-key"]]
        .forEach(([s,d]) => { const sv = $(s), dv = $(d); if (sv?.value && dv) dv.value = sv.value; });
      const data = {
        first_run_complete: true,
        checkpoint_dir: $("settings-checkpoint-dir")?.value,
        audio_dir: $("settings-audio-dir")?.value,
        preprocessed_tensors_dir: $("settings-tensors-dir")?.value,
        gemini_api_key: $("settings-gemini-key")?.value,
        openai_api_key: $("settings-openai-key")?.value,
      };
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
    gemini_api_key:           "settings-gemini-key",
    openai_api_key:           "settings-openai-key",
    openai_base_url:          "settings-openai-base",
    genius_api_token:         "settings-genius-token",
  };

  function _settingsDomId(key) {
    return _SETTINGS_DOM_MAP[key] || ("settings-" + key.replace(/_/g, "-"));
  }

  async function boot() {
    try {
      console.log('Workspace boot starting...');

      // Hide overlay FIRST so it cannot block even if later steps throw
      const ov = document.getElementById("welcome-overlay");
      if (ov) { ov.classList.add("hidden"); ov.style.display = "none"; ov.style.pointerEvents = "none"; }

      // Token fallback: some backends (e.g. GTK/WebKit) may not preserve URL params or injection
      if (!new URLSearchParams(window.location.search).get('token') &&
          typeof window.__SIDESTEP_TOKEN__ === 'undefined' &&
          window.pywebview?.api?.getToken) {
        try {
          window.__SIDESTEP_TOKEN__ = await window.pywebview.api.getToken();
        } catch (e) {
          console.warn('[boot] getToken fallback failed:', e);
        }
      }

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
        }
      } catch (e) {
        console.warn('[boot] could not load saved settings:', e);
      }

      initWelcome(savedSettings);
      
      console.log('Calling init functions...');
      init();
      
      // Populate pickers AFTER settings are in DOM (fixes boot race condition)
      if (typeof WorkspaceSetup !== "undefined") {
        await WorkspaceSetup.populatePickers();
        if (typeof AppState !== "undefined") AppState.setStatus("idle");
        console.log('WorkspaceSetup initialized');
      } else {
        console.error('WorkspaceSetup not available');
      }
      console.log('Workspace boot completed successfully');
      
    } catch (error) {
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
      if (window.pywebview?.api?.onBootError) {
        window.pywebview.api.onBootError(msg);
      }
    }
  }

  // Signal server shutdown when the tab/window closes.
  // Server uses a 3s delayed exit — cancelled if the page reloads (refresh).
  window.addEventListener("beforeunload", () => {
    if (typeof API !== "undefined" && API.signalShutdown) API.signalShutdown();
  });

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", boot);
  } else {
    boot();
  }
})();
