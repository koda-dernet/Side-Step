/* ============================================================
   Side-Step GUI — Setup & Navigation
   File browser, presets grid, settings slide-out, preset browser,
   populate pickers, Ez-to-Full transitions.
   ============================================================ */

const WorkspaceSetup = (() => {
  "use strict";

  const _e = (s) => typeof Validation !== 'undefined' ? Validation.esc(s) : String(s ?? '').replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;');

  /* ---- File Browser Modal ---- */
  let _browseTarget = null;

  function _defaultBrowsePath() {
    const candidates = ["settings-audio-dir", "settings-checkpoint-dir", "settings-tensors-dir", "settings-adapters-dir"];
    for (const id of candidates) { const v = $(id)?.value?.trim(); if (v) return v; }
    return ".";
  }

  function initFileBrowser() {
    document.addEventListener("click", function(e) {
      const btn = e.target.closest(".browse-btn");
      if (!btn) return;
      _browseTarget = btn.dataset.target;
      _openFileBrowser($(btn.dataset.target)?.value || _defaultBrowsePath());
    });
    const bc = $("file-browser-breadcrumb");
    if (bc) bc.addEventListener("click", function(e) {
      const seg = e.target.closest(".fb-breadcrumb__seg");
      if (seg?.dataset.fbPath) _navigateFileBrowser(seg.dataset.fbPath);
    });

    $("file-browser-close")?.addEventListener("click", _closeFileBrowser);
    $("file-browser-cancel")?.addEventListener("click", _closeFileBrowser);

    $("file-browser-select")?.addEventListener("click", () => {
      const path = ($("file-browser-path")?.value || "/").trim();
      if (_browseTarget) {
        const input = $(_browseTarget);
        if (input) { input.value = path; input.dispatchEvent(new Event("input")); input.dispatchEvent(new Event("change")); }
      }
      _closeFileBrowser();
      showToast("Path selected: " + path, "info");
    });

    $("file-browser-path")?.addEventListener("keydown", function(e) {
      if (e.key === "Enter") {
        e.preventDefault();
        _navigateFileBrowser(($("file-browser-path")?.value || "/").trim());
      }
    });

    $("file-browser-go")?.addEventListener("click", () => {
      _navigateFileBrowser(($("file-browser-path")?.value || "/").trim());
    });

    $("file-browser-list")?.addEventListener("click", function(e) {
      const item = e.target.closest(".file-browser__item--dir");
      if (item?.dataset.path) _navigateFileBrowser(item.dataset.path);
    });
  }

  async function _openFileBrowser(startPath) {
    $("file-browser-modal")?.classList.add("open");
    const errEl = $("file-browser-error"); if (errEl) errEl.style.display = "none";
    await _navigateFileBrowser(startPath || "/");
  }

  function _closeFileBrowser() { $("file-browser-modal")?.classList.remove("open"); _browseTarget = null; }

  function _renderBreadcrumb(path) {
    const bc = $("file-browser-breadcrumb"); if (!bc) return;
    const parts = path.split("/").filter(Boolean);
    let html = '<span class="fb-breadcrumb__seg" data-fb-path="/">/</span>';
    let cumPath = "";
    parts.forEach((p) => {
      cumPath += "/" + p;
      html += '<span class="fb-breadcrumb__sep">\u203a</span><span class="fb-breadcrumb__seg" data-fb-path="' + _e(cumPath) + '">' + _e(p) + '</span>';
    });
    bc.innerHTML = html;
  }

  async function _navigateFileBrowser(path) {
    const cleaned = (path || "/").replace(/\/+$/, "") || "/";
    const pathInput = $("file-browser-path"); if (pathInput) pathInput.value = cleaned;
    const hintEl = $("file-browser-selected-hint"); if (hintEl) hintEl.textContent = "Selected: " + cleaned;
    const errEl = $("file-browser-error"); if (errEl) errEl.style.display = "none";
    _renderBreadcrumb(cleaned);
    let result;
    try {
      result = await API.browseDir(cleaned);
      if (result.error && /outside allowed|allowed scope|403/i.test(result.error)) {
        result = { error: "This path is outside the allowed scope. Use Settings to configure your directories first, then browse within them.", entries: [] };
      }
    } catch (e) { result = { error: e.message || "Failed to browse" }; }
    const list = $("file-browser-list"); if (!list) return;
    if (result.error) {
      list.innerHTML = "";
      if (errEl) { errEl.textContent = result.error; errEl.style.display = ""; }
      return;
    }
    const entries = result.entries || [];
    if (!entries.length) { list.innerHTML = '<div class="fb-empty">Empty directory</div>'; return; }
    let html = "";
    if (cleaned !== "/" && cleaned !== ".") {
      const parent = cleaned.replace(/[\\/][^\\/]+$/, "") || "/";
      html += '<div class="file-browser__item file-browser__item--dir" data-path="' + _e(parent) + '"><span class="fb-icon">..</span><span>Parent directory</span></div>';
    }
    entries.forEach((e) => {
      const full = cleaned === "/" ? "/" + e.name : cleaned + "/" + e.name;
      const isDir = e.is_dir !== false;
      const cls = isDir ? "file-browser__item--dir" : "file-browser__item--file";
      const dp = isDir ? ' data-path="' + _e(full) + '"' : "";
      html += '<div class="file-browser__item ' + cls + '"' + dp + '><span class="fb-icon">' + (isDir ? "\u25b8" : "\u00b7") + '</span><span>' + _e(e.name) + (isDir ? "/" : "") + '</span></div>';
    });
    list.innerHTML = html;
  }

  /* ---- Ez -> Advanced Mode Transitions ---- */
  function initEzToFull() {
    on("click", ".ez-default-val", function() {
      const fieldId = this.dataset.field;
      switchMode("full");
      if (fieldId) {
        const target = $(fieldId);
        if (target) {
          const group = target.closest(".section-group");
          if (group && !group.classList.contains("open")) group.classList.add("open");
          target.scrollIntoView({ behavior: "smooth", block: "center" });
          target.classList.add('highlight-field');
          target.focus();
          setTimeout(() => target.classList.remove('highlight-field'), 1500);
        }
      }
    });
    on("click", ".ez-to-full-link", function(e) { e.preventDefault(); switchMode("full"); });
  }

  /* ---- Presets Grid ---- */
  async function initPresets() {
    const grid = $("presets-grid");
    if (!grid) return;
    const presets = await API.fetchPresets();
    grid.innerHTML = "";
    presets.forEach((p) => {
      const card = document.createElement("div");
      card.className = "preset-card";
      const tag = p.builtin
        ? '<span class="preset-card__tag preset-card__tag--builtin">built-in</span>'
        : '<span class="preset-card__tag preset-card__tag--user">user</span>';
      card.innerHTML = `
        <div class="preset-card__name">${_e(p.name)} ${tag}</div>
        <div class="preset-card__desc">${_e(p.description || "")}</div>
        <div class="preset-card__meta">${_e(p.adapter_type || "lora")} | r=${_e(p.rank || "?")} | lr=${_e(p.lr || "?")} | ${_e(p.epochs || "?")} epochs</div>
        <div class="preset-card__actions">
          <button class="btn btn--sm btn--primary preset-apply" data-name="${_e(p.name)}">Apply to Configure</button>
          ${!p.builtin ? '<button class="btn btn--sm btn--danger preset-delete" data-name="' + _e(p.name) + '">Delete</button>' : ""}
        </div>
      `;
      grid.appendChild(card);
    });

    const _aliases = {
      learning_rate:"lr", gradient_accumulation:"grad_accum", gradient_accumulation_steps:"grad_accum",
      scheduler_type:"scheduler", max_epochs:"epochs", save_every_n_epochs:"save_every",
      log_every_n_steps:"log_every", gradient_checkpointing:"gradient_checkpointing_ratio",
      target_modules_str:"projections", target_modules:"projections",
    };
    const valMap = {
      adapter_type:"full-adapter-type", model_variant:"full-model-variant", rank:"full-rank", alpha:"full-alpha", dropout:"full-dropout",
      lr:"full-lr", batch_size:"full-batch", grad_accum:"full-grad-accum", epochs:"full-epochs", warmup_steps:"full-warmup",
      max_steps:"full-max-steps", dataset_repeats:"full-dataset-repeats", shift:"full-shift", num_inference_steps:"full-inference-steps",
      cfg_ratio:"full-cfg-dropout", loss_weighting:"full-loss-weighting", snr_gamma:"full-snr-gamma", gradient_checkpointing_ratio:"full-grad-ckpt-ratio",
      chunk_duration:"full-chunk-duration", chunk_decay_every:"full-chunk-decay-every", optimizer_type:"full-optimizer", scheduler:"full-scheduler",
      scheduler_formula:"full-scheduler-formula", device:"full-device", precision:"full-precision", save_every:"full-save-every",
      log_every:"full-log-every", log_heavy_every:"full-log-heavy-every", save_best_after:"full-save-best-after", early_stop:"full-early-stop",
      weight_decay:"full-weight-decay", max_grad_norm:"full-max-grad-norm", seed:"full-seed", warmup_start_factor:"full-warmup-start-factor",
      cosine_eta_min_ratio:"full-cosine-eta-min", cosine_restarts_count:"full-cosine-restarts", ema_decay:"full-ema-decay", val_split:"full-val-split",
      adaptive_timestep_ratio:"full-adaptive-timestep", save_best_every_n_steps:"full-save-best-every-n-steps",
      timestep_mu:"full-timestep-mu", timestep_sigma:"full-timestep-sigma", num_workers:"full-num-workers", prefetch_factor:"full-prefetch-factor",
      bias:"full-bias", attention_type:"full-attention-type", lokr_linear_dim:"full-lokr-dim", lokr_linear_alpha:"full-lokr-alpha",
      lokr_factor:"full-lokr-factor", loha_linear_dim:"full-loha-dim", loha_linear_alpha:"full-loha-alpha", loha_factor:"full-loha-factor",
      oft_block_size:"full-oft-block-size", oft_eps:"full-oft-eps",
    };
    const chkMap = {
      target_mlp:"full-target-mlp", offload_encoder:"full-offload-encoder",
      save_best:"full-save-best", pin_memory:"full-pin-memory",
      persistent_workers:"full-persistent-workers",
      lokr_decompose_both:"full-lokr-decompose-both", lokr_use_tucker:"full-lokr-use-tucker",
      lokr_use_scalar:"full-lokr-use-scalar", lokr_weight_decompose:"full-lokr-weight-decompose",
      loha_use_tucker:"full-loha-use-tucker", loha_use_scalar:"full-loha-use-scalar",
      oft_coft:"full-oft-coft",
    };
    const onPresetApply = async function() {
      const raw = await API.loadPreset(this.dataset.name);
      if (!raw) return;
      const p = {};
      Object.entries(raw).forEach(([k, v]) => { p[_aliases[k] || k] = v; });
      if (typeof raw.gradient_checkpointing === "boolean") p.gradient_checkpointing_ratio = raw.gradient_checkpointing ? "1.0" : "0.0";
      const _touched = [];
      Object.entries(valMap).forEach(([k, id]) => { if (p[k] != null) { const el = $(id); if (el) { el.value = p[k]; _touched.push(el); } } });
      Object.entries(chkMap).forEach(([k, id]) => { if (p[k] != null) { const el = $(id); if (el) { el.checked = !!p[k]; _touched.push(el); } } });
      _touched.forEach((el) => {
        el.dispatchEvent(new Event("change", { bubbles: true }));
        if (el.type !== "checkbox" && el.tagName !== "SELECT") el.dispatchEvent(new Event("input", { bubbles: true }));
      });
      $("preset-browser-modal")?.classList.remove("open", "active");
      switchMode("full");
      showToast("Preset '" + this.dataset.name + "' applied (" + Object.keys(p).length + " fields)", "ok");
      if (typeof Validation !== "undefined") Validation.validateAll();
      if (typeof WorkspaceConfig !== "undefined") WorkspaceConfig.updateEzReview();
      if (typeof VRAM !== "undefined") VRAM.recalculate();
    };
    const onPresetDelete = function() {
      const name = this.dataset.name;
      const _doDelete = async () => {
        await API.deletePreset(name);
        showToast("Preset '" + name + "' deleted", "ok");
        initPresets();
      };
      if (typeof WorkspaceBehaviors !== "undefined" && WorkspaceBehaviors.showConfirmModal) {
        WorkspaceBehaviors.showConfirmModal("Delete Preset", "Delete preset '" + name + "'? This cannot be undone.", "Delete", _doDelete);
      } else { _doDelete(); }
    };
    on("click", ".preset-apply", onPresetApply);
    on("click", ".preset-delete", onPresetDelete);
  }

  /* ---- Settings Slide-out ---- */
  function initSettings() {
    const panel = $("settings-panel");
    const _emitSettingsSaved = (data) => {
      document.dispatchEvent(new CustomEvent("sidestep:settings-saved", { detail: data || {} }));
    };

    const _MASK_CHAR = "\u2022";
    function _isMasked(v) { return typeof v === "string" && v.includes(_MASK_CHAR); }
    function _gatherSettings() {
      const raw = {
        checkpoint_dir: $("settings-checkpoint-dir")?.value,
        trained_adapters_dir: $("settings-adapters-dir")?.value,
        preprocessed_tensors_dir: $("settings-tensors-dir")?.value,
        audio_dir: $("settings-audio-dir")?.value,
        gemini_api_key: $("settings-gemini-key")?.value,
        gemini_model: $("settings-gemini-model")?.value,
        openai_api_key: $("settings-openai-key")?.value,
        openai_base_url: $("settings-openai-base")?.value,
        genius_api_token: $("settings-genius-token")?.value,
      };
      const out = {};
      Object.entries(raw).forEach(([k, v]) => { if (v != null && !_isMasked(v)) out[k] = v; });
      return out;
    }

    on("click", "#btn-open-settings", () => { if (panel) panel.classList.add("open"); });

    on("click", "#settings-panel-close", () => {
      const settings = _gatherSettings();
      API.saveSettings(settings)
        .then(() => { populatePickers(); _emitSettingsSaved(settings); })
        .catch(() => { showToast("Failed to save settings", "error"); });
      if (panel) panel.classList.remove("open");
    });

    on("click", "#btn-save-settings", async () => {
      const settings = _gatherSettings();
      try {
        await API.saveSettings(settings);
        if (panel) panel.classList.remove("open");
        showToast("Settings saved", "ok");
        populatePickers();
        _emitSettingsSaved(settings);
      } catch (e) {
        showToast("Failed to save settings: " + e.message, "error");
      }
    });

    on("click", "#btn-validate-gemini", async () => {
      const key = $("settings-gemini-key")?.value;
      if (!key || key.includes("\u2022")) { showToast("Enter a Gemini API key first", "warn"); return; }
      const model = $("settings-gemini-model")?.value || "gemini-2.0-flash";
      showToast("Validating Gemini key...", "info");
      try {
        const r = await API.validateApiKey("gemini", key, { model });
        showToast(r.valid ? "Gemini key is valid" : "Gemini key invalid: " + (r.error || "unknown"), r.valid ? "ok" : "error");
      } catch (e) { showToast("Validation failed: " + e.message, "error"); }
    });

    on("click", "#btn-validate-openai", async () => {
      const key = $("settings-openai-key")?.value;
      if (!key || key.includes("\u2022")) { showToast("Enter an OpenAI API key first", "warn"); return; }
      const base = $("settings-openai-base")?.value || null;
      showToast("Validating OpenAI key...", "info");
      try {
        const r = await API.validateApiKey("openai", key, { base_url: base });
        showToast(r.valid ? "OpenAI key is valid" : "OpenAI key invalid: " + (r.error || "unknown"), r.valid ? "ok" : "error");
      } catch (e) { showToast("Validation failed: " + e.message, "error"); }
    });

    on("click", "#settings-restart-tutorial", function(e) {
      e.preventDefault();
      $("settings-panel")?.classList.remove("open");
      if (typeof Tutorial !== "undefined") { Tutorial.reset(); Tutorial.start(); }
    });

    on("click", "#settings-open-keybinds", function(e) {
      e.preventDefault();
      $("settings-panel")?.classList.remove("open");
      if (typeof Palette !== "undefined") Palette.openKeybindModal();
    });

    try { localStorage.removeItem('sidestep_crt'); } catch (e) {}
    document.documentElement.classList.remove('crt-active');
  }

  /* ---- Preset Browser Modal ---- */
  function initPresetBrowser() {
    const modal = $("preset-browser-modal");
    const openModal = async () => { if (modal) { await initPresets(); modal.classList.add("open"); } };
    const closeModal = () => { if (modal) modal.classList.remove("open"); };
    on("click", "#ez-load-preset, #btn-load-preset", (e) => { e.preventDefault(); openModal(); });
    on("click", "#preset-browser-close", closeModal);
  }

  /* ---- Populate pickers from Settings paths ---- */
  async function populatePickers() {
    try {
      const ckptDir = $("settings-checkpoint-dir")?.value || "./checkpoints";
      const models = await API.fetchModels(ckptDir);
      const variants = (models.models || []).map(m => m.name || m);
      document.querySelectorAll(".model-picker").forEach((sel) => {
        const current = sel.value;
        BatchDOM.setOptions(sel, variants, current);
      });
      const det = $("ez-checkpoint-detect");
      if (det) {
        if (variants.length) { det.textContent = "[ok] Found: " + variants.join(", "); det.className = "detect detect--ok"; }
        else { det.textContent = "No models found in " + ckptDir; det.className = "detect detect--warn"; }
      }
    } catch (e) {
      document.querySelectorAll(".model-picker").forEach((sel) => {
        if (!sel.options.length) {
          const opt = document.createElement("option");
          opt.value = "";
          opt.textContent = "Failed to load \u2014 check Settings & token";
          sel.appendChild(opt);
        }
      });
      const det = $("ez-checkpoint-detect");
      if (det) { det.textContent = "Could not load models \u2014 check checkpoint dir in Settings"; det.className = "detect detect--warn"; }
      if (typeof showToast === "function") showToast("Could not load models (check checkpoint dir and token)", "warn");
    }

    try {
      const tensorsDir = $("settings-tensors-dir")?.value || "./preprocessed_tensors";
      const result = await API.scanTensorsDir(tensorsDir);
      const folders = (result.datasets || result.folders || []).map(d => ({
        name: d.name, files: d.count || d.files || 0, duration: d.duration_label || d.duration || "?",
        pp_map: d.pp_map ?? false, path: d.path || (tensorsDir + "/" + d.name).replace(/\/+/g, "/"),
      }));
      document.querySelectorAll(".dataset-picker").forEach((sel) => {
        const current = sel.value;
        const datasetOptions = folders.map((f) => ({
          value: f.path,
          label: f.name + " \u2014 " + f.files + " files, " + f.duration + (f.pp_map ? " [PP++]" : "")
        }));
        BatchDOM.setOptions(sel, [{value: "", label: "Select a dataset..."}, ...datasetOptions], current);
      });
    } catch (e) { /* silent */ }

    const gemKey = $("settings-gemini-key")?.value || "";
    const oaiKey = $("settings-openai-key")?.value || "";
    const genKey = $("settings-genius-token")?.value || "";
    const _setBadgeState = (el, configured) => {
      if (!el) return;
      el.textContent = configured ? "configured [ok]" : "not set";
      el.style.color = "";
      el.classList.toggle("u-text-success", configured);
      el.classList.toggle("u-text-muted", !configured);
    };
    _setBadgeState($("caption-gemini-badge"), gemKey && !gemKey.includes("Not set"));
    _setBadgeState($("caption-openai-badge"), !!oaiKey);
    _setBadgeState($("caption-genius-badge"), !!genKey);
  }

  function init() {
    initFileBrowser();
    initEzToFull();
    initSettings();
    initPresetBrowser();
  }

  return { init, initPresets, populatePickers };
})();
