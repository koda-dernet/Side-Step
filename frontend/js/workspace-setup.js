/* ============================================================
   Side-Step GUI — Setup & Navigation
   File browser, presets grid, settings slide-out, preset browser,
   populate pickers, Ez-to-Full transitions.
   ============================================================ */

const WorkspaceSetup = (() => {
  "use strict";

  const _e = window._esc || ((s) => String(s ?? ''));

  /** Normalize path separators to forward slash for consistent handling. */
  function _normPath(p) {
    return String(p || '').replace(/\\/g, '/');
  }

  /** True when a path segment looks like a Windows drive letter (e.g. "C:"). */
  function _isDrive(seg) {
    return /^[A-Za-z]:$/.test(seg);
  }

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
    const norm = _normPath(path);
    const parts = norm.split("/").filter(Boolean);
    let html = '<span class="fb-breadcrumb__seg" data-fb-path="/">/</span>';
    let cumPath = "";
    parts.forEach((p) => {
      if (!cumPath && _isDrive(p)) {
        cumPath = p + "/";
      } else {
        cumPath += (cumPath && !cumPath.endsWith("/") ? "/" : "") + p;
      }
      html += '<span class="fb-breadcrumb__sep">\u203a</span><span class="fb-breadcrumb__seg" data-fb-path="' + _e(cumPath) + '">' + _e(p) + '</span>';
    });
    bc.innerHTML = html;
  }

  async function _navigateFileBrowser(path) {
    let cleaned = _normPath(path || "/").replace(/\/+$/, "") || "/";
    if (/^[A-Za-z]:$/.test(cleaned)) cleaned += "/";
    const pathInput = $("file-browser-path"); if (pathInput) pathInput.value = cleaned;
    const hintEl = $("file-browser-selected-hint"); if (hintEl) hintEl.textContent = "Selected: " + cleaned;
    const errEl = $("file-browser-error"); if (errEl) errEl.style.display = "none";
    _renderBreadcrumb(cleaned);
    let result;
    try {
      result = await API.browseDir(cleaned);
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
      let parent;
      if (/^[A-Za-z]:\/$/.test(cleaned)) {
        parent = "/";
      } else {
        parent = cleaned.replace(/\/[^\/]+\/?$/, "") || "/";
        if (/^[A-Za-z]:$/.test(parent)) parent += "/";
      }
      html += '<div class="file-browser__item file-browser__item--dir" data-path="' + _e(parent) + '"><span class="fb-icon">..</span><span>Parent directory</span></div>';
    }
    entries.forEach((e) => {
      const full = _normPath(e.path) || (cleaned === "/" ? "/" + e.name : cleaned + "/" + e.name);
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
        <div class="preset-card__meta">${_e(p.adapter_type || "lora")} | r=${_e(p.rank || "?")} | lr=${_e(p.learning_rate || "?")} | ${_e(p.epochs || "?")} epochs</div>
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
      self_target_modules_str:"self_projections", cross_target_modules_str:"cross_projections",
      early_stop_patience:"early_stop",
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
      target_loss:"full-target-loss", target_loss_floor:"full-target-loss-floor",
      target_loss_warmup:"full-target-loss-warmup", target_loss_smoothing:"full-target-loss-smoothing",
      timestep_mu:"full-timestep-mu", timestep_sigma:"full-timestep-sigma", num_workers:"full-num-workers", prefetch_factor:"full-prefetch-factor",
      bias:"full-bias", attention_type:"full-attention-type", lokr_linear_dim:"full-lokr-dim", lokr_linear_alpha:"full-lokr-alpha",
      lokr_factor:"full-lokr-factor", loha_linear_dim:"full-loha-dim", loha_linear_alpha:"full-loha-alpha", loha_factor:"full-loha-factor",
      oft_block_size:"full-oft-block-size", oft_eps:"full-oft-eps",
      projections:"full-projections", self_projections:"full-self-projections", cross_projections:"full-cross-projections",
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
    function _syncProjChipsFromHidden() {
      const _syncOne = (hiddenId, containerId) => {
        const hidden = $(hiddenId), container = $(containerId);
        if (!hidden || !container) return;
        const vals = new Set((hidden.value || "").split(/\s+/).filter(Boolean));
        container.querySelectorAll("input[type=checkbox]").forEach(cb => {
          cb.checked = vals.has(cb.value);
          const chip = cb.closest(".proj-chip");
          if (chip) chip.classList.toggle("active", cb.checked);
        });
      };
      const attnType = $("full-attention-type")?.value || "both";
      if (attnType === "both") {
        _syncOne("full-self-projections", "proj-chips-self");
        _syncOne("full-cross-projections", "proj-chips-cross");
      } else {
        _syncOne("full-projections", "proj-chips");
      }
    }
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
      // Sync projection chip checkboxes to match hidden input values
      _syncProjChipsFromHidden();
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
    const _SENSITIVE_KEYS = new Set(["gemini_api_key", "openai_api_key", "genius_api_token", "hf_token"]);
    function _isMasked(v) { return typeof v === "string" && v.includes(_MASK_CHAR); }
    function _gatherSettings() {
      const raw = {
        checkpoint_dir: $("settings-checkpoint-dir")?.value,
        trained_adapters_dir: $("settings-adapters-dir")?.value,
        preprocessed_tensors_dir: $("settings-tensors-dir")?.value,
        audio_dir: $("settings-audio-dir")?.value,
        exported_loras_dir: $("settings-exported-loras-dir")?.value,
        gemini_api_key: $("settings-gemini-key")?.value,
        gemini_model: $("settings-gemini-model")?.value,
        openai_api_key: $("settings-openai-key")?.value,
        openai_base_url: $("settings-openai-base")?.value,
        genius_api_token: $("settings-genius-token")?.value,
        transcriber_server_url: $("settings-transcriber-server-url")?.value,
        music_flamingo_url: $("settings-music-flamingo-url")?.value,
        hf_token: $("settings-hf-token")?.value,
      };
      const out = {};
      Object.entries(raw).forEach(([k, v]) => {
        if (v == null || _isMasked(v)) return;
        if (_SENSITIVE_KEYS.has(k) && v === "") return;
        out[k] = v;
      });
      return out;
    }

    on("click", "#btn-open-settings", () => { if (panel) panel.classList.add("open"); });

    on("click", "#settings-panel-close", () => {
      const settings = _gatherSettings();
      API.saveSettings(settings)
        .then(() => { populatePickers(); _emitSettingsSaved(settings); if (panel) panel.classList.remove("open"); })
        .catch(() => { showToast("Failed to save settings", "error"); });
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
      const model = $("settings-gemini-model")?.value || "gemini-2.5-flash";
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
      const ckptDir = $("settings-checkpoint-dir")?.value || Defaults.get("settings-checkpoint-dir") || "checkpoints";
      const models = await API.fetchModels(ckptDir);
      const variants = (models.models || []).map(m => m.name || m);
      // Priority: base > sft > turbo when current value doesn't match any scanned model
      const _pickDefault = (variants, current) => {
        if (variants.includes(current)) return current;
        const lv = variants.map(v => v.toLowerCase());
        const baseIdx = lv.findIndex(v => v.includes("base"));
        if (baseIdx >= 0) return variants[baseIdx];
        const sftIdx = lv.findIndex(v => v.includes("sft"));
        if (sftIdx >= 0) return variants[sftIdx];
        const turboIdx = lv.findIndex(v => v.includes("turbo"));
        if (turboIdx >= 0) return variants[turboIdx];
        return variants[0] || current;
      };
      document.querySelectorAll(".model-picker").forEach((sel) => {
        const current = sel.value;
        const savedDefault = sel.dataset.default;
        const preferred = _pickDefault(variants, current);
        BatchDOM.setOptions(sel, variants, preferred);
        if (savedDefault) sel.dataset.default = savedDefault;
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
      const allResult = await API.fetchAllDatasets();
      const allDatasets = allResult.datasets || [];
      const tensorSets = allDatasets.filter(d => d.type === "tensors").map(d => ({
        name: d.name, files: d.count || d.files || 0, duration: d.duration_label || d.duration || "?",
        pp_map: d.pp_map ?? false, path: d.path, type: "tensors",
      }));
      const audioSets = allDatasets.filter(d => d.type === "audio").map(d => ({
        name: d.name, files: d.count || d.files || 0, duration: d.duration_label || d.duration || "?",
        pp_map: false, path: d.path, type: "audio",
      }));
      document.querySelectorAll(".dataset-picker").forEach((sel) => {
        const current = sel.value;
        const tensorOpts = tensorSets.map((f) => ({
          value: f.path,
          label: "\u25A0 " + f.name + " \u2014 " + f.files + " files, " + f.duration + (f.pp_map ? " [PP++]" : ""),
        }));
        const audioOpts = audioSets.map((f) => ({
          value: "audio:" + f.path,
          label: "\u266A " + f.name + " \u2014 " + f.files + " files, " + f.duration + " [needs preprocessing]",
        }));
        const options = [{value: "", label: "Select a dataset..."}];
        if (tensorOpts.length) options.push(...tensorOpts);
        if (audioOpts.length) options.push(...audioOpts);
        BatchDOM.setOptions(sel, options, current);
      });
    } catch (e) { console.error('[Setup] dataset scan failed:', e); }

    const gemKey = $("settings-gemini-key")?.value || "";
    const oaiKey = $("settings-openai-key")?.value || "";
    const genKey = $("settings-genius-token")?.value || "";
    const transcriberUrl = $("settings-transcriber-server-url")?.value || "";
    const musicFlamingoUrl = $("settings-music-flamingo-url")?.value || "";
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
    _setBadgeState($("caption-transcriber-badge"), !!transcriberUrl);
    _setBadgeState($("caption-music-flamingo-badge"), !!musicFlamingoUrl);

    if (typeof CustomSelect !== "undefined" && CustomSelect.refresh) CustomSelect.refresh();
  }

  function initRefreshButtons() {
    document.querySelectorAll(".btn--refresh").forEach((btn) => {
      btn.addEventListener("click", async () => {
        btn.disabled = true;
        const orig = btn.textContent;
        btn.textContent = "\u23f3";
        try { await populatePickers(); } catch (e) { console.warn("[refresh]", e); }
        btn.textContent = orig;
        btn.disabled = false;
        if (typeof showToast === "function") showToast("Picker lists refreshed", "ok");
      });
    });
  }

  function init() {
    initFileBrowser();
    initEzToFull();
    initSettings();
    initPresetBrowser();
    initRefreshButtons();
  }

  return { init, initPresets, populatePickers };
})();
