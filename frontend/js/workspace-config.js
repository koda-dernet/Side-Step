/* Side-Step GUI — Config / Review / Cross-mode Sync */

const WorkspaceConfig = (() => {
  "use strict";

  const $ = (id) => document.getElementById(id);

  /* ---- CLI Export ---- */
  function initCLIExport() {
    $("btn-export-cli")?.addEventListener("click", () => {
      const config = gatherFullConfig();
      const cmd = API.buildCLICommand(config);
      const panel = $("cli-export-panel");
      const content = $("cli-export-content");
      if (content) content.textContent = cmd;
      if (panel) panel.style.display = "block";
      _updateConsole("CLI command exported");
    });

    $("btn-copy-cli")?.addEventListener("click", async () => {
      const text = $("cli-export-content")?.textContent;
      if (!text) return;
      const copied = typeof copyTextToClipboard === "function"
        ? await copyTextToClipboard(text)
        : false;
      showToast(copied ? "CLI command copied" : "Failed to copy CLI command", copied ? "ok" : "error");
    });
  }

  /* ---- Review Panel Live Update ---- */
  function initReviewSync() {
    const fields = [
      ["full-model-variant", "rev-variant", v => v],
      ["full-model-variant", "rev-ckpt", v => _joinPath($("settings-checkpoint-dir")?.value || "./checkpoints", v)],
      ["full-adapter-type", "rev-adapter", v => v.toUpperCase()],
      ["full-rank", "rev-rank", v => v],
      ["full-alpha", "rev-alpha", v => v],
      ["full-dropout", "rev-dropout", v => v],
      ["full-lr", "rev-lr", v => v],
      ["full-epochs", "rev-epochs", v => v],
      ["full-warmup", "rev-warmup", v => v + " steps"],
      ["full-max-steps", "rev-max-steps", v => v === "0" ? "off" : v],
      ["full-dataset-repeats", "rev-repeats", v => v + "\u00d7"],
      ["full-grad-ckpt-ratio", "rev-ckpt-mode", v => parseFloat(v) > 0 ? "ratio " + v : "off"],
      ["full-chunk-duration", "rev-chunk", v => v === "0" ? "off" : v + "s"],
      ["full-optimizer", "rev-optimizer", v => v],
      ["full-scheduler", "rev-scheduler", v => v],
      ["full-weight-decay", "rev-weight-decay", v => v],
      ["full-max-grad-norm", "rev-grad-norm", v => v],
      ["full-save-every", "rev-save-every", v => v + " epochs"],
    ];

    fields.forEach(([srcId, targetId, fmt]) => {
      const src = $(srcId);
      if (!src) return;
      const update = () => {
        const t = $(targetId);
        if (t) {
          const val = fmt(src.value);
          const isDefault = src.dataset && src.dataset.default ? src.value === src.dataset.default : true;
          t.textContent = val;
          t.style.color = "";
          t.classList.toggle("review-table__val--modified", !isDefault);
        }
      };
      src.addEventListener("input", update);
      src.addEventListener("change", update);
      update();
    });

    const updateBatch = () => {
      const bs = parseInt($("full-batch")?.value) || 1;
      const ga = parseInt($("full-grad-accum")?.value) || 4;
      const t = $("rev-batch");
      if (t) t.textContent = `${bs} \u00d7 ${ga} = ${bs * ga}`;
    };
    $("full-batch")?.addEventListener("input", updateBatch);
    $("full-grad-accum")?.addEventListener("input", updateBatch);

    const updateOffload = () => {
      const t = $("rev-offload");
      if (t) t.textContent = $("full-offload-encoder")?.checked ? "yes" : "no";
    };
    $("full-offload-encoder")?.addEventListener("change", updateOffload);

    const updateSaveBest = () => {
      const t = $("rev-save-best");
      if (t) t.textContent = $("full-save-best")?.checked ? "yes" : "no";
    };
    $("full-save-best")?.addEventListener("change", updateSaveBest);

    const updateTargeting = () => {
      const t = $("rev-targeting");
      if (!t) return;
      const attn = $("full-attention-type")?.value || "both";
      const mlp = $("full-target-mlp")?.checked ? " + MLP" : "";
      t.textContent = attn + mlp;
    };
    $("full-attention-type")?.addEventListener("change", updateTargeting);
    $("full-target-mlp")?.addEventListener("change", updateTargeting);

    const updateResume = () => {
      const t = $("rev-resume");
      if (t) t.textContent = $("full-resume-from")?.value.trim() || "--";
    };
    $("full-resume-from")?.addEventListener("input", updateResume);
    $("full-resume-from")?.addEventListener("change", updateResume);

    const cfgGroup = $("cfg-dropout-group");
    const updateCfgVisibility = () => {
      const variant = String($("full-model-variant")?.value || "turbo").toLowerCase();
      if (cfgGroup) cfgGroup.style.display = variant.includes("turbo") ? "none" : "";
    };
    $("full-model-variant")?.addEventListener("change", updateCfgVisibility);
    $("ez-model-variant")?.addEventListener("change", updateCfgVisibility);
    updateCfgVisibility();

    document.querySelectorAll("#full-review-content .review-table__group[data-jump]").forEach((grp) => {
      grp.addEventListener("click", () => {
        const target = $(grp.dataset.jump);
        if (!target) return;
        const section = target.closest(".section-group");
        if (section && !section.classList.contains("open")) {
          section.querySelector(".section-group__toggle")?.click();
        }
        target.scrollIntoView({ behavior: "smooth", block: "center" });
        target.classList.add("highlight-field");
        setTimeout(() => target.classList.remove("highlight-field"), 1500);
      });
    });
  }

  /* ---- Cross-mode Sync (Ez <-> Full) ---- */
  function initCrossModeSync() {
    const modelPickers = document.querySelectorAll(".model-picker");
    modelPickers.forEach((picker) => {
      picker.addEventListener("change", () => {
        const val = picker.value;
        modelPickers.forEach((other) => {
          if (other !== picker) other.value = val;
        });
        _autoRunName();
        updateEzReview();
        _updateConsole("Model: " + val);
      });
    });

    const ezCards = document.querySelectorAll("#ez-adapter-cards .adapter-card");
    const fullAdapterSel = $("full-adapter-type");

    ezCards.forEach((card) => {
      card.addEventListener("click", () => {
        const radio = card.querySelector("input[type=radio]");
        if (radio && fullAdapterSel) {
          fullAdapterSel.value = radio.value;
          fullAdapterSel.dispatchEvent(new Event("change"));
        }
      });
    });

    if (fullAdapterSel) {
      fullAdapterSel.addEventListener("change", () => {
        const val = fullAdapterSel.value;
        ezCards.forEach((card) => {
          const radio = card.querySelector("input[type=radio]");
          card.classList.toggle("selected", radio && radio.value === val);
          if (radio) radio.checked = radio.value === val;
        });
        _autoRunName();
        _updateAdapterSection(val);
        updateEzReview();
        _updateConsole("Adapter: " + val.toUpperCase());
      });
    }

    // W3: Sync Ez/Full model variant pickers (bidirectional)
    $("ez-model-variant")?.addEventListener("change", () => {
      const val = $("ez-model-variant").value;
      if ($("full-model-variant")) {
        $("full-model-variant").value = val;
        $("full-model-variant").dispatchEvent(new Event("change"));
      }
      _autoRunName();
      updateEzReview();
    });
    $("full-model-variant")?.addEventListener("change", () => {
      const val = $("full-model-variant").value;
      if ($("ez-model-variant")) $("ez-model-variant").value = val;
      if ($("pp-model-variant")) $("pp-model-variant").value = val;
      if ($("ppplus-model-variant")) $("ppplus-model-variant").value = val;
    });

    $("ez-dataset-dir")?.addEventListener("change", () => {
      const val = $("ez-dataset-dir").value;
      if ($("full-dataset-dir")) $("full-dataset-dir").value = val;
      _onDatasetChange(val, "ez");
    });
    $("full-dataset-dir")?.addEventListener("change", () => {
      const val = $("full-dataset-dir").value;
      if ($("ez-dataset-dir")) $("ez-dataset-dir").value = val;
      if ($("ppplus-dataset-dir")) { $("ppplus-dataset-dir").value = val; $("ppplus-dataset-dir").dispatchEvent(new Event("change")); }
      _onDatasetChange(val, "full");
    });

    const fullInputs = document.querySelectorAll("#mode-full .input, #mode-full .select, #mode-full input[type=checkbox]");
    fullInputs.forEach((input) => {
      const evt = input.type === "checkbox" || input.tagName === "SELECT" ? "change" : "input";
      input.addEventListener(evt, updateEzReview);
    });
  }

  /* ---- Run Name Auto-generation ---- */
  let _runNameManuallyEdited = false;

  function _autoRunName() {
    if (_runNameManuallyEdited) return;
    const adapter = $("full-adapter-type")?.value || "lora";
    const variant = $("full-model-variant")?.value || "turbo";
    const name = adapter + "_" + variant;
    const input = $("full-run-name");
    if (input) {
      input.value = name;
      input.dataset.default = name;
      input.dispatchEvent(new Event("input"));
    }
  }

  function initRunNameTracking() {
    const input = $("full-run-name");
    if (!input) return;
    input.addEventListener("input", () => {
      _runNameManuallyEdited = true;
    });
    input.addEventListener("focus", () => {
      const val = input.value;
      if (val === input.dataset.default) _runNameManuallyEdited = false;
    });
  }

  /* ---- Ez Review Table (reactive) ---- */
  function updateEzReview() {
    const map = {
      "ez-rev-variant": () => $("full-model-variant")?.value || "turbo",
      "ez-rev-ckpt": () => _joinPath($("settings-checkpoint-dir")?.value || "./checkpoints", $("full-model-variant")?.value || "turbo"),
      "ez-rev-adapter": () => { const v = $("full-adapter-type")?.value || "lora"; return v.charAt(0).toUpperCase() + v.slice(1); },
      "ez-rev-rank": () => _v("full-rank","64"), "ez-rev-alpha": () => _v("full-alpha","128"),
      "ez-rev-dropout": () => _v("full-dropout","0.1"), "ez-rev-attention": () => _v("full-attention-type","both"),
      "ez-rev-mlp": () => _c("full-target-mlp") ? "yes" : "no", "ez-rev-lr": () => _v("full-lr","3e-4"),
      "ez-rev-batch": () => _v("full-batch","1") + " \u00d7 " + _v("full-grad-accum","4"), "ez-rev-epochs": () => _v("full-epochs","1000"),
      "ez-rev-warmup": () => _v("full-warmup","100") + " steps",
      "ez-rev-ckpt-mode": () => { const r = parseFloat(_v("full-grad-ckpt-ratio","1")); return r >= 1 ? "full" : r > 0 ? "ratio " + r : "off"; },
      "ez-rev-offload": () => _c("full-offload-encoder") ? "yes" : "no",
      "ez-rev-save-every": () => _v("full-save-every","50") + " epochs",
      "ez-rev-save-best": () => _c("full-save-best") ? "yes" : "no", "ez-rev-optimizer": () => _v("full-optimizer","adamw8bit"),
    };
    Object.entries(map).forEach(([id, fn]) => {
      const el = $(id);
      if (el) el.textContent = fn();
    });

    const defaultsRow = $("ez-defaults-row");
    if (defaultsRow) {
      const spans = defaultsRow.querySelectorAll(".ez-default-val");
      spans.forEach((span) => {
        const field = span.dataset.field;
        if (!field) return;
        const src = $(field);
        if (!src) return;
        if (field === "full-batch") {
          const bs = $("full-batch")?.value || "1";
          const ga = $("full-grad-accum")?.value || "4";
          span.textContent = bs + " \u00d7 " + ga;
        } else {
          span.textContent = src.value;
        }
        const isDefault = src.dataset && src.dataset.default ? src.value === src.dataset.default : true;
        span.style.color = "";
        span.classList.toggle("u-text-secondary", isDefault);
        span.classList.toggle("u-text-changed", !isDefault);
      });
    }

    if (typeof VRAM !== "undefined") {
      VRAM.recalculate();
    }
  }

  /* ---- Adapter Section Reactivity ---- */
  function _updateAdapterSection(adapter) {
    const names = { lora: "LoRA", dora: "DoRA", lokr: "LoKR", loha: "LoHA", oft: "OFT" };
    const title = $("full-adapter-section-title");
    if (title) title.textContent = (names[adapter] || adapter.toUpperCase()) + " Settings";

    const fieldMap = { lora: "lora", dora: "lora", lokr: "lokr", loha: "loha", oft: "oft" };
    const activeBlock = fieldMap[adapter] || "lora";
    document.querySelectorAll(".adapter-fields").forEach((el) => {
      el.style.display = "none";
    });
    const target = $("adapter-fields-" + activeBlock);
    if (target) target.style.display = "";

    const biasSelect = $("full-bias");
    if (biasSelect) {
      const loraOnly = biasSelect.querySelector('option[value="lora_only"]');
      if (loraOnly) loraOnly.textContent = (adapter === "lora" || adapter === "dora") ? "lora_only" : "adapter_only";
    }
  }

  /* ---- Dataset Picker Reactivity ---- */
  async function _onDatasetChange(val, source) {
    if (!val) return;
    try {
      const result = await API.scanTensorsDir($("settings-tensors-dir")?.value || "./preprocessed_tensors");
      const datasets = (result.datasets || result.folders || []);
      const _normPath = (p) => String(p || "").replace(/\\/g, "/").replace(/\/+$/, "");
      const _baseName = (p) => _normPath(p).split("/").filter(Boolean).pop() || "";
      const selectedPath = _normPath(val);
      const selectedName = _baseName(val);
      const tensorsRoot = $("settings-tensors-dir")?.value || "./preprocessed_tensors";
      const raw = datasets.find((f) => _normPath(f.path) === selectedPath)
        || datasets.find((f) => {
          const candidate = f.path || (typeof _joinPath === "function"
            ? _joinPath(tensorsRoot, f.name || "")
            : `${tensorsRoot}/${f.name || ""}`);
          return _normPath(candidate) === selectedPath;
        })
        || datasets.find((f) => String(f.name || "") === selectedName);
      if (!raw) return;
      const folder = { name: raw.name, files: raw.count || raw.files || 0, duration: raw.duration_label || raw.duration || "?", pp_map: raw.pp_map ?? false };

      const info = `[ok] ${folder.files} .pt tensors detected | ${folder.duration} total${folder.pp_map ? " | PP++ map: found" : ""}`;

      ["ez-dataset-detect", "full-dataset-detect"].forEach(id => {
        const el = $(id); if (el) { el.textContent = info; el.className = "detect detect--ok"; }
      });
      const fi = $("full-dataset-info");
      if (fi) {
        const ppCls = folder.pp_map ? "u-text-success" : "u-text-muted";
        const ppT = folder.pp_map ? "PP++ map: detected [ok]" : "PP++ map: not found";
        fi.innerHTML = '<div class="u-text-success">' + folder.files + ' samples, ' + folder.duration + ' total</div>' +
          '<div class="u-text-muted">~' + Math.ceil(folder.files * 1.5) + ' steps/epoch</div>' +
          '<div id="full-pp-status" class="' + ppCls + '">' + ppT + '</div>';
      }
      const hasPP = !!folder.pp_map;
      const ppTR = $("full-ppplus-toggle-row"); if (ppTR) ppTR.style.display = hasPP ? "" : "none";
      const ppTg = $("full-use-ppplus"); if (ppTg) ppTg.checked = hasPP;
      const ppRun = $("full-pp-run"); if (ppRun) ppRun.style.display = hasPP ? "none" : "block";

      _updateConsole("Dataset: " + folder.name + " (" + folder.files + " files)");
    } catch (e) { console.error('[Config] tensor scan failed:', e); }
  }

  /* ---- PP++ Fisher Map Status Check ---- */
  function initPPPlusStatus() {
    const _updateStatus = async () => {
      const dsDir = $("ppplus-dataset-dir").value, status = $("ppplus-status-content");
      if (!dsDir || !status) return;
      try {
        const variant = $("ppplus-model-variant")?.value || "";
        const r = await API.checkFisherMap(dsDir, variant);
        status.innerHTML = r.exists
          ? '<span class="u-text-success">[ok] fisher_map.json — ' + r.modules + ' modules, ranks ' + r.rank_min + '-' + r.rank_max + '</span>' + (r.stale ? ' <span class="u-text-warning">(stale?)</span>' : '')
          : '<span class="u-text-muted">No fisher_map.json found. Run PP++ to generate one.</span>';
      } catch (e) { status.innerHTML = '<span class="u-text-error">Error checking fisher map</span>'; }
    };

    $("ppplus-dataset-dir")?.addEventListener("change", _updateStatus);
    $("ppplus-model-variant")?.addEventListener("change", _updateStatus);
  }

  function _updateConsole(msg) { const el = $("console-line"); if (el) { el.textContent = msg; el.className = "console__line"; } }

  /* ---- Gather Full Config ---- */
  const _d = (id) => typeof Defaults !== 'undefined' && Defaults.get(id) || '';
  const _v = (id, def) => ($(id)?.value || _d(id) || def).toString().trim();
  const _c = (id) => $(id)?.checked;
  const _platformDefaultWorkers = () => {
    const p = String(window.__SIDESTEP_PLATFORM__ || "").toLowerCase();
    return p.startsWith("win") ? 0 : 4;
  };
  function gatherFullConfig() {
    const adapter = _v("full-adapter-type", "lora");
    const gradCkptRatio = parseFloat(_v("full-grad-ckpt-ratio", "1.0"));
    const gradCkptEnabled = !Number.isNaN(gradCkptRatio) ? gradCkptRatio > 0 : true;
    const cfg = {
      checkpoint_dir: _v("settings-checkpoint-dir", "./checkpoints"), model_variant: _v("full-model-variant", "turbo"),
      adapter_type: adapter, dataset_dir: _v("full-dataset-dir", ""), run_name: _v("full-run-name", ""),
      output_dir: _v("full-output-dir", ""), attention_type: _v("full-attention-type", "both"),
      projections: _v("full-projections", "q_proj k_proj v_proj o_proj"),
      self_projections: _v("full-self-projections", "q_proj k_proj v_proj o_proj"),
      cross_projections: _v("full-cross-projections", "q_proj k_proj v_proj o_proj"),
      target_mlp: _c("full-target-mlp"),
      bias: _v("full-bias", "none"), lr: _v("full-lr", "3e-4"),
      batch_size: _v("full-batch", "1"), grad_accum: _v("full-grad-accum", "4"),
      epochs: _v("full-epochs", "1000"), warmup_steps: _v("full-warmup", "100"), max_steps: _v("full-max-steps", "0"),
      shift: _v("full-shift", "3.0"), num_inference_steps: _v("full-inference-steps", "8"),
      cfg_ratio: _v("full-cfg-dropout", "0.15"), loss_weighting: _v("full-loss-weighting", "none"),
      snr_gamma: _v("full-snr-gamma", "5.0"), offload_encoder: _c("full-offload-encoder"),
      gradient_checkpointing: gradCkptEnabled,
      gradient_checkpointing_ratio: _v("full-grad-ckpt-ratio", "1.0"),
      chunk_duration: _v("full-chunk-duration", "0"), chunk_decay_every: _v("full-chunk-decay-every", "10"),
      optimizer_type: _v("full-optimizer", "adamw8bit"), scheduler: _v("full-scheduler", "cosine"),
      scheduler_formula: _v("full-scheduler-formula", ""),
      device: _v("full-device", "auto"), precision: _v("full-precision", "auto"),
      save_every: _v("full-save-every", "50"), log_every: _v("full-log-every", "10"),
      log_heavy_every: _v("full-log-heavy-every", "50"), save_best: _c("full-save-best"),
      save_best_after: _v("full-save-best-after", "200"), early_stop: _v("full-early-stop", "0"),
      resume_from: _v("full-resume-from", ""), strict_resume: $("full-strict-resume")?.checked ?? true,
      weight_decay: _v("full-weight-decay", "0.01"),
      max_grad_norm: _v("full-max-grad-norm", "1.0"), seed: _v("full-seed", "42"),
      dataset_repeats: _v("full-dataset-repeats", "1"), warmup_start_factor: _v("full-warmup-start-factor", "0.1"),
      cosine_eta_min_ratio: _v("full-cosine-eta-min", "0.01"), cosine_restarts_count: _v("full-cosine-restarts", "4"),
      ema_decay: _v("full-ema-decay", "0"), val_split: _v("full-val-split", "0"),
      adaptive_timestep_ratio: _v("full-adaptive-timestep", "0"),
      save_best_every_n_steps: _v("full-save-best-every-n-steps", "0"),
      num_workers: _v("full-num-workers", String(_platformDefaultWorkers())),
      prefetch_factor: _v("full-prefetch-factor", "2"),
      pin_memory: _c("full-pin-memory"),
      persistent_workers: _c("full-persistent-workers"),
      ignore_fisher_map: $("full-use-ppplus") ? !$("full-use-ppplus").checked : false,
    };

    const numWorkers = parseInt(cfg.num_workers, 10);
    const normalizedWorkers = Number.isFinite(numWorkers) ? numWorkers : _platformDefaultWorkers();
    cfg.num_workers = String(normalizedWorkers);
    if (normalizedWorkers <= 0) {
      cfg.prefetch_factor = "0";
      cfg.persistent_workers = false;
    } else if (cfg.prefetch_factor === "" || cfg.prefetch_factor == null) {
      cfg.prefetch_factor = "2";
    }

    const logDir = _v("full-log-dir", "");
    if (logDir) cfg.log_dir = logDir;

    const muVal = _v("full-timestep-mu", "-0.4");
    const sigmaVal = _v("full-timestep-sigma", "1.0");
    const muDefault = (($("full-timestep-mu")?.dataset?.default) || _d("full-timestep-mu") || "-0.4").toString().trim();
    const sigmaDefault = (($("full-timestep-sigma")?.dataset?.default) || _d("full-timestep-sigma") || "1.0").toString().trim();
    if (muVal !== muDefault) cfg.timestep_mu = muVal;
    if (sigmaVal !== sigmaDefault) cfg.timestep_sigma = sigmaVal;

    if (adapter === "lora" || adapter === "dora") {
      cfg.rank = _v("full-rank", "64"); cfg.alpha = _v("full-alpha", "128"); cfg.dropout = _v("full-dropout", "0.1");
      if (adapter === "dora") cfg.use_dora = true;
    } else if (adapter === "lokr") {
      cfg.lokr_linear_dim = _v("full-lokr-dim", "64"); cfg.lokr_linear_alpha = _v("full-lokr-alpha", "128");
      cfg.lokr_factor = _v("full-lokr-factor", "-1");
      cfg.lokr_decompose_both = _c("full-lokr-decompose-both") || false; cfg.lokr_use_tucker = _c("full-lokr-use-tucker") || false;
      cfg.lokr_use_scalar = _c("full-lokr-use-scalar") || false; cfg.lokr_weight_decompose = _c("full-lokr-weight-decompose") || false;
    } else if (adapter === "loha") {
      cfg.loha_linear_dim = _v("full-loha-dim", "64"); cfg.loha_linear_alpha = _v("full-loha-alpha", "128");
      cfg.loha_factor = _v("full-loha-factor", "-1");
      cfg.loha_use_tucker = _c("full-loha-use-tucker") || false; cfg.loha_use_scalar = _c("full-loha-use-scalar") || false;
    } else if (adapter === "oft") {
      cfg.oft_block_size = _v("full-oft-block-size", "64"); cfg.oft_coft = _c("full-oft-coft") || false;
      cfg.oft_eps = _v("full-oft-eps", "6e-5");
    }
    return cfg;
  }

  function init() {
    initCLIExport();
    initReviewSync();
    initCrossModeSync();
    initRunNameTracking();
    initPPPlusStatus();
  }

  return { init, gatherFullConfig, updateEzReview, _autoRunName: _autoRunName };
})();
