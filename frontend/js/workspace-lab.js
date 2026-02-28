/* Side-Step GUI — Lab Functions (Preprocess, PP++, AI Captions, Trigger Tag, Resume) */

const WorkspaceLab = (() => {
  "use strict";

  const $ = (id) => document.getElementById(id);
  const _e = (s) => typeof Validation !== 'undefined' ? Validation.esc(s) : String(s ?? '');

  /* ---- Shared task WS streaming ---- */
  const _tasks = {};  // { pp: { ws, taskId }, ppplus: ..., captions: ... }
  let _resumeBaseConfig = null;
  let _lastCanonicalAudio = "";

  function _canonicalAudioDir() {
    return ($("settings-audio-dir")?.value || "").trim();
  }

  function _syncAudioLibraryPath() {
    const canonical = _canonicalAudioDir();
    const labDs = $("lab-dataset-path");
    if (labDs) {
      labDs.value = canonical;
      labDs.readOnly = true;
      labDs.style.opacity = "0.75";
    }
    const browseBtn = document.querySelector('.browse-btn[data-target="lab-dataset-path"]');
    if (browseBtn) browseBtn.style.display = "none";
    if (canonical && typeof Dataset !== "undefined" && typeof Dataset.scan === "function") {
      Dataset.scan(canonical);
    }
  }

  function _syncPreprocessAudioPath() {
    const canonical = _canonicalAudioDir();
    const ppAudio = $("pp-audio-dir");
    const ppSource = $("pp-audio-source");
    if (ppAudio && (!ppAudio.value || ppAudio.value === _lastCanonicalAudio)) {
      ppAudio.value = canonical;
      ppAudio.dispatchEvent(new Event("change"));
      if (ppSource) ppSource.textContent = canonical ? "Using Settings \u203a Audio directory" : "";
    }
    _lastCanonicalAudio = canonical;
  }

  function _syncCanonicalAudioPaths() {
    _syncAudioLibraryPath();
    _syncPreprocessAudioPath();
  }

  function _timestamp() {
    const d = new Date();
    return d.getFullYear().toString() +
      String(d.getMonth() + 1).padStart(2, "0") +
      String(d.getDate()).padStart(2, "0") + "_" +
      String(d.getHours()).padStart(2, "0") +
      String(d.getMinutes()).padStart(2, "0");
  }

  function _normalizeResumeConfig(raw) {
    const cfg = { ...(raw || {}) };
    if (cfg.learning_rate != null && cfg.lr == null) cfg.lr = cfg.learning_rate;
    if (cfg.gradient_accumulation_steps != null && cfg.grad_accum == null) cfg.grad_accum = cfg.gradient_accumulation_steps;
    if (cfg.max_epochs != null && cfg.epochs == null) cfg.epochs = cfg.max_epochs;
    if (cfg.save_every_n_epochs != null && cfg.save_every == null) cfg.save_every = cfg.save_every_n_epochs;
    if (cfg.log_every_n_steps != null && cfg.log_every == null) cfg.log_every = cfg.log_every_n_steps;
    if (Array.isArray(cfg.target_modules) && cfg.projections == null) cfg.projections = cfg.target_modules.join(" ");
    if (Array.isArray(cfg.self_target_modules) && cfg.self_projections == null) cfg.self_projections = cfg.self_target_modules.join(" ");
    if (Array.isArray(cfg.cross_target_modules) && cfg.cross_projections == null) cfg.cross_projections = cfg.cross_target_modules.join(" ");
    return cfg;
  }

  function _streamTask(taskId, key, opts) {
    // opts: { barId, labelId, pctId, logId, onDone(msg), onProgress(msg) }
    if (_tasks[key]) { try { _tasks[key].ws.close(); } catch {} }
    const ws = API.connectTaskWS(taskId, (msg) => {
      if (msg.type === "progress") {
        const pct = msg.percent || 0;
        const bar = $(opts.barId); if (bar) bar.style.width = pct + "%";
        const lbl = $(opts.labelId); if (lbl) lbl.textContent = msg.label || `Processing ${msg.current || 0} / ${msg.total || 0}`;
        const pctEl = $(opts.pctId); if (pctEl) pctEl.textContent = pct + "%";
        if (opts.logId && msg.log) {
          const log = $(opts.logId);
          if (log) { const d = document.createElement("div"); d.className = "log-entry log-entry--step"; d.textContent = "  " + msg.log; log.appendChild(d); if (typeof autoScrollLog === "function") autoScrollLog(log); else log.scrollTop = log.scrollHeight; }
        }
        if (opts.onProgress) opts.onProgress(msg);
      } else if (msg.type === "done" || msg.type === "error" || msg.type === "cancelled") {
        if (opts.onDone) opts.onDone(msg);
        try { ws.close(); } catch {}
        delete _tasks[key];
      }
    });
    _tasks[key] = { ws, taskId };
  }

  function _stopTask(key) {
    const entry = _tasks[key];
    if (!entry) return;
    if (entry.cancelRequested) return;
    entry.cancelRequested = true;
    // Keep WS open until terminal done/error/cancelled event arrives.
    // This guarantees UI reset handlers run with backend-confirmed state.
    if (entry.taskId) {
      API.stopTask(entry.taskId).catch(() => {
        entry.cancelRequested = false;
      });
    }
  }

  /* ---- Preprocess queue ---- */
  const _ppQueue = [];      // [{audioDir, outputDir, status:"pending"|"running"|"done"|"failed"}]
  let _ppQueueActive = false;

  // _pathBasename and _joinPath provided by fallback.js globals

  function _renderQueue() {
    const panel = $("pp-queue-panel");
    const list = $("pp-queue-list");
    if (!panel || !list) return;
    if (_ppQueue.length === 0) { panel.style.display = "none"; return; }
    panel.style.display = "block";
    list.innerHTML = _ppQueue.map((item, i) => {
      const icon = item.status === "done" ? '<span class="u-text-success">[ok]</span>'
        : item.status === "failed" ? '<span class="u-text-error">[x]</span>'
        : item.status === "running" ? '<span class="u-text-primary">[...]</span>'
        : '<span class="u-text-muted">[--]</span>';
      const name = _pathBasename(item.audioDir);
      return `<div style="padding:2px 0;">${icon} ${_e(name)} <span class="u-text-muted">${_e(item.status)}</span></div>`;
    }).join("");
  }

  function _cancelQueue() {
    _ppQueue.forEach(item => { if (item.status === "pending") item.status = "cancelled"; });
    _ppQueueActive = false;
    _stopTask("pp");
    _renderQueue();
    showToast("Preprocessing queue cancelled", "warn");
  }

  async function _runNextInQueue() {
    const next = _ppQueue.find(item => item.status === "pending");
    if (!next) {
      _ppQueueActive = false;
      const done = _ppQueue.filter(i => i.status === "done").length;
      const failed = _ppQueue.filter(i => i.status === "failed").length;
      showToast(`Queue complete: ${done} done, ${failed} failed`, done > 0 ? "ok" : "warn");
      _renderQueue();
      return;
    }

    next.status = "running";
    _ppQueueActive = true;
    _renderQueue();

    const ppAudio = $("pp-audio-dir");
    const ppOut = $("pp-output-dir");
    if (ppAudio) { ppAudio.value = next.audioDir; ppAudio.dispatchEvent(new Event("change")); }
    if (ppOut && ppOut.readOnly) ppOut.value = next.outputDir;

    const normMethod = $("pp-normalize")?.value || "none";
    const config = {
      checkpoint_dir: $("settings-checkpoint-dir")?.value, model_variant: $("pp-model-variant")?.value,
      audio_dir: next.audioDir, output_dir: next.outputDir,
      normalize: normMethod, trigger_tag: $("pp-trigger-tag")?.value, tag_position: $("pp-tag-position")?.value,
      target_db: normMethod === "peak" ? parseFloat($("pp-peak-target")?.value) || -1.0 : -1.0,
      target_lufs: normMethod === "lufs" ? parseFloat($("pp-lufs-target")?.value) || -14.0 : -14.0,
    };

    const prog = $("pp-progress-panel"); if (prog) prog.style.display = "block";
    const log = $("pp-log"); if (log) log.innerHTML = "";

    try {
      const result = await API.runPreprocess(config);
      if (result.task_id) {
        _streamTask(result.task_id, "pp", {
          barId: "pp-progress-bar", labelId: "pp-progress-label", pctId: "pp-progress-pct", logId: "pp-log",
          onDone: (msg) => {
            if (prog) prog.style.display = "none";
            if (msg.type === "cancelled") {
              next.status = "failed";
              _ppQueueActive = false;
              _ppQueue.forEach(item => { if (item.status === "pending") item.status = "cancelled"; });
              _renderQueue();
              showToast("Preprocessing cancelled", "warn");
              return;
            }
            next.status = msg.type === "error" ? "failed" : "done";
            _renderQueue();
            _runNextInQueue();
          },
        });
      } else {
        next.status = "done";
        if (prog) prog.style.display = "none";
        _renderQueue();
        _runNextInQueue();
      }
    } catch (e) {
      next.status = "failed";
      if (prog) prog.style.display = "none";
      showToast("Preprocessing failed: " + e.message, "error");
      _renderQueue();
      _runNextInQueue();
    }
  }

  function queuePreprocess(folderPaths) {
    const tensorsDir = $("settings-tensors-dir")?.value || "./preprocessed_tensors";
    folderPaths.forEach(audioDir => {
      _ppQueue.push({
        audioDir,
        outputDir: _joinPath(tensorsDir, _pathBasename(audioDir) || "tensors"),
        status: "pending",
      });
    });
    _renderQueue();
    if (!_ppQueueActive) _runNextInQueue();
  }

  /* ---- Preprocess output override ---- */
  function _deriveOutputDir() {
    const audioDir = $("pp-audio-dir")?.value || "";
    const tensorsDir = $("settings-tensors-dir")?.value || "./preprocessed_tensors";
    const folderName = _pathBasename(audioDir) || "tensors";
    return _joinPath(tensorsDir, folderName);
  }

  function initOutputOverride() {
    const input = $("pp-output-dir");
    const editLink = $("pp-output-edit");
    const resetLink = $("pp-output-reset");

    editLink?.addEventListener("click", (e) => {
      e.preventDefault();
      if (input) { input.readOnly = false; input.style.opacity = "1"; }
      if (editLink) editLink.style.display = "none";
      if (resetLink) resetLink.style.display = "";
    });

    resetLink?.addEventListener("click", (e) => {
      e.preventDefault();
      if (input) { input.readOnly = true; input.style.opacity = "0.7"; input.value = _deriveOutputDir(); }
      if (resetLink) resetLink.style.display = "none";
      if (editLink) editLink.style.display = "";
    });

    $("pp-audio-dir")?.addEventListener("change", () => {
      if (input && input.readOnly) input.value = _deriveOutputDir();
    });
  }

  /* ---- Normalization target visibility ---- */
  function initNormalizeTargets() {
    const sel = $("pp-normalize");
    const peakGrp = $("pp-peak-target-group");
    const lufsGrp = $("pp-lufs-target-group");
    if (!sel) return;
    const update = () => {
      const v = sel.value;
      if (peakGrp) peakGrp.style.display = v === "peak" ? "" : "none";
      if (lufsGrp) lufsGrp.style.display = v === "lufs" ? "" : "none";
    };
    sel.addEventListener("change", update);
    update();
  }

  function _fmtSec(s) {
    const m = Math.floor(s / 60);
    const sec = s % 60;
    return m > 0 ? `${m}m ${String(sec).padStart(2, "0")}s` : `${sec}s`;
  }

  /* ---- Preprocess Tab ---- */
  function initPreprocess() {
    // Canonical source audio path comes from settings-audio-dir.
    _syncPreprocessAudioPath();
    $("btn-scan-audio")?.addEventListener("click", async () => {
      const path = $("pp-audio-dir")?.value;
      if (!path) { showToast("Enter an audio folder path", "warn"); return; }
      const result = await API.scanAudioFolder(path);
      const detect = $("pp-audio-detect");
      if (detect) {
        // Genre ratio summary
        const genres = {};
        result.files.forEach(f => { const g = f.genre || '--'; genres[g] = (genres[g] || 0) + 1; });
        const genreStr = Object.entries(genres).sort((a,b) => b[1] - a[1]).map(([g,c]) => `${g}: ${c}`).join(', ');
        detect.innerHTML = `[ok] ${result.files.length} audio files | Total: ${_fmtSec(result.total_duration)} | Longest: ${_fmtSec(result.longest)}` +
          (genreStr ? `<br><span class="u-text-muted">Genres: ${_e(genreStr)}</span>` : '');
        detect.className = "detect detect--ok";
      }
      const scanPanel = $("pp-duration-scan"), scanLog = $("pp-duration-log"), scanSummary = $("pp-duration-summary");
      if (scanPanel) scanPanel.style.display = "block";
      if (scanLog) { scanLog.innerHTML = ""; result.files.forEach(f => { const d = document.createElement("div"); d.className = "log-entry log-entry--info"; d.textContent = `  ${f.name.padEnd(40)} ${_fmtSec(f.duration)}`; scanLog.appendChild(d); }); }
      if (scanSummary) scanSummary.textContent = `${result.files.length} files, longest ${_fmtSec(result.longest)}, total ${_fmtSec(result.total_duration)}`;
      const outDir = $("pp-output-dir");
      if (outDir && !outDir.value) outDir.value = _joinPath($("settings-tensors-dir")?.value || "./preprocessed_tensors", _pathBasename(path) || "tensors");
      const startBtn = $("btn-start-preprocess"); if (startBtn) startBtn.disabled = false;
      showToast("Audio scan complete: " + result.files.length + " files", "ok");
    });

    $("btn-start-preprocess")?.addEventListener("click", async () => {
      const normMethod = $("pp-normalize")?.value || "none";
      const config = {
        checkpoint_dir: $("settings-checkpoint-dir")?.value, model_variant: $("pp-model-variant")?.value,
        audio_dir: $("pp-audio-dir")?.value, output_dir: $("pp-output-dir")?.value,
        normalize: normMethod, trigger_tag: $("pp-trigger-tag")?.value, tag_position: $("pp-tag-position")?.value,
        target_db: normMethod === "peak" ? parseFloat($("pp-peak-target")?.value) || -1.0 : -1.0,
        target_lufs: normMethod === "lufs" ? parseFloat($("pp-lufs-target")?.value) || -14.0 : -14.0,
        genre_ratio: parseInt($("pp-genre-ratio")?.value) || 0,
      };
      const prog = $("pp-progress-panel"); if (prog) prog.style.display = "block";
      const log = $("pp-log"); if (log) log.innerHTML = "";
      const _ppDone = (r, err) => {
        if (prog) prog.style.display = "none";
        const cp = $("pp-complete-panel"); if (cp) cp.style.display = "block";
        const s = $("pp-complete-summary");
        const processed = r.processed || 0, failed = r.failed || 0, total = r.total || 0;
        const statusClass = err ? "u-text-error" : failed > 0 ? "u-text-warning" : "u-text-success";
        const statusIcon = err ? "[x]" : failed > 0 ? "[!]" : "[ok]";
        if (s) s.innerHTML = `<div class="${statusClass}">${statusIcon} ${processed}/${total} preprocessed${failed > 0 ? `, ${failed} failed` : ""}</div><div class="u-text-muted">Output: ${_e(r.output_dir || config.output_dir)}</div>`;
        if (err) showToast("Preprocessing failed: " + err, "error");
        else if (failed > 0) showToast(`Preprocessing done with ${failed} failure${failed > 1 ? "s" : ""} (${processed}/${total})`, "warn");
        else showToast("Preprocessing complete!", "ok");
      };
      const result = await API.runPreprocess(config);
      if (result.task_id) {
        _streamTask(result.task_id, "pp", {
          barId: "pp-progress-bar", labelId: "pp-progress-label", pctId: "pp-progress-pct", logId: "pp-log",
          onDone: (msg) => {
            if (msg.type === "cancelled") { if (prog) prog.style.display = "none"; showToast("Preprocessing cancelled", "warn"); return; }
            const r = msg.result || { processed: msg.processed, failed: msg.failed, total: msg.total, output_dir: msg.output_dir } || result;
            _ppDone(r, msg.type === "error" ? (msg.msg || msg.error || "unknown") : null);
          },
        });
      } else { _ppDone(result, null); }
    });

    $("btn-stop-preprocess")?.addEventListener("click", () => { _stopTask("pp"); showToast("Preprocessing cancellation requested", "info"); });

    $("btn-pp-chain-train")?.addEventListener("click", () => {
      const outDir = $("pp-output-dir")?.value;
      const fullDs = $("full-dataset-dir");
      if (fullDs && outDir) fullDs.value = outDir;
      switchMode("full");
      showToast("Dataset dir set to preprocessed tensors", "ok");
    });
    $("btn-pp-chain-ppplus")?.addEventListener("click", () => {
      const outDir = $("pp-output-dir")?.value;
      const pppDs = $("ppplus-dataset-dir");
      if (pppDs && outDir) pppDs.value = outDir;
      document.querySelectorAll(".lab-nav__item").forEach(b => b.classList.remove("active"));
      document.querySelectorAll(".lab-panel").forEach(p => p.classList.remove("active"));
      document.querySelector('[data-lab="ppplus"]')?.classList.add("active");
      $("lab-ppplus")?.classList.add("active");
    });
  }

  /* ---- PP++ Tab ---- */
  function _showPPResults(result) {
    const results = $("ppplus-results-panel");
    if (results) results.style.display = "block";
    const summary = $("ppplus-results-summary");
    const modules = Array.isArray(result.modules) ? result.modules : [];
    const budget = result.rank_budget || {};
    const minRank = budget.min ?? "?";
    const maxRank = budget.max ?? "?";
    const fallbackSummary = modules.length
      ? `PP++ map generated: ${modules.length} modules (ranks ${minRank}-${maxRank})`
      : "PP++ map generated";
    if (summary) summary.textContent = result.summary || fallbackSummary;
    const tbody = $("ppplus-results-tbody");
    if (tbody && modules.length) {
      tbody.innerHTML = "";
      modules.forEach(m => {
        const fisher = Number(m.fisher ?? m.fisher_score ?? 0);
        const spectral = Number(m.spectral ?? m.effective_rank ?? 0);
        const rank = m.rank ?? m.assigned_rank ?? "-";
        const tr = document.createElement("tr");
        tr.innerHTML = `<td class="u-meta-muted-xs">${_e(m.name)}</td><td>${fisher.toFixed(3)}</td><td>${spectral.toFixed(2)}</td><td class="u-text-bold u-text-primary">${_e(rank)}</td>`;
        tbody.appendChild(tr);
      });
    }
  }

  function initPPPlus() {
    $("btn-run-ppplus")?.addEventListener("click", async () => {
      const config = {
        dataset_dir: $("ppplus-dataset-dir")?.value, checkpoint_dir: $("settings-checkpoint-dir")?.value,
        model_variant: $("ppplus-model-variant")?.value, timestep_focus: $("ppplus-timestep-focus")?.value,
        base_rank: parseInt($("ppplus-base-rank")?.value) || 64, rank_min: parseInt($("ppplus-rank-min")?.value) || 16,
        rank_max: parseInt($("ppplus-rank-max")?.value) || 128,
      };
      if (!config.dataset_dir) { showToast("Enter a dataset folder", "warn"); return; }
      const prog = $("ppplus-progress-panel"); if (prog) prog.style.display = "block";
      const log = $("ppplus-log"); if (log) log.innerHTML = "";
      const stopBtn = $("btn-stop-ppplus"), runBtn = $("btn-run-ppplus");
      if (stopBtn) stopBtn.style.display = "inline-block"; if (runBtn) runBtn.style.display = "none";
      const _ppDone = (r, msg) => {
        if (prog) prog.style.display = "none"; if (stopBtn) stopBtn.style.display = "none"; if (runBtn) runBtn.style.display = "inline-block";
        if (msg?.type === "cancelled") { showToast("PP++ cancelled", "warn"); return; }
        if (msg?.type === "error") { showToast("PP++ failed: " + (msg.msg || msg.error || "unknown"), "error"); return; }
        _showPPResults(r); showToast("PP++ analysis complete!", "ok");
      };
      const result = await API.runFisherAnalysis(config);
      if (result.task_id) {
        _streamTask(result.task_id, "ppplus", {
          barId: "ppplus-progress-bar", labelId: "ppplus-progress-label", pctId: "ppplus-progress-pct", logId: "ppplus-log",
          onDone: (msg) => _ppDone(msg.result || result, msg),
        });
      } else { _ppDone(result, null); }
    });
    $("btn-stop-ppplus")?.addEventListener("click", () => { _stopTask("ppplus"); showToast("PP++ cancellation requested", "info"); });

    $("btn-ppplus-chain-train")?.addEventListener("click", () => {
      const ds = $("ppplus-dataset-dir")?.value;
      const fullDs = $("full-dataset-dir");
      if (fullDs && ds) fullDs.value = ds;
      switchMode("full");
      showToast("Dataset dir set — PP++ map will be auto-detected", "ok");
    });
  }

  /* ---- Resume Training ---- */
  function initResume() {
    $("btn-resume-selected")?.addEventListener("click", async () => {
      const rows = [...document.querySelectorAll("#history-tbody tr.selected")]
        .filter((r) => r.dataset.detected !== "1");
      if (rows.length === 0) { showToast("Select a run to resume (click or Ctrl+click)", "warn"); return; }
      const idx = parseInt(rows[0].dataset.idx, 10);
      if (!Number.isFinite(idx)) { showToast("Select a valid run to resume", "warn"); return; }
      const runs = await API.fetchHistory();
      if (!runs[idx]) return;
      _openResumeModal(runs[idx]);
    });

    $("resume-modal-close")?.addEventListener("click", _closeResumeModal);
    $("btn-resume-cancel")?.addEventListener("click", _closeResumeModal);

    $("resume-edit-mode")?.addEventListener("change", () => {
      const mode = $("resume-edit-mode").value;
      $("resume-safe-fields").style.display = (mode === "safe" || mode === "all") ? "block" : "none";
      $("resume-danger-fields").style.display = mode === "all" ? "block" : "none";
    });

    $("btn-resume-start")?.addEventListener("click", async () => {
      const runName = $("resume-run-name")?.textContent;
      const ckpt = $("resume-checkpoint-select")?.value;
      if (!runName || !ckpt) {
        showToast("Select a checkpoint to resume", "warn");
        return;
      }

      let baseCfg = _resumeBaseConfig;
      if (!baseCfg) {
        baseCfg = await API.fetchRunConfig(runName);
      }
      if (!baseCfg || baseCfg.error) {
        showToast("Could not load run configuration", "error");
        return;
      }

      const cfg = _normalizeResumeConfig(baseCfg);
      const extraEpochs = Math.max(0, parseInt($("resume-extra-epochs")?.value, 10) || 0);
      const baseEpochs = parseInt(cfg.epochs, 10) || 0;
      const runNameResumed = runName + "_resumed_" + _timestamp();

      cfg.run_name = runNameResumed;
      cfg.resume_from = ckpt;
      cfg.strict_resume = true;
      cfg.epochs = String(baseEpochs + extraEpochs);

      const lr = $("resume-lr")?.value?.trim();
      if (lr) cfg.lr = lr;
      const saveEvery = parseInt($("resume-save-every")?.value, 10);
      if (!Number.isNaN(saveEvery) && saveEvery > 0) cfg.save_every = String(saveEvery);

      if (!cfg.adapter_type) cfg.adapter_type = "lora";
      cfg.output_dir = _joinPath(
        $("settings-adapters-dir")?.value || "./trained_adapters",
        cfg.adapter_type,
        runNameResumed,
      );

      if (!cfg.checkpoint_dir || !cfg.dataset_dir) {
        showToast("Resume config is missing checkpoint_dir or dataset_dir", "error");
        return;
      }

      _closeResumeModal();
      Training.start(cfg);
    });
  }

  async function _openResumeModal(run) {
    $("resume-modal")?.classList.add("open");
    $("resume-run-name").textContent = run.run_name;
    const cfg = await API.fetchRunConfig(run.run_name);
    if (cfg && !cfg.error) {
      _resumeBaseConfig = cfg;
      const adapter = cfg.adapter_type || cfg.adapter || "--";
      const model = cfg.model_variant || cfg.model || "--";
      const rank = cfg.rank ?? cfg.lokr_linear_dim ?? cfg.loha_linear_dim ?? "--";
      const lr = cfg.learning_rate ?? cfg.lr ?? "--";
      const batch = cfg.batch_size ?? cfg.batch ?? "--";
      const accum = cfg.gradient_accumulation_steps ?? cfg.grad_accum ?? "--";
      const optimizer = cfg.optimizer_type || cfg.optimizer || "--";
      [["resume-cfg-adapter", adapter], ["resume-cfg-model", model], ["resume-cfg-rank", rank],
        ["resume-cfg-lr", lr], ["resume-cfg-batch", batch + " \u00d7 " + accum],
        ["resume-cfg-optimizer", optimizer]].forEach(([id, v]) => { const e = $(id); if (e) e.textContent = v; });
      if ($("resume-lr") && lr !== "--") $("resume-lr").value = String(lr);
      const se = cfg.save_every_n_epochs ?? cfg.save_every;
      if ($("resume-save-every") && se != null) $("resume-save-every").value = String(se);
    } else {
      _resumeBaseConfig = null;
    }
    let ckpts;
    try { ckpts = await API.scanCheckpoints(run.run_name); }
    catch (e) {
      const sel = $("resume-checkpoint-select");
      if (sel) { sel.innerHTML = ""; const opt = document.createElement("option"); opt.value = ""; opt.textContent = "Failed to scan checkpoints"; sel.appendChild(opt); }
      showToast("Checkpoint scan failed: " + e.message, "error");
      return;
    }
    const sel = $("resume-checkpoint-select");
    if (sel) {
      sel.innerHTML = "";
      (ckpts.checkpoints || []).forEach(c => {
        const opt = document.createElement("option");
        const loss = typeof c.loss === "number" ? ` \u2014 loss ${c.loss.toFixed(4)}` : "";
        const epoch = c.epoch ? ` \u2014 epoch ${c.epoch}` : "";
        opt.value = c.path;
        opt.textContent = `${c.name}${loss || epoch}`;
        sel.appendChild(opt);
      });
      if (!sel.options.length) {
        const opt = document.createElement("option");
        opt.value = "";
        opt.textContent = "No checkpoints found";
        sel.appendChild(opt);
      }
    }
  }

  function _closeResumeModal() {
    $("resume-modal")?.classList.remove("open");
    _resumeBaseConfig = null;
  }

  /* ---- AI Captions ---- */
  function initAICaptions() {
    $("caption-provider")?.addEventListener("change", () => {
      const prov = $("caption-provider").value;
      $("caption-gemini-settings").style.display = prov === "gemini" ? "block" : "none";
      $("caption-openai-settings").style.display = prov === "openai" ? "block" : "none";
    });

    $("caption-open-settings")?.addEventListener("click", (e) => {
      e.preventDefault();
      $("settings-panel")?.classList.add("open");
    });

    $("btn-gen-captions")?.addEventListener("click", () => {
      // Switch to the Captions lab tab (not just scroll within current panel)
      document.querySelectorAll(".lab-nav__item").forEach(b => b.classList.remove("active"));
      document.querySelectorAll(".lab-panel").forEach(p => p.classList.remove("active"));
      document.querySelector('[data-lab="captions"]')?.classList.add("active");
      $("lab-captions")?.classList.add("active");
    });

    $("btn-run-captions")?.addEventListener("click", async () => {
      const provider = $("caption-provider")?.value;
      if (provider === "skip") { showToast("Caption provider set to skip", "warn"); return; }
      const config = {
        provider: provider,
        overwrite: $("caption-overwrite")?.value,
        gemini_key: $("settings-gemini-key")?.value,
        gemini_model: $("caption-gemini-model")?.value,
        openai_key: $("settings-openai-key")?.value,
        openai_model: $("caption-openai-model")?.value,
        openai_base: $("caption-openai-base")?.value || $("settings-openai-base")?.value,
        genius_token: $("settings-genius-token")?.value,
        default_artist: $("caption-default-artist")?.value,
        dataset_dir: $("lab-dataset-path")?.value,
      };
      if (!config.dataset_dir) { showToast("Select an audio folder first", "warn"); return; }

      $("caption-batch-progress").style.display = "block";
      $("btn-run-captions").style.display = "none";
      $("btn-stop-captions").style.display = "inline-block";
      const log = $("caption-log"); if (log) log.innerHTML = "";
      let written = 0, skipped = 0, failed = 0;

      const _finish = (msg) => {
        $("btn-run-captions").style.display = "inline-block";
        $("btn-stop-captions").style.display = "none";
        if (msg && msg.type === "cancelled") { showToast("AI Captions cancelled", "warn"); return; }
        const payload = msg?.result || msg || {};
        if (payload.written != null) written = payload.written;
        if (payload.skipped != null) skipped = payload.skipped;
        if (payload.failed != null) failed = payload.failed;
        showToast(`AI Captions: ${written} written, ${skipped} skipped, ${failed} failed`, written > 0 ? "ok" : "warn");
        if (typeof Dataset !== "undefined") Dataset.scan($("lab-dataset-path")?.value);
      };

      const result = await API.runAICaptions(config);
      if (result.error) {
        _finish({ type: "error", failed: failed + 1 });
        showToast("AI Captions failed to start: " + result.error, "error");
        return;
      }
      const taskId = result.task_id;
      if (taskId) {
        _streamTask(taskId, "captions", {
          barId: "caption-progress-bar", labelId: "caption-progress-label", pctId: "caption-progress-pct", logId: "caption-log",
          onProgress: (msg) => {
            if (msg.written != null) { written = msg.written; skipped = msg.skipped || 0; failed = msg.failed || 0; }
            const ws = $("caption-stat-written"); if (ws) ws.textContent = written + " written";
            const ss = $("caption-stat-skipped"); if (ss) ss.textContent = skipped + " skipped";
            const fs = $("caption-stat-failed"); if (fs) fs.textContent = failed + " failed";
          },
          onDone: _finish,
        });
      } else {
        _finish(result);
      }
    });

    $("btn-stop-captions")?.addEventListener("click", async () => {
      _stopTask("captions");
      showToast("Caption generation cancellation requested", "info");
    });
  }

  /* ---- Trigger Tag Bulk Modal ---- */
  function initTriggerTagBulk() {
    $("btn-add-trigger")?.addEventListener("click", () => {
      $("trigger-tag-modal")?.classList.add("open");
    });
    $("trigger-tag-close")?.addEventListener("click", () => {
      $("trigger-tag-modal")?.classList.remove("open");
    });
    $("btn-bulk-trigger-cancel")?.addEventListener("click", () => {
      $("trigger-tag-modal")?.classList.remove("open");
    });

    $("bulk-trigger-tag")?.addEventListener("input", _updateTriggerPreview);
    $("bulk-trigger-position")?.addEventListener("change", _updateTriggerPreview);

    $("btn-bulk-trigger-apply")?.addEventListener("click", async () => {
      const tag = $("bulk-trigger-tag")?.value;
      const position = $("bulk-trigger-position")?.value;
      const dsPath = $("lab-dataset-path")?.value;
      if (!tag) { showToast("Enter a trigger tag", "warn"); return; }
      const result = await API.bulkWriteTriggerTag(dsPath, tag, position);
      $("trigger-tag-modal")?.classList.remove("open");
      showToast(`Trigger tag written to ${result.updated} sidecars`, "ok");
      if (typeof Dataset !== "undefined") Dataset.scan(dsPath);
    });
  }

  function _updateTriggerPreview() {
    const tag = $("bulk-trigger-tag")?.value || "[tag]", pos = $("bulk-trigger-position")?.value || "prepend";
    const p = $("bulk-trigger-preview"); if (!p) return;
    p.innerHTML = "Preview: <code>" + (pos === "prepend" ? _e(tag) + " original caption text..." : pos === "append" ? "original caption text... " + _e(tag) : _e(tag)) + "</code>";
  }

  function init() {
    _syncCanonicalAudioPaths();
    document.addEventListener("sidestep:settings-saved", () => {
      _syncCanonicalAudioPaths();
    });
    $("btn-cancel-pp-queue")?.addEventListener("click", _cancelQueue);
    [initOutputOverride, initNormalizeTargets, initPreprocess, initPPPlus, initResume, initAICaptions, initTriggerTagBulk].forEach(fn => fn());
  }

  return { init, queuePreprocess };
})();
