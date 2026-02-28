/* Side-Step GUI — Training Controller (WebSocket + Monitor DOM) */

const Training = (() => {

  let _running = false, _finalized = false;
  let _ws = null, _gpuWs = null;
  let _step = 0, _epoch = 0, _maxEpochs = 100, _stepsPerEpoch = 47, _stepInEpoch = 0;
  let _loss = 0, _bestLoss = Infinity, _bestEpoch = 0, _lr = 0;
  let _lossHistory = [], _lrHistory = [];
  let _epochLossHistory = [], _epochLrHistory = [];
  let _epochLossAccum = 0, _epochStepCount = 0;
  let _startTime = 0, _epochStartTime = 0, _lastEpochDuration = 0;
  let _config = {}, _viewXMin = null, _viewXMax = null;
  let _smoothingWeight = 0.6, _userZoomed = false;
  let _taskId = null, _domTimer = null;
  let _stopRequested = false, _lastFailureMsg = "";

  const $ = (id) => document.getElementById(id);

  function _esc(v) {
    return String(v ?? "")
      .replaceAll("&", "&amp;")
      .replaceAll("<", "&lt;")
      .replaceAll(">", "&gt;")
      .replaceAll('"', "&quot;")
      .replaceAll("'", "&#39;");
  }

  function _fmtPath(path) {
    const p = String(path || "").trim();
    if (!p) return "--";
    return p;
  }

  async function _copyText(text) {
    try {
      if (navigator?.clipboard?.writeText) {
        await navigator.clipboard.writeText(text);
        return true;
      }
    } catch {}
    try {
      const ta = document.createElement("textarea");
      ta.value = text;
      ta.setAttribute("readonly", "true");
      ta.style.position = "fixed";
      ta.style.top = "-10000px";
      ta.style.left = "-10000px";
      document.body.appendChild(ta);
      ta.select();
      const ok = document.execCommand("copy");
      document.body.removeChild(ta);
      return !!ok;
    } catch {
      return false;
    }
  }

  function _buildRunSummaryText() {
    const c = _config || {};
    const lines = [
      `Run: ${c.run_name || "training"}`,
      `Adapter: ${c.adapter_type || "lora"}`,
      `Model: ${c.model_variant || "turbo"}`,
      `Dataset: ${c.dataset_dir || "--"}`,
      `Output: ${c.output_dir || "--"}`,
      `Resume: ${c.resume_from || "fresh start"}`,
      `Optimizer: ${c.optimizer_type || "adamw"}`,
      `LR: ${c.lr || "1e-4"}`,
      `Batch: ${(c.batch_size || 1)} × ${(c.grad_accum || 4)}`,
      `Epochs: ${c.epochs || "--"}`,
      `Checkpointing: ${c.gradient_checkpointing_ratio || "full"}`,
    ];
    return lines.join("\n");
  }

  function _showTerminalToast(outcome) {
    if (typeof showToast !== 'function') return;
    if (outcome === 'complete') {
      const best = _bestLoss < Infinity ? _bestLoss.toFixed(4) : '--';
      showToast(`Training complete · best ${best} @ epoch ${_bestEpoch || '--'}`, 'ok');
      return;
    }
    if (outcome === 'stopped') {
      showToast('Training stopped. Resume from History to continue.', 'warn');
      return;
    }
    const reason = _lastFailureMsg ? `: ${_lastFailureMsg}` : '';
    const isOOM = /out of memory/i.test(_lastFailureMsg);
    showToast(isOOM ? `Out of memory${reason}` : `Training failed${reason}`, 'error');
  }

  async function _openOutputDirFromMonitor() {
    const dir = $('monitor-output-dir')?.textContent?.replace('Output: ', '').trim() || '';
    if (!dir) {
      if (typeof showToast === 'function') showToast('No output directory available', 'warn');
      return;
    }
    const result = await API.openFolder(dir);
    if (typeof showToast === 'function') {
      showToast(
        result.ok ? 'Output directory opened' : 'Failed to open output directory: ' + (result.error || dir),
        result.ok ? 'ok' : 'error'
      );
    }
  }

  // Cache for diff-based updates
  const _domCache = new Map();
  
  function _updateDOM() {
    const epochPct = _maxEpochs > 0 ? (_epoch / _maxEpochs) * 100 : 0;
    const stepPct = _stepsPerEpoch > 0 ? (_stepInEpoch / _stepsPerEpoch) * 100 : 0;

    // Progress bars (always update as they're style changes)
    const epochBar = $('monitor-epoch-bar');
    const stepBar = $('monitor-step-bar');
    if (epochBar) epochBar.style.width = epochPct + '%';
    if (stepBar) stepBar.style.width = stepPct + '%';

    // Helper for diff-based text updates
    const setText = (id, val) => {
      const key = 'text_' + id;
      if (_domCache.get(key) !== val) {
        const e = $(id);
        if (e) { e.textContent = val; _domCache.set(key, val); }
      }
    };

    // Labels (only update if value changed)
    setText('monitor-epoch-label', `Epoch ${_epoch} / ${_maxEpochs}`);
    setText('monitor-epoch-pct', Math.round(epochPct) + '%');
    setText('monitor-step-label', `Step ${_stepInEpoch} / ${_stepsPerEpoch}`);
    setText('monitor-step-pct', Math.round(stepPct) + '%');
    setText('monitor-loss', _loss.toFixed(4));
    setText('monitor-best-loss', _bestLoss < Infinity ? _bestLoss.toFixed(4) : '--');
    setText('monitor-best-epoch', _bestEpoch > 0 ? _bestEpoch.toString() : '--');
    setText('monitor-lr', _lr > 0 ? _lr.toExponential(2) : '--');

    // MA5
    const ma5 = _calcMA(5);
    setText('monitor-ma5', ma5 !== null ? ma5.toFixed(4) : '--');

    // Timing
    const elapsed = (Date.now() - _startTime) / 1000;
    setText('monitor-elapsed', _fmtDuration(elapsed));
    setText('monitor-step-time', _step > 0 ? (elapsed / _step).toFixed(2) + 's' : '--');
    setText('monitor-epoch-time', _lastEpochDuration > 0 ? _lastEpochDuration.toFixed(1) + 's' : '--');
    setText('monitor-sps', _step > 0 ? ((_step / elapsed) || 0).toFixed(1) + ' steps/s' : '--');

    // ETA
    const remainingEpochs = _maxEpochs - _epoch;
    let eta = '--';
    if (_lastEpochDuration > 0 && remainingEpochs > 0) {
      eta = _fmtDuration(_lastEpochDuration * remainingEpochs);
    }
    setText('monitor-eta', 'ETA: ' + eta);
    setText('monitor-eta-right', eta);

    // Loss chart SVG
    _updateLossChart();
  }

  function _calcMA(window) {
    if (_lossHistory.length < window) return null;
    const slice = _lossHistory.slice(-window);
    return slice.reduce((a, b) => a + b, 0) / slice.length;
  }

  function _fmtDuration(seconds) {
    const s = Math.floor(seconds);
    const h = Math.floor(s / 3600);
    const m = Math.floor((s % 3600) / 60);
    const sec = s % 60;
    if (h > 0) return `${h}h ${String(m).padStart(2, '0')}m ${String(sec).padStart(2, '0')}s`;
    if (m > 0) return `${m}m ${String(sec).padStart(2, '0')}s`;
    return `${sec}s`;
  }

  // Chart display mode: 'step' or 'epoch'
  let _chartMode = 'epoch';

  function _getChartData() {
    if (_chartMode === 'epoch') {
      const loss = _epochLossHistory.slice();
      const lr = _epochLrHistory.slice();
      if (_running && _epochStepCount > 0) {
        loss.push(_epochLossAccum / _epochStepCount);
        lr.push(_lr);
      }
      return { loss, lr, label: 'Epoch' };
    }
    return { loss: _lossHistory, lr: _lrHistory, label: 'Step' };
  }

  function _updateLossChart() {
    const { loss, lr } = _getChartData();
    if (typeof TrainingChart !== "undefined") {
      TrainingChart.render({ fullLoss: loss, fullLr: lr, viewXMin: _viewXMin, viewXMax: _viewXMax, smoothingWeight: _smoothingWeight });
    }
  }

  function _getChartView() {
    const { loss } = _getChartData();
    const len = Math.max(2, loss.length);
    return { totalLen: len, startIdx: _viewXMin ?? 0, endIdx: _viewXMax ?? len };
  }

  function _addLog(msg, kind) {
    const log = $('monitor-log');
    if (!log) return;
    const entry = document.createElement('div');
    entry.className = 'log-entry log-entry--' + (kind || 'info');
    entry.textContent = '  ' + msg;
    log.appendChild(entry);
    if (typeof autoScrollLog === "function") autoScrollLog(log);
    else log.scrollTop = log.scrollHeight;

    // Console strip
    const consoleLine = $('console-line');
    if (consoleLine) {
      consoleLine.textContent = msg;
      consoleLine.className = 'console__line console__line--' + (kind || 'info');
    }
  }

  function _fillConfigSummary() {
    const panel = $('monitor-config-summary');
    if (!panel) return;
    const c = _config;
    const rows = [
      { k: 'Run', v: c.run_name || 'training' },
      { k: 'Dataset', v: _fmtPath(c.dataset_dir), raw: c.dataset_dir || '' },
      { k: 'Output', v: _fmtPath(c.output_dir), raw: c.output_dir || '' },
      { k: 'Resume', v: c.resume_from ? _fmtPath(c.resume_from) : 'fresh start', raw: c.resume_from || '' },
      { k: 'Adapter', v: c.adapter_type || 'lora' },
      { k: 'Model', v: c.model_variant || 'turbo' },
      { k: 'Rank', v: c.rank || 64 },
      { k: 'Optimizer', v: c.optimizer_type || 'adamw' },
      { k: 'LR', v: c.lr || '1e-4' },
      { k: 'Batch', v: `${c.batch_size || 1} × ${c.grad_accum || 4}` },
      { k: 'Epochs', v: c.epochs || '--' },
      { k: 'Checkpointing', v: c.gradient_checkpointing_ratio || 'full' },
    ];
    const rowsHtml = rows.map(({ k, v, raw }) =>
      `<div style="display:flex;justify-content:space-between;padding:3px 0;gap:12px;"><span class="u-text-muted" style="white-space:nowrap;flex-shrink:0;">${_esc(k)}</span><span title="${_esc(raw || String(v))}" style="text-align:right;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;min-width:0;max-width:220px;">${_esc(v)}</span></div>`
    ).join('');
    panel.innerHTML = rowsHtml +
      '<div style="margin-top:var(--space-sm);display:flex;justify-content:flex-end;">' +
      '<button class="btn btn--sm" id="monitor-copy-summary" title="Copy current monitor run summary">Copy Run Summary</button>' +
      '</div>';

    const copyBtn = $('monitor-copy-summary');
    if (copyBtn) {
      copyBtn.onclick = async () => {
        const ok = await _copyText(_buildRunSummaryText());
        if (typeof showToast === 'function') {
          showToast(ok ? 'Run summary copied' : 'Failed to copy run summary', ok ? 'ok' : 'error');
        }
      };
    }
  }

  // ---- WebSocket message handler ----------------------------------------

  let _progressComplete = false;

  function _onTrainingMessage(msg) {
    if (!msg) return;

    if (_finalized) return;

    // Console log lines from subprocess stdout
    if (msg.type === 'log') {
      const text = String(msg.msg || '');
      if (/CUDA out of memory|torch\.OutOfMemoryError|torch\.cuda\.OutOfMemoryError/.test(text)) {
        _lastFailureMsg = 'CUDA out of memory — reduce batch size, enable gradient checkpointing, or shorten audio';
        _addLog('[OOM]  ' + _lastFailureMsg, 'fail');
      } else {
        _addLog(text, 'info');
      }
      return;
    }

    // Terminal status from subprocess — the ONLY path that should finalize.
    // Exit code is the final authority on success vs failure.
    if (msg.type === 'status') {
      if (msg.oom) {
        _lastFailureMsg = _lastFailureMsg || 'CUDA out of memory — reduce batch size, enable gradient checkpointing, or shorten audio';
      }
      if (msg.reason && !_lastFailureMsg) {
        _lastFailureMsg = String(msg.reason);
      }

      if (msg.status === 'done' && !msg.oom) {
        _finalize('complete');
      } else if (msg.status === 'failed' || msg.oom) {
        if (!_stopRequested && !_lastFailureMsg) {
          _lastFailureMsg = 'Process exited with code ' + (msg.exit_code ?? '?');
        }
        _finalize(_stopRequested ? 'stopped' : 'failed');
      }
      return;
    }

    // Progress updates from .progress.jsonl
    if (msg.type === 'progress') {
      const kind = msg.kind || 'step';

      if (kind === 'step') {
        _step = msg.step || _step;
        _loss = msg.loss ?? _loss;
        _lr = msg.lr ?? _lr;
        _maxEpochs = msg.max_epochs || _maxEpochs;
        _stepsPerEpoch = msg.steps_per_epoch || _stepsPerEpoch;
        _epoch = msg.epoch || _epoch;
        _bestLoss = msg.best_loss ?? _bestLoss;
        _bestEpoch = msg.best_epoch ?? _bestEpoch;
        _stepInEpoch = _stepsPerEpoch > 0 ? _step % _stepsPerEpoch : _step;

        _lossHistory.push(_loss);
        _lrHistory.push(_lr);
        _epochLossAccum += _loss;
        _epochStepCount++;
      }

      if (kind === 'epoch') {
        _epoch = msg.epoch || _epoch;
        _maxEpochs = msg.max_epochs || _maxEpochs;
        _lastEpochDuration = msg.epoch_time || 0;
        _bestLoss = msg.best_loss ?? _bestLoss;
        _bestEpoch = msg.best_epoch ?? _bestEpoch;
        _stepInEpoch = 0;
        _epochStartTime = Date.now();

        const epochAvgLoss = _epochStepCount > 0 ? _epochLossAccum / _epochStepCount : (msg.loss ?? _loss);
        _epochLossHistory.push(epochAvgLoss);
        _epochLrHistory.push(_lr);
        _epochLossAccum = 0;
        _epochStepCount = 0;
        _addLog(`[epoch ${_epoch}/${_maxEpochs}]  avg_loss=${epochAvgLoss.toFixed(4)}  epoch_time=${_lastEpochDuration.toFixed(1)}s`, 'epoch');
      }

      if (kind === 'checkpoint') {
        _addLog(`[checkpoint]  saved at step ${msg.step}`, 'ckpt');
      }

      // Progress 'complete' does NOT finalize — it only updates the UI.
      // The actual 'status' message with exit code is the final authority.
      if (kind === 'complete') {
        _progressComplete = true;
        const infoMsg = String(msg.msg || '');
        if (/stopped by user/i.test(infoMsg)) {
          _addLog('[stopped]  Training stopped by user', 'warn');
        } else {
          _bestLoss = msg.best_loss ?? _bestLoss;
          _bestEpoch = msg.best_epoch ?? _bestEpoch;
          _addLog(`[complete]  Training finished. Best loss: ${_bestLoss.toFixed(4)} at epoch ${_bestEpoch}`, 'ckpt');
          _addLog('[info]  Waiting for process cleanup...', 'info');
        }
      }

      if (kind === 'fail') {
        _lastFailureMsg = String(msg.msg || 'Training failed');
        _addLog(`[fail]  ${_lastFailureMsg}`, 'fail');
      }

      _updateDOM();
    }
  }

  function _refreshHistory() {
    if (typeof History !== 'undefined' && typeof History.loadHistory === 'function') {
      Promise.resolve(History.loadHistory()).catch(() => {});
    }
    document.dispatchEvent(new CustomEvent('sidestep:history-updated'));
  }

  function _finalize(outcome) {
    if (_finalized) return;
    _finalized = true;
    const finalOutcome = outcome || 'failed';
    const wasRunning = _running;
    _running = false;
    _closeWebSockets();
    if (_domTimer) { clearInterval(_domTimer); _domTimer = null; }
    document.querySelector('[data-mode="monitor"]')?.classList.remove('training');
    _setVRAMLock(false);
    _updateDOM();

    if (typeof AppState !== 'undefined') {
      const appStatus = finalOutcome === 'complete'
        ? 'complete'
        : finalOutcome === 'failed' ? 'error' : 'idle';
      AppState.setStatus(appStatus);
      API.fetchGPU().then(g => AppState.setGPU(g)).catch(() => {});
    }

    _refreshHistory();

    if (wasRunning) _showCompletionState(finalOutcome);
    if (wasRunning) _showTerminalToast(finalOutcome);
    _stopRequested = false;
  }

  function _closeWebSockets() {
    if (_ws) { try { _ws.close(); } catch (e) { console.warn('[Training] ws close error:', e); } _ws = null; }
    if (_gpuWs) { try { _gpuWs.close(); } catch (e) { console.warn('[Training] gpu ws close error:', e); } _gpuWs = null; }
  }

  // ---- Public API -------------------------------------------------------

  async function start(config) {
    if (_running) return;

    _config = config || {};
    _running = true;
    _step = 0; _epoch = 0; _stepInEpoch = 0;
    _loss = 0; _bestLoss = Infinity; _bestEpoch = 0; _lr = 0;
    _lossHistory = []; _lrHistory = [];
    _epochLossHistory = []; _epochLrHistory = [];
    _epochLossAccum = 0; _epochStepCount = 0;
    _viewXMin = null; _viewXMax = null; _userZoomed = false;
    _stopRequested = false; _lastFailureMsg = ''; _finalized = false; _progressComplete = false;
    _maxEpochs = parseInt(_config.epochs) || 100;
    _stepsPerEpoch = parseInt(_config.steps_per_epoch) || 47;
    _startTime = Date.now(); _epochStartTime = Date.now(); _lastEpochDuration = 0;

    const _show = (id, disp) => { const e = $(id); if (e) e.style.display = disp; };
    _show('monitor-idle', 'none'); _show('monitor-active', 'block'); _show('monitor-controls', 'flex');
    const completion = $('monitor-completion');
    if (completion) { completion.style.display = 'none'; completion.innerHTML = ''; }
    const _el = (id, val) => { const e = $(id); if (e) e.textContent = val; };
    _el('monitor-run-name', _config.run_name || 'training');
    _el('monitor-output-dir', 'Output: ' + (_config.output_dir || ''));
    const log = $('monitor-log'); if (log) log.innerHTML = '';
    _fillConfigSummary();
    _addLog('[info]  Starting training...', 'info');
    document.querySelector('[data-mode="monitor"]')?.classList.add('training');
    _setVRAMLock(true);
    if (typeof AppState !== 'undefined') AppState.setStatus('training');
    if (typeof switchMode === 'function') switchMode('monitor');

    // Call backend to start training subprocess
    try {
      const result = await API.startTraining(_config);
      if (result.error) {
        _lastFailureMsg = String(result.error || 'Could not start training');
        _addLog('[fail]  ' + _lastFailureMsg, 'fail');
        _finalize('failed');
        return;
      }
      _taskId = result.task_id;
      _addLog('[info]  Subprocess started (task: ' + _taskId + ')', 'info');
    } catch (err) {
      _lastFailureMsg = 'Could not start training: ' + err.message;
      _addLog('[fail]  ' + _lastFailureMsg, 'fail');
      _finalize('failed');
      return;
    }

    // Connect WebSockets for real-time updates (auto-reconnects)
    _ws = API.connectTrainingWS(_onTrainingMessage);

    _gpuWs = API.connectGpuWS((data) => {
      if (data && typeof data === 'object' && typeof AppState !== 'undefined' && data.available !== false) AppState.setGPU(data);
    });

    // Periodic DOM refresh (timing display, ETA, etc.)
    _domTimer = setInterval(() => { if (_running) _updateDOM(); }, 1000);
  }

  async function stop() {
    if (!_running) return;
    _stopRequested = true;
    const stopBtn = $('btn-stop-training');
    if (stopBtn) { stopBtn.textContent = 'Stopping\u2026'; stopBtn.disabled = true; }
    _addLog('[info]  Stop requested — waiting for trainer to exit...', 'info');
    try {
      await API.stopTraining();
    } catch {}
    // Finalization happens when the subprocess exits and WS sends status
  }

  function _showCompletionState(outcome) {
    const elapsed = (Date.now() - _startTime) / 1000;
    const done = outcome === 'complete';
    const failed = outcome === 'failed';
    const controls = $('monitor-controls'), completion = $('monitor-completion');
    if (controls) controls.style.display = 'none';
    if (completion) {
      const c = done ? 'var(--success)' : failed ? 'var(--error)' : 'var(--warning)';
      const icon = done ? '[ok]' : failed ? '[x]' : '[!]';
      const label = done ? 'Training Complete' : failed ? 'Training Failed' : 'Training Stopped';
      const best = _bestLoss < Infinity ? _bestLoss.toFixed(4) : '--';
      const reason = failed && _lastFailureMsg
        ? '<div class="u-text-error" style="margin-top:var(--space-xs);">Reason: ' + _esc(_lastFailureMsg) + '</div>'
        : '';
      completion.innerHTML =
        '<div style="padding:var(--space-md);border:1px solid '+c+';border-radius:var(--radius);">' +
        '<div style="color:'+c+';font-weight:bold;font-size:var(--font-size-lg);margin-bottom:var(--space-sm);">'+icon+' '+label+'</div>' +
        '<div style="font-size:var(--font-size-sm);color:var(--text);">' +
        '<div>Epochs: '+_epoch+' / '+_maxEpochs+' · Steps: '+_step+'</div>' +
        '<div>Best loss: '+best+' at epoch '+_bestEpoch+'</div><div>Duration: '+_fmtDuration(elapsed)+'</div>' + reason + '</div>' +
        '<div style="margin-top:var(--space-md);display:flex;gap:var(--space-sm);">' +
        '<button class="btn btn--primary" id="btn-new-run">New Run</button>' +
        '<button class="btn" id="btn-completed-open-output">Open Output Dir</button>' +
        (!done ? '<button class="btn btn--success" id="btn-completed-resume">Resume from History</button>' : '') +
        '</div></div>';
      completion.style.display = 'block';
      $('btn-new-run')?.addEventListener('click', () => { if (typeof switchMode === 'function') switchMode('ez'); });
      $('btn-completed-open-output')?.addEventListener('click', _openOutputDirFromMonitor);
      $('btn-completed-resume')?.addEventListener('click', () => {
        if (typeof switchMode === 'function') switchMode('lab');
        _refreshHistory();
        setTimeout(() => {
          document.querySelectorAll('.lab-nav__item').forEach(i => i.classList.remove('active'));
          document.querySelectorAll('.lab-panel').forEach(p => p.classList.remove('active'));
          document.querySelector('.lab-nav__item[data-lab="history"]')?.classList.add('active');
          $('lab-history')?.classList.add('active');
        }, 50);
      });
    }
    const consoleLine = $('console-line');
    if (consoleLine) {
      const statusText = done ? 'Training complete' : failed ? 'Training failed' : 'Training stopped';
      consoleLine.textContent = statusText +
        ' — ' + _epoch + ' epochs, best: ' + (_bestLoss < Infinity ? _bestLoss.toFixed(4) : '--');
      consoleLine.className = 'console__line console__line--' + (done ? 'epoch' : failed ? 'fail' : 'warn');
    }
  }

  function _setVRAMLock(locked) {
    document.querySelectorAll('.vram-action').forEach(btn => {
      btn.disabled = locked;
    });
  }

  function isRunning() { return _running; }

  // Set visible X range (absolute data indices). null = show all.
  function setViewRange(xMin, xMax) {
    if (xMin === null || xMax === null) {
      _viewXMin = null;
      _viewXMax = null;
      _userZoomed = false;
    } else {
      _viewXMin = xMin;
      _viewXMax = xMax;
      _userZoomed = true;
    }
    _updateLossChart();
  }

  function zoomReset() {
    setViewRange(null, null);
  }

  function setSmoothing(weight) {
    _smoothingWeight = Math.max(0, Math.min(weight, 0.99));
    _updateLossChart();
  }

  function setChartMode(mode) {
    _chartMode = mode; // 'step' or 'epoch'
    _viewXMin = null;
    _viewXMax = null;
    _userZoomed = false;
    _updateLossChart();
  }

  function getChartView() { return _getChartView(); }
  function getChartMode() { return _chartMode; }

  return { init() {}, start, stop, isRunning, setViewRange, zoomReset, setSmoothing, setChartMode, getChartView, getChartMode };

})();
