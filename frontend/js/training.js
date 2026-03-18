/* Side-Step GUI — Training Controller (WebSocket + Monitor DOM) */

const Training = (() => {

  let _running = false, _finalized = false;
  let _ws = null, _gpuWs = null;

  // ---- Training Queue ---------------------------------------------------
  let _queue = [];  // { id, config, status: 'pending'|'running'|'done'|'failed'|'stopped' }
  let _queueIdCounter = 0;
  let _step = 0, _epoch = 0, _maxEpochs = 100, _stepsPerEpoch = 0, _stepInEpoch = 0;
  let _loss = 0, _bestLoss = Infinity, _bestEpoch = 0, _lr = 0;
  let _lossHistory = [], _lrHistory = [];
  let _epochLossHistory = [], _epochLrHistory = [];
  let _epochLossAccum = 0, _epochStepCount = 0;
  let _startTime = 0, _epochStartTime = 0, _lastEpochDuration = 0;
  let _config = {}, _viewXMin = null, _viewXMax = null;
  let _smoothingWeight = 0.6, _userZoomed = false;
  let _taskId = null, _domTimer = null;
  let _stopRequested = false, _lastFailureMsg = "";

  // TensorBoard scalar data from tfevents reader — { tag: [{step, value, wall_time}] }
  let _tbScalars = {};
  // TensorBoard histogram data — { tag: [{step, wall_time, bins: [{x, dx, y}]}] }
  let _tbHistograms = {};
  let _snapToGrid = true;        // S button: snap cursor to nearest data point
  let _histMode = 'overlay';     // 'overlay' | 'offset' — expanded histogram display mode
  let _histTipState = null;       // stored state for histogram tooltip interaction
  let _expandedTag = null;        // currently expanded mini-chart tag (null = collapsed)
  let _expandedViewMin = null, _expandedViewMax = null;  // expanded chart zoom state
  let _chartMode = 'epoch';       // 'step' | 'epoch'
  let _demoTimer = null;          // live demo interval handle

  const _esc = window._esc || ((v) => String(v ?? ''));

  function _fmtPath(path) {
    const p = String(path || "").trim();
    if (!p) return "--";
    return p;
  }

  const _copyText = typeof copyTextToClipboard === 'function' ? copyTextToClipboard : async () => false;

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
    setText('monitor-step-label', _stepsPerEpoch > 0 ? `Step ${_stepInEpoch} / ${_stepsPerEpoch}` : `Step ${_step}`);
    setText('monitor-step-pct', _stepsPerEpoch > 0 ? Math.round(stepPct) + '%' : '');
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

  const _fmtDuration = window._fmtDuration || ((seconds) => {
    const s = Math.floor(seconds), h = Math.floor(s / 3600), m = Math.floor((s % 3600) / 60), sec = s % 60;
    if (h > 0) return `${h}h ${String(m).padStart(2, '0')}m ${String(sec).padStart(2, '0')}s`;
    if (m > 0) return `${m}m ${String(sec).padStart(2, '0')}s`;
    return `${sec}s`;
  });

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
    // Hide hover dots — Y range may have changed, dots would be stale
    const dotsG = $('monitor-hover-dots');
    if (dotsG) dotsG.style.display = 'none';
  }

  // ---- Mini-charts for additional TensorBoard scalars --------------------
  // Known tags get curated labels/colors; any unknown tag auto-discovers
  const _KNOWN_TAG_META = {
    'train/loss':         { label: 'Step Loss',      color: 'var(--primary)' },
    'train/epoch_loss':   { label: 'Epoch Loss',     color: 'var(--primary)' },
    'train/lr':           { label: 'Learning Rate',  color: 'var(--secondary)' },
    'train/grad_norm':    { label: 'Grad Norm',      color: 'var(--accent)' },
    'val_loss':           { label: 'Val Loss',       color: 'var(--success)' },
    'target_loss_scale':  { label: 'Loss Scale',     color: 'var(--warning)' },
    'target_loss_ema':    { label: 'Loss EMA',       color: 'var(--changed)' },
  };
  // Auto-color palette for unknown tags (HSL hues)
  const _AUTO_COLORS = [
    'hsl(180,60%,55%)', 'hsl(45,80%,55%)', 'hsl(270,55%,60%)',
    'hsl(120,50%,50%)', 'hsl(330,60%,55%)', 'hsl(200,70%,55%)',
    'hsl(60,65%,50%)',  'hsl(0,60%,55%)',
  ];
  let _autoColorIdx = 0;
  const _tagColorCache = {};

  function _getMiniChartTags() {
    const tags = {};
    // The main chart already shows one of these — show the OTHER as a mini-chart
    const mainTag = _chartMode === 'epoch' ? 'train/epoch_loss' : 'train/loss';
    const altTag  = _chartMode === 'epoch' ? 'train/loss' : 'train/epoch_loss';

    // Preferred order: alt loss first, then lr, then everything else alphabetically
    const ordered = [];
    if (_tbScalars[altTag]) ordered.push(altTag);
    if (_tbScalars['train/lr']) ordered.push('train/lr');

    // Collect remaining tags
    const seen = new Set([mainTag, altTag, 'train/lr']);
    for (const tag of Object.keys(_tbScalars).sort()) {
      if (!seen.has(tag)) ordered.push(tag);
      seen.add(tag);
    }

    for (const tag of ordered) {
      const data = _tbScalars[tag];
      if (!data || data.length < 2) continue;
      const known = _KNOWN_TAG_META[tag];
      if (known) {
        tags[tag] = { label: known.label, color: known.color };
      } else {
        // Auto-generate label from tag name
        const label = tag.replace(/^train\//, '').replace(/_/g, ' ');
        // Assign stable color per tag
        if (!_tagColorCache[tag]) {
          _tagColorCache[tag] = _AUTO_COLORS[_autoColorIdx % _AUTO_COLORS.length];
          _autoColorIdx++;
        }
        tags[tag] = { label, color: _tagColorCache[tag] };
      }
    }
    return tags;
  }
  let _miniChartDebounce = null;

  function _renderMiniCharts() {
    if (_miniChartDebounce) return;
    _miniChartDebounce = requestAnimationFrame(() => {
      _miniChartDebounce = null;
      _doRenderMiniCharts();
      if (_expandedTag) {
        if (_tbHistograms[_expandedTag]) _renderExpandedHistogram();
        else _renderExpandedChart();
      }
    });
  }

  function _tagId(tag) { return tag.replace(/\//g, '-'); }
  function _fmtVal(v) { return Math.abs(v) < 0.01 ? v.toExponential(2) : v.toFixed(4); }
  /** TB-style axis tick label (auto precision) */
  function _fmtAxisLabel(v) {
    const a = Math.abs(v);
    if (a === 0) return '0';
    if (a >= 1e6 || a < 0.001) return v.toExponential(1);
    if (a >= 100) return v.toFixed(0);
    if (a >= 1) return v.toFixed(2);
    return v.toPrecision(3);
  }

  // Smart tick formatter: picks the shortest unambiguous representation
  function _fmtTick(v) {
    const a = Math.abs(v);
    if (a === 0) return '0';
    if (a >= 1e6)  return (v / 1e6).toPrecision(2) + 'M';
    if (a >= 1e3)  return (v / 1e3).toPrecision(2) + 'k';
    if (a >= 1)    return v.toPrecision(3);
    if (a >= 0.01) return v.toFixed(3);
    return v.toExponential(1);
  }

  // Compute nice Y ticks (3 ticks: top, middle, bottom)
  function _niceYTicks(yMin, yMax, n) {
    if (n < 2) n = 2;
    const ticks = [];
    const step = (yMax - yMin) / (n - 1);
    for (let i = 0; i < n; i++) ticks.push(yMax - i * step);
    return ticks;
  }

  // ---- Mini-chart drag reorder helpers ----
  let _dragSrcPanel = null;

  function _wireDrag(panel) {
    panel.draggable = true;
    panel.addEventListener('dragstart', (e) => {
      _dragSrcPanel = panel;
      panel.classList.add('mini-chart-panel--dragging');
      e.dataTransfer.effectAllowed = 'move';
      e.dataTransfer.setData('text/plain', panel.id);
    });
    panel.addEventListener('dragend', () => {
      panel.classList.remove('mini-chart-panel--dragging');
      _dragSrcPanel = null;
      document.querySelectorAll('.mini-chart-panel--dragover').forEach(el => el.classList.remove('mini-chart-panel--dragover'));
    });
    panel.addEventListener('dragover', (e) => {
      if (!_dragSrcPanel || _dragSrcPanel === panel) return;
      e.preventDefault();
      e.dataTransfer.dropEffect = 'move';
      panel.classList.add('mini-chart-panel--dragover');
    });
    panel.addEventListener('dragleave', () => {
      panel.classList.remove('mini-chart-panel--dragover');
    });
    panel.addEventListener('drop', (e) => {
      e.preventDefault();
      panel.classList.remove('mini-chart-panel--dragover');
      if (!_dragSrcPanel || _dragSrcPanel === panel) return;
      const container = panel.parentNode;
      const panels = [...container.querySelectorAll('.mini-chart-panel')];
      const srcIdx = panels.indexOf(_dragSrcPanel);
      const dstIdx = panels.indexOf(panel);
      if (srcIdx < 0 || dstIdx < 0) return;
      if (srcIdx < dstIdx) {
        container.insertBefore(_dragSrcPanel, panel.nextSibling);
      } else {
        container.insertBefore(_dragSrcPanel, panel);
      }
    });
  }

  function _doRenderMiniCharts() {
    const container = $('monitor-mini-charts');
    if (!container) return;
    const tags = _getMiniChartTags();
    const vbW = 200, vbH = 60;
    const numYTicks = 3;

    // Remove panels for tags no longer in config (epoch/step swap)
    container.querySelectorAll('.mini-chart-panel').forEach(el => {
      const id = el.id.replace('mini-chart-', '').replace(/-/g, '/');
      if (!tags[id] && !_tbHistograms[id]) el.remove();
    });

    // ---- Scalar mini-charts ----
    for (const [tag, meta] of Object.entries(tags)) {
      const data = _tbScalars[tag];
      if (!data || data.length < 2) continue;
      const tid = _tagId(tag);
      let panel = $('mini-chart-' + tid);
      if (!panel) {
        panel = document.createElement('div');
        panel.id = 'mini-chart-' + tid;
        panel.className = 'mini-chart-panel';
        panel.dataset.tag = tag;
        panel.innerHTML =
          '<div class="mini-chart-panel__header">' +
            '<span class="mini-chart-panel__label">' + meta.label + '</span>' +
            '<span class="mini-chart-panel__value" id="mini-val-' + tid + '">--</span>' +
          '</div>' +
          '<div class="mini-chart-panel__grid">' +
            '<div class="mini-chart-panel__yaxis" id="mini-yaxis-' + tid + '"></div>' +
            '<div class="mini-chart-panel__series">' +
              '<svg class="mini-chart-panel__svg" preserveAspectRatio="none" viewBox="0 0 ' + vbW + ' ' + vbH + '">' +
                '<g class="mini-grid-lines"></g>' +
                '<defs><linearGradient id="grad-' + tid + '" x1="0" y1="0" x2="0" y2="1">' +
                  '<stop offset="0%" stop-color="' + meta.color + '" stop-opacity="0.18"/>' +
                  '<stop offset="100%" stop-color="' + meta.color + '" stop-opacity="0.01"/>' +
                '</linearGradient></defs>' +
                '<polygon class="mini-chart-panel__fill" fill="url(#grad-' + tid + ')" points="" />' +
                '<polyline class="mini-raw" fill="none" stroke="' + meta.color + '" stroke-width="2" vector-effect="non-scaling-stroke" opacity="0.3" points="" />' +
                '<polyline class="mini-smooth" fill="none" stroke="' + meta.color + '" stroke-width="2" vector-effect="non-scaling-stroke" points="" />' +
              '</svg>' +
              '<div class="mini-end-dot" style="background:' + meta.color + ';"></div>' +
            '</div>' +
            '<div class="mini-chart-panel__xaxis" id="mini-xaxis-' + tid + '"></div>' +
          '</div>';
        panel.addEventListener('click', () => _toggleExpandedChart(tag));
        _wireDrag(panel);
        container.appendChild(panel);
      }
      panel.classList.toggle('mini-chart-panel--active', _expandedTag === tag);
      const values = data.map(d => d.value);
      const last = values[values.length - 1];
      const valEl = $('mini-val-' + tid);
      if (valEl) valEl.textContent = _fmtVal(last);

      // Smoothed values
      const smoothed = typeof TrainingChart !== 'undefined'
        ? TrainingChart.expSmooth(values, _smoothingWeight)
        : values;

      // TB-faithful Y domain (exclude raw from domain — raw is aux)
      const { yMin, yMax, yTicks: miniYTicks } = _computeYDomain(smoothed, numYTicks);

      const n = values.length;
      const toX = (i) => n > 1 ? (i / (n - 1)) * vbW : vbW / 2;
      const toY = (v) => yMax > yMin ? ((yMax - v) / (yMax - yMin)) * vbH : vbH / 2;

      // Grid lines (use ticks from TB domain computation)
      const gridG = panel.querySelector('.mini-grid-lines');
      if (gridG) {
        let gl = '';
        const ySpacing = miniYTicks.length > 1 ? Math.abs(miniYTicks[1] - miniYTicks[0]) : 1;
        for (const t of miniYTicks) {
          const y = toY(t).toFixed(1);
          const cls = Math.abs(t) < ySpacing * 0.01 ? 'grid-line zero' : 'grid-line';
          gl += '<line class="' + cls + '" x1="0" y1="' + y + '" x2="' + vbW + '" y2="' + y + '" />';
        }
        gridG.innerHTML = gl;
      }

      // Y-axis ticks
      const yAxisEl = $('mini-yaxis-' + tid);
      if (yAxisEl) {
        yAxisEl.innerHTML = miniYTicks.slice().reverse().map(t => '<span>' + _fmtTick(t) + '</span>').join('');
      }

      // X-axis ticks (first and last step)
      const xAxisEl = $('mini-xaxis-' + tid);
      if (xAxisEl) {
        const firstStep = data[0].step;
        const lastStep = data[data.length - 1].step;
        xAxisEl.innerHTML = '<span>' + firstStep + '</span><span>' + lastStep + '</span>';
      }

      // Raw line (faded)
      const rawLine = panel.querySelector('.mini-raw');
      if (rawLine) {
        rawLine.setAttribute('points', values.map((v, i) => toX(i).toFixed(1) + ',' + toY(v).toFixed(1)).join(' '));
      }
      // Smoothed line (bold)
      const smoothLine = panel.querySelector('.mini-smooth');
      if (smoothLine) {
        smoothLine.setAttribute('points', smoothed.map((v, i) => toX(i).toFixed(1) + ',' + toY(v).toFixed(1)).join(' '));
      }
      // End dot at last smoothed value (HTML overlay)
      const endDot = panel.querySelector('.mini-end-dot');
      if (endDot && smoothed.length > 0) {
        const pctX = n > 1 ? ((n - 1) / (n - 1)) * 100 : 50;
        const lastSm = smoothed[smoothed.length - 1];
        const pctY = yMax > yMin ? ((yMax - lastSm) / (yMax - yMin)) * 100 : 50;
        endDot.style.left = pctX.toFixed(1) + '%';
        endDot.style.top = pctY.toFixed(1) + '%';
      }
      // Gradient fill under smoothed line
      const fill = panel.querySelector('.mini-chart-panel__fill');
      if (fill) {
        const pts = smoothed.map((v, i) => toX(i).toFixed(1) + ',' + toY(v).toFixed(1)).join(' ');
        fill.setAttribute('points', pts + ' ' + vbW + ',' + vbH + ' 0,' + vbH);
      }
    }

    // ---- Histogram mini-charts (overlay mode like TB) ----
    for (const [tag, histData] of Object.entries(_tbHistograms)) {
      if (!histData || histData.length < 1) continue;
      const tid = _tagId(tag);
      let panel = $('mini-chart-' + tid);
      if (!panel) {
        const label = tag.replace('train/', '').replace(/_/g, ' ');
        panel = document.createElement('div');
        panel.id = 'mini-chart-' + tid;
        panel.className = 'mini-chart-panel';
        panel.dataset.tag = tag;
        panel.dataset.histogram = '1';
        panel.innerHTML =
          '<div class="mini-chart-panel__header">' +
            '<span class="mini-chart-panel__label">' + label + '</span>' +
            '<span class="mini-chart-panel__value" id="mini-val-' + tid + '">step ' + histData[histData.length - 1].step + '</span>' +
          '</div>' +
          '<div class="mini-chart-panel__grid">' +
            '<div class="mini-chart-panel__yaxis" id="mini-yaxis-' + tid + '"></div>' +
            '<div class="mini-chart-panel__series">' +
              '<svg class="mini-chart-panel__svg" preserveAspectRatio="none" viewBox="0 0 ' + vbW + ' ' + vbH + '">' +
                '<g class="mini-grid-lines"></g>' +
                '<g class="mini-hist-paths"></g>' +
              '</svg>' +
            '</div>' +
            '<div class="mini-chart-panel__xaxis" id="mini-xaxis-' + tid + '"></div>' +
          '</div>';
        panel.addEventListener('click', () => _toggleExpandedChart(tag));
        _wireDrag(panel);
        container.appendChild(panel);
      }
      panel.classList.toggle('mini-chart-panel--active', _expandedTag === tag);
      // Update value label
      const valEl = $('mini-val-' + tid);
      if (valEl) valEl.textContent = 'step ' + histData[histData.length - 1].step;

      // TB: Normalize bins to equal width (30 bins) before rendering
      const maxShow = Math.min(8, histData.length);
      const recent = histData.slice(-maxShow);
      const normRecent = _normalizeHistograms(recent);

      // Compute extents on normalized data
      let binMin = Infinity, binMax = -Infinity, countMax = 0;
      for (const d of normRecent) {
        for (const b of d.bins) {
          if (b.x < binMin) binMin = b.x;
          if (b.x + b.dx > binMax) binMax = b.x + b.dx;
          if (b.y > countMax) countMax = b.y;
        }
      }
      if (binMin >= binMax) { binMax = binMin + 1; }
      if (countMax <= 0) countMax = 1;
      const bRange = binMax - binMin;

      const toHX = (bx) => ((bx - binMin) / bRange) * vbW;
      const toHY = (cy) => vbH - (cy / countMax) * vbH;

      // Grid lines (3 horizontal)
      const gridG = panel.querySelector('.mini-grid-lines');
      if (gridG) {
        const ticks = _niceYTicks(0, countMax, 3);
        let gl = '';
        for (const t of ticks) {
          const y = toHY(t).toFixed(1);
          gl += '<line class="grid-line" x1="0" y1="' + y + '" x2="' + vbW + '" y2="' + y + '" />';
        }
        gridG.innerHTML = gl;
      }

      // Y-axis
      const yAxisEl = $('mini-yaxis-' + tid);
      if (yAxisEl) {
        const ticks = _niceYTicks(0, countMax, 3);
        yAxisEl.innerHTML = ticks.map(t => '<span>' + _fmtTick(t) + '</span>').join('');
      }
      // X-axis
      const xAxisEl = $('mini-xaxis-' + tid);
      if (xAxisEl) {
        xAxisEl.innerHTML = '<span>' + _fmtTick(binMin) + '</span><span>' + _fmtTick(binMax) + '</span>';
      }

      // Render histogram paths using normalized equal-width bins
      const pathsG = panel.querySelector('.mini-hist-paths');
      if (pathsG) {
        let html = '';
        for (let si = 0; si < normRecent.length; si++) {
          const d = normRecent[si];
          const opacity = 0.15 + 0.85 * (si / Math.max(1, normRecent.length - 1));
          // TB path: M(firstCentroid, baseline) → L(centroids) → L(lastCentroid, baseline)
          if (d.bins.length === 0) continue;
          const first = d.bins[0], last = d.bins[d.bins.length - 1];
          let path = 'M' + toHX(first.x + first.dx / 2).toFixed(1) + ',' + vbH;
          for (const b of d.bins) {
            path += ' L' + toHX(b.x + b.dx / 2).toFixed(1) + ',' + toHY(b.y).toFixed(1);
          }
          path += ' L' + toHX(last.x + last.dx / 2).toFixed(1) + ',' + vbH;
          html += '<path class="hist-path" d="' + path + '" fill="var(--primary)" fill-opacity="' + (opacity * 0.7).toFixed(2) + '" stroke="var(--primary)" stroke-opacity="' + opacity.toFixed(2) + '" />';
        }
        pathsG.innerHTML = html;
      }
    }
  }

  // ---- Expanded mini-chart (click to expand/collapse) --------------------
  function _removeHistToggle() {
    const btn = $('hist-mode-toggle');
    if (btn) btn.remove();
  }

  function _toggleExpandedChart(tag) {
    if (_expandedTag === tag) {
      _expandedTag = null;
      _expandedViewMin = null; _expandedViewMax = null;
      _removeHistToggle();
      const ec = $('monitor-expanded-chart');
      if (ec) ec.style.display = 'none';
      // Remove active highlight
      document.querySelectorAll('.mini-chart-panel--active').forEach(el => el.classList.remove('mini-chart-panel--active'));
    } else {
      _expandedTag = tag;
      _expandedViewMin = null; _expandedViewMax = null;
      _removeHistToggle();
      const ec = $('monitor-expanded-chart');
      if (ec) { ec.style.display = ''; ec.style.animation = 'none'; ec.offsetHeight; ec.style.animation = ''; }
      if (_tbHistograms[tag]) {
        _renderExpandedHistogram();
      } else {
        _renderExpandedChart();
        _wireExpandedInteraction();
      }
      // Highlight active panel
      document.querySelectorAll('.mini-chart-panel').forEach(el => {
        el.classList.toggle('mini-chart-panel--active', el.dataset.tag === tag);
      });
    }
  }

  function _getExpandedData() {
    if (!_expandedTag) return null;
    const data = _tbScalars[_expandedTag];
    if (!data || data.length < 2) return null;
    return data.map(d => d.value);
  }

  function _getExpandedView() {
    const values = _getExpandedData();
    if (!values) return { totalLen: 1, startIdx: 0, endIdx: 1 };
    const len = values.length;
    return { totalLen: len, startIdx: _expandedViewMin ?? 0, endIdx: _expandedViewMax ?? len };
  }

  // TB: d3.scaleLinear().domain([min,max]).nice() equivalent for bin axis
  function _niceBinDomain(lo, hi) {
    if (hi <= lo) return { lo: lo - 1, hi: lo + 1 };
    const span = hi - lo;
    const rawStep = span / 9; // ~10 ticks
    const exp = Math.floor(Math.log10(rawStep));
    const frac = rawStep / Math.pow(10, exp);
    let niceStep;
    if (frac <= 1) niceStep = 1;
    else if (frac <= 2) niceStep = 2;
    else if (frac <= 5) niceStep = 5;
    else niceStep = 10;
    niceStep *= Math.pow(10, exp);
    return {
      lo: Math.floor(lo / niceStep) * niceStep,
      hi: Math.ceil(hi / niceStep) * niceStep
    };
  }

  // ---- TB Histogram Bin Normalization (histogram_util.ts) ----
  // Redistributes variable-width bins into equal-width bins.
  const _HIST_BIN_COUNT = 30; // TB DEFAULT_BIN_COUNT

  function _getBinRange(histograms) {
    let left = null, right = null;
    for (const d of histograms) {
      if (!d.bins || !d.bins.length) continue;
      const last = d.bins[d.bins.length - 1];
      const hLeft = d.bins[0].x;
      const hRight = last.x + last.dx;
      if (left === null || hLeft < left) left = hLeft;
      if (right === null || hRight > right) right = hRight;
    }
    if (left === null || right === null) return null;
    return { left, right };
  }

  function _getBinContribution(bin, resultLeft, resultRight, hasRightNeighbor) {
    const binLeft = bin.x, binRight = bin.x + bin.dx;
    if (binLeft > resultRight || binRight < resultLeft) return { curr: 0, next: 0 };
    if (bin.dx === 0) {
      if (hasRightNeighbor && binRight >= resultRight) return { curr: 0, next: bin.y };
      return { curr: bin.y, next: 0 };
    }
    const intersection = Math.min(binRight, resultRight) - Math.max(binLeft, resultLeft);
    return { curr: (bin.y * intersection) / bin.dx, next: 0 };
  }

  function _rebuildBins(bins, range, binCount) {
    const results = [];
    const dx = (range.right - range.left) / binCount;
    let binIndex = 0, nextContrib = 0;
    for (let i = 0; i < binCount; i++) {
      const rLeft = range.left + i * dx;
      const rRight = rLeft + dx;
      const isLast = i === binCount - 1;
      let y = nextContrib;
      nextContrib = 0;
      while (binIndex < bins.length) {
        const b = bins[binIndex];
        const c = _getBinContribution(b, rLeft, rRight, !isLast);
        y += c.curr;
        nextContrib += c.next;
        if (b.x + b.dx > rRight) break;
        binIndex++;
      }
      results.push({ x: rLeft, dx: dx, y: y });
    }
    return results;
  }

  function _normalizeHistograms(histograms) {
    if (!histograms.length) return [];
    const range = _getBinRange(histograms);
    if (!range) return histograms;
    // TB: if range is 0 width, expand
    if (range.left === range.right) {
      range.right = range.right * 1.1 + 1;
      range.left = range.left / 1.1 - 1;
    }
    return histograms.map(d => ({
      step: d.step,
      wallTime: d.wallTime,
      bins: range ? _rebuildBins(d.bins || [], range, _HIST_BIN_COUNT) : []
    }));
  }

  function _renderExpandedHistogram() {
    const histData = _tbHistograms[_expandedTag];
    if (!histData || histData.length < 1) return;
    const label = _expandedTag.replace(/^train\//, '').replace(/_/g, ' ');
    const titleEl = $('expanded-chart-title');
    const valueEl = $('expanded-chart-value');
    if (titleEl) titleEl.textContent = label;
    if (valueEl) {
      valueEl.textContent = 'step ' + histData[histData.length - 1].step;
      // Add overlay/offset toggle if not already present
      if (!$('hist-mode-toggle')) {
        const btn = document.createElement('button');
        btn.id = 'hist-mode-toggle';
        btn.className = 'btn btn--sm';
        btn.style.cssText = 'margin-left:8px;font-size:0.7rem;padding:2px 8px;';
        btn.textContent = _histMode === 'overlay' ? 'Overlay' : 'Offset';
        btn.addEventListener('click', (e) => {
          e.stopPropagation();
          _histMode = _histMode === 'overlay' ? 'offset' : 'overlay';
          btn.textContent = _histMode === 'overlay' ? 'Overlay' : 'Offset';
          _renderExpandedHistogram();
        });
        valueEl.parentNode.insertBefore(btn, valueEl.nextSibling);
      }
    }

    const svg = $('expanded-svg');
    const yLabels = $('expanded-y-labels');
    const xLabels = $('expanded-x-labels');
    const gridG = $('expanded-grid-lines');
    if (!svg) return;

    // Clean up ChartInteraction if previously wired (prevents tooltip conflicts)
    const area = $('expanded-series-area');
    if (area && area._ciCleanup) { area._ciCleanup(); delete area._ciCleanup; _expandedWired = false; }

    // Hide scalar elements + tooltip/crosshair/hover (prevent leak into histogram)
    const eLine = $('expanded-line');
    const eSmoothLine = $('expanded-smooth-line');
    const eCrosshair = $('expanded-crosshair');
    const eTooltip = $('expanded-tooltip');
    const eHoverDots = $('expanded-hover-dots');
    if (eLine) eLine.setAttribute('points', '');
    if (eSmoothLine) eSmoothLine.setAttribute('points', '');
    if (eCrosshair) eCrosshair.style.display = 'none';
    if (eTooltip) eTooltip.style.display = 'none';
    if (eHoverDots) eHoverDots.style.display = 'none';
    // Also hide SVG hover circles
    const hcG = svg.querySelector('.expanded-hover-circles');
    if (hcG) hcG.style.display = 'none';

    // ---- TensorBoard-faithful histogram rendering ----
    // Step 1: Normalize bins (TB: buildNormalizedHistograms, 30 equal-width bins)
    const maxShow = Math.min(30, histData.length);
    const rawData = histData.slice(-maxShow);
    const data = _normalizeHistograms(rawData);
    const nSlices = data.length;

    // Compute extents on normalized data (TB: binScale domain from min/max)
    let xMin = Infinity, xMax = -Infinity, yMax = 0;
    for (const d of data) {
      for (const b of d.bins) {
        if (b.x < xMin) xMin = b.x;
        if (b.x + b.dx > xMax) xMax = b.x + b.dx;
        if (b.y > yMax) yMax = b.y;
      }
    }
    if (xMin >= xMax) xMax = xMin + 1;
    if (yMax <= 0) yMax = 1;

    // TB: binScale uses .nice() for clean axis boundaries
    const niceBin = _niceBinDomain(xMin, xMax);
    xMin = niceBin.lo; xMax = niceBin.hi;

    // Outline canvas (TB uses 500)
    const C = 500;
    const xToC = (x) => ((x - xMin) / (xMax - xMin)) * C;
    const yToC = (y) => C - (y / yMax) * C;

    // Generate closed-area path (TB: getHistogramPath using getXCentroid)
    // M(firstCentroid, baseline) → L(centroids...) → L(lastCentroid, baseline)
    function makePath(bins) {
      if (!bins.length) return '';
      const first = bins[0], last = bins[bins.length - 1];
      let d = 'M' + xToC(first.x + first.dx / 2).toFixed(2) + ',' + C;
      for (const b of bins) {
        d += 'L' + xToC(b.x + b.dx / 2).toFixed(2) + ',' + yToC(b.y).toFixed(2);
      }
      d += 'L' + xToC(last.x + last.dx / 2).toFixed(2) + ',' + C;
      return d;
    }

    // Color: oldest (back) = dark, newest (front) = bright.
    // HSL — base hue 200° (teal). si=0 is oldest (drawn first, behind).
    function sliceColor(si) {
      const t = nSlices > 1 ? si / (nSlices - 1) : 1; // 0=oldest → 1=newest
      const L = 30 + 38 * t;   // 30% (dark/old) → 68% (bright/new)
      const S = 70 - 15 * t;   // slight saturation boost for darker shades
      return 'hsl(200,' + S.toFixed(0) + '%,' + L.toFixed(0) + '%)';
    }

    // ViewBox
    const vbW = 1000, vbH = 500;
    svg.setAttribute('viewBox', '0 0 ' + vbW + ' ' + vbH);

    // Ensure histogram paths group exists
    let pathsG = svg.querySelector('.expanded-hist-paths');
    if (!pathsG) {
      pathsG = document.createElementNS('http://www.w3.org/2000/svg', 'g');
      pathsG.classList.add('expanded-hist-paths');
      svg.appendChild(pathsG);
    }

    // Sort by time — oldest first (back of DOM), newest last (front, occludes)
    const sorted = [...data].sort((a, b) => a.step - b.step);

    // Scale factors: path drawn in CxC, stretched to viewBox
    const sxFull = vbW / C;
    const syFull = vbH / C;

    if (_histMode === 'offset') {
      // TB offset: sliceHeight = outerHeight / 2.5
      const sliceH = vbH / 2.5;                  // 200 in vb units
      const baselineTop = sliceH;                 // oldest baseline
      const baselineBot = vbH;                    // newest baseline
      const baseSpan = baselineBot - baselineTop;
      const stepMin = sorted[0].step, stepMax = sorted[nSlices - 1].step;
      const yScale = (step) => stepMax === stepMin ? baselineTop + baseSpan / 2
        : baselineTop + ((step - stepMin) / (stepMax - stepMin)) * baseSpan;
      const sx = vbW / C;
      const sy = sliceH / C;

      if (gridG) gridG.innerHTML = '';

      // Y-axis: step labels (TB shows step numbers on right axis)
      if (yLabels) {
        const nTicks = Math.max(2, Math.min(5, nSlices));
        const labels = [];
        for (let i = 0; i < nTicks; i++) {
          const idx = Math.round(i * (nSlices - 1) / Math.max(1, nTicks - 1));
          labels.push(sorted[idx].step);
        }
        yLabels.innerHTML = labels.map(s =>
          '<span style="text-align:right;font-size:0.65rem;">' + s + '</span>'
        ).join('');
      }

      // Render slices oldest-first (back) → newest-last (front, occludes)
      let html = '';
      for (let si = 0; si < sorted.length; si++) {
        const d = sorted[si];
        if (!d.bins.length) continue;
        const pathD = makePath(d.bins);
        const color = sliceColor(si);
        const baseline = yScale(d.step);
        const ty = baseline - sliceH;

        // Baseline
        html += '<line x1="0" y1="' + sliceH + '" x2="' + vbW + '" y2="' + sliceH + '" ' +
          'transform="translate(0,' + ty.toFixed(2) + ')" ' +
          'stroke="currentColor" stroke-opacity="0.1" vector-effect="non-scaling-stroke" />';

        // Filled outline path (TB: fill=color, stroke=outline 0.5)
        html += '<path d="' + pathD + '" ' +
          'transform="translate(0,' + ty.toFixed(2) + ') scale(' + sx.toFixed(6) + ',' + sy.toFixed(6) + ')" ' +
          'fill="' + color + '" ' +
          'stroke="var(--surface, #1E1F28)" stroke-opacity="0.5" stroke-width="1" ' +
          'vector-effect="non-scaling-stroke" />';
      }
      pathsG.innerHTML = html;
    } else {
      // TB overlay: filled areas (not just strokes), colored by time
      if (gridG) {
        const ticks = _niceYTicks(0, yMax, 5);
        let gl = '';
        for (const t of ticks) {
          const y = (vbH - (t / yMax) * vbH).toFixed(1);
          const isZero = Math.abs(t) < (yMax * 0.001);
          const sw = isZero ? '1.5' : '1';
          const color = isZero ? 'rgba(255,255,255,0.15)' : 'rgba(255,255,255,0.08)';
          gl += '<line x1="0" y1="' + y + '" x2="' + vbW + '" y2="' + y + '" ' +
            'stroke="' + color + '" stroke-width="' + sw + '" vector-effect="non-scaling-stroke" />';
        }
        gridG.innerHTML = gl;
      }
      if (yLabels) {
        const ticks = _niceYTicks(0, yMax, 5);
        yLabels.innerHTML = ticks.map(t =>
          '<span style="text-align:right;">' + _fmtTick(t) + '</span>'
        ).join('');
      }

      let html = '';
      for (let si = 0; si < sorted.length; si++) {
        const d = sorted[si];
        if (!d.bins.length) continue;
        const pathD = makePath(d.bins);
        const color = sliceColor(si);

        // TB overlay: fill=color with opacity, stroke=color (filled areas, not just outlines)
        html += '<path d="' + pathD + '" ' +
          'transform="scale(' + sxFull.toFixed(6) + ',' + syFull.toFixed(6) + ')" ' +
          'fill="' + color + '" fill-opacity="0.7" ' +
          'stroke="' + color + '" stroke-width="1.5" ' +
          'vector-effect="non-scaling-stroke" />';
      }
      pathsG.innerHTML = html;
    }

    // X-axis: nicely formatted bin-value ticks
    if (xLabels) {
      const fmt = (v) => Math.abs(v) >= 1000 ? v.toFixed(0) :
        Math.abs(v) >= 1 ? v.toFixed(2) : v.toPrecision(3);
      const count = 6;
      let html = '';
      for (let i = 0; i < count; i++) {
        const v = xMin + (i / (count - 1)) * (xMax - xMin);
        const left = (i / (count - 1)) * 100;
        html += '<span class="axis-label-row__tick" style="left:' + left.toFixed(1) + '%;">' + fmt(v) + '</span>';
      }
      xLabels.innerHTML = html;
    }

    // Store state for histogram tooltip and wire interaction
    _histTipState = {
      sorted: sorted, xMin: xMin, xMax: xMax, yMax: yMax,
      nSlices: nSlices, sliceColor: sliceColor, mode: _histMode,
      vbW: vbW, vbH: vbH, C: C
    };
    if (_histMode === 'offset') {
      _histTipState.yScale = yScale;
      _histTipState.sliceH = sliceH;
      _histTipState.baselineTop = baselineTop;
      _histTipState.baseSpan = baseSpan;
    }
    _wireHistogramTooltip();
  }

  /**
   * TB-faithful Y-domain (matches LinearScale.niceDomain + computeDataSeriesExtent).
   * P5-P95 percentile indices, PADDING_RATIO=0.05, d3-like nice domain.
   */
  function _computeYDomain(values, gridCount) {
    const finite = values.filter(v => Number.isFinite(v));
    if (finite.length === 0) return { yMin: -1, yMax: 1, yTicks: [-1, 0, 1] };
    const sorted = finite.slice().sort((a, b) => a - b);
    let lo = sorted[0], hi = sorted[sorted.length - 1];
    if (sorted.length > 2) {
      lo = sorted[Math.ceil((sorted.length - 1) * 0.05)];
      hi = sorted[Math.floor((sorted.length - 1) * 0.95)];
    }
    if (hi === lo) {
      if (lo === 0) { lo = -1; hi = 1; }
      else if (lo < 0) { lo = 2 * lo; hi = 0; }
      else { lo = 0; hi = 2 * hi; }
    }
    const PADDING_RATIO = 0.05;
    const padding = (hi - lo + Number.EPSILON) * PADDING_RATIO;
    let domLo = lo - padding, domHi = hi + padding;
    const gc = Math.max(2, gridCount || 6);
    const rawStep = (domHi - domLo) / (gc - 1);
    const exp = Math.floor(Math.log10(rawStep));
    const frac = rawStep / Math.pow(10, exp);
    let niceStep;
    if (frac <= 1) niceStep = 1;
    else if (frac <= 2) niceStep = 2;
    else if (frac <= 5) niceStep = 5;
    else niceStep = 10;
    niceStep *= Math.pow(10, exp);
    const niceMin = Math.floor(domLo / niceStep) * niceStep;
    const niceMax = Math.ceil(domHi / niceStep) * niceStep;
    const ticks = [];
    for (let v = niceMin; v <= niceMax + niceStep * 0.5; v += niceStep) {
      ticks.push(parseFloat(v.toPrecision(12)));
    }
    return { yMin: niceMin, yMax: niceMax, yTicks: ticks };
  }

  function _renderExpandedChart() {
    // Clean up histogram paths if switching from histogram to scalar
    const svg = $('expanded-svg');
    if (svg) { const hp = svg.querySelector('.expanded-hist-paths'); if (hp) hp.innerHTML = ''; }
    const data = _tbScalars[_expandedTag];
    if (!data || data.length < 2) return;
    const tags = _getMiniChartTags();
    const meta = tags[_expandedTag] || { label: _expandedTag, color: 'var(--primary)' };
    const titleEl = $('expanded-chart-title');
    const valueEl = $('expanded-chart-value');
    if (titleEl) titleEl.textContent = meta.label;
    if (valueEl) valueEl.textContent = _fmtVal(data[data.length - 1].value);
    // Update expanded line color
    const eLine = $('expanded-line');
    const eSmoothLine = $('expanded-smooth-line');
    if (eLine) eLine.setAttribute('stroke', meta.color);
    if (eSmoothLine) eSmoothLine.setAttribute('stroke', meta.color);

    const yLabels = $('expanded-y-labels');
    const xLabels = $('expanded-x-labels');
    const gridG = $('expanded-grid-lines');
    if (!svg) return;

    // Slice visible data by index range (zoom model is index-based)
    const totalLen = data.length;
    let startIdx = _expandedViewMin ?? 0, endIdx = _expandedViewMax ?? totalLen;
    startIdx = Math.max(0, Math.round(startIdx));
    endIdx = Math.min(totalLen, Math.round(endIdx));
    if (endIdx - startIdx < 2) startIdx = Math.max(0, endIdx - 2);

    const visData = data.slice(startIdx, endIdx);
    const visValues = visData.map(d => d.value);
    const allValues = data.map(d => d.value);
    const smoothedAll = typeof TrainingChart !== 'undefined'
      ? TrainingChart.expSmooth(allValues, _smoothingWeight) : allValues;
    const smoothedVis = smoothedAll.slice(startIdx, endIdx);

    // Step-based X positioning (TB uses d.step as x accessor)
    const stepMin = visData[0].step;
    const stepMax = visData[visData.length - 1].step;
    const stepSpan = stepMax - stepMin || 1;

    // TB-faithful Y domain: only smoothed (raw is aux, excluded from domain)
    const { yMin, yMax, yTicks: expandedYTicks } = _computeYDomain(smoothedVis, 6);

    const vbW = 1000, vbH = 500;
    svg.setAttribute('viewBox', `0 0 ${vbW} ${vbH}`);
    const toX = (step) => ((step - stepMin) / stepSpan) * vbW;
    const toY = (v) => yMax > yMin ? ((yMax - v) / (yMax - yMin)) * vbH : vbH / 2;

    // Draw lines positioned by step value
    if (eLine) eLine.setAttribute('points', visData.map((d, i) => `${toX(d.step).toFixed(1)},${toY(d.value).toFixed(1)}`).join(' '));
    if (eSmoothLine) eSmoothLine.setAttribute('points', visData.map((d, i) => `${toX(d.step).toFixed(1)},${toY(smoothedVis[i]).toFixed(1)}`).join(' '));

    // Update SVG hover circles (TB uses circle r=4, no stroke, fill=color)
    let hoverG = svg.querySelector('.expanded-hover-circles');
    if (!hoverG) {
      hoverG = document.createElementNS('http://www.w3.org/2000/svg', 'g');
      hoverG.classList.add('expanded-hover-circles');
      hoverG.style.display = 'none';
      svg.appendChild(hoverG);
    }
    // Ensure circles exist
    let hcSmooth = hoverG.querySelector('.hc-smooth');
    if (!hcSmooth) {
      hcSmooth = document.createElementNS('http://www.w3.org/2000/svg', 'circle');
      hcSmooth.classList.add('hc-smooth');
      hcSmooth.setAttribute('r', '4');
      hcSmooth.setAttribute('stroke', 'none');
      hcSmooth.setAttribute('vector-effect', 'non-scaling-stroke');
      hoverG.appendChild(hcSmooth);
    }
    let hcRaw = hoverG.querySelector('.hc-raw');
    if (!hcRaw) {
      hcRaw = document.createElementNS('http://www.w3.org/2000/svg', 'circle');
      hcRaw.classList.add('hc-raw');
      hcRaw.setAttribute('r', '3');
      hcRaw.setAttribute('stroke', 'none');
      hcRaw.setAttribute('opacity', '0.5');
      hcRaw.setAttribute('vector-effect', 'non-scaling-stroke');
      hoverG.appendChild(hcRaw);
    }
    hcSmooth.setAttribute('fill', meta.color);
    hcRaw.setAttribute('fill', meta.color);

    // Store render state for hover lookup
    svg._expandedRender = { data, startIdx, endIdx, stepMin, stepSpan, yMin, yMax, vbW, vbH, smoothedAll, meta };

    // Y-axis ticks from TB domain computation
    if (yLabels) {
      yLabels.innerHTML = expandedYTicks.slice().reverse().map(t =>
        `<span style="text-align:right;">${_fmtAxisLabel(t)}</span>`
      ).join('');
    }
    // X-axis labels (TB X_GRID_COUNT = 10)
    if (xLabels) {
      const count = Math.min(10, visData.length);
      let html = '';
      for (let i = 0; i < count; i++) {
        const frac = count > 1 ? i / (count - 1) : 0;
        const stepVal = stepMin + frac * stepSpan;
        const left = frac * 100;
        const label = Math.round(stepVal).toLocaleString();
        html += `<span class="axis-label-row__tick" style="left:${left.toFixed(1)}%;">${label}</span>`;
      }
      xLabels.innerHTML = html;
    }
    // Grid lines from TB domain ticks
    if (gridG) {
      let html = '';
      const ySpacing = expandedYTicks.length > 1 ? Math.abs(expandedYTicks[1] - expandedYTicks[0]) : 1;
      expandedYTicks.forEach(tick => {
        const y = toY(tick).toFixed(1);
        const isZero = Math.abs(tick) < ySpacing * 0.01;
        const sw = isZero ? '1.5' : '1';
        const color = isZero ? 'rgba(255,255,255,0.15)' : 'rgba(255,255,255,0.08)';
        html += `<line x1="0" y1="${y}" x2="${vbW}" y2="${y}" stroke="${color}" stroke-width="${sw}" vector-effect="non-scaling-stroke" />`;
      });
      // Vertical grid lines (TB X_GRID_COUNT = 10)
      const xCount = Math.min(10, visData.length);
      for (let i = 0; i < xCount; i++) {
        const frac = xCount > 1 ? i / (xCount - 1) : 0;
        const x = (frac * vbW).toFixed(1);
        html += `<line x1="${x}" y1="0" x2="${x}" y2="${vbH}" stroke="rgba(255,255,255,0.08)" stroke-width="1" vector-effect="non-scaling-stroke" />`;
      }
      gridG.innerHTML = html;
    }
  }

  // ---- Histogram tooltip interaction ----
  function _wireHistogramTooltip() {
    const area = $('expanded-series-area');
    if (!area) return;
    const crosshair = $('expanded-crosshair');
    const tooltip = $('expanded-tooltip');

    // Clean up previous histogram tooltip listeners
    if (area._histCleanup) area._histCleanup();

    function onMove(e) {
      const st = _histTipState;
      if (!st) return;
      const rect = area.getBoundingClientRect();
      const px = e.clientX - rect.left;
      const py = e.clientY - rect.top;
      const frac = Math.max(0, Math.min(1, px / rect.width));
      const binVal = st.xMin + frac * (st.xMax - st.xMin);

      // Show crosshair
      if (crosshair) { crosshair.style.display = ''; crosshair.style.left = px + 'px'; }

      // Find density at binVal for a slice's bins
      function getDensity(bins) {
        for (const b of bins) {
          if (binVal >= b.x && binVal < b.x + b.dx) return b.y;
        }
        if (bins.length) {
          const last = bins[bins.length - 1];
          if (binVal >= last.x && binVal <= last.x + last.dx) return last.y;
        }
        return 0;
      }

      const fmtV = (v) => Math.abs(v) >= 1000 ? v.toFixed(0) :
        Math.abs(v) >= 1 ? v.toFixed(2) : v.toPrecision(3);
      const fmtD = (v) => v === 0 ? '0' : Math.abs(v) >= 1000 ? v.toFixed(0) :
        Math.abs(v) >= 0.01 ? v.toFixed(4) : v.toPrecision(3);

      // In offset mode, find the nearest slice by Y proximity
      let highlightIdx = st.nSlices - 1; // default: newest
      if (st.mode === 'offset' && st.yScale) {
        const yFrac = py / rect.height;
        const yVb = yFrac * st.vbH;
        let bestDist = Infinity;
        for (let si = 0; si < st.sorted.length; si++) {
          const baseline = st.yScale(st.sorted[si].step);
          const dist = Math.abs(yVb - baseline);
          if (dist < bestDist) { bestDist = dist; highlightIdx = si; }
        }
      }

      // Build tooltip — newest slices first, highlight closest
      const slices = [...st.sorted].reverse();
      const maxRows = Math.min(6, slices.length);
      let html = '<div style="margin-bottom:4px;font-size:0.8rem;font-weight:600;">Value: ' + fmtV(binVal) + '</div>';
      html += '<table style="border-spacing:4px 2px;font-size:0.75rem;">';
      html += '<tr style="color:var(--muted);font-size:0.65rem;"><td></td><td>Step</td><td style="text-align:right;">Count</td></tr>';

      for (let i = 0; i < maxRows; i++) {
        const d = slices[i];
        const si = st.nSlices - 1 - i;
        const density = getDensity(d.bins);
        const color = st.sliceColor(si);
        const isHL = si === highlightIdx;
        const rowStyle = isHL ? 'font-weight:600;' : 'opacity:0.7;';
        html += '<tr style="' + rowStyle + '">' +
          '<td><span style="display:inline-block;width:8px;height:8px;border-radius:50%;background:' + color + ';' + (isHL ? 'box-shadow:0 0 3px ' + color + ';' : '') + '"></span></td>' +
          '<td style="font-family:var(--font-mono);">' + d.step + '</td>' +
          '<td style="text-align:right;font-family:var(--font-mono);">' + fmtD(density) + '</td>' +
          '</tr>';
      }
      if (slices.length > maxRows) {
        html += '<tr><td colspan="3" style="color:var(--muted);font-size:0.65rem;text-align:center;">+' + (slices.length - maxRows) + ' older</td></tr>';
      }
      html += '</table>';

      if (tooltip) {
        tooltip.innerHTML = html;
        tooltip.style.display = '';
        tooltip.style.bottom = 'auto';
        const tipW = tooltip.offsetWidth || 160;
        const tipH = tooltip.offsetHeight || 100;
        let left = px + 14;
        if (left + tipW > rect.width - 4) left = px - tipW - 14;
        let top = py - tipH / 2;
        top = Math.max(4, Math.min(top, rect.height - tipH - 4));
        tooltip.style.left = left + 'px';
        tooltip.style.top = top + 'px';
      }
    }

    function onLeave() {
      if (crosshair) crosshair.style.display = 'none';
      if (tooltip) tooltip.style.display = 'none';
    }

    area.addEventListener('mousemove', onMove);
    area.addEventListener('mouseleave', onLeave);
    area._histCleanup = () => {
      area.removeEventListener('mousemove', onMove);
      area.removeEventListener('mouseleave', onLeave);
      _histTipState = null;
      if (crosshair) crosshair.style.display = 'none';
      if (tooltip) tooltip.style.display = 'none';
    };
  }

  let _expandedWired = false;
  function _wireExpandedInteraction() {
    // Clean up histogram tooltip if switching to scalar view
    const area = $('expanded-series-area');
    if (area && area._histCleanup) { area._histCleanup(); delete area._histCleanup; }
    if (_expandedWired || typeof ChartInteraction === 'undefined') return;
    _expandedWired = true;
    ChartInteraction.wire({
      areaId: 'expanded-series-area', svgId: 'expanded-svg',
      getView: () => _getExpandedView(),
      setRange: (lo, hi) => { _expandedViewMin = lo; _expandedViewMax = hi; _renderExpandedChart(); },
      zoomReset: () => { _expandedViewMin = null; _expandedViewMax = null; _renderExpandedChart(); },
      minRange: 2, zoomStep: 0.16, enableDoubleClickReset: true,
      formatTip: (idx) => {
        const data = _tbScalars[_expandedTag];
        if (!data || idx < 0 || idx >= data.length) return '#' + idx;
        const d = data[idx];
        const smoothed = typeof TrainingChart !== 'undefined'
          ? TrainingChart.expSmooth(data.map(x => x.value), _smoothingWeight) : null;
        const svg = $('expanded-svg');
        const meta = svg && svg._expandedRender ? svg._expandedRender.meta : { color: 'var(--primary)' };
        let tip = '<table><tr><th colspan="3">Step ' + d.step + '</th></tr>';
        if (smoothed) tip += '<tr><td><span class="chart-tooltip__swatch" style="background:' + meta.color + '"></span></td><td>Smoothed</td><td>' + _fmtVal(smoothed[idx]) + '</td></tr>';
        tip += '<tr><td><span class="chart-tooltip__swatch" style="background:' + meta.color + ';opacity:0.3"></span></td><td>Value</td><td>' + _fmtVal(d.value) + '</td></tr>';
        return tip + '</table>';
      },
      onHover: (idx) => {
        const svg = $('expanded-svg');
        if (!svg || !svg._expandedRender) return;
        const r = svg._expandedRender;
        if (idx < r.startIdx || idx >= r.endIdx) return;
        const d = r.data[idx];
        const x = ((d.step - r.stepMin) / r.stepSpan) * r.vbW;
        const toY = (v) => r.yMax > r.yMin ? ((r.yMax - v) / (r.yMax - r.yMin)) * r.vbH : r.vbH / 2;
        const smoothY = toY(r.smoothedAll[idx]);
        const rawY = toY(d.value);
        // Position SVG hover circles (TB-style: r=4, no stroke)
        const hoverG = svg.querySelector('.expanded-hover-circles');
        if (hoverG) {
          hoverG.style.display = '';
          const hcSmooth = hoverG.querySelector('.hc-smooth');
          const hcRaw = hoverG.querySelector('.hc-raw');
          if (hcSmooth) { hcSmooth.setAttribute('cx', x.toFixed(1)); hcSmooth.setAttribute('cy', smoothY.toFixed(1)); }
          if (hcRaw) { hcRaw.setAttribute('cx', x.toFixed(1)); hcRaw.setAttribute('cy', rawY.toFixed(1)); }
        }
      },
      onLeave: () => {
        const svg = $('expanded-svg');
        if (svg) { const g = svg.querySelector('.expanded-hover-circles'); if (g) g.style.display = 'none'; }
      },
    });
  }

  function _getChartView() {
    const { loss } = _getChartData();
    const len = Math.max(2, loss.length);
    return { totalLen: len, startIdx: _viewXMin ?? 0, endIdx: _viewXMax ?? len };
  }

  function _renderConditioningPanel(info) {
    const panel = $('monitor-conditioning-panel');
    const container = $('monitor-conditioning');
    if (!panel || !container) return;
    panel.style.display = '';
    container.innerHTML = '';
    for (const item of info) {
      const row = document.createElement('div');
      row.style.cssText = 'display:flex;justify-content:space-between;padding:2px 0;';
      const nameSpan = document.createElement('span');
      nameSpan.style.cssText = 'color:var(--muted);overflow:hidden;text-overflow:ellipsis;white-space:nowrap;max-width:60%;';
      nameSpan.textContent = item.name || '?';
      nameSpan.title = item.name || '?';
      const typeSpan = document.createElement('span');
      const isGenre = item.type === 'genre';
      typeSpan.style.cssText = 'font-weight:bold;color:' + (isGenre ? 'var(--accent)' : 'var(--secondary)') + ';';
      typeSpan.textContent = item.type || 'caption';
      row.appendChild(nameSpan);
      row.appendChild(typeSpan);
      container.appendChild(row);
    }
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

    // TensorBoard scalar events from tfevents reader
    if (msg.type === 'tb_scalar') {
      const tag = msg.tag;
      // Tags already fed by progress handler — skip to avoid duplicates
      const _progressTags = new Set(['train/loss', 'train/lr', 'train/epoch_loss']);
      if (_progressTags.has(tag) && _tbScalars[tag] && _tbScalars[tag].length > 0) {
        // Progress already populated this tag; only update main chart if ahead
        if (tag === 'train/loss' && msg.step > _lossHistory.length) {
          _lossHistory.push(msg.value); _loss = msg.value;
        }
        if (tag === 'train/lr' && msg.step > _lrHistory.length) {
          _lrHistory.push(msg.value); _lr = msg.value;
        }
        return;
      }
      if (!_tbScalars[tag]) {
        _tbScalars[tag] = [];
        console.log('[tb_scalar] New tag discovered:', tag);
      }
      _tbScalars[tag].push({ step: msg.step, value: msg.value, wall_time: msg.wall_time });
      // train/loss and train/lr also update the main chart
      if (tag === 'train/loss' && msg.step > _lossHistory.length) {
        _lossHistory.push(msg.value); _loss = msg.value;
      }
      if (tag === 'train/lr' && msg.step > _lrHistory.length) {
        _lrHistory.push(msg.value); _lr = msg.value;
      }
      _renderMiniCharts();
      _updateLossChart();
      return;
    }

    // TensorBoard histogram events from tfevents reader
    if (msg.type === 'tb_histogram') {
      const tag = msg.tag;
      if (!_tbHistograms[tag]) {
        _tbHistograms[tag] = [];
        console.log('[tb_histogram] New tag discovered:', tag);
      }
      _tbHistograms[tag].push({ step: msg.step, wall_time: msg.wall_time, bins: msg.bins });
      _renderMiniCharts();
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

        // Also feed into _tbScalars so mini-charts work immediately
        // (the tfevents reader may lag or not yet be running)
        const wt = Date.now() / 1000;
        if (!_tbScalars['train/loss']) _tbScalars['train/loss'] = [];
        _tbScalars['train/loss'].push({ step: _step, value: _loss, wall_time: wt });
        if (!_tbScalars['train/lr']) _tbScalars['train/lr'] = [];
        _tbScalars['train/lr'].push({ step: _step, value: _lr, wall_time: wt });

        // Cruise control scalars (piped through progress writer)
        if (msg.target_loss_scale != null) {
          if (!_tbScalars['target_loss_scale']) _tbScalars['target_loss_scale'] = [];
          _tbScalars['target_loss_scale'].push({ step: _step, value: msg.target_loss_scale, wall_time: wt });
        }
        if (msg.target_loss_ema != null) {
          if (!_tbScalars['target_loss_ema']) _tbScalars['target_loss_ema'] = [];
          _tbScalars['target_loss_ema'].push({ step: _step, value: msg.target_loss_ema, wall_time: wt });
        }

        // Conditioning info: per-sample genre/caption selection
        if (msg.conditioning_info && Array.isArray(msg.conditioning_info)) {
          _renderConditioningPanel(msg.conditioning_info);
        }
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

        // Feed epoch loss into _tbScalars for mini-chart
        const wt = Date.now() / 1000;
        if (!_tbScalars['train/epoch_loss']) _tbScalars['train/epoch_loss'] = [];
        _tbScalars['train/epoch_loss'].push({ step: _epoch, value: epochAvgLoss, wall_time: wt });
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
          const _bl = _bestLoss < Infinity ? _bestLoss.toFixed(4) : '--';
          _addLog(`[complete]  Training finished. Best loss: ${_bl} at epoch ${_bestEpoch || '--'}`, 'ckpt');
          _addLog('[info]  Waiting for process cleanup...', 'info');
        }
      }

      if (kind === 'fail') {
        _lastFailureMsg = String(msg.msg || 'Training failed');
        _addLog(`[fail]  ${_lastFailureMsg}`, 'fail');
      }

      _renderMiniCharts();
      _updateLossChart();
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

    // Update queue entry status
    const runningEntry = _queue.find(e => e.status === 'running');
    if (runningEntry) {
      runningEntry.status = finalOutcome === 'complete' ? 'done' : finalOutcome === 'stopped' ? 'stopped' : 'failed';
    }

    if (typeof AppState !== 'undefined') {
      const appStatus = finalOutcome === 'complete'
        ? 'complete'
        : finalOutcome === 'failed' ? 'error' : 'idle';
      AppState.setStatus(appStatus);
      API.fetchGPU().then(g => AppState.setGPU(g)).catch(() => {});
    }

    _refreshHistory();

    // Check if there are more queued runs
    const hasPending = _queue.some(e => e.status === 'pending');

    if (wasRunning) _showTerminalToast(finalOutcome);

    if (hasPending) {
      // Auto-chain: start next queued run after a brief delay
      _renderQueue();
      _syncStartButtons();
      if (typeof showToast === 'function') {
        const next = _queue.find(e => e.status === 'pending');
        showToast('Starting queued run: ' + (next?.config?.run_name || 'training'), 'info');
      }
      setTimeout(() => _runNext(), 1500);
    } else {
      // Queue is empty — check if we just finished a multi-run queue
      const finishedCount = _queue.filter(e => e.status === 'done' || e.status === 'failed' || e.status === 'stopped').length;
      if (wasRunning) _showCompletionState(finalOutcome);
      if (finishedCount > 1 && typeof showToast === 'function') {
        showToast('Queue complete -- ' + finishedCount + ' runs finished', 'ok');
      }
      _renderQueue();
      _syncStartButtons();
    }
    _stopRequested = false;
  }

  function _closeWebSockets() {
    if (_ws) { try { _ws.close(); } catch (e) { console.warn('[Training] ws close error:', e); } _ws = null; }
    if (_gpuWs) { try { _gpuWs.close(); } catch (e) { console.warn('[Training] gpu ws close error:', e); } _gpuWs = null; }
  }

  // ---- Queue management -------------------------------------------------

  function _nextQueueId() { return 'tq_' + (++_queueIdCounter); }

  function enqueue(config) {
    const entry = { id: _nextQueueId(), config: { ...config }, status: 'pending' };
    _queue.push(entry);
    _renderQueue();
    _syncStartButtons();
    if (!_running) {
      _runNext();
    } else {
      if (typeof showToast === 'function') showToast('Queued: ' + (config.run_name || 'training'), 'info');
    }
  }

  function _runNext() {
    // Prune finished entries to prevent unbounded growth
    _queue = _queue.filter(e => e.status === 'pending' || e.status === 'running');
    const next = _queue.find(e => e.status === 'pending');
    if (!next) {
      _syncStartButtons();
      _renderQueue();
      return;
    }
    next.status = 'running';
    _renderQueue();
    start(next.config);
  }

  function removeFromQueue(id) {
    _queue = _queue.filter(e => e.id !== id || e.status === 'running');
    _renderQueue();
    _syncStartButtons();
  }

  function clearQueue() {
    _queue = _queue.filter(e => e.status === 'running' || e.status === 'done' || e.status === 'failed' || e.status === 'stopped');
    _renderQueue();
    _syncStartButtons();
    if (typeof showToast === 'function') showToast('Queue cleared', 'info');
  }

  function getQueue() { return _queue.slice(); }
  function queueLength() { return _queue.filter(e => e.status === 'pending').length; }

  function _renderQueue() {
    const panel = $('training-queue-panel');
    const list = $('training-queue-list');
    if (!panel || !list) return;
    const running = _queue.filter(e => e.status === 'running');
    const pending = _queue.filter(e => e.status === 'pending');
    if (running.length === 0 && pending.length === 0) { panel.style.display = 'none'; return; }
    panel.style.display = 'block';
    const rows = [];
    running.forEach(entry => {
      const name = _esc(entry.config.run_name || 'training');
      const ds = _esc(_pathBasename(entry.config.dataset_dir || ''));
      rows.push('<div style="display:flex;align-items:center;gap:var(--space-sm);padding:3px 0;">' +
        '<span class="u-text-success">[...]</span> ' +
        '<span>' + name + '</span>' +
        (ds ? '<span class="u-text-muted" style="font-size:var(--font-size-xs);">' + ds + '</span>' : '') +
        '<span class="u-text-muted" style="margin-left:auto;font-size:var(--font-size-xs);">running</span>' +
        '</div>');
    });
    pending.forEach(entry => {
      const name = _esc(entry.config.run_name || 'training');
      const ds = _esc(_pathBasename(entry.config.dataset_dir || ''));
      rows.push('<div style="display:flex;align-items:center;gap:var(--space-sm);padding:3px 0;">' +
        '<span class="u-text-muted">[--]</span> ' +
        '<span>' + name + '</span>' +
        (ds ? '<span class="u-text-muted" style="font-size:var(--font-size-xs);">' + ds + '</span>' : '') +
        '<button class="btn btn--sm" data-queue-remove="' + entry.id + '" style="margin-left:auto;">[x]</button>' +
        '</div>');
    });
    list.innerHTML = rows.join('');
    list.querySelectorAll('[data-queue-remove]').forEach(btn => {
      btn.addEventListener('click', () => removeFromQueue(btn.dataset.queueRemove));
    });
  }

  const _pathBasename = window._pathBasename || ((p) => String(p || '').split(/[/\\]/).filter(Boolean).pop() || '');

  function _syncStartButtons() {
    const runCount = _queue.filter(e => e.status === 'running').length;
    const pendingCount = _queue.filter(e => e.status === 'pending').length;
    const total = runCount + pendingCount;
    const isActive = _running || runCount > 0;
    const label = isActive ? 'Queue Training' : 'Start Training';
    ['btn-start-ez', 'btn-start-full'].forEach(id => {
      const btn = $(id);
      if (btn) btn.textContent = label;
    });
    // Update queue status indicators on ez/full mode pages
    ['ez-queue-status', 'full-queue-status'].forEach(id => {
      const el = $(id);
      if (!el) return;
      if (total === 0) { el.style.display = 'none'; return; }
      el.style.display = 'block';
      const parts = ['Queue: ' + total];
      if (runCount > 0) parts.push(runCount + ' running');
      if (pendingCount > 0) parts.push(pendingCount + ' waiting');
      el.textContent = parts.join(' -- ');
    });
    // Update Monitor tab badge in topbar
    const monitorTab = document.querySelector('.topbar__mode[data-mode="monitor"]');
    if (monitorTab) {
      let badge = monitorTab.querySelector('.queue-badge');
      if (total > 0) {
        if (!badge) {
          badge = document.createElement('span');
          badge.className = 'queue-badge';
          badge.style.cssText = 'margin-left:var(--space-xs);color:var(--changed);';
          monitorTab.appendChild(badge);
        }
        badge.textContent = '[' + total + ']';
      } else if (badge) {
        badge.remove();
      }
    }
    // Update Stop All button visibility in monitor controls
    const stopAllBtn = $('btn-stop-all');
    if (stopAllBtn) stopAllBtn.style.display = pendingCount > 0 ? '' : 'none';
  }

  // ---- Internal start (called by queue) ---------------------------------

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
    _stopRequested = false; _lastFailureMsg = ''; _finalized = false; _progressComplete = false; _domCache.clear();
    _tbScalars = {}; _tbHistograms = {};
    const miniContainer = $('monitor-mini-charts');
    if (miniContainer) miniContainer.innerHTML = '';
    _maxEpochs = parseInt(_config.epochs) || 100;
    _stepsPerEpoch = parseInt(_config.steps_per_epoch) || 0;
    _startTime = Date.now(); _epochStartTime = Date.now(); _lastEpochDuration = 0;

    // Strip frontend-only keys and zero-means-off fields before sending to backend
    delete _config.steps_per_epoch;
    // Only strip disabled crop fields; chunk_decay_every=0 is still a valid user choice.
    if (_config.chunk_duration !== undefined && (Number(_config.chunk_duration) === 0 || _config.chunk_duration === '0')) delete _config.chunk_duration;
    if (_config.max_latent_length !== undefined && (Number(_config.max_latent_length) === 0 || _config.max_latent_length === '0')) delete _config.max_latent_length;
    // When both crop modes are disabled, chunk_decay_every is irrelevant — strip it too
    if (!_config.chunk_duration && !_config.max_latent_length && _config.chunk_decay_every !== undefined && (Number(_config.chunk_decay_every) === 0 || _config.chunk_decay_every === '0')) delete _config.chunk_decay_every;

    // Reset stop button state from previous run
    const stopBtn = $('btn-stop');
    if (stopBtn) { stopBtn.textContent = 'Stop Training'; stopBtn.disabled = false; }

    // Clear chart SVG from previous run
    ['monitor-loss-line', 'monitor-loss-raw', 'monitor-ma-line', 'monitor-lr-line'].forEach(id => {
      const el = $(id); if (el) el.setAttribute('points', '');
    });
    ['monitor-grid-lines', 'monitor-y-labels', 'monitor-y-labels-right', 'monitor-x-labels'].forEach(id => {
      const el = $(id); if (el) el.innerHTML = '';
    });

    const _show = (id, disp) => { const e = $(id); if (e) e.style.display = disp; };
    _show('monitor-idle', 'none'); _show('monitor-active', 'block'); _show('monitor-controls', 'flex');
    const completion = $('monitor-completion');
    if (completion) { completion.style.display = 'none'; completion.innerHTML = ''; }
    const _el = (id, val) => { const e = $(id); if (e) e.textContent = val; };
    _el('monitor-run-name', _config.run_name || 'training');
    _el('monitor-output-dir', 'Output: ' + (_config.output_dir || ''));
    const log = $('monitor-log'); if (log) log.innerHTML = '';
    const condPanel = $('monitor-conditioning-panel');
    if (condPanel) { condPanel.style.display = 'none'; }
    const condContent = $('monitor-conditioning');
    if (condContent) { condContent.innerHTML = '<div style="color:var(--muted);padding:3px 0;">Waiting for data...</div>'; }
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
    const stopBtn = $('btn-stop');
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
        (done ? '<button class="btn" id="btn-completed-export-comfyui">Export ComfyUI</button>' : '') +
        (!done ? '<button class="btn btn--success" id="btn-completed-resume">Resume from History</button>' : '') +
        '</div></div>';
      completion.style.display = 'block';
      $('btn-new-run')?.addEventListener('click', () => { if (typeof switchMode === 'function') switchMode('ez'); });
      $('btn-completed-open-output')?.addEventListener('click', _openOutputDirFromMonitor);
      $('btn-completed-export-comfyui')?.addEventListener('click', async () => {
        const dir = (_config.output_dir || '') + '/final';
        const btn = $('btn-completed-export-comfyui');
        if (btn) { btn.disabled = true; btn.textContent = 'Exporting...'; }
        try {
          const res = await API.exportComfyUI(dir);
          if (res?.ok) {
            const msg = res.already_compatible
              ? res.adapter_type.toUpperCase() + ' already ComfyUI-compatible'
              : 'Exported: ' + (res.output_path || '').split(/[/\\]/).pop() + ' (' + (res.size_mb || '?') + ' MB)';
            if (typeof showToast === 'function') showToast(msg, 'ok');
          } else {
            if (typeof showToast === 'function') showToast('Export failed: ' + (res?.message || res?.error || 'unknown'), 'error');
          }
        } catch (err) {
          if (typeof showToast === 'function') showToast('Export failed: ' + err.message, 'error');
        } finally {
          if (btn) { btn.disabled = false; btn.textContent = 'Export ComfyUI'; }
        }
      });
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
    _smoothingWeight = Math.max(0, Math.min(weight, 0.999));
    _updateLossChart();
    if (_expandedTag) _renderExpandedChart();
  }

  function setChartMode(mode) {
    _chartMode = mode; // 'step' or 'epoch'
    _viewXMin = null;
    _viewXMax = null;
    _userZoomed = false;
    _updateLossChart();
    // Rebuild mini-charts — epoch/step swap (#4)
    const miniContainer = $('monitor-mini-charts');
    if (miniContainer) miniContainer.innerHTML = '';
    _renderMiniCharts();
  }

  function getChartView() { return _getChartView(); }
  function getChartMode() { return _chartMode; }
  function getSnap() { return _snapToGrid; }
  function setSnap(v) { _snapToGrid = !!v; }
  function collapseExpanded() { if (_expandedTag) _toggleExpandedChart(_expandedTag); }

  function getChartOpts() {
    const { loss, lr } = _getChartData();
    return { fullLoss: loss, fullLr: lr, viewXMin: _viewXMin, viewXMax: _viewXMax, smoothingWeight: _smoothingWeight };
  }

  function getDataAtIndex(idx) {
    const { loss, lr } = _getChartData();
    if (idx < 0 || idx >= loss.length) return null;
    const smoothed = TrainingChart.expSmooth(loss, _smoothingWeight);
    const ma5 = TrainingChart.simpleMA(loss, 5);
    return {
      raw: loss[idx],
      smoothed: smoothed[idx],
      ma5: ma5[idx],
      lr: lr[idx] ?? null,
    };
  }

  function stopAll() {
    clearQueue();
    stop();
  }

  /**
   * Inject fake tb_scalar data to preview all mini-charts and the main chart.
   * Call from console: Training.demoMiniCharts()
   */
  function demoMiniCharts() {
    _tbScalars = {}; _tbHistograms = {};
    _lossHistory = []; _lrHistory = [];
    _epochLossHistory = []; _epochLrHistory = [];
    _finalized = false; _running = true;
    const miniContainer = $('monitor-mini-charts');
    if (miniContainer) miniContainer.innerHTML = '';
    // Show the monitor UI if hidden
    const _show = (id, disp) => { const e = $(id); if (e) e.style.display = disp; };
    _show('monitor-idle', 'none'); _show('monitor-active', 'block'); _show('monitor-controls', 'flex');

    const N = 80;
    const now = Date.now() / 1000;

    // Generate realistic training curves
    for (let i = 0; i < N; i++) {
      const t = i / N;
      const noise = () => (Math.random() - 0.5) * 0.02;
      const step = i + 1;
      const wt = now - (N - i) * 30;

      // train/loss: starts high, decays with noise
      const loss = 0.45 * Math.exp(-3 * t) + 0.08 + noise() * (1 - t * 0.5);
      _onTrainingMessage({ type: 'tb_scalar', tag: 'train/loss', step, value: loss, wall_time: wt });

      // train/lr: warmup then cosine decay
      const lr = t < 0.1 ? 1e-4 * (t / 0.1) : 1e-4 * (0.5 + 0.5 * Math.cos(Math.PI * (t - 0.1) / 0.9));
      _onTrainingMessage({ type: 'tb_scalar', tag: 'train/lr', step, value: lr, wall_time: wt });

      // train/epoch_loss: every ~10 steps
      if (i % 10 === 9) {
        const epochLoss = 0.4 * Math.exp(-3 * t) + 0.09 + noise() * 0.3;
        _onTrainingMessage({ type: 'tb_scalar', tag: 'train/epoch_loss', step: Math.floor(i / 10) + 1, value: epochLoss, wall_time: wt });
      }

      // val_loss: every ~20 steps, slightly above train
      if (i % 20 === 19) {
        const valLoss = 0.42 * Math.exp(-2.5 * t) + 0.11 + noise() * 0.3;
        _onTrainingMessage({ type: 'tb_scalar', tag: 'val_loss', step: Math.floor(i / 20) + 1, value: valLoss, wall_time: wt });
      }

      // target_loss_scale: starts at 1, gradually decreases
      const scale = Math.max(0.3, 1.0 - t * 0.7 + noise() * 0.5);
      _onTrainingMessage({ type: 'tb_scalar', tag: 'target_loss_scale', step, value: scale, wall_time: wt });

      // target_loss_ema: smooth downward trend
      const ema = 0.35 * Math.exp(-2 * t) + 0.1 + noise() * 0.2;
      _onTrainingMessage({ type: 'tb_scalar', tag: 'target_loss_ema', step, value: ema, wall_time: wt });

      // Histogram: timestep distribution — every 5 steps
      if (i % 5 === 4) {
        const bins = [];
        const nBins = 20;
        for (let b = 0; b < nBins; b++) {
          const center = b / nBins;
          // Shift distribution from uniform toward lower timesteps as training progresses
          const mu = 0.5 - t * 0.3;
          const sigma = 0.2;
          const count = Math.max(0, 100 * Math.exp(-((center - mu) ** 2) / (2 * sigma * sigma)) + Math.random() * 10);
          bins.push({ x: b * 50, dx: 50, y: count });
        }
        _onTrainingMessage({ type: 'tb_histogram', tag: 'train/timestep_distribution', step, wall_time: wt, bins });
      }
    }

    // Also populate the main chart if it was empty
    _updateLossChart();
    _updateDOM();
    console.log('[demo] Injected', N, 'fake data points for 6 scalar tags + histograms. Mini-charts should be visible now.');
  }

  /**
   * Animated live demo — streams fake data every 200ms so you can watch
   * the charts update in real-time. Call Training.demoLive() to start;
   * call again to stop.
   */
  function demoLive() {
    if (_demoTimer) { clearInterval(_demoTimer); _demoTimer = null; console.log('[demoLive] Stopped.'); return; }
    // Bootstrap with 20 initial points
    demoMiniCharts();
    let stepCounter = _lossHistory.length;
    let epochCounter = (_tbScalars['train/epoch_loss'] || []).length;
    _demoTimer = setInterval(() => {
      stepCounter++;
      const t = Math.min(1, stepCounter / 300);
      const noise = () => (Math.random() - 0.5) * 0.02;
      const wt = Date.now() / 1000;
      const loss = 0.45 * Math.exp(-3 * t) + 0.08 + noise() * (1 - t * 0.5);
      _onTrainingMessage({ type: 'tb_scalar', tag: 'train/loss', step: stepCounter, value: loss, wall_time: wt });
      const lr = t < 0.05 ? 1e-4 * (t / 0.05) : 1e-4 * (0.5 + 0.5 * Math.cos(Math.PI * (t - 0.05) / 0.95));
      _onTrainingMessage({ type: 'tb_scalar', tag: 'train/lr', step: stepCounter, value: lr, wall_time: wt });
      const scale = Math.max(0.3, 1.0 - t * 0.7 + noise() * 0.5);
      _onTrainingMessage({ type: 'tb_scalar', tag: 'target_loss_scale', step: stepCounter, value: scale, wall_time: wt });
      const ema = 0.35 * Math.exp(-2 * t) + 0.1 + noise() * 0.2;
      _onTrainingMessage({ type: 'tb_scalar', tag: 'target_loss_ema', step: stepCounter, value: ema, wall_time: wt });
      if (stepCounter % 10 === 0) {
        epochCounter++;
        const epochLoss = 0.4 * Math.exp(-3 * t) + 0.09 + noise() * 0.3;
        _onTrainingMessage({ type: 'tb_scalar', tag: 'train/epoch_loss', step: epochCounter, value: epochLoss, wall_time: wt });
      }
      if (stepCounter % 25 === 0) {
        const valLoss = 0.42 * Math.exp(-2.5 * t) + 0.11 + noise() * 0.3;
        _onTrainingMessage({ type: 'tb_scalar', tag: 'val_loss', step: Math.ceil(epochCounter), value: valLoss, wall_time: wt });
      }
      if (stepCounter % 8 === 0) {
        const bins = [];
        for (let b = 0; b < 20; b++) {
          const center = b / 20;
          const mu = 0.5 - t * 0.3;
          const count = Math.max(0, 100 * Math.exp(-((center - mu) ** 2) / 0.08) + Math.random() * 10);
          bins.push({ x: b * 50, dx: 50, y: count });
        }
        _onTrainingMessage({ type: 'tb_histogram', tag: 'train/timestep_distribution', step: stepCounter, wall_time: wt, bins });
      }
      _updateDOM();
    }, 200);
    console.log('[demoLive] Started — call Training.demoLive() again to stop.');
  }

  function init() {
    $('btn-clear-training-queue')?.addEventListener('click', () => clearQueue());
    $('btn-stop-all')?.addEventListener('click', () => stopAll());
  }

  return { init, start, enqueue, stop, stopAll, isRunning, setViewRange, zoomReset, setSmoothing, setChartMode, getChartView, getChartMode, getChartOpts, getDataAtIndex, getSnap, setSnap, collapseExpanded, demoMiniCharts, demoLive, getQueue, queueLength, removeFromQueue, clearQueue };

})();
