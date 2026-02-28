/* ============================================================
   Side-Step GUI — History Table + Compare View
   Populates the run history table and wires the compare view
   with overlaid loss curves and config diff.
   ============================================================ */

const History = (() => {

  const $ = (id) => document.getElementById(id);
  const _esc = window._esc;
  let _runs = [];
  // Compare chart view state
  let _cCurveA = [], _cCurveB = [], _cNameA = '', _cNameB = '';
  let _cViewMin = 0, _cViewMax = 0, _cSmoothing = 0;

  async function loadHistory() {
    try {
      _runs = await API.fetchHistory();
    } catch {
      _runs = [];
    }
    if (typeof AppState !== 'undefined') AppState.setRuns(_runs);

    // Count — distinguish completed from detected-only
    const count = $('history-count');
    if (count) {
      const completed = _runs.filter(r => !r.detected_only).length;
      const detected = _runs.length - completed;
      count.textContent = completed + ' run' + (completed !== 1 ? 's' : '') +
        (detected > 0 ? ' + ' + detected + ' detected' : '');
    }

    // Table
    const tbody = $('history-tbody');
    if (!tbody) return;
    tbody.innerHTML = '';

    if (_runs.length === 0) {
      tbody.innerHTML = '<tr><td colspan="8" class="data-table-empty">No training runs found. Start training to see your history here.</td></tr>';
      _populateCompareSelectors();
      return;
    }

    _runs.forEach((r, idx) => {
      const tr = document.createElement('tr');
      const detectedOnly = !!r.detected_only;
      const statusClass = detectedOnly
        ? 'history-detected-chip'
        : (r.status === 'complete' ? 'status--ok' : r.status === 'stopped' ? 'status--warn' : 'status--fail');
      const statusLabel = detectedOnly
        ? 'DNF'
        : (r.artifact_source ? `${r.status} (${r.artifact_source})` : r.status);
      const bestLoss = typeof r.best_loss === 'number' ? r.best_loss.toFixed(4) : '--';
      const actions = detectedOnly
        ? `<button class="btn btn--danger btn--sm history-delete-chip" data-action="delete-detected" data-path="${_esc(r.path)}">Delete</button>`
        : '<span class="u-text-muted">--</span>';
      tr.dataset.idx = idx;
      tr.dataset.detected = detectedOnly ? '1' : '0';
      tr.classList.toggle('history-row--detected', detectedOnly);
      tr.innerHTML = `
        <td>${_esc(r.run_name)}</td>
        <td>${_esc(r.adapter)}</td>
        <td>${_esc(r.model)}</td>
        <td>${r.epochs}</td>
        <td>${bestLoss}</td>
        <td>${_esc(r.duration)}</td>
        <td><span class="${statusClass}" title="${detectedOnly ? 'Did Not Finish — detected folder at: ' + _esc(r.path || '') : _esc(r.path || '')}">${_esc(statusLabel)}</span></td>
        <td>${actions}</td>
      `;
      tbody.appendChild(tr);
    });

    tbody.querySelectorAll('[data-action="delete-detected"]').forEach((btn) => {
      btn.addEventListener('click', (e) => {
        e.stopPropagation();
        const path = btn.dataset.path || '';
        if (!path) return;
        const _doDelete = async () => {
          const res = await API.deleteHistoryFolder(path);
          if (!res?.ok) {
            if (typeof showToast === 'function') showToast('Failed to delete folder: ' + (res?.error || 'unknown error'), 'error');
            return;
          }
          if (typeof showToast === 'function') showToast('Detected folder deleted', 'ok');
          document.dispatchEvent(new CustomEvent('sidestep:history-updated'));
          await loadHistory();
        };
        if (typeof WorkspaceBehaviors !== 'undefined' && WorkspaceBehaviors.showConfirmModal) {
          WorkspaceBehaviors.showConfirmModal('Delete Folder', 'Delete detected folder recursively?<br><br><code style="font-size:var(--font-size-xs);color:var(--accent);word-break:break-all;display:block;padding:var(--space-xs);background:var(--surface);border-radius:var(--radius);margin-top:var(--space-xs);">' + _esc(path) + '</code>', 'Delete', _doDelete);
        } else { _doDelete(); }
      });
    });

    // Populate compare selectors
    _populateCompareSelectors();
  }

  function _populateCompareSelectors() {
    const selA = $('compare-run-a');
    const selB = $('compare-run-b');
    if (!selA || !selB) return;

    const compareRuns = _runs.filter((r) => !r.detected_only);

    selA.innerHTML = '<option value="">Select run...</option>';
    selB.innerHTML = '<option value="">Select run...</option>';

    compareRuns.forEach(r => {
      const opt = `<option value="${_esc(r.run_name)}">${_esc(r.run_name)} (${_esc(r.adapter)}, ${r.best_loss != null ? r.best_loss.toFixed(4) : '--'})</option>`;
      selA.innerHTML += opt;
      selB.innerHTML += opt;
    });

    // Pre-select first two if available
    if (compareRuns.length >= 2) {
      selA.value = compareRuns[0].run_name;
      selB.value = compareRuns[1].run_name;
      _runCompare();
    }
  }

  async function _runCompare() {
    const nameA = $('compare-run-a')?.value;
    const nameB = $('compare-run-b')?.value;
    if (!nameA || !nameB) return;

    // Use AppState for deterministic curves + cached configs
    const hasAS = typeof AppState !== 'undefined';
    const [configA, configB, curveA, curveB] = await Promise.all([
      API.fetchRunConfig(nameA), API.fetchRunConfig(nameB),
      API.fetchRunLossCurve(nameA), API.fetchRunLossCurve(nameB),
    ]);
    if (hasAS) { AppState.setRunConfig(nameA, configA); AppState.setRunConfig(nameB, configB); }

    _renderDiff(configA, configB);
    _cCurveA = _normalizeCurve(curveA); _cCurveB = _normalizeCurve(curveB); _cNameA = nameA; _cNameB = nameB;
    _cViewMin = 0; _cViewMax = Math.max(_cCurveA.length, _cCurveB.length);
    _renderOverlaidCurves();
    _renderSummary(configA, configB, nameA, nameB);
  }

  function _renderSummary(cfgA, cfgB, nameA, nameB) {
    const summary = $('compare-summary');
    if (!summary || !cfgA || !cfgB) return;

    const diffs = [];
    const keys = new Set([...Object.keys(cfgA), ...Object.keys(cfgB)]);
    keys.forEach(k => {
      if (String(cfgA[k]) !== String(cfgB[k])) {
        diffs.push(k);
      }
    });

    if (diffs.length === 0) {
      summary.innerHTML = '<span class="u-text-success">Identical configs</span>';
    } else {
      summary.innerHTML = `<span class="u-text-changed">${diffs.length} parameter${diffs.length > 1 ? 's' : ''} differ:</span> ` +
        diffs.map(k => `<code class="u-code-chip u-text-changed">${_esc(k)}</code>`).join(' ');
    }
  }

  function _renderDiff(cfgA, cfgB) {
    const tbody = $('compare-diff-tbody');
    if (!tbody) return;
    tbody.innerHTML = '';

    if (!cfgA || !cfgB) {
      tbody.innerHTML = '<tr><td colspan="3" class="u-text-muted">Select two runs to compare</td></tr>';
      return;
    }

    const keys = new Set([...Object.keys(cfgA), ...Object.keys(cfgB)]);
    const sorted = [...keys].sort();

    sorted.forEach(k => {
      const vA = cfgA[k] !== undefined ? String(cfgA[k]) : '--';
      const vB = cfgB[k] !== undefined ? String(cfgB[k]) : '--';
      const isDiff = vA !== vB;

      const tr = document.createElement('tr');
      const diffCls = isDiff ? 'u-text-changed u-text-bold' : '';
      tr.innerHTML = `
        <td>${_esc(k)}</td>
        <td class="${diffCls}">${_esc(vA)}</td>
        <td class="${diffCls}">${_esc(vB)}</td>
      `;
      tbody.appendChild(tr);
    });
  }

  function _expSmooth(arr, w) {
    if (w <= 0 || !arr.length) return arr;
    const out = [arr[0]]; let last = arr[0];
    for (let i = 1; i < arr.length; i++) { last = w * last + (1 - w) * arr[i]; out.push(last); }
    return out;
  }

  function _normalizeCurve(rawCurve) {
    const out = [];
    (rawCurve || []).forEach((point) => {
      const loss = Number(point?.loss);
      if (!Number.isFinite(loss)) return;
      const stepNum = Number(point?.step);
      const epochNum = Number(point?.epoch);
      out.push({
        loss,
        step: Number.isFinite(stepNum) ? stepNum : null,
        epoch: Number.isFinite(epochNum) ? epochNum : null,
      });
    });
    return out;
  }

  function _niceNum(range, round) {
    if (!Number.isFinite(range) || range <= 0) return 1;
    const exp = Math.floor(Math.log10(range));
    const frac = range / Math.pow(10, exp);
    let nice;
    if (round) nice = frac < 1.5 ? 1 : frac < 3 ? 2 : frac < 7 ? 5 : 10;
    else nice = frac <= 1 ? 1 : frac <= 2 ? 2 : frac <= 5 ? 5 : 10;
    return nice * Math.pow(10, exp);
  }

  function _buildXTicks(lo, hi, widthPx) {
    const range = Math.max(1, hi - lo);
    const targetCount = Math.max(2, Math.floor((widthPx || 400) / 72));
    const rawStep = _niceNum(range / Math.max(1, targetCount - 1), true);
    const step = Math.max(1, Math.round(rawStep));
    const first = Math.ceil(lo / step) * step;
    const ticks = [];
    for (let t = first; t <= hi; t += step) ticks.push(t);
    if (!ticks.length || ticks[0] !== lo) ticks.unshift(lo);
    if (ticks[ticks.length - 1] !== hi) ticks.push(hi);
    return [...new Set(ticks.filter((t) => Number.isFinite(t)))].sort((a, b) => a - b);
  }

  function _seriesInView(curve, lo, hi) {
    if (!curve.length) return { idx: [], val: [] };
    const start = Math.max(0, Math.floor(lo));
    const end = Math.min(curve.length, Math.ceil(hi));
    const idx = [];
    const val = [];
    for (let i = start; i < end; i++) {
      const point = curve[i];
      const loss = Number(point?.loss);
      if (!Number.isFinite(loss)) continue;
      idx.push(i);
      val.push(loss);
    }
    return { idx, val };
  }

  function _renderOverlaidCurves() {
    const lineA = $('compare-line-a'), lineB = $('compare-line-b');
    const rawA = $('compare-raw-a'), rawB = $('compare-raw-b');
    const endA = $('compare-end-a'), endB = $('compare-end-b');
    const legendA = $('compare-legend-a'), legendB = $('compare-legend-b');
    if (!lineA || !lineB) return;
    if (legendA) { legendA.textContent = _cNameA; legendA.title = _cNameA; }
    if (legendB) { legendB.textContent = _cNameB; legendB.title = _cNameB; }

    const totalLen = Math.max(1, _cCurveA.length, _cCurveB.length);
    const lo = Math.max(0, _cViewMin);
    const hi = Math.max(lo + 1, Math.min(totalLen, _cViewMax));

    const rawSeriesA = _seriesInView(_cCurveA, lo, hi);
    const rawSeriesB = _seriesInView(_cCurveB, lo, hi);
    const smoothFullA = _expSmooth(_cCurveA.map((p) => Number(p?.loss) || 0), _cSmoothing);
    const smoothFullB = _expSmooth(_cCurveB.map((p) => Number(p?.loss) || 0), _cSmoothing);
    const smoothSeriesA = _seriesInView(smoothFullA.map((loss) => ({ loss })), lo, hi);
    const smoothSeriesB = _seriesInView(smoothFullB.map((loss) => ({ loss })), lo, hi);

    const all = [
      ...rawSeriesA.val, ...rawSeriesB.val,
      ...smoothSeriesA.val, ...smoothSeriesB.val,
    ];
    if (!all.length) return;
    const rawMax = Math.max(...all), rawMin = Math.min(...all);
    const yPad = (rawMax - rawMin) * 0.08 || 0.001;
    const maxLoss = rawMax + yPad, minLoss = Math.max(0, rawMin - yPad);

    const w = 400, h = 160, pad = 4;
    const toY = (v) => pad + ((maxLoss - v) / (maxLoss - minLoss)) * (h - 2 * pad);
    const toX = (idx) => pad + ((idx - lo) / Math.max(1, hi - lo)) * (w - 2 * pad);
    const makePts = (idx, val) => idx.map((epoch, i) => {
      const x = toX(epoch);
      return x.toFixed(1) + ',' + toY(val[i]).toFixed(1);
    }).join(' ');

    lineA.setAttribute('points', smoothSeriesA.idx.length > 1 ? makePts(smoothSeriesA.idx, smoothSeriesA.val) : '');
    lineB.setAttribute('points', smoothSeriesB.idx.length > 1 ? makePts(smoothSeriesB.idx, smoothSeriesB.val) : '');
    if (rawA) rawA.setAttribute('points', _cSmoothing > 0 && rawSeriesA.idx.length > 1 ? makePts(rawSeriesA.idx, rawSeriesA.val) : '');
    if (rawB) rawB.setAttribute('points', _cSmoothing > 0 && rawSeriesB.idx.length > 1 ? makePts(rawSeriesB.idx, rawSeriesB.val) : '');

    const _setEndMarker = (el, runLen) => {
      if (!el || runLen <= 0) return;
      const idx = runLen - 1;
      if (idx < lo || idx > hi) {
        el.style.display = 'none';
        return;
      }
      const x = toX(idx).toFixed(1);
      el.setAttribute('x1', x);
      el.setAttribute('x2', x);
      el.style.display = '';
    };
    _setEndMarker(endA, _cCurveA.length);
    _setEndMarker(endB, _cCurveB.length);

    // Y-axis labels
    const yLabels = $('compare-y-labels');
    if (yLabels) {
      const prec = maxLoss < 0.01 ? 4 : 3;
      yLabels.innerHTML = [maxLoss, (maxLoss + minLoss) / 2, minLoss].map(v =>
        '<span class="u-axis-label">' + v.toFixed(prec) + '</span>'
      ).join('');
    }
    // X-axis labels
    const xLabels = $('compare-x-labels');
    if (xLabels) {
      const loTick = Math.max(0, Math.floor(lo));
      const hiTick = Math.max(loTick + 1, Math.ceil(hi));
      const width = $('compare-loss-svg')?.getBoundingClientRect?.().width || 400;
      const ticks = _buildXTicks(loTick, hiTick, width);
      xLabels.innerHTML = ticks.map((t) => {
        const left = ((t - lo) / Math.max(1, hi - lo)) * 100;
        return `<span class="axis-label-row__tick" style="left:${left.toFixed(3)}%;">${t}</span>`;
      }).join('');
    }
  }

  // Compare chart view API for ChartInteraction
  function _getCompareView() {
    return {
      startIdx: _cViewMin,
      endIdx: _cViewMax,
      totalLen: Math.max(1, _cCurveA.length, _cCurveB.length),
    };
  }
  function _setCompareRange(lo, hi) {
    const total = Math.max(1, _cCurveA.length, _cCurveB.length);
    const minRange = Math.min(16, total);
    let start = Math.max(0, Number(lo) || 0);
    let end = Math.min(total, Number(hi) || total);
    if (end - start < minRange) {
      const mid = (start + end) / 2;
      start = Math.max(0, mid - minRange / 2);
      end = Math.min(total, start + minRange);
      start = Math.max(0, end - minRange);
    }
    _cViewMin = start;
    _cViewMax = end;
    _renderOverlaidCurves();
  }
  function _compareZoomReset() { _cViewMin = 0; _cViewMax = Math.max(1, _cCurveA.length, _cCurveB.length); _renderOverlaidCurves(); }
  function _compareFormatTip(idx) {
    const a = _cCurveA[idx], b = _cCurveB[idx];
    const _meta = (p) => `ep ${p?.epoch ?? '--'} · step ${p?.step ?? '--'}`;
    let h = '<span class="u-text-muted">idx ' + idx + '</span>';
    if (a) h += '<br><span class="u-text-primary">' + a.loss.toFixed(4) + '</span> <span class="u-text-muted">' + _meta(a) + '</span>';
    if (b) h += '<br><span class="u-text-warning">' + b.loss.toFixed(4) + '</span> <span class="u-text-muted">' + _meta(b) + '</span>';
    return h;
  }

  function _updateSelectedCount() {
    const count = document.querySelectorAll('#history-tbody tr.selected:not([data-detected="1"])').length;
    const badge = $('history-selected-count');
    if (badge) badge.textContent = count > 0 ? '(' + count + ' selected)' : '';
    const compareBtn = $('btn-compare-selected');
    const resumeBtn = $('btn-resume-selected');
    if (compareBtn) compareBtn.disabled = count < 2;
    if (resumeBtn) resumeBtn.disabled = count < 1;
  }

  function compareSelected() {
    const rows = [...document.querySelectorAll('#history-tbody tr.selected')]
      .filter((r) => r.dataset.detected !== '1');
    const indices = rows.map(r => parseInt(r.dataset.idx));

    if (indices.length < 2) {
      if (typeof showToast === 'function') showToast('Select exactly 2 runs to compare', 'warn');
      return;
    }

    if (indices.length > 2) {
      if (typeof showToast === 'function') showToast('Compare uses exactly 2 runs. Using first 2 selected.', 'warn');
      rows.slice(2).forEach(r => r.classList.remove('selected'));
      _updateSelectedCount();
    }

    const selA = $('compare-run-a');
    const selB = $('compare-run-b');
    if (selA) selA.value = _runs[indices[0]].run_name;
    if (selB) selB.value = _runs[indices[1]].run_name;

    const inlinePanel = $('history-compare-panel');
    if (inlinePanel) {
      inlinePanel.style.display = 'block';
      inlinePanel.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }

    _runCompare();
  }

  function init() {
    loadHistory();
    document.addEventListener('sidestep:history-updated', () => { loadHistory(); });
    document.addEventListener('sidestep:settings-saved', () => { loadHistory(); });

    // Compare selectors
    $('compare-run-a')?.addEventListener('change', _runCompare);
    $('compare-run-b')?.addEventListener('change', _runCompare);

    // Compare selected button
    $('btn-compare-selected')?.addEventListener('click', compareSelected);

    // Close compare
    $('btn-close-compare')?.addEventListener('click', () => {
      const p = $('history-compare-panel'); if (p) p.style.display = 'none';
    });

    // Compare smoothing slider
    const slider = $('compare-smoothing');
    const sVal = $('compare-smoothing-val');
    if (slider) slider.addEventListener('input', () => {
      _cSmoothing = parseInt(slider.value, 10) / 100;
      if (sVal) sVal.textContent = _cSmoothing.toFixed(2);
      _renderOverlaidCurves();
    });

    // Compare zoom reset
    $('btn-compare-zoom-reset')?.addEventListener('click', _compareZoomReset);

    // Wire ChartInteraction for compare chart
    if (typeof ChartInteraction !== 'undefined') {
      ChartInteraction.wire({
        areaId: 'compare-chart-area', svgId: 'compare-loss-svg',
        getView: _getCompareView, setRange: _setCompareRange,
        zoomReset: _compareZoomReset, formatTip: _compareFormatTip,
        minRange: 16,
        zoomStep: 0.16,
        panModifier: 'shift',
        enableDoubleClickReset: true,
      });
    }

    // Update count when selection changes (via MutationObserver on class changes)
    const htbody = $('history-tbody');
    if (htbody) {
      new MutationObserver(_updateSelectedCount)
        .observe(htbody, { attributes: true, attributeFilter: ['class'], subtree: true });
    }
  }

  return { init, loadHistory, compareSelected };

})();
