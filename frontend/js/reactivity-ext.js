/* Side-Step GUI — Reactivity Extensions (PP++ validation, adapter lock, dataset preflight, formula preview) */

const ReactivityExt = (() => {
  'use strict';
  const $ = (id) => document.getElementById(id);

  function _showWarning(id, msg) {
    const el = $(id); if (!el) return;
    el.textContent = msg; el.style.display = msg ? 'block' : 'none';
  }

  /* ---- PP++ Cross-Validation ---- */
  function initPPPlusCrossValidation() {
    const baseRank = $('ppplus-base-rank'), rankMin = $('ppplus-rank-min'), rankMax = $('ppplus-rank-max');
    const warnMin = $('warn-ppplus-rank-min'), warnMax = $('warn-ppplus-rank-max');
    const warnOrder = $('warn-ppplus-rank-order'), warnSmall = $('warn-ppplus-small-dataset');
    const timeEst = $('ppplus-time-estimate'), datasetSel = $('ppplus-dataset-dir');
    if (!rankMin || !rankMax) return;

    const check = () => {
      const min = parseInt(rankMin.value, 10) || 0;
      const max = parseInt(rankMax.value, 10) || 0;
      const base = parseInt(baseRank?.value, 10) || 0;
      if (warnMin) { warnMin.textContent = min < 4 ? 'Rank min should be at least 4' : ''; warnMin.style.display = min < 4 ? '' : 'none'; }
      if (warnMax) { warnMax.textContent = max > 512 ? 'Extreme rank \u2014 VRAM usage will be very high' : ''; warnMax.style.display = max > 512 ? '' : 'none'; }
      if (warnOrder) {
        const bad = min > max || (base > 0 && (base < min || base > max));
        warnOrder.textContent = bad ? 'Rank order violated: need rank_min \u2264 base_rank \u2264 rank_max' : '';
        warnOrder.style.display = bad ? '' : 'none';
      }
    };
    [baseRank, rankMin, rankMax].forEach(el => { el?.addEventListener('input', check); el?.addEventListener('change', check); });

    const checkDataset = () => {
      const detectEl = $('ppplus-dataset-detect');
      const text = detectEl?.textContent || '';
      const match = text.match(/(\d+)\s*(\.pt|tensor|sample)/i);
      const count = match ? parseInt(match[1], 10) : 0;
      if (warnSmall) {
        warnSmall.textContent = count > 0 && count < 5 ? 'Very few samples (' + count + ') \u2014 Fisher results may be unreliable' : '';
        warnSmall.style.display = count > 0 && count < 5 ? '' : 'none';
      }
      if (timeEst) { timeEst.textContent = count > 0 ? 'Estimated time: ~' + (count * 0.12).toFixed(1) + ' min (' + count + ' samples)' : ''; }
    };
    datasetSel?.addEventListener('change', checkDataset);
    const detectEl = $('ppplus-dataset-detect');
    if (detectEl) new MutationObserver(checkDataset).observe(detectEl, { childList: true, characterData: true, subtree: true });
  }

  /* ---- PP++ Attention/MLP Lock ---- */
  function initPPPlusAdapterLock() {
    const attnSel = $('full-attention-type'), mlpToggle = $('full-target-mlp');
    if (!attnSel || !mlpToggle) return;
    const lock = () => {
      const ppStatus = $('full-pp-status');
      const ppToggle = $('full-use-ppplus');
      const hasPP = ppStatus && ppStatus.textContent.includes('[ok]') && (!ppToggle || ppToggle.checked);
      attnSel.disabled = hasPP; attnSel.style.opacity = hasPP ? '0.5' : '';
      mlpToggle.disabled = hasPP; mlpToggle.closest('.toggle').style.opacity = hasPP ? '0.5' : '';
      if (hasPP) { attnSel.value = 'both'; mlpToggle.checked = true; }
      const hint = $('pp-adapter-lock-hint'); if (hint) hint.style.display = hasPP ? '' : 'none';
    };
    // Observe parent container for dynamic full-pp-status changes
    const infoEl = $('full-dataset-info');
    if (infoEl) new MutationObserver(lock).observe(infoEl, { childList: true, characterData: true, subtree: true });
    $('full-use-ppplus')?.addEventListener('change', lock);
    $('full-dataset-dir')?.addEventListener('change', () => setTimeout(lock, 400));
    lock();
  }

  /* ---- Dataset Preflight ---- */
  function initDatasetPreflight() {
    const fullDs = $('full-dataset-dir'), fullDetect = $('full-dataset-info');
    if (!fullDs || !fullDetect) return;
    fullDs.addEventListener('change', () => {
      const v = fullDs.value.trim(); if (!v) return;
      const looksValid = v.includes('tensor') || v.includes('.pt') || v.endsWith('/');
      if (!looksValid) fullDetect.innerHTML = '<div class="u-text-warning">Path may not be a preprocessed tensor directory</div>';
    });
  }

  /* ---- Formula Visualizer ---- */
  function initFormulaPreview() {
    const input = $('full-scheduler-formula'), preview = $('formula-preview');
    const line = $('formula-preview-line'), yMaxLabel = $('formula-y-max');
    if (!input || !preview || !line) return;

    const SAFE_FNS = { cos: Math.cos, sin: Math.sin, exp: Math.exp, log: Math.log, sqrt: Math.sqrt, pow: Math.pow, min: Math.min, max: Math.max, abs: Math.abs, pi: Math.PI, e: Math.E, clamp: (v, lo, hi) => Math.min(Math.max(v, lo), hi) };

    function _py2js(src) {
      let s = src;
      s = s.replace(/([a-zA-Z0-9_.\)]+)\s*\*\*\s*([a-zA-Z0-9_.\(]+)/g, 'pow($1, $2)');
      s = s.replace(/([a-zA-Z0-9_.\)]+)\s*\/\/\s*([a-zA-Z0-9_.\(]+)/g, 'Math.floor($1 / $2)');
      s = s.replace(/(.+?)\s+if\s+(.+?)\s+else\s+(.+)/g, '(($2) ? ($1) : ($3))');
      s = s.replace(/\bTrue\b/g, 'true').replace(/\bFalse\b/g, 'false');
      return s;
    }

    let _debounce = null;
    const warnEl = $('warn-scheduler-formula');
    const render = () => {
      const raw = input.value.trim();
      if (!raw) { preview.style.display = 'none'; if (warnEl) warnEl.style.display = 'none'; return; }
      const expr = _py2js(raw.replace(/^\s*lr\s*=\s*/, ''));
      const baseLr = parseFloat($('full-lr')?.value) || 1e-4;
      try {
        if (/\b(while|for|do|eval|Function|import|require|fetch|XMLHttp)\b/.test(expr)) throw new Error('Blocked keyword');
        if (expr.length > 500) throw new Error('Formula too long');
        const fn = new Function('step', 'total_steps', 'progress', 'epoch', 'base_lr', ...Object.keys(SAFE_FNS), 'return (' + expr + ')');
        const pts = []; let yMax = 0;
        const t0 = performance.now();
        for (let i = 0; i <= 100; i++) {
          if (performance.now() - t0 > 200) throw new Error('Preview timeout');
          const p = i / 100;
          const y = fn(Math.round(p * 1000), 1000, p, Math.round(p * 100), baseLr, ...Object.values(SAFE_FNS));
          if (typeof y !== 'number' || !isFinite(y)) throw new Error('NaN');
          pts.push(y); if (y > yMax) yMax = y;
        }
        if (yMax <= 0) yMax = baseLr;
        const w = 200, h = 120;
        line.setAttribute('points', pts.map((v, i) => ((i / 100) * w).toFixed(1) + ',' + (h - (v / yMax) * h).toFixed(1)).join(' '));
        if (yMaxLabel) yMaxLabel.textContent = yMax < 0.001 ? yMax.toExponential(1) : yMax.toFixed(5);
        const yMidLabel = $('formula-y-mid');
        if (yMidLabel) { const mid = yMax / 2; yMidLabel.textContent = mid < 0.001 ? mid.toExponential(1) : mid.toFixed(5); }
        // Update epoch/step axis labels
        const estEl = $('full-step-estimate-text');
        const epochs = parseInt(estEl?.dataset?.epochs || $('full-epochs')?.value, 10) || 100;
        const total = parseInt(estEl?.dataset?.total, 10) || 0;
        const ep25 = $('formula-epoch-25'), ep50 = $('formula-epoch-50'), ep75 = $('formula-epoch-75'), ep100 = $('formula-epoch-100');
        if (ep25) ep25.textContent = Math.round(epochs * 0.25);
        if (ep50) ep50.textContent = Math.round(epochs * 0.5);
        if (ep75) ep75.textContent = Math.round(epochs * 0.75);
        if (ep100) ep100.textContent = 'ep ' + epochs;
        const topLabel = $('formula-top-label');
        if (topLabel) topLabel.textContent = total ? 'LR over ' + total + ' steps' : 'LR over training (load dataset for step count)';
        preview.style.display = ''; if (warnEl) warnEl.style.display = 'none';
      } catch (e) {
        preview.style.display = 'none';
        if (warnEl) { warnEl.textContent = 'Preview error: ' + e.message; warnEl.style.display = ''; }
      }
    };

    input.addEventListener('input', () => { clearTimeout(_debounce); _debounce = setTimeout(render, 200); });
    $('formula-template')?.addEventListener('change', (e) => { if (e.target.value) { input.value = e.target.value; render(); } });
    $('full-lr')?.addEventListener('input', () => { if (input.value.trim()) render(); });
  }

  function init() {
    initPPPlusCrossValidation();
    initPPPlusAdapterLock();
    initDatasetPreflight();
    initFormulaPreview();
  }

  return { init };
})();
