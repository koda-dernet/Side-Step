/* Side-Step GUI — Training Chart Rendering (extracted from training.js) */

const TrainingChart = (() => {
  "use strict";
  const $ = (id) => document.getElementById(id);

  function _niceNum(range, round) {
    if (range <= 0) return 1;
    const exp = Math.floor(Math.log10(range));
    const frac = range / Math.pow(10, exp);
    let nice;
    if (round) { nice = frac < 1.5 ? 1 : frac < 3 ? 2 : frac < 7 ? 5 : 10; }
    else { nice = frac <= 1 ? 1 : frac <= 2 ? 2 : frac <= 5 ? 5 : 10; }
    return nice * Math.pow(10, exp);
  }

  function _calcTicks(minVal, maxVal, targetCount) {
    if (maxVal <= minVal) maxVal = minVal + 0.001;
    const range = _niceNum(maxVal - minVal, false);
    const spacing = _niceNum(range / (targetCount - 1), true);
    const niceMin = Math.floor(minVal / spacing) * spacing;
    const niceMax = Math.ceil(maxVal / spacing) * spacing;
    const ticks = [];
    for (let v = niceMin; v <= niceMax + spacing * 0.5; v += spacing) ticks.push(parseFloat(v.toPrecision(6)));
    return { ticks, min: niceMin, max: niceMax };
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

  function expSmooth(data, weight) {
    if (weight <= 0 || data.length === 0) return data.slice();
    const out = [data[0]];
    for (let i = 1; i < data.length; i++) out.push(out[i - 1] * weight + data[i] * (1 - weight));
    return out;
  }

  function simpleMA(data, window) {
    if (data.length === 0) return [];
    return data.map((_, i) => {
      const start = Math.max(0, i - window + 1);
      const slice = data.slice(start, i + 1);
      return slice.reduce((a, b) => a + b, 0) / slice.length;
    });
  }

  /**
   * Render the loss/LR chart SVG.
   * @param {Object} opts - { fullLoss, fullLr, viewXMin, viewXMax, smoothingWeight }
   */
  function render(opts) {
    const { fullLoss, fullLr, viewXMin, viewXMax, smoothingWeight } = opts;
    if (fullLoss.length < 2) return;
    const svg = $('monitor-loss-svg');
    const lossLine = $('monitor-loss-line');
    const rawLine = $('monitor-loss-raw');
    const maLine = $('monitor-ma-line');
    const lrLine = $('monitor-lr-line');
    const gridG = $('monitor-grid-lines');
    const yLabelsEl = $('monitor-y-labels');
    const yLabelsRightEl = $('monitor-y-labels-right');
    const xLabelsEl = $('monitor-x-labels');
    if (!svg || !lossLine) return;

    const totalLen = fullLoss.length;
    let startIdx, endIdx;
    if (viewXMin !== null && viewXMax !== null) {
      startIdx = Math.max(0, Math.round(viewXMin));
      endIdx = Math.min(totalLen, Math.round(viewXMax));
      if (endIdx - startIdx < 2) startIdx = Math.max(0, endIdx - 2);
    } else { startIdx = 0; endIdx = totalLen; }

    const visibleLoss = fullLoss.slice(startIdx, endIdx);
    const visibleLr = fullLr.slice(startIdx, endIdx);
    const sliderSmoothed = expSmooth(fullLoss, smoothingWeight);
    const visibleSliderSmoothed = sliderSmoothed.slice(startIdx, endIdx);
    const ma5Full = simpleMA(fullLoss, 5);
    const visibleMA5 = ma5Full.slice(startIdx, endIdx);

    const allVis = visibleLoss.concat(visibleSliderSmoothed).concat(visibleMA5);
    const rawMin = Math.min(...allVis), rawMax = Math.max(...allVis);
    const yPad = (rawMax - rawMin) * 0.08 || 0.001;
    const { ticks: yTicks, min: yMin, max: yMax } = _calcTicks(rawMin - yPad, rawMax + yPad, 5);

    const lrMin = 0, lrMax = visibleLr.length > 0 ? Math.max(...visibleLr) * 1.15 : 1e-4;
    const vbW = 1000, vbH = 500;
    svg.setAttribute('viewBox', `0 0 ${vbW} ${vbH}`);

    const n = visibleLoss.length;
    const toX = (i) => n > 1 ? (i / (n - 1)) * vbW : vbW / 2;
    const toY = (v) => yMax > yMin ? ((yMax - v) / (yMax - yMin)) * vbH : vbH / 2;
    const toLrY = (v) => lrMax > 0 ? ((lrMax - v) / (lrMax - lrMin)) * vbH : vbH / 2;

    lossLine.setAttribute('points', visibleSliderSmoothed.map((v, i) => `${toX(i).toFixed(1)},${toY(v).toFixed(1)}`).join(' '));
    if (rawLine) rawLine.setAttribute('points', visibleLoss.map((v, i) => `${toX(i).toFixed(1)},${toY(v).toFixed(1)}`).join(' '));
    if (maLine) maLine.setAttribute('points', visibleMA5.map((v, i) => `${toX(i).toFixed(1)},${toY(v).toFixed(1)}`).join(' '));
    if (lrLine) lrLine.setAttribute('points', visibleLr.map((v, i) => `${toX(i).toFixed(1)},${toLrY(v).toFixed(1)}`).join(' '));

    if (gridG) {
      let gridHTML = '';
      yTicks.forEach(tick => {
        const y = toY(tick).toFixed(1);
        gridHTML += `<line x1="0" y1="${y}" x2="${vbW}" y2="${y}" stroke="var(--border)" stroke-width="1" vector-effect="non-scaling-stroke" />`;
      });
      gridG.innerHTML = gridHTML;
    }

    if (yLabelsEl) {
      const precision = yTicks.length > 0 && yTicks[0] < 0.01 ? 4 : 3;
      yLabelsEl.innerHTML = yTicks.slice().reverse().map(tick =>
        `<span style="font-size:11px;font-family:var(--font-mono);color:var(--muted);text-align:right;padding-right:4px;">${tick.toFixed(precision)}</span>`
      ).join('');
    }

    if (yLabelsRightEl && visibleLr.length > 0) {
      const lrTicks = [];
      const lrStep = lrMax / 4;
      for (let i = 0; i <= 4; i++) lrTicks.push(lrMax - i * lrStep);
      yLabelsRightEl.innerHTML = lrTicks.map(v =>
        `<span style="font-size:10px;font-family:var(--font-mono);color:var(--secondary);text-align:left;padding-left:4px;">${v.toExponential(1)}</span>`
      ).join('');
    }

    if (xLabelsEl) {
      // TensorBoard-style axis behavior: choose tick count by available pixels,
      // then place each label at its true fractional position in view.
      const width = svg.getBoundingClientRect().width || 400;
      const loTick = Math.max(0, startIdx);
      const hiTick = Math.max(loTick + 1, endIdx - 1);
      const ticks = _buildXTicks(loTick, hiTick, width);
      xLabelsEl.innerHTML = ticks.map((tick) => {
        const left = ((tick - startIdx) / Math.max(1, endIdx - startIdx - 1)) * 100;
        return `<span class="axis-label-row__tick" style="left:${left.toFixed(3)}%;">${tick}</span>`;
      }).join('');
    }
  }

  return { render, expSmooth, simpleMA };
})();
