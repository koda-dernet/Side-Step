/* Side-Step GUI — Training Chart Rendering (extracted from training.js) */

const TrainingChart = (() => {
  "use strict";

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

  /**
   * TB-faithful Y-domain computation (matches LinearScale.niceDomain + computeDataSeriesExtent).
   * 1) P5-P95 percentile indices: ceil((n-1)*0.05), floor((n-1)*0.95)
   * 2) PADDING_RATIO = 0.05
   * 3) d3-like nice domain (round to clean tick boundaries)
   * Returns { yMin, yMax, yTicks }
   */
  function _tbYDomain(values, gridCount) {
    const finite = values.filter(v => Number.isFinite(v));
    if (finite.length === 0) return { yMin: -1, yMax: 1, yTicks: [-1, 0, 1] };
    const sorted = finite.slice().sort((a, b) => a - b);
    let lo = sorted[0], hi = sorted[sorted.length - 1];
    // TB ignoreYOutliers: use P5-P95 when length > 2
    if (sorted.length > 2) {
      lo = sorted[Math.ceil((sorted.length - 1) * 0.05)];
      hi = sorted[Math.floor((sorted.length - 1) * 0.95)];
    }
    // TB niceDomain logic (LinearScale.niceDomain from scale.ts)
    if (hi === lo) {
      if (lo === 0) { lo = -1; hi = 1; }
      else if (lo < 0) { lo = 2 * lo; hi = 0; }
      else { lo = 0; hi = 2 * hi; }
    }
    const PADDING_RATIO = 0.05;
    const padding = (hi - lo + Number.EPSILON) * PADDING_RATIO;
    let domLo = lo - padding, domHi = hi + padding;
    // d3-like nice: round to clean tick boundaries
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

  /** Format axis tick label (TB-style: auto precision) */
  function _fmtAxisTick(v) {
    const abs = Math.abs(v);
    if (abs === 0) return '0';
    if (abs >= 1e6 || abs < 0.001) return v.toExponential(1);
    if (abs >= 100) return v.toFixed(0);
    if (abs >= 1) return v.toFixed(2);
    return v.toPrecision(3);
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

  /**
   * TensorBoard-compatible debiased exponential moving average.
   * 1st-order IIR low-pass filter (classicSmoothing from data_transformer.ts).
   * - NaN/Infinity pass through unchanged and don't affect the accumulator.
   * - Constant series pass through unchanged (avoids FP noise, see TB #786).
   * - Debiasing uses numAccum (count of finite values), not index.
   */
  function expSmooth(data, weight) {
    if (data.length === 0) return [];
    weight = Math.max(0, Math.min(weight, 1));
    if (weight === 0) return data.slice();
    // Constant series detection (TB: isConstant check)
    const first = data[0];
    if (data.every(v => v === first)) return data.slice();
    const out = new Array(data.length);
    let last = 0;
    let numAccum = 0;
    for (let i = 0; i < data.length; i++) {
      const v = data[i];
      if (!Number.isFinite(v)) {
        out[i] = v; // NaN/Infinity pass through
      } else {
        last = last * weight + (1 - weight) * v;
        numAccum++;
        const debiasWeight = weight === 1 ? 1 : 1 - Math.pow(weight, numAccum);
        out[i] = last / debiasWeight;
      }
    }
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
    let { fullLoss, fullLr, viewXMin, viewXMax, smoothingWeight } = opts;
    if (fullLoss.length < 1) return;
    // SVG polyline with a single point is invisible. Duplicate so callers that
    // pass exactly one sample (e.g. first step before epoch completes) still
    // see a flat baseline rather than a blank canvas.
    if (fullLoss.length === 1) {
      fullLoss = [fullLoss[0], fullLoss[0]];
      fullLr = fullLr.length === 1 ? [fullLr[0], fullLr[0]] : fullLr;
    }
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

    // TB-faithful Y domain: only smoothed+MA (not raw — raw is aux in TB)
    const domainVals = visibleSliderSmoothed.concat(visibleMA5).filter(v => Number.isFinite(v));
    const { yMin, yMax, yTicks } = _tbYDomain(domainVals, 6);

    let lrMaxVal = 1e-4;
    for (let i = 0; i < visibleLr.length; i++) { if (visibleLr[i] > lrMaxVal) lrMaxVal = visibleLr[i]; }
    const lrMin = 0, lrMax = lrMaxVal * 1.15;
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
      const ySpacing = yTicks.length > 1 ? Math.abs(yTicks[1] - yTicks[0]) : 1;
      yTicks.forEach(tick => {
        const y = toY(tick).toFixed(1);
        const isZero = Math.abs(tick) < ySpacing * 0.01;
        const sw = isZero ? '1.5' : '1';
        const color = isZero ? 'rgba(255,255,255,0.15)' : 'rgba(255,255,255,0.08)';
        gridHTML += `<line x1="0" y1="${y}" x2="${vbW}" y2="${y}" stroke="${color}" stroke-width="${sw}" vector-effect="non-scaling-stroke" />`;
      });
      // Vertical grid lines (TB X_GRID_COUNT = 10)
      const width = svg.getBoundingClientRect().width || 400;
      const loTick = Math.max(0, startIdx);
      const hiTick = Math.max(loTick + 1, endIdx - 1);
      const vTicks = _buildXTicks(loTick, hiTick, width);
      vTicks.forEach(tick => {
        const frac = (tick - startIdx) / Math.max(1, endIdx - startIdx - 1);
        if (frac < 0 || frac > 1) return;
        const x = (frac * vbW).toFixed(1);
        gridHTML += `<line x1="${x}" y1="0" x2="${x}" y2="${vbH}" stroke="rgba(255,255,255,0.08)" stroke-width="1" vector-effect="non-scaling-stroke" />`;
      });
      gridG.innerHTML = gridHTML;
    }

    if (yLabelsEl) {
      yLabelsEl.innerHTML = yTicks.slice().reverse().map(tick =>
        `<span style="text-align:right;">${_fmtAxisTick(tick)}</span>`
      ).join('');
    }

    if (yLabelsRightEl && visibleLr.length > 0) {
      const lrTicks = [];
      const lrStep = lrMax / 4;
      for (let i = 0; i <= 4; i++) lrTicks.push(lrMax - i * lrStep);
      yLabelsRightEl.innerHTML = lrTicks.map(v =>
        `<span style="text-align:left;">${v.toExponential(1)}</span>`
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
        return `<span class="axis-label-row__tick" style="left:${left.toFixed(3)}%;">${tick.toLocaleString()}</span>`;
      }).join('');
    }
  }

  /**
   * Return SVG-space coordinates for hover dot markers.
   * Called by chart-interaction to position dots at the nearest data point.
   * @param {number} dataIdx — index into the FULL data array
   * @param {Object} opts — same opts passed to render()
   * @returns {Object|null} { x, lossY, smoothY, ma5Y, lrY } in viewBox coords
   */
  function getPointCoords(dataIdx, opts) {
    const { fullLoss, fullLr, viewXMin, viewXMax, smoothingWeight } = opts;
    if (fullLoss.length < 1) return null;
    const totalLen = fullLoss.length;
    let startIdx, endIdx;
    if (viewXMin !== null && viewXMax !== null) {
      startIdx = Math.max(0, Math.round(viewXMin));
      endIdx = Math.min(totalLen, Math.round(viewXMax));
      if (endIdx - startIdx < 2) startIdx = Math.max(0, endIdx - 2);
    } else { startIdx = 0; endIdx = totalLen; }
    if (dataIdx < startIdx || dataIdx >= endIdx) return null;

    const visLen = endIdx - startIdx;
    const localIdx = dataIdx - startIdx;
    const vbW = 1000, vbH = 500;
    const x = visLen > 1 ? (localIdx / (visLen - 1)) * vbW : vbW / 2;

    const visibleLoss = fullLoss.slice(startIdx, endIdx);
    const allSmoothed = expSmooth(fullLoss, smoothingWeight);
    const allMA5 = simpleMA(fullLoss, 5);
    const visSmooth = allSmoothed.slice(startIdx, endIdx);
    const visMA5 = allMA5.slice(startIdx, endIdx);
    const visLr = fullLr.slice(startIdx, endIdx);

    // Recompute Y range (same TB-faithful logic as render: exclude raw, use _tbYDomain)
    const domainVals = visSmooth.concat(visMA5).filter(v => Number.isFinite(v));
    const { yMin, yMax } = _tbYDomain(domainVals, 6);
    const toY = (v) => yMax > yMin ? ((yMax - v) / (yMax - yMin)) * vbH : vbH / 2;

    let lrMaxVal = 1e-4;
    for (let i = 0; i < visLr.length; i++) { if (visLr[i] > lrMaxVal) lrMaxVal = visLr[i]; }
    const lrMax = lrMaxVal * 1.15;
    const toLrY = (v) => lrMax > 0 ? ((lrMax - v) / lrMax) * vbH : vbH / 2;

    return {
      x,
      lossY: toY(visibleLoss[localIdx]),
      smoothY: toY(visSmooth[localIdx]),
      ma5Y: toY(visMA5[localIdx]),
      lrY: toLrY(visLr[localIdx] ?? 0),
    };
  }

  return { render, expSmooth, simpleMA, getPointCoords };
})();
