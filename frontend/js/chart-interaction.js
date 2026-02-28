/* ============================================================
   Side-Step GUI — Chart Interaction (shared)
   TensorBoard-like interactions:
   - drag box: zoom to selected span
   - modifier+drag: pan current window
   - wheel: cursor-centered zoom with clamped bounds
   - double-click: reset zoom
   ============================================================ */

const ChartInteraction = (() => {
  "use strict";

  /**
   * Wire interactive chart controls on a chart area.
   * @param {object} opts
   * @param {string} opts.areaId   - chart-area container element ID
   * @param {string} opts.svgId    - SVG element ID inside the area
   * @param {function} opts.getView      - () => { startIdx, endIdx, totalLen }
   * @param {function} opts.setRange     - (lo, hi) => void
   * @param {function} opts.zoomReset    - () => void
   * @param {function} [opts.formatTip]  - (dataIdx, frac) => HTML string
   * @param {number}   [opts.minRange]   - minimum zoom window in samples
   * @param {number}   [opts.zoomStep]   - wheel zoom step (fraction, default 0.16)
   * @param {string}   [opts.panModifier]- 'shift' | 'alt' | 'ctrl' | 'meta'
   * @param {boolean}  [opts.enableDoubleClickReset] - reset zoom on dblclick
   */
  function wire(opts) {
    const area = document.getElementById(opts.areaId);
    const svg = document.getElementById(opts.svgId);
    if (!area || !svg) return;

    // Reuse existing overlay/crosshair/tooltip or create new ones
    const _find = (cls) => area.querySelector("." + cls) || area.querySelector("[id*='" + cls.replace("chart-","") + "']");
    let overlay = _find("chart-drag-overlay");
    if (!overlay) { overlay = document.createElement("div"); overlay.className = "chart-drag-overlay"; overlay.style.cssText = "display:none;position:absolute;background:rgba(0,200,255,0.12);pointer-events:none;z-index:2;"; area.style.position = "relative"; area.appendChild(overlay); }
    let crosshair = _find("chart-crosshair");
    if (!crosshair) { crosshair = document.createElement("div"); crosshair.className = "chart-crosshair"; crosshair.style.cssText = "display:none;position:absolute;top:0;width:1px;height:100%;background:var(--border);pointer-events:none;z-index:3;"; area.appendChild(crosshair); }
    let tooltip = _find("chart-tooltip");
    if (!tooltip) { tooltip = document.createElement("div"); tooltip.className = "chart-tooltip"; tooltip.style.cssText = "display:none;position:absolute;background:var(--panel);border:1px solid var(--border);border-radius:var(--radius);padding:2px 6px;font-size:10px;font-family:var(--font-mono);pointer-events:none;z-index:4;white-space:nowrap;"; area.appendChild(tooltip); }

    const minRange = Math.max(2, Number(opts.minRange) || 2);
    const zoomStep = Math.min(0.45, Math.max(0.05, Number(opts.zoomStep) || 0.16));
    const panModifier = String(opts.panModifier || "shift").toLowerCase();
    const allowDoubleClickReset = opts.enableDoubleClickReset !== false;

    let dragStart = null, panning = false, panStartX = 0, panView = null;

    function pxToFrac(clientX) {
      const r = svg.getBoundingClientRect();
      return Math.max(0, Math.min(1, (clientX - r.left) / (r.width || 1)));
    }

    function isPanGesture(e) {
      if (panModifier === "alt") return !!e.altKey;
      if (panModifier === "ctrl") return !!e.ctrlKey;
      if (panModifier === "meta") return !!e.metaKey;
      return !!e.shiftKey;
    }

    function clampRange(lo, hi, totalLen) {
      const total = Math.max(1, Number(totalLen) || 1);
      const minAllowed = Math.max(1, Math.min(minRange, total));
      let start = Math.max(0, Number(lo) || 0);
      let end = Math.min(total, Number(hi) || total);
      if (end - start < minAllowed) {
        const mid = (start + end) / 2;
        start = Math.max(0, mid - minAllowed / 2);
        end = Math.min(total, start + minAllowed);
        start = Math.max(0, end - minAllowed);
      }
      return { lo: start, hi: end };
    }

    area.addEventListener("mousedown", (e) => {
      if (e.button !== 0) return;
      if (isPanGesture(e)) {
        panning = true;
        panStartX = e.clientX;
        panView = opts.getView();
        area.style.cursor = "grabbing";
        e.preventDefault();
        return;
      }
      const r = area.getBoundingClientRect();
      dragStart = { x: e.clientX, left: e.clientX - r.left };
      overlay.style.display = "none";
      e.preventDefault();
    });

    document.addEventListener("mousemove", (e) => {
      // Pan
      if (panning && panView) {
        const r = svg.getBoundingClientRect();
        const dx = e.clientX - panStartX;
        const dataRange = panView.endIdx - panView.startIdx;
        const shift = -(dx / (r.width || 1)) * dataRange;
        let lo = panView.startIdx + shift, hi = lo + dataRange;
        if (lo < 0) { lo = 0; hi = dataRange; }
        if (hi > panView.totalLen) { hi = panView.totalLen; lo = Math.max(0, hi - dataRange); }
        const clamped = clampRange(lo, hi, panView.totalLen);
        opts.setRange(clamped.lo, clamped.hi);
        return;
      }
      // Drag-to-zoom overlay
      if (dragStart) {
        const r = area.getBoundingClientRect();
        const svgR = svg.getBoundingClientRect();
        const curX = e.clientX - r.left;
        const x1 = Math.max(svgR.left - r.left, Math.min(dragStart.left, curX));
        const x2 = Math.min(svgR.right - r.left, Math.max(dragStart.left, curX));
        if (x2 - x1 > 4) { overlay.style.display = "block"; overlay.style.left = x1 + "px"; overlay.style.top = "0"; overlay.style.width = (x2 - x1) + "px"; overlay.style.height = svgR.height + "px"; }
        return;
      }
      // Tooltip
      _updateTip(e);
    });

    document.addEventListener("mouseup", (e) => {
      if (panning) { panning = false; panView = null; area.style.cursor = ""; return; }
      if (!dragStart) return;
      const dx = Math.abs(e.clientX - dragStart.x);
      overlay.style.display = "none";
      if (dx > 8) {
        const view = opts.getView();
        const f1 = pxToFrac(Math.min(dragStart.x, e.clientX));
        const f2 = pxToFrac(Math.max(dragStart.x, e.clientX));
        const lo = view.startIdx + f1 * (view.endIdx - view.startIdx);
        const hi = view.startIdx + f2 * (view.endIdx - view.startIdx);
        if (hi - lo >= 1) {
          const clamped = clampRange(lo, hi, view.totalLen);
          opts.setRange(clamped.lo, clamped.hi);
        }
      }
      dragStart = null;
    });

    area.addEventListener("wheel", (e) => {
      e.preventDefault();
      const view = opts.getView();
      if (view.totalLen <= 2) return;
      const frac = pxToFrac(e.clientX);
      const cursor = view.startIdx + frac * (view.endIdx - view.startIdx);
      const range = view.endIdx - view.startIdx;
      const factor = e.deltaY > 0 ? (1 + zoomStep) : Math.max(0.2, 1 - zoomStep);
      const nr = Math.max(minRange, range * factor);
      if (nr >= view.totalLen * 0.995) {
        opts.zoomReset();
      } else {
        const lo = cursor - frac * nr;
        const hi = cursor + (1 - frac) * nr;
        const clamped = clampRange(lo, hi, view.totalLen);
        opts.setRange(clamped.lo, clamped.hi);
      }
    }, { passive: false });

    if (allowDoubleClickReset) {
      area.addEventListener("dblclick", (e) => {
        e.preventDefault();
        opts.zoomReset();
      });
    }

    function _updateTip(e) {
      const r = svg.getBoundingClientRect();
      const ar = area.getBoundingClientRect();
      if (e.clientX < r.left || e.clientX > r.right || e.clientY < r.top || e.clientY > r.bottom) {
        tooltip.style.display = "none"; crosshair.style.display = "none"; return;
      }
      const view = opts.getView();
      const frac = (e.clientX - r.left) / r.width;
      const dataIdx = Math.max(0, Math.min(
        Math.max(0, view.totalLen - 1),
        Math.round(view.startIdx + frac * (view.endIdx - view.startIdx))
      ));
      const xPos = e.clientX - ar.left;
      crosshair.style.display = "block"; crosshair.style.left = xPos + "px";
      tooltip.style.display = "block"; tooltip.style.left = (xPos + 12) + "px"; tooltip.style.top = (e.clientY - ar.top - 30) + "px";
      tooltip.innerHTML = opts.formatTip ? opts.formatTip(dataIdx, frac) : '<span class="u-text-muted">#' + dataIdx + '</span>';
    }

    area.addEventListener("mouseleave", () => { tooltip.style.display = "none"; crosshair.style.display = "none"; });
  }

  return { wire };
})();
