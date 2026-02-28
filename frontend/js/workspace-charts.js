/* ============================================================
   Side-Step GUI — Chart Controls
   TensorBoard button, series toggles, axis toggle, smoothing,
   drag-to-zoom, shift+pan, wheel zoom, hover tooltip.
   Extracted from workspace.js for 400 LOC cap.
   ============================================================ */

const WorkspaceCharts = (() => {
  "use strict";

  const $ = (id) => document.getElementById(id);

  /* ---- TensorBoard Button ---- */
  function initTensorBoardBtn() {
    $("btn-open-tensorboard")?.addEventListener("click", async () => {
      const outputDir = $("monitor-output-dir")?.textContent?.replace("Output: ", "").trim() || "";
      const logDir = $("full-log-dir")?.value || (outputDir ? outputDir + "/runs" : "");
      const port = 6006;
      try {
        const result = await API.startTensorBoard(logDir, port, outputDir);
        if (result.ok) {
          window.open(result.url || ("http://localhost:" + port), "_blank");
          const verb = result.restarted ? "TensorBoard restarted" : "TensorBoard started";
          if (typeof showToast === "function") showToast(verb + (result.log_dir ? " (" + result.log_dir + ")" : ""), "ok");
        } else {
          if (typeof showToast === "function") showToast("TensorBoard failed: " + (result.error || "unknown error"), "error");
        }
      } catch (e) {
        console.error('[TensorBoard] API call failed, opening fallback URL:', e);
        window.open("http://localhost:" + port, "_blank");
      }
    });
  }

  /* ---- Chart Controls (TensorBoard-style) ---- */
  function initChartControls() {
    const _setPressed = (btn, isPressed) => {
      if (!btn) return;
      btn.setAttribute("aria-pressed", isPressed ? "true" : "false");
    };

    document.querySelectorAll(".chart-toggle").forEach((btn) => {
      _setPressed(btn, btn.classList.contains("active"));
      btn.addEventListener("click", () => {
        btn.classList.toggle("active");
        const isActive = btn.classList.contains("active");
        _setPressed(btn, isActive);
        const series = btn.dataset.series;
        const lineMap = { loss: "monitor-loss-line", ma5: "monitor-ma-line", lr: "monitor-lr-line" };
        const lineEl = $(lineMap[series]);
        if (lineEl) lineEl.style.display = isActive ? "" : "none";
      });
    });

    const axisBtns = Array.from(document.querySelectorAll(".chart-axis"));
    const _setAxisActive = (activeBtn) => {
      axisBtns.forEach((b) => {
        const isActive = b === activeBtn;
        b.classList.toggle("active", isActive);
        _setPressed(b, isActive);
      });
    };

    axisBtns.forEach((btn) => {
      btn.addEventListener("click", () => {
        _setAxisActive(btn);
        if (typeof Training !== "undefined") {
          Training.setChartMode(btn.dataset.axis);
        }
      });
    });
    const initialAxis = axisBtns.find((b) => b.classList.contains("active")) || axisBtns[0];
    if (initialAxis) _setAxisActive(initialAxis);

    document.querySelectorAll(".chart-zoom").forEach((btn) => {
      btn.addEventListener("click", () => {
        if (btn.dataset.zoom === "reset" && typeof Training !== "undefined") Training.zoomReset();
      });
    });

    const smoothSlider = $("chart-smoothing");
    const smoothVal = $("chart-smoothing-val");
    if (smoothSlider) {
      smoothSlider.addEventListener("input", () => {
        const w = parseInt(smoothSlider.value, 10) / 100;
        if (smoothVal) smoothVal.textContent = w.toFixed(2);
        if (typeof Training !== "undefined") Training.setSmoothing(w);
      });
    }

    // Delegate to shared ChartInteraction utility (Training loaded later, callbacks are lazy)
    if (typeof ChartInteraction !== "undefined") {
      const T = () => typeof Training !== "undefined" ? Training : null;
      ChartInteraction.wire({
        areaId: "monitor-chart-area", svgId: "monitor-loss-svg",
        getView: () => T()?.getChartView() || { startIdx: 0, endIdx: 1, totalLen: 1 },
        setRange: (lo, hi) => T()?.setViewRange(lo, hi),
        zoomReset: () => T()?.zoomReset(),
        minRange: 2,
        zoomStep: 0.16,
        panModifier: "shift",
        enableDoubleClickReset: true,
        formatTip: (idx) => {
          const mode = T()?.getChartMode?.() || "epoch";
          const label = mode === "step" ? "step" : "epoch";
          return '<span class="u-text-muted">' + label + ' #' + idx + '</span>';
        },
      });
    }
  }

  function init() {
    initTensorBoardBtn();
    initChartControls();
  }

  return { init };
})();
