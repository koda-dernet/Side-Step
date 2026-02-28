/* ============================================================
   Side-Step GUI — AppState (single source of truth)
   Centralizes GPU, training status, run history, and
   deterministic loss curves so every UI component reads
   from one place.
   ============================================================ */

const AppState = (() => {
  "use strict";

  // ---- Internal state ----------------------------------------------------
  let _status = "idle"; // "idle" | "training" | "error" | "complete"
  let _gpu = { name: "detecting...", vram_used_mb: 0, vram_total_mb: 0, utilization: 0, temperature: 0, power_draw_w: 0 };
  let _runs = [];
  const _configCache = {};


  // ---- Status ------------------------------------------------------------
  function status() { return _status; }
  function setStatus(s) {
    _status = s;
    document.dispatchEvent(new CustomEvent("appstate:status", { detail: s }));
  }

  // ---- GPU ---------------------------------------------------------------
  function gpu() { return { ..._gpu }; }
  function setGPU(data) {
    if (data && data.power_draw_w == null && data.power_draw != null) data.power_draw_w = data.power_draw;
    if (data && data.power_draw == null && data.power_draw_w != null) data.power_draw = data.power_draw_w;
    Object.assign(_gpu, data);
    document.dispatchEvent(new CustomEvent("appstate:gpu", { detail: _gpu }));
  }

  // ---- Runs --------------------------------------------------------------
  function runs() { return _runs; }
  function setRuns(arr) { _runs = arr; document.dispatchEvent(new CustomEvent("appstate:runs", { detail: _runs })); }

  // ---- Run configs (cached from API) -------------------------------------
  function runConfig(name) { return _configCache[name] || null; }
  function setRunConfig(name, cfg) { _configCache[name] = cfg; }

  // ---- Topbar + monitor GPU updater (listens to appstate:gpu) ------------
  function initGPUListeners() {
    document.addEventListener("appstate:gpu", (e) => {
      const g = e.detail;
      const pct = g.vram_total_mb > 0 ? (g.vram_used_mb / g.vram_total_mb) * 100 : 0;
      // Topbar
      const topName = document.getElementById("topbar-gpu-name");
      const topVram = document.getElementById("topbar-gpu-vram");
      if (topName) topName.textContent = "GPU: " + g.name;
      if (topVram) topVram.textContent = (g.vram_used_mb / 1024).toFixed(1) + " / " + (g.vram_total_mb / 1024).toFixed(1) + " GB";
      // Monitor panel
      const el = (id, val) => { const e = document.getElementById(id); if (e) e.textContent = val; };
      el("monitor-vram-val", (g.vram_used_mb / 1024).toFixed(1) + " / " + (g.vram_total_mb / 1024).toFixed(1) + " GB");
      el("monitor-util-val", g.utilization + "%");
      el("monitor-temp", "Temp: " + g.temperature + "\u00b0C");
      const power = g.power_draw_w ?? g.power_draw ?? 0;
      el("monitor-power", "Power: " + power + "W");
      const vramBar = document.getElementById("monitor-vram-bar");
      const utilBar = document.getElementById("monitor-util-bar");
      if (vramBar) vramBar.style.width = pct + "%";
      if (utilBar) utilBar.style.width = g.utilization + "%";
    });
  }

  // ---- Topbar status chip updater ----------------------------------------
  function initStatusListener() {
    document.addEventListener("appstate:status", (e) => {
      const chip = document.getElementById("topbar-status-chip");
      if (!chip) return;
      chip.className = "topbar__status-chip topbar__status-chip--" + e.detail;
      const labels = { idle: "Idle", training: "Training", error: "Error", complete: "Complete" };
      chip.textContent = labels[e.detail] || e.detail;
    });
  }

  function init() { initGPUListeners(); initStatusListener(); }

  return { status, setStatus, gpu, setGPU, runs, setRuns, runConfig, setRunConfig, init };
})();
