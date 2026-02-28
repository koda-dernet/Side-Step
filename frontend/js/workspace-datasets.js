/* Side-Step GUI — Datasets Tab (extracted from workspace-behaviors.js for LOC cap) */

const WorkspaceDatasets = (() => {
  "use strict";

  const $ = (id) => document.getElementById(id);
  const _esc = window._esc || ((s) => String(s ?? ''));
  const _fmtWhen = (iso) => {
    if (!iso) return "";
    const d = new Date(iso);
    if (Number.isNaN(d.getTime())) return "";
    return d.toLocaleString();
  };
  const _baseName = (p) => String(p || "").split(/[\\/]/).filter(Boolean).pop() || String(p || "");

  let _linkTarget = null;

  function _setDatasetPickerValue(selectId, path) {
    const sel = $(selectId);
    if (!sel || !path) return;
    let opt = [...sel.options].find((o) => o.value === path);
    if (!opt) {
      opt = document.createElement("option");
      opt.value = path;
      opt.textContent = _baseName(path) + " (manual)";
      sel.appendChild(opt);
    }
    sel.value = path;
    sel.dispatchEvent(new Event("change"));
  }

  async function _refreshDatasets() {
    const tbody = $("datasets-tbody"), footer = $("datasets-footer");
    if (!tbody) return;
    try {
      const result = await API.fetchAllDatasets();
      const datasets = (result.datasets || []).filter(ds => ds.type === "tensors");
      tbody.innerHTML = "";
      if (datasets.length === 0) {
        tbody.innerHTML = '<tr><td colspan="6" class="data-table-empty">No preprocessed tensor datasets found. Run preprocessing from the Audio Library or Preprocess tab to create one.</td></tr>';
        if (footer) footer.textContent = "0 tensor datasets found";
        return;
      }
      datasets.forEach((ds) => {
        const tr = document.createElement("tr");
        const linked = ds.audio_linked
          ? `<span class="u-text-success">[ok] ${_esc(ds.audio_linked)}</span>`
          : '<span class="u-text-muted">not linked</span>';
        const meta = [];
        if (ds.model_variant) meta.push(`model: ${_esc(ds.model_variant)}`);
        if (ds.normalize) meta.push(`norm: ${_esc(ds.normalize)}`);
        if (ds.created_at) meta.push(`created: ${_esc(_fmtWhen(ds.created_at))}`);
        const metaLine = meta.length
          ? `<div class="u-meta-muted-xs" style="margin-top: 2px;">${meta.join(" · ")}</div>`
          : "";
        const sourceBtn = ds.audio_linked
          ? ` <button class="btn btn--sm" data-action="open-source-audio" data-path="${_esc(ds.audio_linked)}">Open Source</button>`
          : "";
        tr.innerHTML = `
          <td class="u-text-bold">${_esc(ds.name)}${metaLine}</td>
          <td>${_esc(ds.files_label)}</td><td>${_esc(ds.duration_label)}</td><td>${linked}</td>
          <td class="u-meta-muted-xs">${_esc(ds.path)}</td>
          <td>
            <div style="display:flex;flex-wrap:wrap;gap:4px;align-items:center;">
              <button class="btn btn--sm" data-action="open-dataset" data-path="${_esc(ds.path)}">Open</button>
              <button class="btn btn--sm" data-action="copy-path" data-path="${_esc(ds.path)}" title="Copy path to clipboard">Copy</button>
              <button class="btn btn--sm btn--primary" data-action="use-dataset" data-path="${_esc(ds.path)}">Use for Training</button>${sourceBtn}
              <button class="btn btn--sm" data-action="link-audio" data-name="${_esc(ds.name)}">Set Source</button>
            </div>
          </td>`;
        tbody.appendChild(tr);
      });
      if (footer) footer.textContent = `${datasets.length} dataset${datasets.length !== 1 ? "s" : ""} found · scanned from Settings roots`;
      _bindDatasetButtons();
      if (typeof initShiftClickTable === "function") initShiftClickTable("datasets-tbody");
    } catch (e) {
      if (footer) footer.textContent = "Failed to scan datasets";
    }
  }

  function _bindDatasetButtons() {
    document.querySelectorAll('[data-action="open-dataset"]').forEach((btn) => {
      btn.addEventListener("click", async () => {
        const path = btn.dataset.path;
        const result = await API.openFolder(path);
        if (result.ok) showToast("Opened: " + path, "ok");
        else showToast("Failed to open folder: " + (result.error || path), "error");
      });
    });
    document.querySelectorAll('[data-action="copy-path"]').forEach((btn) => {
      btn.addEventListener("click", async () => {
        const path = btn.dataset.path;
        const copied = typeof copyTextToClipboard === "function"
          ? await copyTextToClipboard(path)
          : false;
        showToast(copied ? "Path copied" : "Failed to copy path", copied ? "ok" : "error");
      });
    });
    document.querySelectorAll('[data-action="use-dataset"]').forEach((btn) => {
      btn.addEventListener("click", () => {
        const path = btn.dataset.path;
        if (!path) return;
        _setDatasetPickerValue("full-dataset-dir", path);
        _setDatasetPickerValue("ez-dataset-dir", path);
        if (typeof WorkspaceConfig !== "undefined" && typeof WorkspaceConfig.updateEzReview === "function") {
          WorkspaceConfig.updateEzReview();
        }
        showToast("Tensor dataset selected for training", "ok");
      });
    });
    document.querySelectorAll('[data-action="open-source-audio"]').forEach((btn) => {
      btn.addEventListener("click", async () => {
        const path = btn.dataset.path;
        if (!path) return;
        const result = await API.openFolder(path);
        if (result.ok) showToast("Opened source audio", "ok");
        else showToast("Failed to open source audio folder", "error");
      });
    });
    document.querySelectorAll('[data-action="link-audio"]').forEach((btn) => {
      btn.addEventListener("click", () => {
        _linkTarget = btn.dataset.name;
        const dialog = $("datasets-link-dialog");
        if (dialog) dialog.style.display = "block";
        const pathInput = $("datasets-link-path");
        const audioDefault = $("settings-audio-dir")?.value || "";
        if (pathInput) pathInput.value = audioDefault;
        if (pathInput) pathInput.focus();
      });
    });
  }

  function init() {
    $("btn-datasets-link-confirm")?.addEventListener("click", async () => {
      const name = _linkTarget;
      if (!name) { showToast("No tensor dataset selected", "warn"); return; }
      const path = $("datasets-link-path")?.value;
      if (!path) { showToast("Enter an audio folder path", "warn"); return; }
      try {
        const result = await API.linkSourceAudio(name, path);
        if (result.ok) {
          showToast("Source audio linked for '" + name + "'", "ok");
          _refreshDatasets();
        } else {
          showToast("Failed to link source audio: " + (result.error || "unknown error"), "error");
        }
      } catch (e) {
        showToast("Failed to link source audio: " + e.message, "error");
      }
      const dialog = $("datasets-link-dialog"); if (dialog) dialog.style.display = "none";
      _linkTarget = null;
    });

    $("btn-datasets-link-cancel")?.addEventListener("click", () => {
      const dialog = $("datasets-link-dialog"); if (dialog) dialog.style.display = "none";
      _linkTarget = null;
    });

    _refreshDatasets();

    document.addEventListener("sidestep:settings-saved", () => {
      _refreshDatasets();
    });

    $("btn-refresh-datasets")?.addEventListener("click", () => {
      _refreshDatasets();
      showToast("Datasets refreshed from Settings roots", "ok");
    });
  }

  return { init, refresh: _refreshDatasets };
})();
