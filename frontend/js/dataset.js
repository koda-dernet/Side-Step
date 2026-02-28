/* ============================================================
   Side-Step GUI — Dataset Browser + Sidecar Editor
   Populates the dataset table from API.scanDataset() and
   handles the slide-out sidecar editor panel.
   ============================================================ */

const Dataset = (() => {

  const $ = (id) => document.getElementById(id);
  const _esc = window._esc;
  let _files = [];
  let _folders = [];
  let _currentFile = null;
  let _scanRoot = '';
  const _expandedFolders = new Set();

  function _canonicalAudioPath() {
    return ($('settings-audio-dir')?.value || '').trim();
  }

  function _setAudioPathInput(path) {
    const input = $('lab-dataset-path');
    if (!input) return;
    input.value = path || '';
    input.readOnly = true;
    input.style.opacity = '0.75';
  }

  function _sidecarPath(file) {
    const explicit = file?.sidecar_path;
    if (explicit) return explicit;
    const p = file?.path || '';
    return p.replace(/\.[^.]+$/, '.txt');
  }

  function _fmtDuration(seconds) {
    const m = Math.floor(seconds / 60);
    const s = seconds % 60;
    return `${m}:${String(s).padStart(2, '0')}`;
  }

  function _fmtTotalDuration(seconds) {
    const h = Math.floor(seconds / 3600);
    const m = Math.floor((seconds % 3600) / 60);
    const s = seconds % 60;
    if (h > 0) return `${h}h ${m}m ${s}s`;
    return `${m}m ${s}s`;
  }

  function _setActiveLabPanel(panelId) {
    switchMode('lab');
    document.querySelectorAll('.lab-nav__item').forEach((b) => b.classList.remove('active'));
    document.querySelectorAll('.lab-panel').forEach((p) => p.classList.remove('active'));
    document.querySelector(`.lab-nav__item[data-lab="${panelId}"]`)?.classList.add('active');
    $(`lab-${panelId}`)?.classList.add('active');
  }

  function _openPreprocessForFolder(folderPath) {
    const root = _scanRoot || _canonicalAudioPath();
    if (!root) return;
    const target = folderPath === '.' ? root : _joinPath(root, folderPath);
    _setActiveLabPanel('preprocess');
    const ppAudio = $('pp-audio-dir');
    if (ppAudio) {
      ppAudio.value = target;
      ppAudio.dispatchEvent(new Event('change'));
    }
    const ppOut = $('pp-output-dir');
    if (ppOut && ppOut.readOnly) {
      ppOut.value = _joinPath($('settings-tensors-dir')?.value || './preprocessed_tensors', _pathBasename(target) || 'tensors');
    }
    if (typeof showToast === 'function') showToast('Preprocess path set from Audio Library folder', 'info');
  }

  function _normalizeFolders(scanResult) {
    const rows = Array.isArray(scanResult?.folders) ? scanResult.folders.slice() : [];
    if (!rows.some((f) => (f.path || '.') === '.')) {
      rows.unshift({
        path: '.',
        name: _pathBasename(_scanRoot) || 'Root',
        parent_path: '',
        depth: 0,
        file_count: _files.length,
        sidecar_count: Number(scanResult?.sidecar_count || 0),
        total_duration: Number(scanResult?.total_duration || 0),
        duration_label: _fmtTotalDuration(Number(scanResult?.total_duration || 0)),
      });
    }
    return rows;
  }

  function _renderHierarchy(tbody) {
    tbody.innerHTML = '';
    if (_files.length === 0) {
      tbody.innerHTML = '<tr><td colspan="7" class="data-table-empty">No audio files found in the current Audio directory.</td></tr>';
      return;
    }

    const map = new Map();
    _folders.forEach((f) => {
      map.set(f.path || '.', {
        ...f,
        path: f.path || '.',
        parent_path: f.parent_path || '',
        depth: Number(f.depth || 0),
        children: [],
        files: [],
      });
    });
    if (!map.has('.')) {
      map.set('.', {
        path: '.',
        name: _pathBasename(_scanRoot) || 'Root',
        parent_path: '',
        depth: 0,
        children: [],
        files: [],
      });
    }

    _files.forEach((f) => {
      const fp = String(f.folder_path || '.');
      if (!map.has(fp)) {
        map.set(fp, {
          path: fp,
          name: fp.split('/').pop() || fp,
          parent_path: fp.includes('/') ? fp.slice(0, fp.lastIndexOf('/')) : '.',
          depth: fp === '.' ? 0 : fp.split('/').length,
          children: [],
          files: [],
        });
      }
      map.get(fp).files.push(f);
    });

    map.forEach((folder) => {
      const p = folder.path;
      if (p === '.') return;
      const parentPath = folder.parent_path || '.';
      if (!map.has(parentPath)) {
        map.set(parentPath, {
          path: parentPath,
          name: parentPath === '.' ? (_pathBasename(_scanRoot) || 'Root') : (parentPath.split('/').pop() || parentPath),
          parent_path: parentPath.includes('/') ? parentPath.slice(0, parentPath.lastIndexOf('/')) : '.',
          depth: parentPath === '.' ? 0 : parentPath.split('/').length,
          children: [],
          files: [],
        });
      }
      map.get(parentPath).children.push(folder);
    });

    const sortByName = (a, b) => String(a.name || a.path).localeCompare(String(b.name || b.path));
    map.forEach((folder) => {
      folder.children.sort(sortByName);
      folder.files.sort((a, b) => String(a.name).localeCompare(String(b.name)));
    });

    const appendFile = (f, depth) => {
      const idx = _files.findIndex((x) => x.path === f.path);
      const sidecarStatus = f.has_sidecar
        ? '<span class="status--ok">[ok] exists</span>'
        : '<span class="status--warn">missing</span>';
      const editLabel = f.has_sidecar ? 'Edit' : 'Create';
      const trFile = document.createElement('tr');
      trFile.className = 'dataset-file-row';
      trFile.innerHTML = `
        <td>
          <span class="dataset-folder-indent" style="margin-left:${Math.max(0, depth) * 12}px;"></span>
          <span title="${_esc(f.relative_path || f.name)}">${_esc(f.name.length > 44 ? f.name.slice(0, 41) + '...' : f.name)}</span>
        </td>
        <td>${_fmtDuration(f.duration)}</td>
        <td>${sidecarStatus}</td>
        <td>${f.genre ? _esc(f.genre) : '<span class="u-text-muted">--</span>'}</td>
        <td>${f.tags ? _esc(f.tags) : '<span class="u-text-muted">--</span>'}</td>
        <td>${f.trigger ? _esc(f.trigger) : '<span class="u-text-muted">--</span>'}</td>
        <td><button class="btn btn--sm sidecar-edit-btn" data-idx="${idx}">${editLabel}</button></td>
      `;
      tbody.appendChild(trFile);
    };

    const appendFolder = (folder) => {
      const folderPath = folder.path || '.';
      const hasChildren = folder.children.length > 0 || folder.files.length > 0;
      const expanded = _expandedFolders.has(folderPath);
      const duration = Number(folder.total_duration || 0);
      const sidecars = Number(folder.sidecar_count || folder.files.filter((f) => f.has_sidecar).length || 0);
      const count = Number(folder.file_count || folder.files.length || 0);
      const depth = Math.max(0, Number(folder.depth || 0));

      if (folderPath !== '.') {
        const tr = document.createElement('tr');
        tr.className = 'dataset-folder-row';
        tr.innerHTML = `
          <td>
            <span class="dataset-folder-indent" style="margin-left:${depth * 12}px;"></span>
            <button class="dataset-folder-toggle ${expanded ? 'open' : ''}" data-action="toggle-folder" data-folder="${_esc(folderPath)}" ${hasChildren ? '' : 'disabled'}>${_esc(folder.name || folderPath)}</button>
          </td>
          <td>${_fmtTotalDuration(duration)}</td>
          <td><span class="u-text-muted">${sidecars}/${count} sidecars</span></td>
          <td><span class="u-text-muted">--</span></td>
          <td><span class="u-text-muted">--</span></td>
          <td>${folder.common_trigger ? `<span class="u-text-secondary">${_esc(folder.common_trigger)}</span>` : '<span class="u-text-muted">--</span>'}</td>
          <td><button class="btn btn--sm" data-action="preprocess-folder" data-folder="${_esc(folderPath)}">Preprocess</button></td>
        `;
        tbody.appendChild(tr);
      }

      if (folderPath !== '.' && !expanded) return;

      folder.children.forEach((child) => appendFolder(child));
      folder.files.forEach((f) => appendFile(f, folderPath === '.' ? 0 : depth + 1));
    };

    appendFolder(map.get('.'));

    tbody.querySelectorAll('[data-action="toggle-folder"]').forEach((btn) => {
      btn.addEventListener('click', () => {
        const path = btn.dataset.folder || '.';
        if (_expandedFolders.has(path)) _expandedFolders.delete(path);
        else _expandedFolders.add(path);
        _renderHierarchy(tbody);
      });
    });

    tbody.querySelectorAll('[data-action="preprocess-folder"]').forEach((btn) => {
      btn.addEventListener('click', () => {
        _openPreprocessForFolder(btn.dataset.folder || '.');
      });
    });

    tbody.querySelectorAll('.sidecar-edit-btn').forEach((btn) => {
      btn.addEventListener('click', () => {
        const idx = parseInt(btn.dataset.idx, 10);
        if (!Number.isFinite(idx) || idx < 0) return;
        openEditor(_files[idx]);
      });
    });
  }

  async function scan(path) {
    const targetPath = (path || _canonicalAudioPath() || '').trim();
    _scanRoot = targetPath;
    _setAudioPathInput(targetPath);

    const detect = $('lab-dataset-detect');
    const footer = $('dataset-footer');
    const tbody = $('dataset-tbody');

    if (!targetPath) {
      _files = [];
      _folders = [];
      if (detect) {
        detect.textContent = '[warn] Set Settings > Audio directory to scan source audio';
        detect.className = 'detect detect--warn';
      }
      if (footer) footer.textContent = '0 files scanned';
      if (tbody) tbody.innerHTML = '';
      return;
    }

    let result = { files: [], sidecar_count: 0, total_duration: 0 };
    try {
      result = await API.scanDataset(targetPath);
    } catch {
      result = { files: [], sidecar_count: 0, total_duration: 0, error: 'Scan failed' };
    }

    _files = result.files || [];
    _folders = _normalizeFolders(result);
    _expandedFolders.clear();

    // Update detect line
    if (detect) {
      if (result.error) {
        detect.textContent = `[warn] ${result.error}`;
        detect.className = 'detect detect--warn';
      } else if (_files.length === 0) {
        detect.textContent = '[warn] No audio files found in the configured Audio directory';
        detect.className = 'detect detect--warn';
      } else {
        const sidecars = Number(result.sidecar_count || 0);
        const coveragePct = _files.length > 0 ? Math.round((sidecars / _files.length) * 100) : 0;
        const folderCount = Math.max(0, _folders.length - 1);
        detect.textContent = `[ok] ${_files.length} audio files across ${folderCount} folder${folderCount !== 1 ? 's' : ''} | Sidecars: ${sidecars}/${_files.length} (${coveragePct}%) | Total: ${_fmtTotalDuration(result.total_duration)}`;
        detect.className = 'detect detect--ok';
      }
    }
    if (footer) footer.textContent = `${_files.length} files scanned across ${Math.max(0, _folders.length - 1)} folders`;

    if (!tbody) return;
    _renderHierarchy(tbody);
    if (typeof initShiftClickTable === 'function') initShiftClickTable('dataset-tbody');
  }

  async function openEditor(file) {
    _currentFile = file;
    const editor = $('sidecar-editor');
    if (!editor) return;

    // Fill fields
    $('sidecar-editor-title').textContent = file.has_sidecar ? 'Edit Sidecar' : 'Create Sidecar';
    $('sidecar-filename').textContent = file.name;

    if (file.has_sidecar) {
      const wrapped = await API.readSidecar(_sidecarPath(file));
      const data = wrapped?.data || wrapped || {};
      $('sidecar-caption').value = data.caption || '';
      $('sidecar-genre').value = data.genre || '';
      $('sidecar-bpm').value = data.bpm || '';
      $('sidecar-key').value = data.key || '';
      $('sidecar-signature').value = data.signature || '';
      $('sidecar-tags').value = data.tags || '';
      $('sidecar-trigger').value = data.custom_tag || data.trigger || '';
      $('sidecar-lyrics').value = data.lyrics || '';
      $('sidecar-instrumental').checked = data.is_instrumental === true || data.is_instrumental === 'true';
    } else {
      $('sidecar-caption').value = '';
      $('sidecar-genre').value = '';
      $('sidecar-bpm').value = '';
      $('sidecar-key').value = '';
      $('sidecar-signature').value = '';
      $('sidecar-tags').value = '';
      $('sidecar-trigger').value = '';
      $('sidecar-lyrics').value = '';
      $('sidecar-instrumental').checked = false;
    }

    editor.classList.add('open');
  }

  function closeEditor() {
    const editor = $('sidecar-editor');
    if (editor) editor.classList.remove('open');
    _currentFile = null;
  }

  async function saveEditor() {
    if (!_currentFile) return;

    const data = {
      caption: $('sidecar-caption').value,
      genre: $('sidecar-genre').value,
      bpm: $('sidecar-bpm').value,
      key: $('sidecar-key').value,
      signature: $('sidecar-signature').value,
      tags: $('sidecar-tags').value,
      custom_tag: $('sidecar-trigger').value,
      lyrics: $('sidecar-lyrics').value,
      is_instrumental: $('sidecar-instrumental').checked,
    };

    // Warn if all text fields are empty
    const allEmpty = !data.caption && !data.genre && !data.bpm && !data.key &&
      !data.signature && !data.tags && !data.custom_tag && !data.lyrics;
    if (allEmpty && typeof WorkspaceBehaviors !== 'undefined' && WorkspaceBehaviors.showConfirmModal) {
      await new Promise((resolve) => {
        WorkspaceBehaviors.showConfirmModal('Empty Sidecar', 'All fields are empty. Save anyway?', 'Save', resolve);
      });
    }

    const savePath = _sidecarPath(_currentFile);
    try {
      await API.writeSidecar(savePath, data);
    } catch (e) {
      if (typeof showToast === 'function') showToast('Failed to save sidecar: ' + e.message, 'error');
      return;
    }

    _currentFile.has_sidecar = true;
    _currentFile.genre = data.genre;
    _currentFile.tags = data.tags;
    _currentFile.trigger = data.custom_tag;

    const savedName = _currentFile.name;
    closeEditor();
    await refreshFromSettings();

    if (typeof showToast === 'function') {
      showToast('Sidecar saved for ' + savedName, 'ok');
    }
  }

  /* ---- Bulk selection toolbar ---- */
  function _updateBulkToolbar() {
    const toolbar = $('dataset-bulk-toolbar'), countEl = $('dataset-bulk-count'), warnEl = $('dataset-bulk-warn');
    if (!toolbar) return;
    const selected = document.querySelectorAll('#dataset-tbody tr.selected');
    const folders = [...selected].filter((r) => r.classList.contains('dataset-folder-row'));
    const files = selected.length - folders.length;
    if (!selected.length) { toolbar.style.display = 'none'; return; }
    toolbar.style.display = '';
    if (countEl) countEl.textContent = selected.length + ' selected' + (folders.length ? ' (' + folders.length + ' folder' + (folders.length > 1 ? 's' : '') + ')' : '');
    if (warnEl) {
      const msg = folders.length && files ? 'Bulk actions apply to folders only' : (!folders.length && files ? 'Select folders for bulk actions' : '');
      warnEl.textContent = msg; warnEl.style.display = msg ? '' : 'none';
    }
  }

  function _getSelectedFolderPaths() {
    return [...document.querySelectorAll('#dataset-tbody tr.dataset-folder-row.selected')]
      .map((r) => r.querySelector('[data-folder]')?.dataset.folder).filter(Boolean);
  }

  function _initBulkActions() {
    const tbody = $('dataset-tbody');
    if (tbody) new MutationObserver(_updateBulkToolbar).observe(tbody, { attributes: true, attributeFilter: ['class'], subtree: true });
    $('dataset-bulk-clear')?.addEventListener('click', () => { document.querySelectorAll('#dataset-tbody tr.selected').forEach((r) => r.classList.remove('selected')); _updateBulkToolbar(); });
    $('dataset-bulk-trigger')?.addEventListener('click', () => {
      if (!_getSelectedFolderPaths().length) { if (typeof showToast === 'function') showToast('Select at least one folder', 'warn'); return; }
      $('trigger-tag-modal')?.classList.add('open');
    });
    $('dataset-bulk-preprocess')?.addEventListener('click', () => {
      const paths = _getSelectedFolderPaths();
      if (!paths.length) { if (typeof showToast === 'function') showToast('Select at least one folder', 'warn'); return; }
      const root = _scanRoot || _canonicalAudioPath();
      if (!root) { if (typeof showToast === 'function') showToast('No audio directory configured', 'warn'); return; }
      const fullPaths = paths.map(p => p === '.' ? root : _joinPath(root, p));
      if (fullPaths.length === 1) {
        _openPreprocessForFolder(paths[0]);
      } else {
        _setActiveLabPanel('preprocess');
        if (typeof WorkspaceLab !== 'undefined' && WorkspaceLab.queuePreprocess) {
          WorkspaceLab.queuePreprocess(fullPaths);
          showToast(fullPaths.length + ' folders queued for preprocessing', 'ok');
        } else {
          _openPreprocessForFolder(paths[0]);
        }
      }
    });
  }

  function init() {
    $('sidecar-editor-close')?.addEventListener('click', closeEditor);
    $('sidecar-cancel')?.addEventListener('click', closeEditor);
    $('sidecar-save')?.addEventListener('click', saveEditor);
    $('btn-refresh-audio-library')?.addEventListener('click', () => refreshFromSettings());

    document.addEventListener('sidestep:settings-saved', () => refreshFromSettings());

    _initBulkActions();
    refreshFromSettings();
  }

  async function refreshFromSettings() {
    const path = _canonicalAudioPath();
    await scan(path);
  }

  return { init, scan, openEditor, closeEditor, refreshFromSettings };

})();
