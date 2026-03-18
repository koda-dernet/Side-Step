/* ============================================================
   Side-Step GUI — Dataset Browser + Sidecar Editor
   Populates the dataset table from API.scanDataset() and
   handles the slide-out sidecar editor panel.
   ============================================================ */

const Dataset = (() => {

  const _esc = window._esc;
  let _files = [];
  let _folders = [];
  let _currentFile = null;
  let _scanRoot = '';
  let _lastScannedPath = '';
  const _expandedFolders = new Set();
  const _thumbCache = new Map();  // path → blob URL

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
    const total = Math.floor(seconds);
    const m = Math.floor(total / 60);
    const s = total % 60;
    return `${m}:${String(s).padStart(2, '0')}`;
  }

  function _fmtTotalDuration(seconds) {
    const total = Math.floor(seconds);
    const h = Math.floor(total / 3600);
    const m = Math.floor((total % 3600) / 60);
    const s = total % 60;
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

  function _updateRowVisibility(tbody) {
    tbody.querySelectorAll('tr[data-ancestors]').forEach((tr) => {
      try {
        const ancestors = JSON.parse(tr.dataset.ancestors);
        const visible = ancestors.every((a) => _expandedFolders.has(a));
        tr.style.display = visible ? '' : 'none';
      } catch (_) { /* ignore */ }
    });
    tbody.querySelectorAll('[data-action="toggle-folder"]').forEach((btn) => {
      btn.classList.toggle('open', _expandedFolders.has(btn.dataset.folder));
    });
  }

  // IntersectionObserver-based lazy thumbnail loading (viewport-only)
  let _thumbObserver = null;
  function _initThumbObserver(tbody) {
    if (_thumbObserver) _thumbObserver.disconnect();
    const scrollRoot = tbody.closest('.main') || null;
    _thumbObserver = new IntersectionObserver((entries) => {
      entries.forEach((entry) => {
        if (!entry.isIntersecting) return;
        const tr = entry.target;
        const img = tr.querySelector('.ap-row-thumb');
        if (!img || img.dataset.loaded) { _thumbObserver.unobserve(tr); return; }
        const url = img.dataset.coverUrl;
        if (!url) { _thumbObserver.unobserve(tr); return; }

        // Check blob cache first
        const cached = _thumbCache.get(url);
        if (cached) {
          img.src = cached;
          img.dataset.loaded = '1';
          const wrap = tr.querySelector('.ap-row-thumb-wrap');
          if (wrap) wrap.style.display = '';
          _thumbObserver.unobserve(tr);
          return;
        }

        // Fetch and cache as blob
        fetch(url).then(r => r.ok ? r.blob() : Promise.reject()).then(blob => {
          const blobUrl = URL.createObjectURL(blob);
          _thumbCache.set(url, blobUrl);
          img.src = blobUrl;
          img.dataset.loaded = '1';
          const wrap = tr.querySelector('.ap-row-thumb-wrap');
          if (wrap) wrap.style.display = '';
        }).catch(() => {
          img.dataset.loaded = '1';
          const wrap = tr.querySelector('.ap-row-thumb-wrap');
          if (wrap) wrap.style.display = 'none';
        });
        _thumbObserver.unobserve(tr);
      });
    }, { root: scrollRoot, rootMargin: '200px 0px' });

    tbody.querySelectorAll('tr.dataset-file-row').forEach((tr) => {
      const img = tr.querySelector('.ap-row-thumb');
      if (img && !img.dataset.loaded) _thumbObserver.observe(tr);
    });
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

    const appendFile = (f, depth, ancestorPaths) => {
      const idx = _files.findIndex((x) => x.path === f.path);
      const sidecarStatus = f.has_sidecar
        ? '<span class="status--ok">[ok] exists</span>'
        : '<span class="status--warn">missing</span>';
      const editLabel = f.has_sidecar ? 'Edit' : 'Create';
      const coverUrl = API.audioCoverUrl(f.path);
      const trFile = document.createElement('tr');
      trFile.className = 'dataset-file-row';
      trFile.dataset.apPath = f.path;
      trFile.dataset.ancestors = JSON.stringify(ancestorPaths);
      trFile.innerHTML = `
        <td>
          <button class="ap-row-btn" title="Play">&#9654;</button>
          <span class="ap-row-thumb-wrap" style="display:none;">
            <img class="ap-row-thumb" alt="">
            <span class="ap-row-thumb-scanlines"></span>
          </span>
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
      // Store cover URL for lazy loading (don't fetch until row is visible)
      const thumbImg = trFile.querySelector('.ap-row-thumb');
      if (thumbImg) {
        thumbImg.dataset.coverUrl = coverUrl;
      }
      // Play button handler
      const playBtn = trFile.querySelector('.ap-row-btn');
      if (playBtn) {
        const filePath = f.path;
        playBtn.addEventListener('click', (e) => {
          e.stopPropagation();
          _apBuildPlaylist();
          const pIdx = _apPlaylist.findIndex((t) => t.path === filePath);
          if (pIdx >= 0) {
            if (_apCurrentIdx === pIdx && _apAudio && !_apAudio.paused) {
              _apAudio.pause();
            } else {
              _apPlayIndex(pIdx);
            }
          }
        });
      }
      tbody.appendChild(trFile);
    };

    const appendFolder = (folder, ancestorPaths) => {
      const folderPath = folder.path || '.';
      const hasChildren = folder.children.length > 0 || folder.files.length > 0;
      const duration = Number(folder.total_duration || 0);
      const sidecars = Number(folder.sidecar_count || folder.files.filter((f) => f.has_sidecar).length || 0);
      const count = Number(folder.file_count || folder.files.length || 0);
      const depth = Math.max(0, Number(folder.depth || 0));

      // Ancestor chain for this folder's row (not including self)
      const myAncestors = folderPath === '.' ? [] : ancestorPaths;

      if (folderPath !== '.') {
        const tr = document.createElement('tr');
        tr.className = 'dataset-folder-row';
        tr.dataset.ancestors = JSON.stringify(myAncestors);
        tr.innerHTML = `
          <td>
            <span class="dataset-folder-indent" style="margin-left:${depth * 12}px;"></span>
            <button class="dataset-folder-toggle ${_expandedFolders.has(folderPath) ? 'open' : ''}" data-action="toggle-folder" data-folder="${_esc(folderPath)}" ${hasChildren ? '' : 'disabled'}>${_esc(folder.name || folderPath)}</button>
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

      // Children need this folder in their ancestor chain (unless root)
      const childAncestors = folderPath === '.' ? [] : [...ancestorPaths, folderPath];

      folder.children.forEach((child) => appendFolder(child, childAncestors));
      folder.files.forEach((f) => appendFile(f, folderPath === '.' ? 0 : depth + 1, childAncestors));
    };

    appendFolder(map.get('.'), []);

    // Apply initial visibility based on _expandedFolders
    _updateRowVisibility(tbody);

    // Start IntersectionObserver for lazy thumbnail loading
    _initThumbObserver(tbody);

    // Folder toggle — just flip visibility, no DOM rebuild
    tbody.querySelectorAll('[data-action="toggle-folder"]').forEach((btn) => {
      btn.addEventListener('click', () => {
        const path = btn.dataset.folder || '.';
        if (_expandedFolders.has(path)) _expandedFolders.delete(path);
        else _expandedFolders.add(path);
        _updateRowVisibility(tbody);
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
    } catch (e) {
      console.error('[Dataset] scan failed:', e);
      result = { files: [], sidecar_count: 0, total_duration: 0, error: 'Scan failed: ' + (e.message || e) };
    }

    _files = result.files || [];
    _folders = _normalizeFolders(result);
    // Only reset expanded folders when scanning a different directory
    if (targetPath !== _lastScannedPath) {
      _expandedFolders.clear();
      _thumbCache.clear();
    }
    _lastScannedPath = targetPath;

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
    if (typeof initShiftClickTable === 'function') initShiftClickTable('dataset-tbody', { requireModifier: true });
    document.dispatchEvent(new CustomEvent('sidestep:dataset-scanned', { detail: { fileCount: _files.length } }));
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
      const confirmed = await new Promise((resolve) => {
        WorkspaceBehaviors.showConfirmModal('Empty Sidecar', 'All fields are empty. Save anyway?', 'Save',
          () => resolve(true), () => resolve(false));
      });
      if (!confirmed) return;
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
    const mixBtn = $('dataset-bulk-mix');
    if (!toolbar) return;
    const selected = document.querySelectorAll('#dataset-tbody tr.selected');
    const folders = [...selected].filter((r) => r.classList.contains('dataset-folder-row'));
    const files = selected.length - folders.length;
    if (!selected.length) { toolbar.style.display = 'none'; return; }
    toolbar.style.display = '';
    const parts = [];
    if (folders.length) parts.push(folders.length + ' folder' + (folders.length > 1 ? 's' : ''));
    if (files) parts.push(files + ' file' + (files > 1 ? 's' : ''));
    if (countEl) countEl.textContent = parts.join(', ') + ' selected';
    if (mixBtn) mixBtn.style.display = (files > 0 || folders.length > 0) ? '' : 'none';
    if (warnEl) {
      warnEl.textContent = '';
      warnEl.style.display = 'none';
    }
  }

  function _getSelectedFolderPaths() {
    return [...document.querySelectorAll('#dataset-tbody tr.dataset-folder-row.selected')]
      .map((r) => r.querySelector('[data-folder]')?.dataset.folder).filter(Boolean);
  }

  function _getSelectedFilePaths() {
    return [...document.querySelectorAll('#dataset-tbody tr.dataset-file-row.selected')]
      .map((r) => {
        const idx = parseInt(r.querySelector('.sidecar-edit-btn')?.dataset.idx, 10);
        return Number.isFinite(idx) && _files[idx] ? _files[idx].path : null;
      }).filter(Boolean);
  }

  function getSelectedAudioPaths() {
    const filePaths = new Set(_getSelectedFilePaths());
    const folderPaths = _getSelectedFolderPaths();
    if (folderPaths.length) {
      _files.forEach((f) => {
        const fp = String(f.folder_path || '.');
        if (folderPaths.includes(fp) || folderPaths.some((sel) => fp.startsWith(sel + '/'))) {
          filePaths.add(f.path);
        }
      });
    }
    return [...filePaths];
  }

  function hasSelection() {
    return document.querySelectorAll('#dataset-tbody tr.selected').length > 0;
  }

  function _initBulkActions() {
    const tbody = $('dataset-tbody');
    if (tbody) new MutationObserver(_updateBulkToolbar).observe(tbody, { attributes: true, attributeFilter: ['class'], subtree: true });
    $('dataset-bulk-clear')?.addEventListener('click', () => { document.querySelectorAll('#dataset-tbody tr.selected').forEach((r) => r.classList.remove('selected')); _updateBulkToolbar(); });
    $('dataset-bulk-trigger')?.addEventListener('click', () => {
      if (!_getSelectedFolderPaths().length && !_getSelectedFilePaths().length) { if (typeof showToast === 'function') showToast('Select files or folders first', 'warn'); return; }
      if (typeof WorkspaceLab !== 'undefined' && WorkspaceLab._updateTriggerModalScope) WorkspaceLab._updateTriggerModalScope();
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

    $('dataset-bulk-mix')?.addEventListener('click', async () => {
      const filePaths = getSelectedAudioPaths();
      if (!filePaths.length) { if (typeof showToast === 'function') showToast('Select files or folders to create a mix', 'warn'); return; }
      const defaultName = 'mix_' + filePaths.length + '_tracks';
      const mixName = typeof WorkspaceBehaviors !== 'undefined' && WorkspaceBehaviors.showPromptModal
        ? await WorkspaceBehaviors.showPromptModal('Create Mix Dataset', 'Name for the mix dataset:', defaultName)
        : prompt('Name for the mix dataset:', defaultName);
      if (!mixName) return;
      // Ask about sidecar handling
      let sidecarMode = 'empty';
      const hasSidecars = filePaths.some((fp) => {
        const f = _files.find((x) => x.path === fp);
        return f && f.has_sidecar;
      });
      if (hasSidecars && typeof WorkspaceBehaviors !== 'undefined' && WorkspaceBehaviors.showConfirmModal) {
        sidecarMode = await new Promise((resolve) => {
          WorkspaceBehaviors.showConfirmModal(
            'Sidecar Metadata',
            'Some selected tracks have existing sidecar metadata (.txt). Copy them into the mix, or start fresh with empty sidecars?',
            'Copy Existing',
            () => resolve('copy'),
            () => resolve('empty'),
          );
        });
      }
      const root = _scanRoot || _canonicalAudioPath();
      const destRoot = $('settings-audio-dir')?.value || root || '.';
      try {
        const result = await API.createMixDataset(root, destRoot, mixName, filePaths, sidecarMode);
        if (result.ok) {
          showToast('Mix dataset created: ' + mixName + ' (' + result.created + ' files)', 'ok');
          await refreshFromSettings();
          if (typeof WorkspaceSetup !== 'undefined') WorkspaceSetup.populatePickers();
        } else {
          showToast('Mix failed: ' + (result.error || 'unknown'), 'error');
        }
      } catch (e) {
        showToast('Mix failed: ' + e.message, 'error');
      }
    });
  }

  /* ============================================================
     Audio Player — integrated mini player
     ============================================================ */

  let _apAudio = null;
  let _apPlaylist = [];
  let _apCurrentIdx = -1;
  let _apAutoPlay = false;
  let _apSeekDragging = false;

  // Web Audio API — analyser for dynamic bloom
  let _apAudioCtx = null;
  let _apAnalyser = null;
  let _apSourceNode = null;
  let _apFreqData = null;
  let _apBloomRAF = null;
  let _apBloomSmooth = 0;

  function _apFmtTime(s) {
    if (!Number.isFinite(s) || s < 0) return "0:00";
    const m = Math.floor(s / 60);
    const sec = Math.floor(s % 60);
    return m + ":" + String(sec).padStart(2, "0");
  }

  function _apUpdateSeekUI(pct) {
    const fill = $("ap-seek-fill");
    const cursor = $("ap-seek-cursor");
    if (fill) fill.style.width = pct + "%";
    if (cursor) cursor.style.left = "calc(" + pct + "% - 1px)";
  }

  // --- Marquee: start/stop scroll based on overflow ---
  let _apMarqueeName = "";
  function _apUpdateMarquee(name) {
    const wrap = $("ap-marquee-wrap");
    const el = $("ap-track-name");
    if (!wrap || !el) return;
    if (name !== undefined) _apMarqueeName = name;
    el.classList.remove("ap-marquee--scroll");
    el.style.removeProperty("--ap-marquee-dur");
    el.style.removeProperty("--ap-marquee-dist");
    // Set single name first to measure
    el.textContent = _apMarqueeName;
    requestAnimationFrame(() => {
      const singleW = el.scrollWidth;
      const wrapW = wrap.clientWidth;
      if (singleW > wrapW) {
        // Duplicate with spacer for seamless loop
        const spacer = "\u00a0\u00a0\u00a0\u2022\u00a0\u00a0\u00a0";
        el.textContent = _apMarqueeName + spacer + _apMarqueeName;
        requestAnimationFrame(() => {
          const spacerW = el.scrollWidth - singleW * 2;
          const scrollDist = singleW + spacerW;
          const speed = 40; // px/s
          const dur = scrollDist / speed;
          el.style.setProperty("--ap-marquee-dur", dur + "s");
          el.style.setProperty("--ap-marquee-dist", "-" + scrollDist + "px");
          el.classList.add("ap-marquee--scroll");
        });
      }
    });
  }

  // --- Web Audio analyser for dynamic bloom ---
  let _apPrevBins = null;
  let _apTimeData = null;
  let _apBloomLo = 0;
  let _apBloomHi = 0;

  // ═══ BLOOM TUNER — all knobs in one place ═══
  const _BT = {
    subW:    1.1,   // sub-bass (20–120 Hz) weight for LO envelope
    lmW:     0,     // low-mid (120–500 Hz) weight for LO envelope
    hmW:     0.4,   // hi-mid (500–4 kHz) weight for HI envelope
    hiW:     0.4,   // high (4–12 kHz) weight for HI envelope
    loFluxW: 1.0,   // low-band flux multiplier
    hiFluxW: 0.5,   // high-band flux multiplier
    peakW:   0,     // waveform peak added to LO envelope
    loAtk:   0.1,   // lo attack speed (0=slow, 1=instant)
    loDcy:   0,     // lo decay hold (0=instant drop, 0.99=long sustain)
    hiAtk:   0.09,  // hi attack speed
    hiDcy:   0,     // hi decay hold (lower = faster fade)
    fftSize: 512,
    smooth:  0,     // analyser smoothingTimeConstant
  };

  function _apInitAnalyser() {
    if (_apAnalyser) return;
    try {
      _apAudioCtx = new (window.AudioContext || window.webkitAudioContext)();
      _apAnalyser = _apAudioCtx.createAnalyser();
      _apAnalyser.fftSize = _BT.fftSize;
      _apAnalyser.smoothingTimeConstant = _BT.smooth;
      _apAnalyser.minDecibels = -90;
      _apAnalyser.maxDecibels = -10;
      _apFreqData = new Uint8Array(_apAnalyser.frequencyBinCount);
      _apTimeData = new Uint8Array(_apAnalyser.fftSize);
      _apPrevBins = new Float32Array(_apAnalyser.frequencyBinCount);
      _apSourceNode = _apAudioCtx.createMediaElementSource(_apAudio);
      _apSourceNode.connect(_apAnalyser);
      _apAnalyser.connect(_apAudioCtx.destination);
    } catch (e) {
      console.warn("[AudioPlayer] Web Audio API unavailable:", e);
      _apAnalyser = null;
    }
  }

  function _apBloomApply() {
    const bar = $("ap-bar");
    if (!bar) return;
    const bloom = Math.min(1, Math.max(_apBloomLo, _apBloomHi));
    bar.style.setProperty("--ap-bloom", bloom.toFixed(3));
  }

  function _apBloomLoop() {
    if (!_apAnalyser || !_apAudio || _apAudio.paused) {
      _apBloomLo *= _BT.loDcy;
      _apBloomHi *= _BT.hiDcy;
      // Decay EQ bars
      const els = _apEqGetEls();
      let anyEq = false;
      for (let i = 0; i < 7; i++) {
        _eqSmooth[i] *= 0.85;
        if (_eqSmooth[i] < 0.005) _eqSmooth[i] = 0;
        else anyEq = true;
        if (els[i]) els[i].style.height = (_eqSmooth[i] * 100).toFixed(1) + "%";
      }
      if (_apBloomLo < 0.005 && _apBloomHi < 0.005 && !anyEq) {
        _apBloomLo = 0; _apBloomHi = 0; _apBloomSmooth = 0;
        _apBloomApply();
        _apBloomRAF = null;
        return;
      }
      _apBloomApply();
      _apBloomRAF = requestAnimationFrame(_apBloomLoop);
      return;
    }

    const bins = _apAnalyser.frequencyBinCount;
    const sr = _apAudioCtx.sampleRate;
    const binHz = sr / _apAnalyser.fftSize;

    // Waveform peak
    _apAnalyser.getByteTimeDomainData(_apTimeData);
    let peak = 0;
    for (let i = 0; i < _apTimeData.length; i++) {
      const s = Math.abs((_apTimeData[i] - 128) / 128);
      if (s > peak) peak = s;
    }

    // Frequency band energy
    _apAnalyser.getByteFrequencyData(_apFreqData);
    const subEnd = Math.min(bins, Math.ceil(120 / binHz));
    const lmEnd  = Math.min(bins, Math.ceil(500 / binHz));
    const hmEnd  = Math.min(bins, Math.ceil(4000 / binHz));
    const hiEnd  = Math.min(bins, Math.ceil(12000 / binHz));

    let subSum = 0, lmSum = 0, hmSum = 0, hiSum = 0;
    for (let i = 0; i < hiEnd; i++) {
      const v = _apFreqData[i] / 255;
      if (i < subEnd) subSum += v;
      else if (i < lmEnd) lmSum += v;
      else if (i < hmEnd) hmSum += v;
      else hiSum += v;
    }
    const subE = subEnd > 0 ? subSum / subEnd : 0;
    const lmE  = (lmEnd - subEnd) > 0 ? lmSum / (lmEnd - subEnd) : 0;
    const hmE  = (hmEnd - lmEnd) > 0 ? hmSum / (hmEnd - lmEnd) : 0;
    const hiE  = (hiEnd - hmEnd) > 0 ? hiSum / (hiEnd - hmEnd) : 0;

    // Spectral flux per band
    let loFlux = 0, hiFlux = 0;
    for (let i = 0; i < bins; i++) {
      const cur = _apFreqData[i] / 255;
      const diff = cur - _apPrevBins[i];
      if (diff > 0) {
        if (i < lmEnd) loFlux += diff;
        else hiFlux += diff;
      }
      _apPrevBins[i] = cur;
    }
    loFlux /= lmEnd || 1;
    hiFlux /= (bins - lmEnd) || 1;

    // LOW envelope
    const loTarget = Math.min(1,
      subE * _BT.subW + lmE * _BT.lmW + loFlux * _BT.loFluxW + peak * _BT.peakW
    );
    if (loTarget > _apBloomLo) {
      _apBloomLo = _apBloomLo * (1 - _BT.loAtk) + loTarget * _BT.loAtk;
    } else {
      _apBloomLo = _apBloomLo * _BT.loDcy + loTarget * (1 - _BT.loDcy);
    }

    // HIGH envelope
    const hiTarget = Math.min(1,
      hmE * _BT.hmW + hiE * _BT.hiW + hiFlux * _BT.hiFluxW
    );
    if (hiTarget > _apBloomHi) {
      _apBloomHi = _apBloomHi * (1 - _BT.hiAtk) + hiTarget * _BT.hiAtk;
    } else {
      _apBloomHi = _apBloomHi * _BT.hiDcy + hiTarget * (1 - _BT.hiDcy);
    }

    // Update tuner meters if visible
    if (_btPanel) _btUpdateMeters(subE, lmE, hmE, hiE, loFlux, hiFlux, peak);

    // 7-band EQ visualizer
    _apEqUpdate(binHz, bins);

    _apBloomApply();
    _apBloomRAF = requestAnimationFrame(_apBloomLoop);
  }

  // ═══ 7-BAND EQ VISUALIZER ═══
  const _EQ_BANDS = [63, 160, 400, 1000, 2500, 6300, 16000]; // Hz center freqs
  const _eqSmooth = new Float32Array(7); // smoothed levels
  let _eqEls = null; // cached fill elements

  function _apEqGetEls() {
    if (_eqEls) return _eqEls;
    _eqEls = [];
    for (let i = 0; i < 7; i++) {
      const col = document.getElementById("ap-eq-" + i);
      if (col) _eqEls.push(col.querySelector(".ap-eq-fill"));
      else _eqEls.push(null);
    }
    return _eqEls;
  }

  function _apEqUpdate(binHz, bins) {
    const els = _apEqGetEls();
    for (let b = 0; b < 7; b++) {
      const fc = _EQ_BANDS[b];
      // Band range: half-octave below to half-octave above center
      const lo = Math.max(0, Math.floor((fc / 1.414) / binHz));
      const hi = Math.min(bins - 1, Math.ceil((fc * 1.414) / binHz));
      let sum = 0, count = 0;
      for (let i = lo; i <= hi; i++) {
        sum += _apFreqData[i] / 255;
        count++;
      }
      const raw = count > 0 ? sum / count : 0;
      // Light smoothing for fluid motion
      _eqSmooth[b] = _eqSmooth[b] * 0.3 + raw * 0.7;
      if (els[b]) els[b].style.height = (_eqSmooth[b] * 100).toFixed(1) + "%";
    }
  }

  function _apEqZero() {
    const els = _apEqGetEls();
    for (let i = 0; i < 7; i++) {
      _eqSmooth[i] = 0;
      if (els[i]) els[i].style.height = "0%";
    }
  }

  // ═══ BLOOM TUNER PANEL (dev tool — toggled with Ctrl+Shift+B) ═══
  let _btPanel = null;
  const _btFields = [
    { key: "subW",    label: "sub weight",    min: 0, max: 10, step: 0.1 },
    { key: "lmW",     label: "low-mid weight", min: 0, max: 10, step: 0.1 },
    { key: "hmW",     label: "hi-mid weight",  min: 0, max: 10, step: 0.1 },
    { key: "hiW",     label: "high weight",    min: 0, max: 10, step: 0.1 },
    { key: "loFluxW", label: "lo flux",        min: 0, max: 20, step: 0.5 },
    { key: "hiFluxW", label: "hi flux",        min: 0, max: 20, step: 0.5 },
    { key: "peakW",   label: "peak contrib",   min: 0, max: 5,  step: 0.1 },
    { key: "loAtk",   label: "lo attack",      min: 0, max: 1,  step: 0.01 },
    { key: "loDcy",   label: "lo decay",       min: 0, max: 0.99, step: 0.01 },
    { key: "hiAtk",   label: "hi attack",      min: 0, max: 1,  step: 0.01 },
    { key: "hiDcy",   label: "hi decay",       min: 0, max: 0.99, step: 0.01 },
    { key: "smooth",  label: "fft smooth",     min: 0, max: 0.95, step: 0.05 },
  ];

  function _btBuildPanel() {
    if (_btPanel) { _btPanel.remove(); _btPanel = null; return; }
    const p = document.createElement("div");
    p.id = "bloom-tuner";
    p.style.cssText = "position:fixed;top:10px;right:10px;z-index:99999;width:280px;"
      + "background:#0e1016;border:1px solid rgba(92,207,230,0.3);border-radius:3px;"
      + "padding:8px 10px;font:11px/1.5 var(--font-mono,monospace);color:#8995bc;"
      + "box-shadow:0 4px 20px rgba(0,0,0,0.6);max-height:90vh;overflow-y:auto;";
    let html = '<div style="color:#5ccfe6;margin-bottom:6px;font-size:12px;">[ bloom tuner ]</div>';
    // Live meters
    html += '<div id="bt-meters" style="display:flex;gap:4px;margin-bottom:8px;font-size:9px;">'
      + '<span>lo:<b id="bt-m-lo" style="color:#5ccfe6">0</b></span>'
      + '<span>hi:<b id="bt-m-hi" style="color:#5ccfe6">0</b></span>'
      + '<span>bloom:<b id="bt-m-bl" style="color:#5ccfe6">0</b></span>'
      + '</div>';
    // Band energy meters
    html += '<div id="bt-bands" style="display:flex;gap:3px;margin-bottom:8px;height:24px;align-items:flex-end;">'
      + '<div id="bt-b-sub" style="flex:1;background:#5ccfe6;min-height:1px;" title="sub"></div>'
      + '<div id="bt-b-lm" style="flex:1;background:#5ccfe6;min-height:1px;" title="lo-mid"></div>'
      + '<div id="bt-b-hm" style="flex:1;background:#5ccfe6;min-height:1px;" title="hi-mid"></div>'
      + '<div id="bt-b-hi" style="flex:1;background:#5ccfe6;min-height:1px;" title="high"></div>'
      + '<div id="bt-b-lf" style="flex:1;background:#8995bc;min-height:1px;" title="lo-flux"></div>'
      + '<div id="bt-b-hf" style="flex:1;background:#8995bc;min-height:1px;" title="hi-flux"></div>'
      + '<div id="bt-b-pk" style="flex:1;background:#e6d75c;min-height:1px;" title="peak"></div>'
      + '</div>';
    // Sliders
    _btFields.forEach(f => {
      html += '<div style="display:flex;align-items:center;gap:4px;margin-bottom:3px;">'
        + '<label style="width:95px;flex-shrink:0;font-size:10px;">' + f.label + '</label>'
        + '<input type="range" data-bt="' + f.key + '" min="' + f.min + '" max="' + f.max
        + '" step="' + f.step + '" value="' + _BT[f.key]
        + '" style="flex:1;height:3px;accent-color:#5ccfe6;cursor:pointer;">'
        + '<span data-btv="' + f.key + '" style="width:36px;text-align:right;font-size:10px;color:#5ccfe6;">'
        + _BT[f.key] + '</span>'
        + '</div>';
    });
    // Copy button
    html += '<button id="bt-copy" style="margin-top:8px;width:100%;padding:4px;'
      + 'background:rgba(92,207,230,0.08);border:1px solid rgba(92,207,230,0.3);'
      + 'border-radius:2px;color:#5ccfe6;font:10px var(--font-mono,monospace);cursor:pointer;">'
      + '[ copy config ]</button>';
    p.innerHTML = html;
    document.body.appendChild(p);
    _btPanel = p;

    // Wire sliders
    p.querySelectorAll("input[data-bt]").forEach(inp => {
      inp.addEventListener("input", () => {
        const key = inp.dataset.bt;
        _BT[key] = parseFloat(inp.value);
        p.querySelector("[data-btv='" + key + "']").textContent = inp.value;
        // Live-update analyser smoothing
        if (key === "smooth" && _apAnalyser) {
          _apAnalyser.smoothingTimeConstant = _BT.smooth;
        }
      });
    });

    // Copy config
    p.querySelector("#bt-copy").addEventListener("click", () => {
      const cfg = {};
      _btFields.forEach(f => { cfg[f.key] = _BT[f.key]; });
      const txt = JSON.stringify(cfg, null, 2);
      navigator.clipboard.writeText(txt).then(() => {
        const btn = p.querySelector("#bt-copy");
        btn.textContent = "[ copied! ]";
        setTimeout(() => { btn.textContent = "[ copy config ]"; }, 1500);
      });
    });
  }

  function _btUpdateMeters(subE, lmE, hmE, hiE, loFlux, hiFlux, peak) {
    const s = (id, v) => { const el = document.getElementById(id); if (el) el.style.height = Math.round(v * 100) + "%"; };
    s("bt-b-sub", subE); s("bt-b-lm", lmE); s("bt-b-hm", hmE); s("bt-b-hi", hiE);
    s("bt-b-lf", loFlux * 2); s("bt-b-hf", hiFlux * 2); s("bt-b-pk", peak);
    const mlo = document.getElementById("bt-m-lo");
    const mhi = document.getElementById("bt-m-hi");
    const mbl = document.getElementById("bt-m-bl");
    if (mlo) mlo.textContent = _apBloomLo.toFixed(2);
    if (mhi) mhi.textContent = _apBloomHi.toFixed(2);
    if (mbl) mbl.textContent = Math.min(1, Math.max(_apBloomLo, _apBloomHi)).toFixed(2);
  }

  // Toggle with Ctrl+Shift+B
  document.addEventListener("keydown", (e) => {
    if (e.ctrlKey && e.shiftKey && e.key === "B") {
      e.preventDefault();
      _btBuildPanel();
    }
  });

  function _apStartBloom() {
    if (!_apBloomRAF) {
      _apBloomRAF = requestAnimationFrame(_apBloomLoop);
    }
  }

  function _apInitAudio() {
    if (_apAudio) return _apAudio;
    _apAudio = new Audio();
    _apAudio.preload = "metadata";
    _apAudio.volume = 0.8;
    _apAudio.crossOrigin = "anonymous";

    _apAudio.addEventListener("timeupdate", () => {
      if (_apSeekDragging) return;
      const cur = $("ap-time-current");
      if (Number.isFinite(_apAudio.duration) && _apAudio.duration > 0) {
        const pct = (_apAudio.currentTime / _apAudio.duration) * 100;
        _apUpdateSeekUI(pct);
      }
      if (cur) cur.textContent = _apFmtTime(_apAudio.currentTime);
    });

    _apAudio.addEventListener("loadedmetadata", () => {
      const dur = $("ap-time-duration");
      if (dur) dur.textContent = _apFmtTime(_apAudio.duration);
      _apUpdateSeekUI(0);
    });

    _apAudio.addEventListener("ended", () => {
      _apSetPlayState(false);
      if (_apAutoPlay && _apCurrentIdx < _apPlaylist.length - 1) {
        _apPlayIndex(_apCurrentIdx + 1);
      }
    });

    _apAudio.addEventListener("play", () => {
      _apSetPlayState(true);
      // Lazily initialise analyser on first play
      _apInitAnalyser();
      if (_apAudioCtx && _apAudioCtx.state === "suspended") {
        _apAudioCtx.resume();
      }
      _apStartBloom();
    });
    _apAudio.addEventListener("pause", () => {
      _apSetPlayState(false);
      // Bloom loop will decay naturally
    });

    return _apAudio;
  }

  function _apSetPlayState(playing) {
    const btn = $("ap-play");
    if (btn) btn.textContent = playing ? "[||]" : "[>]";
    const currentPath = (_apCurrentIdx >= 0 && _apCurrentIdx < _apPlaylist.length) ? _apPlaylist[_apCurrentIdx].path : "";
    document.querySelectorAll("#dataset-tbody .ap-row-btn").forEach((b) => {
      const row = b.closest("tr");
      const rowPath = row ? row.dataset.apPath : "";
      const isActive = currentPath && rowPath === currentPath;
      b.textContent = (isActive && playing) ? "\u275A\u275A" : "\u25B6";
      b.classList.toggle("ap-row-btn--active", isActive);
    });
  }

  function _apLoadCover(filePath) {
    const wrap = $("ap-cover-wrap");
    const img = $("ap-cover");
    if (!img || !wrap) return;
    const url = API.audioCoverUrl(filePath);
    img.src = url;
    img.onerror = () => { wrap.style.display = "none"; };
    img.onload = () => { wrap.style.display = ""; };
  }

  function _apBuildPlaylist() {
    _apPlaylist = [];
    const rows = document.querySelectorAll("#dataset-tbody tr.dataset-file-row");
    rows.forEach((row, i) => {
      const path = row.dataset.apPath || "";
      if (!path) return;
      const nameSpan = row.querySelector("td:first-child span[title]");
      const name = nameSpan ? (nameSpan.getAttribute("title") || nameSpan.textContent.trim()) : "Track " + (i + 1);
      _apPlaylist.push({ path, name, rowIndex: i });
    });
  }

  function _apPlayIndex(idx) {
    if (idx < 0 || idx >= _apPlaylist.length) return;
    const audio = _apInitAudio();
    const track = _apPlaylist[idx];
    _apCurrentIdx = idx;

    audio.src = API.audioStreamUrl(track.path);
    audio.play().catch(() => {});

    _apUpdateMarquee(track.name);
    _apLoadCover(track.path);

    const bar = $("ap-bar");
    if (bar) bar.classList.remove("ap-bar--standby");
  }

  function _apTogglePlay() {
    if (!_apAudio) return;
    if (_apAudio.paused) _apAudio.play().catch(() => {});
    else _apAudio.pause();
  }

  function _apPrev() {
    if (_apCurrentIdx > 0) _apPlayIndex(_apCurrentIdx - 1);
  }

  function _apNext() {
    if (_apCurrentIdx < _apPlaylist.length - 1) _apPlayIndex(_apCurrentIdx + 1);
  }

  function _apSeekFromEvent(e, el) {
    const rect = el.getBoundingClientRect();
    const pct = Math.max(0, Math.min(100, ((e.clientX - rect.left) / rect.width) * 100));
    _apUpdateSeekUI(pct);
    if (_apAudio && Number.isFinite(_apAudio.duration)) {
      _apAudio.currentTime = (_apAudio.duration * pct) / 100;
    }
  }

  function _apVolFromEvent(e, el) {
    const rect = el.getBoundingClientRect();
    const pct = Math.max(0, Math.min(100, ((e.clientX - rect.left) / rect.width) * 100));
    const fill = $("ap-vol-fill");
    if (fill) fill.style.width = pct + "%";
    if (_apAudio) _apAudio.volume = pct / 100;
  }

  function _apInit() {
    // Custom seek bar interactions
    const seekTrack = $("ap-seek-track");
    if (seekTrack) {
      seekTrack.addEventListener("mousedown", (e) => {
        _apSeekDragging = true;
        _apSeekFromEvent(e, seekTrack);
        const onMove = (ev) => _apSeekFromEvent(ev, seekTrack);
        const onUp = () => {
          _apSeekDragging = false;
          document.removeEventListener("mousemove", onMove);
          document.removeEventListener("mouseup", onUp);
        };
        document.addEventListener("mousemove", onMove);
        document.addEventListener("mouseup", onUp);
      });
    }

    // Custom volume bar interactions
    const volTrack = $("ap-vol-track");
    if (volTrack) {
      volTrack.addEventListener("mousedown", (e) => {
        _apVolFromEvent(e, volTrack);
        const onMove = (ev) => _apVolFromEvent(ev, volTrack);
        const onUp = () => {
          document.removeEventListener("mousemove", onMove);
          document.removeEventListener("mouseup", onUp);
        };
        document.addEventListener("mousemove", onMove);
        document.addEventListener("mouseup", onUp);
      });
    }

    // Transport buttons
    $("ap-play")?.addEventListener("click", _apTogglePlay);
    $("ap-prev")?.addEventListener("click", _apPrev);
    $("ap-next")?.addEventListener("click", _apNext);

    // Autoplay bracket toggle
    const apToggle = $("ap-autoplay");
    if (apToggle) {
      apToggle.addEventListener("click", () => {
        _apAutoPlay = !_apAutoPlay;
        apToggle.textContent = _apAutoPlay ? "[x] auto" : "[ ] auto";
        apToggle.classList.toggle("ap-auto--on", _apAutoPlay);
      });
    }

    // Dock system
    _apDockInit();

  }

  // ═══ DOCK / UNDOCK / HIDE / SHOW / CLOSE ═══
  let _apActivated = false;  // true after first Audio Library visit
  let _apDocked = false;
  let _apClosed = false;
  let _apHidden = false;
  let _apHomeVisible = false;

  function _apActivate() {
    // Show the player for the first time
    if (_apActivated) return;
    _apActivated = true;
    const home = $("ap-home");
    if (home) home.style.display = "";
  }

  function _apDockInit() {
    const bar = $("ap-bar");
    const home = $("ap-home");
    const dock = $("ap-dock");
    if (!bar || !home || !dock) return;

    // IntersectionObserver: watches the home slot inside .main scroll container
    const scrollRoot = document.querySelector(".main");
    const observer = new IntersectionObserver((entries) => {
      _apHomeVisible = entries[0].isIntersecting;
      _apDockUpdate();
    }, { root: scrollRoot, threshold: 0.1 });
    observer.observe(home);

    // Hide button [--]
    const btnHide = $("ap-btn-hide");
    if (btnHide) {
      btnHide.addEventListener("click", () => {
        _apHidden = !_apHidden;
        bar.classList.toggle("ap-bar--hidden", _apHidden);
        btnHide.textContent = _apHidden ? "[+]" : "[--]";
        btnHide.title = _apHidden ? "Show player" : "Hide player";
      });
    }

    // Close button [x]
    const btnClose = $("ap-btn-close");
    if (btnClose) {
      btnClose.addEventListener("click", () => {
        _apClosed = true;
        _apDocked = false;
        bar.classList.add("ap-bar--closed");
        dock.classList.remove("active");
      });
    }
  }

  function _apDockUpdate() {
    if (!_apActivated || _apClosed) return;
    const bar = $("ap-bar");
    const home = $("ap-home");
    const dock = $("ap-dock");
    if (!bar || !home || !dock) return;

    if (_apHomeVisible) {
      // Undock: move player back to home
      if (_apDocked) {
        home.appendChild(bar);
        dock.classList.remove("active");
        _apDocked = false;
      }
    } else {
      // Dock: move player to sticky dock
      if (!_apDocked) {
        dock.appendChild(bar);
        dock.classList.add("active");
        _apDocked = true;
      }
    }
  }

  function _apReopen() {
    // Called when user visits Audio Library tab
    _apActivate();

    const bar = $("ap-bar");
    const home = $("ap-home");
    const dock = $("ap-dock");

    if (_apClosed && bar) {
      _apClosed = false;
      _apDocked = false;
      bar.classList.remove("ap-bar--closed");
      if (home) home.appendChild(bar);
      if (dock) dock.classList.remove("active");
    }

    // Explicitly check if home is in the viewport (observer may not have fired yet)
    if (home) {
      const scrollRoot = document.querySelector(".main");
      if (scrollRoot) {
        const rootRect = scrollRoot.getBoundingClientRect();
        const homeRect = home.getBoundingClientRect();
        _apHomeVisible = homeRect.top < rootRect.bottom && homeRect.bottom > rootRect.top;
      }
    }
    _apDockUpdate();
  }

  // ═══ BOOT-UP BEEP: car stereo "tee-too tee-too" ═══
  let _apBooted = false;

  function _apBootBeep() {
    if (_apBooted) return;
    _apBooted = true;
    try {
      const ctx = new (window.AudioContext || window.webkitAudioContext)();
      const vol = ctx.createGain();
      vol.gain.value = 0.08;
      vol.connect(ctx.destination);

      // Two pairs: tee-too ... tee-too
      // Each "tee" = high tone, "too" = low tone
      const tones = [
        { f: 1200, t: 0.00, d: 0.06 },  // tee
        { f: 800,  t: 0.08, d: 0.06 },  // too
        { f: 1200, t: 0.22, d: 0.06 },  // tee
        { f: 800,  t: 0.30, d: 0.06 },  // too
      ];

      tones.forEach(({ f, t, d }) => {
        const osc = ctx.createOscillator();
        const env = ctx.createGain();
        osc.type = "sine";
        osc.frequency.value = f;
        env.gain.setValueAtTime(0, ctx.currentTime + t);
        env.gain.linearRampToValueAtTime(1, ctx.currentTime + t + 0.005);
        env.gain.setValueAtTime(1, ctx.currentTime + t + d - 0.01);
        env.gain.linearRampToValueAtTime(0, ctx.currentTime + t + d);
        osc.connect(env);
        env.connect(vol);
        osc.start(ctx.currentTime + t);
        osc.stop(ctx.currentTime + t + d + 0.01);
      });

      // Close context after beeps finish
      setTimeout(() => ctx.close().catch(() => {}), 600);
    } catch (e) {
      // Web Audio not available, silent fail
    }
  }

  /* ============================================================
     Dataset init + exports
     ============================================================ */

  function init() {
    $('sidecar-editor-close')?.addEventListener('click', closeEditor);
    $('sidecar-cancel')?.addEventListener('click', closeEditor);
    $('sidecar-save')?.addEventListener('click', saveEditor);
    $('btn-refresh-audio-library')?.addEventListener('click', () => refreshFromSettings(true));

    // Single-file audio analysis from sidecar editor
    $('sidecar-analyze-btn')?.addEventListener('click', async () => {
      if (!_currentFile) return;
      const btn = $('sidecar-analyze-btn');
      const status = $('sidecar-analyze-status');
      btn.disabled = true;
      btn.textContent = '[...]';
      const mode = $('analyze-mode')?.value || 'mid';
      const chunks = parseInt($('analyze-chunks')?.value || '5', 10);
      const modeLabels = { faf: 'F-A-F (fast)', mid: 'Mid (ensemble)', sas: 'S-A-S (deep)' };
      if (status) { status.style.display = 'inline'; status.textContent = 'Running ' + (modeLabels[mode] || mode) + '...'; status.style.color = 'var(--muted)'; }

      try {
        const result = await API.analyzeOneFile(_currentFile.path, { mode, chunks });
        if (result.error) {
          if (status) { status.textContent = '[!] ' + result.error; status.style.color = 'var(--error)'; }
          if (typeof showToast === 'function') showToast('Analysis failed: ' + result.error, 'error');
        } else {
          const f = result.fields || {};
          const conf = f.confidence || {};
          if (f.bpm) $('sidecar-bpm').value = f.bpm;
          if (f.key) $('sidecar-key').value = f.key;
          if (f.signature) $('sidecar-signature').value = f.signature;
          // Build confidence summary for status display
          const parts = [];
          if (f.bpm) parts.push('bpm=' + f.bpm + (conf.bpm ? ' (' + conf.bpm + ')' : ''));
          if (f.key) parts.push('key=' + f.key + (conf.key ? ' (' + conf.key + ')' : ''));
          if (f.signature) parts.push('sig=' + f.signature + (conf.signature ? ' (' + conf.signature + ')' : ''));
          const confVals = Object.values(conf);
          const allHigh = confVals.length > 0 && confVals.every(c => c === 'high');
          const anyLow = confVals.some(c => c === 'low');
          if (status) {
            status.textContent = '[ok] ' + parts.join(', ');
            status.style.color = allHigh ? 'var(--success)' : anyLow ? 'var(--warning)' : 'var(--accent)';
          }
          if (typeof showToast === 'function') showToast('Audio analysis complete', 'ok');
        }
      } catch (e) {
        if (status) { status.textContent = '[!] ' + (e.message || e); status.style.color = 'var(--error)'; }
      } finally {
        btn.disabled = false;
        btn.textContent = 'Analyze';
      }
    });

    document.addEventListener('sidestep:settings-saved', () => refreshFromSettings(true));

    _initBulkActions();
    _apInit();
    refreshFromSettings();
  }

  async function refreshFromSettings(force) {
    const path = _canonicalAudioPath();
    // Skip re-scan if same path and we already have data (tab switch)
    if (!force && path && path === _lastScannedPath && _files.length > 0) return;
    await scan(path);
  }

  return { init, scan, openEditor, closeEditor, refreshFromSettings, getSelectedAudioPaths, hasSelection, bootBeep: _apBootBeep, reopen: _apReopen };

})();
