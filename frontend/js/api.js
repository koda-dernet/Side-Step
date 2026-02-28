/* ============================================================
   Side-Step GUI — API Layer
   fetch() + WebSocket calls to the FastAPI backend.
   ============================================================ */

const API = (() => {

  // ---- Auth token (lazy read to avoid race with inline script) ----------

  function _getToken() {
    return new URLSearchParams(window.location.search).get('token') ||
      (typeof window.__SIDESTEP_TOKEN__ !== 'undefined' ? window.__SIDESTEP_TOKEN__ : '') ||
      '';
  }

  function _authHeaders() {
    const t = _getToken();
    return t ? { 'Authorization': 'Bearer ' + t } : {};
  }

  // Diagnostic: log token presence and first failure once
  let _firstCall = true;
  let _firstFailureLogged = false;
  let _authFailureLogged = false;
  function _logTokenOnce() {
    if (_firstCall) {
      _firstCall = false;
      const t = _getToken();
      const fromUrl = !!new URLSearchParams(window.location.search).get('token');
      const fromInject = typeof window.__SIDESTEP_TOKEN__ !== 'undefined';
      console.log('[API] token present:', !!t, '| from URL:', fromUrl, '| from __SIDESTEP_TOKEN__:', fromInject);
      if (!t) {
        console.warn('[API] No auth token — all /api/* requests will return 401. GPU, models, datasets will show placeholders.');
      }
    }
  }

  function _logAuthFailure(url, status) {
    if (status === 401 && !_authFailureLogged) {
      _authFailureLogged = true;
      console.warn('[API] 401 Unauthorized — token missing or invalid. Check that the page loaded with ?token= or __SIDESTEP_TOKEN__ was injected.');
      document.dispatchEvent(new CustomEvent('sidestep:api-auth-failed', { detail: { url, status } }));
    }
  }

  // ---- Helpers ----------------------------------------------------------

  async function _get(url) {
    _logTokenOnce();
    const r = await fetch(url, { headers: _authHeaders() });
    if (!r.ok) {
      const t = await r.text().catch(() => '');
      _logAuthFailure(url, r.status);
      if (!_firstFailureLogged) {
        _firstFailureLogged = true;
        console.log('[API] first request failed:', url, r.status, t.slice(0, 100));
      }
      throw new Error(`GET ${url} ${r.status}: ${t.slice(0, 200)}`);
    }
    return r.json();
  }

  async function _post(url, body) {
    _logTokenOnce();
    const r = await fetch(url, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json', ..._authHeaders() },
      body: JSON.stringify(body),
    });
    if (!r.ok) {
      _logAuthFailure(url, r.status);
      const t = await r.text().catch(() => '');
      throw new Error(`POST ${url} ${r.status}: ${t.slice(0, 200)}`);
    }
    return r.json();
  }

  async function _del(url) {
    _logTokenOnce();
    const r = await fetch(url, { method: 'DELETE', headers: _authHeaders() });
    if (!r.ok) {
      _logAuthFailure(url, r.status);
      const t = await r.text().catch(() => '');
      throw new Error(`DELETE ${url} ${r.status}: ${t.slice(0, 200)}`);
    }
    return r.json();
  }

  // ---- Settings ---------------------------------------------------------

  async function fetchSettings() { return _get('/api/settings'); }
  async function saveSettings(data) { return _post('/api/settings', { data }); }

  // ---- GPU --------------------------------------------------------------

  async function fetchGPU() { return _get('/api/gpu'); }

  // ---- VRAM Estimation (calls backend Python estimation) ----------------

  async function estimateVRAM(config) {
    return _post('/api/vram/estimate', config);
  }

  // ---- Models -----------------------------------------------------------

  async function fetchModels(checkpointDir) {
    return _get('/api/models?checkpoint_dir=' + encodeURIComponent(checkpointDir || ''));
  }

  // ---- Dataset ----------------------------------------------------------

  async function scanDataset(path) { return _get('/api/dataset/scan?path=' + encodeURIComponent(path || '')); }
  async function readSidecar(filePath) { return _get('/api/sidecar?path=' + encodeURIComponent(filePath)); }
  async function writeSidecar(filePath, data) { return _post('/api/sidecar', { path: filePath, data }); }

  // ---- Presets ----------------------------------------------------------

  async function fetchPresets() { return _get('/api/presets'); }
  async function loadPreset(name) { return _get('/api/presets/' + encodeURIComponent(name)); }

  async function savePreset(name, description, data) {
    return _post('/api/presets', { name, data: { ...data, description } });
  }

  async function deletePreset(name) { return _del('/api/presets/' + encodeURIComponent(name)); }

  // ---- History ----------------------------------------------------------

  async function fetchHistory() { return _get('/api/history'); }
  async function fetchRunConfig(runName) { return _get('/api/history/' + encodeURIComponent(runName) + '/config'); }
  async function fetchRunLossCurve(runName) { return _get('/api/history/' + encodeURIComponent(runName) + '/curve'); }
  async function deleteHistoryFolder(path) { return _del('/api/history/folder?path=' + encodeURIComponent(path || '')); }

  // ---- Training ---------------------------------------------------------

  async function startTraining(config) { return _post('/api/train/start', { config }); }
  async function stopTraining() { return _post('/api/train/stop', {}); }

  // ---- Dataset Scanning -------------------------------------------------

  async function scanTensorsDir(tensorsDir) { return _get('/api/datasets?tensors_dir=' + encodeURIComponent(tensorsDir || '')); }

  // ---- Preprocessing ----------------------------------------------------

  async function scanAudioFolder(path) { return _get('/api/dataset/scan?path=' + encodeURIComponent(path || '')); }
  async function runPreprocess(config) { return _post('/api/preprocess/start', config); }

  // ---- PP++ (Fisher Analysis) -------------------------------------------

  async function checkFisherMap(datasetDir, modelVariant) {
    const q = new URLSearchParams({ dataset_dir: datasetDir || '' });
    if (modelVariant) q.set('model_variant', modelVariant);
    return _get('/api/fisher-map/status?' + q.toString());
  }

  async function runFisherAnalysis(config) { return _post('/api/ppplus/start', config); }

  // ---- Resume -----------------------------------------------------------

  async function scanCheckpoints(runName) {
    return _get('/api/checkpoints/' + encodeURIComponent(runName));
  }


  // ---- AI Captions ------------------------------------------------------

  async function validateApiKey(provider, key, opts) {
    return _post('/api/validate-key', {
      provider,
      key,
      model: opts?.model || null,
      base_url: opts?.base_url || null,
    });
  }

  async function runAICaptions(config) { return _post('/api/captions/start', config); }
  async function stopTask(taskId) { return _post('/api/task/' + encodeURIComponent(taskId) + '/stop', {}); }

  function signalShutdown() {
    // Use sendBeacon for reliable delivery during beforeunload
    const t = _getToken();
    const url = '/api/shutdown' + (t ? '?token=' + t : '');
    if (navigator.sendBeacon) {
      const blob = new Blob(['{}'], { type: 'application/json' });
      navigator.sendBeacon(url, blob);
    } else {
      fetch('/api/shutdown', { method: 'POST', headers: { 'Content-Type': 'application/json', ..._authHeaders() }, body: '{}', keepalive: true });
    }
  }

  async function startTensorBoard(logDir, port, outputDir) {
    return _post('/api/tensorboard/start', {
      log_dir: logDir || '',
      output_dir: outputDir || '',
      port: port || 6006,
    });
  }

  // ---- Trigger Tag Bulk -------------------------------------------------

  async function bulkWriteTriggerTag(datasetDir, tag, position) {
    // Scan for ALL audio files (not just those with existing sidecars)
    const scan = await scanDataset(datasetDir).catch(() => ({ files: [] }));
    const paths = (scan.files || [])
      .map(f => f.sidecar_path || f.path.replace(/\.[^.]+$/, '.txt'));
    return _post('/api/trigger-tag/bulk', { paths, tag, position });
  }

  // ---- CLI Export (delegated to APICli) ---------------------------------

  function buildCLICommand(config) {
    return typeof APICli !== 'undefined' ? APICli.buildCLICommand(config) : '';
  }

  // ---- Datasets Manager --------------------------------------------------

  async function fetchAllDatasets() { return _get('/api/datasets/all'); }
  async function linkSourceAudio(tensorName, audioPath) {
    return _post('/api/datasets/link-audio', { tensor_name: tensorName, audio_path: audioPath });
  }
  async function openFolder(path) { return _post('/api/open-folder', { path }); }

  // ---- File Browser -----------------------------------------------------

  async function browseDir(path) {
    const target = path || '.';
    try {
      const r = await fetch('/api/browse', { method: 'POST', headers: { 'Content-Type': 'application/json', ..._authHeaders() }, body: JSON.stringify({ path: target, dirs_only: false }) });
      const data = await r.json().catch(() => ({}));
      if (!r.ok || data.error) return { error: data.error || 'Browse failed (' + r.status + ')', entries: [] };
      return { current: data.path || target, entries: (data.entries || []).map(e => ({ name: e.name, type: e.is_dir ? 'dir' : 'file', path: e.path, is_dir: e.is_dir !== false, size: e.size })) };
    } catch (e) { return { error: e.message || 'Network error', entries: [] }; }
  }

  // ---- WebSocket helpers ------------------------------------------------

  function _wsAuth() {
    const t = _getToken();
    return t ? `?token=${t}` : '';
  }

  function connectTrainingWS(onMessage) {
    let ws, _timer, _closed = false;
    function _connect() {
      ws = new WebSocket(`ws://${location.host}/ws/training${_wsAuth()}`);
      ws.onmessage = (e) => {
        try { onMessage(JSON.parse(e.data)); }
        catch (err) { console.warn('[WS:training] message handler error:', err); }
      };
      ws.onerror = (ev) => { console.warn('[WS:training] error', ev); };
      ws.onclose = () => {
        if (!_closed) { clearTimeout(_timer); _timer = setTimeout(_connect, 3000); }
      };
    }
    _connect();
    return {
      close: () => { _closed = true; clearTimeout(_timer); ws?.close(); },
      get readyState() { return ws?.readyState; },
    };
  }

  function connectGpuWS(onMessage) {
    let ws, _timer, _closed = false;
    function _connect() {
      ws = new WebSocket(`ws://${location.host}/ws/gpu${_wsAuth()}`);
      ws.onmessage = (e) => {
        try { onMessage(JSON.parse(e.data)); }
        catch (err) { console.warn('[WS:gpu] message handler error:', err); }
      };
      ws.onerror = (ev) => { console.warn('[WS:gpu] error', ev); };
      ws.onclose = () => { if (!_closed) { clearTimeout(_timer); _timer = setTimeout(_connect, 3000); } };
    }
    _connect();
    return { close: () => { _closed = true; clearTimeout(_timer); ws?.close(); } };
  }

  function connectTaskWS(taskId, onMessage) {
    const ws = new WebSocket(`ws://${location.host}/ws/task/${taskId}${_wsAuth()}`);
    ws.onmessage = (e) => {
      try { onMessage(JSON.parse(e.data)); }
      catch (err) { console.warn('[WS:task] message handler error:', err); }
    };
    ws.onerror = (ev) => { console.warn('[WS:task] error', ev); };
    return ws;
  }

  // ---- Public API -------------------------------------------------------

  return {
    fetchSettings,
    saveSettings,
    fetchGPU,
    estimateVRAM,
    fetchModels,
    scanDataset,
    readSidecar,
    writeSidecar,
    fetchPresets,
    loadPreset,
    savePreset,
    deletePreset,
    fetchHistory,
    fetchRunConfig,
    fetchRunLossCurve,
    deleteHistoryFolder,
    startTraining,
    stopTraining,
    browseDir,
    scanTensorsDir,
    scanAudioFolder,
    runPreprocess,
    checkFisherMap,
    runFisherAnalysis,
    scanCheckpoints,
    validateApiKey,
    runAICaptions,
    bulkWriteTriggerTag,
    buildCLICommand,
    fetchAllDatasets,
    linkSourceAudio,
    openFolder,
    stopTask,
    startTensorBoard,
    connectTrainingWS,
    connectGpuWS,
    connectTaskWS,
    signalShutdown,
  };

})();
