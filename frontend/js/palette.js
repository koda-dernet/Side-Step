/* ============================================================
   Side-Step GUI — Command Palette + Keybind System (Ctrl+K)
   VSCode-style command palette with customizable keybinds.
   ============================================================ */

const Palette = (() => {
  'use strict';

  const KEYBIND_STORAGE = 'sidestep_keybinds';

  // --- Default keybinds (actionId → combo string) ---
  const DEFAULT_BINDS = {
    'nav-ez':       'Alt+1',
    'nav-full':     'Alt+2',
    'nav-monitor':  'Alt+3',
    'nav-lab':      'Alt+4',
    'open-settings':'Ctrl+,',
  };

  // Non-rebindable keys
  const FIXED_BINDS = new Set(['nav-ez', 'nav-full', 'nav-monitor', 'nav-lab']);

  let _customBinds = {};

  function _loadBinds() {
    try {
      const raw = localStorage.getItem(KEYBIND_STORAGE);
      _customBinds = raw ? JSON.parse(raw) : {};
    } catch { _customBinds = {}; }
  }

  function _saveBinds() {
    try { localStorage.setItem(KEYBIND_STORAGE, JSON.stringify(_customBinds)); } catch (e) {}
  }

  function getKeybind(actionId) {
    if (FIXED_BINDS.has(actionId)) return DEFAULT_BINDS[actionId] || '';
    return _customBinds[actionId] ?? DEFAULT_BINDS[actionId] ?? '';
  }

  function setKeybind(actionId, combo) {
    if (FIXED_BINDS.has(actionId)) return;
    if (combo === DEFAULT_BINDS[actionId]) { delete _customBinds[actionId]; }
    else { _customBinds[actionId] = combo; }
    _saveBinds();
  }

  function resetAllKeybinds() {
    _customBinds = {};
    _saveBinds();
  }

  // Check if combo conflicts with another action
  function findConflict(actionId, combo) {
    if (!combo) return null;
    const lower = combo.toLowerCase();
    for (const cmd of commands) {
      if (cmd.id === actionId) continue;
      const existing = getKeybind(cmd.id);
      if (existing && existing.toLowerCase() === lower) return cmd;
    }
    return null;
  }

  // --- Command Registry ---
  const commands = [
    // Navigation (fixed keybinds)
    { id: 'nav-ez',        label: 'Switch to Ez Mode',     category: 'Navigate', icon: '›', action: () => _switchMode('ez') },
    { id: 'nav-full',      label: 'Switch to Advanced',    category: 'Navigate', icon: '›', action: () => _switchMode('full') },
    { id: 'nav-monitor',   label: 'Switch to Monitor',     category: 'Navigate', icon: '›', action: () => _switchMode('monitor') },
    { id: 'nav-lab',       label: 'Switch to Lab',         category: 'Navigate', icon: '›', action: () => _switchMode('lab') },

    // Lab sub-tabs
    { id: 'lab-history',   label: 'Open History',          category: 'Lab',      icon: '›', action: () => _openLabTab('history') },
    { id: 'lab-datasets',  label: 'Open Tensor Datasets',  category: 'Lab',      icon: '›', action: () => _openLabTab('datasets') },
    { id: 'lab-dataset',   label: 'Open Audio Library',    category: 'Lab',      icon: '›', action: () => _openLabTab('dataset') },
    { id: 'lab-preprocess',label: 'Open Preprocess',       category: 'Lab',      icon: '›', action: () => _openLabTab('preprocess') },
    { id: 'lab-ppplus',    label: 'Open PP++',             category: 'Lab',      icon: '›', action: () => _openLabTab('ppplus') },
    { id: 'lab-captions',  label: 'Open AI Captions',      category: 'Lab',      icon: '›', action: () => _openLabTab('captions') },
    { id: 'lab-resume',    label: 'Open Resume',           category: 'Lab',      icon: '›', action: () => _openLabTab('resume') },

    // Actions
    { id: 'start-training',label: 'Start Training',        category: 'Training', icon: '▶', action: () => _clickBtn('btn-start-ez') },
    { id: 'stop-training', label: 'Stop Training',         category: 'Training', icon: '■', action: () => { if (typeof Training !== 'undefined') Training.stop(); } },
    { id: 'export-cli',    label: 'Export CLI Command',    category: 'Training', icon: '⌘', action: () => _clickBtn('btn-export-cli') },
    { id: 'export-csv',    label: 'Export History CSV',    category: 'Training', icon: '⌘', action: () => _clickBtn('btn-export-csv') },

    // Settings
    { id: 'open-settings', label: 'Open Settings',         category: 'Settings', icon: '⚙', action: () => _clickBtn('btn-open-settings') },
    { id: 'load-preset',   label: 'Load Preset',           category: 'Settings', icon: '›', action: () => _clickBtn('ez-load-preset') },
    { id: 'keybinds',      label: 'Keyboard Shortcuts',    category: 'Settings', icon: '⌨', action: () => _openKeybindModal() },
    
    // Help
    { id: 'show-tutorial', label: 'Show Tutorial',         category: 'Help',     icon: '?', action: () => { if (typeof Tutorial !== 'undefined') Tutorial.start(); } },
    { id: 'reset-newcomer',label: 'Reset Newcomer Experience', category: 'Help', icon: '↺', action: _resetNewcomer },

    // Jump to fields
    { id: 'j-lr',          label: 'Learning Rate',         category: 'Jump To',  icon: '›', action: () => _jumpTo('full-lr') },
    { id: 'j-epochs',      label: 'Epochs',                category: 'Jump To',  icon: '›', action: () => _jumpTo('full-epochs') },
    { id: 'j-batch',       label: 'Batch Size',            category: 'Jump To',  icon: '›', action: () => _jumpTo('full-batch') },
    { id: 'j-warmup',      label: 'Warmup Steps',          category: 'Jump To',  icon: '›', action: () => _jumpTo('full-warmup') },
    { id: 'j-adapter',     label: 'Adapter Type',          category: 'Jump To',  icon: '›', action: () => _jumpTo('full-adapter-type') },
    { id: 'j-rank',        label: 'Rank',                  category: 'Jump To',  icon: '›', action: () => _jumpTo('full-rank') },
    { id: 'j-optimizer',   label: 'Optimizer',             category: 'Jump To',  icon: '›', action: () => _jumpTo('full-optimizer') },
    { id: 'j-scheduler',   label: 'Scheduler',             category: 'Jump To',  icon: '›', action: () => _jumpTo('full-scheduler') },
    { id: 'j-model',       label: 'Model Variant',         category: 'Jump To',  icon: '›', action: () => _jumpTo('full-model-variant') },
    { id: 'j-dataset',     label: 'Dataset',               category: 'Jump To',  icon: '›', action: () => _jumpTo('full-dataset-dir') },
    { id: 'j-run-name',    label: 'Run Name',              category: 'Jump To',  icon: '›', action: () => _jumpTo('full-run-name') },
    { id: 'j-ckpt',        label: 'Gradient Checkpointing',category: 'Jump To',  icon: '›', action: () => _jumpTo('full-grad-ckpt-ratio') },
    { id: 'j-save-every',  label: 'Save Every N Epochs',   category: 'Jump To',  icon: '›', action: () => _jumpTo('full-save-every') },
  ];

  // --- DOM refs ---
  let backdrop, input, resultsList;
  let activeIndex = -1;
  let filtered = [];

  function init() {
    try { _loadBinds(); } catch (e) { _customBinds = {}; }
    backdrop = document.getElementById('palette-backdrop');
    input = document.getElementById('palette-input');
    resultsList = document.getElementById('palette-results');
    if (!backdrop || !input || !resultsList) return;

    try {
      input.addEventListener('input', _onInput);
      backdrop.addEventListener('click', (e) => { if (e.target === backdrop) close(); });
      document.addEventListener('keydown', _onGlobalKeydown);
    } catch (e) { console.warn('[Palette] Could not attach listeners:', e); }
  }

  function _onGlobalKeydown(e) {
    // Ctrl+K always opens palette (not rebindable)
    if ((e.ctrlKey || e.metaKey) && e.key === 'k') {
      e.preventDefault();
      backdrop.classList.contains('open') ? close() : open();
      return;
    }
    // Escape closes palette
    if (e.key === 'Escape' && backdrop.classList.contains('open')) {
      e.preventDefault();
      close();
      return;
    }
    // Arrow/Enter navigation inside palette
    if (backdrop.classList.contains('open')) {
      if (e.key === 'ArrowDown') { e.preventDefault(); _navigate(1); return; }
      if (e.key === 'ArrowUp') { e.preventDefault(); _navigate(-1); return; }
      if (e.key === 'Enter') { e.preventDefault(); _execute(); return; }
      return;
    }
    // Don't fire keybinds when typing in inputs
    if (_isTyping(e)) return;
    // Tutorial active — don't fire keybinds
    if (typeof Tutorial !== 'undefined' && Tutorial.isActive()) return;
    // Match custom keybinds
    const combo = _eventToCombo(e);
    if (!combo) return;
    const match = commands.find((cmd) => {
      const bind = getKeybind(cmd.id);
      return bind && bind.toLowerCase() === combo.toLowerCase();
    });
    if (match) {
      e.preventDefault();
      if (typeof match.action === 'function') match.action();
    }
  }

  function _isTyping(e) {
    const tag = (e.target.tagName || '').toLowerCase();
    return tag === 'input' || tag === 'textarea' || tag === 'select' || e.target.isContentEditable;
  }

  function _eventToCombo(e) {
    const parts = [];
    if (e.ctrlKey || e.metaKey) parts.push('Ctrl');
    if (e.altKey) parts.push('Alt');
    if (e.shiftKey) parts.push('Shift');
    const key = e.key.length === 1 ? e.key.toUpperCase() : e.key;
    if (['Control', 'Alt', 'Shift', 'Meta'].includes(e.key)) return '';
    // Map number keys for Alt+1 etc.
    if (/^[0-9]$/.test(e.key)) parts.push(e.key);
    else if (e.key === ',') parts.push(',');
    else parts.push(key);
    return parts.join('+');
  }

  function open() {
    backdrop.classList.add('open');
    input.value = '';
    activeIndex = -1;
    _render(commands);
    setTimeout(() => input.focus(), 50);
  }

  function close() {
    backdrop.classList.remove('open');
    input.value = '';
  }

  function _onInput() {
    const q = input.value.toLowerCase().trim();
    if (!q) { filtered = commands; }
    else {
      filtered = commands.filter((cmd) =>
        cmd.label.toLowerCase().includes(q) ||
        cmd.category.toLowerCase().includes(q) ||
        (getKeybind(cmd.id) || '').toLowerCase().includes(q)
      );
    }
    activeIndex = filtered.length > 0 ? 0 : -1;
    _render(filtered);
  }

  function _render(items) {
    filtered = items;
    if (!items.length) {
      resultsList.innerHTML = '<div class="palette__empty">No commands found</div>';
      return;
    }
    let html = '', lastCat = '';
    items.forEach((cmd, i) => {
      if (cmd.category !== lastCat) {
        lastCat = cmd.category;
        html += '<div class="palette__group">' + _esc(cmd.category) + '</div>';
      }
      const cls = i === activeIndex ? 'palette__item active' : 'palette__item';
      const shortcut = getKeybind(cmd.id);
      html += '<div class="' + cls + '" data-idx="' + i + '">' +
        '<span class="palette__item-label"><span class="palette__item-icon">' + _esc(cmd.icon || '') + '</span> ' + _esc(cmd.label) + '</span>' +
        (shortcut ? '<span class="palette__item-shortcut"><span class="kbd">' + _esc(shortcut) + '</span></span>' : '') +
        '</div>';
    });
    resultsList.innerHTML = html;
    resultsList.querySelectorAll('.palette__item').forEach((el) => {
      el.addEventListener('click', () => { activeIndex = parseInt(el.dataset.idx, 10); _execute(); });
    });
  }

  function _navigate(dir) {
    if (!filtered.length) return;
    activeIndex = (activeIndex + dir + filtered.length) % filtered.length;
    _render(filtered);
    const active = resultsList.querySelector('.palette__item.active');
    if (active) active.scrollIntoView({ block: 'nearest' });
  }

  function _execute() {
    if (activeIndex < 0 || activeIndex >= filtered.length) return;
    const cmd = filtered[activeIndex];
    close();
    if (typeof cmd.action === 'function') cmd.action();
  }

  // --- Reset Newcomer Experience ---
  function _resetNewcomer() {
    try { localStorage.removeItem('sidestep_welcomed'); } catch (e) {}
    try { localStorage.removeItem('sidestep_tutorial_done'); } catch (e) {}
    try { localStorage.removeItem(KEYBIND_STORAGE); } catch (e) {}
    if (typeof API !== 'undefined' && API.saveSettings) {
      API.saveSettings({ first_run_complete: false }).catch(() => {});
    }
    if (typeof showToast === 'function') showToast('Newcomer experience reset — reloading...', 'ok');
    setTimeout(() => location.reload(), 800);
  }

  // --- Helpers ---
  function _switchMode(mode) {
    if (typeof switchMode === 'function') switchMode(mode);
  }

  function _openLabTab(tab) {
    _switchMode('lab');
    setTimeout(() => {
      document.querySelectorAll('.lab-nav__item').forEach((i) => i.classList.remove('active'));
      document.querySelectorAll('.lab-panel').forEach((p) => p.classList.remove('active'));
      const btn = document.querySelector('.lab-nav__item[data-lab="' + tab + '"]');
      if (btn) btn.classList.add('active');
      const panel = document.getElementById('lab-' + tab);
      if (panel) panel.classList.add('active');
    }, 50);
  }

  function _clickBtn(id) {
    const el = document.getElementById(id);
    if (el) el.click();
  }

  function _jumpTo(id) {
    _switchMode('full');
    setTimeout(() => {
      const el = document.getElementById(id);
      if (!el) return;
      const section = el.closest('.section-group');
      if (section && !section.classList.contains('open')) {
        section.querySelector('.section-group__toggle')?.click();
      }
      el.scrollIntoView({ behavior: 'smooth', block: 'center' });
      el.classList.add('highlight-field');
      el.focus();
      setTimeout(() => el.classList.remove('highlight-field'), 1500);
    }, 200);
  }

  function _esc(str) {
    const d = document.createElement('div');
    d.textContent = str;
    return d.innerHTML;
  }

  // --- Keybind Settings Modal ---
  function _openKeybindModal() {
    let modal = document.getElementById('keybind-modal');
    if (!modal) {
      modal = document.createElement('div');
      modal.className = 'modal'; modal.id = 'keybind-modal';
      modal.innerHTML = '<div class="modal__content" style="max-width:520px;"><div class="modal__header"><span class="modal__title">Keyboard Shortcuts</span><button class="modal__close" id="keybind-modal-close">x</button></div><div class="modal__body"><div id="keybind-list"></div></div><div class="modal__footer"><button class="btn btn--sm" id="keybind-reset-all">Reset to Defaults</button><button class="btn btn--primary btn--sm" id="keybind-modal-done">Done</button></div></div>';
      document.body.appendChild(modal);
      const _close = () => modal.classList.remove('open');
      modal.addEventListener('click', (e) => { if (e.target === modal) _close(); });
      const btnClose = document.getElementById('keybind-modal-close');
      const btnDone = document.getElementById('keybind-modal-done');
      const btnReset = document.getElementById('keybind-reset-all');
      if (btnClose) btnClose.addEventListener('click', _close);
      if (btnDone) btnDone.addEventListener('click', _close);
      if (btnReset) btnReset.addEventListener('click', () => { resetAllKeybinds(); _renderKeybindList(); if (typeof showToast === 'function') showToast('Keybinds reset to defaults', 'ok'); });
    }
    _renderKeybindList();
    modal.classList.add('open');
  }

  let _captureTarget = null;

  function _renderKeybindList() {
    const container = document.getElementById('keybind-list');
    if (!container) return;
    const bindable = commands.filter((c) => c.id in DEFAULT_BINDS || !c.id.startsWith('j-'));
    const grouped = {};
    bindable.forEach((c) => { (grouped[c.category] || (grouped[c.category] = [])).push(c); });
    let html = '';
    Object.entries(grouped).forEach(([cat, cmds]) => {
      html += '<div style="color:var(--primary);font-weight:bold;font-size:var(--font-size-xs);text-transform:uppercase;letter-spacing:1px;margin:var(--space-md) 0 var(--space-xs) 0;">' + _esc(cat) + '</div>';
      cmds.forEach((c) => {
        const bind = getKeybind(c.id), fixed = FIXED_BINDS.has(c.id);
        html += '<div style="display:flex;justify-content:space-between;align-items:center;padding:4px 0;' + (fixed ? 'opacity:0.5;' : '') + '"><span style="font-size:var(--font-size-sm);color:var(--text);">' + _esc(c.label) + '</span><span class="keybind-chip kbd" data-cmd-id="' + c.id + '" style="min-width:60px;text-align:center;' + (fixed ? 'cursor:default;' : 'cursor:pointer;') + '" ' + (fixed ? 'title="Not customizable"' : 'title="Click to change"') + '>' + (bind || '<span class="u-text-muted">none</span>') + '</span></div>';
      });
    });
    container.innerHTML = html;
    container.querySelectorAll('.keybind-chip').forEach((chip) => {
      if (FIXED_BINDS.has(chip.dataset.cmdId)) return;
      chip.addEventListener('click', () => _startCapture(chip, chip.dataset.cmdId));
    });
  }

  function _startCapture(chip, cmdId) {
    if (_captureTarget) _stopCapture();
    _captureTarget = { chip, cmdId };
    chip.textContent = 'Press key combo...'; chip.style.borderColor = 'var(--primary)'; chip.style.color = 'var(--primary)';
    const handler = (e) => {
      e.preventDefault(); e.stopPropagation();
      if (e.key === 'Escape') { _stopCapture(); return; }
      if (['Control', 'Alt', 'Shift', 'Meta'].includes(e.key)) return;
      const combo = _eventToCombo(e); if (!combo) return;
      const conflict = findConflict(cmdId, combo);
      if (conflict) { if (!window.confirm('Already used for "' + conflict.label + '". Replace?')) { _stopCapture(); return; } setKeybind(conflict.id, ''); }
      setKeybind(cmdId, combo); _stopCapture(); _renderKeybindList();
    };
    document.addEventListener('keydown', handler, true); _captureTarget.handler = handler;
  }

  function _stopCapture() {
    if (!_captureTarget) return;
    document.removeEventListener('keydown', _captureTarget.handler, true); _captureTarget = null; _renderKeybindList();
  }

  // Init when DOM ready
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }

  return { open, close, openKeybindModal: _openKeybindModal, getKeybind, setKeybind, resetAllKeybinds, findConflict };
})();
