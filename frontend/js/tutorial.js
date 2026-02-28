/* ============================================================
   Side-Step GUI — Interactive Guided Tutorial (v3)
   Rules:
   - ONLY the Next/Skip buttons advance. No backdrop clicks.
   - User must perform every action themselves (click tabs, etc.)
   - Click-interact steps: user clicks target, Next unlocks, user
     clicks Next to proceed. Nothing auto-advances.
   - Scroll/resize repositions spotlight + tooltip.
   - Tooltip always clamped inside viewport.
   ============================================================ */

const Tutorial = (() => {
  'use strict';

  const STORAGE_KEY = 'sidestep_tutorial_done';

  /*  Step schema:
        target    — CSS selector to spotlight (null = centered card)
        text      — HTML content for tooltip
        position  — 'bottom' | 'top' | 'center'  (hint, auto-flips)
        interact  — CSS selector user must click to unlock Next
                    (null = Next is immediately available)
        nextLabel — override for Next button text                  */

  const STEPS = [
    // ── INTRO ──
    { target: null,
      text: '<strong>Welcome to Side-Step!</strong><br>This tour takes about 2 minutes. I\'ll highlight things — you click them.<br><br>You can <strong>skip anytime</strong>, or re-run later from <span class="kbd">Ctrl+K</span> → "Show Tutorial".' },

    // ── MODES ──
    { target: '.topbar__modes',
      text: '<strong>These are your four modes.</strong> Everything in Side-Step lives in one of these. Let\'s visit each one.' },

    // Ez Mode (user is already here after welcome)
    { target: '.topbar__mode[data-mode="ez"]',
      text: '<strong>Ez Mode</strong> — pick a dataset, pick a model, hit Start. That\'s it.<br><br>You\'re already here. Let\'s look around.' },

    { target: '#ez-model-variant',
      text: '<strong>Model variant.</strong> Turbo is fast. Base is balanced. SFT follows prompts closely.<br><br><em>Click the dropdown to see options.</em>',
      interact: '#ez-model-variant' },

    { target: '#ez-dataset-dir',
      text: '<strong>Dataset.</strong> Shows folders from your configured directories. Empty? Set them in Settings later.' },

    { target: '#ez-adapter-cards',
      text: '<strong>Adapter type.</strong> LoRA is the standard. The rest are experimental.<br><br><em>Try clicking a different card.</em>',
      interact: '#ez-adapter-cards' },

    { target: '.ez-start-row', position: 'top',
      text: '<strong>Start Training.</strong> Once dataset + model are set, this button activates. Don\'t worry — we won\'t start now.' },

    // ── ADVANCED ──
    { target: '.topbar__mode[data-mode="full"]',
      text: '<strong>Advanced mode</strong> — same engine, every knob exposed.<br><br><em>Click "Advanced" to switch.</em>',
      interact: '.topbar__mode[data-mode="full"]' },

    { target: '#full-adapter-section-title',
      text: '<strong>Adapter settings.</strong> Rank, alpha, dropout. Every field has a <span class="help-icon" style="display:inline-flex;width:16px;height:16px;font-size:9px;pointer-events:none;">?</span> — click any for a plain-English explanation.' },

    { target: '#full-lr',
      text: '<strong>Learning rate</strong> — the most important number.<br><code>1e-4</code> is safe. PP++ auto-lowers it.<br><br><em>Click the field and type a value.</em>',
      interact: '#full-lr' },

    { target: '.two-col__right',
      text: '<strong>Review panel.</strong> Live summary of all settings. Click any header to jump to that field.' },

    // ── MONITOR ──
    { target: '.topbar__mode[data-mode="monitor"]',
      text: '<strong>Monitor</strong> — real-time loss charts, logs, GPU usage.<br><br><em>Click "Monitor".</em>',
      interact: '.topbar__mode[data-mode="monitor"]' },

    { target: '#mode-monitor',
      text: 'This is idle now. When training starts, it switches here automatically with live charts and progress bars.' },

    // ── LAB ──
    { target: '.topbar__mode[data-mode="lab"]',
      text: '<strong>The Lab</strong> — past runs, audio library, preprocessing, PP++.<br><br><em>Click "Lab".</em>',
      interact: '.topbar__mode[data-mode="lab"]' },

    { target: '.lab-nav',
      text: '<strong>Lab tabs.</strong> History, Tensor Datasets, Audio Library, Preprocess, PP++. Let\'s visit a few.' },

    { target: '.lab-nav__item[data-lab="datasets"]',
      text: '<strong>Tensor Datasets</strong> — preprocessed .pt files ready for training.<br><br><em>Click the tab.</em>',
      interact: '.lab-nav__item[data-lab="datasets"]' },

    { target: '#lab-datasets',
      text: 'Shows file count, duration, source audio link for each dataset. You can link back to original audio to edit sidecars.' },

    { target: '.lab-nav__item[data-lab="dataset"]',
      text: '<strong>Audio Library</strong> — browse raw audio, edit captions/tags, manage triggers.<br><br><em>Click "Audio Library".</em>',
      interact: '.lab-nav__item[data-lab="dataset"]' },

    { target: '#lab-dataset',
      text: 'Each file shows duration, sidecar status, genre, tags. Click <strong>Edit</strong> to open the sidecar editor. <strong>Generate AI Captions</strong> uses Gemini/OpenAI.' },

    { target: '.lab-nav__item[data-lab="preprocess"]',
      text: '<strong>Preprocess</strong> — converts audio to .pt tensors.<br><br><em>Click "Preprocess".</em>',
      interact: '.lab-nav__item[data-lab="preprocess"]' },

    { target: '#lab-preprocess',
      text: 'Point at audio, pick output, hit Start. <strong>Tip:</strong> Ez Mode auto-preprocesses raw audio folders — you often don\'t need this.' },

    { target: '.lab-nav__item[data-lab="ppplus"]',
      text: '<strong>PP++</strong> — per-layer rank allocation via Fisher analysis. Optional, skip for first run.<br><br><em>Click "PP++".</em>',
      interact: '.lab-nav__item[data-lab="ppplus"]' },

    { target: '#lab-ppplus',
      text: 'Generates <code>fisher_map.json</code>. Training auto-detects it and applies variable ranks. Purely optional.' },

    // ── SETTINGS ──
    { target: '#btn-open-settings', position: 'bottom',
      text: '<strong>Settings</strong> — configure directories and API keys.<br><br><em>Click the gear icon.</em>',
      interact: '#btn-open-settings' },

    { target: '.settings-panel__body',
      text: '<strong>Directories:</strong> checkpoint, audio, tensors, adapters.<br><strong>API keys:</strong> Gemini, OpenAI, Genius (all optional).<br><br>Close the panel when you\'re done — settings auto-save.' },

    // ── CONSOLE ──
    { target: '#console-strip', position: 'top',
      text: '<strong>Console strip.</strong> Mode, status, device info. Click to expand into a log.' },

    // ── SHORTCUTS ──
    { target: null,
      text: '<strong>Keyboard shortcuts:</strong><br><span class="kbd">Ctrl+K</span> Command palette<br><span class="kbd">Alt+1/2/3/4</span> Switch modes<br><span class="kbd">Ctrl+,</span> Open Settings<br><span class="kbd">Esc</span> Close panels<br><br>Customize in <span class="kbd">Ctrl+K</span> → "Keyboard Shortcuts".' },

    // ── DONE ──
    { target: null,
      text: '<strong>You\'re all set!</strong><br><br>1. Set directories in <strong>Settings</strong><br>2. Pick dataset + model in <strong>Ez Mode</strong><br>3. Hit <strong>Start</strong><br><br>Re-run anytime: <span class="kbd">Ctrl+K</span> → "Show Tutorial".<br><br><em>Happy training!</em>',
      nextLabel: 'Finish' },
  ];

  let _idx = -1, _active = false, _locked = false;
  let _overlay, _spot, _tip, _cleanup = null;
  let _scrollTimer = null;

  // ── DOM setup ──
  function _build() {
    if (_overlay) return;
    _overlay = document.createElement('div');
    _overlay.className = 'tutorial-overlay';
    _spot = document.createElement('div');
    _spot.className = 'tutorial-spotlight';
    _overlay.appendChild(_spot);
    _tip = document.createElement('div');
    _tip.className = 'tutorial-tooltip';
    _tip.innerHTML =
      '<div class="tutorial-tooltip__progress" id="tutorial-progress"></div>' +
      '<div class="tutorial-tooltip__text" id="tutorial-text"></div>' +
      '<div class="tutorial-tooltip__nav">' +
      '<span class="tutorial-tooltip__counter" id="tutorial-counter"></span>' +
      '<span class="tutorial-tooltip__actions">' +
      '<button class="btn btn--sm tutorial-tooltip__skip" id="tutorial-skip">Skip tutorial</button>' +
      '<button class="btn btn--sm btn--primary tutorial-tooltip__next" id="tutorial-next">Next</button>' +
      '</span></div>';
    _overlay.appendChild(_tip);
    document.body.appendChild(_overlay);

    const btnNext = document.getElementById('tutorial-next');
    const btnSkip = document.getElementById('tutorial-skip');
    if (btnNext) btnNext.addEventListener('click', () => { if (!_locked) _advance(); });
    if (btnSkip) btnSkip.addEventListener('click', _end);
    // NO overlay click handler — only buttons advance
    document.addEventListener('keydown', (e) => {
      if (_active && e.key === 'Escape') { _end(); e.preventDefault(); }
    });
    // Reposition on scroll/resize
    const _reposition = () => { if (_active && _idx >= 0) _position(STEPS[_idx]); };
    const main = document.querySelector('.main');
    if (main) main.addEventListener('scroll', () => { clearTimeout(_scrollTimer); _scrollTimer = setTimeout(_reposition, 60); });
    window.addEventListener('resize', _reposition);
  }

  function start() { _build(); _idx = -1; _active = true; _locked = false; _overlay.classList.add('active'); _advance(); }
  function _advance() { _teardown(); _idx++; if (_idx >= STEPS.length) { _end(); return; } requestAnimationFrame(() => _show(STEPS[_idx])); }

  function _end() {
    _teardown(); _active = false; _locked = false;
    try { localStorage.setItem(STORAGE_KEY, 'done'); } catch (e) {}
    _overlay?.classList.remove('active');
    _spot.style.display = 'none';
    document.getElementById('settings-panel')?.classList.remove('open');
  }

  function _teardown() { if (_cleanup) { _cleanup(); _cleanup = null; } }

  // ── Show a step ──
  function _show(step) {
    const txt = document.getElementById('tutorial-text');
    const ctr = document.getElementById('tutorial-counter');
    const btn = document.getElementById('tutorial-next');
    const bar = document.getElementById('tutorial-progress');
    if (!txt) return;
    txt.innerHTML = step.text;
    ctr.textContent = (_idx + 1) + ' / ' + STEPS.length;
    if (bar) bar.style.width = ((_idx + 1) / STEPS.length * 100) + '%';

    if (step.interact) {
      // User must click the target first
      _locked = true;
      btn.textContent = 'Try it ↑';
      btn.classList.add('tutorial-tooltip__next--locked');
      _wireInteract(step.interact, btn, step);
    } else {
      _locked = false;
      btn.textContent = step.nextLabel || (_idx === STEPS.length - 1 ? 'Finish' : 'Next');
      btn.classList.remove('tutorial-tooltip__next--locked');
    }
    _position(step);
  }

  function _wireInteract(condition, btn, step) {
    // Poll every 200ms for the expected state change
    const unlock = () => {
      _locked = false;
      btn.textContent = step.nextLabel || (_idx === STEPS.length - 1 ? 'Finish' : 'Next');
      btn.classList.remove('tutorial-tooltip__next--locked');
      requestAnimationFrame(() => _position(step));
    };
    // Snapshot initial state for inputs
    const initEl = typeof condition === 'string' ? document.querySelector(condition) : null;
    const initVal = initEl?.value;
    const initFocus = document.activeElement;

    const smartCheck = () => {
      if (typeof condition === 'function') return condition();
      const el = typeof condition === 'string' ? document.querySelector(condition) : null;
      if (!el) return false;
      if (el.classList.contains('topbar__mode')) return el.classList.contains('active');
      if (el.classList.contains('lab-nav__item')) return el.classList.contains('active');
      if (el.id === 'btn-open-settings') return !!document.getElementById('settings-panel')?.classList.contains('open');
      if (el.tagName === 'SELECT') return document.activeElement === el || el.value !== initVal;
      if (el.tagName === 'INPUT') return document.activeElement === el || el.value !== initVal;
      // For generic containers (adapter cards etc): check if a child changed
      return document.activeElement !== initFocus;
    };

    const timer = setInterval(() => {
      if (smartCheck()) { clearInterval(timer); unlock(); }
    }, 200);
    _cleanup = () => clearInterval(timer);
  }

  // ── Position spotlight + tooltip ──
  function _position(step) {
    if (!step.target) {
      _spot.style.display = 'none';
      _tip.className = 'tutorial-tooltip tutorial-tooltip--center';
      _tip.style.cssText = '';
      return;
    }
    const el = document.querySelector(step.target);
    if (!el) { _spot.style.display = 'none'; _tip.className = 'tutorial-tooltip tutorial-tooltip--center'; _tip.style.cssText = ''; return; }

    // Scroll element into view if needed
    const main = document.querySelector('.main');
    if (main) {
      const elRect = el.getBoundingClientRect();
      const mainRect = main.getBoundingClientRect();
      if (elRect.bottom > mainRect.bottom || elRect.top < mainRect.top) {
        el.scrollIntoView({ behavior: 'smooth', block: 'center' });
      }
    }

    // Measure after potential scroll
    requestAnimationFrame(() => _placeElements(el, step));
  }

  function _placeElements(el, step) {
    const r = el.getBoundingClientRect();
    const pad = 8;
    const vw = window.innerWidth, vh = window.innerHeight;

    // Spotlight
    _spot.style.display = 'block';
    _spot.style.top = Math.max(0, r.top - pad) + 'px';
    _spot.style.left = Math.max(0, r.left - pad) + 'px';
    _spot.style.width = Math.min(r.width + pad * 2, vw) + 'px';
    _spot.style.height = Math.min(r.height + pad * 2, vh) + 'px';

    // Tooltip positioning with viewport clamping
    _tip.className = 'tutorial-tooltip tutorial-tooltip--positioned';
    const tw = 380, th = _tip.offsetHeight || 200;
    const pref = step.position || 'bottom';
    let top, left;

    if (pref === 'top' || r.bottom + th + 16 > vh) {
      // Place above
      top = Math.max(8, r.top - th - 12);
    } else {
      // Place below
      top = r.bottom + 12;
    }
    left = Math.max(8, Math.min(r.left, vw - tw - 16));
    // Final clamp
    top = Math.max(8, Math.min(top, vh - th - 8));

    _tip.style.cssText = 'position:fixed;top:' + top + 'px;left:' + left + 'px;';
  }

  function isDone() { try { return localStorage.getItem(STORAGE_KEY) === 'done'; } catch (e) { return false; } }
  function reset() { try { localStorage.removeItem(STORAGE_KEY); } catch (e) {} }
  function isActive() { return _active; }

  return { start, next: _advance, skip: _end, isDone, reset, isActive };
})();
