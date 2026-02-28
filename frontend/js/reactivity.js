/* ============================================================
   Side-Step GUI — Cross-field Reactivity & Contextual Warnings
   Translates wizard UX decisions to GUI interactions.
   ============================================================ */

const Reactivity = (() => {
  'use strict';

  function _showWarning(id, msg) {
    const el = $(id);
    if (!el) return;
    el.textContent = msg;
    el.style.display = msg ? 'block' : 'none';
  }

  function _setLocked(groupEl, locked, badge) {
    if (!groupEl) return;
    if (locked) {
      groupEl.classList.add('field-locked');
      const existing = groupEl.querySelector('.field-locked-badge');
      if (!existing && badge) {
        const span = document.createElement('span');
        span.className = 'field-locked-badge';
        span.textContent = badge;
        const label = groupEl.querySelector('.form-group__label');
        if (label) label.appendChild(span);
      }
    } else {
      groupEl.classList.remove('field-locked');
      const b = groupEl.querySelector('.field-locked-badge');
      if (b) b.remove();
    }
  }

  // ---- Chunk < 60s Warning ------------------------------------------------

  function initChunkWarning() {
    const input = $('full-chunk-duration');
    if (!input) return;
    const handler = () => {
      const val = parseInt(input.value, 10);
      if (val > 0 && val < 60) {
        _showWarning('warn-chunk-short',
          'Chunks below 60s may reduce training quality. Use 60s+ unless you need VRAM savings.');
        input.classList.add('warned');
      } else {
        _showWarning('warn-chunk-short', '');
        input.classList.remove('warned');
      }
    };
    input.addEventListener('input', handler);
    input.addEventListener('change', handler);
  }

  // ---- PP++ LR Warning ----------------------------------------------------

  function initPPLrWarning() {
    const lrInput = $('full-lr');
    if (!lrInput) return;

    function check() {
      const ppStatus = $('full-pp-status');
      const hasPP = ppStatus && ppStatus.textContent.includes('detected');
      const lr = parseFloat(lrInput.value);
      if (hasPP && !isNaN(lr) && lr > 1e-4) {
        _showWarning('warn-pp-lr',
          'PP++ detected \u2014 consider LR \u2264 1e-4 for stability.');
      } else {
        _showWarning('warn-pp-lr', '');
      }
    }
    lrInput.addEventListener('input', check);
    lrInput.addEventListener('change', check);
    $('full-dataset-dir')?.addEventListener('change', () => setTimeout(check, 400));
  }

  // ---- Warmup > 25% Warning --------------

  function initWarmupWarning() {
    const warnEl = $('full-warmup-warning');
    if (!warnEl) return;

    function check() {
      const warmup = parseInt($('full-warmup')?.value, 10) || 0;
      const bs = parseInt($('full-batch')?.value, 10) || 1;
      const ga = parseInt($('full-grad-accum')?.value, 10) || 4;
      const epochs = parseInt($('full-epochs')?.value, 10) || 100;
      const repeats = parseInt($('full-dataset-repeats')?.value, 10) || 1;
      const maxSteps = parseInt($('full-max-steps')?.value, 10) || 0;

      const infoEl = $('full-dataset-info');
      const match = infoEl?.textContent?.match(/(\d+)\s*samples/);
      const samples = match ? parseInt(match[1], 10) : 0;
      if (!samples) return;

      const stepsPerEpoch = Math.ceil((samples * repeats) / (bs * ga));
      const totalSteps = maxSteps > 0 ? maxSteps : stepsPerEpoch * epochs;

      if (warmup > 0 && warmup > totalSteps * 0.25) {
        warnEl.style.display = '';
        warnEl.textContent = ' | Warning: warmup (' + warmup + ') is > 25% of total steps (' + totalSteps + ')';
      } else {
        warnEl.style.display = 'none';
      }
    }

    ['full-batch', 'full-grad-accum', 'full-epochs', 'full-dataset-repeats', 'full-max-steps', 'full-warmup'].forEach((id) => {
      $(id)?.addEventListener('input', check);
      $(id)?.addEventListener('change', check);
    });
    $('full-dataset-dir')?.addEventListener('change', () => setTimeout(check, 300));
  }

  // ---- Save Best Gating ---------------------------------------------------

  function initSaveBestGating() {
    const toggle = $('full-save-best');
    const afterEl = $('full-save-best-after');
    const earlyEl = $('full-early-stop');
    if (!toggle) return;

    function update() {
      const show = toggle.checked;
      if (afterEl) afterEl.closest('.form-group').style.opacity = show ? '' : '0.4';
      if (afterEl) afterEl.disabled = !show;
      if (earlyEl) earlyEl.closest('.form-group').style.opacity = show ? '' : '0.4';
      if (earlyEl) earlyEl.disabled = !show;
    }

    toggle.addEventListener('change', update);
    update();
  }

  // ---- PP++ Adapter Lock --------------------------------------------------

  function initPPAdapterLock() {
    const adapterSel = $('full-adapter-type');
    if (!adapterSel) return;

    const update = debounce(() => {
      const ppStatus = $('full-pp-status');
      const ppToggle = $('full-use-ppplus');
      const hasPP = ppStatus && ppStatus.textContent.includes('detected') && (!ppToggle || ppToggle.checked);
      const adapter = adapterSel.value;
      const ppCompatible = adapter === 'lora' || adapter === 'dora';
      const shouldLock = hasPP && ppCompatible;

      ['full-rank', 'full-alpha', 'full-projections', 'full-target-mlp'].forEach((id) => {
        const el = $(id);
        if (!el) return;
        const group = el.closest('.form-group');
        _setLocked(group, shouldLock, shouldLock ? 'Locked by PP++' : '');
      });

      const warnEl = $('warn-pp-compat');
      if (warnEl) {
        if (hasPP && !ppCompatible) {
          warnEl.textContent = 'PP++ map found but ' + adapter.toUpperCase() +
            ' does not support per-module rank assignment. Map will be ignored.';
          warnEl.style.display = 'block';
        } else {
          warnEl.style.display = 'none';
        }
      }
    }, 200);

    adapterSel.addEventListener('change', update);
    $('full-dataset-dir')?.addEventListener('change', () => setTimeout(update, 400));
    $('full-use-ppplus')?.addEventListener('change', update);
    setTimeout(update, 100);
  }

  // ---- Resume Warmup Auto-Zero -------------------------------------------

  function initResumeWarmupZero() {
    const resumeInput = $('full-resume-from');
    const warmupInput = $('full-warmup');
    if (!resumeInput || !warmupInput) return;

    let _prevWarmup = warmupInput.value;
    const update = debounce(() => {
      const hasResume = resumeInput.value.trim() !== '';
      if (hasResume && warmupInput.value !== '0') {
        _prevWarmup = warmupInput.value;
        warmupInput.value = '0';
        warmupInput.dispatchEvent(new Event('input'));
        _showWarning('warn-resume-warmup', 'Warmup set to 0 for resume (warmup already completed).');
      } else if (!hasResume) {
        _showWarning('warn-resume-warmup', '');
        if (warmupInput.value === '0' && _prevWarmup && _prevWarmup !== '0') {
          warmupInput.value = _prevWarmup;
          warmupInput.dispatchEvent(new Event('input'));
        }
      }
    }, 200);

    resumeInput.addEventListener('input', update);
    resumeInput.addEventListener('change', update);
  }

  // ---- Custom Scheduler Safety -------------------------------------------

  function initSchedulerSafety() {
    const schedulerSel = $('full-scheduler');
    const formulaInput = $('full-scheduler-formula');
    if (!schedulerSel || !formulaInput) return;

    const check = debounce(() => {
      if (schedulerSel.value === 'custom' && !formulaInput.value.trim()) {
        _showWarning('warn-scheduler-formula',
          'Custom scheduler selected but no formula provided. Will fall back to cosine.');
        formulaInput.classList.add('warned');
      } else {
        _showWarning('warn-scheduler-formula', '');
        formulaInput.classList.remove('warned');
      }
    }, 300);

    schedulerSel.addEventListener('change', check);
    formulaInput.addEventListener('input', check);
    formulaInput.addEventListener('change', check);
  }

  // ---- Smart Defaults Fix -------------------------------------------------

  function initSmartDefaultsFix() {
    const epochsEl = $('full-epochs');
    const warmupEl = $('full-warmup');
    const saveBestEl = $('full-save-best-after');
    if (!epochsEl || !saveBestEl) return;

    function updateSaveBest() {
      if (!saveBestEl) return;
      const isDefault = saveBestEl.value === saveBestEl.dataset.default || saveBestEl.value === '200';
      if (!isDefault) return;

      const warmup = parseInt(warmupEl?.value, 10) || 0;
      const epochs = parseInt(epochsEl.value, 10) || 100;

      const infoEl = $('full-dataset-info');
      const match = infoEl?.textContent?.match(/(\d+)\s*samples/);
      const samples = match ? parseInt(match[1], 10) : 0;
      const bs = parseInt($('full-batch')?.value, 10) || 1;
      const ga = parseInt($('full-grad-accum')?.value, 10) || 4;
      const stepsPerEpoch = samples ? Math.ceil(samples / (bs * ga)) : 47;
      const total = stepsPerEpoch * epochs;

      const smart = Math.max(warmup + 10, Math.min(200, Math.floor(total / 10)));
      saveBestEl.value = smart;
      saveBestEl.dataset.default = String(smart);
    }

    epochsEl.addEventListener('change', updateSaveBest);
    if (warmupEl) warmupEl.addEventListener('change', updateSaveBest);
    $('full-dataset-dir')?.addEventListener('change', () => setTimeout(updateSaveBest, 300));
  }

  // ---- Timestep Override Warning ------------------------------------------

  function initTimestepWarning() {
    const muInput = $('full-timestep-mu');
    const sigmaInput = $('full-timestep-sigma');
    if (!muInput && !sigmaInput) return;

    function check() {
      if (muInput) {
        const changed = muInput.value !== muInput.dataset.default;
        _showWarning('warn-timestep-mu', changed ? 'Overriding model default \u2014 changes sampling distribution.' : '');
      }
      if (sigmaInput) {
        const changed = sigmaInput.value !== sigmaInput.dataset.default;
        _showWarning('warn-timestep-sigma', changed ? 'Overriding model default \u2014 changes sampling distribution.' : '');
      }
    }
    muInput?.addEventListener('input', check);
    muInput?.addEventListener('change', check);
    sigmaInput?.addEventListener('input', check);
    sigmaInput?.addEventListener('change', check);
  }

  // ---- Init ---------------------------------------------------------------

  function init() {
    initChunkWarning();
    initPPLrWarning();
    initWarmupWarning();
    initSaveBestGating();
    initPPAdapterLock();
    initResumeWarmupZero();
    initSchedulerSafety();
    initSmartDefaultsFix();
    initTimestepWarning();
    if (typeof ReactivityExt !== 'undefined') ReactivityExt.init();
  }

  return { init };
})();
