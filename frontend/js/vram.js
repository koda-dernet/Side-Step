/* ============================================================
   Side-Step GUI — Reactive VRAM Budget Bar
   Listens to config form changes in Full Configure mode and
   recalculates the VRAM estimation, updating the bar + legend.
   ============================================================ */

const VRAM = (() => {

  const _TEXT_STATE_CLASSES = ["u-text-success", "u-text-warning", "u-text-error", "u-text-muted"];

  function _setTextState(el, state) {
    if (!el) return;
    el.style.color = "";
    el.classList.remove(..._TEXT_STATE_CLASSES);
    if (state === "success") el.classList.add("u-text-success");
    else if (state === "warning") el.classList.add("u-text-warning");
    else if (state === "error") el.classList.add("u-text-error");
    else if (state === "muted") el.classList.add("u-text-muted");
  }

  function _gatherConfig() {
    const adapter = $('full-adapter-type')?.value || 'lora';
    let rank = 64;
    if (adapter === 'lora' || adapter === 'dora') rank = parseInt($('full-rank')?.value) || 64;
    else if (adapter === 'lokr') rank = parseInt($('full-lokr-dim')?.value) || 64;
    else if (adapter === 'loha') rank = parseInt($('full-loha-dim')?.value) || 64;
    else if (adapter === 'oft') rank = parseInt($('full-oft-block-size')?.value) || 64;
    const ratio = parseFloat($('full-grad-ckpt-ratio')?.value);
    return {
      adapter_type: adapter,
      rank: rank,
      batch_size: parseInt($('full-batch')?.value) || 1,
      offload_encoder: $('full-offload-encoder')?.checked ?? true,
      gradient_checkpointing: !isNaN(ratio) && ratio > 0 ? 'on' : 'off',
      gradient_checkpointing_ratio: !isNaN(ratio) ? ratio : 1.0,
      optimizer_type: $('full-optimizer')?.value || 'adamw',
      chunk_duration: parseInt($('full-chunk-duration')?.value) || 0,
      target_mlp: $('full-target-mlp')?.checked ?? true,
    };
  }

  async function recalculate() {
    const config = _gatherConfig();

    try {
      const est = await API.estimateVRAM(config);
      _render(est);
    } catch (e) {
      console.warn('[VRAM] estimation failed:', e);
      _render({ peak_mb: 0, total_gpu_mb: 0, available_mb: 0, verdict: 'unknown', system_used_mb: 0 });
    }
  }

  function _verdictToState(verdict) {
    if (verdict === 'red') return 'error';
    if (verdict === 'yellow') return 'warning';
    return 'success';
  }

  function _render(est) {
    const totalGpu = est.gpu_total_mb || 0;
    const sysUsed = est.system_used_mb || 0;
    const available = totalGpu > 0 ? totalGpu - sysUsed : 0;
    const peak = est.peak_mb || 0;
    const gradOh = (est.gradient_mb || 0) + (est.cuda_overhead_mb || 0) + (est.fragmentation_mb || 0);
    const optTotal = (est.optimizer_mb || 0) + (est.adapter_mb || 0) + gradOh;
    const verdict = est.verdict || 'green';
    const state = _verdictToState(verdict);

    if (!totalGpu) {
      const el = (id, val) => { const e = $(id); if (e) e.textContent = val; };
      el('vram-estimated', `~${(peak / 1024).toFixed(1)} GB estimated`);
      el('vram-available', '/ GPU not detected');
      el('ez-vram-estimate', `~${(peak / 1024).toFixed(1)} GB`);
      el('ez-vram-total', '? GB');
      _setTextState($('vram-estimated'), 'muted');
      _setTextState($('ez-vram-estimate'), 'muted');
      const ezStatus = $('ez-vram-status');
      if (ezStatus) { ezStatus.textContent = 'GPU not detected'; _setTextState(ezStatus, 'muted'); }
      return;
    }

    // Bar segments scale against total GPU (system-used is implicit context)
    const barBase = totalGpu;
    const modelPct = (est.model_mb / barBase) * 100;
    const actPct = (est.activation_mb / barBase) * 100;
    const optPct = (optTotal / barBase) * 100;

    const segModel = $('vram-seg-model');
    const segAct = $('vram-seg-act');
    const segOpt = $('vram-seg-opt');
    if (segModel) { segModel.style.width = modelPct.toFixed(1) + '%'; segModel.title = 'Model weights: ~' + (est.model_mb / 1024).toFixed(1) + ' GB'; }
    if (segAct) { segAct.style.width = actPct.toFixed(1) + '%'; segAct.title = 'Activations: ~' + (est.activation_mb / 1024).toFixed(1) + ' GB'; }
    if (segOpt) { segOpt.style.width = optPct.toFixed(1) + '%'; segOpt.title = 'Optimizer+Grad+Overhead: ~' + (optTotal / 1024).toFixed(1) + ' GB'; }
    const bar = $('vram-bar');
    if (bar) bar.title = 'Model: ~' + (est.model_mb / 1024).toFixed(1) + ' GB\nActivations: ~' + (est.activation_mb / 1024).toFixed(1) + ' GB\nOptimizer+Grad: ~' + (optTotal / 1024).toFixed(1) + ' GB\nPeak: ~' + (peak / 1024).toFixed(1) + ' GB' + (sysUsed > 100 ? '\nSystem in use: ~' + (sysUsed / 1024).toFixed(1) + ' GB' : '');

    // Labels
    const el = (id, val) => { const e = $(id); if (e) e.textContent = val; };
    el('vram-lbl-model', `Model ${(est.model_mb / 1024).toFixed(1)}G`);
    el('vram-lbl-act', `Activations ${(est.activation_mb / 1024).toFixed(1)}G`);
    el('vram-lbl-opt', `Opt+Grad ${(optTotal / 1024).toFixed(1)}G`);

    // Totals: show effective available (total minus system usage)
    const estEl = $('vram-estimated');
    const availEl = $('vram-available');
    if (estEl) {
      estEl.textContent = `~${(peak / 1024).toFixed(1)} GB estimated`;
      _setTextState(estEl, state);
    }
    if (availEl) {
      if (sysUsed > 100) {
        availEl.textContent = `/ ${(available / 1024).toFixed(1)} GB available (${(sysUsed / 1024).toFixed(1)} GB in use)`;
      } else {
        availEl.textContent = `/ ${(totalGpu / 1024).toFixed(1)} GB available`;
      }
    }

    // Ez mode VRAM mirror
    const ezEst = $('ez-vram-estimate');
    const ezTotal = $('ez-vram-total');
    const ezStatus = $('ez-vram-status');
    if (ezEst) {
      ezEst.textContent = `~${(peak / 1024).toFixed(1)} GB`;
      _setTextState(ezEst, state);
    }
    if (ezTotal) ezTotal.textContent = `${Math.floor((available > 0 ? available : totalGpu) / 1024)} GB`;
    if (ezStatus) {
      const msg = verdict === 'red' ? '[!] WILL OOM' : verdict === 'yellow' ? '[!] tight fit' : '[ok] fits';
      ezStatus.textContent = msg;
      _setTextState(ezStatus, state);
    }
  }

  let _debounce = null;
  function _debouncedRecalculate() {
    clearTimeout(_debounce);
    _debounce = setTimeout(recalculate, 150);
  }

  function init() {
    const inputs = document.querySelectorAll('.vram-input');
    inputs.forEach(input => {
      const event = input.type === 'checkbox' || input.tagName === 'SELECT' ? 'change' : 'input';
      input.addEventListener(event, _debouncedRecalculate);
    });

    recalculate();
  }

  return { init, recalculate };

})();
