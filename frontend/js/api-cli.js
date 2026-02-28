/* Side-Step GUI — CLI Command Builder (extracted from api.js) */

const APICli = (() => {
  "use strict";

  function buildCLICommand(config) {
    const parts = ['sidestep train'];
    const _platformPrefetchDefault = () => {
      const p = String(window.__SIDESTEP_PLATFORM__ || '').toLowerCase();
      return p.startsWith('win') ? '0' : '2';
    };
    const _isWin = String(window.__SIDESTEP_PLATFORM__ || '').toLowerCase().startsWith('win');
    const _sh = (val) => {
      const s = String(val);
      if (/^[A-Za-z0-9_./:@%+=,-]+$/.test(s)) return s;
      if (_isWin) return '"' + s.replace(/"/g, '\\"') + '"';
      return "'" + s.replace(/'/g, "'\"'\"'") + "'";
    };
    const add = (flag, val) => { if (val !== undefined && val !== null && val !== '') parts.push(flag + ' ' + _sh(val)); };
    const addList = (flag, val) => {
      if (val === undefined || val === null || val === '') return;
      const items = Array.isArray(val) ? val : String(val).trim().split(/\s+/).filter(Boolean);
      if (!items.length) return;
      parts.push(flag + ' ' + items.map(_sh).join(' '));
    };
    const addBool = (flag, val) => { if (val) parts.push(flag); };
    const addNoBool = (flag, val) => { if (val === false) parts.push('--no-' + flag.replace(/^--/, '')); };
    const addNonDefault = (flag, val, def) => { if (val !== undefined && val !== null && val !== '' && String(val) !== String(def)) add(flag, val); };

    add('--checkpoint-dir', config.checkpoint_dir);
    add('--model', config.model_variant);
    add('--adapter', config.adapter_type);
    add('--dataset-dir', config.dataset_dir);
    add('--run-name', config.run_name);
    if (config.output_dir) add('--output-dir', config.output_dir);

    const at = config.adapter_type;
    if (at === 'lora' || at === 'dora') {
      add('--rank', config.rank); add('--alpha', config.alpha);
      addNonDefault('--dropout', config.dropout, '0.1');
    } else if (at === 'lokr') {
      add('--lokr-linear-dim', config.lokr_linear_dim); add('--lokr-linear-alpha', config.lokr_linear_alpha);
      addNonDefault('--lokr-factor', config.lokr_factor, '-1');
      addBool('--lokr-decompose-both', config.lokr_decompose_both); addBool('--lokr-use-tucker', config.lokr_use_tucker);
      addBool('--lokr-use-scalar', config.lokr_use_scalar); addBool('--lokr-weight-decompose', config.lokr_weight_decompose);
    } else if (at === 'loha') {
      add('--loha-linear-dim', config.loha_linear_dim); add('--loha-linear-alpha', config.loha_linear_alpha);
      addNonDefault('--loha-factor', config.loha_factor, '-1');
      addBool('--loha-use-tucker', config.loha_use_tucker); addBool('--loha-use-scalar', config.loha_use_scalar);
    } else if (at === 'oft') {
      add('--oft-block-size', config.oft_block_size); addBool('--oft-coft', config.oft_coft);
      addNonDefault('--oft-eps', config.oft_eps, '6e-5');
    }

    addNonDefault('--attention-type', config.attention_type, 'both');
    if (config.attention_type === 'both') {
      if (config.self_projections && config.self_projections !== 'q_proj k_proj v_proj o_proj') {
        addList('--self-target-modules', config.self_projections);
      }
      if (config.cross_projections && config.cross_projections !== 'q_proj k_proj v_proj o_proj') {
        addList('--cross-target-modules', config.cross_projections);
      }
    }
    if (config.projections && config.projections !== 'q_proj k_proj v_proj o_proj') addList('--target-modules', config.projections);
    addBool('--target-mlp', config.target_mlp);
    addNoBool('--target-mlp', config.target_mlp);
    addNonDefault('--bias', config.bias, 'none');

    add('--lr', config.lr); add('--batch-size', config.batch_size);
    add('--gradient-accumulation', config.grad_accum); add('--epochs', config.epochs);
    add('--warmup-steps', config.warmup_steps);
    if (config.max_steps && config.max_steps !== '0') add('--max-steps', config.max_steps);

    addNonDefault('--shift', config.shift, '3.0');
    addNonDefault('--num-inference-steps', config.num_inference_steps, '8');
    addNonDefault('--cfg-ratio', config.cfg_ratio, '0.15');
    addNonDefault('--loss-weighting', config.loss_weighting, 'none');
    if (config.loss_weighting === 'min_snr') addNonDefault('--snr-gamma', config.snr_gamma, '5.0');

    add('--optimizer-type', config.optimizer_type); add('--scheduler-type', config.scheduler);
    if (config.scheduler === 'custom' && config.scheduler_formula) add('--scheduler-formula', config.scheduler_formula);
    if (config.device && config.device !== 'auto') add('--device', config.device);
    if (config.precision && config.precision !== 'auto') add('--precision', config.precision);

    addNoBool('--gradient-checkpointing', config.gradient_checkpointing);
    addBool('--offload-encoder', config.offload_encoder);
    addNoBool('--offload-encoder', config.offload_encoder);
    addNonDefault('--gradient-checkpointing-ratio', config.gradient_checkpointing_ratio, '1.0');
    if (config.chunk_duration && config.chunk_duration !== '0') {
      add('--chunk-duration', config.chunk_duration);
      addNonDefault('--chunk-decay-every', config.chunk_decay_every, '10');
    }

    add('--save-every', config.save_every); add('--log-every', config.log_every);
    addNonDefault('--log-heavy-every', config.log_heavy_every, '50');
    addBool('--save-best', config.save_best);
    addNoBool('--save-best', config.save_best);
    if (config.save_best_after && config.save_best_after !== '0') add('--save-best-after', config.save_best_after);
    if (config.early_stop && config.early_stop !== '0') add('--early-stop-patience', config.early_stop);
    if (config.resume_from) add('--resume-from', config.resume_from);
    if (config.resume_from && config.strict_resume === false) parts.push('--no-strict-resume');
    if (config.log_dir) add('--log-dir', config.log_dir);

    add('--weight-decay', config.weight_decay); add('--max-grad-norm', config.max_grad_norm);
    add('--seed', config.seed);
    if (config.dataset_repeats && config.dataset_repeats !== '1') add('--dataset-repeats', config.dataset_repeats);
    addNonDefault('--warmup-start-factor', config.warmup_start_factor, '0.1');
    addNonDefault('--cosine-eta-min-ratio', config.cosine_eta_min_ratio, '0.01');
    addNonDefault('--cosine-restarts-count', config.cosine_restarts_count, '4');
    if (config.ema_decay && config.ema_decay !== '0') add('--ema-decay', config.ema_decay);
    if (config.val_split && config.val_split !== '0') add('--val-split', config.val_split);
    if (config.adaptive_timestep_ratio && config.adaptive_timestep_ratio !== '0') add('--adaptive-timestep-ratio', config.adaptive_timestep_ratio);
    if (config.save_best_every_n_steps && config.save_best_every_n_steps !== '0') add('--save-best-every-n-steps', config.save_best_every_n_steps);
    add('--timestep-mu', config.timestep_mu);
    add('--timestep-sigma', config.timestep_sigma);
    addBool('--ignore-fisher-map', config.ignore_fisher_map);
    add('--num-workers', config.num_workers);
    addNonDefault('--prefetch-factor', config.prefetch_factor, _platformPrefetchDefault());
    addBool('--pin-memory', config.pin_memory);
    addNoBool('--pin-memory', config.pin_memory);
    if (config.persistent_workers === false) parts.push('--no-persistent-workers');

    return parts.join(_isWin ? ' ^\n  ' : ' \\\n  ');
  }

  return { buildCLICommand };
})();
