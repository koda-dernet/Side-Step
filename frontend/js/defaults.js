/* Side-Step GUI — Centralized Defaults Loader
   Reads defaults.json (later: /api/defaults) and applies to all form fields. */

const Defaults = (() => {
  'use strict';
  let _data = {};

  async function load() {
    try {
      const resp = await fetch('js/defaults.json');
      const raw = await resp.json();
      // Strip _comment keys
      Object.keys(_data).forEach(k => delete _data[k]);
      Object.keys(raw).forEach(k => { if (!k.startsWith('_')) _data[k] = raw[k]; });
    } catch (e) { console.warn('[Defaults] Could not load defaults.json:', e); }
  }

  function get(id) {
    return _data[id] !== undefined ? String(_data[id]) : undefined;
  }

  function getBool(id) {
    return _data[id] === true;
  }

  function apply() {
    Object.entries(_data).forEach(([id, val]) => {
      const el = document.getElementById(id);
      if (!el) return;
      if (el.type === 'checkbox') {
        el.checked = val === true;
      } else {
        el.value = String(val);
        el.dataset.default = String(val);
      }
    });
  }

  function all() { return Object.assign({}, _data); }

  return { load, get, getBool, apply, all };
})();
