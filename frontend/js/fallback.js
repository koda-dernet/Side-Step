/* Side-Step GUI — Global Utilities (fallback.js)
   Defines $, debounce, on, BatchDOM used by all modules. */

window.$ = (id) => document.getElementById(id);

window.debounce = (fn, delay) => {
  let timeout;
  return (...args) => {
    clearTimeout(timeout);
    timeout = setTimeout(() => fn(...args), delay);
  };
};

window.on = (evt, sel, fn) => {
  document.addEventListener(evt, (e) => {
    const t = e.target.closest(sel);
    if (t) fn.call(t, e);
  });
};

window._pathBasename = (p) => String(p || '').split(/[/\\]/).filter(Boolean).pop() || "";
window._joinPath = function() { return Array.from(arguments).filter(Boolean).join("/").replace(/[/\\]+/g, "/"); };

window._fmtDuration = (seconds) => {
  const s = Math.floor(seconds);
  const h = Math.floor(s / 3600);
  const m = Math.floor((s % 3600) / 60);
  const sec = s % 60;
  if (h > 0) return `${h}h ${String(m).padStart(2, '0')}m ${String(sec).padStart(2, '0')}s`;
  if (m > 0) return `${m}m ${String(sec).padStart(2, '0')}s`;
  return `${sec}s`;
};

window._esc = (s) => typeof Validation !== 'undefined'
  ? Validation.esc(s)
  : String(s ?? '').replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;');

window.BatchDOM = {
  fragment: () => document.createDocumentFragment(),
  appendChildren: (parent, children) => {
    children.forEach(child => parent.appendChild(child));
    return parent;
  },
  setOptions: (select, options, currentValue = '') => {
    select.innerHTML = '';
    (options || []).forEach(opt => {
      const option = document.createElement('option');
      option.value = opt.value ?? opt;
      option.textContent = opt.label ?? opt;
      if ((opt.value ?? opt) === currentValue) option.selected = true;
      select.appendChild(option);
    });
    if (currentValue && [...select.options].some(o => o.value === currentValue)) {
      select.value = currentValue;
    }
    return select;
  }
};
