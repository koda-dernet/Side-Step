/* ============================================================
   Side-Step GUI — Field Validation Engine
   Data-driven validation using data-validate attributes.
   ============================================================ */

const Validation = (() => {
  'use strict';

  /**
   * Parse a data-validate attribute string into an array of rule objects.
   * Supported rules:
   *   gt:N   — value > N
   *   gte:N  — value >= N
   *   lt:N   — value < N
   *   lte:N  — value <= N
   *   range:MIN:MAX — MIN <= value <= MAX
   *   required — non-empty
   *   int — must be an integer
   */
  function _parseRules(raw) {
    if (!raw) return [];
    return raw.split(/\s+/).map((token) => {
      const parts = token.split(':');
      const type = parts[0];
      if (type === 'gt' || type === 'gte' || type === 'lt' || type === 'lte') {
        return { type, val: parseFloat(parts[1]) };
      }
      if (type === 'range') {
        return { type, min: parseFloat(parts[1]), max: parseFloat(parts[2]) };
      }
      return { type };
    });
  }

  // ---- Single-field check -------------------------------------------------

  function _checkRules(value, rules) {
    for (const rule of rules) {
      if (rule.type !== 'required' && value !== '' && isNaN(Number(value))) return 'Must be a number';
      switch (rule.type) {
        case 'required':
          if (value === '' || value === null || value === undefined) return 'Required';
          break;
        case 'int':
          if (value !== '' && !Number.isInteger(Number(value))) return 'Must be an integer';
          break;
        case 'gt':
          if (value !== '' && Number(value) <= rule.val) return 'Must be > ' + rule.val;
          break;
        case 'gte':
          if (value !== '' && Number(value) < rule.val) return 'Must be \u2265 ' + rule.val;
          break;
        case 'lt':
          if (value !== '' && Number(value) >= rule.val) return 'Must be < ' + rule.val;
          break;
        case 'lte':
          if (value !== '' && Number(value) > rule.val) return 'Must be \u2264 ' + rule.val;
          break;
        case 'range':
          if (value !== '') {
            const n = Number(value);
            if (n < rule.min || n > rule.max) return 'Must be ' + rule.min + ' \u2013 ' + rule.max;
          }
          break;
      }
    }
    return null;
  }

  // ---- Validate one input -------------------------------------------------

  function validateField(input) {
    const raw = input.getAttribute('data-validate');
    if (!raw) return null;
    const rules = _parseRules(raw);
    const error = _checkRules(input.value, rules);
    const errorEl = input.parentElement.querySelector('.field-error');
    if (error) {
      input.classList.add('invalid');
      if (errorEl) {
        errorEl.textContent = error;
        errorEl.style.display = 'block';
      }
    } else {
      input.classList.remove('invalid');
      if (errorEl) {
        errorEl.style.display = 'none';
      }
    }
    return error;
  }

  // ---- Validate all fields ------------------------------------------------

  function validateAll() {
    let valid = true;
    document.querySelectorAll('[data-validate]').forEach((input) => {
      if (input.offsetParent === null) return; // skip hidden
      const err = validateField(input);
      if (err) valid = false;
    });
    return valid;
  }

  // ---- Wire listeners -----------------------------------------------------

  function init() {
    document.querySelectorAll('[data-validate]').forEach((input) => {
      const handler = () => validateField(input);
      input.addEventListener('input', handler);
      input.addEventListener('change', handler);
    });
  }

  // ---- XSS-safe HTML escaping ---------------------------------------------

  function esc(str) {
    if (typeof str !== 'string') return String(str ?? '');
    return str.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;').replace(/'/g, '&#39;');
  }

  return { init, validateField, validateAll, esc };
})();
