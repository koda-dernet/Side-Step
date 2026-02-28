/* Side-Step GUI — DOM Cache
   Provides a caching $ that avoids repeated getElementById lookups.
   Loaded after fallback.js — overwrites the basic $ with a cached version. */

const _domCache = new Map();
window.$ = (id) => {
  if (_domCache.has(id)) return _domCache.get(id);
  const el = document.getElementById(id);
  if (el) _domCache.set(id, el);
  return el || null;
};
