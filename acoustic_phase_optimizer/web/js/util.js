// Small DOM and formatting helpers. No dependencies.

/** Create an element. attrs: class, text, style (object), dataset, attrs (raw attributes), on<Event>, or DOM props. */
export function h(tag, attrs, ...children) {
  const el = document.createElement(tag);
  if (attrs) {
    for (const [k, v] of Object.entries(attrs)) {
      if (v == null || v === false) continue;
      if (k === 'class') el.className = v;
      else if (k === 'text') el.textContent = v;
      else if (k === 'style' && typeof v === 'object') Object.assign(el.style, v);
      else if (k === 'dataset') Object.assign(el.dataset, v);
      else if (k === 'attrs') for (const [a, b] of Object.entries(v)) { if (b != null && b !== false) el.setAttribute(a, b); }
      else if (k.startsWith('on') && typeof v === 'function') el.addEventListener(k.slice(2).toLowerCase(), v);
      else el[k] = v;
    }
  }
  append(el, children);
  return el;
}

function append(el, children) {
  for (const c of children) {
    if (c == null || c === false) continue;
    if (Array.isArray(c)) append(el, c);
    else el.append(c instanceof Node ? c : document.createTextNode(String(c)));
  }
}

export function clear(el) { while (el.firstChild) el.removeChild(el.firstChild); return el; }
export const clamp = (v, lo, hi) => Math.min(hi, Math.max(lo, v));
export const round = (v, nd = 0) => { const k = 10 ** nd; return Math.round(v * k) / k; };
export const clone = (o) => JSON.parse(JSON.stringify(o));
export const uid = () => Math.random().toString(16).slice(2, 10);

export function fmt(v, nd = 1, signed = false) {
  if (v == null || !Number.isFinite(v)) return '–';
  const s = v.toFixed(nd);
  const zero = Number(s) === 0;
  return (signed && !zero && v > 0 ? '+' : '') + (zero ? (0).toFixed(nd) : s).replace('-', '−');
}

export function fmtHz(f) {
  if (f >= 1000) { const k = f / 1000; return (Number.isInteger(round(k, 2)) ? round(k) : round(k, k < 10 ? 2 : 1)) + ' kHz'; }
  return (f < 100 ? round(f, 1) : round(f)) + ' Hz';
}

export function debounce(fn, ms) {
  let t = null;
  const wrapped = (...a) => { clearTimeout(t); t = setTimeout(() => { t = null; fn(...a); }, ms); };
  wrapped.flush = (...a) => { if (t) { clearTimeout(t); t = null; fn(...a); } };
  wrapped.cancel = () => { clearTimeout(t); t = null; };
  return wrapped;
}

export function download(filename, content, mime = 'application/octet-stream') {
  const blob = content instanceof Blob ? content : new Blob([content], { type: mime });
  const url = URL.createObjectURL(blob);
  const a = h('a', { href: url, download: filename });
  document.body.append(a);
  a.click();
  a.remove();
  setTimeout(() => URL.revokeObjectURL(url), 4000);
}

export function toast(message, { error = false, action = null, ms = 4200 } = {}) {
  const box = document.getElementById('toasts');
  const el = h('div', { class: 'toast' + (error ? ' error' : ''), attrs: { role: error ? 'alert' : 'status' } },
    h('span', { text: message }));
  if (action) el.append(h('button', { class: 'btn small', text: action.label, onclick: () => { action.run(); el.remove(); } }));
  box.append(el);
  setTimeout(() => el.remove(), error ? Math.max(ms, 7000) : ms);
  return el;
}

const ICONS = {
  undo: 'M9 14 4 9l5-5M4 9h10.5a5.5 5.5 0 0 1 0 11H11',
  redo: 'm15 14 5-5-5-5M20 9H9.5a5.5 5.5 0 0 0 0 11H13',
  save: 'M5 3h11l4 4v13a1 1 0 0 1-1 1H5a1 1 0 0 1-1-1V4a1 1 0 0 1 1-1zM8 3v5h7V3M8 21v-7h8v7',
  folder: 'M3 6a1 1 0 0 1 1-1h5l2 2h9a1 1 0 0 1 1 1v10a1 1 0 0 1-1 1H4a1 1 0 0 1-1-1z',
  export: 'M12 3v12m0-12 4 4m-4-4-4 4M5 14v5a1 1 0 0 0 1 1h12a1 1 0 0 0 1-1v-5',
  sun: 'M12 4V2m0 20v-2m8-8h2M2 12h2m13.7-5.7 1.4-1.4M4.9 19.1l1.4-1.4m0-11.4L4.9 4.9m14.2 14.2-1.4-1.4M16 12a4 4 0 1 1-8 0 4 4 0 0 1 8 0z',
  moon: 'M20 14.5A8 8 0 0 1 9.5 4 8 8 0 1 0 20 14.5z',
  plus: 'M12 5v14M5 12h14',
  trash: 'M4 7h16M10 11v6m4-6v6M6 7l1 12a1 1 0 0 0 1 1h8a1 1 0 0 0 1-1l1-12M9 7V4h6v3',
  copy: 'M9 9h10a1 1 0 0 1 1 1v10a1 1 0 0 1-1 1H9a1 1 0 0 1-1-1V10a1 1 0 0 1 1-1zM5 15H4a1 1 0 0 1-1-1V4a1 1 0 0 1 1-1h10a1 1 0 0 1 1 1v1',
  mirror: 'M12 3v18M8 7 4 12l4 5V7zm8 0 4 5-4 5V7z',
  link: 'M10 14a4 4 0 0 0 5.7 0l3-3a4 4 0 0 0-5.7-5.7l-1 1m2 3.700a4 4 0 0 0-5.7 0l-3 3A4 4 0 0 0 11 20.700l1-1',
  mute: 'M4 9v6h4l5 4V5L8 9H4zm13 0 4 6m0-6-4 6',
  sound: 'M4 9v6h4l5 4V5L8 9H4zm12-1a5 5 0 0 1 0 8m2.500-10.500a9 9 0 0 1 0 13',
  chevron: 'm6 9 6 6 6-6',
  bolt: 'M13 2 4 14h7l-1 8 9-12h-7l1-8z',
  target: 'M12 21a9 9 0 1 0 0-18 9 9 0 0 0 0 18zm0-4a5 5 0 1 0 0-10 5 5 0 0 0 0 10zm0-4a1 1 0 1 0 0-2 1 1 0 0 0 0 2z',
  sliders: 'M4 6h9m4 0h3M4 12h3m4 0h9M4 18h11m4 0h1M15 4v4M9 10v4M17 16v4',
  table: 'M4 5h16v14H4zM4 10h16M4 15h16M10 5v14',
  x: 'M6 6l12 12M18 6 6 18',
  lock: 'M7 11V8a5 5 0 0 1 10 0v3M6 11h12a1 1 0 0 1 1 1v8a1 1 0 0 1-1 1H6a1 1 0 0 1-1-1v-8a1 1 0 0 1 1-1z',
  cube: 'M12 2 3 7v10l9 5 9-5V7l-9-5zM3 7l9 5 9-5M12 12v10',
  search: 'M11 18a7 7 0 1 0 0-14 7 7 0 0 0 0 14zm9 3-4.35-4.35',
};

export function icon(name) {
  const svg = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
  svg.setAttribute('viewBox', '0 0 24 24');
  svg.setAttribute('fill', 'none');
  svg.setAttribute('stroke', 'currentColor');
  svg.setAttribute('stroke-width', '1.8');
  svg.setAttribute('stroke-linecap', 'round');
  svg.setAttribute('stroke-linejoin', 'round');
  svg.setAttribute('aria-hidden', 'true');
  const p = document.createElementNS('http://www.w3.org/2000/svg', 'path');
  p.setAttribute('d', ICONS[name] || '');
  svg.append(p);
  return svg;
}

export const STATUS_ICON = { good: '✓', warning: '!', serious: '▲', critical: '✕' };

export function statusBadge(level, label) {
  return h('span', { class: 'status' },
    h('i', { class: level, style: { background: `var(--${level})` }, text: STATUS_ICON[level] || '' }),
    label ? h('span', { text: label }) : null);
}
