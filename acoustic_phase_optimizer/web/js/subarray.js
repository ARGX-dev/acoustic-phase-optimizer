// Sub array builder dialog: line / arc, cardioid (gradient) and end-fire. The engine builds the boxes;
// this shows a live top-view sketch with each box's delay and polarity before anything is added.

import { api } from './api.js';
import { openDialog } from './dialog.js';
import { mutate, select, state } from './store.js';
import { clamp, fmt, h, toast } from './util.js';
import { dispStep, fromDisp, len, lenUnit, toDisp } from './units.js';

const KINDS = [
  ['line', 'Line / arc', 'Boxes in a row across the room. A spread angle delays the outer boxes as if the row were bent into an arc, widening the narrow beam a long straight row makes.'],
  ['cardioid', 'Cardioid', 'Gradient pairs: a second box behind each front box, polarity inverted and delayed by the gap. They cancel behind the array, quieting the stage, and still add in front.'],
  ['endfire', 'End-fire', 'A column of boxes pointing into the audience, each delayed by its distance from the rearmost one. Everything arrives together in front; behind, the arrivals spread out and cancel.'],
];

function speedOfSound(env) {
  // For the preview only; the engine uses the full temperature / humidity / pressure model.
  return 331.3 * Math.sqrt(1 + env.temperature_c / 273.15);
}

/** Preview of box positions and pattern delays, mirroring engine/subarray.py. */
function layout(o, c) {
  const out = [];
  const ms = (m) => (m / c) * 1000;
  if (o.kind === 'line') {
    const hw = ((o.count - 1) * o.spacing) / 2;
    const r = o.spread > 0 && o.count > 1 ? hw / Math.sin((Math.min(o.spread, 170) * Math.PI) / 360) : 0;
    for (let i = 0; i < o.count; i++) {
      const dx = (i - (o.count - 1) / 2) * o.spacing;
      out.push({ dx, dy: 0, d: r ? ms(r - Math.sqrt(Math.max(r * r - dx * dx, 0))) : 0, inv: false });
    }
  } else if (o.kind === 'cardioid') {
    for (let i = 0; i < o.count; i++) {
      const dx = (i - (o.count - 1) / 2) * o.spacing;
      out.push({ dx, dy: 0, d: 0, inv: false }, { dx, dy: -o.depth, d: ms(o.depth), inv: true });
    }
  } else {
    for (let i = 0; i < o.count; i++) out.push({ dx: 0, dy: -(o.count - 1 - i) * o.spacing, d: ms(i * o.spacing), inv: false });
  }
  return out;
}

function sketch(boxes, o) {
  const NS = 'http://www.w3.org/2000/svg';
  const el = (tag, attrs, text) => {
    const e = document.createElementNS(NS, tag);
    for (const [k, v] of Object.entries(attrs)) e.setAttribute(k, v);
    if (text != null) e.textContent = text;
    return e;
  };
  const W = 520, H = 190;
  const xs = boxes.map((b) => b.dx), ys = boxes.map((b) => b.dy);
  const spanX = Math.max(2, Math.max(...xs) - Math.min(...xs) + 2), spanY = Math.max(2, Math.max(...ys) - Math.min(...ys) + 2);
  const s = Math.min((W - 60) / spanX, (H - 70) / spanY, 70);
  const cx = W / 2 - ((Math.max(...xs) + Math.min(...xs)) / 2) * s, cy = 40 - Math.max(...ys) * s + ((H - 70) - spanY * s) / 2 + s;
  const svg = el('svg', { viewBox: `0 0 ${W} ${H}`, class: 'sub-sketch', role: 'img', 'aria-label': 'Top view of the array with each box’s delay' });
  svg.append(el('text', { x: 10, y: 16, class: 'k' }, 'Audience ↓   (top view, stage at the top)'));
  const bw = Math.max(26, Math.min(40, s * 0.6)), bh = Math.max(20, Math.min(28, s * 0.45));
  for (const b of boxes) {
    const x = cx + b.dx * s, y = cy - b.dy * s; // further from stage = further down
    const g = el('g', {});
    g.append(el('rect', { x: x - bw / 2, y: y - bh / 2, width: bw, height: bh, rx: 3, class: b.inv ? 'box inv' : 'box' }));
    g.append(el('text', { x, y: y + 4, 'text-anchor': 'middle', class: 'pol' }, b.inv ? '−' : '+'));
    g.append(el('text', { x, y: y + bh / 2 + 13, 'text-anchor': 'middle', class: 'd' }, `${fmt(b.d, 2)} ms`));
    svg.append(g);
  }
  svg.append(el('text', { x: W - 10, y: H - 8, 'text-anchor': 'end', class: 'k' },
    o.kind === 'cardioid' ? '− = polarity inverted · delays are added to the array’s own alignment' : 'Delays are added to the array’s own alignment'));
  return svg;
}

export async function openSubArrayBuilder() {
  const p = state.project;
  const subs = state.meta.speaker_types.filter((t) => t.role === 'sub');
  const existing = p.speakers.filter((s) => s.role === 'sub');
  const o = {
    kind: 'cardioid', model: existing[0]?.model && subs.some((t) => t.id === existing[0].model) ? existing[0].model : (subs[0]?.id || 'sub_18'),
    count: 2, spacing: 1.4, depth: 1.0, spread: 30,
    x: Math.round((p.room.width / 2) * 10) / 10,
    y: Math.round(clamp(existing[0]?.y ?? Math.max(1, p.audience.y0 - 2), 0.5, p.room.length - 0.5) * 10) / 10,
    replace: existing.length > 0,
  };
  const c = speedOfSound(p.environment);
  const preview = h('div', { class: 'sub-preview' });
  const about = h('p', { class: 'hint' });
  const fields = h('div', { class: 'grid3' });
  const kindSeg = h('div', { class: 'seg block', attrs: { role: 'radiogroup', 'aria-label': 'Array type' } });

  const numIn = (label, key, { min, max, step, unit = '', nd = 1 }) => {
    // Lengths are kept in metres (min/max too) and shown in the chosen unit.
    const isLen = unit === 'm';
    const D = (v) => (isLen ? toDisp(v) : v), M = (v) => (isLen ? fromDisp(v) : v);
    const dstep = isLen ? dispStep(step) : step;
    const show = () => String(Number(D(o[key]).toFixed(nd)));
    const input = h('input', { type: 'text', class: 'num', value: show(), attrs: { 'aria-label': label, inputmode: 'decimal' } });
    input.addEventListener('input', () => {
      const v = parseFloat(input.value.replace(',', '.'));
      if (Number.isFinite(v)) { o[key] = clamp(M(v), min, max); draw(); }
    });
    input.addEventListener('change', () => { input.value = show(); });
    input.addEventListener('keydown', (e) => {
      if (e.key !== 'ArrowUp' && e.key !== 'ArrowDown') return;
      e.preventDefault();
      o[key] = clamp(M(Math.round((D(o[key]) + (e.key === 'ArrowUp' ? dstep : -dstep)) / dstep) * dstep), min, max);
      input.value = show();
      draw();
    });
    return h('label', { class: 'field' }, h('span', { text: label }), h('span', { class: 'input-wrap' }, input, unit ? h('em', { text: isLen ? lenUnit() : unit }) : null));
  };

  function renderFields() {
    fields.replaceChildren(...[
      h('label', { class: 'field wide' }, h('span', { text: 'Model' }),
        h('select', { attrs: { 'aria-label': 'Subwoofer model' }, onchange: (e) => { o.model = e.target.value; } },
          subs.map((t) => h('option', { value: t.id, text: t.label, selected: t.id === o.model })))),
      numIn(o.kind === 'endfire' ? 'Boxes' : o.kind === 'cardioid' ? 'Pairs' : 'Boxes', 'count', { min: 1, max: o.kind === 'cardioid' ? 12 : 24, step: 1, nd: 0 }),
      numIn(o.kind === 'endfire' ? 'Gap front to back' : 'Spacing', 'spacing', { min: 0.3, max: 10, step: 0.1, unit: 'm' }),
      o.kind === 'cardioid' ? numIn('Rear box behind by', 'depth', { min: 0.3, max: 4, step: 0.05, unit: 'm', nd: 2 }) : null,
      o.kind === 'line' ? numIn('Spread', 'spread', { min: 0, max: 170, step: 5, unit: '°', nd: 0 }) : null,
      numIn(o.kind === 'endfire' ? 'Front box, from stage' : 'From stage', 'y', { min: 0.3, max: p.room.length - 0.3, step: 0.1, unit: 'm' }),
      numIn('Centre, across', 'x', { min: 0.3, max: p.room.width - 0.3, step: 0.1, unit: 'm' }),
    ].filter(Boolean));
  }
  function renderKinds() {
    kindSeg.replaceChildren(...KINDS.map(([id, label]) => h('button', { text: label, attrs: { role: 'radio', 'aria-checked': String(o.kind === id), 'aria-pressed': String(o.kind === id) },
      onclick: () => { o.kind = id; if (id === 'endfire') o.spacing = Math.min(o.spacing, 1.2); renderKinds(); renderFields(); draw(); } })));
  }
  function draw() {
    const boxes = layout({ ...o, count: Math.round(o.count) }, c);
    const kind = KINDS.find((k) => k[0] === o.kind);
    about.textContent = kind[2] + (o.kind === 'cardioid'
      ? ` With a ${len(o.depth, 2)} gap the pair adds most in front around ${fmt(c / (4 * o.depth), 0)} Hz; much lower, the front gets weaker and needs gain.`
      : o.kind === 'endfire' ? ` Spacing of ${len(o.spacing, 2)}: the rear rejection is strongest around ${fmt(c / (4 * o.spacing), 0)} Hz.` : '');
    preview.replaceChildren(sketch(boxes, o));
  }
  renderKinds(); renderFields(); draw();
  const replace = h('input', { type: 'checkbox', checked: o.replace, onchange: (e) => { o.replace = e.target.checked; } });

  const body = h('div', { class: 'sub-builder' }, kindSeg, about, preview, fields,
    h('label', { class: 'check' }, replace, h('span', { text: existing.length ? `Replace the ${existing.length} subwoofer${existing.length > 1 ? 's' : ''} already in the venue` : 'Replace existing subwoofers' })));
  const r = await openDialog({ title: 'Build a sub array', body, wide: true,
    actions: [{ label: 'Cancel' }, { label: 'Add the array', kind: 'primary', run: async () => {
      try {
        const res = await api.subarray(state.project, { kind: o.kind, model: o.model, count: Math.round(o.count), spacing: o.spacing,
          depth: o.depth, spread_deg: o.kind === 'line' ? o.spread : 0, x: o.x, y: o.y, replace: o.replace });
        const ids = new Set(state.project.speakers.map((s) => s.id));
        mutate((pr) => { pr.speakers = res.speakers; }, { structural: true });
        const first = res.speakers.find((s) => !ids.has(s.id));
        if (first) select('speaker', first.id);
        toast(`Added ${res.added} subwoofer${res.added > 1 ? 's' : ''}${res.removed ? `, replaced ${res.removed}` : ''}. Run Quick align to time the array to the mains.`);
        return true;
      } catch (e) {
        toast(e.message, { error: true });
        return false;
      }
    } }] });
  return r;
}
