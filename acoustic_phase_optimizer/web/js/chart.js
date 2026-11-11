// Response panel: magnitude, phase and arrival times at the probe, a mic, or the audience average.
// One y-scale per plot; a crosshair reads every series at once; a table view mirrors the data.

import { resample } from './measure.js';
import { on, setUI, state } from './store.js';
import { clamp, clear, fmt, fmtHz, h, icon } from './util.js';
import { cssVar, seriesColor } from './theme.js';

const OCTAVES = [31.5, 63, 125, 250, 500, 1000, 2000, 4000, 8000, 16000];
const BANDS = [63, 125, 250, 500, 1000, 2000, 4000, 8000];
const ORDER_NAMES = ['Direct sound', '1st-order reflection', '2nd-order reflection', '3rd-order reflection', '4th-order reflection'];

export function createChart(toolbar, legend, root) {
  const canvas = h('canvas', { attrs: { 'aria-label': 'Frequency response chart' } });
  const tip = h('div', { class: 'tip', hidden: true });
  const tableBox = h('div', { class: 'chart-table', hidden: true });
  const empty = h('div', { class: 'empty-state', hidden: true });
  root.append(canvas, tableBox, empty, tip);
  const ctx = canvas.getContext('2d');
  const hidden = new Set();
  let W = 0, H = 0, dpr = 1, raf = 0;
  let model = null; // what is currently plotted
  let hoverX = null;

  // ---------------------------------------------------------------- toolbar
  function renderToolbar() {
    clear(toolbar);
    const src = h('select', { attrs: { 'aria-label': 'Response location' }, onchange: (e) => setUI({ source: e.target.value }) },
      h('option', { value: 'probe', text: 'Probe' }),
      state.project.mics.map((m) => h('option', { value: m.id, text: m.name })),
      h('option', { value: 'average', text: 'Audience average' }));
    src.value = state.ui.source;
    const modes = [['magnitude', 'Magnitude'], ['phase', 'Phase'], ['arrivals', 'Arrivals']];
    const average = state.ui.source === 'average';
    const seg = h('div', { class: 'seg', attrs: { role: 'group', 'aria-label': 'Chart type' } }, modes.map(([id, label]) =>
      h('button', { text: label, disabled: average && id !== 'magnitude', attrs: { 'aria-pressed': String(effectiveMode() === id) },
        onclick: () => setUI({ chart: id }) })));
    toolbar.append(h('span', { class: 'title', text: 'Response at' }), src, seg, h('span', { class: 'spacer' }),
      h('button', { class: 'icon-btn' + (state.ui.table ? ' on' : ''), title: 'Show the numbers as a table',
        attrs: { 'aria-pressed': String(state.ui.table), 'aria-label': 'Table view' }, onclick: () => setUI({ table: !state.ui.table }) }, icon('table')));
  }

  const effectiveMode = () => (state.ui.source === 'average' ? 'magnitude' : state.ui.chart);

  // ------------------------------------------------------------------ model
  function buildModel() {
    const mode = effectiveMode();
    const p = state.project;
    const c = { ink: cssVar('--ink'), muted: cssVar('--muted'), accent: cssVar('--accent') };
    if (state.ui.source === 'average') {
      const avg = state.sim?.average;
      if (!avg) return null;
      const f = state.sim.freqs;
      const tilt = p.settings.target_tilt_db_per_oct;
      const dev = f.map((x, i) => (x >= 100 && x <= 10000 && avg.spl[i] != null ? avg.spl[i] - tilt * Math.log2(x / 1000) : null)).filter((v) => v != null).sort((a, b) => a - b);
      const offset = dev.length ? dev[Math.floor(dev.length / 2)] : 0;
      return { mode, kind: 'lines', x: f, unit: 'dB', series: [
        { id: 'avg', name: 'Direct sound, audience average', color: c.ink, width: 2, values: avg.spl },
        { id: 'tot', name: 'Including reverberation', color: c.muted, width: 1.5, values: avg.total },
        { id: 'target', name: `Target (${fmt(tilt, 1, true)} dB/oct)`, color: c.accent, width: 1.5, values: f.map((x) => offset + tilt * Math.log2(x / 1000)) },
        ...measured(f, 'spl'),
      ] };
    }
    const pt = state.point;
    if (!pt) return null;
    const byId = new Map(pt.speakers.map((s) => [s.id, s]));
    const live = p.speakers.filter((s) => !s.mute && byId.get(s.id)?.spl);
    if (mode === 'arrivals') {
      const t0 = pt.first_arrival_ms;
      return { mode, kind: 'stems', unit: 'dB', events: pt.arrivals.map((a) => ({ ...a, t: a.t - t0, spk: p.speakers[a.speaker] })).filter((a) => a.spk && !hidden.has(a.spk.id)),
        series: live.map((s) => ({ id: s.id, name: s.name, color: seriesColor(s.color) })) };
    }
    const per = live.map((s) => ({ id: s.id, name: s.name, color: seriesColor(s.color), width: 1.5, values: byId.get(s.id)[mode === 'phase' ? 'phase' : 'spl'] }));
    if (mode === 'phase') {
      return { mode, kind: 'lines', x: pt.freqs, unit: '°', wrap: true,
        series: [...per, { id: 'sum', name: 'All speakers', color: c.ink, width: 2, values: pt.phase }, ...measured(pt.freqs, 'phase')] };
    }
    return { mode, kind: 'lines', x: pt.freqs, unit: 'dB', series: [
      ...per,
      { id: 'psum', name: 'If nothing cancelled', color: c.muted, width: 1.25, values: pt.power_sum },
      { id: 'sum', name: 'All speakers', color: c.ink, width: 2, values: pt.spl_smooth },
      ...measured(pt.freqs, 'spl'),
    ] };
  }

  // ----------------------------------------------------------------- legend
  function renderLegend() {
    clear(legend);
    if (!model) return;
    for (const s of model.series) {
      legend.append(h('button', { class: 'chip' + (hidden.has(s.id) ? ' off' : ''), title: 'Show or hide this trace',
        attrs: { 'aria-pressed': String(!hidden.has(s.id)) },
        onclick: () => { hidden.has(s.id) ? hidden.delete(s.id) : hidden.add(s.id); refresh(); } },
      h('span', { class: 'key' + (s.dash ? ' dashed' : ''), style: { borderColor: s.color } }), h('span', { text: s.name })));
    }
  }

  // ------------------------------------------------------------------- plot
  const plot = () => ({ l: 44, t: 10, r: W - 14, b: H - 26 });

  function axes(a, xTicks, yTicks, c) {
    ctx.lineWidth = 1;
    ctx.strokeStyle = cssVar('--line');
    ctx.fillStyle = c.muted;
    ctx.font = '10.5px system-ui, -apple-system, "Segoe UI", sans-serif';
    ctx.textBaseline = 'middle';
    ctx.textAlign = 'right';
    for (const t of yTicks) {
      const y = Math.round(t.pos) + 0.5;
      ctx.beginPath(); ctx.moveTo(a.l, y); ctx.lineTo(a.r, y); ctx.stroke();
      ctx.fillText(t.label, a.l - 7, y);
    }
    ctx.textAlign = 'center';
    ctx.textBaseline = 'top';
    for (const t of xTicks) {
      const x = Math.round(t.pos) + 0.5;
      ctx.beginPath(); ctx.moveTo(x, a.t); ctx.lineTo(x, a.b); ctx.stroke();
      ctx.fillText(t.label, x, a.b + 7);
    }
    ctx.strokeStyle = cssVar('--line-strong');
    ctx.beginPath(); ctx.moveTo(a.l, a.b + 0.5); ctx.lineTo(a.r, a.b + 0.5); ctx.stroke();
  }

  // Measured traces: dashed, in the measurement's own colour, resampled onto the chart's grid.
  function measured(grid, key) {
    return (state.project.measurements || []).filter((m) => key === 'spl' || m.phase).map((m) => {
      const v = resample(m, grid, { key });
      return { id: `meas_${m.id}`, name: `${m.name} (measured)`, color: seriesColor(m.color), width: 1.75, dash: [6, 4],
        values: key === 'spl' ? v.map((x) => (x == null ? null : x + (m.offset_db || 0))) : v };
    });
  }

  function draw() {
    raf = 0;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, W, H);
    if (!model || W < 60 || H < 60 || state.ui.table) return;
    const a = plot();
    const c = { ink: cssVar('--ink'), muted: cssVar('--muted'), surface: cssVar('--surface') };
    const shown = model.series.filter((s) => !hidden.has(s.id));

    if (model.kind === 'stems') {
      const ev = model.events;
      const tMax = clamp(Math.max(60, ...ev.map((e) => e.t)) * 1.04, 60, 250);
      const top = Math.ceil((Math.max(0, ...ev.map((e) => e.db)) + 1) / 6) * 6;
      const lo = top - 42;
      const X = (t) => a.l + ((t + 2) / (tMax + 2)) * (a.r - a.l);
      const Y = (d) => a.t + ((top - d) / (top - lo)) * (a.b - a.t);
      const xt = [];
      const stepT = tMax > 160 ? 40 : tMax > 80 ? 20 : 10;
      for (let t = 0; t <= tMax; t += stepT) xt.push({ pos: X(t), label: t === 0 ? 'first arrival' : `+${t} ms` });
      const yt = [];
      for (let d = top; d >= lo; d -= 6) yt.push({ pos: Y(d), label: `${d}` });
      axes(a, xt, yt, c);
      ctx.fillStyle = c.muted;
      ctx.textAlign = 'left';
      ctx.textBaseline = 'top';
      if (tMax > 34) { ctx.fillRect(Math.round(X(30)), a.t, 1, a.b - a.t); ctx.fillText('30 ms: echo risk beyond here', X(30) + 5, a.t + 2); }
      for (const e of ev) {
        if (e.t > tMax || e.db < lo) continue;
        const x = X(e.t), y = Y(e.db);
        ctx.strokeStyle = seriesColor(e.spk.color);
        ctx.globalAlpha = e.order === 0 ? 1 : 0.55;
        ctx.lineWidth = e.order === 0 ? 2 : 1;
        ctx.beginPath(); ctx.moveTo(x, a.b); ctx.lineTo(x, y); ctx.stroke();
        if (e.order === 0) {
          ctx.globalAlpha = 1;
          ctx.fillStyle = seriesColor(e.spk.color);
          ctx.strokeStyle = c.surface;
          ctx.lineWidth = 2;
          ctx.beginPath(); ctx.arc(x, y, 4.5, 0, Math.PI * 2); ctx.fill(); ctx.stroke();
        }
      }
      ctx.globalAlpha = 1;
      model.geom = { X, Y, tMax, lo };
      if (hoverX != null) {
        const near = nearestEvent(hoverX);
        if (near) { ctx.strokeStyle = c.ink; ctx.lineWidth = 1; ctx.beginPath(); ctx.arc(X(near.t), Y(Math.max(near.db, lo)), 7, 0, Math.PI * 2); ctx.stroke(); }
      }
      return;
    }

    const f = model.x;
    const f0 = f[0], f1 = f[f.length - 1];
    const X = (hz) => a.l + (Math.log(hz / f0) / Math.log(f1 / f0)) * (a.r - a.l);
    let hi, lo, stepY;
    if (model.wrap) { hi = 180; lo = -180; stepY = 90; } else {
      const all = shown.flatMap((s) => s.values.filter((v) => v != null));
      const lead = (shown.find((s) => s.id === 'sum' || s.id === 'avg') || shown[0])?.values.filter((v) => v != null).sort((p, q) => p - q) || [];
      hi = Math.ceil(((all.length ? Math.max(...all) : 90) + 1.5) / 6) * 6;
      const floor = lead.length ? lead[Math.floor(lead.length * 0.04)] - 9 : hi - 48;
      lo = hi - clamp(Math.ceil((hi - floor) / 6) * 6, 30, 66);
      stepY = hi - lo > 48 ? 12 : 6;
    }
    const Y = (v) => a.t + ((hi - v) / (hi - lo)) * (a.b - a.t);
    const yt = [];
    for (let v = hi; v >= lo; v -= stepY) yt.push({ pos: Y(v), label: model.wrap ? `${v}°` : `${v}` });
    axes(a, OCTAVES.filter((o) => o >= f0 * 0.99 && o <= f1 * 1.01).map((o) => ({ pos: X(o), label: fmtHz(o).replace(' Hz', '').replace(' kHz', 'k') })), yt, c);

    ctx.save();
    ctx.beginPath();
    ctx.rect(a.l, a.t, a.r - a.l, a.b - a.t);
    ctx.clip();
    ctx.lineJoin = 'round';
    ctx.lineCap = 'round';
    for (const s of shown) {
      ctx.strokeStyle = s.color;
      ctx.lineWidth = s.width;
      ctx.setLineDash(s.dash || []);
      ctx.beginPath();
      let pen = false, prev = null;
      for (let i = 0; i < f.length; i++) {
        const v = s.values[i];
        if (v == null) { pen = false; continue; }
        const x = X(f[i]), y = Y(v);
        if (!pen || (model.wrap && prev != null && Math.abs(v - prev) > 180)) { ctx.moveTo(x, y); pen = true; } else ctx.lineTo(x, y);
        prev = v;
      }
      ctx.stroke();
    }
    ctx.setLineDash([]);
    ctx.restore();
    model.geom = { X, Y, f0, f1 };

    if (hoverX != null) {
      const i = nearestIndex(hoverX);
      const x = Math.round(X(f[i])) + 0.5;
      ctx.strokeStyle = c.muted;
      ctx.lineWidth = 1;
      ctx.beginPath(); ctx.moveTo(x, a.t); ctx.lineTo(x, a.b); ctx.stroke();
      for (const s of shown) {
        const v = s.values[i];
        if (v == null || v < lo || v > hi) continue;
        ctx.fillStyle = s.color;
        ctx.strokeStyle = c.surface;
        ctx.lineWidth = 2;
        ctx.beginPath(); ctx.arc(x, Y(v), 4, 0, Math.PI * 2); ctx.fill(); ctx.stroke();
      }
    }
  }

  function nearestIndex(px) {
    const { f0, f1 } = model.geom;
    const a = plot();
    const hz = f0 * Math.exp((clamp(px, a.l, a.r) - a.l) / (a.r - a.l) * Math.log(f1 / f0));
    let best = 0, bd = Infinity;
    for (let i = 0; i < model.x.length; i++) {
      const d = Math.abs(Math.log(model.x[i] / hz));
      if (d < bd) { bd = d; best = i; }
    }
    return best;
  }

  function nearestEvent(px) {
    const { X, tMax } = model.geom;
    let best = null, bd = 14;
    for (const e of model.events) {
      if (e.t > tMax) continue;
      const d = Math.abs(X(e.t) - px) + (e.order === 0 ? -3 : 0);
      if (d < bd) { bd = d; best = e; }
    }
    return best;
  }

  function showTip(px, py) {
    if (!model?.geom || state.ui.table) { tip.hidden = true; return; }
    const rows = [];
    if (model.kind === 'stems') {
      const e = nearestEvent(px);
      if (!e) { tip.hidden = true; return; }
      rows.push(h('div', { class: 'v', text: e.t < 0.05 ? 'First arrival' : `+${fmt(e.t, 1)} ms` }),
        h('div', { class: 'l', text: `${fmt(e.db, 1)} dB at 1 kHz` }),
        h('div', { class: 'row' }, h('span', { class: 'key', style: { borderColor: seriesColor(e.spk.color) } }), h('span', { class: 'l', text: `${e.spk.name} · ${ORDER_NAMES[e.order] || 'Reflection'}` })));
    } else {
      const i = nearestIndex(px);
      rows.push(h('div', { class: 'v', text: fmtHz(model.x[i]) }), h('hr'));
      for (const s of [...model.series].reverse()) {
        if (hidden.has(s.id) || s.values[i] == null) continue;
        rows.push(h('div', { class: 'row' }, h('span', { class: 'key', style: { borderColor: s.color } }),
          h('span', { class: 'v', text: `${fmt(s.values[i], model.wrap ? 0 : 1)}${model.wrap ? '°' : ' dB'}` }), h('span', { class: 'l', text: s.name })));
      }
    }
    clear(tip).append(...rows);
    tip.hidden = false;
    tip.style.left = `${px}px`;
    tip.style.top = `${py}px`;
    tip.style.transform = `translate(${px > W - 270 ? 'calc(-100% - 14px)' : '14px'}, ${py > H / 2 ? 'calc(-100% - 10px)' : '10px'})`;
  }

  // ------------------------------------------------------------------ table
  const bandAverage = (f, values, centre) => {
    let sum = 0, n = 0;
    for (let i = 0; i < f.length; i++) {
      if (values[i] == null || f[i] < centre / Math.SQRT2 || f[i] >= centre * Math.SQRT2) continue;
      sum += 10 ** (values[i] / 10); n++;
    }
    return n ? 10 * Math.log10(sum / n) : null;
  };

  function renderTable() {
    clear(tableBox);
    if (!model) return;
    const table = h('table', { class: 'data' });
    if (model.kind === 'stems') {
      table.append(h('tr', null, ['Speaker', 'Path', 'After first arrival', 'Level at 1 kHz'].map((t) => h('th', { text: t }))));
      for (const e of model.events.slice(0, 60)) {
        table.append(h('tr', null, h('td', null, h('span', { class: 'key', style: { borderColor: seriesColor(e.spk.color) } }), e.spk.name),
          h('td', { text: ORDER_NAMES[e.order] || 'Reflection' }), h('td', { text: `${fmt(e.t, 1)} ms` }), h('td', { text: `${fmt(e.db, 1)} dB` })));
      }
    } else if (model.wrap) {
      table.append(h('tr', null, h('th', { text: 'Phase at band centre' }), BANDS.map((b) => h('th', { text: fmtHz(b) }))));
      for (const s of model.series) {
        table.append(h('tr', null, h('td', null, h('span', { class: 'key', style: { borderColor: s.color } }), s.name),
          BANDS.map((b) => { let k = 0; for (let i = 1; i < model.x.length; i++) if (Math.abs(Math.log(model.x[i] / b)) < Math.abs(Math.log(model.x[k] / b))) k = i; return h('td', { text: `${fmt(s.values[k], 0)}°` }); })));
      }
    } else {
      table.append(h('tr', null, h('th', { text: 'Octave band level, dB' }), BANDS.map((b) => h('th', { text: fmtHz(b) }))));
      for (const s of [...model.series].reverse()) {
        table.append(h('tr', null, h('td', null, h('span', { class: 'key', style: { borderColor: s.color } }), s.name),
          BANDS.map((b) => h('td', { text: fmt(bandAverage(model.x, s.values, b), 1) }))));
      }
    }
    tableBox.append(table);
  }

  // ---------------------------------------------------------------- refresh
  function refresh() {
    model = state.project ? buildModel() : null;
    renderLegend();
    tableBox.hidden = !state.ui.table || !model;
    canvas.style.visibility = state.ui.table ? 'hidden' : 'visible';
    empty.hidden = !!model;
    if (!model) {
      const none = !state.project?.speakers.some((s) => !s.mute);
      empty.textContent = none ? 'Add or unmute a speaker to see its response.' : 'Calculating the response…';
    }
    if (state.ui.table) renderTable();
    if (!raf) raf = requestAnimationFrame(draw);
  }

  canvas.addEventListener('pointermove', (e) => {
    const b = canvas.getBoundingClientRect();
    const px = e.clientX - b.left, py = e.clientY - b.top;
    const a = plot();
    hoverX = px >= a.l - 4 && px <= a.r + 4 ? px : null;
    if (hoverX == null) tip.hidden = true; else showTip(px, py);
    if (!raf) raf = requestAnimationFrame(draw);
  });
  canvas.addEventListener('pointerleave', () => { hoverX = null; tip.hidden = true; if (!raf) raf = requestAnimationFrame(draw); });

  new ResizeObserver(() => {
    const b = root.getBoundingClientRect();
    dpr = window.devicePixelRatio || 1;
    W = Math.floor(b.width);
    H = Math.floor(b.height);
    canvas.width = Math.max(1, Math.round(W * dpr));
    canvas.height = Math.max(1, Math.round(H * dpr));
    draw();
  }).observe(root);

  on('point sim theme', refresh);
  on('ui project', (d) => { if (!d?.interacting) renderToolbar(); refresh(); });
  return { refresh };
}
