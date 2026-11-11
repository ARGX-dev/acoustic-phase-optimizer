// Plan view: coverage heat map with draggable speakers, mics and a probe.
// The stage is on the left; distance from the stage runs left to right.

import { commit, editSpeaker, mutate, on, select, setProbe, speakerById, state } from './store.js';
import { clamp, clear, fmt, h, round } from './util.js';
import { cssVar, makeScale, seriesColor } from './theme.js';

export const MAPS = {
  direct: { label: 'Direct sound level', unit: 'dB', kind: 'sequential', bands: true,
    help: 'Direct sound plus early reflections: what the speakers themselves deliver to each seat.' },
  total: { label: 'Total level incl. reverberation', unit: 'dB', kind: 'sequential',
    help: 'What an SPL meter reads. In a live room this is always even, so it says little about coverage.' },
  clarity: { label: 'Clarity (early vs late)', unit: 'dB', kind: 'sequential', range: [-6, 12],
    help: 'Early energy relative to late reverberant energy. Above about 6 dB speech is easy to follow.' },
  interference: { label: 'Summation and cancellation', unit: 'dB', kind: 'diverging', range: [-6, 6],
    help: 'How the speakers add together (blue) or cancel each other (red), 1/3-octave smoothed.' },
  spread: { label: 'Arrival time spread', unit: 'ms', kind: 'sequential', range: [0, 60],
    help: 'Time between the first and last audible speaker at each seat. Beyond 30 ms it is heard as an echo.' },
};

export const mapKey = () => (MAPS[state.ui.map].bands && state.ui.band !== 'all' ? `direct_${state.ui.band}` : state.ui.map);

function levelRange(values) {
  const v = values.filter((x) => x != null).sort((a, b) => a - b);
  if (!v.length) return [66, 78];
  const hi = Math.ceil(v[Math.min(v.length - 1, Math.floor(v.length * 0.985))]);
  return [hi - 12, hi];
}

export function currentScale() {
  const def = MAPS[state.ui.map];
  const ref = state.sim?.maps?.[mapKey()] || shownSim()?.maps?.[mapKey()];
  const [lo, hi] = def.range || levelRange(ref || []);
  return makeScale(def.kind, lo, hi);
}

export const shownSim = () => (state.ui.compare === 'before' && state.simBaseline ? state.simBaseline : state.sim);

let heatCache = { sig: '', sim: null, canvas: null };
/**
 * The current map as an image: one pixel per grid point, columns = distance from the stage (room y),
 * rows = across the room (room x). Shared by the plan and the 3D view.
 */
export function heatImage(sim, key, scale) {
  const sig = [key, scale.lo, scale.hi, scale.css].join('|');
  if (heatCache.sig === sig && heatCache.sim === sim) return heatCache.canvas;
  const { nx, ny } = sim.grid;
  const vals = sim.maps[key];
  const off = document.createElement('canvas');
  off.width = ny;
  off.height = nx;
  const octx = off.getContext('2d');
  const img = octx.createImageData(ny, nx);
  const lut = scale.lut;
  for (let iy = 0; iy < ny; iy++) {
    for (let ix = 0; ix < nx; ix++) {
      const v = vals[iy * nx + ix];
      const o = (ix * ny + iy) * 4;
      if (v == null) continue;
      const k = scale.index(v) * 3;
      img.data[o] = lut[k]; img.data[o + 1] = lut[k + 1]; img.data[o + 2] = lut[k + 2]; img.data[o + 3] = 255;
    }
  }
  octx.putImageData(img, 0, 0);
  heatCache = { sig, sim, canvas: off };
  return off;
}

/** Map value and loudest speaker at a room position, or null outside the grid. */
export function valueAt(x, y) {
  const sim = shownSim();
  const vals = sim?.maps?.[mapKey()];
  if (!vals) return null;
  const g = sim.grid;
  const cx = (g.x1 - g.x0) / Math.max(g.nx - 1, 1), cy = (g.y1 - g.y0) / Math.max(g.ny - 1, 1);
  if (x < g.x0 - cx / 2 || x > g.x1 + cx / 2 || y < g.y0 - cy / 2 || y > g.y1 + cy / 2) return null;
  const ix = clamp(Math.round((x - g.x0) / cx), 0, g.nx - 1), iy = clamp(Math.round((y - g.y0) / cy), 0, g.ny - 1);
  const k = iy * g.nx + ix;
  return { value: vals[k], dominant: state.project.speakers[sim.maps.dominant?.[k]] };
}

export function createStage(root, foot) {
  const canvas = h('canvas', { class: 'plan', tabIndex: 0, attrs: { 'aria-label': 'Plan view of the venue. Drag speakers and microphones; click to place the probe.' } });
  const tip = h('div', { class: 'tip plan-tip', hidden: true });
  root.append(canvas, tip);
  const ctx = canvas.getContext('2d');
  let W = 0, H = 0, dpr = 1;
  let view = { scale: 1, ox: 0, oy: 0 };
  let drag = null;
  let hover = null;
  let raf = 0;

  const toScreen = (x, y) => [view.ox + y * view.scale, view.oy + x * view.scale];
  const toRoom = (sx, sy) => [(sy - view.oy) / view.scale, (sx - view.ox) / view.scale];

  function layout() {
    const r = state.project.room;
    const pad = { l: 46, r: 18, t: 16, b: 30 };
    const aw = Math.max(10, W - pad.l - pad.r), ah = Math.max(10, H - pad.t - pad.b);
    const scale = Math.min(aw / r.length, ah / r.width);
    view = { scale, ox: pad.l + (aw - r.length * scale) / 2, oy: pad.t + (ah - r.width * scale) / 2 };
  }

  function niceStep(pxPerMetre) {
    for (const s of [1, 2, 5, 10, 20, 50]) if (s * pxPerMetre >= 46) return s;
    return 100;
  }

  function text(str, x, y, { color, align = 'left', base = 'alphabetic', size = 11, weight = 500, halo = true } = {}) {
    ctx.font = `${weight} ${size}px system-ui, -apple-system, "Segoe UI", sans-serif`;
    ctx.textAlign = align;
    ctx.textBaseline = base;
    if (halo) {
      ctx.lineWidth = 3;
      ctx.lineJoin = 'round';
      ctx.strokeStyle = cssVar('--surface');
      ctx.globalAlpha = 0.85;
      ctx.strokeText(str, x, y);
      ctx.globalAlpha = 1;
    }
    ctx.fillStyle = color || cssVar('--ink');
    ctx.fillText(str, x, y);
  }

  function draw() {
    raf = 0;
    if (!state.project || W < 20 || H < 20) return;
    layout();
    const p = state.project, r = p.room, a = p.audience;
    const c = { surface: cssVar('--surface'), floor: cssVar('--room-floor'), ink: cssVar('--ink'), ink2: cssVar('--ink-2'),
      muted: cssVar('--muted'), line: cssVar('--line-strong'), accent: cssVar('--accent') };
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, W, H);

    const [rx, ry] = toScreen(0, 0);
    const rw = r.length * view.scale, rh = r.width * view.scale;
    ctx.fillStyle = c.floor;
    ctx.fillRect(rx, ry, rw, rh);

    // heat map over the audience area
    const sim = shownSim();
    const key = mapKey();
    const [ax, ay] = toScreen(a.x0, a.y0);
    const aw = (a.y1 - a.y0) * view.scale, ah = (a.x1 - a.x0) * view.scale;
    if (sim?.maps?.[key]) {
      const g = sim.grid;
      const img = heatImage(sim, key, currentScale());
      const cy = (g.y1 - g.y0) / Math.max(g.ny - 1, 1), cx = (g.x1 - g.x0) / Math.max(g.nx - 1, 1);
      const [hx, hy] = toScreen(g.x0 - cx / 2, g.y0 - cy / 2);
      ctx.save();
      ctx.beginPath();
      ctx.rect(ax, ay, aw, ah);
      ctx.clip();
      ctx.imageSmoothingEnabled = true;
      ctx.imageSmoothingQuality = 'high';
      ctx.drawImage(img, hx, hy, (g.y1 - g.y0 + cy) * view.scale, (g.x1 - g.x0 + cx) * view.scale);
      ctx.restore();
    }
    ctx.strokeStyle = c.muted;
    ctx.globalAlpha = 0.7;
    ctx.lineWidth = 1;
    ctx.strokeRect(Math.round(ax) + 0.5, Math.round(ay) + 0.5, Math.round(aw), Math.round(ah));
    ctx.globalAlpha = 1;

    // room outline, stage wall, rulers
    ctx.strokeStyle = c.line;
    ctx.lineWidth = 1.5;
    ctx.strokeRect(rx, ry, rw, rh);
    ctx.fillStyle = c.ink2;
    ctx.fillRect(rx - 4, ry, 4, rh);
    ctx.save();
    ctx.translate(rx - 12, ry + rh / 2);
    ctx.rotate(-Math.PI / 2);
    text('STAGE', 0, 0, { color: c.muted, align: 'center', base: 'middle', size: 10, weight: 650, halo: false });
    ctx.restore();
    const step = niceStep(view.scale);
    ctx.fillStyle = c.muted;
    for (let m = 0; m <= r.length + 1e-6; m += step) {
      const [sx] = toScreen(0, m);
      ctx.fillRect(Math.round(sx), ry + rh, 1, 4);
      text(m === 0 ? '0' : `${m} m`, sx, ry + rh + 8, { color: c.muted, align: 'center', base: 'top', size: 10.5, halo: false });
    }
    for (let m = step; m <= r.width + 1e-6; m += step) {
      const [, sy] = toScreen(m, 0);
      ctx.fillStyle = c.muted;
      ctx.fillRect(rx - 26, Math.round(sy), 4, 1);
      text(`${m}`, rx - 29, sy, { color: c.muted, align: 'right', base: 'middle', size: 10.5, halo: false });
    }

    // speakers
    const selId = state.selection?.kind === 'speaker' ? state.selection.id : null;
    for (const s of p.speakers) {
      const [sx, sy] = toScreen(s.x, s.y);
      const th = (s.yaw * Math.PI) / 180;
      const color = seriesColor(s.color);
      const isSel = s.id === selId;
      const omni = s.coverage_h >= 300;
      ctx.globalAlpha = s.mute ? 0.4 : 1;
      if (!omni) {
        const drop = s.z - a.ear_height;
        const reach = s.pitch < -1 && drop > 0 ? drop / Math.tan((-s.pitch * Math.PI) / 180) : 14;
        const len = clamp(reach, 4, 30) * view.scale;
        const half = (s.coverage_h * Math.PI) / 360;
        ctx.beginPath();
        ctx.moveTo(sx, sy);
        ctx.arc(sx, sy, len, th - half, th + half);
        ctx.closePath();
        if (isSel) { ctx.fillStyle = color; ctx.globalAlpha = 0.12; ctx.fill(); ctx.globalAlpha = s.mute ? 0.4 : 1; }
        ctx.strokeStyle = c.ink;
        ctx.globalAlpha *= isSel ? 0.7 : 0.28;
        ctx.lineWidth = 1;
        ctx.stroke();
        ctx.globalAlpha = s.mute ? 0.4 : 1;
      }
      if (isSel) {
        const [hx, hy] = [sx + Math.cos(th) * 38, sy + Math.sin(th) * 38];
        ctx.strokeStyle = c.accent;
        ctx.lineWidth = 1.5;
        ctx.beginPath(); ctx.moveTo(sx, sy); ctx.lineTo(hx, hy); ctx.stroke();
        ctx.beginPath(); ctx.arc(sx, sy, 12, 0, Math.PI * 2); ctx.stroke();
        ctx.fillStyle = c.surface;
        ctx.beginPath(); ctx.arc(hx, hy, 5, 0, Math.PI * 2); ctx.fill(); ctx.stroke();
      } else if (!omni) {
        ctx.strokeStyle = c.ink;
        ctx.lineWidth = 2;
        ctx.beginPath(); ctx.moveTo(sx, sy); ctx.lineTo(sx + Math.cos(th) * 14, sy + Math.sin(th) * 14); ctx.stroke();
      }
      ctx.fillStyle = color;
      ctx.strokeStyle = c.surface;
      ctx.lineWidth = 2;
      ctx.beginPath();
      if (s.role === 'sub') ctx.rect(sx - 6.5, sy - 6.5, 13, 13); else ctx.arc(sx, sy, 7, 0, Math.PI * 2);
      ctx.fill();
      ctx.stroke();
      text(s.name, sx + 11, sy - 9, { size: 11, weight: isSel ? 650 : 550 });
      ctx.globalAlpha = 1;
    }

    // microphones
    const micSel = state.selection?.kind === 'mic' ? state.selection.id : null;
    for (const m of p.mics) {
      const [sx, sy] = toScreen(m.x, m.y);
      ctx.fillStyle = c.surface;
      ctx.strokeStyle = m.id === micSel || state.ui.source === m.id ? c.accent : c.ink;
      ctx.lineWidth = 2;
      ctx.beginPath(); ctx.arc(sx, sy, 5.5, 0, Math.PI * 2); ctx.fill(); ctx.stroke();
      ctx.fillStyle = ctx.strokeStyle;
      ctx.beginPath(); ctx.arc(sx, sy, 1.8, 0, Math.PI * 2); ctx.fill();
      text(m.name, sx + 9, sy + 14, { size: 10.5, color: c.ink });
    }

    // probe
    if (state.probe) {
      const [sx, sy] = toScreen(state.probe.x, state.probe.y);
      const on = state.ui.source === 'probe';
      ctx.strokeStyle = on ? c.accent : c.muted;
      ctx.lineWidth = 2;
      ctx.beginPath();
      for (const [dx, dy] of [[1, 0], [-1, 0], [0, 1], [0, -1]]) { ctx.moveTo(sx + dx * 4, sy + dy * 4); ctx.lineTo(sx + dx * 11, sy + dy * 11); }
      ctx.stroke();
      ctx.beginPath(); ctx.arc(sx, sy, 1.5, 0, Math.PI * 2); ctx.fillStyle = ctx.strokeStyle; ctx.fill();
    }
  }

  const schedule = () => { if (!raf) raf = requestAnimationFrame(draw); };

  // ------------------------------------------------------------- interaction
  function hit(sx, sy) {
    const p = state.project;
    const sel = state.selection?.kind === 'speaker' ? speakerById(state.selection.id) : null;
    if (sel) {
      const [x, y] = toScreen(sel.x, sel.y);
      const th = (sel.yaw * Math.PI) / 180;
      if (Math.hypot(sx - (x + Math.cos(th) * 38), sy - (y + Math.sin(th) * 38)) <= 10) return { kind: 'rotate', obj: sel };
    }
    for (let i = p.speakers.length - 1; i >= 0; i--) {
      const [x, y] = toScreen(p.speakers[i].x, p.speakers[i].y);
      if (Math.hypot(sx - x, sy - y) <= 13) return { kind: 'speaker', obj: p.speakers[i] };
    }
    for (let i = p.mics.length - 1; i >= 0; i--) {
      const [x, y] = toScreen(p.mics[i].x, p.mics[i].y);
      if (Math.hypot(sx - x, sy - y) <= 11) return { kind: 'mic', obj: p.mics[i] };
    }
    return null;
  }

  function describe(sx, sy) {
    const target = hit(sx, sy);
    const [x, y] = toRoom(sx, sy);
    const r = state.project.room;
    const rows = [];
    if (target?.kind === 'speaker' || target?.kind === 'rotate') {
      const s = target.obj;
      rows.push(h('div', { class: 'row' }, h('span', { class: 'swatch', style: { background: seriesColor(s.color) } }), h('span', { class: 'v', style: { textAlign: 'left' }, text: s.name })));
      rows.push(h('div', { class: 'l', text: `Delay ${fmt(s.delay_ms, 2)} ms · Gain ${fmt(s.gain_db, 1, true)} dB${s.invert ? ' · Inverted' : ''}${s.mute ? ' · Muted' : ''}` }));
      rows.push(h('div', { class: 'l', text: target.kind === 'rotate' ? `Aim ${fmt(s.yaw, 0)}° - drag to turn` : `${fmt(s.y, 1)} m from stage, ${fmt(s.z, 1)} m high - drag to move` }));
      return { rows, readout: `${s.name}: delay ${fmt(s.delay_ms, 2)} ms, gain ${fmt(s.gain_db, 1, true)} dB` };
    }
    if (target?.kind === 'mic') {
      rows.push(h('div', { class: 'v', text: target.obj.name }), h('div', { class: 'l', text: 'Measurement position - drag to move' }));
      return { rows, readout: `${target.obj.name}: ${fmt(target.obj.y, 1)} m from stage` };
    }
    if (x < 0 || y < 0 || x > r.width || y > r.length) return null;
    const cell = valueAt(x, y);
    const def = MAPS[state.ui.map];
    const where = `${fmt(y, 1)} m from stage, ${fmt(x, 1)} m across`;
    if (!cell || cell.value == null) return { rows: [h('div', { class: 'l', text: where }), h('div', { class: 'l', text: 'Outside the audience area' })], readout: where };
    rows.push(h('div', { class: 'v', text: `${fmt(cell.value, def.unit === 'ms' ? 0 : 1, def.kind === 'diverging')} ${def.unit}` }));
    rows.push(h('div', { class: 'l', text: def.label + (state.ui.band !== 'all' && def.bands ? `, ${state.ui.band} Hz band` : '') }));
    rows.push(h('div', { class: 'l', text: where }));
    if (cell.dominant) rows.push(h('div', { class: 'row' }, h('span', { class: 'swatch', style: { background: seriesColor(cell.dominant.color) } }), h('span', { class: 'l', text: `Loudest here: ${cell.dominant.name}` })));
    return { rows, readout: `${where}: ${fmt(cell.value, 1, def.kind === 'diverging')} ${def.unit}` };
  }

  const local = (e) => { const b = canvas.getBoundingClientRect(); return [e.clientX - b.left, e.clientY - b.top]; };
  const snap = (v, lo, hi) => round(clamp(v, lo, hi), 1);

  canvas.addEventListener('pointerdown', (e) => {
    if (e.button !== 0 || !state.project) return;
    const [sx, sy] = local(e);
    const target = hit(sx, sy);
    const [x, y] = toRoom(sx, sy);
    const r = state.project.room;
    canvas.focus({ preventScroll: true });
    if (target?.kind === 'rotate') drag = { mode: 'rotate', id: target.obj.id, moved: false };
    else if (target) {
      select(target.kind, target.obj.id);
      drag = { mode: 'move', kind: target.kind, id: target.obj.id, dx: target.obj.x - x, dy: target.obj.y - y, moved: false, sx, sy };
    } else if (x >= 0 && y >= 0 && x <= r.width && y <= r.length) {
      setProbe(x, y, { final: false });
      drag = { mode: 'probe', moved: true };
    } else return;
    canvas.setPointerCapture(e.pointerId);
    tip.hidden = true;
    e.preventDefault();
  });

  canvas.addEventListener('pointermove', (e) => {
    if (!state.project) return;
    const [sx, sy] = local(e);
    const [x, y] = toRoom(sx, sy);
    const r = state.project.room;
    if (drag) {
      const live = { commit: false, interacting: true, source: 'stage' };
      if (drag.mode === 'rotate') {
        const s = speakerById(drag.id);
        if (!s) return;
        let yaw = (Math.atan2(x - s.x, y - s.y) * 180) / Math.PI;
        yaw = e.shiftKey ? Math.round(yaw / 15) * 15 : Math.round(yaw);
        if (yaw !== s.yaw) { drag.moved = true; editSpeaker(drag.id, { yaw }, live); }
      } else if (drag.mode === 'move') {
        if (!drag.moved && Math.hypot(sx - drag.sx, sy - drag.sy) < 3) return;
        drag.moved = true;
        const nx = snap(x + drag.dx, 0.1, r.width - 0.1), ny = snap(y + drag.dy, 0.1, r.length - 0.1);
        if (drag.kind === 'speaker') editSpeaker(drag.id, { x: nx, y: ny }, live);
        else mutate((p) => { const m = p.mics.find((q) => q.id === drag.id); if (m) { m.x = nx; m.y = ny; } }, live);
      } else setProbe(x, y, { final: false });
      return;
    }
    const info = describe(sx, sy);
    hover = info;
    canvas.style.cursor = hit(sx, sy) ? 'grab' : info ? 'crosshair' : 'default';
    if (info) {
      clear(tip).append(...info.rows);
      tip.hidden = false;
      const flipX = sx > W - 250, flipY = sy > H - 120;
      tip.style.left = `${sx}px`;
      tip.style.top = `${sy}px`;
      tip.style.transform = `translate(${flipX ? 'calc(-100% - 12px)' : '12px'}, ${flipY ? 'calc(-100% - 12px)' : '12px'})`;
    } else tip.hidden = true;
    renderFoot();
  });

  const endDrag = (e) => {
    if (!drag) return;
    const d = drag;
    drag = null;
    try { canvas.releasePointerCapture(e.pointerId); } catch (err) { /* already released */ }
    if (d.mode === 'probe') setProbe(state.probe.x, state.probe.y, { final: true });
    else if (d.moved) { commit(); mutate(() => {}, { source: 'stage' }); }
  };
  canvas.addEventListener('pointerup', endDrag);
  canvas.addEventListener('pointercancel', endDrag);
  canvas.addEventListener('pointerleave', () => { if (!drag) { tip.hidden = true; hover = null; renderFoot(); } });

  // ------------------------------------------------------------------ footer
  function renderFoot() {
    clear(foot);
    const def = MAPS[state.ui.map];
    const sim = shownSim();
    if (sim?.maps?.[mapKey()]) {
      const sc = currentScale();
      const d = def.unit === 'ms' ? 0 : 0;
      const lo = def.kind === 'diverging' ? `${fmt(sc.lo, d)} ${def.unit} cancelling` : `${fmt(sc.lo, d)} ${def.unit}`;
      const hi = def.kind === 'diverging' ? `${fmt(sc.hi, d, true)} ${def.unit} adding` : `${fmt(sc.hi, d)} ${def.unit}`;
      foot.append(h('div', { class: 'scale', title: def.help },
        h('span', { text: lo }), h('div', { class: 'bar', style: { background: sc.css } }), h('span', { text: hi })));
    } else if (state.simError) foot.append(h('span', { text: 'No result: ' + state.simError }));
    else foot.append(h('span', { text: state.project?.speakers.length ? 'Calculating…' : 'Add a speaker to see coverage.' }));
    if (sim?.preview) foot.append(h('span', { class: 'badge', text: 'Preview quality while editing' }));
    foot.append(h('span', { class: 'readout', text: hover?.readout || 'Drag speakers and mics · click the map to move the probe' }));
  }

  new ResizeObserver(() => {
    const b = root.getBoundingClientRect();
    dpr = window.devicePixelRatio || 1;
    W = Math.floor(b.width);
    H = Math.floor(b.height);
    canvas.width = Math.max(1, Math.round(W * dpr));
    canvas.height = Math.max(1, Math.round(H * dpr));
    draw();
  }).observe(root);

  on('project sim selection probe ui theme', () => { schedule(); renderFoot(); });
  on('simstate', () => root.classList.toggle('busy', state.simBusy && !state.sim?.preview));
  renderFoot();
  return { draw: schedule };
}
