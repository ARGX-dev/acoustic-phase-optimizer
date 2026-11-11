// Section view: a vertical slice of direct level through the room, the way a system designer checks tilt,
// splay and the vertical coverage of a hang. The slice runs through the probe, either away from the stage
// or across the room. Speakers can be dragged (distance and height) and their aim handle tilts them.

import { api } from './api.js';
import { commit, editSpeaker, mutate, on, select, setProbe, setUI, state } from './store.js';
import { clamp, clear, debounce, fmt, h } from './util.js';
import { cssVar, makeScale, seriesColor } from './theme.js';
import { levelRange, setFootOverride } from './stage.js';

const AIM_LEN_PX = 70;

export function createSection(root) {
  const host = h('div', { class: 'section-view', hidden: true });
  const canvas = h('canvas', { tabIndex: 0, attrs: { 'aria-label': 'Section view: direct level on a vertical slice through the probe. Drag speakers to move them, drag an aim handle to tilt.' } });
  const tip = h('div', { class: 'tip', hidden: true });
  const bar = h('div', { class: 'seg view-buttons section-buttons', attrs: { role: 'group', 'aria-label': 'Slice direction' } });
  host.append(canvas, tip, bar);
  root.append(host);
  const ctx = canvas.getContext('2d');
  let W = 0, H = 0, dpr = 1, visible = false;
  let data = null, pending = false, error = null, ctrl = null, image = null, imageKey = '';
  let view = { s: 1, ox: 0, oy: 0 };
  let drag = null, hover = null, raf = 0;

  const axis = () => (state.ui.sectionAxis === 'width' ? 'width' : 'length');
  const at = () => (axis() === 'length' ? state.probe.x : state.probe.y);
  const along = (o) => (axis() === 'length' ? o.y : o.x);
  const off = (o) => Math.abs((axis() === 'length' ? o.x : o.y) - at());
  const span = () => (axis() === 'length' ? state.project.room.length : state.project.room.width);
  const toS = (u, z) => [view.ox + u * view.s, view.oy + (state.project.room.height - z) * view.s];
  const toR = (sx, sy) => [(sx - view.ox) / view.s, state.project.room.height - (sy - view.oy) / view.s];

  function renderBar() {
    clear(bar);
    bar.append(...[['length', 'Away from stage'], ['width', 'Across the room']].map(([id, label]) =>
      h('button', { text: label, attrs: { 'aria-pressed': String(axis() === id) }, onclick: () => setUI({ sectionAxis: id }) })));
  }

  // ------------------------------------------------------------------ data
  const fetchNow = async () => {
    if (!visible || !state.project) return;
    ctrl?.abort();
    ctrl = new AbortController();
    pending = true;
    schedule();
    try {
      const band = state.ui.band || 'all';
      data = await api.section(state.project, axis(), at(), band, ctrl.signal);
      error = null;
    } catch (e) {
      if (e.name === 'AbortError') return;
      error = e.message;
    }
    pending = false;
    schedule();
    renderFoot();
  };
  const refetch = debounce(fetchNow, 180);

  function scale() {
    const vals = data?.values || [];
    // Colour the slice with the plan's range so a colour means the same level in both views.
    const plan = state.sim?.maps?.[state.ui.band && state.ui.band !== 'all' ? `direct_${state.ui.band}` : 'direct'];
    const [lo, hi] = levelRange(plan?.length ? plan : vals);
    return makeScale('sequential', lo, hi);
  }

  function heat(sc) {
    const key = [data, sc.lo, sc.hi, sc.css].map(String).join('|') + data?.timing_ms;
    if (image && imageKey === key) return image;
    const { nu, nz, values } = data;
    const c = document.createElement('canvas');
    c.width = nu; c.height = nz;
    const g = c.getContext('2d');
    const img = g.createImageData(nu, nz);
    for (let iz = 0; iz < nz; iz++) {
      for (let iu = 0; iu < nu; iu++) {
        const v = values[iz * nu + iu];
        if (v == null) continue;
        const o = ((nz - 1 - iz) * nu + iu) * 4, k = sc.index(v) * 3;
        img.data[o] = sc.lut[k]; img.data[o + 1] = sc.lut[k + 1]; img.data[o + 2] = sc.lut[k + 2]; img.data[o + 3] = 255;
      }
    }
    g.putImageData(img, 0, 0);
    image = c; imageKey = key;
    return c;
  }

  function valueAt(u, z) {
    if (!data) return null;
    const cu = (data.u1 - data.u0) / (data.nu - 1), cz = (data.z1 - data.z0) / (data.nz - 1);
    if (u < 0 || u > data.span || z < 0 || z > data.height) return null;
    const iu = clamp(Math.round((u - data.u0) / cu), 0, data.nu - 1), iz = clamp(Math.round((z - data.z0) / cz), 0, data.nz - 1);
    return data.values[iz * data.nu + iu];
  }

  // ------------------------------------------------------------------ geometry of speakers in the slice
  function aimDir(s) {
    const yw = (s.yaw * Math.PI) / 180, pt = (s.pitch * Math.PI) / 180;
    const horiz = axis() === 'length' ? Math.cos(yw) * Math.cos(pt) : Math.sin(yw) * Math.cos(pt);
    return { du: horiz, dz: Math.sin(pt), inPlane: Math.hypot(horiz, Math.sin(pt)) };
  }
  function handle(s) {
    const [x, y] = toS(along(s), s.z);
    const d = aimDir(s);
    const n = Math.hypot(d.du, d.dz) || 1;
    return [x + (d.du / n) * AIM_LEN_PX, y - (d.dz / n) * AIM_LEN_PX];
  }
  const nearSlice = (s) => off(s) <= Math.max(1.5, span() * 0.04);

  // ------------------------------------------------------------------ drawing
  function text(str, x, y, { color, align = 'left', base = 'alphabetic', size = 11, weight = 500 } = {}) {
    ctx.font = `${weight} ${size}px system-ui, -apple-system, "Segoe UI", sans-serif`;
    ctx.textAlign = align; ctx.textBaseline = base;
    ctx.lineWidth = 3; ctx.lineJoin = 'round'; ctx.strokeStyle = cssVar('--surface'); ctx.globalAlpha = 0.85;
    ctx.strokeText(str, x, y);
    ctx.globalAlpha = 1; ctx.fillStyle = color || cssVar('--ink');
    ctx.fillText(str, x, y);
  }
  const niceStep = (px) => [0.5, 1, 2, 5, 10, 20, 50].find((s) => s * px >= 40) || 100;

  function draw() {
    raf = 0;
    if (!visible || !state.project || W < 20 || H < 20) return;
    const p = state.project, r = p.room, a = p.audience;
    const L = span();
    const pad = { l: 46, r: 22, t: 22, b: 34 };
    const aw = W - pad.l - pad.r, ah = H - pad.t - pad.b;
    const s = Math.min(aw / L, ah / r.height);
    view = { s, ox: pad.l + (aw - L * s) / 2, oy: pad.t + (ah - r.height * s) / 2 };
    const c = { ink: cssVar('--ink'), ink2: cssVar('--ink-2'), muted: cssVar('--muted'), line: cssVar('--line-strong'), accent: cssVar('--accent'), floor: cssVar('--room-floor') };
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, W, H);
    const [x0, yTop] = toS(0, r.height), [x1, yFloor] = toS(L, 0);
    ctx.fillStyle = c.floor;
    ctx.fillRect(x0, yTop, x1 - x0, yFloor - yTop);
    if (data && data.axis === axis()) {
      const sc = scale();
      ctx.imageSmoothingEnabled = true;
      ctx.imageSmoothingQuality = 'high';
      const [hx0, hy0] = toS(data.u0, data.z1), [hx1, hy1] = toS(data.u1, data.z0);
      ctx.globalAlpha = pending ? 0.75 : 1;
      ctx.drawImage(heat(sc), hx0, hy0, hx1 - hx0, hy1 - hy0);
      ctx.globalAlpha = 1;
    }
    // grid & axes
    const step = niceStep(s);
    ctx.strokeStyle = c.line; ctx.globalAlpha = 0.35; ctx.lineWidth = 1; ctx.setLineDash([]);
    for (let u = step; u < L; u += step) { const [gx] = toS(u, 0); ctx.beginPath(); ctx.moveTo(gx + 0.5, yTop); ctx.lineTo(gx + 0.5, yFloor); ctx.stroke(); }
    const zstep = niceStep(s);
    for (let z = zstep; z < r.height; z += zstep) { const [, gy] = toS(0, z); ctx.beginPath(); ctx.moveTo(x0, gy + 0.5); ctx.lineTo(x1, gy + 0.5); ctx.stroke(); }
    ctx.globalAlpha = 1;
    for (let u = 0; u <= L + 1e-6; u += step) text(`${fmt(u, step < 1 ? 1 : 0)}${u === 0 ? '' : ' m'}`, toS(u, 0)[0], yFloor + 16, { color: c.muted, align: 'center', size: 10.5 });
    for (let z = 0; z <= r.height + 1e-6; z += zstep) text(`${fmt(z, zstep < 1 ? 1 : 0)} m`, x0 - 8, toS(0, z)[1], { color: c.muted, align: 'right', base: 'middle', size: 10.5 });
    // walls, floor, roof
    ctx.strokeStyle = c.ink2; ctx.lineWidth = 1.5;
    ctx.strokeRect(x0, yTop, x1 - x0, yFloor - yTop);
    ctx.lineWidth = 3; ctx.strokeStyle = c.ink; ctx.beginPath(); ctx.moveTo(x0, yFloor); ctx.lineTo(x1, yFloor); ctx.stroke();
    if (r.kind !== 'indoor') {
      ctx.lineWidth = 2; ctx.strokeStyle = c.muted; ctx.setLineDash([6, 6]);
      ctx.beginPath(); ctx.moveTo(x0, yTop); ctx.lineTo(x1, yTop); ctx.stroke(); ctx.setLineDash([]);
      text('Open sky', x1 - 6, yTop + 14, { color: c.muted, align: 'right', size: 10.5 });
    }
    if (axis() === 'length') text('STAGE', x0 + 6, yFloor - 8, { color: c.muted, size: 10, weight: 600 });
    // audience and ear height
    const [ua, ub] = axis() === 'length' ? [a.y0, a.y1] : [a.x0, a.x1];
    const [ax0, eary] = toS(ua, a.ear_height), [ax1] = toS(ub, a.ear_height);
    ctx.strokeStyle = c.accent; ctx.lineWidth = 4; ctx.beginPath(); ctx.moveTo(ax0, yFloor - 2); ctx.lineTo(ax1, yFloor - 2); ctx.stroke();
    ctx.lineWidth = 1.5; ctx.setLineDash([5, 4]); ctx.strokeStyle = c.ink; ctx.globalAlpha = 0.8;
    ctx.beginPath(); ctx.moveTo(ax0, eary); ctx.lineTo(ax1, eary); ctx.stroke(); ctx.setLineDash([]); ctx.globalAlpha = 1;
    text(`Ear height ${fmt(a.ear_height, 1)} m`, ax1 - 4, eary - 6, { color: c.ink2, align: 'right', size: 10.5 });
    // probe
    const pu = axis() === 'length' ? state.probe.y : state.probe.x;
    const [px] = toS(pu, 0);
    ctx.strokeStyle = c.ink; ctx.globalAlpha = 0.55; ctx.setLineDash([2, 4]); ctx.lineWidth = 1;
    ctx.beginPath(); ctx.moveTo(px + 0.5, yTop); ctx.lineTo(px + 0.5, yFloor); ctx.stroke(); ctx.setLineDash([]); ctx.globalAlpha = 1;
    ctx.beginPath(); ctx.arc(px, eary, 5, 0, Math.PI * 2); ctx.fillStyle = cssVar('--surface'); ctx.fill(); ctx.lineWidth = 2; ctx.strokeStyle = c.ink; ctx.stroke();
    // mics near the slice
    for (const m of p.mics) {
      if (!nearSlice(m)) continue;
      const [mx, my] = toS(along(m), m.z);
      ctx.beginPath(); ctx.arc(mx, my, 4.5, 0, Math.PI * 2); ctx.fillStyle = c.ink; ctx.fill();
      text(m.name, mx + 7, my - 6, { color: c.ink2, size: 10.5 });
    }
    // speakers: those near the slice in full, the rest as ghosts (their level still counts)
    const order = [...p.speakers].sort((q, w) => off(w) - off(q));
    for (const sp of order) {
      const near = nearSlice(sp);
      const sel = state.selection?.kind === 'speaker' && state.selection.id === sp.id;
      const col = seriesColor(sp.color);
      const [sx, sy] = toS(along(sp), sp.z);
      ctx.globalAlpha = sp.mute ? 0.3 : near || sel ? 1 : 0.35;
      const d = aimDir(sp);
      const faceOn = Math.abs(d.du) < 0.35 * Math.cos((sp.pitch * Math.PI) / 180); // aimed across the slice
      if ((near || sel) && (sp.coverage_v || 0) < 170 && !faceOn) {
        // vertical coverage wedge, drawn in the slice plane, clipped to the room
        ctx.save();
        ctx.beginPath(); ctx.rect(x0, yTop, x1 - x0, yFloor - yTop); ctx.clip();
        const n = Math.hypot(d.du, d.dz) || 1;
        const base = Math.atan2(d.dz / n, d.du / n), half = ((sp.coverage_v || 60) * Math.PI) / 360;
        const R = Math.min(view.s * L * 0.5, 260);
        ctx.fillStyle = col; ctx.globalAlpha *= 0.12;
        ctx.beginPath(); ctx.moveTo(sx, sy);
        ctx.arc(sx, sy, R, -(base + half), -(base - half));
        ctx.closePath(); ctx.fill();
        ctx.globalAlpha = sp.mute ? 0.3 : 1;
        ctx.strokeStyle = col; ctx.lineWidth = 1; ctx.setLineDash([4, 4]);
        for (const ang of [base + half, base - half]) {
          ctx.beginPath(); ctx.moveTo(sx, sy); ctx.lineTo(sx + Math.cos(ang) * R, sy - Math.sin(ang) * R); ctx.stroke();
        }
        ctx.setLineDash([]);
        // aim line and handle
        const [hx, hy] = handle(sp);
        ctx.lineWidth = 2; ctx.beginPath(); ctx.moveTo(sx, sy); ctx.lineTo(hx, hy); ctx.stroke();
        ctx.restore();
        ctx.beginPath(); ctx.arc(hx, hy, sel ? 6 : 5, 0, Math.PI * 2);
        ctx.fillStyle = cssVar('--surface'); ctx.fill(); ctx.strokeStyle = col; ctx.lineWidth = 2; ctx.stroke();
      }
      // cabinet: a stack of boxes for a line-array hang, otherwise one box
      const boxes = sp.array_boxes > 1 ? Math.min(sp.array_boxes, 24) : 1;
      const bh = boxes > 1 ? Math.max(3, Math.min(9, view.s * 0.35)) : 12, bw = boxes > 1 ? Math.max(10, view.s * 0.9) : 12;
      ctx.save();
      ctx.translate(sx, sy);
      const tilt = faceOn ? 0 : Math.atan2(d.dz, Math.abs(d.du) || 1e-6) * (d.du < 0 ? -1 : 1);
      const flip = d.du < 0 ? -1 : 1;
      let ang = -tilt * flip;
      let cx = 0, cy = -(boxes - 1) * bh / 2;
      for (let i = 0; i < boxes; i++) {
        ctx.save(); ctx.translate(cx, cy); ctx.rotate(ang * flip);
        ctx.fillStyle = col; ctx.strokeStyle = sel ? c.ink : cssVar('--surface'); ctx.lineWidth = sel ? 2 : 1;
        ctx.beginPath(); ctx.roundRect(-bw / 2, -bh / 2, bw, bh, 2); ctx.fill(); ctx.stroke();
        ctx.restore();
        cy += bh * Math.cos(ang); cx += -bh * Math.sin(ang) * flip;
        ang += ((sp.array_splay || 0) * Math.PI) / 180;
      }
      ctx.restore();
      if (near || sel) text(sp.name, sx + bw / 2 + 6, sy - (boxes - 1) * bh / 2 - 6, { color: c.ink, size: 11, weight: 600 });
      ctx.globalAlpha = 1;
    }
    if (pending && !data) text('Calculating the section…', (x0 + x1) / 2, (yTop + yFloor) / 2, { color: c.ink2, align: 'center', size: 13 });
    if (error) text(`No section: ${error}`, (x0 + x1) / 2, (yTop + yFloor) / 2, { color: cssVar('--critical'), align: 'center', size: 13 });
    text(`Slice ${axis() === 'length' ? 'away from the stage' : 'across the room'} at ${fmt(at(), 1)} m ${axis() === 'length' ? 'across' : 'from the stage'} · through the probe`,
      x0, yTop - 8, { color: c.ink2, size: 11 });
  }
  function schedule() { if (!raf) raf = requestAnimationFrame(draw); }

  // ------------------------------------------------------------------ footer (the stage owns the element)
  function renderFoot() {
    if (!visible) return;
    setFootOverride(visible ? footContent : null);
  }
  function footContent() {
    const items = [];
    if (data?.values?.length) {
      const sc = scale();
      items.push(h('div', { class: 'scale', title: 'Direct sound and early reflections, as in the plan’s Direct level map.' },
        h('span', { text: `${fmt(sc.lo, 0)} dB` }), h('div', { class: 'bar', style: { background: sc.css } }), h('span', { text: `${fmt(sc.hi, 0)} dB` })));
    }
    if (pending) items.push(h('span', { class: 'badge', text: 'Updating' }));
    items.push(h('span', { class: 'readout', text: hover || 'Direct level on a vertical slice through the probe · drag a speaker to move it, drag its aim handle to tilt · click to move the probe' }));
    return items;
  }

  // ------------------------------------------------------------------ pointer
  function local(e) { const b = canvas.getBoundingClientRect(); return [e.clientX - b.left, e.clientY - b.top]; }
  function hit(sx, sy) {
    const sp = [...state.project.speakers].sort((q, w) => off(q) - off(w));
    for (const s of sp) {
      if (!nearSlice(s) && !(state.selection?.id === s.id)) continue;
      const [hx, hy] = handle(s);
      const d = aimDir(s);
      if (Math.abs(d.du) < 0.35 * Math.cos((s.pitch * Math.PI) / 180) || (s.coverage_v || 0) >= 170) continue;
      if (Math.hypot(sx - hx, sy - hy) < 10) return { kind: 'aim', id: s.id };
    }
    for (const s of sp) {
      const [x, y] = toS(along(s), s.z);
      const tall = s.array_boxes > 1 ? Math.min(s.array_boxes, 24) * Math.max(3, Math.min(9, view.s * 0.35)) / 2 + 4 : 10;
      if (Math.abs(sx - x) < 12 && Math.abs(sy - y) < tall) return { kind: 'speaker', id: s.id };
    }
    return null;
  }
  const live = { commit: false, interacting: true, source: 'section' };
  canvas.addEventListener('pointerdown', (e) => {
    if (e.button !== 0 || !state.project) return;
    const [sx, sy] = local(e);
    const t = hit(sx, sy);
    if (!t) {
      const [u] = toR(sx, sy);
      if (u >= 0 && u <= span()) {
        if (axis() === 'length') setProbe(state.probe.x, u); else setProbe(u, state.probe.y);
      }
      return;
    }
    select('speaker', t.id);
    drag = { ...t, moved: false };
    canvas.setPointerCapture(e.pointerId);
  });
  canvas.addEventListener('pointermove', (e) => {
    const [sx, sy] = local(e);
    if (drag) {
      const r = state.project.room;
      const s = state.project.speakers.find((q) => q.id === drag.id);
      if (!s) return;
      drag.moved = true;
      if (drag.kind === 'speaker') {
        const [u, z] = toR(sx, sy);
        const uu = Math.round(clamp(u, 0.1, span() - 0.1) * 10) / 10, zz = Math.round(clamp(z, 0.1, r.height - 0.1) * 10) / 10;
        editSpeaker(s.id, axis() === 'length' ? { y: uu, z: zz } : { x: uu, z: zz }, live);
      } else {
        const [x, y] = toS(along(s), s.z);
        const d = aimDir(s);
        const du = sx - x, dz = y - sy;
        const pitch = Math.round(clamp((Math.atan2(dz, Math.abs(du)) * 180) / Math.PI, -90, 90));
        if (pitch !== s.pitch && (Math.sign(du) === Math.sign(d.du) || Math.abs(d.du) < 1e-3 || Math.abs(du) < 4)) editSpeaker(s.id, { pitch }, live);
      }
      schedule();
      return;
    }
    const t = hit(sx, sy);
    canvas.style.cursor = t?.kind === 'aim' ? 'crosshair' : t ? 'grab' : 'default';
    const [u, z] = toR(sx, sy);
    const v = valueAt(u, z);
    if (t) {
      const s = state.project.speakers.find((q) => q.id === t.id);
      hover = t.kind === 'aim' ? `${s.name}: tilt ${fmt(s.pitch, 0)}° · drag to tilt` : `${s.name}: ${fmt(along(s), 1)} m, ${fmt(s.z, 1)} m high · drag to move`;
    } else hover = v == null ? null : `${fmt(v, 1)} dB at ${fmt(u, 1)} m ${axis() === 'length' ? 'from the stage' : 'across'}, ${fmt(z, 1)} m high`;
    tip.hidden = true;
    renderFoot();
  });
  const end = (e) => {
    if (!drag) return;
    const moved = drag.moved;
    drag = null;
    try { canvas.releasePointerCapture(e.pointerId); } catch (err) { /* released */ }
    if (moved) { commit(); mutate(() => {}, { source: 'section' }); }
  };
  canvas.addEventListener('pointerup', end);
  canvas.addEventListener('pointercancel', end);
  canvas.addEventListener('pointerleave', () => { if (!drag) { hover = null; renderFoot(); } });

  new ResizeObserver(() => {
    const b = host.getBoundingClientRect();
    dpr = window.devicePixelRatio || 1;
    W = Math.floor(b.width); H = Math.floor(b.height);
    canvas.width = Math.max(1, Math.round(W * dpr)); canvas.height = Math.max(1, Math.round(H * dpr));
    schedule();
  }).observe(host);

  let lastAt = null;
  on('project', () => { if (visible) { schedule(); refetch(); } });
  on('probe', () => {
    if (!visible) return;
    schedule();
    if (at() !== lastAt) { lastAt = at(); refetch(); }
  });
  on('ui', (patch) => {
    if (!visible) return;
    if (!patch || 'sectionAxis' in patch || 'band' in patch) { renderBar(); data = null; fetchNow(); }
    schedule();
  });
  on('selection sim theme', () => { if (visible) { image = null; schedule(); renderFoot(); } });

  // For the UI tests: where a speaker's aim handle is drawn, in canvas pixels.
  window.__apoSectionHandle = (name) => {
    const s = state.project.speakers.find((q) => q.name === name);
    return s && visible ? handle(s) : null;
  };

  return {
    show() {
      if (visible) return;
      visible = true; host.hidden = false; lastAt = at();
      renderBar(); renderFoot(); fetchNow();
    },
    hide() {
      visible = false; host.hidden = true;
      ctrl?.abort();
      setFootOverride(null);
    },
  };
}
