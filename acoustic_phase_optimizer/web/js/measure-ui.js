// Measurements that carry an impulse response (an imported WAV or a live sweep): add them to the venue, look
// at them (energy-time curve, energy decay per octave, the ISO 3382 table beside the prediction) and compare
// the arrival times of several takes to set delays.

import { state, mutate, partnerOf } from './store.js';
import { clear, download, fmt, fmtHz, h, toast, uid } from './util.js';
import { openDialog } from './dialog.js';
import { cssVar } from './theme.js';
import { matchOffset } from './measure.js';

const P = () => state.project;
export const BAND_KEYS = ['edt', 't20', 't30', 'c50', 'c80', 'd50', 'ts'];
const LABEL = { edt: 'EDT', t20: 'T20', t30: 'T30', c50: 'C50', c80: 'C80', d50: 'D50', ts: 'Ts' };
const UNIT = { edt: 's', t20: 's', t30: 's', c50: 'dB', c80: 'dB', d50: '', ts: 'ms' };
const ND = { edt: 2, t20: 2, t30: 2, c50: 1, c80: 1, d50: 2, ts: 0 };
const TIP = {
  edt: 'Early decay time: the first 10 dB of the decay, times 6. Closest to how reverberant a room sounds',
  t20: 'Reverberation time from the decay between −5 and −25 dB',
  t30: 'Reverberation time from the decay between −5 and −35 dB (the usual RT60)',
  c50: 'Clarity for speech: sound in the first 50 ms against the rest',
  c80: 'Clarity for music: sound in the first 80 ms against the rest',
  d50: 'Definition: share of the sound that arrives in the first 50 ms',
  ts: 'Centre time: the balance point of the sound energy in time',
};

function predictedCurve() {
  if (state.ui.source === 'average') return state.sim?.average ? { f: state.sim.freqs, v: state.sim.average.spl } : null;
  return state.point ? { f: state.point.freqs, v: state.point.spl_smooth } : null;
}

function freeColor() {
  const used = new Set([...(P().measurements || []).map((x) => x.color), ...P().speakers.map((x) => x.color)]);
  return [5, 6, 7, 3, 2, 1, 0, 4].find((c) => !used.has(c)) ?? 5;
}

async function postBytes(url, body) {
  let res;
  try {
    res = await fetch(url, { method: 'POST', body, headers: { 'Content-Type': 'application/octet-stream' } });
  } catch (e) {
    throw new Error('The engine is not responding. Is the app still running?');
  }
  const out = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error(typeof out.detail === 'string' ? out.detail : res.statusText);
  return out;
}

/** The measurement record for an analysed impulse response (from /api/measure/ir or /capture). */
export function measurementFrom(name, res, extra = {}) {
  const a = res.analysis;
  const m = {
    id: `m${uid()}`, name: String(name).slice(0, 80) || 'Measurement', color: freeColor(), offset_db: 0,
    freqs: a.response.freqs, spl: a.response.spl, phase: a.response.phase,
    source: 'ir', ir_id: res.ir_id, start_ms: res.start_ms ?? 0, arrival_ms: a.arrival_ms, polarity: a.polarity,
    bands: a.bands.map((b) => Object.fromEntries([['f', b.f], ...BAND_KEYS.map((k) => [k, b[k]])])),
    taken: new Date().toISOString().slice(0, 19), ...extra,
  };
  alignLevel(m);
  return m;
}

/** Line a measurement's level up with the prediction (200 Hz–2 kHz). A file export in dB SPL gets an offset; a
 *  level that is digital (an impulse response or a live take, in dBFS, often 100 dB away) has the match put into
 *  the curve itself, so the offset control stays free for fine adjustment. Changes `m`; false with no prediction. */
export function alignLevel(m) {
  const pred = predictedCurve();
  if (!pred) return false;
  const k = matchOffset(m, pred.f, pred.v);
  if ((m.source || 'file') === 'file' && Math.abs(k) <= 80) { m.offset_db = k; return true; }
  m.spl = m.spl.map((v) => Math.round((v + k) * 100) / 100);
  m.offset_db = 0;
  return true;
}

export function addMeasurementRecord(m) {
  if ((P().measurements || []).length >= 8) throw new Error('Up to 8 measurements per venue. Delete one first.');
  mutate((p) => { p.measurements = [...(p.measurements || []), m]; }, { structural: true });
  return m;
}

const summary = (a) => {
  const s = a.single;
  const rt = s.t30 != null ? `T30 ${fmt(s.t30, 2)} s` : s.t20 != null ? `T20 ${fmt(s.t20, 2)} s` : s.edt != null ? `EDT ${fmt(s.edt, 2)} s` : 'no decay time (too noisy)';
  return `${rt}, C80 ${fmt(s.c80, 1)} dB at 500 Hz–1 kHz`;
};

/** Import an impulse response WAV (REW, Smaart, ARTA, a convolution reverb...). Returns the toast text. */
export async function addImpulseResponse(name, bytes) {
  if ((P().measurements || []).length >= 8) throw new Error('Up to 8 measurements per venue. Delete one first.');
  const res = await postBytes('/api/measure/ir', bytes);
  const m = addMeasurementRecord(measurementFrom(name.replace(/^.*[\\/]/, '').replace(/\.[^.]+$/, ''), res));
  return `Imported "${m.name}" as an impulse response: ${summary(res.analysis)}.`;
}

/** Measured decay time per octave band, averaged over the measurements that have one (T30, else T20). */
export function measuredRt(project = P()) {
  const list = (project.measurements || []).filter((m) => m.bands?.length);
  if (!list.length) return null;
  const freqs = list[0].bands.map((b) => b.f);
  return freqs.map((f) => {
    const v = list.map((m) => m.bands.find((b) => b.f === f)).map((b) => b && (b.t30 ?? b.t20)).filter((x) => x != null);
    return v.length ? v.reduce((a, b) => a + b, 0) / v.length : null;
  });
}

// ------------------------------------------------------------------------------ viewer

/** Draw the energy-time curve and one band's energy decay (dB) on a canvas. */
function drawIr(cv, a, band, span) {
  const ctx = cv.getContext('2d');
  const W = cv.width, H = cv.height, L = 44, R = 10, T = 10, B = 26;
  ctx.fillStyle = cssVar('--page');
  ctx.fillRect(0, 0, W, H);
  const c = a.curves;
  const tEnd = c.t0_ms + c.etc.length * c.dt_ms;
  const t0 = span === 'early' ? Math.max(c.t0_ms, a.arrival_ms - 5) : c.t0_ms;
  const t1 = span === 'early' ? Math.min(tEnd, t0 + 100) : tEnd;
  const X = (t) => L + ((t - t0) / (t1 - t0)) * (W - L - R);
  const Y = (db) => T + (Math.min(0, Math.max(-90, db)) / -90) * (H - T - B);
  ctx.strokeStyle = cssVar('--line');
  ctx.fillStyle = cssVar('--ink-2');
  ctx.font = '11px system-ui, sans-serif';
  ctx.lineWidth = 1;
  ctx.textAlign = 'right';
  ctx.textBaseline = 'middle';
  for (let db = 0; db >= -90; db -= 15) {
    ctx.beginPath(); ctx.moveTo(L, Y(db)); ctx.lineTo(W - R, Y(db)); ctx.stroke();
    ctx.fillText(`${db}`, L - 6, Y(db));
  }
  const step = [1, 2, 5, 10, 20, 50, 100, 200, 500, 1000].find((s) => (t1 - t0) / s <= 8) || 2000;
  ctx.textAlign = 'center';
  ctx.textBaseline = 'top';
  for (let t = Math.ceil(t0 / step) * step; t <= t1; t += step) {
    ctx.beginPath(); ctx.moveTo(X(t), T); ctx.lineTo(X(t), H - B); ctx.stroke();
    ctx.fillText(`${t} ms`, X(t), H - B + 6);
  }
  ctx.save();
  ctx.beginPath(); ctx.rect(L, T, W - L - R, H - T - B); ctx.clip();
  // ETC: filled, quiet
  ctx.fillStyle = cssVar('--muted');
  ctx.globalAlpha = 0.35;
  ctx.beginPath(); ctx.moveTo(X(c.t0_ms), Y(-90));
  c.etc.forEach((v, i) => { if (v != null) ctx.lineTo(X(c.t0_ms + i * c.dt_ms), Y(v)); });
  ctx.lineTo(X(tEnd), Y(-90)); ctx.closePath(); ctx.fill();
  ctx.globalAlpha = 1;
  // the band's energy decay
  const edc = c.edc[String(band)];
  if (edc) {
    ctx.strokeStyle = cssVar('--accent');
    ctx.lineWidth = 2;
    ctx.beginPath();
    let pen = false;
    edc.forEach((v, i) => {
      if (v == null) { pen = false; return; }
      const x = X(c.edc_t0_ms + i * c.edc_dt_ms), y = Y(v);
      if (pen) ctx.lineTo(x, y); else ctx.moveTo(x, y);
      pen = true;
    });
    ctx.stroke();
  }
  // the direct sound
  ctx.strokeStyle = cssVar('--ink');
  ctx.setLineDash([4, 3]);
  ctx.beginPath(); ctx.moveTo(X(a.arrival_ms), T); ctx.lineTo(X(a.arrival_ms), H - B); ctx.stroke();
  ctx.setLineDash([]);
  ctx.restore();
  ctx.fillStyle = cssVar('--ink');
  ctx.textAlign = 'left';
  ctx.fillText(`direct sound ${fmt(a.arrival_ms, 2)} ms`, Math.min(X(a.arrival_ms) + 6, W - 150), T + 2);
}

/** The ISO 3382 table: one row per value, a column per octave, the prediction under the decay times. */
function bandTable(a) {
  const pred = state.sim?.room && P().room.kind !== 'outdoor' ? (state.sim.room.rt60 || state.sim.room.rt60_eyring) : null;
  const bands = a.bands;
  const head = h('tr', {}, h('th', { text: '' }), ...bands.map((b) => h('th', { text: fmtHz(b.f).replace(' kHz', 'k').replace(' Hz', '') })),
    h('th', { text: '500–1k', title: 'The single-number value: average of 500 Hz and 1 kHz' }));
  const rows = BAND_KEYS.map((k) => h('tr', {}, h('th', { text: `${LABEL[k]}${UNIT[k] ? ` (${UNIT[k]})` : ''}`, title: TIP[k] }),
    ...bands.map((b) => h('td', { text: b[k] == null ? '—' : fmt(k === 'd50' ? b[k] : b[k], ND[k]), title: b.note || null, class: b[k] == null ? 'na' : null })),
    h('td', { class: 'single', text: a.single[k] == null ? '—' : fmt(a.single[k], ND[k]) })));
  if (pred) {
    const pb = state.sim.room.bands || [];
    rows.splice(3, 0, h('tr', { class: 'pred' }, h('th', { text: 'Predicted', title: 'The prediction\'s reverb time for this venue, per octave' }),
      ...bands.map((b) => { const i = pb.indexOf(b.f); return h('td', { text: i >= 0 && pred[i] != null ? fmt(pred[i], 2) : '—' }); }),
      h('td', { class: 'single', text: fmt(state.sim.room.rt60_mid, 2) })));
  }
  return h('table', { class: 'ir-table' }, h('thead', {}, head), h('tbody', {}, rows));
}

/** The viewer's content for an analysis: chart with band and span choice, table, notes. */
export function irView(a, { warnings = [] } = {}) {
  const cv = h('canvas', { class: 'ir-chart', width: 720, height: 260, attrs: { role: 'img', 'aria-label': `Energy over time after the sound, with the decay at the chosen octave. ${summary(a)}.` } });
  let band = 1000, span = 'all';
  const redraw = () => drawIr(cv, a, band, span);
  const seg = (items, get, set) => h('div', { class: 'seg' }, items.map(([v, label]) => {
    const b = h('button', { class: get() === v ? 'on' : '', text: label, attrs: { 'aria-pressed': String(get() === v) },
      onclick: () => { set(v); [...b.parentNode.children].forEach((x) => { const on = x === b; x.classList.toggle('on', on); x.setAttribute('aria-pressed', String(on)); }); redraw(); } });
    return b;
  }));
  const usable = a.bands.filter((b) => a.curves.edc[String(b.f)]);
  if (!usable.some((b) => b.f === band)) band = usable[0]?.f ?? band;
  const notes = [...warnings];
  if (a.inr_db < 45) notes.push(`${fmt(a.inr_db, 0)} dB of decay above the noise: T30 needs 45 dB, T20 35.`);
  if (a.polarity < 0) notes.push('The direct sound arrives inverted (negative polarity).');
  const el = h('div', { class: 'ir-view' },
    h('div', { class: 'ir-tools' },
      seg(usable.map((b) => [b.f, fmtHz(b.f).replace(' kHz', 'k').replace(' Hz', '')]), () => band, (v) => { band = v; }),
      seg([['early', 'First 100 ms'], ['all', 'Whole decay']], () => span, (v) => { span = v; })),
    cv,
    h('p', { class: 'hint', text: `Grey: energy over time (dB). Line: energy decay of the chosen octave, used for EDT and T20/T30. ${a.loopback ? 'Times count from when the sound left the interface (loopback).' : 'Times include the fixed latency of the audio chain.'}` }),
    bandTable(a),
    notes.length ? h('ul', { class: 'ir-notes' }, notes.map((n) => h('li', { text: n }))) : null);
  requestAnimationFrame(redraw);
  return el;
}

/** Open the viewer for a measurement that has an impulse response. */
export async function showIR(m) {
  let a;
  try {
    const q = new URLSearchParams({ start_ms: String(m.start_ms || 0) });
    const res = await fetch(`/api/irs/${m.ir_id}/analysis?${q}`);
    a = await res.json();
    if (!res.ok) throw new Error(a.detail || res.statusText);
  } catch (e) {
    toast(`${m.name}: ${e.message}`, { error: true });
    return;
  }
  a.loopback = !!m.loopback;
  openDialog({ title: `${m.name}: impulse response`, body: irView(a), wide: true, actions: [
    { label: 'Download WAV', run: async () => {
      const res = await fetch(`/api/irs/${m.ir_id}.wav`);
      download(`${m.name.replace(/[<>:"/\\|?*\u0000-\u001f]+/g, ' ').trim() || 'impulse response'}.wav`, await res.blob(), 'audio/wav');
      return false;
    } },
    { label: 'Close', kind: 'primary' },
  ] });
}

// ------------------------------------------------------------------------------ delay finder

const ROLE_PRECEDENCE = new Set(['fill', 'front_fill', 'delay']);

/** Suggested delay changes from measured arrivals: each take's speaker is moved to arrive with the
 *  reference take (plus the precedence offset for fills). Pure, for the dialog and the tests. */
export function suggestDelays(project, takes, refId) {
  const ref = takes.find((t) => t.id === refId);
  if (!ref) return [];
  const prec = project.settings?.precedence_ms || 0;
  return takes.filter((t) => t.id !== refId).map((t) => {
    const s = project.speakers.find((x) => x.id === t.speaker_ids?.[0]);
    const row = { id: t.id, name: t.name, speaker: s?.name ?? null, speakerId: s?.id ?? null, rel: t.arrival_ms - ref.arrival_ms, now: s?.delay_ms ?? null, next: null, note: null };
    if (!s) { row.note = 'Say which speaker played in this take to get a suggestion.'; return row; }
    if (t.speaker_ids.length > 1) { row.note = 'More than one speaker played in this take.'; return row; }
    if (s.lock_delay) { row.note = 'Its delay is locked.'; return row; }
    const target = ROLE_PRECEDENCE.has(s.role) ? prec : 0;
    const want = s.delay_ms - row.rel + target;
    row.next = Math.round(Math.min(500, Math.max(0, want)) * 100) / 100;
    if (want < 0) row.note = `Arrives ${fmt(-want, 1)} ms too late even with no delay: delay the reference instead, or check the take.`;
    return row;
  });
}

export function arrivalsDialog() {
  const takes = (P().measurements || []).filter((m) => m.arrival_ms != null);
  if (takes.length < 2) { toast('Measure at least two speakers from the same mic position to compare arrivals.', { error: true }); return; }
  const chains = new Set(takes.map((t) => t.chain || '?'));
  const refDefault = takes.find((t) => P().speakers.find((s) => s.id === t.speaker_ids?.[0])?.role === 'main') || takes[0];
  let refId = refDefault.id;
  const body = h('div', { class: 'arrivals' });
  let rows = [];
  const render = () => {
    rows = suggestDelays(P(), takes, refId);
    clear(body).append(
      h('label', { class: 'field' }, h('span', { text: 'Reference take (normally a main)' }),
        h('select', { onchange: (e) => { refId = e.target.value; render(); } }, takes.map((t) => h('option', { value: t.id, text: t.name, selected: t.id === refId })))),
      h('table', { class: 'ir-table' },
        h('thead', {}, h('tr', {}, ['Take', 'Speaker', 'Arrives (vs reference)', 'Delay now', 'Suggested'].map((x) => h('th', { text: x })))),
        h('tbody', {}, rows.map((r) => h('tr', {},
          h('td', { text: r.name }), h('td', { text: r.speaker || '—' }),
          h('td', { text: `${fmt(r.rel, 2, true)} ms` }), h('td', { text: r.now == null ? '—' : `${fmt(r.now, 2)} ms` }),
          h('td', { text: r.next == null ? '—' : `${fmt(r.next, 2)} ms`, title: r.note })))),
      ),
      ...rows.filter((r) => r.note).map((r) => h('p', { class: 'hint', text: `${r.name}: ${r.note}` })),
      h('p', { class: 'hint', text: (chains.size > 1 || takes.some((t) => !t.loopback)
        ? 'Arrival times without a loopback include the audio chain\'s latency. That is the same for takes made with the same interface and settings, so compare those; takes from different set-ups can be a few ms apart. '
        : '') + 'Each speaker is moved to arrive with the reference, plus the precedence offset for fills. Mirror-image partners get the same change.' }));
  };
  render();
  openDialog({ title: 'Compare arrivals', body, wide: true, actions: [
    { label: 'Close' },
    { label: 'Apply suggested delays', kind: 'primary', run: () => {
      const apply = rows.filter((r) => r.next != null && r.speakerId && Math.abs(r.next - r.now) >= 0.01);
      if (!apply.length) { toast('Nothing to change.'); return true; }
      const twins = Object.fromEntries(apply.map((r) => [r.speakerId, partnerOf(r.speakerId)?.id]));
      mutate((p) => {
        for (const r of apply) {
          const s = p.speakers.find((x) => x.id === r.speakerId);
          if (!s) continue;
          s.delay_ms = r.next;
          const twin = twins[r.speakerId] && p.speakers.find((x) => x.id === twins[r.speakerId]);
          if (twin && !twin.lock_delay && !apply.some((o) => o.speakerId === twin.id)) twin.delay_ms = r.next;
        }
      }, { structural: true });
      toast(`Changed the delay of ${apply.length} speaker${apply.length === 1 ? '' : 's'}. Undo takes it back.`);
      return true;
    } },
  ] });
}

/** Speaker names for a take's "what was playing". */
export const playingLabel = (m) => (m.speaker_ids || []).map((id) => P().speakers.find((s) => s.id === id)?.name).filter(Boolean).join(', ');

