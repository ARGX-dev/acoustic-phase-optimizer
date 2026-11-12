// Real-time analyser: the mic's spectrum in 1/3-octave bars, with a pink-noise generator, and snapshots kept as
// measurements over the prediction. Pink noise has equal energy per octave, so through a flat system the bars are
// level; what the room and speakers do shows up as their shape.

import { state } from './store.js';
import { clamp, fmt, h, toast } from './util.js';
import { openDialog } from './dialog.js';
import { cssVar } from './theme.js';
import { addMeasurementRecord, newMeasurement } from './measure-ui.js';
import { inputSelect, livePrefs, openInput, saveLivePrefs } from './live.js';

const P = () => state.project;
const THIRDS = Array.from({ length: 31 }, (_, i) => 1000 * 2 ** ((i - 17) / 3)); // 20 Hz … 20 kHz
const NOMINAL = ['20', '25', '31.5', '40', '50', '63', '80', '100', '125', '160', '200', '250', '315', '400', '500', '630', '800',
  '1k', '1.25k', '1.6k', '2k', '2.5k', '3.15k', '4k', '5k', '6.3k', '8k', '10k', '12.5k', '16k', '20k'];
// AnalyserNode bins are |X|/N under a Blackman window: one-sided and scaled by the window's mean square (0.3046),
// so this factor makes the bins of a band add up to its power in dBFS.
const BIN_POWER = 2 / (0.42 ** 2 + 0.5 ** 2 / 2 + 0.08 ** 2 / 2);
const AVERAGE = [['fast', 'Fast', 0.125], ['slow', 'Slow', 1], ['inf', 'Until reset', Infinity]];

let graph = null; // { ctx, out, noise, stream, analyser, raf }

/** The RTA's audio graph while it runs (for tests): `out` is the generator's output. */
export const rtaGraph = () => graph;

export async function stopRta() {
  const g = graph;
  graph = null;
  if (!g) return;
  cancelAnimationFrame(g.raf);
  try { g.noise?.stop(); } catch { /* not started */ }
  g.stream?.getTracks().forEach((t) => t.stop());
  try { await g.ctx.close(); } catch { /* closed */ }
}

/** Ten seconds of pink noise (Paul Kellet's filter), scaled to an RMS of 1. */
export function pinkNoise(ctx, seconds = 10) {
  const n = Math.round(ctx.sampleRate * seconds);
  const buf = ctx.createBuffer(1, n, ctx.sampleRate);
  const x = buf.getChannelData(0);
  let b0 = 0, b1 = 0, b2 = 0, b3 = 0, b4 = 0, b5 = 0, b6 = 0, sum = 0;
  for (let i = 0; i < n; i++) {
    const w = Math.random() * 2 - 1;
    b0 = 0.99886 * b0 + w * 0.0555179; b1 = 0.99332 * b1 + w * 0.0750759; b2 = 0.969 * b2 + w * 0.153852;
    b3 = 0.8665 * b3 + w * 0.3104856; b4 = 0.55 * b4 + w * 0.5329522; b5 = -0.7616 * b5 - w * 0.016898;
    x[i] = b0 + b1 + b2 + b3 + b4 + b5 + b6 + w * 0.5362;
    b6 = w * 0.115926;
    sum += x[i] * x[i];
  }
  const k = 1 / Math.sqrt(sum / n);
  // fade the loop seam over 20 ms so it does not click
  const f = Math.round(ctx.sampleRate * 0.02);
  for (let i = 0; i < n; i++) x[i] *= k * Math.min(1, (i + 1) / f, (n - i) / f);
  return buf;
}

/** Band levels (dB) from a power spectrum: the energy in each band, with the nearest bin standing in for a band
 *  too narrow to hold one. `power` is linear per bin, `df` the bin width. */
export function bandLevels(power, df, centres, fraction = 3) {
  const half = 2 ** (1 / (2 * fraction));
  return centres.map((fc) => {
    const lo = fc / half, hi = fc * half;
    const i0 = Math.max(1, Math.ceil(lo / df)), i1 = Math.min(power.length - 1, Math.floor(hi / df));
    let e = 0;
    if (i1 >= i0) for (let i = i0; i <= i1; i++) e += power[i];
    else e = power[clamp(Math.round(fc / df), 1, power.length - 1)] * ((hi - lo) / df);
    return 10 * Math.log10(Math.max(e, 1e-20));
  });
}

function drawBars(cv, levels, peak) {
  const ctx = cv.getContext('2d');
  const W = cv.width, H = cv.height, L = 40, R = 8, T = 8, B = 22;
  ctx.fillStyle = cssVar('--page');
  ctx.fillRect(0, 0, W, H);
  const top = Math.ceil((Math.max(...levels.filter(Number.isFinite), -60) + 3) / 10) * 10;
  const span = 60;
  const Y = (db) => T + clamp((top - db) / span, 0, 1) * (H - T - B);
  ctx.strokeStyle = cssVar('--line');
  ctx.fillStyle = cssVar('--ink-2');
  ctx.font = '11px system-ui, sans-serif';
  ctx.textAlign = 'right';
  ctx.textBaseline = 'middle';
  for (let db = top; db >= top - span; db -= 10) {
    ctx.beginPath(); ctx.moveTo(L, Y(db)); ctx.lineTo(W - R, Y(db)); ctx.stroke();
    ctx.fillText(`${db}`, L - 6, Y(db));
  }
  const bw = (W - L - R) / levels.length;
  ctx.textAlign = 'center';
  ctx.textBaseline = 'top';
  levels.forEach((v, i) => {
    const x = L + i * bw;
    ctx.fillStyle = cssVar('--accent');
    ctx.fillRect(x + 1, Y(v), bw - 2, H - B - Y(v));
    if (peak && Number.isFinite(peak[i])) { ctx.fillStyle = cssVar('--ink'); ctx.fillRect(x + 1, Y(peak[i]) - 1, bw - 2, 2); }
    if (i % 3 === 1) { ctx.fillStyle = cssVar('--ink-2'); ctx.fillText(NOMINAL[i], x + bw / 2, H - B + 5); }
  });
}

/** The RTA dialog. */
export async function rtaDialog() {
  if (!navigator.mediaDevices?.getUserMedia) {
    toast('This window cannot record audio. Open the app in Edge or Chrome.', { error: true });
    return;
  }
  const pr = livePrefs();
  const opt = { deviceId: pr.deviceId || '', noiseDb: pr.noiseDb ?? -20, avg: 'slow', micId: state.selection?.kind === 'mic' ? state.selection.id : (P().mics[0]?.id || '') };
  const devSel = inputSelect(opt);
  const cv = h('canvas', { class: 'ir-chart', width: 720, height: 240, attrs: { role: 'img', 'aria-label': 'Spectrum of the mic in 1/3-octave bands' } });
  const status = h('p', { class: 'hint', attrs: { 'aria-live': 'polite' }, text: 'Starting the input…' });
  const noiseBtn = h('button', { class: 'btn small', attrs: { 'aria-pressed': 'false' }, text: 'Pink noise' });
  const level = h('input', { type: 'number', class: 'num', min: -60, max: -10, step: 1, value: opt.noiseDb, attrs: { 'aria-label': 'Noise level, dBFS RMS' },
    onchange: () => { opt.noiseDb = clamp(Number(level.value) || -20, -60, -10); level.value = opt.noiseDb; if (graph?.out) graph.out.gain.value = 10 ** (opt.noiseDb / 20); } });
  let power = null, frames = 0, peak = null;
  const reset = () => { power = null; frames = 0; peak = null; };
  const avgSeg = h('div', { class: 'seg' }, AVERAGE.map(([k, label]) => h('button', { text: label, attrs: { 'aria-pressed': String(opt.avg === k) },
    onclick: (e) => { opt.avg = k; reset(); [...e.currentTarget.parentNode.children].forEach((b) => b.setAttribute('aria-pressed', String(b === e.currentTarget))); } })));
  const micSel = h('select', { attrs: { 'aria-label': 'Mic position' }, onchange: (e) => { opt.micId = e.target.value; } },
    h('option', { value: '', text: 'Not marked' }), P().mics.map((m) => h('option', { value: m.id, text: m.name, selected: m.id === opt.micId })));
  const field = (label, el, title) => h('label', { class: 'field', title }, h('span', { text: label }), el);
  const body = h('div', { class: 'live' },
    h('div', { class: 'grid2' }, field('Input', devSel), field('Mic position', micSel)),
    h('div', { class: 'live-row' },
      h('div', { class: 'field' }, h('span', { text: 'Averaging' }), avgSeg),
      h('div', { class: 'field' }, h('span', { text: 'Generator' }), noiseBtn),
      field('Noise level (dBFS)', level, 'RMS level of the pink noise at the output')),
    cv, status,
    h('p', { class: 'hint', text: 'Bars: 1/3-octave levels of the mic, dB below full scale. Line: the highest each band reached. Pink noise reads flat through a flat system. Keep a snapshot to compare it with the prediction on the response chart.' }));

  const start = async () => {
    await stopRta();
    const ctx = new AudioContext({ latencyHint: 'interactive' });
    const g = { ctx };
    graph = g;
    await ctx.resume();
    g.out = ctx.createGain();
    g.out.gain.value = 10 ** (opt.noiseDb / 20);
    g.out.connect(ctx.destination);
    const { stream, warnings } = await openInput(opt.deviceId, 1);
    if (graph !== g) { stream.getTracks().forEach((t) => t.stop()); return; }
    g.stream = stream;
    g.analyser = ctx.createAnalyser();
    g.analyser.fftSize = 32768;
    g.analyser.smoothingTimeConstant = 0;
    ctx.createMediaStreamSource(stream).connect(g.analyser);
    devSel.refresh();
    reset();
    status.textContent = warnings.join(' ') || 'Listening.';
    const db = new Float32Array(g.analyser.frequencyBinCount);
    const df = ctx.sampleRate / g.analyser.fftSize;
    let last = performance.now();
    const tick = (now) => {
      if (graph !== g) return;
      g.analyser.getFloatFrequencyData(db);
      const dt = Math.min(0.5, (now - last) / 1000);
      last = now;
      const tau = AVERAGE.find((a) => a[0] === opt.avg)[2];
      const a = tau === Infinity ? 1 / (frames + 1) : 1 - Math.exp(-dt / tau);
      if (!power) power = new Float64Array(db.length);
      for (let i = 0; i < db.length; i++) {
        const p = Number.isFinite(db[i]) ? BIN_POWER * 10 ** (db[i] / 10) : 0;
        power[i] = frames ? power[i] + a * (p - power[i]) : p;
      }
      frames++;
      const levels = bandLevels(power, df, THIRDS);
      peak = peak ? peak.map((v, i) => Math.max(v, levels[i])) : levels.slice();
      g.levels = levels;
      drawBars(cv, levels, peak);
      g.raf = requestAnimationFrame(tick);
    };
    g.raf = requestAnimationFrame(tick);
    g.df = df;
  };
  noiseBtn.onclick = () => {
    const g = graph;
    if (!g?.out) return;
    if (g.noise) {
      try { g.noise.stop(); } catch { /* stopped */ }
      g.noise = null;
    } else {
      g.noise = g.ctx.createBufferSource();
      g.noise.buffer = pinkNoise(g.ctx);
      g.noise.loop = true;
      g.noise.connect(g.out);
      g.noise.start();
      reset();
    }
    noiseBtn.setAttribute('aria-pressed', String(!!g.noise));
    noiseBtn.classList.toggle('primary', !!g.noise);
  };
  devSel.addEventListener('change', () => { start().catch((e) => { status.textContent = e.message; }); });

  const snapshot = () => {
    if (!power || !graph?.df) { toast('Nothing measured yet.', { error: true }); return false; }
    if ((P().measurements || []).length >= 8) { toast('Up to 8 measurements per venue. Delete one first.', { error: true }); return false; }
    // 1/12-octave levels from 20 Hz to 20 kHz: fine enough to show the response, smooth enough to read
    const freqs = Array.from({ length: 121 }, (_, i) => Math.round(20 * 2 ** (i / 12) * 100) / 100).filter((f) => f <= Math.min(20000, graph.ctx.sampleRate * 0.45));
    const spl = bandLevels(power, graph.df, freqs, 12).map((v) => Math.round(v * 100) / 100);
    const mic = P().mics.find((m) => m.id === opt.micId);
    const n = (P().measurements || []).filter((m) => m.source === 'rta').length + 1;
    const m = addMeasurementRecord(newMeasurement(`RTA ${n}${mic ? ` at ${mic.name}` : ''}`, { freqs, spl, source: 'rta', mic_id: mic?.id ?? null }));
    toast(`Kept "${m.name}" on the response chart. ${opt.avg === 'inf' ? `Average of ${fmt(frames, 0)} frames.` : ''}`, { ms: 6000 });
    return false;
  };
  const done = openDialog({ title: 'Real-time analyser', body, wide: true,
    onClose: () => { saveLivePrefs({ deviceId: opt.deviceId, noiseDb: opt.noiseDb }); stopRta(); },
    actions: [{ label: 'Close' }, { label: 'Reset', run: () => { reset(); return false; } }, { label: 'Keep snapshot', kind: 'primary', run: snapshot }] });
  start().catch((e) => { status.textContent = e.message; status.setAttribute('role', 'alert'); });
  await done;
}

