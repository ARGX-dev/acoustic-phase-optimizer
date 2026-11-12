// Live measurement with the computer's audio interface: play an exponential sweep, record the mic (and optionally a
// loopback of the output on a second input), and let the engine turn it into an impulse response.
// One AudioWorklet plays the sweep and records in the same callback (capture-worklet.js), so the reference the
// engine deconvolves with is exactly what was sent out.

import { state } from './store.js';
import { clamp, clear, fmt, h, toast } from './util.js';
import { currentDialog, openDialog } from './dialog.js';
import { addMeasurementRecord, irView, measurementFrom } from './measure-ui.js';

const P = () => state.project;
const PREFS = 'apo-live';
const LENGTHS = [[1.5, '1.5 s'], [3, '3 s'], [6, '6 s'], [12, '12 s']];

let graph = null; // the take in progress: { ctx, node, stream, fail }

/** The audio graph of the take in progress (for tests and for stopping it). */
export const audioGraph = () => graph;

function prefs() {
  try { return JSON.parse(localStorage.getItem(PREFS)) || {}; } catch { return {}; }
}
function savePrefs(p) {
  try { localStorage.setItem(PREFS, JSON.stringify(p)); } catch { /* private window: settings are not kept */ }
}

/** Stop everything: the mic tracks (the browser's recording indicator goes off) and the audio context. */
export async function stopLive(reason) {
  const g = graph;
  graph = null;
  if (!g) return;
  if (reason) g.fail?.(new Error(reason));
  try { g.node?.port.postMessage({ type: 'stop' }); } catch { /* already gone */ }
  g.stream?.getTracks().forEach((t) => t.stop());
  try { await g.ctx?.close(); } catch { /* already closed */ }
}

const constraints = (deviceId, channels) => ({ audio: {
  ...(deviceId ? { deviceId: { exact: deviceId } } : {}),
  echoCancellation: false, noiseSuppression: false, autoGainControl: false, channelCount: { ideal: channels } } });

/** Open an input for measuring, with the browser's voice processing off. Throws a readable message. */
export async function openInput(deviceId, channels = 1) {
  let stream;
  try {
    stream = await navigator.mediaDevices.getUserMedia(constraints(deviceId, channels));
  } catch (e) {
    throw new Error(e.name === 'NotAllowedError' ? 'The microphone is blocked. Allow it for this app (the icon in the address bar, or the browser\'s site settings) and try again.'
      : e.name === 'NotFoundError' || e.name === 'OverconstrainedError' ? 'That input is not available. Is the interface connected?' : `The input could not be opened (${e.message}).`);
  }
  const track = stream.getAudioTracks()[0];
  const got = track?.getSettings?.() || {};
  const warnings = [];
  const on = [['echoCancellation', 'echo cancellation'], ['noiseSuppression', 'noise suppression'], ['autoGainControl', 'automatic gain']]
    .filter(([k]) => got[k] === true).map(([, label]) => label);
  if (on.length) warnings.push(`The browser kept ${on.join(' and ')} on for this input. The result can be wrong; choose the audio interface instead of a headset or webcam mic.`);
  return { stream, track, got, warnings };
}

/** The input chooser shared by the sweep and the RTA. `opt.deviceId` follows the choice. */
export function inputSelect(opt) {
  const sel = h('select', { attrs: { 'aria-label': 'Input' }, onchange: (e) => { opt.deviceId = e.target.value; } });
  sel.refresh = async () => {
    let list = [];
    try { list = (await navigator.mediaDevices.enumerateDevices()).filter((d) => d.kind === 'audioinput'); } catch { /* none */ }
    clear(sel).append(h('option', { value: '', text: 'Default input' }),
      ...list.filter((d) => d.deviceId && d.deviceId !== 'default').map((d, i) => h('option', { value: d.deviceId, text: d.label || `Input ${i + 1}`, selected: d.deviceId === opt.deviceId })));
    if (![...sel.options].some((o) => o.selected && o.value === opt.deviceId)) opt.deviceId = sel.value;
  };
  sel.refresh();
  return sel;
}

export const livePrefs = prefs;
export const saveLivePrefs = (patch) => savePrefs({ ...prefs(), ...patch });

/** Play the sweep and record. Resolves with the raw takes; the caller sends them to the engine. */
async function takeSweep({ deviceId, seconds, levelDb, loopback, onLevel }) {
  await stopLive();
  const ctx = new AudioContext({ latencyHint: 'interactive' });
  graph = { ctx };
  const mine = graph;
  await ctx.resume();
  const fs = ctx.sampleRate;
  const f2 = Math.min(20000, Math.floor(fs * 0.45));
  await ctx.audioWorklet.addModule(new URL('./capture-worklet.js', import.meta.url));
  const channels = loopback ? 2 : 1;
  const node = new AudioWorkletNode(ctx, 'apo-capture', { numberOfInputs: 1, numberOfOutputs: 1, outputChannelCount: [2],
    channelCount: channels, channelCountMode: 'explicit', channelInterpretation: 'discrete', processorOptions: { channels } });
  mine.node = node;
  node.connect(ctx.destination);

  const { stream, got, warnings } = await openInput(deviceId, channels);
  if (graph !== mine) { stream.getTracks().forEach((t) => t.stop()); throw new Error('Stopped.'); }
  mine.stream = stream;
  const track = stream.getAudioTracks()[0];
  if (loopback && got.channelCount != null && got.channelCount < 2) {
    throw new Error('This input has only one channel, so there is no second input for the loopback. Turn the loopback off, or choose the interface\'s two-channel input.');
  }
  ctx.createMediaStreamSource(stream).connect(node);

  const res = await fetch(`/api/measure/sweep?${new URLSearchParams({ fs: String(fs), seconds: String(seconds), f1: '20', f2: String(f2) })}`);
  if (!res.ok) throw new Error('The engine could not make the sweep.');
  const sweep = new Float32Array(await res.arrayBuffer());
  const g = 10 ** (levelDb / 20);
  const pre = Math.round(fs * 0.1);
  const signal = new Float32Array(pre + sweep.length);
  for (let i = 0; i < sweep.length; i++) signal[pre + i] = sweep[i] * g;
  const rt = state.sim?.room?.rt60_mid;
  const tail = clamp((rt || 1.5) * 1.5 + 0.5, 1.5, 8); // room for the decay and the latency of the chain
  const total = signal.length + Math.round(fs * tail);

  const out = await new Promise((resolve, reject) => {
    mine.fail = reject;
    const guard = setTimeout(() => reject(new Error('The audio device did not run. Is the output muted or in use by another program?')), (total / fs + 10) * 1000);
    node.port.onmessage = (e) => {
      if (e.data.type === 'level') onLevel?.(e.data);
      else if (e.data.type === 'done') { clearTimeout(guard); resolve(e.data); }
    };
    node.port.postMessage({ type: 'start', signal, total }, [signal.buffer]);
  });
  await stopLive();
  return { fs, f2, mic: out.inputs[0], loop: loopback ? out.inputs[1] : null, ref: out.ref, warnings, device: track?.label || '' };
}

/** The body of POST /api/measure/capture: u32 header length, JSON header, padding to 4, planar float32. */
export function captureBody({ fs, f1 = 20, f2, mic, ref, loop = null }) {
  const names = loop ? ['mic', 'ref', 'loop'] : ['mic', 'ref'];
  const n = ref.length;
  const head = new TextEncoder().encode(JSON.stringify({ fs, f1, f2, n, channels: names }));
  const off = Math.ceil((4 + head.length) / 4) * 4;
  const buf = new ArrayBuffer(off + 4 * n * names.length);
  new DataView(buf).setUint32(0, head.length, true);
  new Uint8Array(buf, 4, head.length).set(head);
  const f = new Float32Array(buf, off); // little-endian, as on every platform browsers run on
  [mic, ref, loop].filter(Boolean).forEach((x, i) => f.set(x, i * n));
  return buf;
}

async function analyse(take) {
  let res;
  try {
    res = await fetch('/api/measure/capture', { method: 'POST', body: captureBody(take), headers: { 'Content-Type': 'application/octet-stream' } });
  } catch {
    throw new Error('The engine is not responding. Is the app still running?');
  }
  const out = await res.json().catch(() => ({}));
  if (!res.ok) throw new Error(typeof out.detail === 'string' ? out.detail : res.statusText);
  return out;
}

// ------------------------------------------------------------------------------ dialog

/** Measure with a sweep: choose the input, where the mic is and what is playing, then take it. */
export async function liveSweepDialog() {
  if (!navigator.mediaDevices?.getUserMedia || !window.AudioWorkletNode) {
    toast('This window cannot record audio. Open the app in Edge or Chrome.', { error: true });
    return;
  }
  const pr = prefs();
  const opt = {
    deviceId: pr.deviceId || '', seconds: pr.seconds || 3, levelDb: pr.levelDb ?? -12, loopback: !!pr.loopback,
    micId: state.selection?.kind === 'mic' ? state.selection.id : (P().mics[0]?.id || ''), speakerId: '',
  };
  const devSel = inputSelect(opt);
  const seg = (items, key) => h('div', { class: 'seg' }, items.map(([v, label]) => h('button', { text: label, attrs: { 'aria-pressed': String(opt[key] === v) },
    onclick: (e) => { opt[key] = v; [...e.currentTarget.parentNode.children].forEach((b) => b.setAttribute('aria-pressed', String(b === e.currentTarget))); } })));
  const field = (label, el, title) => h('label', { class: 'field', title }, h('span', { text: label }), el);
  const level = h('input', { type: 'number', min: -60, max: 0, step: 1, value: opt.levelDb, class: 'num', attrs: { 'aria-label': 'Sweep level, dBFS' },
    onchange: () => { opt.levelDb = clamp(Number(level.value) || -12, -60, 0); level.value = opt.levelDb; } });
  const loop = h('input', { type: 'checkbox', checked: opt.loopback, onchange: () => { opt.loopback = loop.checked; } });
  const name = h('input', { type: 'text', maxLength: 80, placeholder: 'Named from the mic and speaker', attrs: { 'aria-label': 'Name' } });
  const meter = h('div', { class: 'live-meter' }, h('div'));
  const meterText = h('span', { class: 'hint', text: 'Input level appears here during the take.' });
  const bar = h('div', { class: 'progress' }, h('div', { style: { width: '0%' } }));
  const result = h('div', { class: 'live-result' });

  const body = h('div', { class: 'live' },
    h('div', { class: 'grid2' },
      field('Input', devSel, 'The audio interface input the measurement mic is on'),
      field('Mic position', h('select', { attrs: { 'aria-label': 'Mic position' }, onchange: (e) => { opt.micId = e.target.value; } },
        h('option', { value: '', text: 'Not marked' }), P().mics.map((m) => h('option', { value: m.id, text: m.name, selected: m.id === opt.micId })))),
      field('Speaker playing', h('select', { attrs: { 'aria-label': 'Speaker playing' }, onchange: (e) => { opt.speakerId = e.target.value; } },
        h('option', { value: '', text: 'All / several' }), P().speakers.map((s) => h('option', { value: s.id, text: s.name }))), 'Measure one speaker at a time to compare arrivals and set delays'),
      field('Name', name)),
    h('div', { class: 'live-row' },
      h('div', { class: 'field' }, h('span', { text: 'Sweep length' }), seg(LENGTHS, 'seconds')),
      field('Level (dBFS)', level, 'Peak level of the sweep at the output. Start low, then raise it until the take is clean'),
      h('label', { class: 'check', title: 'Feed the interface output straight back into input 2. Arrival times then count from when the sound left, not including the computer\'s latency' },
        loop, h('span', null, 'Loopback on input 2', h('small', { text: 'For absolute arrival times' })))),
    h('p', { class: 'hint', text: 'The sweep plays on this computer\'s default output. Turn the system down before the first take, keep the room quiet, and use a measurement mic on the audio interface: built-in and headset mics filter the sound.' }),
    h('div', { class: 'live-status' }, bar, meter, meterText),
    result);

  let busy = false;
  const measure = async () => {
    if (busy) return false;
    if ((P().measurements || []).length >= 8) { toast('Up to 8 measurements per venue. Delete one first.', { error: true }); return false; }
    busy = true;
    const btn = currentDialog()?.buttons.find((b) => b.textContent === 'Measure');
    if (btn) btn.disabled = true;
    saveLivePrefs({ deviceId: opt.deviceId, seconds: opt.seconds, levelDb: opt.levelDb, loopback: opt.loopback });
    clear(result);
    meterText.textContent = 'Starting…';
    try {
      const take = await takeSweep({ deviceId: opt.deviceId, seconds: opt.seconds, levelDb: opt.levelDb, loopback: opt.loopback,
        onLevel: ({ peak, t, total }) => {
          const db = 20 * Math.log10(Math.max(peak[0], 1e-6));
          meter.firstChild.style.width = `${clamp((db + 60) / 60, 0, 1) * 100}%`;
          meter.classList.toggle('hot', db > -3);
          meterText.textContent = `Mic ${db > -99 ? fmt(db, 0) : '–∞'} dBFS${peak[1] != null ? ` · loopback ${fmt(20 * Math.log10(Math.max(peak[1], 1e-6)), 0)} dBFS` : ''}`;
          bar.firstChild.style.width = `${Math.min(100, (t / total) * 100)}%`;
        } });
      devSel.refresh(); // labels are only given once the mic is allowed
      meterText.textContent = 'Analysing…';
      const res = await analyse(take);
      const mic = P().mics.find((m) => m.id === opt.micId);
      const spk = P().speakers.find((s) => s.id === opt.speakerId);
      const label = name.value.trim() || [spk?.name, mic ? `at ${mic.name}` : null].filter(Boolean).join(' ') || `Sweep ${(P().measurements || []).length + 1}`;
      const m = addMeasurementRecord(measurementFrom(label, res, {
        source: 'sweep', loopback: !!take.loop, mic_id: mic?.id ?? null, speaker_ids: spk ? [spk.id] : [] }));
      meterText.textContent = `Added "${m.name}". ${take.device ? `Input: ${take.device}.` : ''}`;
      res.analysis.loopback = !!take.loop;
      result.append(irView(res.analysis, { warnings: [...take.warnings, ...res.warnings] }));
      name.value = '';
    } catch (e) {
      await stopLive();
      meterText.textContent = '';
      bar.firstChild.style.width = '0%';
      result.append(h('p', { class: 'error-text', attrs: { role: 'alert' }, text: e.message }));
    } finally {
      busy = false;
      if (btn) btn.disabled = false;
    }
    return false;
  };
  await openDialog({ title: 'Measure with a sweep', body, wide: true,
    onClose: () => stopLive('Stopped.'),
    actions: [{ label: 'Close' }, { label: 'Measure', kind: 'primary', run: measure }] });
}
