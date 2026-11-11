// Application state, undo history and the request pumps that keep results in step with the model.

import { api } from './api.js';
import { clone, debounce, round, toast, uid } from './util.js';

const listeners = new Map();
export function on(events, fn) {
  for (const evt of events.split(' ')) {
    if (!listeners.has(evt)) listeners.set(evt, new Set());
    listeners.get(evt).add(fn);
  }
}
function emit(evt, data = {}) {
  for (const fn of listeners.get(evt) || []) {
    try { fn(data); } catch (e) { console.error(`listener for "${evt}" failed`, e); }
  }
}
/** Tell every view the colour theme changed so canvases repaint with the new palette. */
export const themeChanged = () => emit('theme');

/** Ear height at distance y from the stage: raked seating rises evenly from the front row to the back. */
export function earAt(a, y) {
  const span = a.y1 - a.y0;
  const t = span > 0 ? Math.min(1, Math.max(0, (y - a.y0) / span)) : 0;
  return a.ear_height + (a.rake || 0) * t;
}

export const DEFAULT_WEIGHTS = { coverage: 1, interference: 1, clarity: 1, timing: 1, tonal: 1 };

export const state = {
  meta: null,
  project: null,
  baseline: null, // the project as it was before the last optimization (for before/after)
  sim: null,
  simBaseline: null,
  simBusy: false,
  simError: null,
  point: null,
  selection: null, // { kind: 'speaker' | 'mic', id }
  probe: null, // { x, y }
  lastAction: null, // { title, notes, changes, before, after }
  job: null, // running deep optimization
  busy: null, // name of a short blocking action
  dirty: false,
  savedName: null, // the name this venue was last saved under, if it ever was
  ui: {
    tab: 'room', map: 'direct', band: 'all', compare: 'after', view: 'plan', sectionAxis: 'length',
    source: 'probe', chart: 'magnitude', table: false,
    weights: { ...DEFAULT_WEIGHTS }, eqBands: 5, budget: 15, units: 'metric',
  },
};

// ------------------------------------------------------------------ history
let committed = '';
let savedJson = '';
const past = [];
let future = [];
const LIMIT = 150;

const snapshot = () => JSON.stringify(state.project);
export const canUndo = () => past.length > 0 || snapshot() !== committed;
export const canRedo = () => future.length > 0;

export function commit() {
  const now = snapshot();
  if (now === committed) return false;
  past.push(committed);
  if (past.length > LIMIT) past.shift();
  future = [];
  committed = now;
  emit('history');
  return true;
}

function restore(json) {
  state.project = JSON.parse(json);
  committed = json;
  if (state.lastAction) dropComparison(); // the before/after no longer describes what is on screen
  if (state.selection && !findSelection()) state.selection = null;
  afterChange({ structural: true, source: 'history' });
  emit('history');
  emit('selection');
}

export function undo() {
  commit();
  if (!past.length) return;
  future.push(committed);
  restore(past.pop());
}

export function redo() {
  if (!future.length) return;
  past.push(committed);
  restore(future.pop());
}

// ---------------------------------------------------------------- mutation
/** Change the project. commit:false is for live drags; call commit() when the gesture ends. */
export function mutate(fn, { commit: doCommit = true, interacting = false, source = null, structural = false } = {}) {
  fn(state.project);
  if (doCommit) commit();
  afterChange({ interacting, source, structural });
}

function afterChange({ interacting = false, source = null, structural = false } = {}) {
  const dirty = snapshot() !== savedJson;
  if (dirty !== state.dirty) { state.dirty = dirty; emit('saved'); }
  emit('project', { interacting, source, structural });
  requestSim(interacting);
  autosave();
}

/**
 * Show a project. saved: it was just opened from disk. savedName/dirty: restore that status from a session.
 * lastAction goes with baseline so a restored comparison still has its change list.
 */
export function loadProject(project, { baseline = null, lastAction = null, saved = false, savedName = null, dirty = false,
  ui = null, probe = null } = {}) {
  state.project = project;
  state.savedName = saved ? project.name : savedName;
  if (!lastAction) baseline = null;
  state.baseline = baseline;
  state.simBaseline = null;
  state.sim = null;
  state.point = null;
  state.lastAction = lastAction;
  state.selection = null;
  if (ui) Object.assign(state.ui, ui, { weights: { ...DEFAULT_WEIGHTS, ...(ui.weights || {}) } });
  const a = project.audience;
  state.probe = probe || { x: round((a.x0 + a.x1) / 2, 1), y: round((a.y0 + a.y1) / 2, 1) };
  if (state.ui.source !== 'probe' && state.ui.source !== 'average' && !project.mics.some((m) => m.id === state.ui.source)) {
    state.ui.source = 'probe';
  }
  state.ui.compare = 'after';
  committed = snapshot();
  past.length = 0;
  future = [];
  // An example or a fresh room is not "changed" until you change it; a restored edited session still is.
  const isDirty = !saved && dirty;
  savedJson = isDirty ? '' : committed;
  state.dirty = isDirty;
  emit('saved');
  emit('history');
  emit('selection');
  emit('action');
  emit('ui');
  emit('project', { structural: true, source: 'load' });
  requestSim(false);
  if (baseline) fetchBaseline();
  autosave();
}

export function markSaved() {
  savedJson = snapshot();
  state.savedName = state.project.name;
  state.dirty = false;
  emit('saved');
  autosave();
}

/** The saved copy is gone (deleted); what is on screen is now the only copy. */
export function forgetSaved() {
  savedJson = '';
  state.savedName = null;
  state.dirty = true;
  emit('saved');
  autosave();
}

// -------------------------------------------------------------- selections
function findSelection() {
  const s = state.selection;
  if (!s) return null;
  return (s.kind === 'speaker' ? state.project.speakers : state.project.mics).find((o) => o.id === s.id) || null;
}
export const selected = findSelection;
export const speakerById = (id) => state.project.speakers.find((s) => s.id === id);

export function select(kind, id) {
  const next = kind ? { kind, id } : null;
  if (JSON.stringify(next) === JSON.stringify(state.selection)) return;
  state.selection = next;
  if (kind === 'speaker' && state.ui.tab !== 'speakers') state.ui.tab = 'speakers';
  if (kind === 'mic' && state.ui.tab !== 'mics') state.ui.tab = 'mics';
  emit('selection');
}

export function setUI(patch) {
  Object.assign(state.ui, patch);
  emit('ui', patch);
  if ('source' in patch) requestPoint();
  autosave();
}

export function setProbe(x, y, { final = true } = {}) {
  const r = state.project.room;
  state.probe = { x: round(Math.min(r.width, Math.max(0, x)), 2), y: round(Math.min(r.length, Math.max(0, y)), 2) };
  if (state.ui.source !== 'probe') state.ui.source = 'probe';
  emit('probe', { final });
  requestPoint();
  if (final) autosave();
}

// ---------------------------------------------------------------- speakers
/** The mirror-image partner of a speaker, if the engine has linked them. */
export function partnerOf(id) {
  if (!state.project.settings.link_symmetric || !state.sim?.units) return null;
  const unit = state.sim.units.find((u) => u.includes(id));
  if (!unit || unit.length !== 2) return null;
  return speakerById(unit.find((x) => x !== id)) || null;
}

const MIRRORED = new Set(['y', 'z', 'pitch', 'coverage_h', 'coverage_v', 'level_db', 'hp_hz', 'lp_hz', 'directivity_hz',
  'role', 'model', 'delay_ms', 'gain_db', 'invert', 'lock_delay', 'lock_gain',
  'array_boxes', 'array_splay', 'array_box_h', 'array_box_curve']);

/** Edit a speaker; a linked partner follows as a mirror image so the pair stays matched. */
export function editSpeaker(id, patch, opts) {
  const partnerId = partnerOf(id)?.id;
  mutate((p) => {
    const s = p.speakers.find((x) => x.id === id);
    if (!s) return;
    Object.assign(s, patch);
    const partner = partnerId && p.speakers.find((x) => x.id === partnerId);
    if (!partner) return;
    for (const [k, v] of Object.entries(patch)) {
      if (k === 'x') partner.x = round(p.room.width - v, 3);
      else if (k === 'yaw') partner.yaw = -v;
      else if (k === 'peq') partner.peq = v.map((b) => ({ ...b, id: (b.id.startsWith('auto') ? 'auto-' : '') + uid() }));
      else if (MIRRORED.has(k)) partner[k] = v;
    }
  }, opts);
}

function freeColor(p) {
  const used = new Set(p.speakers.map((s) => s.color));
  for (let c = 0; c < 8; c++) if (!used.has(c)) return c;
  return p.speakers.length % 8;
}

function uniqueName(p, base) {
  const names = new Set(p.speakers.map((s) => s.name));
  if (!names.has(base)) return base;
  for (let n = 2; n < 200; n++) if (!names.has(`${base} ${n}`)) return `${base} ${n}`;
  return base;
}

const TYPE_FIELDS = ['coverage_h', 'coverage_v', 'level_db', 'hp_hz', 'lp_hz', 'directivity_hz', 'role', 'array_boxes', 'array_splay', 'array_box_h', 'array_box_curve'];
/** The acoustic settings a speaker type puts on a speaker (never its label, brand or datasheet notes). */
export function typeFields(type) {
  return { array_boxes: 0, array_splay: 0, array_box_h: 0, array_box_curve: 0, ...Object.fromEntries(TYPE_FIELDS.filter((k) => k in type).map((k) => [k, type[k]])) };
}

/** A line-array hang's level and vertical coverage from its box spec (same rule as engine/library.py). */
export function hangFields(type, boxes, splay) {
  const n = Math.max(1, Math.round(boxes));
  return { array_boxes: n, array_splay: splay,
    level_db: Math.min(160, Math.round((type.spec.box_spl + 10 * Math.log10(n)) * 10) / 10),
    coverage_v: Math.max(5, Math.min(180, Math.round(n * splay * 10) / 10)) };
}

/** Fetch the real loudspeakers (library + the user's own) and add them after the generic types. */
export async function loadSpeakerLibrary(api) {
  const res = await api.speakers();
  const generic = state.meta.speaker_types.filter((t) => t.group === 'Generic');
  state.meta.speaker_types = [...generic, ...res.library, ...res.user];
  emit('library', {});
  return res;
}
export const speakerType = (id) => state.meta.speaker_types.find((t) => t.id === id);

/** Near (x, y) but not on top of another speaker. */
function freeSpot(p, x, y) {
  const taken = (cx) => p.speakers.some((s) => Math.hypot(s.x - cx, s.y - y) < 0.8);
  for (let i = 0; i < 40; i++) {
    const cx = round(x + (i % 2 ? 1 : -1) * Math.ceil(i / 2) * 1.0, 1);
    if (cx > 0.2 && cx < p.room.width - 0.2 && !taken(cx)) return { x: cx, y };
  }
  return { x, y };
}

export function addSpeaker(typeId) {
  const type = speakerType(typeId) || state.meta.speaker_types[0];
  const id = uid();
  mutate((p) => {
    const model = type.id;
    const acoustic = typeFields(type);
    const sub = acoustic.role === 'sub';
    p.speakers.push({
      id, name: uniqueName(p, sub ? 'Sub' : acoustic.role === 'main' ? 'Main' : 'Fill'), model,
      ...freeSpot(p, round(p.room.width / 2, 1), sub ? 1 : Math.min(2, p.room.length / 4)),
      z: sub ? 0.5 : round(Math.min(p.room.height - 0.5, 4), 1), yaw: 0, pitch: sub ? 0 : -10,
      ...acoustic, delay_ms: 0, gain_db: 0, invert: false, mute: false, peq: [],
      lock_delay: false, lock_gain: false, color: freeColor(p),
    });
  }, { structural: true });
  select('speaker', id);
  return id;
}

export function duplicateSpeaker(id) {
  const src = speakerById(id);
  if (!src) return;
  const copy = clone(src);
  copy.id = uid();
  mutate((p) => {
    copy.name = uniqueName(p, src.name + ' copy');
    copy.x = round(Math.min(p.room.width - 0.2, src.x + 1), 2);
    copy.color = freeColor(p);
    copy.peq = copy.peq.map((b) => ({ ...b, id: (b.id.startsWith('auto') ? 'auto-' : '') + uid() }));
    p.speakers.push(copy);
  }, { structural: true });
  select('speaker', copy.id);
}

export function mirrorSpeaker(id) {
  const src = speakerById(id);
  if (!src) return;
  const copy = clone(src);
  copy.id = uid();
  mutate((p) => {
    const swap = /\bL\b/.test(src.name) ? src.name.replace(/\bL\b/, 'R') : /\bR\b/.test(src.name) ? src.name.replace(/\bR\b/, 'L')
      : /left/i.test(src.name) ? src.name.replace(/left/i, 'Right') : /right/i.test(src.name) ? src.name.replace(/right/i, 'Left') : src.name + ' mirror';
    copy.name = uniqueName(p, swap);
    copy.x = round(p.room.width - src.x, 3);
    copy.yaw = -src.yaw;
    copy.color = freeColor(p);
    copy.peq = copy.peq.map((b) => ({ ...b, id: (b.id.startsWith('auto') ? 'auto-' : '') + uid() }));
    p.speakers.push(copy);
  }, { structural: true });
  select('speaker', copy.id);
}

export function removeSpeaker(id) {
  if (state.selection?.id === id) state.selection = null;
  mutate((p) => { p.speakers = p.speakers.filter((s) => s.id !== id); }, { structural: true });
  emit('selection');
}

export function addMic() {
  const id = uid();
  mutate((p) => {
    const a = p.audience;
    const names = new Set(p.mics.map((m) => m.name));
    let n = p.mics.length + 1;
    while (names.has(`Mic ${n}`)) n++;
    const at = state.probe || { x: (a.x0 + a.x1) / 2, y: (a.y0 + a.y1) / 2 };
    p.mics.push({ id, name: `Mic ${n}`, x: round(at.x, 1), y: round(at.y, 1), z: round(earAt(a, at.y), 2) });
  }, { structural: true });
  select('mic', id);
}

export function removeMic(id) {
  if (state.selection?.id === id) state.selection = null;
  if (state.ui.source === id) state.ui.source = 'probe';
  mutate((p) => { p.mics = p.mics.filter((m) => m.id !== id); }, { structural: true });
  emit('selection');
  emit('ui');
}

// ------------------------------------------------------------ optimization
function applyResult(title, res) {
  const before = clone(state.project);
  const previousSim = !state.simBusy && state.sim && !state.sim.preview ? state.sim : null;
  mutate((p) => { p.speakers = res.speakers; }, { structural: true, source: 'optimizer' });
  state.baseline = before;
  state.simBaseline = previousSim;
  state.ui.compare = 'after';
  state.lastAction = { title, notes: res.notes || [], changes: res.changes || [], before: res.before, after: res.after };
  emit('action');
  emit('ui');
  if (!previousSim) fetchBaseline();
  autosave();
}

async function fetchBaseline() {
  const target = state.baseline;
  if (!target) return;
  try {
    const sim = await api.simulate(target, false);
    if (state.baseline === target) { state.simBaseline = sim; emit('sim'); }
  } catch (e) { /* comparison map is optional */ }
}

function dropComparison() {
  state.baseline = null;
  state.simBaseline = null;
  state.lastAction = null;
  state.ui.compare = 'after';
  emit('action');
  emit('ui');
  emit('sim');
}

export function clearComparison() {
  dropComparison();
  autosave();
}

/** Put the speakers back as they were before the last optimizer run. Works after a restart too (undoable). */
export function revertLastAction() {
  const before = state.baseline;
  if (!before) return;
  dropComparison();
  mutate((p) => { p.speakers = clone(before.speakers); }, { structural: true, source: 'revert' });
  toast('Back to the settings from before.');
}

async function runBlocking(name, title, call) {
  if (state.busy || state.job) return;
  commit();
  state.busy = name;
  emit('busy');
  try {
    const res = await call();
    applyResult(title, res);
    const d = (res.after?.score ?? 0) - (res.before?.score ?? 0);
    toast(res.changes?.length ? `${title}: score ${res.before.score} → ${res.after.score}${d > 0.05 ? ' ▲' : ''}` : `${title}: nothing needed changing`);
  } catch (e) {
    toast(`${title} failed: ${e.message}`, { error: true });
  } finally {
    state.busy = null;
    emit('busy');
  }
}

export const quickAlign = () => runBlocking('align', 'Quick align', () => api.align(state.project));
export const autoEq = () => runBlocking('eq', 'Auto EQ', () => api.autoEq(state.project, state.ui.eqBands));

export async function deepOptimize() {
  if (state.busy || state.job) return;
  commit();
  try {
    const { job_id } = await api.optimize(state.project, state.ui.weights, state.ui.budget);
    state.job = { id: job_id, progress: {}, cancelling: false };
    emit('busy');
    for (;;) {
      await new Promise((r) => setTimeout(r, 300));
      const view = await api.job(job_id);
      if (!state.job) return;
      state.job.progress = view.progress || {};
      emit('job');
      if (view.state === 'running') continue;
      state.job = null;
      emit('busy');
      if (view.state === 'done') {
        applyResult('Deep optimize', view.result);
        toast(`Deep optimize: score ${view.result.before.score} → ${view.result.after.score}`);
      } else if (view.state === 'cancelled') toast('Optimization cancelled. Nothing was changed.');
      else toast(`Optimization failed: ${view.error}`, { error: true });
      return;
    }
  } catch (e) {
    state.job = null;
    emit('busy');
    toast(`Optimization failed: ${e.message}`, { error: true });
  }
}

export function cancelOptimize() {
  if (!state.job || state.job.cancelling) return;
  state.job.cancelling = true;
  emit('job');
  api.cancelJob(state.job.id).catch(() => {});
}

// ------------------------------------------------------------ request pumps
let simWanted = null;
let simRunning = false;
let lastSimError = '';

function requestSim(preview) {
  simWanted = { preview };
  pumpSim();
}

async function pumpSim() {
  if (simRunning) return;
  simRunning = true;
  state.simBusy = true;
  emit('simstate');
  while (simWanted) {
    const want = simWanted;
    simWanted = null;
    try {
      state.sim = await api.simulate(state.project, want.preview);
      state.simError = null;
      lastSimError = '';
    } catch (e) {
      state.simError = e.message;
      if (e.message !== lastSimError) toast(e.message, { error: true });
      lastSimError = e.message;
    }
    emit('sim');
    requestPoint();
  }
  simRunning = false;
  state.simBusy = false;
  emit('simstate');
}

let pointWanted = false;
let pointRunning = false;

export function pointTarget() {
  const src = state.ui.source;
  if (src === 'average') return null;
  if (src === 'probe') return state.probe ? { ...state.probe, z: null, label: 'Probe' } : null;
  const mic = state.project.mics.find((m) => m.id === src);
  return mic ? { x: mic.x, y: mic.y, z: mic.z, label: mic.name } : null;
}

export function requestPoint() {
  pointWanted = true;
  pumpPoint();
}

async function pumpPoint() {
  if (pointRunning) return;
  pointRunning = true;
  while (pointWanted) {
    pointWanted = false;
    const t = pointTarget();
    if (!t || !state.project.speakers.length) { state.point = null; emit('point'); continue; }
    try {
      state.point = await api.point(state.project, t.x, t.y, t.z);
    } catch (e) { state.point = null; }
    emit('point');
  }
  pointRunning = false;
}

const sessionPayload = () => ({
  project: state.project, baseline: state.baseline,
  ui: { ...state.ui, probe: state.probe, dirty: state.dirty, savedName: state.savedName, lastAction: state.lastAction },
});

const autosave = debounce(() => {
  if (!state.project) return;
  api.saveSession(sessionPayload()).catch(() => {});
}, 900);

/** Write the session now; used when the window is closing. */
export function flushSession() {
  if (!state.project) return;
  autosave.cancel();
  api.saveSession(sessionPayload(), true).catch(() => {});
}
