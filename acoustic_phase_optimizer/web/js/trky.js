// .trky archives: open them (a whole project saved by this app, or any archive holding models, measurements or
// loudspeaker lists) and save the current venue as one, with its scan inside, to share or keep.
// The format and its coders come from the TRKY converter page (vendored unchanged in ../vendor/trky); the work runs in
// trky-worker.js. Every file is checked against its SHA-256 as it is unpacked.

import { api } from './api.js';
import { commit, loadProject, state } from './store.js';
import { download, h, toast } from './util.js';
import { ask, currentDialog, openDialog } from './dialog.js';

export const MAX_ARCHIVE = 300 * 1024 * 1024; // the same as the scan limit
export const MANIFEST = 'apo-manifest.json';
export const PROJECT = 'project.apo.json';

const MODEL_EXTS = ['glb', 'gltf', 'obj', 'ply', 'mvr', '3ds', 'dae', 'fbx', 'stl', 'usdz', 'usd', 'usda', 'usdc', 'dxf', 'dwg', 'ifc', 'step', 'stp', 'iges', 'igs', 'brep', 'brp'];
const GLTF_PARTS = ['bin', 'png', 'jpg', 'jpeg', 'webp', 'ktx2'];
const MEASURE_EXTS = ['txt', 'csv', 'tsv', 'frd', 'dat'];

const extOf = (name) => (/\.([^./\\]+)$/.exec(name)?.[1] || '').toLowerCase();
const baseName = (name) => name.replace(/^.*[\\/]/, '');
const dirOf = (name) => (name.includes('/') ? name.slice(0, name.lastIndexOf('/') + 1) : '');
const mb = (n) => (n >= 1e6 ? `${(n / 1e6).toFixed(n >= 1e8 ? 0 : 1)} MB` : `${Math.max(1, Math.round(n / 1e3))} kB`);

/** One worker per archive job; terminated when done so its memory is returned. */
class Trky {
  constructor(onProgress) {
    this.w = new Worker(new URL('./trky-worker.js', import.meta.url));
    this.next = 1;
    this.wait = new Map();
    this.onProgress = onProgress;
    this.w.onmessage = (e) => {
      const m = e.data, p = this.wait.get(m.id);
      if (!p) return;
      if (m.progress) { this.onProgress?.(m.progress); return; }
      this.wait.delete(m.id);
      if (m.ok) p.resolve(m); else p.reject(new Error(m.error));
    };
    this.w.onerror = (e) => {
      for (const p of this.wait.values()) p.reject(new Error(e.message || 'The archive worker stopped.'));
      this.wait.clear();
    };
  }

  call(msg, transfer = []) {
    const id = this.next++;
    return new Promise((resolve, reject) => {
      this.wait.set(id, { resolve, reject });
      this.w.postMessage({ ...msg, id }, transfer);
    });
  }

  open(buffer) { return this.call({ op: 'open', buffer }, [buffer]); }
  async extract(name) { return new Uint8Array((await this.call({ op: 'extract', name })).data); }
  pack(files) { return this.call({ op: 'pack', files }, files.map((f) => f.data)); }
  cancel() { this.w.postMessage({ op: 'cancel' }); }
  close() { this.w.terminate(); }
}

/** A toast that shows how far a long job is, with a Cancel button. */
function progressToast(verb, job) {
  const el = toast(`${verb}…`, { ms: 3_600_000, action: { label: 'Cancel', run: () => job.cancel() } });
  const span = el.querySelector('span');
  let last = 0;
  return {
    update: (p) => {
      const now = performance.now();
      if (now - last < 150) return;
      last = now;
      span.textContent = `${verb}: ${baseName(p.file || '')} ${Math.round(100 * Math.min(1, p.frac || 0))}%`;
    },
    done: () => el.remove(),
  };
}

// --------------------------------------------------------------------- open

const discardOk = async () => !state.dirty || ask('This venue has changes that are not saved. If you continue they will be lost.',
  { title: 'Discard changes?', yes: 'Discard changes', danger: true });

/** Open a .trky file. Before a saved project replaces the venue on screen, unsaved changes are confirmed. */
export async function openTrky(file) {
  if (file.size > MAX_ARCHIVE) { toast(`${file.name} is ${mb(file.size)}; archives up to ${mb(MAX_ARCHIVE)} can be opened.`, { error: true }); return; }
  let job = null, bar = null;
  try {
    const buffer = await file.arrayBuffer();
    job = new Trky((p) => bar?.update(p));
    bar = progressToast(`Opening ${file.name}`, job);
    const arc = await job.open(buffer);
    const names = arc.files.map((f) => f.name);
    if (names.includes(MANIFEST) && names.includes(PROJECT)) {
      const manifest = JSON.parse(new TextDecoder().decode(await job.extract(MANIFEST)));
      if (manifest?.kind === 'apo-project') {
        bar.done();
        if (!(await discardOk())) return;
        bar = progressToast(`Opening ${file.name}`, job);
        await openBundle(job, manifest, arc.files, file.name);
        return;
      }
    }
    bar.done();
    await openContents(job, arc.files, file.name);
  } catch (e) {
    if (e.message !== 'Cancelled') toast(`Could not open ${file.name}: ${e.message}`, { error: true, ms: 9000 });
    else toast('Stopped.');
  } finally {
    bar?.done();
    job?.close();
  }
}

/** A project saved by this app: the venue, plus its scan if it has one. */
async function openBundle(job, manifest, entries, archiveName) {
  const raw = JSON.parse(new TextDecoder().decode(await job.extract(PROJECT)));
  if (raw?.scan) {
    const path = manifest.scan || `scan/${raw.scan.id}.${raw.scan.ext}`;
    if (entries.some((e) => e.name === path)) {
      const bytes = await job.extract(path);
      const res = await fetch(`/api/scans?ext=${encodeURIComponent(raw.scan.ext)}`, { method: 'POST', body: bytes, headers: { 'Content-Type': 'application/octet-stream' } });
      if (!res.ok) throw new Error(`the scan could not be stored: ${(await res.json().catch(() => ({}))).detail || res.statusText}`);
      raw.scan.id = (await res.json()).id; // the same id (it is the content hash), unless the file was edited
    } else {
      // keep the venue; the 3D view will say the scan is missing
      toast('The archive does not contain the scan this project uses; the room keeps its measured size.', { ms: 8000 });
    }
  }
  const project = await api.checkProject(raw);
  loadProject(project, { dirty: true });
  toast(`Opened ${project.name} from ${archiveName}. Press Save to keep it with your venues.`, { ms: 6000 });
}

/** Any other archive: offer what this app can use. A single model is imported straight away. */
async function openContents(job, entries, archiveName) {
  const models = entries.filter((e) => MODEL_EXTS.includes(extOf(e.name)));
  const texts = entries.filter((e) => MEASURE_EXTS.includes(extOf(e.name)) || extOf(e.name) === 'json');
  const projects = entries.filter((e) => e.name.toLowerCase().endsWith('.apo.json'));
  if (!models.length && !texts.length) {
    toast(`${archiveName} holds ${entries.length} file${entries.length === 1 ? '' : 's'}, none of them a model, measurement or speaker list this app reads.`, { error: true, ms: 9000 });
    return;
  }
  if (models.length === 1 && !texts.length) { await useModel(job, models[0], entries); return; }

  // a choice closes the dialog and is carried out afterwards, while the archive is still open
  let chosen = null;
  const pick = (label, e, run) => h('button', { class: 'btn small', attrs: { 'aria-label': `${label}: ${e.name}` },
    onclick: () => { chosen = { e, run }; currentDialog()?.close(label); } }, label);
  const row = (e, ...buttons) => h('div', { class: 'trky-row' },
    h('span', { class: 'name', text: e.name, title: e.name }), h('small', { text: mb(e.size) }), ...buttons);
  const body = h('div', { class: 'trky-list' },
    h('p', { class: 'hint', text: `${entries.length} file${entries.length === 1 ? '' : 's'} in ${archiveName}, each checked as it is unpacked. Choose one to use:` }),
    ...models.map((e) => row(e, pick('Use as the room', e, () => useModel(job, e, entries)))),
    ...texts.filter((e) => !projects.includes(e)).map((e) => row(e,
      MEASURE_EXTS.includes(extOf(e.name)) ? pick('Add as a measurement', e, () => useMeasurement(job, e)) : null,
      ['csv', 'json'].includes(extOf(e.name)) ? pick('Add to my speakers', e, () => useSpeakers(job, e)) : null)));
  await openDialog({ title: `Open ${archiveName}`, body, wide: true, actions: [{ label: 'Close' }] });
  if (!chosen) return;
  try { await chosen.run(); } catch (err) {
    if (err.message === 'Cancelled') throw err;
    toast(`${baseName(chosen.e.name)}: ${err.message}`, { error: true });
  }
}

async function asFile(job, e) {
  return new File([await job.extract(e.name)], baseName(e.name));
}

async function useModel(job, e, entries) {
  const files = [await asFile(job, e)];
  if (extOf(e.name) === 'gltf') {
    // its buffers and textures, from the same folder
    const dir = dirOf(e.name);
    for (const part of entries) {
      if (part !== e && part.name.startsWith(dir) && GLTF_PARTS.includes(extOf(part.name))) files.push(await asFile(job, part));
    }
  }
  const { importFiles } = await import('./scan.js');
  await importFiles(files);
}

async function useMeasurement(job, e) {
  const text = new TextDecoder().decode(await job.extract(e.name));
  const { addMeasurement } = await import('./panels.js');
  toast(addMeasurement(e.name, text), { ms: 8000 });
}

async function useSpeakers(job, e) {
  const text = new TextDecoder().decode(await job.extract(e.name));
  const { addSpeakerList } = await import('./panels.js');
  const r = await addSpeakerList(text);
  toast(r.message, { error: r.error, ms: r.long ? 12000 : 5000 });
}

// --------------------------------------------------------------------- save

/** The files of a project bundle: manifest, project and the scan's original bytes. */
export async function bundleFiles(project, appVersion) {
  const enc = new TextEncoder();
  const manifest = { kind: 'apo-project', app: 'Acoustic Phase Optimizer', app_version: appVersion || null, schema: project.version ?? 2, created: new Date().toISOString(), name: project.name };
  const files = [];
  if (project.scan) {
    const res = await fetch(`/api/scans/${project.scan.id}.${project.scan.ext}`);
    if (!res.ok) throw new Error('the scan file of this venue is missing from this computer');
    manifest.scan = `scan/${project.scan.id}.${project.scan.ext}`;
    files.push({ name: manifest.scan, data: await res.arrayBuffer() });
  }
  files.unshift(
    { name: MANIFEST, data: enc.encode(JSON.stringify(manifest, null, 2)).buffer },
    { name: PROJECT, data: enc.encode(JSON.stringify(project, null, 2)).buffer });
  return files;
}

/** Save the venue on screen, with its scan, as one .trky file. */
export async function saveTrky() {
  commit();
  const project = state.project;
  const name = (project.name || 'venue').replace(/[<>:"/\\|?*\u0000-\u001f]+/g, ' ').trim() || 'venue';
  let job = null, bar = null;
  try {
    const files = await bundleFiles(project, state.meta?.version);
    job = new Trky((p) => bar?.update(p));
    bar = progressToast('Packing', job);
    const res = await job.pack(files);
    bar.done();
    download(`${name}.trky`, new Blob([res.data], { type: 'application/octet-stream' }));
    const saved = res.totalIn ? Math.round(100 * (1 - res.size / res.totalIn)) : 0;
    toast(`Saved ${name}.trky: ${mb(res.size)}${saved > 0 ? ` (${saved}% smaller than the ${mb(res.totalIn)} it holds)` : ''}.`, { ms: 7000 });
  } catch (e) {
    if (e.message === 'Cancelled') toast('Stopped.');
    else toast(`Could not save the archive: ${e.message}`, { error: true });
  } finally {
    bar?.done();
    job?.close();
  }
}

/** Ask for a .trky file and open it. */
export function chooseTrky() {
  const input = h('input', { type: 'file', accept: '.trky', style: { display: 'none' },
    onchange: () => { const f = input.files?.[0]; input.remove(); if (f) openTrky(f); } });
  document.body.append(input);
  input.click();
}
