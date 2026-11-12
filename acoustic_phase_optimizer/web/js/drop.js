// Drop files anywhere on the window: .trky archives, 3D scans and models (or a folder holding an unpacked MVR or a
// glTF with its parts), a saved project (.apo.json) or measurement exports. Each goes where its own button would take it.

import { api } from './api.js';
import { loadProject, state } from './store.js';
import { h, toast } from './util.js';
import { ask, currentDialog } from './dialog.js';

const MEASURE_EXTS = ['txt', 'csv', 'tsv', 'frd', 'dat'];
const MAX_FILES = 20000;
const extOf = (name) => (/\.([^./\\]+)$/.exec(name)?.[1] || '').toLowerCase();

/** Every file in the drop, walking into folders (Chromium's FileSystemEntry API). The entries are taken before the
 * first await: the browser empties the DataTransfer once the drop event is over. */
async function droppedFiles(dt) {
  const entries = [...dt.items].map((it) => (it.kind === 'file' ? it.webkitGetAsEntry?.() : null));
  if (!entries.length || entries.some((e) => !e)) return [...dt.files];
  const files = [];
  const walk = async (entry) => {
    if (files.length >= MAX_FILES) return;
    if (entry.isFile) { files.push(await new Promise((res, rej) => entry.file(res, rej))); return; }
    const reader = entry.createReader();
    for (;;) {
      const batch = await new Promise((res, rej) => reader.readEntries(res, rej)); // gives at most 100 at a time
      if (!batch.length) break;
      for (const e of batch) await walk(e);
    }
  };
  for (const e of entries) await walk(e);
  return files;
}

async function openProjectFile(file) {
  if (file.size > 50_000_000) throw new Error('that project file is larger than 50 MB');
  let raw;
  try { raw = JSON.parse(await file.text()); } catch (e) { throw new Error('it is not a project file (not valid JSON)'); }
  const project = await api.checkProject(raw.project && raw.project.room ? raw.project : raw);
  if (state.dirty && !(await ask('This venue has changes that are not saved. If you continue they will be lost.',
    { title: 'Discard changes?', yes: 'Discard changes', danger: true }))) return;
  loadProject(project, { dirty: true });
  toast(`Opened ${project.name} from ${file.name}. Press Save to keep it with your venues.`, { ms: 6000 });
}

/** Send dropped files where they belong. Exported for tests and for other entry points. */
export async function routeFiles(files) {
  if (!files.length) return;
  const { SCAN_EXTS, importFiles, importScan, folderToMvr } = await import('./scan.js');
  const trky = files.find((f) => extOf(f.name) === 'trky');
  if (trky) { await (await import('./trky.js')).openTrky(trky); return; }
  const project = files.find((f) => f.name.toLowerCase().endsWith('.apo.json'));
  if (project) {
    try { await openProjectFile(project); } catch (e) { toast(`Could not open ${project.name}: ${e.message}`, { error: true }); }
    return;
  }
  if (files.some((f) => f.name.toLowerCase() === 'generalscenedescription.xml')) {
    try { await importScan(await folderToMvr(files)); } catch (e) { toast(`Could not read the folder: ${e.message}`, { error: true }); }
    return;
  }
  if (files.some((f) => SCAN_EXTS.includes(extOf(f.name)) || extOf(f.name) === 'gltf')) { await importFiles(files); return; }
  const texts = files.filter((f) => MEASURE_EXTS.includes(extOf(f.name)) || extOf(f.name) === 'wav');
  if (texts.length) {
    const { addMeasurement } = await import('./panels.js');
    const { addImpulseResponse } = await import('./measure-ui.js');
    for (const f of texts.slice(0, 8)) {
      try {
        const wav = extOf(f.name) === 'wav';
        if (f.size > (wav ? 64 : 20) * 1_000_000) throw new Error(`larger than ${wav ? 64 : 20} MB`);
        toast(wav ? await addImpulseResponse(f.name, await f.arrayBuffer()) : addMeasurement(f.name, await f.text()), { ms: 8000 });
      } catch (e) { toast(`${f.name}: ${e.message}`, { error: true }); }
    }
    return;
  }
  toast(`${files.length === 1 ? `${files[0].name} cannot` : `None of these ${files.length} files can`} be opened here. Drop a .trky archive, a 3D scan or model, a project (.apo.json), a measurement export or an impulse response (.wav).`, { error: true, ms: 8000 });
}

/** Turn on dropping for the whole window. */
export function enableDrop() {
  const overlay = h('div', { class: 'drop-overlay', attrs: { 'aria-hidden': 'true' } },
    h('div', { class: 'drop-card' },
      h('strong', { text: 'Drop to open' }),
      h('span', { text: '.trky archive, 3D scan or model, project (.apo.json) or measurement' })));
  document.body.append(overlay);
  let depth = 0;
  const hasFiles = (e) => [...(e.dataTransfer?.types || [])].includes('Files');
  const show = (on) => overlay.classList.toggle('show', on);
  window.addEventListener('dragenter', (e) => {
    if (!hasFiles(e)) return;
    e.preventDefault();
    depth++;
    if (!currentDialog()) show(true);
  });
  window.addEventListener('dragover', (e) => {
    if (!hasFiles(e)) return;
    e.preventDefault(); // without this the browser would open the file itself
    e.dataTransfer.dropEffect = currentDialog() ? 'none' : 'copy';
  });
  window.addEventListener('dragleave', (e) => {
    if (!hasFiles(e)) return;
    depth = Math.max(0, depth - 1);
    if (!depth) show(false);
  });
  window.addEventListener('drop', async (e) => {
    if (!hasFiles(e)) return;
    e.preventDefault();
    depth = 0;
    show(false);
    if (currentDialog() || !state.project) return;
    try {
      await routeFiles(await droppedFiles(e.dataTransfer));
    } catch (err) { toast(`Could not read what was dropped: ${err.message}`, { error: true }); }
  });
}
