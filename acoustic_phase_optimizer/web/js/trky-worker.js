// A classic worker for .trky archives (the lossless format from the TRKY converter page, vendored unchanged in
// ../vendor/trky). Decoding and packing are CPU-heavy, so they never run on the page's own thread.
//
// in:  { id, op: 'open', buffer }           -> { id, ok, version, files: [{ name, method, size, hash }] }
//      { id, op: 'extract', name }          -> { id, ok, data: ArrayBuffer, status: 'identical' | 'content-identical' }
//      { id, op: 'pack', files: [{ name, data: ArrayBuffer }] } -> { id, ok, data: ArrayBuffer, size, totalIn }
//      { op: 'cancel' }                     -> the running job stops with 'Cancelled'
// progress while working: { id, progress: { stage, file, frac } }
// A failure is { id, ok: false, error }. The archive given to 'open' stays loaded for later 'extract' calls.

/* global importScripts, TRKY */
importScripts('../vendor/trky/core.js', '../vendor/trky/audio.js', '../vendor/trky/flac.js');

// Limits against hostile or damaged files (the same spirit as the MVR zip reader).
const MAX_ENTRIES = 5000;
const MAX_ENTRY = 1024 ** 3; // 1 GB restored per file
const MAX_TOTAL = 2 * 1024 ** 3; // 2 GB restored in all

let arc = null;

function checkIndex(a) {
  if (!Array.isArray(a.files)) throw new Error('The archive index is damaged.');
  if (a.files.length > MAX_ENTRIES) throw new Error(`The archive lists ${a.files.length.toLocaleString('en-US')} files; up to ${MAX_ENTRIES.toLocaleString('en-US')} can be opened.`);
  const room = a.u8.length - a.base;
  let total = 0;
  for (const e of a.files) {
    const ok = e && typeof e.n === 'string' && e.n.length > 0 && e.n.length < 1024 && typeof e.m === 'string'
      && Number.isSafeInteger(e.o) && Number.isSafeInteger(e.l) && Number.isSafeInteger(e.z)
      && e.o >= 0 && e.l >= 0 && e.z >= 0 && e.o + e.l <= room && typeof e.h === 'string';
    if (!ok) throw new Error('The archive index is damaged or the file is cut short.');
    if (e.z > MAX_ENTRY) throw new Error(`"${e.n}" would unpack to more than 1 GB.`);
    total += e.z;
  }
  if (total > MAX_TOTAL) throw new Error('The archive would unpack to more than 2 GB.');
}

onmessage = async (ev) => {
  const msg = ev.data;
  if (msg.op === 'cancel') { TRKY.state.cancel = true; return; }
  const { id } = msg;
  TRKY.state.cancel = false;
  const progress = (p) => postMessage({ id, progress: { stage: p.stage, file: p.file, frac: p.frac ?? p.f ?? 0 } });
  try {
    if (msg.op === 'open') {
      const a = await TRKY.open(new Uint8Array(msg.buffer));
      checkIndex(a);
      arc = a;
      postMessage({ id, ok: true, version: a.version, files: a.files.map((e) => ({ name: e.n, method: e.m, size: e.z, hash: e.h })) });
    } else if (msg.op === 'extract') {
      if (!arc) throw new Error('No archive is open.');
      const entry = arc.files.find((e) => e.n === msg.name);
      if (!entry) throw new Error(`"${msg.name}" is not in the archive.`);
      const { data, status } = await TRKY.extract(arc, entry, progress, true);
      if (status !== 'identical' && status !== 'content-identical') {
        throw new Error(`"${entry.n}" did not unpack correctly (its checksum does not match). The archive is damaged.`);
      }
      const out = data.byteOffset === 0 && data.byteLength === data.buffer.byteLength ? data.buffer : data.slice().buffer;
      postMessage({ id, ok: true, data: out, status }, [out]);
    } else if (msg.op === 'pack') {
      const files = msg.files.map((f) => ({ name: f.name, data: new Uint8Array(f.data) }));
      const r = await TRKY.pack(files, { verify: true }, progress);
      const out = new Uint8Array(r.size);
      let o = 0;
      for (const p of r.parts) { out.set(p, o); o += p.length; }
      postMessage({ id, ok: true, data: out.buffer, size: r.size, totalIn: r.totalIn }, [out.buffer]);
    } else {
      throw new Error(`unknown request ${msg.op}`);
    }
  } catch (e) {
    postMessage({ id, ok: false, error: (e && e.message) || String(e) });
  }
};
