// A classic worker: reads STEP, IGES and BREP with occt-import-js (OpenCascade compiled to WebAssembly,
// LGPL-2.1, vendored unmodified) off the UI thread, since tessellating a large model can take minutes.
//
// in:  { format: 'step' | 'iges' | 'brep', buffer: ArrayBuffer, params: object | null }
// out: { ok: true, meshes: [{ name, color, position: Float32Array, index: Uint32Array }], origin: [x, y, z] }
//      or { ok: false, error: string }
// Positions are moved so the model's lowest corner is at the origin (in double precision, before they become
// 32-bit floats), which keeps models drawn far from the origin exact.

/* global importScripts, occtimportjs */
importScripts('../vendor/occt-import-js/occt-import-js.js');

let ready = null;

onmessage = async (ev) => {
  const { format, buffer, params } = ev.data;
  try {
    ready = ready || occtimportjs({ locateFile: (path) => `../vendor/occt-import-js/${path}` });
    const occt = await ready;
    const result = occt.ReadFile(format, new Uint8Array(buffer), params);
    if (!result || !result.success) throw new Error('OpenCascade could not read the file');
    const lo = [Infinity, Infinity, Infinity];
    for (const m of result.meshes) {
      const p = m.attributes.position.array;
      for (let i = 0; i < p.length; i += 3) for (let k = 0; k < 3; k++) if (p[i + k] < lo[k]) lo[k] = p[i + k];
    }
    const origin = lo.map((v) => (Number.isFinite(v) ? v : 0));
    const meshes = [];
    const transfer = [];
    for (const m of result.meshes) {
      const src = m.attributes.position.array;
      const position = new Float32Array(src.length);
      for (let i = 0; i < src.length; i++) position[i] = src[i] - origin[i % 3];
      const index = new Uint32Array(m.index.array);
      meshes.push({ name: m.name || '', color: m.color || null, position, index });
      transfer.push(position.buffer, index.buffer);
    }
    postMessage({ ok: true, meshes, origin }, transfer);
  } catch (e) {
    postMessage({ ok: false, error: String((e && e.message) || e) });
  }
};
