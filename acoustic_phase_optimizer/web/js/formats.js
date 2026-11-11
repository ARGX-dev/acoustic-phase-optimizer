// Readers for formats that need more than a stock loader call:
//
//  * USD: USDZ packages (as iPhone RoomPlan, Polycam and 3d Scanner App export them, usually binary "crate"
//    layers inside), and standalone .usd, .usda and .usdc layers. three.js's USDLoader (vendored from r186,
//    which reads crate files) does the parsing and already turns metersPerUnit and Z up into metres, Y up.
//  * glTF with separate files: the .gltf, its .bin buffers and its textures are packed into one GLB here in
//    the browser, so storage, reload and the strict content policy all go through the GLB path.
//
// Nothing is fetched from outside the files the user chose, and nothing in them is executed.

import * as THREE from 'three';

const MAX_UNPACKED = 1024 * 1024 * 1024; // the declared size of everything in a USDZ
const MAX_ENTRIES = 20000;

/** Check a USDZ's ZIP directory before it is unpacked: bounded entry count and unpacked size, no encryption. */
function checkUsdz(buffer) {
  const v = new DataView(buffer);
  const n = buffer.byteLength;
  let eocd = -1;
  for (let i = n - 22; i >= Math.max(0, n - 22 - 65535); i--) if (v.getUint32(i, true) === 0x06054b50) { eocd = i; break; }
  if (eocd < 0) throw new Error('this USDZ is damaged (no ZIP directory found)');
  const count = v.getUint16(eocd + 10, true);
  let p = v.getUint32(eocd + 16, true);
  if (count === 0xffff || p === 0xffffffff) throw new Error('this USDZ uses ZIP64, which the app does not read');
  if (count > MAX_ENTRIES) throw new Error(`the package lists ${count} files; the limit is ${MAX_ENTRIES}`);
  let total = 0;
  for (let i = 0; i < count; i++) {
    if (p + 46 > n || v.getUint32(p, true) !== 0x02014b50) throw new Error('the package directory is damaged');
    if (v.getUint16(p + 8, true) & 1) throw new Error('the package is encrypted');
    const method = v.getUint16(p + 10, true);
    if (method !== 0 && method !== 8) throw new Error('the package uses a compression USDZ does not allow');
    total += v.getUint32(p + 24, true); // the unpacker allocates exactly this much per file, so it bounds memory
    if (total > MAX_UNPACKED) throw new Error('the package unpacks to more than the app will read');
    p += 46 + v.getUint16(p + 28, true) + v.getUint16(p + 30, true) + v.getUint16(p + 32, true);
  }
}

/** Parse .usdz, .usd, .usda or .usdc into a group in metres, Y up. */
export async function parseUsd(buffer, ext) {
  const bytes = new Uint8Array(buffer);
  if (ext === 'usdz' || (bytes[0] === 0x50 && bytes[1] === 0x4b)) checkUsdz(buffer);
  const { USDLoader } = await import('three/addons/loaders/USDLoader.js');
  // parse() builds the geometry at once; onLoad follows when embedded textures have decoded. A texture that
  // fails to decode only costs its colour, so the geometry is used either way (and after 20 s at most).
  const group = await new Promise((resolve, reject) => {
    let g = null;
    const done = () => resolve(g);
    try { g = new USDLoader().parse(buffer, '', done, done); } catch (e) { reject(e); return; }
    setTimeout(done, 20000);
  });
  let meshes = 0;
  group.traverse((o) => {
    if (!o.isMesh) return;
    meshes++;
    (Array.isArray(o.material) ? o.material : [o.material]).forEach((m) => { m.side = THREE.DoubleSide; });
    if (!o.geometry.attributes.normal) o.geometry.computeVertexNormals();
  });
  if (!meshes) throw new Error('the file has no 3D geometry');
  return group;
}

// ------------------------------------------------------------------ glTF to GLB
const IMAGE_TYPES = { png: 'image/png', jpg: 'image/jpeg', jpeg: 'image/jpeg', webp: 'image/webp', ktx2: 'image/ktx2' };
// a 1x1 grey PNG standing in for a texture that was not among the chosen files
const BLANK_PNG = Uint8Array.from(atob('iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAAAAAA6fptVAAAACklEQVR4nGNYAAAAogCh3I2xzAAAAABJRU5ErkJggg=='), (c) => c.charCodeAt(0));

function dataUri(uri) {
  const m = /^data:([^;,]*)(;base64)?,(.*)$/s.exec(uri);
  if (!m) return null;
  const body = m[2] ? Uint8Array.from(atob(m[3]), (c) => c.charCodeAt(0)) : new TextEncoder().encode(decodeURIComponent(m[3]));
  return { type: m[1], bytes: body };
}

/**
 * Pack a .gltf and the files it references (buffers, textures) into one GLB File.
 * @param {File} gltfFile the .gltf
 * @param {File[]} others the files chosen with it (or the folder it sits in); matched by path, then by name
 * @returns {Promise<{file: File, missing: string[]}>}
 */
export async function gltfToGlb(gltfFile, others = []) {
  let json;
  try { json = JSON.parse(await gltfFile.text()); } catch { throw new Error('the .gltf is not valid JSON'); }
  if (!json?.asset) throw new Error('the .gltf has no asset header, so it is not a glTF file');
  const dir = (gltfFile.webkitRelativePath || '').split('/').slice(0, -1).join('/');
  const byPath = new Map(), byName = new Map();
  for (const f of others) {
    if (f === gltfFile) continue;
    const rel = f.webkitRelativePath || f.name;
    byPath.set(rel.toLowerCase(), f);
    if (!byName.has(f.name.toLowerCase())) byName.set(f.name.toLowerCase(), f);
  }
  const missing = [];
  const shown = (uri) => { try { return decodeURIComponent(uri).split('/').pop(); } catch { return uri; } }; // as the user sees it
  const resolve = async (uri) => {
    const d = dataUri(uri);
    if (d) return d.bytes;
    if (/^[a-z][a-z0-9+.-]*:/i.test(uri)) return null; // http and the like: never fetched
    let rel;
    try { rel = decodeURIComponent(uri); } catch { rel = uri; }
    rel = rel.replace(/\\/g, '/').replace(/^\.\//, '');
    const full = (dir ? `${dir}/${rel}` : rel).toLowerCase();
    const f = byPath.get(full) || byName.get(rel.split('/').pop().toLowerCase());
    return f ? new Uint8Array(await f.arrayBuffer()) : null;
  };

  const chunks = [];
  let size = 0;
  const append = (bytes) => {
    const offset = size;
    chunks.push(bytes);
    size += bytes.length;
    const pad = (4 - (size % 4)) % 4;
    if (pad) { chunks.push(new Uint8Array(pad)); size += pad; }
    return offset;
  };

  // buffers: one after another in the single GLB buffer
  const bufferBase = [];
  for (const [i, b] of (json.buffers || []).entries()) {
    const bytes = b.uri != null ? await resolve(b.uri) : null;
    if (!bytes) {
      const name = b.uri ? (b.uri.startsWith('data:') ? `buffer ${i}` : shown(b.uri)) : `buffer ${i}`;
      throw new Error(`the .gltf needs "${name}", which was not chosen with it. Choose the .gltf together with its .bin and texture files, or import the folder they are in`);
    }
    if (bytes.length < (b.byteLength || 0)) throw new Error(`"${b.uri}" is shorter than the .gltf says`);
    bufferBase[i] = append(bytes);
  }
  for (const bv of json.bufferViews || []) {
    bv.byteOffset = (bv.byteOffset || 0) + (bufferBase[bv.buffer] ?? 0);
    bv.buffer = 0;
  }
  // images: moved into the buffer as their own buffer views
  json.bufferViews = json.bufferViews || [];
  for (const img of json.images || []) {
    if (img.uri == null) continue;
    const d = dataUri(img.uri);
    let bytes = d ? d.bytes : await resolve(img.uri);
    let type = d?.type || img.mimeType || IMAGE_TYPES[img.uri.split('?')[0].split('.').pop().toLowerCase()] || 'image/png';
    if (!bytes) { missing.push(shown(img.uri)); bytes = BLANK_PNG; type = 'image/png'; }
    json.bufferViews.push({ buffer: 0, byteOffset: append(bytes), byteLength: bytes.length });
    img.bufferView = json.bufferViews.length - 1;
    img.mimeType = type;
    delete img.uri;
  }
  json.buffers = size ? [{ byteLength: size }] : [];
  if (!json.bufferViews.length) delete json.bufferViews;

  const enc = new TextEncoder().encode(JSON.stringify(json));
  const jsonPad = (4 - (enc.length % 4)) % 4;
  const jsonLen = enc.length + jsonPad;
  const total = 12 + 8 + jsonLen + (size ? 8 + size : 0);
  const head = new DataView(new ArrayBuffer(20));
  head.setUint32(0, 0x46546c67, true); head.setUint32(4, 2, true); head.setUint32(8, total, true);
  head.setUint32(12, jsonLen, true); head.setUint32(16, 0x4e4f534a, true); // JSON
  const parts = [new Uint8Array(head.buffer), enc, new Uint8Array(jsonPad).fill(0x20)];
  if (size) {
    const bin = new DataView(new ArrayBuffer(8));
    bin.setUint32(0, size, true); bin.setUint32(4, 0x004e4942, true); // BIN
    parts.push(new Uint8Array(bin.buffer), ...chunks);
  }
  const name = gltfFile.name.replace(/\.gltf$/i, '') + '.glb';
  return { file: new File(parts, name, { type: 'model/gltf-binary' }), missing };
}

// ------------------------------------------------------------------ compressed glTF / GLB
const LIBS = new URL('../vendor/three/examples/jsm/libs/', import.meta.url).href;

/** The JSON part of a GLB (or a .gltf text), to see which extensions a file needs before loading it. */
function gltfJson(data) {
  try {
    if (typeof data === 'string') return JSON.parse(data);
    const v = new DataView(data);
    if (v.getUint32(0, true) !== 0x46546c67) return JSON.parse(new TextDecoder().decode(data));
    const len = v.getUint32(12, true);
    return JSON.parse(new TextDecoder().decode(new Uint8Array(data, 20, len)));
  } catch { return {}; }
}

/**
 * Parse a GLB or .gltf, bringing in the decoders it needs: Draco and meshopt geometry, KTX2 (Basis) textures.
 * Each decoder runs in WebAssembly, loaded only for a file that uses it.
 */
export async function parseGltf(data, manager) {
  const { GLTFLoader } = await import('three/addons/loaders/GLTFLoader.js');
  const used = new Set([...(gltfJson(data).extensionsUsed || [])]);
  const loader = new GLTFLoader(manager);
  const cleanup = [];
  if (used.has('KHR_draco_mesh_compression')) {
    const { DRACOLoader } = await import('three/addons/loaders/DRACOLoader.js');
    const draco = new DRACOLoader(manager).setDecoderPath(`${LIBS}draco/gltf/`).setDecoderConfig({ type: 'wasm' });
    loader.setDRACOLoader(draco);
    cleanup.push(() => draco.dispose());
  }
  if (used.has('EXT_meshopt_compression')) {
    const { MeshoptDecoder } = await import('three/addons/libs/meshopt_decoder.module.js');
    loader.setMeshoptDecoder(MeshoptDecoder);
  }
  if (used.has('KHR_texture_basisu')) {
    const { KTX2Loader } = await import('three/addons/loaders/KTX2Loader.js');
    // the texture formats this GPU can show, asked of a throwaway renderer (the 3D view's uses the same GPU)
    const renderer = new THREE.WebGLRenderer();
    const ktx2 = new KTX2Loader(manager).setTranscoderPath(`${LIBS}basis/`).detectSupport(renderer);
    renderer.dispose();
    renderer.forceContextLoss();
    loader.setKTX2Loader(ktx2);
    cleanup.push(() => ktx2.dispose());
  }
  try {
    const gltf = await new Promise((res, rej) => loader.parse(data, '', res, rej));
    const root = gltf.scene || gltf.scenes?.[0];
    if (!root) throw new Error('the file has no scene');
    return root;
  } finally { cleanup.forEach((f) => f()); }
}

// ------------------------------------------------------------------ converted models
// CAD formats (DXF, DWG, IFC, STEP, IGES, Rhino) are converted once, on import, into a GLB in the model's own
// units: storing that means a reload never converts (or asks) again. The import dialog gets the unit as a hint.

const KNOWN_UNITS = [1, 0.1, 0.01, 0.001, 0.3048, 0.0254, 0.9144];

/** The closest unit the dialog offers, as its option value. */
export function unitOption(metersPerUnit) {
  if (!(metersPerUnit > 0)) return null;
  const best = KNOWN_UNITS.reduce((a, b) => (Math.abs(Math.log(b / metersPerUnit)) < Math.abs(Math.log(a / metersPerUnit)) ? b : a));
  return Math.abs(Math.log(best / metersPerUnit)) < 0.01 ? String(best) : null;
}

/** A Z-up group (in its own units) as a Y-up GLB file. */
export async function toGlbFile(group, name) {
  const { GLTFExporter } = await import('three/addons/exporters/GLTFExporter.js');
  const holder = new THREE.Group();
  holder.rotation.x = -Math.PI / 2; // Z up -> Y up, as glTF expects
  holder.add(group);
  holder.updateMatrixWorld(true);
  const data = await new Promise((resolve, reject) => new GLTFExporter().parse(holder, resolve, reject, { binary: true, onlyVisible: false }));
  return new File([data], name.replace(/\.[^.]+$/, '') + '.glb', { type: 'model/gltf-binary' });
}

const plural = (n, one, many) => `${n} ${n === 1 ? one : many}`;

async function convertDxf(text, name, ui) {
  const { parseDxf, extrudePlan } = await import('./dxf.js');
  const group = parseDxf(text);
  const d = group.userData.dxf;
  if (!d.hasTris && !d.lines) {
    throw new Error(d.acis
      ? `the drawing holds ${plural(d.acis, 'ACIS solid', 'ACIS solids')} (3DSOLID, BODY, REGION or SURFACE) and nothing else. ACIS needs a commercial modeller to read; export the model as 3D faces or a mesh (in AutoCAD: CONVTOMESH, then save as DXF), or as STEP, IFC, FBX or OBJ instead`
      : 'the drawing has no geometry in model space');
  }
  let mpu = d.metersPerUnit;
  const flat = d.flat;
  if (flat) {
    const answer = await ui.askHeight({ name, metersPerUnit: mpu, size: new THREE.Box3().setFromObject(group).getSize(new THREE.Vector3()) });
    if (!answer) return null;
    mpu = answer.metersPerUnit;
    extrudePlan(group, answer.height / mpu);
  }
  const parts = [];
  if (d.faces) parts.push(plural(d.faces, 'surface', 'surfaces'));
  if (d.lines) parts.push(plural(d.lines, 'line', 'lines'));
  if (d.inserts) parts.push(plural(d.inserts, 'block insert', 'block inserts'));
  let note = `${name}: ${parts.join(', ')}`;
  if (flat) note += '; walls raised from the plan';
  if (d.acis) note += `; ${plural(d.acis, 'ACIS solid', 'ACIS solids')} left out (they need a commercial modeller to read)`;
  return { file: await toGlbFile(group, name), hints: { unit: unitOption(mpu), up: 'y' }, note: note + '.' };
}

/**
 * Convert a CAD file into a GLB to import.
 * @param {File} file
 * @param {string} ext
 * @param {{askHeight: (info: object) => Promise<{height: number, metersPerUnit: number}|null>}} ui
 * @returns {Promise<{file: File, hints: {unit: string|null, up: string}, note: string}|null>} null if cancelled
 */
export async function convertModel(file, ext, ui) {
  if (ext === 'dxf') return convertDxf(await file.text(), file.name, ui);
  if (ext === 'dwg') {
    // DWG is closed; the local server converts it with ODA File Converter or LibreDWG if one is installed
    const res = await fetch('/api/convert/dwg', { method: 'POST', body: file, headers: { 'Content-Type': 'application/octet-stream' } });
    if (!res.ok) {
      let msg = `the converter answered ${res.status}`;
      try { msg = (await res.json()).detail || msg; } catch { /* not JSON */ }
      throw new Error(msg.replace(/\.$/, ''));
    }
    return convertDxf(await res.text(), file.name, ui);
  }
  throw new Error(`.${ext} cannot be converted`);
}
