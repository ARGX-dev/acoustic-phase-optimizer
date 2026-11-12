// MVR (My Virtual Rig, DIN SPEC 15801): a ZIP archive with GeneralSceneDescription.xml and the 3D files it
// places. We read the venue geometry (scene objects, trusses, screens and their symbols) into one three.js
// group in metres, Z up. Fixtures carry no geometry in MVR (it lives in their GDTF files), so they are
// counted and kept as empty markers (userData.fixture) for the speaker finder, not drawn.
//
// Everything is parsed here in the browser: a small ZIP reader (STORE and DEFLATE, the only methods MVR
// allows) with hard limits, DOMParser for the XML (no external entities are ever fetched), and three.js
// loaders for the geometry. Nothing in the archive is executed or fetched from outside it.

import * as THREE from 'three';
import { parseGltf } from './formats.js';
import { TDSLoader } from 'three/addons/loaders/TDSLoader.js';

const MAX_ENTRIES = 20000;
const MAX_ENTRY_BYTES = 512 * 1024 * 1024; // one inflated file
const MAX_TOTAL_BYTES = 1024 * 1024 * 1024; // everything inflated
const MAX_INSTANCES = 20000;
const MAX_DEPTH = 24; // nested groups and symbols

// ------------------------------------------------------------------ ZIP
function readZip(buffer) {
  const v = new DataView(buffer);
  const n = buffer.byteLength;
  let eocd = -1;
  for (let i = n - 22; i >= Math.max(0, n - 22 - 65535); i--) if (v.getUint32(i, true) === 0x06054b50) { eocd = i; break; }
  if (eocd < 0) throw new Error('this is not an MVR file (no ZIP directory found)');
  const count = v.getUint16(eocd + 10, true);
  const cdOffset = v.getUint32(eocd + 16, true);
  if (count === 0xffff || cdOffset === 0xffffffff) throw new Error('the archive uses ZIP64, which MVR files do not need; export it again');
  if (count > MAX_ENTRIES) throw new Error(`the archive lists ${count} files; the limit is ${MAX_ENTRIES}`);
  const entries = new Map();
  let p = cdOffset;
  const dec = new TextDecoder();
  for (let i = 0; i < count; i++) {
    if (p + 46 > n || v.getUint32(p, true) !== 0x02014b50) throw new Error('the archive directory is damaged');
    const flags = v.getUint16(p + 8, true), method = v.getUint16(p + 10, true);
    const csize = v.getUint32(p + 20, true), usize = v.getUint32(p + 24, true);
    const nl = v.getUint16(p + 28, true), xl = v.getUint16(p + 30, true), cl = v.getUint16(p + 32, true);
    const local = v.getUint32(p + 42, true);
    const name = dec.decode(new Uint8Array(buffer, p + 46, nl));
    p += 46 + nl + xl + cl;
    if (name.endsWith('/')) continue; // a folder
    if (flags & 1) throw new Error('the archive is encrypted');
    // MVR keeps every file at the root; look files up by their base name, case-insensitively.
    const key = name.split('/').pop().toLowerCase();
    entries.set(key, { name, method, csize, usize, local });
  }
  return { entries, has: (f) => entries.has(base(f)), read: (f) => extract(buffer, entries.get(base(f))) };
}
const base = (f) => String(f).replace(/\\/g, '/').split('/').pop().toLowerCase();

let inflated = 0;
async function extract(buffer, e) {
  if (!e) return null;
  const v = new DataView(buffer);
  if (e.local + 30 > buffer.byteLength || v.getUint32(e.local, true) !== 0x04034b50) throw new Error(`"${e.name}" is damaged`);
  const start = e.local + 30 + v.getUint16(e.local + 26, true) + v.getUint16(e.local + 28, true);
  if (start + e.csize > buffer.byteLength) throw new Error(`"${e.name}" is cut short`);
  const raw = new Uint8Array(buffer, start, e.csize);
  if (e.method === 0) return raw.slice().buffer;
  if (e.method !== 8) throw new Error(`"${e.name}" uses a compression MVR does not allow`);
  // Inflate as a stream and stop at the limits, whatever sizes the archive claims (no zip bombs).
  const reader = new Blob([raw]).stream().pipeThrough(new DecompressionStream('deflate-raw')).getReader();
  const parts = [];
  let size = 0;
  for (;;) {
    const { done, value } = await reader.read();
    if (done) break;
    size += value.length;
    inflated += value.length;
    if (size > MAX_ENTRY_BYTES || inflated > MAX_TOTAL_BYTES) { reader.cancel(); throw new Error('the archive unpacks to more than the app will read'); }
    parts.push(value);
  }
  const out = new Uint8Array(size);
  let o = 0;
  for (const part of parts) { out.set(part, o); o += part.length; }
  return out.buffer;
}

// ------------------------------------------------------------------ scene
const kids = (el, tag) => (el ? [...el.children].filter((c) => !tag || c.tagName === tag) : []);
const kid = (el, tag) => kids(el, tag)[0] || null;

/** An MVR <Matrix>: {u}{v}{w}{o}, row vectors (point = x·u + y·v + z·w + o), offsets in millimetres. */
function matrixOf(el) {
  const m = kid(el, 'Matrix');
  const nums = m ? (m.textContent.match(/[-+]?(?:\d+\.?\d*|\.\d+)(?:e[-+]?\d+)?/gi) || []).map(Number) : [];
  if (nums.length < 12 || !nums.slice(0, 12).every(Number.isFinite)) return new THREE.Matrix4();
  const [u1, u2, u3, v1, v2, v3, w1, w2, w3, o1, o2, o3] = nums;
  return new THREE.Matrix4().set(u1, v1, w1, o1, u2, v2, w2, o2, u3, v3, w3, o3, 0, 0, 0, 1);
}

const GEOMETRY_OWNERS = new Set(['SceneObject', 'Truss', 'Support', 'VideoScreen', 'Projector']);

/**
 * Parse an .mvr file into a group in metres, Z up. userData.mvr reports what was found.
 * @param {ArrayBuffer} buffer
 */
export async function parseMvr(buffer) {
  inflated = 0;
  const zip = readZip(buffer);
  const xmlBuf = await zip.read('GeneralSceneDescription.xml');
  if (!xmlBuf) throw new Error('GeneralSceneDescription.xml is missing, so this is not a valid MVR file');
  const doc = new DOMParser().parseFromString(new TextDecoder().decode(xmlBuf), 'application/xml');
  if (doc.getElementsByTagName('parsererror').length) throw new Error('the scene description is not valid XML');
  const scene = doc.querySelector('GeneralSceneDescription > Scene');
  if (!scene) throw new Error('the file has no scene');
  const symdefs = new Map(kids(kid(scene, 'AUXData'), 'Symdef').map((s) => [s.getAttribute('uuid'), s]));

  // Blob URLs for every archive file a glTF or 3DS may reference (buffers, textures), made on demand.
  const urls = new Map();
  const blobUrl = async (name) => {
    const k = base(name);
    if (!urls.has(k)) {
      const data = zip.has(k) ? await zip.read(k) : null;
      urls.set(k, data ? URL.createObjectURL(new Blob([data])) : null);
    }
    return urls.get(k);
  };

  // Geometry files: each parsed once, then cloned per instance (clones share the meshes' buffers).
  const files = new Map();
  const missing = new Set();
  async function geometry(fileName) {
    const k = base(fileName.includes('.') ? fileName : `${fileName}.3ds`); // no extension means 3DS
    if (files.has(k)) return files.get(k);
    const job = (async () => {
      if (!zip.has(k)) { missing.add(fileName); return null; }
      const data = await zip.read(k);
      const ext = k.split('.').pop();
      // Pre-resolve any file the loader may ask for, then hand the loader blob URLs instead of paths.
      const manager = new THREE.LoadingManager();
      for (const name of zip.entries.keys()) if (/\.(png|jpe?g|bmp|tga|webp|bin)$/i.test(name)) await blobUrl(name);
      manager.setURLModifier((u) => (/^(data|blob):/.test(u) ? u : urls.get(base(decodeURIComponent(u))) || 'data:,'));
      if (ext === '3ds') {
        const obj = new TDSLoader(manager).parse(data, ''); // 3DS in MVR: 1 unit = 1 mm
        return obj;
      }
      if (ext === 'glb' || ext === 'gltf') {
        const src = ext === 'gltf' ? new TextDecoder().decode(data) : data;
        const root = await parseGltf(src, manager);
        // glTF is metres, Y up: turn it into MVR's millimetres, Z up.
        const holder = new THREE.Group();
        holder.add(root);
        holder.scale.setScalar(1000);
        holder.rotation.x = Math.PI / 2;
        return holder;
      }
      missing.add(fileName);
      return null;
    })();
    files.set(k, job);
    return job;
  }

  const out = new THREE.Group();
  const info = { objects: 0, instances: 0, fixtures: 0, files: 0, missing: [] };
  const placed = [];
  const fixtures = [];

  async function addGeometries(owner, world, depth) {
    for (const g of kids(kid(owner, 'Geometries'))) await addGeometryNode(g, world, depth);
  }
  async function addGeometryNode(g, world, depth) {
    if (depth > MAX_DEPTH || info.instances >= MAX_INSTANCES) return;
    const m = world.clone().multiply(matrixOf(g));
    if (g.tagName === 'Geometry3D') {
      const src = await geometry(g.getAttribute('fileName') || '');
      if (!src) return;
      const inst = src.clone();
      inst.matrixAutoUpdate = false;
      inst.matrix.copy(m).multiply(src.matrix.clone().compose(src.position, src.quaternion, src.scale));
      placed.push(inst);
      info.instances++;
    } else if (g.tagName === 'Symbol') {
      const def = symdefs.get(g.getAttribute('symdef'));
      for (const c of kids(kid(def, 'ChildList'))) await addGeometryNode(c, m, depth + 1);
    }
  }
  async function walk(list, world, depth) {
    if (depth > MAX_DEPTH) return;
    for (const el of kids(list)) {
      const m = world.clone().multiply(matrixOf(el));
      if (el.tagName === 'Fixture') {
        info.fixtures++;
        // no shape, but a place and an aim: an empty marker the speaker finder reads (GDTF beams point along -Z)
        if (fixtures.length < MAX_INSTANCES) {
          const mark = new THREE.Object3D();
          mark.matrixAutoUpdate = false;
          mark.matrix.copy(m);
          mark.userData.fixture = { name: el.getAttribute('name') || '', spec: kid(el, 'GDTFSpec')?.textContent.trim() || '' };
          fixtures.push(mark);
        }
      }
      if (GEOMETRY_OWNERS.has(el.tagName)) { info.objects++; await addGeometries(el, m, depth); }
      await walk(kid(el, 'ChildList'), m, depth + 1); // groups (and anything else) nest in their parent's frame
    }
  }
  for (const layer of kids(kid(scene, 'Layers'), 'Layer')) await walk(kid(layer, 'ChildList'), matrixOf(layer), 0);

  // 3DS textures load after parsing; let them finish before the archive's blob URLs go.
  setTimeout(() => { for (const u of urls.values()) if (u) URL.revokeObjectURL(u); }, 60000);
  if (!placed.length) {
    throw new Error(info.fixtures
      ? `the file holds ${info.fixtures} fixtures but no venue geometry (MVR keeps fixture shapes in GDTF files). Export the venue model into the MVR, or use GLB or OBJ`
      : 'the file has no 3D geometry');
  }
  // Millimetres to metres, still Z up.
  const mm = new THREE.Group();
  mm.scale.setScalar(0.001);
  for (const inst of placed) mm.add(inst);
  for (const mark of fixtures) mm.add(mark);
  out.add(mm);
  out.updateMatrixWorld(true);
  info.files = [...files.keys()].length;
  info.missing = [...missing];
  out.userData.mvr = info;
  out.traverse((o) => {
    if (!o.isMesh) return;
    (Array.isArray(o.material) ? o.material : [o.material]).forEach((mat) => { mat.side = THREE.DoubleSide; });
    if (!o.geometry.attributes.normal) o.geometry.computeVertexNormals();
  });
  return out;
}
