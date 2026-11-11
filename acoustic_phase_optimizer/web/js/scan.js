// Importing a 3D scan or venue model (GLB/glTF, USDZ, OBJ, PLY, MVR, 3DS, FBX, COLLADA, STL): read it in the browser, measure it, let the user confirm the room,
// then store the file and keep only a small reference in the project. Loaded on demand with three.js.

import * as THREE from 'three';
import { OBJLoader } from 'three/addons/loaders/OBJLoader.js';
import { PLYLoader } from 'three/addons/loaders/PLYLoader.js';
import { api } from './api.js';
import { mutate, setUI } from './store.js';
import { clamp, fmt, h, round, toast } from './util.js';
import { cssVar } from './theme.js';
import { openDialog } from './dialog.js';
import { fromDisp, len, lenNum, lenUnit, toDisp } from './units.js';

export const SCAN_EXTS = ['glb', 'obj', 'ply', 'mvr', '3ds', 'dae', 'fbx', 'stl', 'usdz', 'usd', 'usda', 'usdc', 'dxf', 'dwg'];
// CAD formats converted once into a GLB on import (formats.js), so they are stored and reloaded as GLB.
const CONVERT = new Set(['dxf', 'dwg']);
// Chosen alongside a .gltf (which is packed into a GLB before import), never imported on their own.
const GLTF_PARTS = ['gltf', 'bin', 'png', 'jpg', 'jpeg', 'webp'];
const USD = new Set(['usdz', 'usd', 'usda', 'usdc']);
// Native up axis of each format, as the import's first guess (the dialog lets you change it).
const Z_UP = new Set(['mvr', '3ds', 'stl']);
const NOT_READABLE = {
  vwx: 'Vectorworks files are a closed format. Use File > Export > Export MVR (or 3DS, FBX, COLLADA, OBJ, STL).',
  rvt: 'Revit files are a closed format. Export FBX or OBJ from Revit instead.',
  ifc: 'IFC cannot be read yet. Export MVR, 3DS, FBX, COLLADA (DAE), OBJ or STL instead.',
  step: 'STEP needs a full CAD kernel to read. Export a mesh format instead: MVR, 3DS, FBX, COLLADA (DAE), OBJ or STL.',
  stp: 'STEP needs a full CAD kernel to read. Export a mesh format instead: MVR, 3DS, FBX, COLLADA (DAE), OBJ or STL.',
  igs: 'IGES needs a full CAD kernel to read. Export a mesh format instead: MVR, 3DS, FBX, COLLADA (DAE), OBJ or STL.',
  iges: 'IGES needs a full CAD kernel to read. Export a mesh format instead: MVR, 3DS, FBX, COLLADA (DAE), OBJ or STL.',
  sat: 'SAT needs a full CAD kernel to read. Export a mesh format instead: MVR, 3DS, FBX, COLLADA (DAE), OBJ or STL.',
};
/** Loaders ask for textures by path; serve them nothing rather than fetching from the app's server. */
function quietManager() {
  const m = new THREE.LoadingManager();
  m.setURLModifier((u) => (/^(data|blob):/.test(u) ? u : 'data:,'));
  return m;
}
const plain = (geo) => {
  if (!geo.attributes.normal) geo.computeVertexNormals();
  return new THREE.Mesh(geo, new THREE.MeshStandardMaterial({ color: 0xb9b6ae, roughness: 0.9, side: THREE.DoubleSide }));
};
const MAX_BYTES = 300 * 1024 * 1024;
const UNITS = [['1', 'metres'], ['0.1', 'decimetres'], ['0.01', 'centimetres'], ['0.001', 'millimetres'], ['0.3048', 'feet'], ['0.0254', 'inches'], ['0.9144', 'yards']];
const SIDES = ['-y', '+x', '+y', '-x'];

// ------------------------------------------------------------------ loading
function friendly(e) {
  const m = String(e?.message || e);
  return `The scan could not be read: ${m}`;
}

/** Parse a scan or model file into a three.js object (native coordinates, whatever units the file uses). */
export async function parseScan(buffer, ext) {
  const both = (o) => { o.traverse((c) => { if (c.isMesh) { (Array.isArray(c.material) ? c.material : [c.material]).forEach((mt) => { mt.side = THREE.DoubleSide; }); if (!c.geometry.attributes.normal) c.geometry.computeVertexNormals(); } }); return o; };
  if (ext === 'mvr') return (await import('./mvr.js')).parseMvr(buffer); // metres, Z up
  if (USD.has(ext)) return (await import('./formats.js')).parseUsd(buffer, ext); // metres, Y up
  if (ext === 'stl') {
    const { STLLoader } = await import('three/addons/loaders/STLLoader.js');
    return plain(new STLLoader().parse(buffer));
  }
  if (ext === '3ds') {
    const { TDSLoader } = await import('three/addons/loaders/TDSLoader.js');
    return both(new TDSLoader(quietManager()).parse(buffer, ''));
  }
  if (ext === 'dae') {
    const { ColladaLoader } = await import('three/addons/loaders/ColladaLoader.js');
    // The loader warns on every Z-up file (most CAD exports) that it rotates rather than converts: expected here.
    const warn = console.warn;
    console.warn = (...a) => { if (!/Z-UP coordinate system/.test(String(a[0]))) warn(...a); };
    let dae;
    try { dae = new ColladaLoader(quietManager()).parse(new TextDecoder().decode(buffer), ''); } finally { console.warn = warn; }
    if (!dae?.scene) throw new Error('the file has no scene');
    return both(dae.scene); // COLLADA states its unit and up axis; the loader turns it into metres, Y up
  }
  if (ext === 'fbx') {
    const { FBXLoader } = await import('three/addons/loaders/FBXLoader.js');
    return both(new FBXLoader(quietManager()).parse(buffer, ''));
  }
  if (ext === 'glb') {
    const root = await (await import('./formats.js')).parseGltf(buffer); // Draco, meshopt and KTX2 included
    root.traverse((o) => { if (o.isMesh) (Array.isArray(o.material) ? o.material : [o.material]).forEach((m) => { m.side = THREE.DoubleSide; }); });
    return root;
  }
  if (ext === 'obj') {
    const root = new OBJLoader().parse(new TextDecoder().decode(buffer));
    root.traverse((o) => {
      if (!o.isMesh) return;
      const hasColor = !!o.geometry.attributes.color;
      o.material = new THREE.MeshStandardMaterial({ color: hasColor ? 0xffffff : 0xb9b6ae, vertexColors: hasColor, roughness: 0.9, side: THREE.DoubleSide });
      if (!o.geometry.attributes.normal) o.geometry.computeVertexNormals();
    });
    return root;
  }
  const geo = new PLYLoader().parse(buffer);
  const hasColor = !!geo.attributes.color;
  if (geo.index) {
    if (!geo.attributes.normal) geo.computeVertexNormals();
    return new THREE.Mesh(geo, new THREE.MeshStandardMaterial({ color: hasColor ? 0xffffff : 0xb9b6ae, vertexColors: hasColor, roughness: 0.9, side: THREE.DoubleSide }));
  }
  return new THREE.Points(geo, new THREE.PointsMaterial({ size: 0.035, vertexColors: hasColor, color: hasColor ? 0xffffff : 0xb9b6ae, sizeAttenuation: true }));
}

/** Points spread evenly over the scanned surfaces (by area), with their normals; vertices for a bare point cloud. */
export function samplePoints(root, max = 40000) {
  root.updateMatrixWorld(true);
  const tris = []; // per mesh: { pos, index, matrix, cum }
  const clouds = [];
  let total = 0;
  const a = new THREE.Vector3(), b = new THREE.Vector3(), c = new THREE.Vector3(), ab = new THREE.Vector3(), ac = new THREE.Vector3();
  root.traverse((o) => {
    const pos = o.geometry?.attributes?.position;
    if (!pos) return;
    if (o.isPoints) { clouds.push({ pos, matrix: o.matrixWorld }); return; }
    if (o.isLineSegments) { clouds.push({ pos: alongLines(pos), matrix: o.matrixWorld }); return; } // a wireframe drawing
    if (!o.isMesh) return;
    const index = o.geometry.index;
    const n = index ? index.count / 3 : pos.count / 3;
    const cum = new Float64Array(n);
    for (let t = 0; t < n; t++) {
      const i0 = index ? index.getX(3 * t) : 3 * t, i1 = index ? index.getX(3 * t + 1) : 3 * t + 1, i2 = index ? index.getX(3 * t + 2) : 3 * t + 2;
      a.fromBufferAttribute(pos, i0).applyMatrix4(o.matrixWorld);
      b.fromBufferAttribute(pos, i1).applyMatrix4(o.matrixWorld);
      c.fromBufferAttribute(pos, i2).applyMatrix4(o.matrixWorld);
      total += ab.subVectors(b, a).cross(ac.subVectors(c, a)).length() / 2;
      cum[t] = total;
    }
    if (n) tris.push({ pos, index, matrix: o.matrixWorld, cum, start: total - cum[n - 1] });
  });
  const box = new THREE.Box3().setFromObject(root);
  if (tris.length && total > 0) {
    const pts = new Float32Array(max * 3), nrm = new Float32Array(max * 3);
    for (let k = 0; k < max; k++) {
      const r = Math.random() * total;
      const m = tris.find((q) => r <= q.cum[q.cum.length - 1]) || tris.at(-1);
      let lo = 0, hi = m.cum.length - 1; // binary search the triangle
      while (lo < hi) { const mid = (lo + hi) >> 1; if (m.cum[mid] < r) lo = mid + 1; else hi = mid; }
      const t = lo;
      const i0 = m.index ? m.index.getX(3 * t) : 3 * t, i1 = m.index ? m.index.getX(3 * t + 1) : 3 * t + 1, i2 = m.index ? m.index.getX(3 * t + 2) : 3 * t + 2;
      a.fromBufferAttribute(m.pos, i0).applyMatrix4(m.matrix);
      b.fromBufferAttribute(m.pos, i1).applyMatrix4(m.matrix);
      c.fromBufferAttribute(m.pos, i2).applyMatrix4(m.matrix);
      let u = Math.random(), v = Math.random();
      if (u + v > 1) { u = 1 - u; v = 1 - v; }
      const p = a.clone().addScaledVector(ab.subVectors(b, a), u).addScaledVector(ac.subVectors(c, a), v);
      const n = ab.clone().cross(ac).normalize();
      pts.set([p.x, p.y, p.z], 3 * k);
      nrm.set([n.x, n.y, n.z], 3 * k);
    }
    return { points: pts, normals: nrm, box };
  }
  const all = [];
  for (const { pos, matrix } of clouds) for (let i = 0; i < pos.count; i++) all.push([pos, i, matrix]);
  if (all.length < 100) throw new Error('the file has no surfaces or points');
  const n = Math.min(max, all.length);
  const pts = new Float32Array(n * 3);
  for (let k = 0; k < n; k++) {
    const [pos, i, matrix] = all[Math.floor((k / n) * all.length)];
    a.fromBufferAttribute(pos, i).applyMatrix4(matrix);
    pts.set([a.x, a.y, a.z], 3 * k);
  }
  return { points: pts, normals: null, box };
}

/** Points spread along line segments (a wireframe has no surfaces to sample). */
function alongLines(pos) {
  const out = [];
  const a = new THREE.Vector3(), b = new THREE.Vector3();
  let total = 0;
  for (let i = 0; i + 1 < pos.count; i += 2) total += a.fromBufferAttribute(pos, i).distanceTo(b.fromBufferAttribute(pos, i + 1));
  const step = total / 20000 || 1;
  for (let i = 0; i + 1 < pos.count; i += 2) {
    a.fromBufferAttribute(pos, i); b.fromBufferAttribute(pos, i + 1);
    const n = Math.max(1, Math.round(a.distanceTo(b) / step));
    for (let k = 0; k <= n; k++) { const t = k / n; out.push(a.x + (b.x - a.x) * t, a.y + (b.y - a.y) * t, a.z + (b.z - a.z) * t); }
  }
  return new THREE.Float32BufferAttribute(out, 3);
}

/** A flat 2D plan has no height: ask how tall to raise its lines. Resolves {height (m), metersPerUnit} or null. */
async function askHeight({ name, metersPerUnit, size }) {
  const span = Math.max(size.x, size.y);
  const guess = metersPerUnit || (span > 1500 ? 0.001 : span > 150 ? 0.01 : 1);
  const unitSel = h('select', { attrs: { 'aria-label': 'Drawing units' } }, UNITS.map(([v, t]) => h('option', { value: v, text: t })));
  unitSel.value = String(UNITS.find(([v]) => Math.abs(Number(v) - guess) < 1e-9)?.[0] || '1');
  const input = h('input', { type: 'text', class: 'num', value: String(round(toDisp(8), 1)), attrs: { 'aria-label': 'Wall height', inputmode: 'decimal', autofocus: '' } });
  const extent = h('p', { class: 'hint' });
  const showExtent = () => { const u = Number(unitSel.value); extent.textContent = `In these units the plan measures ${lenNum(size.x * u)} × ${len(size.y * u)}.`; };
  unitSel.onchange = showExtent;
  showExtent();
  const r = await openDialog({
    title: 'Raise the plan into walls',
    body: h('div', {},
      h('p', { text: `${name} is a flat 2D drawing. Its lines can be raised into walls so the room can be measured.` }),
      h('label', { class: 'field' }, h('span', { text: 'Drawing units' }), unitSel),
      h('label', { class: 'field' }, h('span', { text: 'Wall height' }), h('span', { class: 'input-wrap' }, input, h('em', { text: lenUnit() }))),
      extent),
    actions: [{ label: 'Cancel' }, { label: 'Raise walls', kind: 'primary' }],
  });
  const height = fromDisp(parseFloat(input.value.replace(',', '.')));
  if (r !== 'Raise walls' || !(height > 0.5)) return null;
  return { height: Math.min(height, 300), metersPerUnit: Number(unitSel.value) };
}

/** Native file coordinates -> z-up metres. glTF and most OBJ files are y-up. */
function toZup(arr, up, scale, isNormal = false) {
  if (!arr) return null;
  const out = new Float32Array(arr.length);
  const s = isNormal ? 1 : scale;
  for (let i = 0; i < arr.length; i += 3) {
    if (up === 'y') { out[i] = arr[i] * s; out[i + 1] = -arr[i + 2] * s; out[i + 2] = arr[i + 1] * s; } else { out[i] = arr[i] * s; out[i + 1] = arr[i + 1] * s; out[i + 2] = arr[i + 2] * s; }
  }
  return out;
}

/** The 4x4 matrix that places a scan in the three.js scene (room x, y, z drawn at three.js y, z, x). */
export function scanMatrix(ref) {
  const s = ref.unit_scale;
  const U = ref.up === 'y' ? new THREE.Matrix4().set(1, 0, 0, 0, 0, 0, -1, 0, 0, 1, 0, 0, 0, 0, 0, 1) : new THREE.Matrix4();
  const S = new THREE.Matrix4().makeScale(s, s, s);
  const R = new THREE.Matrix4().makeRotationZ((ref.yaw_deg * Math.PI) / 180);
  const T = new THREE.Matrix4().makeTranslation(ref.t[0], ref.t[1], ref.t[2]);
  const A = new THREE.Matrix4().set(0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1);
  return A.multiply(T).multiply(R).multiply(S).multiply(U);
}

const loaded = new Map(); // scan id -> Promise<Object3D>
/** The scan as a ready-to-add object, placed with the reference's transform. Cached per file. */
export async function scanObject(ref) {
  if (!loaded.has(ref.id)) {
    loaded.set(ref.id, (async () => {
      const res = await fetch(`/api/scans/${ref.id}.${ref.ext}`);
      if (!res.ok) throw new Error((await res.json().catch(() => ({}))).detail || 'scan file missing');
      const obj = await parseScan(await res.arrayBuffer(), ref.ext);
      const holder = new THREE.Group();
      holder.add(obj);
      holder.matrixAutoUpdate = false;
      obj.traverse((o) => { if (o.isMesh) o.receiveShadow = true; });
      return holder;
    })().catch((e) => { loaded.delete(ref.id); throw e; }));
  }
  const holder = await loaded.get(ref.id);
  holder.matrix.copy(scanMatrix(ref));
  holder.matrixWorldNeedsUpdate = true;
  holder.visible = ref.visible !== false;
  holder.traverse((o) => {
    if (!o.material) return;
    (Array.isArray(o.material) ? o.material : [o.material]).forEach((m) => {
      m.transparent = ref.opacity < 1;
      m.opacity = ref.opacity;
      m.depthWrite = ref.opacity >= 1;
    });
  });
  return holder;
}

// ------------------------------------------------------------- the import
const extOf = (f) => (f.name.split('.').pop() || '').toLowerCase();

/** Import from a set of chosen files: one scan or model, or a .gltf with its buffers and textures. */
export async function importFiles(files) {
  const gltf = files.find((f) => extOf(f) === 'gltf');
  if (gltf) {
    let packed;
    try { packed = await (await import('./formats.js')).gltfToGlb(gltf, files); } catch (e) { toast(`The glTF could not be read: ${e.message}`, { error: true }); return; }
    if (packed.missing.length) {
      toast(`Some textures were not among the chosen files and are left plain: ${packed.missing.slice(0, 3).join(', ')}${packed.missing.length > 3 ? '…' : ''}.`, { ms: 8000 });
    }
    return importScan(packed.file);
  }
  const main = files.find((f) => SCAN_EXTS.includes(extOf(f))) || files[0];
  if (main) return importScan(main);
}

export async function importScan(file, hints = {}) {
  const ext = extOf(file);
  if (!SCAN_EXTS.includes(ext)) {
    toast(NOT_READABLE[ext] || (GLTF_PARTS.includes(ext) ? 'Choose the .gltf file together with its .bin and texture files.'
      : 'Choose a scan (GLB, glTF, USDZ, OBJ, PLY) or a model (MVR, 3DS, FBX, COLLADA .dae, STL).'), { error: true });
    return;
  }
  if (file.size > MAX_BYTES) { toast(`That scan is ${Math.round(file.size / 1e6)} MB; the limit is 300 MB. Export it at a lower detail level.`, { error: true }); return; }
  if (CONVERT.has(ext)) {
    const working = toast(`Converting ${file.name}…`, { ms: 600000 });
    let res;
    try { res = await (await import('./formats.js')).convertModel(file, ext, { askHeight }); } catch (e) {
      working.remove();
      toast(friendly(e), { error: true });
      return;
    } finally { working.remove(); }
    if (!res) return;
    if (res.note) toast(res.note, { ms: 8000 });
    return importScan(res.file, res.hints);
  }
  const busy = toast(`Reading ${file.name}…`, { ms: 60000 });
  let raw, buffer, obj;
  try {
    buffer = await file.arrayBuffer();
    obj = await parseScan(buffer, ext);
    raw = samplePoints(obj);
  } catch (e) {
    busy.remove();
    toast(friendly(e), { error: true });
    return;
  } finally { busy.remove(); }
  const mvr = obj?.userData?.mvr;
  if (mvr) {
    toast(`MVR: ${mvr.instances} pieces of geometry from ${mvr.files} file${mvr.files === 1 ? '' : 's'}`
      + (mvr.fixtures ? `; ${mvr.fixtures} fixtures left out (MVR keeps their shapes in GDTF files)` : '')
      + (mvr.missing.length ? `; not in the archive: ${mvr.missing.slice(0, 3).join(', ')}${mvr.missing.length > 3 ? '…' : ''}` : '') + '.', { ms: 8000 });
  }

  const size = new THREE.Vector3();
  raw.box.getSize(size);
  const span = Math.max(size.x, size.y, size.z);
  // FBX states its unit (centimetres per file unit); MVR and COLLADA arrive in metres; otherwise guess from the size.
  const fbxUnit = ext === 'fbx' ? Number(obj?.userData?.unitScaleFactor) / 100 : NaN;
  const known = UNITS.find(([v]) => Math.abs(Number(v) - fbxUnit) < 1e-6);
  const choice = {
    up: hints.up || (Z_UP.has(ext) || (ext === 'ply' && size.z < size.y) ? 'z' : 'y'),
    unit: hints.unit || (ext === 'mvr' || ext === 'dae' ? '1' : known ? known[0] : span > 1500 ? '0.001' : span > 150 ? '0.01' : '1'),
    side: null, fit: null, dims: null, error: null,
  };

  const preview = h('canvas', { class: 'scan-preview', width: 520, height: 300, attrs: { 'aria-label': 'Top view of the scan with the fitted room. The stage is on the left.' } });
  const status = h('p', { class: 'hint', attrs: { 'aria-live': 'polite' } });
  const dimInputs = {};
  const dimField = (key, label) => {
    // text, not type=number: the browser would show the Windows decimal comma, and a typed comma would be misread
    const input = h('input', { type: 'text', class: 'num', attrs: { 'aria-label': label, inputmode: 'decimal', autocomplete: 'off' },
      onchange: () => {
        const v = fromDisp(parseFloat(input.value.replace(',', '.')));
        choice.dims[key] = clamp(Number.isFinite(v) ? v : choice.dims[key], key === 'height' ? 1.6 : 1.1, 300);
        input.value = String(round(toDisp(choice.dims[key]), 2));
        draw();
      } });
    dimInputs[key] = input;
    return h('label', { class: 'field' }, h('span', { text: label }), h('span', { class: 'input-wrap' }, input, h('em', { text: lenUnit() })));
  };
  const unitSel = h('select', { attrs: { 'aria-label': 'Scan units' }, onchange: () => { choice.unit = unitSel.value; refit(); } },
    UNITS.map(([v, t]) => h('option', { value: v, text: t })));
  unitSel.value = choice.unit;
  const upSel = h('select', { attrs: { 'aria-label': 'Up axis' }, onchange: () => { choice.up = upSel.value; refit(); } },
    h('option', { value: 'y', text: 'Y is up (GLB, most apps)' }), h('option', { value: 'z', text: 'Z is up (CAD, point clouds)' }));
  upSel.value = choice.up;
  const turnBtn = h('button', { class: 'btn small', onclick: () => { choice.side = SIDES[(SIDES.indexOf(choice.side) + 1) % 4]; useOption(); } }, 'Use the next wall as the stage');

  async function refit() {
    status.textContent = 'Measuring the room…';
    choice.error = null;
    try {
      const pts = toZup(raw.points, choice.up, Number(choice.unit));
      const nrm = toZup(raw.normals, choice.up, 1, true);
      choice.fit = await api.scanFit(Array.from(pts, (v) => round(v, 3)), nrm && Array.from(nrm, (v) => round(v, 3)));
      choice.side ??= choice.fit.suggested;
      choice.pts = pts;
      useOption();
    } catch (e) {
      choice.fit = null;
      choice.error = e.message;
      status.textContent = e.message;
      draw();
    }
  }

  function useOption() {
    const opt = choice.fit.options[choice.side];
    choice.dims = { ...opt.room };
    for (const k of ['length', 'width', 'height']) dimInputs[k].value = String(round(toDisp(opt.room[k]), 2));
    const how = choice.fit.method === 'surfaces' ? 'from the wall surfaces' : 'from the outline of the points';
    status.textContent = `Measured ${how}: ${lenNum(opt.room.length)} × ${len(opt.room.width)}, ${len(opt.room.height)} high. The stage is on the left.`
      + (choice.fit.note ? ` ${choice.fit.note}; check the sizes.` : '');
    draw();
  }

  function draw() {
    const ctx = preview.getContext('2d');
    const W = preview.width, H = preview.height;
    ctx.fillStyle = cssVar('--page');
    ctx.fillRect(0, 0, W, H);
    if (!choice.fit) return;
    const opt = choice.fit.options[choice.side];
    const d = choice.dims;
    const pad = 26;
    const sc = Math.min((W - 2 * pad) / d.length, (H - 2 * pad) / d.width);
    const ox = (W - d.length * sc) / 2, oy = (H - d.width * sc) / 2;
    const a = (opt.yaw_deg * Math.PI) / 180, ca = Math.cos(a), sa = Math.sin(a);
    ctx.fillStyle = cssVar('--ink-2');
    ctx.globalAlpha = 0.5;
    const p = choice.pts;
    for (let i = 0; i < p.length; i += 9) { // every third point is plenty for a preview
      const x = ca * p[i] - sa * p[i + 1] + opt.t[0], y = sa * p[i] + ca * p[i + 1] + opt.t[1], z = p[i + 2] + opt.t[2];
      if (z < 0.08 * d.height || z > 0.9 * d.height) continue; // walls and furnishings, not floor or ceiling
      ctx.fillRect(ox + y * sc, oy + x * sc, 1.4, 1.4);
    }
    ctx.globalAlpha = 1;
    ctx.strokeStyle = cssVar('--accent');
    ctx.lineWidth = 1.5;
    ctx.strokeRect(ox, oy, d.length * sc, d.width * sc);
    ctx.fillStyle = cssVar('--accent');
    ctx.fillRect(ox - 5, oy, 5, d.width * sc);
    ctx.save();
    ctx.translate(ox - 12, oy + (d.width * sc) / 2);
    ctx.rotate(-Math.PI / 2);
    ctx.font = '600 10px system-ui, sans-serif';
    ctx.textAlign = 'center';
    ctx.fillText('STAGE', 0, 0);
    ctx.restore();
  }

  const body = h('div', { class: 'scan-dialog' },
    h('div', { class: 'scan-left' }, preview, h('div', { style: { display: 'flex', gap: '8px', alignItems: 'center', marginTop: '8px' } }, turnBtn), status),
    h('div', { class: 'scan-right' },
      h('div', { class: 'grid2' },
        h('label', { class: 'field wide' }, h('span', { text: 'The scan is in' }), unitSel),
        h('label', { class: 'field wide' }, h('span', { text: 'Orientation' }), upSel)),
      h('div', { class: 'grid3', style: { marginTop: '12px' } }, dimField('length', 'Stage to back'), dimField('width', 'Width'), dimField('height', 'Height')),
      h('p', { class: 'hint', text: 'The sound prediction uses a box of this size. The scan itself is shown in the 3D view so you can place speakers against the real room.' }),
      h('p', { class: 'hint', text: 'Speakers, mics and the audience area keep their positions relative to the room.' })));

  refit();
  await openDialog({
    title: `Use ${file.name} as the room`, body, wide: true,
    actions: [{ label: 'Cancel' }, {
      label: 'Use this room', kind: 'primary',
      run: async () => {
        if (!choice.fit) { status.textContent = choice.error || 'Still measuring…'; return false; }
        status.textContent = 'Saving the scan…';
        try {
          const res = await fetch(`/api/scans?ext=${ext}`, { method: 'POST', body: buffer, headers: { 'Content-Type': 'application/octet-stream' } });
          if (!res.ok) throw new Error((await res.json().catch(() => ({}))).detail || res.statusText);
          const stored = await res.json();
          applyScan(file.name, ext, stored.id, choice);
        } catch (e) { status.textContent = `Could not save the scan: ${e.message}`; return false; }
        return true;
      },
    }],
  });
}

function applyScan(name, ext, id, choice) {
  const opt = choice.fit.options[choice.side];
  const d = choice.dims;
  mutate((p) => {
    const r = p.room;
    const fx = d.width / r.width, fy = d.length / r.length, fz = d.height / r.height;
    const a = p.audience;
    Object.assign(a, { x0: round(a.x0 * fx, 2), x1: round(a.x1 * fx, 2), y0: round(a.y0 * fy, 2), y1: round(a.y1 * fy, 2) });
    a.ear_height = Math.min(a.ear_height, d.height - 0.2);
    for (const o of [...p.speakers, ...p.mics]) {
      o.x = round(clamp(o.x * fx, 0.1, d.width - 0.1), 2);
      o.y = round(clamp(o.y * fy, 0.1, d.length - 0.1), 2);
      o.z = round(clamp(o.z * Math.min(1, fz), 0.1, d.height - 0.1), 2);
    }
    Object.assign(r, { width: d.width, length: d.length, height: d.height });
    p.scan = { id, ext, name: name.replace(/\.[^.]+$/, ''), up: choice.up, unit_scale: Number(choice.unit),
      yaw_deg: opt.yaw_deg, t: opt.t, visible: true, opacity: 1 };
  }, { structural: true, source: 'scan' });
  setUI({ view: '3d' });
  toast(`Room set from the scan: ${lenNum(d.length)} × ${lenNum(d.width)} × ${len(d.height)}.`);
}

/** Pick a file and import it. */
// ------------------------------------------------- an unpacked MVR folder
const CRC = (() => {
  const t = new Uint32Array(256);
  for (let n = 0; n < 256; n++) { let c = n; for (let k = 0; k < 8; k++) c = c & 1 ? 0xedb88320 ^ (c >>> 1) : c >>> 1; t[n] = c >>> 0; }
  return t;
})();
const crc32 = (b) => { let c = 0xffffffff; for (let i = 0; i < b.length; i++) c = CRC[(c ^ b[i]) & 0xff] ^ (c >>> 8); return (c ^ 0xffffffff) >>> 0; };

/** Pack the files of an unpacked .mvr folder back into one archive (STORE, every file at the root, as MVR requires). */
export async function folderToMvr(files) {
  const xml = files.find((f) => f.name.toLowerCase() === 'generalscenedescription.xml');
  if (!xml) throw new Error('that folder has no GeneralSceneDescription.xml, so it is not an unpacked MVR');
  const total = files.reduce((n, f) => n + f.size, 0);
  if (total > MAX_BYTES) throw new Error(`the folder holds ${Math.round(total / 1e6)} MB; the limit is 300 MB`);
  if (files.length > 20000) throw new Error('the folder holds too many files');
  const enc = new TextEncoder();
  const parts = [], central = [];
  let offset = 0;
  const seen = new Set();
  for (const f of files) {
    const key = f.name.toLowerCase();
    if (seen.has(key)) continue; // MVR keeps one flat folder; a duplicate name in a subfolder is not part of it
    seen.add(key);
    const data = new Uint8Array(await f.arrayBuffer());
    const name = enc.encode(f.name);
    const crc = crc32(data);
    const head = new DataView(new ArrayBuffer(30));
    head.setUint32(0, 0x04034b50, true); head.setUint16(4, 20, true); head.setUint16(6, 0x0800, true);
    head.setUint32(14, crc, true); head.setUint32(18, data.length, true); head.setUint32(22, data.length, true);
    head.setUint16(26, name.length, true);
    const cd = new DataView(new ArrayBuffer(46));
    cd.setUint32(0, 0x02014b50, true); cd.setUint16(4, 20, true); cd.setUint16(6, 20, true); cd.setUint16(8, 0x0800, true);
    cd.setUint32(16, crc, true); cd.setUint32(20, data.length, true); cd.setUint32(24, data.length, true);
    cd.setUint16(28, name.length, true); cd.setUint32(42, offset, true);
    parts.push(new Uint8Array(head.buffer), name, data);
    central.push(new Uint8Array(cd.buffer), name);
    offset += 30 + name.length + data.length;
  }
  const cdSize = central.reduce((n, b) => n + b.length, 0);
  const end = new DataView(new ArrayBuffer(22));
  end.setUint32(0, 0x06054b50, true); end.setUint16(8, seen.size, true); end.setUint16(10, seen.size, true);
  end.setUint32(12, cdSize, true); end.setUint32(16, offset, true);
  const folder = (files[0].webkitRelativePath || '').split('/')[0] || 'venue';
  return new File([...parts, ...central, new Uint8Array(end.buffer)], folder.toLowerCase().endsWith('.mvr') ? folder : `${folder}.mvr`);
}

/** Pick a folder: an unpacked MVR (Windows sometimes extracts .mvr files) or a glTF with its files, and import it. */
export function chooseMvrFolder() {
  const input = h('input', { type: 'file', style: { display: 'none' }, attrs: { webkitdirectory: '', multiple: '' },
    onchange: async () => {
      const files = [...(input.files || [])];
      input.remove();
      if (!files.length) return;
      try {
        if (files.some((f) => f.name.toLowerCase() === 'generalscenedescription.xml')) { importScan(await folderToMvr(files)); return; }
        const gltfs = files.filter((f) => extOf(f) === 'gltf');
        if (gltfs.length) {
          // the top-most .gltf is the model; others deeper down are usually variants or samples
          gltfs.sort((a, b) => a.webkitRelativePath.split('/').length - b.webkitRelativePath.split('/').length);
          importFiles([gltfs[0], ...files.filter((f) => extOf(f) !== 'gltf')]);
          return;
        }
        throw new Error('it holds neither an unpacked MVR (GeneralSceneDescription.xml) nor a .gltf');
      } catch (e) { toast(`Could not read the folder: ${e.message}`, { error: true }); }
    } });
  document.body.append(input);
  input.click();
}

export function chooseScanFile() {
  const input = h('input', { type: 'file', accept: [...SCAN_EXTS, ...GLTF_PARTS].map((e) => `.${e}`).join(','), style: { display: 'none' },
    attrs: { multiple: '' }, // a .gltf is chosen together with its .bin and textures
    onchange: () => { const files = [...(input.files || [])]; input.remove(); if (files.length) importFiles(files); } });
  document.body.append(input);
  input.click();
}

