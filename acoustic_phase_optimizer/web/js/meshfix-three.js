// Repairs every mesh of an imported three.js model (see meshfix.js), keeping the original geometry so the repair can be
// switched off again. Vertex buffers are shared between the two versions; only the triangle list (and, where faces were
// turned, the normals) differ.

import * as THREE from 'three';
import { repair, emptyReport, changed, describe } from './meshfix.js';

const WORKER_TRIANGLES = 150000; // above this, a mesh is repaired in a worker so the page stays responsive

function positionsOf(attr) {
  if (!attr.isInterleavedBufferAttribute && attr.itemSize === 3 && attr.array instanceof Float32Array && !attr.normalized) return attr.array;
  const out = new Float32Array(attr.count * 3);
  for (let i = 0; i < attr.count; i++) { out[3 * i] = attr.getX(i); out[3 * i + 1] = attr.getY(i); out[3 * i + 2] = attr.getZ(i); }
  return out;
}

function inWorker(positions, index) {
  return new Promise((resolve, reject) => {
    const w = new Worker(new URL('./meshfix-worker.js', import.meta.url), { type: 'module' });
    w.onmessage = (e) => { w.terminate(); if (e.data.ok) resolve(e.data); else reject(new Error(e.data.error)); };
    w.onerror = (e) => { w.terminate(); reject(new Error(e.message || 'the repair worker failed')); };
    // copies, so the page keeps its own buffers
    const p = positions.slice(), i = index ? Uint32Array.from(index) : null;
    w.postMessage({ positions: p, index: i }, i ? [p.buffer, i.buffer] : [p.buffer]);
  });
}

/** Per-triangle material group of the input, or null when the geometry has no groups. */
function groupOf(geo, triangles) {
  if (!geo.groups?.length) return null;
  const tag = new Int32Array(triangles).fill(-1);
  for (const g of geo.groups) {
    const from = Math.floor(g.start / 3), to = Math.min(triangles, Math.floor((g.start + g.count) / 3));
    for (let t = from; t < to; t++) tag[t] = g.materialIndex ?? 0;
  }
  return tag;
}

function rebuild(geo, res) {
  const fixed = new THREE.BufferGeometry();
  for (const [name, attr] of Object.entries(geo.attributes)) fixed.setAttribute(name, attr);
  if (res.report.nanRemoved) {
    // the broken corners are no longer used by any face, but three.js measures bounds over every vertex: give the
    // repaired copy a harmless value there (a corner of the first face) and leave the original untouched
    const pos = fixed.attributes.position.clone();
    const k = res.index.length ? res.index[0] : 0;
    const safe = [pos.getX(k), pos.getY(k), pos.getZ(k)].map((x) => (Number.isFinite(x) ? x : 0));
    for (let i = 0; i < pos.count; i++) {
      if (!Number.isFinite(pos.getX(i)) || !Number.isFinite(pos.getY(i)) || !Number.isFinite(pos.getZ(i))) pos.setXYZ(i, ...safe);
    }
    fixed.setAttribute('position', pos);
  }
  fixed.morphAttributes = geo.morphAttributes;
  fixed.morphTargetsRelative = geo.morphTargetsRelative;
  const tris = res.index.length / 3;
  const tag = groupOf(geo, geo.index ? geo.index.count / 3 : geo.attributes.position.count / 3);
  let index = res.index;
  if (tag) {
    // keep material groups: order the triangles by group, then describe each run
    const order = Array.from({ length: tris }, (_, t) => t).sort((a, b) => (tag[res.triSource[a]] - tag[res.triSource[b]]) || (a - b));
    index = new Uint32Array(res.index.length);
    order.forEach((t, k) => index.set(res.index.subarray(3 * t, 3 * t + 3), 3 * k));
    let start = 0;
    for (let k = 1; k <= tris; k++) {
      const prev = tag[res.triSource[order[k - 1]]];
      if (k === tris || tag[res.triSource[order[k]]] !== prev) {
        if (prev >= 0) fixed.addGroup(3 * start, 3 * (k - start), prev);
        start = k;
      }
    }
  }
  fixed.setIndex(new THREE.BufferAttribute(index, 1));
  if (res.report.flipped || res.report.holesFilled || !geo.attributes.normal) {
    fixed.deleteAttribute('normal'); // a new attribute: the original geometry keeps its own normals
    fixed.computeVertexNormals();
  }
  fixed.computeBoundingBox();
  fixed.computeBoundingSphere();
  return fixed;
}

/**
 * Repair every triangle mesh in `root`. Returns { report, changed, summary, use(on) }; use(false) puts the original
 * geometry back, use(true) the repaired one. Repaired is the starting state.
 */
export async function repairObject(root) {
  const report = emptyReport();
  const items = [];
  const meshes = [];
  root.traverse((o) => { if (o.isMesh && !o.isSkinnedMesh && o.geometry?.attributes?.position) meshes.push(o); });
  const t0 = performance.now();
  for (const mesh of meshes) {
    const geo = mesh.geometry;
    const positions = positionsOf(geo.attributes.position);
    const index = geo.index ? geo.index.array : null;
    const tris = index ? index.length / 3 : positions.length / 9;
    if (!tris) continue;
    const res = tris > WORKER_TRIANGLES && typeof Worker !== 'undefined' ? await inWorker(positions, index) : repair(positions, index);
    for (const k of Object.keys(report)) report[k] += res.report[k];
    if (!changed(res.report)) continue;
    items.push({ mesh, original: geo, fixed: rebuild(geo, res) });
  }
  report.ms = Math.round(performance.now() - t0);
  const use = (on) => { for (const it of items) it.mesh.geometry = on ? it.fixed : it.original; };
  use(true);
  return { report, changed: items.length > 0, summary: describe(report), use };
}
