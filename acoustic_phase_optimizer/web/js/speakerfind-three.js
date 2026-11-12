// The three.js side of the speaker finder: turn the placed scan into the plain objects speakerfind.js judges.
// Every mesh with a name gives one object (its node path and material names); every mesh is also split into
// its connected parts, which give unnamed box candidates; MVR fixture markers give fixtures. All in room
// metres, z up (three.js x, y, z shows room y, z, x).

import * as THREE from 'three';

const MAX_TRIANGLES = 3_000_000; // beyond this the shape search is skipped (names and fixtures still count)
const MAX_OBJECTS = 20000;
const MAX_PART = 3; // m: a part bigger than this is a wall, floor or the room, never a speaker

const toRoom = (v) => [v.z, v.x, v.y];

function nameOf(o, root) {
  const names = [];
  for (let n = o; n && n !== root; n = n.parent) if (n.name) names.unshift(n.name);
  const mats = (Array.isArray(o.material) ? o.material : [o.material]).map((m) => m?.name).filter(Boolean);
  return [...new Set([...names, ...mats])].join(' ').slice(0, 300);
}

/** Connected parts of one mesh (triangles sharing a vertex position), each as {lo, hi, area}. */
function parts(pos, index, n, out) {
  const keyOf = new Map();
  const vid = new Int32Array(pos.length / 3);
  for (let i = 0; i < vid.length; i++) { // weld by position (0.1 mm): split vertices for normals still join
    const k = `${Math.round(pos[3 * i] * 1e4)},${Math.round(pos[3 * i + 1] * 1e4)},${Math.round(pos[3 * i + 2] * 1e4)}`;
    let v = keyOf.get(k);
    if (v === undefined) { v = keyOf.size; keyOf.set(k, v); }
    vid[i] = v;
  }
  const parent = new Int32Array(keyOf.size).map((_, i) => i);
  const find = (a) => { while (parent[a] !== a) { parent[a] = parent[parent[a]]; a = parent[a]; } return a; };
  const tri = (t, k) => vid[index ? index[3 * t + k] : 3 * t + k];
  for (let t = 0; t < n; t++) {
    const a = find(tri(t, 0)), b = find(tri(t, 1)), c = find(tri(t, 2));
    parent[b] = a; parent[find(c)] = a;
  }
  const groups = new Map();
  for (let t = 0; t < n; t++) {
    const r = find(tri(t, 0));
    let g = groups.get(r);
    if (!g) { g = { lo: [Infinity, Infinity, Infinity], hi: [-Infinity, -Infinity, -Infinity], area: 0, tris: 0 }; groups.set(r, g); }
    const i0 = index ? index[3 * t] : 3 * t, i1 = index ? index[3 * t + 1] : 3 * t + 1, i2 = index ? index[3 * t + 2] : 3 * t + 2;
    for (const i of [i0, i1, i2]) {
      for (let k = 0; k < 3; k++) { const v = pos[3 * i + k]; if (v < g.lo[k]) g.lo[k] = v; if (v > g.hi[k]) g.hi[k] = v; }
    }
    const ux = pos[3 * i1] - pos[3 * i0], uy = pos[3 * i1 + 1] - pos[3 * i0 + 1], uz = pos[3 * i1 + 2] - pos[3 * i0 + 2];
    const wx = pos[3 * i2] - pos[3 * i0], wy = pos[3 * i2 + 1] - pos[3 * i0 + 1], wz = pos[3 * i2 + 2] - pos[3 * i0 + 2];
    g.area += 0.5 * Math.hypot(uy * wz - uz * wy, uz * wx - ux * wz, ux * wy - uy * wx);
    g.tris++;
  }
  for (const g of groups.values()) {
    if (out.length >= MAX_OBJECTS) return;
    const s = [0, 1, 2].map((k) => g.hi[k] - g.lo[k]);
    if (g.tris < 6 || Math.max(...s) > MAX_PART || Math.min(...s) < 0.03) continue;
    out.push({ name: '', lo: g.lo, hi: g.hi, area: g.area });
  }
}

/**
 * Objects and fixtures in a placed scan (the holder from scan.js:scanObject).
 * Returns {objects: [{name, lo, hi, area}], fixtures: [{name, spec, pos, fwd}], skippedShapes}.
 */
export function extractFound(root) {
  root.updateMatrixWorld(true);
  const objects = [], fixtures = [];
  let triangles = 0;
  root.traverse((o) => { const p = o.isMesh && o.geometry?.attributes?.position; if (p) triangles += o.geometry.index ? o.geometry.index.count / 3 : p.count / 3; });
  const shapes = triangles <= MAX_TRIANGLES;
  const w = new THREE.Vector3(), d = new THREE.Vector3();
  root.traverse((o) => {
    if (o.userData?.fixture) {
      w.setFromMatrixPosition(o.matrixWorld);
      d.set(0, 0, -1).transformDirection(o.matrixWorld);
      fixtures.push({ ...o.userData.fixture, pos: toRoom(w), fwd: toRoom(d) });
      return;
    }
    const g = o.geometry, p = g?.attributes?.position;
    if (!o.isMesh || !p || objects.length >= MAX_OBJECTS) return;
    const pos = new Float32Array(p.count * 3);
    for (let i = 0; i < p.count; i++) {
      w.fromBufferAttribute(p, i).applyMatrix4(o.matrixWorld);
      pos[3 * i] = w.z; pos[3 * i + 1] = w.x; pos[3 * i + 2] = w.y;
    }
    const index = g.index ? g.index.array : null;
    const n = Math.floor((index ? index.length : p.count) / 3);
    const name = nameOf(o, root);
    if (name) {
      const lo = [Infinity, Infinity, Infinity], hi = [-Infinity, -Infinity, -Infinity];
      for (let i = 0; i < pos.length; i++) { const k = i % 3; if (pos[i] < lo[k]) lo[k] = pos[i]; if (pos[i] > hi[k]) hi[k] = pos[i]; }
      if (Number.isFinite(lo[0])) objects.push({ name, lo, hi, area: 0 });
    }
    if (shapes && n) parts(pos, index, n, objects);
  });
  return { objects, fixtures, skippedShapes: !shapes };
}
