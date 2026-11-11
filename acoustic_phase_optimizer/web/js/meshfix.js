// Mesh repair for imported models: pure functions on plain typed arrays (no three.js), so it runs in the page, in a worker
// and under `node --test`. It never moves or deletes vertices: it only rewrites the triangle list, so every vertex
// attribute (normals, UVs, colours) stays valid and the original file is untouched.
//
// What it does, in order:
//   1. drops triangles with non-finite corners or indices outside the vertex array
//   2. joins vertices that sit on the same spot (for the analysis; the vertex buffer keeps its seams)
//   3. drops degenerate (zero-area) and duplicate triangles
//   4. makes the winding consistent across every connected surface (flips the minority)
//   5. optionally (opts.outward) turns closed shells so their faces point outward
//   6. closes small holes (cracks and pinholes); openings larger than holeMaxRel of the model stay open, since doors,
//      windows and stage openings are real
//   7. counts tiny loose pieces (reported, not removed)
// Deterministic: the same input always gives the same output, so a stored model is repaired the same way on reload.

export const DEFAULTS = Object.freeze({
  weldRel: 1e-6, // join tolerance, as a fraction of the model's diagonal
  holeMaxEdges: 200, // holes with more edges than this stay open
  holeMaxRel: 0.01, // holes wider than this fraction of the diagonal stay open (a 30 m hall: 0.3 m)
  islandRel: 0.0005, // pieces with less than this fraction of the total area count as loose bits
  // Turn closed shells so their faces point outward. Off by default: a venue model is usually a room seen from inside,
  // with faces pointing in on purpose, so only consistency between neighbouring faces is enforced.
  outward: false,
});

export function emptyReport() {
  return { triangles: 0, nanRemoved: 0, badIndex: 0, welded: 0, degenerate: 0, duplicates: 0, flipped: 0, inverted: 0,
    holesFilled: 0, holesLeft: 0, openEdges: 0, nonManifold: 0, islands: 0, ms: 0 };
}

/** Fields of a report that mean the triangle list was changed. */
export const CHANGES = ['nanRemoved', 'badIndex', 'degenerate', 'duplicates', 'flipped', 'inverted', 'holesFilled'];
export const changed = (r) => CHANGES.some((k) => r[k] > 0);

const now = () => (typeof performance !== 'undefined' ? performance.now() : Date.now());

/**
 * Repair one triangle mesh.
 * @param {Float32Array|number[]} positions xyz per vertex
 * @param {Uint32Array|Uint16Array|number[]|null} index three vertex indices per triangle; null for a triangle soup
 * @returns {{index: Uint32Array, triSource: Int32Array, report: object}} triSource[i] is the input triangle that output
 *   triangle i came from (for a hole patch: the triangle on the hole's rim), so materials and groups can follow.
 */
export function repair(positions, index, opts = {}) {
  const t0 = now();
  const o = { ...DEFAULTS, ...opts };
  const P = positions;
  const nv = Math.floor(P.length / 3);
  const report = emptyReport();
  const src = index || (() => { const a = new Uint32Array(nv - (nv % 3)); for (let i = 0; i < a.length; i++) a[i] = i; return a; })();
  const nt = Math.floor(src.length / 3);
  report.triangles = nt;

  // ------------------------------------------------------------ 1. finite vertices and the size of the model
  const finite = new Uint8Array(nv);
  let minX = Infinity, minY = Infinity, minZ = Infinity, maxX = -Infinity, maxY = -Infinity, maxZ = -Infinity;
  for (let i = 0; i < nv; i++) {
    const x = P[3 * i], y = P[3 * i + 1], z = P[3 * i + 2];
    if (!Number.isFinite(x) || !Number.isFinite(y) || !Number.isFinite(z)) continue;
    finite[i] = 1;
    if (x < minX) minX = x; if (x > maxX) maxX = x;
    if (y < minY) minY = y; if (y > maxY) maxY = y;
    if (z < minZ) minZ = z; if (z > maxZ) maxZ = z;
  }
  const diag = Number.isFinite(minX) ? Math.hypot(maxX - minX, maxY - minY, maxZ - minZ) : 0;
  const cx = (minX + maxX) / 2 || 0, cy = (minY + maxY) / 2 || 0, cz = (minZ + maxZ) / 2 || 0;

  // ------------------------------------------------------------ 2. join coincident vertices
  const rep = weld(P, finite, nv, Math.max(diag * o.weldRel, 1e-12), minX, minY, minZ);
  let reps = 0, fin = 0;
  for (let i = 0; i < nv; i++) if (finite[i]) { fin++; if (rep[i] === i) reps++; }
  report.welded = fin - reps;

  // ------------------------------------------------------------ 3. drop broken, degenerate and duplicate triangles
  const areaEps = diag * diag * 1e-12;
  const T = []; // kept triangles: original vertex indices, 3 per triangle
  const Tsrc = []; // input triangle number of each kept triangle
  const seen = new Set();
  for (let t = 0; t < nt; t++) {
    const a = src[3 * t], b = src[3 * t + 1], c = src[3 * t + 2];
    if (!(a < nv && b < nv && c < nv && a >= 0 && b >= 0 && c >= 0)) { report.badIndex++; continue; }
    if (!finite[a] || !finite[b] || !finite[c]) { report.nanRemoved++; continue; }
    const ra = rep[a], rb = rep[b], rc = rep[c];
    if (ra === rb || rb === rc || ra === rc || triArea(P, ra, rb, rc) <= areaEps) { report.degenerate++; continue; }
    const s0 = Math.min(ra, rb, rc), s2 = Math.max(ra, rb, rc), s1 = ra + rb + rc - s0 - s2;
    const key = `${s0},${s1},${s2}`;
    if (seen.has(key)) { report.duplicates++; continue; }
    seen.add(key);
    T.push(a, b, c);
    Tsrc.push(t);
  }
  const m = Tsrc.length;

  // ------------------------------------------------------------ edges (on joined vertices)
  // edge key -> list of (triangle * 2 + forward), forward = the triangle walks the edge from the lower id to the higher
  const edges = new Map();
  const ekey = (u, v) => (u < v ? u * nv + v : v * nv + u);
  for (let t = 0; t < m; t++) {
    for (let k = 0; k < 3; k++) {
      const u = rep[T[3 * t + k]], v = rep[T[3 * t + ((k + 1) % 3)]];
      const key = ekey(u, v);
      const e = t * 2 + (u < v ? 1 : 0);
      const list = edges.get(key);
      if (list) list.push(e); else edges.set(key, [e]);
    }
  }
  for (const list of edges.values()) if (list.length > 2) report.nonManifold++;

  // ------------------------------------------------------------ 4. consistent winding per surface patch
  const orient = new Int8Array(m); // +1 keep, -1 flip, 0 not visited
  const patch = new Int32Array(m).fill(-1);
  const patches = []; // { tris: number[], closed: boolean }
  for (let s = 0; s < m; s++) {
    if (orient[s]) continue;
    const id = patches.length;
    const tris = [s];
    let closed = true;
    orient[s] = 1; patch[s] = id;
    for (let qi = 0; qi < tris.length; qi++) {
      const t = tris[qi];
      for (let k = 0; k < 3; k++) {
        const u = rep[T[3 * t + k]], v = rep[T[3 * t + ((k + 1) % 3)]];
        const list = edges.get(ekey(u, v));
        if (list.length !== 2) { closed = false; continue; }
        const mine = list[0] >> 1 === t ? list[0] : list[1];
        const other = mine === list[0] ? list[1] : list[0];
        const ot = other >> 1;
        if (ot === t) { closed = false; continue; }
        // walking directions (+1 / -1) after orientation must be opposite across a shared edge
        const dMine = (mine & 1 ? 1 : -1) * orient[t];
        const want = -dMine * (other & 1 ? 1 : -1);
        if (!orient[ot]) { orient[ot] = want; patch[ot] = id; tris.push(ot); }
      }
    }
    // flip the minority, so a mostly-right model keeps its original winding
    let neg = 0;
    for (const t of tris) if (orient[t] < 0) neg++;
    if (neg * 2 > tris.length) for (const t of tris) orient[t] = -orient[t];
    patches.push({ tris, closed });
  }

  // ------------------------------------------------------------ 5. closed shells face outward
  for (const p of patches) {
    if (!o.outward || !p.closed || p.tris.length < 4) continue;
    let vol = 0;
    for (const t of p.tris) {
      let a = rep[T[3 * t]], b = rep[T[3 * t + 1]], c = rep[T[3 * t + 2]];
      if (orient[t] < 0) { const x = b; b = c; c = x; }
      vol += signedVol(P, a, b, c, cx, cy, cz);
    }
    if (vol < 0) {
      for (const t of p.tris) orient[t] = -orient[t];
      report.inverted++; // shells turned inside-out to outward
    }
  }
  // flipped = faces whose winding differs from the input, whatever the reason
  for (let t = 0; t < m; t++) if (orient[t] < 0) report.flipped++;

  // ------------------------------------------------------------ 6. close small holes
  // A rim edge belongs to one triangle. Walked after orientation u->v, the hole's loop runs v->u.
  const next = new Map(); // loop vertex -> [{ to, tri }]
  let rim = 0;
  for (const [, list] of edges) {
    if (list.length !== 1) continue;
    const t = list[0] >> 1;
    for (let k = 0; k < 3; k++) {
      let u = rep[T[3 * t + k]], v = rep[T[3 * t + ((k + 1) % 3)]];
      const key = ekey(u, v);
      if (edges.get(key) !== list) continue;
      if (orient[t] < 0) { const x = u; u = v; v = x; }
      const arr = next.get(v);
      const e = { to: u, tri: t, used: false };
      if (arr) arr.push(e); else next.set(v, [e]);
      rim++;
      break;
    }
  }
  const fills = [], fillSrc = [];
  const maxWidth = diag * o.holeMaxRel;
  const areaCache = new Map();
  const surfaceArea = (id) => {
    if (!areaCache.has(id)) {
      let s = 0;
      for (const t of patches[id].tris) s += triArea(P, rep[T[3 * t]], rep[T[3 * t + 1]], rep[T[3 * t + 2]]);
      areaCache.set(id, s);
    }
    return areaCache.get(id);
  };
  let leftEdges = 0;
  for (const [start, arr] of next) {
    for (const first of arr) {
      if (first.used) continue;
      const loop = [start], tris = [first.tri];
      first.used = true;
      let cur = first.to, ok = arr.length === 1, guard = 0;
      while (cur !== start && guard++ < rim) {
        const outs = next.get(cur);
        if (!outs) { ok = false; break; }
        const e = outs.find((x) => !x.used);
        if (!e) { ok = false; break; }
        if (outs.length !== 1) ok = false; // a pinched rim: walk it so its edges are used, but do not guess a patch
        e.used = true;
        loop.push(cur); tris.push(e.tri);
        cur = e.to;
      }
      if (cur !== start) ok = false;
      if (ok && loop.length >= 3 && loop.length <= o.holeMaxEdges && loopWidth(P, loop) <= maxWidth) {
        const made = earClip(P, loop);
        // A patch that would be as big as the surface around it is not a hole: it is the edge of a sheet (a panel, a
        // sign, a single wall). Capping it would glue a second face onto the sheet, so leave it.
        let patchArea = 0;
        if (made) for (let i = 0; i < made.length; i += 3) patchArea += triArea(P, made[i], made[i + 1], made[i + 2]);
        if (made && patchArea < 0.5 * surfaceArea(patch[tris[0]])) {
          for (let i = 0; i < made.length; i += 3) { fills.push(made[i], made[i + 1], made[i + 2]); fillSrc.push(Tsrc[tris[0]]); }
          report.holesFilled++;
          continue;
        }
      }
      report.holesLeft++;
      leftEdges += loop.length;
    }
  }
  report.openEdges = leftEdges;

  // ------------------------------------------------------------ 7. loose pieces
  report.islands = countIslands(P, T, m, rep, nv, o.islandRel);

  // ------------------------------------------------------------ output
  const out = new Uint32Array(3 * m + fills.length);
  const triSource = new Int32Array(m + fills.length / 3);
  for (let t = 0; t < m; t++) {
    const a = T[3 * t], b = T[3 * t + 1], c = T[3 * t + 2];
    out[3 * t] = a;
    if (orient[t] < 0) { out[3 * t + 1] = c; out[3 * t + 2] = b; } else { out[3 * t + 1] = b; out[3 * t + 2] = c; }
    triSource[t] = Tsrc[t];
  }
  out.set(fills, 3 * m);
  triSource.set(fillSrc, m);
  report.ms = Math.round(now() - t0);
  return { index: out, triSource, report };
}

// ---------------------------------------------------------------------------------------------- helpers

/** Representative vertex for each vertex: the first vertex within tol of it (itself when none). Spatial hash, 8 cells. */
export function weld(P, finite, nv, tol, ox = 0, oy = 0, oz = 0) {
  const rep = new Int32Array(nv).fill(-1);
  const cell = 2 * tol, inv = 1 / cell, tol2 = tol * tol;
  const buckets = new Map();
  const hash = (x, y, z) => (Math.imul(x, 73856093) ^ Math.imul(y, 19349663) ^ Math.imul(z, 83492791)) | 0;
  for (let i = 0; i < nv; i++) {
    if (!finite[i]) continue;
    const x = P[3 * i] - ox, y = P[3 * i + 1] - oy, z = P[3 * i + 2] - oz;
    const fx = x * inv, fy = y * inv, fz = z * inv;
    const qx = Math.floor(fx), qy = Math.floor(fy), qz = Math.floor(fz);
    // a point within tol can only be in this cell or the neighbour on the near side of each axis
    const nx = fx - qx < 0.5 ? qx - 1 : qx + 1, ny = fy - qy < 0.5 ? qy - 1 : qy + 1, nz = fz - qz < 0.5 ? qz - 1 : qz + 1;
    let found = -1;
    search: for (const X of [qx, nx]) for (const Y of [qy, ny]) for (const Z of [qz, nz]) {
      const b = buckets.get(hash(X, Y, Z));
      if (!b) continue;
      for (const j of b) {
        const dx = P[3 * j] - ox - x, dy = P[3 * j + 1] - oy - y, dz = P[3 * j + 2] - oz - z;
        if (dx * dx + dy * dy + dz * dz <= tol2) { found = j; break search; }
      }
    }
    if (found >= 0) { rep[i] = found; continue; }
    rep[i] = i;
    const h = hash(qx, qy, qz);
    const b = buckets.get(h);
    if (b) b.push(i); else buckets.set(h, [i]);
  }
  for (let i = 0; i < nv; i++) if (rep[i] < 0) rep[i] = i;
  return rep;
}

function triArea(P, a, b, c) {
  const abx = P[3 * b] - P[3 * a], aby = P[3 * b + 1] - P[3 * a + 1], abz = P[3 * b + 2] - P[3 * a + 2];
  const acx = P[3 * c] - P[3 * a], acy = P[3 * c + 1] - P[3 * a + 1], acz = P[3 * c + 2] - P[3 * a + 2];
  const x = aby * acz - abz * acy, y = abz * acx - abx * acz, z = abx * acy - aby * acx;
  return Math.sqrt(x * x + y * y + z * z) / 2;
}

function signedVol(P, a, b, c, cx, cy, cz) {
  const ax = P[3 * a] - cx, ay = P[3 * a + 1] - cy, az = P[3 * a + 2] - cz;
  const bx = P[3 * b] - cx, by = P[3 * b + 1] - cy, bz = P[3 * b + 2] - cz;
  const qx = P[3 * c] - cx, qy = P[3 * c + 1] - cy, qz = P[3 * c + 2] - cz;
  return (ax * (by * qz - bz * qy) - ay * (bx * qz - bz * qx) + az * (bx * qy - by * qx)) / 6;
}

/** Largest distance between two points of the loop (its width). */
function loopWidth(P, loop) {
  let w = 0;
  for (let i = 0; i < loop.length; i++) for (let j = i + 1; j < loop.length; j++) {
    const a = loop[i], b = loop[j];
    const d = Math.hypot(P[3 * a] - P[3 * b], P[3 * a + 1] - P[3 * b + 1], P[3 * a + 2] - P[3 * b + 2]);
    if (d > w) w = d;
  }
  return w;
}

/**
 * Triangulate a closed loop of vertex ids by ear clipping in its best-fit plane. The triangles keep the loop's direction,
 * so they join the surrounding surface with matching winding. Returns a flat id list, or null when it cannot be done.
 */
export function earClip(P, loop) {
  const n = loop.length;
  if (n === 3) return [loop[0], loop[1], loop[2]];
  // Newell normal (follows the loop's direction)
  let nx = 0, ny = 0, nz = 0;
  for (let i = 0; i < n; i++) {
    const a = loop[i], b = loop[(i + 1) % n];
    nx += (P[3 * a + 1] - P[3 * b + 1]) * (P[3 * a + 2] + P[3 * b + 2]);
    ny += (P[3 * a + 2] - P[3 * b + 2]) * (P[3 * a] + P[3 * b]);
    nz += (P[3 * a] - P[3 * b]) * (P[3 * a + 1] + P[3 * b + 1]);
  }
  const nl = Math.hypot(nx, ny, nz);
  if (!(nl > 0)) return null;
  nx /= nl; ny /= nl; nz /= nl;
  // basis (u, v) with u x v = n, so the loop is counter-clockwise in 2D
  let ux, uy, uz;
  if (Math.abs(nx) < 0.9) { ux = 0; uy = nz; uz = -ny; } else { ux = -nz; uy = 0; uz = nx; }
  const ul = Math.hypot(ux, uy, uz); ux /= ul; uy /= ul; uz /= ul;
  const vx = ny * uz - nz * uy, vy = nz * ux - nx * uz, vz = nx * uy - ny * ux;
  const X = new Float64Array(n), Y = new Float64Array(n);
  for (let i = 0; i < n; i++) {
    const p = loop[i];
    X[i] = P[3 * p] * ux + P[3 * p + 1] * uy + P[3 * p + 2] * uz;
    Y[i] = P[3 * p] * vx + P[3 * p + 1] * vy + P[3 * p + 2] * vz;
  }
  const idx = Array.from({ length: n }, (_, i) => i);
  const out = [];
  const cross = (a, b, c) => (X[b] - X[a]) * (Y[c] - Y[a]) - (Y[b] - Y[a]) * (X[c] - X[a]);
  const inside = (p, a, b, c) => cross(a, b, p) >= 0 && cross(b, c, p) >= 0 && cross(c, a, p) >= 0;
  let guard = 0;
  while (idx.length > 3 && guard++ < n * n) {
    let clipped = false;
    for (let i = 0; i < idx.length; i++) {
      const a = idx[(i + idx.length - 1) % idx.length], b = idx[i], c = idx[(i + 1) % idx.length];
      if (cross(a, b, c) <= 0) continue; // reflex or flat corner
      let ear = true;
      for (const q of idx) if (q !== a && q !== b && q !== c && inside(q, a, b, c)) { ear = false; break; }
      if (!ear) continue;
      out.push(loop[a], loop[b], loop[c]);
      idx.splice(i, 1);
      clipped = true;
      break;
    }
    if (!clipped) return null; // self-intersecting or badly non-planar: leave the hole open
  }
  out.push(loop[idx[0]], loop[idx[1]], loop[idx[2]]);
  return out;
}

/** Number of connected pieces whose area is below islandRel of the total (only when there is more than one piece). */
function countIslands(P, T, m, rep, nv, islandRel) {
  if (!m) return 0;
  const parent = new Int32Array(nv);
  for (let i = 0; i < nv; i++) parent[i] = i;
  const find = (x) => { while (parent[x] !== x) { parent[x] = parent[parent[x]]; x = parent[x]; } return x; };
  const unite = (a, b) => { a = find(a); b = find(b); if (a !== b) parent[a < b ? b : a] = a < b ? a : b; };
  for (let t = 0; t < m; t++) { const a = rep[T[3 * t]]; unite(a, rep[T[3 * t + 1]]); unite(a, rep[T[3 * t + 2]]); }
  const area = new Map();
  let total = 0;
  for (let t = 0; t < m; t++) {
    const r = find(rep[T[3 * t]]);
    const s = triArea(P, rep[T[3 * t]], rep[T[3 * t + 1]], rep[T[3 * t + 2]]);
    area.set(r, (area.get(r) || 0) + s);
    total += s;
  }
  if (area.size < 2) return 0;
  let small = 0;
  for (const a of area.values()) if (a < total * islandRel) small++;
  return small;
}

/** One-line, plain-language summary of what was repaired (empty when the model was clean). */
export function describe(r) {
  const n = (v) => v.toLocaleString('en-US');
  const parts = [];
  if (r.flipped) parts.push(`${n(r.flipped)} face${r.flipped === 1 ? '' : 's'} turned to match ${r.flipped === 1 ? 'its' : 'their'} neighbours`);
  if (r.holesFilled) parts.push(`${n(r.holesFilled)} small hole${r.holesFilled === 1 ? '' : 's'} closed`);
  if (r.degenerate) parts.push(`${n(r.degenerate)} zero-size face${r.degenerate === 1 ? '' : 's'} removed`);
  if (r.duplicates) parts.push(`${n(r.duplicates)} duplicate face${r.duplicates === 1 ? '' : 's'} removed`);
  if (r.nanRemoved + r.badIndex) parts.push(`${n(r.nanRemoved + r.badIndex)} broken face${r.nanRemoved + r.badIndex === 1 ? '' : 's'} removed`);
  if (r.welded && parts.length) parts.push(`${n(r.welded)} loose seam point${r.welded === 1 ? '' : 's'} stitched`);
  return parts.join(', ');
}
