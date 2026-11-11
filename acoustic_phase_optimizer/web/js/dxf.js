// DXF (ASCII) reader: the 3D content of a drawing as three.js geometry, in drawing units, Z up.
//
// Read: 3DFACE, SOLID/TRACE, POLYLINE (polyface and polygon meshes, 2D and 3D polylines), LWPOLYLINE (with
// bulges), LINE, CIRCLE, ARC, ELLIPSE, SPLINE (as a polyline), MESH, and INSERT of blocks (nested, scaled,
// rotated, arrayed). Thickness extrudes 2D entities into walls, along each entity's extrusion direction
// (the "arbitrary axis" object coordinate system). Paper space is skipped.
// Not readable: 3DSOLID, BODY, REGION and SURFACE keep their shape as ACIS data, which needs a commercial
// kernel; they are counted so the user can be told.

import * as THREE from 'three';

const MAX_DEPTH = 16;
const MAX_TRIS = 8_000_000;
const ARC_STEP = Math.PI / 24; // 7.5 degrees per segment
// $INSUNITS: metres per drawing unit
const INSUNITS = { 1: 0.0254, 2: 0.3048, 3: 1609.344, 4: 0.001, 5: 0.01, 6: 1, 7: 1000, 8: 2.54e-8, 9: 2.54e-5, 10: 0.9144, 14: 0.1 };
const ACIS = new Set(['3DSOLID', 'BODY', 'REGION', 'SURFACE', 'PLANESURFACE', 'EXTRUDEDSURFACE', 'LOFTEDSURFACE', 'REVOLVEDSURFACE', 'SWEPTSURFACE']);

// ------------------------------------------------------------------ tokens
function* pairs(text) {
  const n = text.length;
  let i = 0;
  const line = () => {
    if (i >= n) return null;
    let j = text.indexOf('\n', i);
    if (j < 0) j = n;
    const s = text.slice(i, j > i && text.charCodeAt(j - 1) === 13 ? j - 1 : j);
    i = j + 1;
    return s;
  };
  for (;;) {
    const c = line();
    const v = line();
    if (c === null || v === null) return;
    const code = parseInt(c, 10);
    if (Number.isNaN(code)) throw new Error('this is not a readable DXF file (a group code is not a number)');
    yield [code, v];
  }
}

class Ent {
  constructor(type) { this.type = type; this.c = []; }
  get(code, dflt) { for (const [k, v] of this.c) if (k === code) return v; return dflt; }
  num(code, dflt = 0) { const v = this.get(code); if (v === undefined) return dflt; const x = parseFloat(v); return Number.isFinite(x) ? x : dflt; }
  int(code, dflt = 0) { return Math.trunc(this.num(code, dflt)); }
}

/** Parse into header values, blocks (name -> {base, ents}) and model-space entities. */
function parse(text) {
  if (text.startsWith('AutoCAD Binary DXF')) throw new Error('this is a binary DXF; save it as ASCII DXF');
  const header = {};
  const blocks = new Map();
  const entities = [];
  let section = null, headerVar = null;
  let list = null; // where entities go now
  let cur = null; // the entity being read
  let owner = null; // POLYLINE collecting VERTEX entities, or INSERT skipping ATTRIBs
  let block = null;
  const finish = () => {
    if (!cur) return;
    if (cur.type === 'VERTEX' && owner?.type === 'POLYLINE') owner.vertices.push(cur);
    else if ((cur.type === 'ATTRIB' || cur.type === 'VERTEX') && owner) { /* part of its owner */ }
    else if (cur.type === 'SEQEND') owner = null;
    else if (list && cur.type !== 'BLOCK' && cur.type !== 'ENDBLK') {
      list.push(cur);
      if (cur.type === 'POLYLINE') { cur.vertices = []; owner = cur; } else if (cur.type === 'INSERT' && cur.int(66) === 1) owner = cur;
      else owner = null;
    }
    cur = null;
  };
  for (const [code, v] of pairs(text)) {
    if (code === 0) {
      finish();
      const t = v.trim();
      if (t === 'SECTION') { section = 'pending'; continue; }
      if (t === 'ENDSEC') { section = null; list = null; continue; }
      if (t === 'EOF') break;
      if (section === 'BLOCKS' && t === 'BLOCK') { block = new Ent('BLOCK'); cur = block; list = null; continue; }
      if (section === 'BLOCKS' && t === 'ENDBLK') { block = null; list = null; cur = new Ent('ENDBLK'); continue; }
      cur = new Ent(t);
      if (block && !list) { // the first entity of a block: the block's own codes are complete
        const name = block.get(2, '').trim();
        const ents = [];
        blocks.set(name.toUpperCase(), { base: [block.num(10), block.num(20), block.num(30)], ents });
        list = ents;
      }
      continue;
    }
    if (section === 'pending' && code === 2) {
      section = v.trim();
      if (section === 'ENTITIES') list = entities;
      continue;
    }
    if (section === 'HEADER') {
      if (code === 9) headerVar = v.trim();
      else if (headerVar && !(headerVar in header)) header[headerVar] = v.trim();
      continue;
    }
    if (cur) cur.c.push([code, v]);
  }
  finish();
  return { header, blocks, entities };
}

// ------------------------------------------------------------------ geometry
/** The object coordinate system of an extrusion direction (AutoCAD's arbitrary axis algorithm). */
function ocs(e) {
  const az = new THREE.Vector3(e.num(210, 0), e.num(220, 0), e.num(230, 1));
  if (az.lengthSq() < 1e-12) az.set(0, 0, 1);
  az.normalize();
  const ax = Math.abs(az.x) < 1 / 64 && Math.abs(az.y) < 1 / 64
    ? new THREE.Vector3(0, 1, 0).cross(az) : new THREE.Vector3(0, 0, 1).cross(az);
  ax.normalize();
  const ay = new THREE.Vector3().crossVectors(az, ax);
  return { m: new THREE.Matrix4().makeBasis(ax, ay, az), dir: az };
}

/** Points along a 2D polyline with bulges (each bulge belongs to the vertex it starts from). */
function bulgePath(pts, bulges, closed) {
  const out = [];
  const n = pts.length;
  const segs = closed ? n : n - 1;
  for (let i = 0; i < segs; i++) {
    const p0 = pts[i], p1 = pts[(i + 1) % n], b = bulges[i] || 0;
    out.push(p0);
    if (Math.abs(b) > 1e-9) {
      const th = 4 * Math.atan(b);
      const dx = p1[0] - p0[0], dy = p1[1] - p0[1];
      const chord = Math.hypot(dx, dy);
      if (chord < 1e-12) continue;
      const k = (chord / 2) / Math.tan(th / 2);
      const cx = (p0[0] + p1[0]) / 2 - (dy / chord) * k, cy = (p0[1] + p1[1]) / 2 + (dx / chord) * k;
      const r = Math.hypot(p0[0] - cx, p0[1] - cy);
      const a0 = Math.atan2(p0[1] - cy, p0[0] - cx);
      const steps = Math.max(2, Math.ceil(Math.abs(th) / ARC_STEP));
      for (let s = 1; s < steps; s++) { const a = a0 + (th * s) / steps; out.push([cx + r * Math.cos(a), cy + r * Math.sin(a)]); }
    }
  }
  if (!closed && n) out.push(pts[n - 1]);
  return out;
}

class Builder {
  constructor() { this.tris = new Map(); this.lines = new Map(); this.count = 0; }
  layer(map, name) { if (!map.has(name)) map.set(name, []); return map.get(name); }
  tri(layer, m, a, b, c) {
    if (++this.count > MAX_TRIS) throw new Error('the drawing has more than 8 million faces');
    const t = this.layer(this.tris, layer);
    for (const p of [a, b, c]) { const v = new THREE.Vector3(p[0], p[1], p[2]).applyMatrix4(m); t.push(v.x, v.y, v.z); }
  }
  poly(layer, m, pts) { // a planar polygon, fanned (DXF faces are convex quads or triangles)
    for (let i = 1; i + 1 < pts.length; i++) this.tri(layer, m, pts[0], pts[i], pts[i + 1]);
  }
  seg(layer, m, a, b) {
    const l = this.layer(this.lines, layer);
    for (const p of [a, b]) { const v = new THREE.Vector3(p[0], p[1], p[2]).applyMatrix4(m); l.push(v.x, v.y, v.z); }
  }
  /** A path, as lines; with thickness, as a wall of quads along `dir` (in the path's own frame). */
  path(layer, m, pts, thickness, dir = [0, 0, 1]) {
    for (let i = 0; i + 1 < pts.length; i++) {
      const a = pts[i], b = pts[i + 1];
      if (thickness) {
        const t = thickness;
        const a2 = [a[0] + dir[0] * t, a[1] + dir[1] * t, a[2] + dir[2] * t], b2 = [b[0] + dir[0] * t, b[1] + dir[1] * t, b[2] + dir[2] * t];
        this.poly(layer, m, [a, b, b2, a2]);
      } else this.seg(layer, m, a, b);
    }
  }
}

function arcPoints(cx, cy, r, a0, a1) {
  let sweep = a1 - a0;
  while (sweep <= 0) sweep += 2 * Math.PI;
  const steps = Math.max(2, Math.ceil(sweep / ARC_STEP));
  const out = [];
  for (let s = 0; s <= steps; s++) { const a = a0 + (sweep * s) / steps; out.push([cx + r * Math.cos(a), cy + r * Math.sin(a)]); }
  return out;
}

function emit(e, m, b, ctx, depth) {
  if (e.int(67) === 1) return; // paper space
  const layer = (e.get(8, '0') || '0').trim();
  const th = e.num(39);
  switch (e.type) {
    case '3DFACE': {
      const p = [0, 1, 2, 3].map((i) => [e.num(10 + i), e.num(20 + i), e.num(30 + i)]);
      const same = p[2][0] === p[3][0] && p[2][1] === p[3][1] && p[2][2] === p[3][2];
      b.poly(layer, m, same ? p.slice(0, 3) : p);
      ctx.faces++;
      break;
    }
    case 'SOLID': case 'TRACE': {
      const o = ocs(e);
      const mm = m.clone().multiply(o.m);
      const z = e.num(30);
      const p = [10, 11, 13, 12].map((k) => [e.num(k), e.num(k + 10), z]); // DXF corner order 1, 2, 4, 3
      const same = e.num(12) === e.num(13) && e.num(22) === e.num(23);
      const ring = same ? [p[0], p[1], p[3]] : p;
      b.poly(layer, mm, ring);
      if (th) {
        b.poly(layer, mm, ring.map((q) => [q[0], q[1], q[2] + th]));
        b.path(layer, mm, [...ring, ring[0]], th);
      }
      ctx.faces++;
      break;
    }
    case 'LINE': {
      const a = [e.num(10), e.num(20), e.num(30)], c = [e.num(11), e.num(21), e.num(31)];
      const d = ocs(e).dir;
      b.path(layer, m, [a, c], th, [d.x, d.y, d.z]); // LINE points are in world coordinates
      ctx.lines++;
      break;
    }
    case 'LWPOLYLINE': {
      const o = ocs(e);
      const z = e.num(38);
      const pts = [], bulges = [];
      for (const [k, v] of e.c) {
        if (k === 10) { pts.push([parseFloat(v), 0]); bulges.push(0); }
        else if (k === 20 && pts.length) pts[pts.length - 1][1] = parseFloat(v);
        else if (k === 42 && pts.length) bulges[bulges.length - 1] = parseFloat(v);
      }
      const path = bulgePath(pts, bulges, (e.int(70) & 1) === 1).map((q) => [q[0], q[1], z]);
      if ((e.int(70) & 1) === 1 && path.length) path.push(path[0]);
      b.path(layer, m.clone().multiply(o.m), path, th);
      ctx.lines++;
      break;
    }
    case 'POLYLINE': polyline(e, m, b, ctx, layer, th); break;
    case 'CIRCLE': case 'ARC': {
      const o = ocs(e);
      const r = e.num(40);
      const a0 = e.type === 'ARC' ? THREE.MathUtils.degToRad(e.num(50)) : 0;
      const a1 = e.type === 'ARC' ? THREE.MathUtils.degToRad(e.num(51)) : 2 * Math.PI;
      const z = e.num(30);
      const pts = arcPoints(e.num(10), e.num(20), r, a0, a1).map((q) => [q[0], q[1], z]);
      b.path(layer, m.clone().multiply(o.m), pts, th);
      ctx.lines++;
      break;
    }
    case 'ELLIPSE': {
      const c = new THREE.Vector3(e.num(10), e.num(20), e.num(30));
      const major = new THREE.Vector3(e.num(11), e.num(21), e.num(31));
      const minor = ocs(e).dir.clone().cross(major).multiplyScalar(e.num(40, 1));
      let t0 = e.num(41, 0), t1 = e.num(42, 2 * Math.PI);
      if (t1 <= t0) t1 += 2 * Math.PI;
      const steps = Math.max(8, Math.ceil((t1 - t0) / ARC_STEP));
      const pts = [];
      for (let s = 0; s <= steps; s++) {
        const t = t0 + ((t1 - t0) * s) / steps;
        const p = c.clone().addScaledVector(major, Math.cos(t)).addScaledVector(minor, Math.sin(t));
        pts.push([p.x, p.y, p.z]);
      }
      b.path(layer, m, pts, 0);
      ctx.lines++;
      break;
    }
    case 'SPLINE': { // through its fit points when it has them, else along its control polygon
      const fit = [], ctl = [];
      let target = null;
      for (const [k, v] of e.c) {
        if (k === 10) { target = ctl; ctl.push([parseFloat(v), 0, 0]); } else if (k === 11) { target = fit; fit.push([parseFloat(v), 0, 0]); }
        else if ((k === 20 || k === 21) && target?.length) target[target.length - 1][1] = parseFloat(v);
        else if ((k === 30 || k === 31) && target?.length) target[target.length - 1][2] = parseFloat(v);
      }
      b.path(layer, m, fit.length >= 2 ? fit : ctl, 0);
      ctx.lines++;
      break;
    }
    case 'MESH': mesh(e, m, b, ctx, layer); break;
    case 'INSERT': insert(e, m, b, ctx, depth); break;
    default:
      if (ACIS.has(e.type)) ctx.acis++;
  }
}

function polyline(e, m, b, ctx, layer, th) {
  const flags = e.int(70);
  const vs = e.vertices || [];
  const vp = (v) => [v.num(10), v.num(20), v.num(30)];
  if (flags & 64) { // polyface mesh: position vertices, then face records of 1-based indices
    const pos = vs.filter((v) => (v.int(70) & 192) === 192).map(vp);
    for (const f of vs.filter((v) => (v.int(70) & 192) === 128)) {
      const idx = [71, 72, 73, 74].map((k) => Math.abs(f.int(k))).filter((i) => i > 0 && i <= pos.length);
      if (idx.length >= 3) b.poly(layer, m, idx.map((i) => pos[i - 1]));
    }
    ctx.faces++;
    return;
  }
  if (flags & 16) { // polygon mesh, M x N
    const smooth = e.int(75) !== 0;
    const M = smooth ? e.int(73) || e.int(71) : e.int(71), N = smooth ? e.int(74) || e.int(72) : e.int(72);
    const pos = vs.filter((v) => !(v.int(70) & 16)).map(vp);
    if (M < 2 || N < 2 || pos.length < M * N) return;
    const at = (i, j) => pos[(i % M) * N + (j % N)];
    const mi = flags & 1 ? M : M - 1, nj = flags & 32 ? N : N - 1;
    for (let i = 0; i < mi; i++) for (let j = 0; j < nj; j++) b.poly(layer, m, [at(i, j), at(i + 1, j), at(i + 1, j + 1), at(i, j + 1)]);
    ctx.faces++;
    return;
  }
  const closed = (flags & 1) === 1;
  if (flags & 8) { // 3D polyline, world coordinates
    const pts = vs.map(vp);
    if (closed && pts.length) pts.push(pts[0]);
    b.path(layer, m, pts, 0);
  } else { // 2D polyline in its object coordinate system, at its elevation, with bulges
    const o = ocs(e);
    const z = e.num(30);
    const path = bulgePath(vs.map((v) => [v.num(10), v.num(20)]), vs.map((v) => v.num(42)), closed).map((q) => [q[0], q[1], z]);
    if (closed && path.length) path.push(path[0]);
    b.path(layer, m.clone().multiply(o.m), path, th);
  }
  ctx.lines++;
}

function mesh(e, m, b, ctx, layer) {
  // codes in order: 92 = vertex count, then 10/20/30 per vertex; 93 = face list size, then 90 per item
  const pos = [];
  const faces = [];
  let want = 0, mode = null, faceLeft = 0;
  for (const [k, v] of e.c) {
    if (k === 92) { want = parseInt(v, 10); mode = 'v'; continue; }
    if (k === 93) { faceLeft = parseInt(v, 10); mode = 'f'; continue; }
    if (mode === 'v') {
      if (k === 10) pos.push([parseFloat(v), 0, 0]); else if (k === 20 && pos.length) pos[pos.length - 1][1] = parseFloat(v);
      else if (k === 30 && pos.length) { pos[pos.length - 1][2] = parseFloat(v); if (pos.length >= want) mode = null; }
    } else if (mode === 'f' && k === 90) { faces.push(parseInt(v, 10)); if (--faceLeft <= 0) mode = null; }
  }
  for (let i = 0; i < faces.length;) {
    const n = faces[i];
    const idx = faces.slice(i + 1, i + 1 + n).filter((j) => j >= 0 && j < pos.length);
    if (idx.length >= 3) b.poly(layer, m, idx.map((j) => pos[j]));
    i += n + 1;
    if (n <= 0) break;
  }
  ctx.faces++;
}

function insert(e, m, b, ctx, depth) {
  const blk = ctx.blocks.get((e.get(2, '') || '').trim().toUpperCase());
  if (!blk || depth >= MAX_DEPTH) return;
  ctx.inserts++;
  const o = ocs(e);
  const sx = e.num(41, 1), sy = e.num(42, 1), sz = e.num(43, 1);
  const rot = THREE.MathUtils.degToRad(e.num(50));
  const cols = Math.max(1, Math.min(e.int(70, 1), 1000)), rows = Math.max(1, Math.min(e.int(71, 1), 1000));
  const cs = e.num(44), rs = e.num(45);
  for (let r = 0; r < rows; r++) {
    for (let c = 0; c < cols; c++) {
      const t = m.clone().multiply(o.m)
        .multiply(new THREE.Matrix4().makeTranslation(e.num(10), e.num(20), e.num(30)))
        .multiply(new THREE.Matrix4().makeRotationZ(rot))
        .multiply(new THREE.Matrix4().makeTranslation(c * cs, r * rs, 0))
        .multiply(new THREE.Matrix4().makeScale(sx, sy, sz))
        .multiply(new THREE.Matrix4().makeTranslation(-blk.base[0], -blk.base[1], -blk.base[2]));
      for (const child of blk.ents) emit(child, t, b, ctx, depth + 1);
    }
  }
}

/**
 * Parse DXF text into a group (drawing units, Z up), one child per layer.
 * userData.dxf: { metersPerUnit (null when the drawing does not say), faces, lines, acis, inserts, flat }.
 */
export function parseDxf(text) {
  const { header, blocks, entities } = parse(text);
  const ctx = { blocks, faces: 0, lines: 0, acis: 0, inserts: 0 };
  const b = new Builder();
  const I = new THREE.Matrix4();
  for (const e of entities) emit(e, I, b, ctx, 0);

  const group = new THREE.Group();
  const mat = new THREE.MeshStandardMaterial({ color: 0xb9b6ae, roughness: 0.9, side: THREE.DoubleSide });
  const lineMat = new THREE.LineBasicMaterial({ color: 0x8a877f });
  const names = new Set([...b.tris.keys(), ...b.lines.keys()]);
  for (const name of names) {
    const g = new THREE.Group();
    g.name = name;
    const t = b.tris.get(name);
    if (t?.length) {
      const geo = new THREE.BufferGeometry();
      geo.setAttribute('position', new THREE.Float32BufferAttribute(t, 3));
      geo.computeVertexNormals();
      g.add(new THREE.Mesh(geo, mat));
    }
    const l = b.lines.get(name);
    if (l?.length) {
      const geo = new THREE.BufferGeometry();
      geo.setAttribute('position', new THREE.Float32BufferAttribute(l, 3));
      g.add(new THREE.LineSegments(geo, lineMat));
    }
    group.add(g);
  }
  const box = new THREE.Box3().setFromObject(group);
  const size = box.isEmpty() ? new THREE.Vector3() : box.getSize(new THREE.Vector3());
  const units = parseInt(header.$INSUNITS ?? '0', 10);
  group.userData.dxf = {
    metersPerUnit: INSUNITS[units] ?? null,
    faces: ctx.faces, lines: ctx.lines, acis: ctx.acis, inserts: ctx.inserts,
    hasTris: b.tris.size > 0 && [...b.tris.values()].some((t) => t.length),
    // a plan: lines only, all at one height (or nearly)
    flat: ![...b.tris.values()].some((t) => t.length) && b.lines.size > 0 && size.z <= Math.max(size.x, size.y) * 0.002,
  };
  return group;
}

/** Raise every line of a flat plan into a wall `height` (drawing units) tall, standing on the plan. */
export function extrudePlan(group, height) {
  const box = new THREE.Box3().setFromObject(group);
  const z0 = box.min.z;
  const mat = new THREE.MeshStandardMaterial({ color: 0xb9b6ae, roughness: 0.9, side: THREE.DoubleSide });
  group.updateMatrixWorld(true);
  const tris = [];
  const a = new THREE.Vector3(), c = new THREE.Vector3();
  group.traverse((o) => {
    if (!o.isLineSegments) return;
    const p = o.geometry.attributes.position;
    for (let i = 0; i + 1 < p.count; i += 2) {
      a.fromBufferAttribute(p, i).applyMatrix4(o.matrixWorld);
      c.fromBufferAttribute(p, i + 1).applyMatrix4(o.matrixWorld);
      if (a.distanceToSquared(c) < 1e-12) continue;
      tris.push(a.x, a.y, z0, c.x, c.y, z0, c.x, c.y, z0 + height, a.x, a.y, z0, c.x, c.y, z0 + height, a.x, a.y, z0 + height);
    }
  });
  if (!tris.length) return group;
  // a floor and a ceiling over the plan's extent, so the room is measured from top to bottom
  const { min, max } = box;
  for (const z of [z0, z0 + height]) tris.push(min.x, min.y, z, max.x, min.y, z, max.x, max.y, z, min.x, min.y, z, max.x, max.y, z, min.x, max.y, z);
  const geo = new THREE.BufferGeometry();
  geo.setAttribute('position', new THREE.Float32BufferAttribute(tris, 3));
  geo.computeVertexNormals();
  const walls = new THREE.Mesh(geo, mat);
  walls.name = 'walls raised from the plan';
  group.add(walls);
  group.userData.dxf = { ...group.userData.dxf, flat: false, hasTris: true, raised: height };
  return group;
}
