// Unit tests for the mesh repair (web/js/meshfix.js). Run: node --test tests/js
import test from 'node:test';
import assert from 'node:assert/strict';
import { repair, earClip, describe, changed } from '../../acoustic_phase_optimizer/web/js/meshfix.js';

// A closed box (12 x 20 x 5) with outward winding, as an indexed mesh.
function box(sx = 12, sy = 20, sz = 5) {
  const P = [];
  for (const z of [0, sz]) for (const y of [0, sy]) for (const x of [0, sx]) P.push(x, y, z);
  // vertex i = x + 2y + 4z (each 0/1)
  const quads = [[0, 2, 3, 1], [4, 5, 7, 6], [0, 1, 5, 4], [2, 6, 7, 3], [0, 4, 6, 2], [1, 3, 7, 5]];
  const I = [];
  for (const [a, b, c, d] of quads) I.push(a, b, c, a, c, d);
  return { P: new Float32Array(P), I: new Uint32Array(I) };
}

function volume(P, I) {
  let v = 0;
  for (let t = 0; t < I.length; t += 3) {
    const [a, b, c] = [I[t], I[t + 1], I[t + 2]].map((i) => [P[3 * i], P[3 * i + 1], P[3 * i + 2]]);
    v += (a[0] * (b[1] * c[2] - b[2] * c[1]) - a[1] * (b[0] * c[2] - b[2] * c[0]) + a[2] * (b[0] * c[1] - b[1] * c[0])) / 6;
  }
  return v;
}

/** Every edge shared by exactly two triangles, walked in opposite directions. */
function watertight(I) {
  const e = new Map();
  for (let t = 0; t < I.length; t += 3) for (let k = 0; k < 3; k++) {
    const u = I[t + k], v = I[t + ((k + 1) % 3)];
    const key = `${u},${v}`;
    e.set(key, (e.get(key) || 0) + 1);
  }
  for (const [key, n] of e) {
    const [u, v] = key.split(',');
    if (n !== 1 || e.get(`${v},${u}`) !== 1) return false;
  }
  return true;
}

/** Unindexed copy (a triangle soup, like STL). */
function soup(P, I) {
  const out = new Float32Array(I.length * 3);
  for (let i = 0; i < I.length; i++) out.set(P.subarray(3 * I[i], 3 * I[i] + 3), 3 * i);
  return out;
}

test('a clean box comes back unchanged', () => {
  const { P, I } = box();
  const r = repair(P, I);
  assert.deepEqual(Array.from(r.index), Array.from(I));
  assert.equal(changed(r.report), false);
  assert.equal(r.report.holesLeft, 0);
  assert.equal(describe(r.report), '');
});

test('flipped faces are turned to match their neighbours', () => {
  const { P, I } = box();
  const bad = I.slice();
  for (const t of [0, 5, 7]) { const x = bad[3 * t + 1]; bad[3 * t + 1] = bad[3 * t + 2]; bad[3 * t + 2] = x; }
  const r = repair(P, bad);
  assert.equal(r.report.flipped, 3);
  assert.ok(watertight(r.index));
  assert.ok(Math.abs(volume(P, r.index) - 12 * 20 * 5) < 1e-3);
});

test('a room seen from inside (faces pointing in) is left alone by default', () => {
  const { P, I } = box();
  const inv = I.slice();
  for (let t = 0; t < inv.length; t += 3) { const x = inv[t + 1]; inv[t + 1] = inv[t + 2]; inv[t + 2] = x; }
  const r = repair(P, inv);
  assert.equal(changed(r.report), false);
  assert.ok(volume(P, r.index) < 0);
});

test('with outward on, an inside-out shell is turned outward', () => {
  const { P, I } = box();
  const inv = I.slice();
  for (let t = 0; t < inv.length; t += 3) { const x = inv[t + 1]; inv[t + 1] = inv[t + 2]; inv[t + 2] = x; }
  const r = repair(P, inv, { outward: true });
  assert.equal(r.report.inverted, 1);
  assert.equal(r.report.flipped, 12);
  assert.ok(volume(P, r.index) > 0);
});

test('a triangle soup is stitched: winding and holes work across unjoined vertices', () => {
  const { P, I } = box();
  const S = soup(P, I);
  const r = repair(S, null);
  assert.equal(r.report.welded, 36 - 8);
  assert.equal(r.report.flipped, 0);
  assert.equal(r.report.holesLeft, 0); // stitched, so no open edges
});

test('NaN corners, bad indices, degenerate and duplicate faces are removed', () => {
  const { P, I } = box();
  const P2 = new Float32Array([...P, NaN, 0, 0]);
  const extra = [8, 0, 1, /* NaN */ 0, 1, 99, /* bad index */ 0, 0, 1, /* degenerate */ 0, 2, 3 /* duplicate of face 0 */];
  const r = repair(P2, new Uint32Array([...I, ...extra]));
  assert.equal(r.report.nanRemoved, 1);
  assert.equal(r.report.badIndex, 1);
  assert.equal(r.report.degenerate, 1);
  assert.equal(r.report.duplicates, 1);
  assert.equal(r.index.length, I.length);
  assert.ok(watertight(r.index));
});

test('a small hole is closed; the box is watertight again', () => {
  // a big box with a finely split floor, one floor cell removed
  const P = [], I = [];
  const N = 40, S = 20;
  const vid = (i, j) => i * (N + 1) + j;
  for (let i = 0; i <= N; i++) for (let j = 0; j <= N; j++) P.push((i / N) * S, (j / N) * S, 0);
  for (let i = 0; i < N; i++) for (let j = 0; j < N; j++) {
    if (i === 20 && j === 20) continue; // the hole: 0.5 m wide in a ~35 m model
    I.push(vid(i, j), vid(i, j + 1), vid(i + 1, j + 1), vid(i, j), vid(i + 1, j + 1), vid(i + 1, j));
  }
  const r = repair(new Float32Array(P), new Uint32Array(I), { holeMaxRel: 0.03 });
  assert.equal(r.report.holesFilled, 1);
  assert.equal(r.index.length, I.length + 6);
  assert.equal(r.report.holesLeft, 1); // the outer rim of the floor plate stays open
});

test('openings wider than the limit stay open (doors are real)', () => {
  const P = [], I = [];
  const N = 10, S = 20;
  const vid = (i, j) => i * (N + 1) + j;
  for (let i = 0; i <= N; i++) for (let j = 0; j <= N; j++) P.push((i / N) * S, (j / N) * S, 0);
  for (let i = 0; i < N; i++) for (let j = 0; j < N; j++) {
    if (i === 5 && j === 5) continue; // 2 m opening
    I.push(vid(i, j), vid(i, j + 1), vid(i + 1, j + 1), vid(i, j), vid(i + 1, j + 1), vid(i + 1, j));
  }
  const r = repair(new Float32Array(P), new Uint32Array(I));
  assert.equal(r.report.holesFilled, 0);
  assert.equal(r.report.holesLeft, 2);
});

test('a lone flat panel is not capped with a second face', () => {
  const P = new Float32Array([0, 0, 0, 0.1, 0, 0, 0.1, 0.1, 0, 0, 0.1, 0, 30, 30, 30]);
  const I = new Uint32Array([0, 1, 2, 0, 2, 3]);
  const r = repair(P, I);
  assert.equal(r.report.holesFilled, 0);
  assert.equal(r.index.length, 6);
});

test('patches keep the triangle they came from', () => {
  const { P, I } = box();
  const r = repair(P, I);
  assert.deepEqual(Array.from(r.triSource), Array.from({ length: 12 }, (_, i) => i));
});

test('ear clipping keeps the loop direction and handles a concave loop', () => {
  // an L shape in the z=0 plane, counter-clockwise seen from +z
  const pts = [[0, 0], [2, 0], [2, 1], [1, 1], [1, 2], [0, 2]];
  const P = new Float32Array(pts.flatMap(([x, y]) => [x, y, 0]));
  const tris = earClip(P, [0, 1, 2, 3, 4, 5]);
  assert.equal(tris.length, 12);
  let area = 0;
  for (let t = 0; t < tris.length; t += 3) {
    const [a, b, c] = [tris[t], tris[t + 1], tris[t + 2]];
    const z = (P[3 * b] - P[3 * a]) * (P[3 * c + 1] - P[3 * a + 1]) - (P[3 * b + 1] - P[3 * a + 1]) * (P[3 * c] - P[3 * a]);
    assert.ok(z > 0, 'every triangle faces +z like the loop');
    area += z / 2;
  }
  assert.ok(Math.abs(area - 3) < 1e-9);
});

test('the same input always gives the same output', () => {
  const { P, I } = box();
  const bad = I.slice(); const x = bad[1]; bad[1] = bad[2]; bad[2] = x;
  assert.deepEqual(Array.from(repair(P, bad).index), Array.from(repair(P, bad).index));
});

test('describe() reads well', () => {
  const s = describe({ flipped: 1204, holesFilled: 3, degenerate: 0, duplicates: 1, nanRemoved: 0, badIndex: 0, welded: 10 });
  assert.equal(s, '1,204 faces turned to match their neighbours, 3 small holes closed, 1 duplicate face removed, 10 loose seam points stitched');
});
