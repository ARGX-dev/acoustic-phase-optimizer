// Unit tests for speaker detection (web/js/speakerfind.js). Run: node --test tests/js
import test from 'node:test';
import assert from 'node:assert/strict';
import { findSpeakers, nameEvidence, shapeEvidence, mergeColumns } from '../../acoustic_phase_optimizer/web/js/speakerfind.js';

const room = { width: 12, length: 20, height: 6 };
const audience = { x0: 1, x1: 11, y0: 4, y1: 19, ear_height: 1.2 };
const types = [
  { id: 'lacoustics_kara_ii', brand: 'L-Acoustics', label: 'L-Acoustics Kara II', role: 'main' },
  { id: 'lacoustics_ks28', brand: 'L-Acoustics', label: 'L-Acoustics KS28', role: 'sub' },
  { id: 'lacoustics_k2', brand: 'L-Acoustics', label: 'L-Acoustics K2', role: 'main' },
  { id: 'qsc_k12_2', brand: 'QSC', label: 'QSC K12.2', role: 'main' },
];
const ctx = { room, audience, types };
// a closed box's surface area, as the extractor reports it
const box = (lo, s, name = '') => ({ name, lo, hi: [lo[0] + s[0], lo[1] + s[1], lo[2] + s[2]], area: 2 * (s[0] * s[1] + s[1] * s[2] + s[0] * s[2]) });

test('names: speaker words, library models, decoys', () => {
  assert.ok(nameEvidence('Speaker_L').score >= 0.8);
  assert.equal(nameEvidence('SUB 2').role, 'sub');
  assert.equal(nameEvidence('KARA_II hang left', types).typeId, 'lacoustics_kara_ii');
  assert.equal(nameEvidence('L-Acoustics KS28 #3', types).typeId, 'lacoustics_ks28');
  assert.equal(nameEvidence('FrontFill 1').role, 'front_fill');
  for (const decoy of ['Wall_03', 'Subfloor', 'AC unit', 'Cabinet', 'Column_01', 'Window glass', 'Monitor stand']) {
    assert.ok(nameEvidence(decoy, types).score < 0.5, decoy);
  }
  // a two-letter model needs a brand or a speaker word around it
  assert.equal(nameEvidence('K2', types).typeId, null);
  assert.equal(nameEvidence('L-Acoustics K2 left', types).typeId, 'lacoustics_k2');
});

test('shapes: speaker boxes on walls, subs at the stage, not an AC unit or a cabinet', () => {
  const spk = shapeEvidence(box([0.05, 3, 3], [0.4, 0.4, 0.65]), room); // on the left wall, high
  assert.equal(spk.kind, 'point');
  assert.ok(spk.score >= 0.5);
  const ac = shapeEvidence(box([0.02, 12, 4.5], [0.25, 0.9, 0.3]), room); // wide and flat, on a wall
  assert.ok(ac.score < 0.3);
  const sub = shapeEvidence(box([4, 0.5, 0], [0.9, 0.75, 0.55]), room); // on the floor at the stage
  assert.equal(sub.kind, 'sub');
  assert.ok(sub.score >= 0.5);
  const cab = shapeEvidence(box([9, 19, 0], [1.0, 0.5, 0.9]), room); // low cabinet at the back wall
  assert.ok(cab.score < 0.5);
  const tall = shapeEvidence(box([9, 19, 0], [0.9, 0.5, 2.0]), room); // wardrobe
  assert.equal(tall.score, 0);
});

test('stacked boxes merge into one line array', () => {
  const parts = [0, 1, 2, 3, 4, 5].map((i) => box([5, 2, 5 - i * 0.3], [1.0, 0.5, 0.29]));
  const merged = mergeColumns(parts);
  assert.equal(merged.length, 1);
  assert.equal(merged[0].stacked, 6);
  const { candidates } = findSpeakers({ objects: parts }, ctx);
  assert.equal(candidates.length, 1);
  assert.equal(candidates[0].boxes, 6);
  assert.ok(candidates[0].checked && candidates[0].pitch < 0);
});

test('a room with speakers and decoys', () => {
  const objects = [
    box([0.05, 3, 3], [0.4, 0.4, 0.65]), // left wall speaker
    box([11.55, 3, 3], [0.4, 0.4, 0.65]), // right wall speaker
    box([3, 0.4, 0], [0.9, 0.75, 0.55]), box([8, 0.4, 0], [0.9, 0.75, 0.55]), // subs
    box([0.02, 12, 4.5], [0.25, 0.9, 0.3]), // AC unit
    box([9, 19.4, 0], [1.0, 0.5, 0.9]), // cabinet at the back
  ];
  const { candidates } = findSpeakers({ objects }, ctx);
  const ticked = candidates.filter((c) => c.checked);
  assert.equal(ticked.length, 4);
  assert.equal(ticked.filter((c) => c.role === 'sub').length, 2);
  const left = ticked.find((c) => c.x < 1);
  assert.ok(left.yaw > 45 && left.yaw <= 90, `left wall speaker faces +x, got ${left.yaw}`); // away from the wall
  const right = ticked.find((c) => c.x > 11 && c.role !== 'sub');
  assert.ok(right.yaw < -45);
  assert.ok(!ticked.some((c) => c.y > 18), 'the cabinet is not ticked');
});

test('named objects win over the bare shape at the same place, and keep their model', () => {
  const objects = [box([5, 1, 4], [0.4, 0.4, 0.6], 'QSC K12.2 left'), box([5, 1, 4], [0.4, 0.4, 0.6])];
  const { candidates } = findSpeakers({ objects }, ctx);
  assert.equal(candidates.length, 1);
  assert.equal(candidates[0].typeId, 'qsc_k12_2');
  assert.equal(candidates[0].source, 'name');
});

test('MVR fixtures: speakers kept, lights skipped', () => {
  const fixtures = [
    { name: 'Main L', spec: 'L-Acoustics@Kara II.gdtf', pos: [2, 1, 7], fwd: [0, 0, -1] },
    { name: 'Sub 1', spec: 'L-Acoustics@KS28.gdtf', pos: [4, 0.5, 0.4], fwd: [0, 1, 0] },
    { name: 'Spot 1', spec: 'Robe@Robin_MegaPointe.gdtf', pos: [6, 5, 7], fwd: [0, 0, -1] },
  ];
  const { candidates, skippedFixtures } = findSpeakers({ fixtures }, { ...ctx, room: { ...room, height: 8 } });
  assert.equal(skippedFixtures, 1);
  assert.equal(candidates.length, 2);
  const main = candidates.find((c) => c.name === 'Main L');
  assert.equal(main.typeId, 'lacoustics_kara_ii');
  assert.ok(main.pitch < 0 && main.yaw > 0, 'a hanging fixture aims down at the seats, toed in');
  assert.equal(candidates.find((c) => c.name === 'Sub 1').role, 'sub');
});
