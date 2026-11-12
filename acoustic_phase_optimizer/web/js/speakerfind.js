// Find loudspeakers in an imported scan or venue model. Pure (no three.js), so it runs in Node tests too.
//
// Three kinds of evidence, strongest first:
//   1. MVR fixtures whose name or GDTF file names a loudspeaker (a rigged system in a CAD export);
//   2. objects and materials named like a speaker ("Speaker_L", "SUB 2", "KARA hang", a model from the library);
//   3. the shape alone: separate box-like parts of speaker size, on a wall or the ceiling or stacked as a line
//      array, and larger boxes on the floor near the stage (subs).
// Every candidate has a confidence; nothing is added to the project without the user ticking it.
//
// Coordinates are room metres: x across, y from the stage wall back, z up. Yaw 0 faces +y (into the audience),
// positive yaw turns toward +x; pitch is positive upward (as in the Speaker model).

const DEG = 180 / Math.PI;

const SPEAKER_WORDS = new Set(['speaker', 'speakers', 'loudspeaker', 'loudspeakers', 'spk', 'spkr', 'ls', 'pa', 'lautsprecher',
  'enceinte', 'altavoz', 'altoparlante', 'subwoofer', 'subwoofers', 'sub', 'subs', 'bass', 'woofer', 'linearray', 'array',
  'hang', 'wedge', 'monitor', 'frontfill', 'outfill', 'infill', 'sidefill', 'delay', 'column', 'cabinet_speaker', 'lsp']);
const WEAK_WORDS = new Set(['array', 'hang', 'monitor', 'delay', 'column', 'bass']); // speaker-ish only with other evidence
const SUB_WORDS = new Set(['sub', 'subs', 'subwoofer', 'subwoofers', 'bass', 'woofer', 'lf', 'lfe']);
const ARRAY_WORDS = new Set(['linearray', 'array', 'hang', 'line']);
const FILL_WORDS = { frontfill: 'front_fill', infill: 'front_fill', outfill: 'fill', sidefill: 'fill', delay: 'delay' };
// loudspeaker makers ("Martin" alone is also a lighting brand, so only "Martin Audio" counts)
const BRANDS = ['lacoustics', 'l-acoustics', 'dbaudiotechnik', 'd&b', 'meyersound', 'meyer', 'jbl', 'qsc', 'rcf', 'electrovoice',
  'electro-voice', 'martinaudio', 'nexo', 'adamson', 'coda', 'outline', 'funktionone', 'bose', 'yamaha', 'eaw', 'danley',
  'turbosound', 'kling', 'klingfreitag', 'fohhn', 'tannoy', 'kv2', 'dynacord', 'ev'];
// well-known product families that are not (yet) in the library: a name match is still evidence
const FAMILIES = ['kara', 'karaii', 'k1', 'k2', 'k3', 'kiva', 'kivaii', 'kara2', 'ks28', 'sb18', 'ks21', 'syva', 'soka', 'x15', 'x12',
  'x8', 'a15', 'a10', 'l2', 'gsl', 'ksl', 'vio', 'vtx', 'srx', 'prx', 'leo', 'lyon', 'leopard', 'lina', 'panther', 'mica', 'upa', 'upj',
  'upq', 'ultra', '1100lfc', '900lfc', '750lfc', 'stp', 'geo', 'stm', 'e15', 'cdd', 'xd', 'mlx', 'mla', 'wpc', 'wpl', 'cs', 'kiva2'];

/** Lower-case tokens of a name: split on non-alphanumerics and camelCase, plus the whole name squashed. */
export function tokens(name) {
  const raw = String(name || '').replace(/([a-z])([A-Z])/g, '$1 $2');
  const parts = raw.toLowerCase().split(/[^a-z0-9&]+/).filter(Boolean);
  // "line array" and "front fill" written apart
  const joined = [];
  for (let i = 0; i + 1 < parts.length; i++) joined.push(parts[i] + parts[i + 1]);
  return new Set([...parts, ...joined]);
}

const squash = (s) => String(s || '').toLowerCase().replace(/[^a-z0-9]+/g, '');

/**
 * What a name says about a loudspeaker.
 * @returns {{score: number, typeId: string|null, role: string|null, array: boolean, words: string[]}}
 *   score 0 (nothing) to 1 (a library model or a clear speaker word).
 */
export function nameEvidence(name, types = []) {
  const t = tokens(name);
  const flat = squash(name);
  const words = [...t].filter((w) => SPEAKER_WORDS.has(w));
  const strong = words.filter((w) => !WEAK_WORDS.has(w));
  let score = strong.length ? 0.85 : words.length ? 0.35 : 0;
  let typeId = null;
  // a model from the library: the model name as a token (short ones like "K2" only with a brand or speaker word)
  const brandHit = BRANDS.some((b) => flat.includes(squash(b)) && (squash(b).length > 2 || t.has(squash(b))));
  let best = 0;
  for (const ty of types) {
    if (!ty.brand || !ty.label) continue;
    const model = squash(ty.label.slice(ty.brand.length));
    if (!model || model.length < 2) continue;
    const hit = t.has(model) || (model.length >= 4 && flat.includes(model));
    if (!hit) continue;
    if (model.length < 4 && !brandHit && !strong.length) continue;
    if (model.length > best) { best = model.length; typeId = ty.id; }
  }
  if (typeId) score = Math.max(score, 0.95);
  else if (FAMILIES.some((f) => t.has(f)) && (brandHit || words.length)) score = Math.max(score, 0.9);
  else if (FAMILIES.some((f) => f.length >= 4 && t.has(f))) score = Math.max(score, 0.6);
  if (brandHit) score = Math.max(score, words.length ? 0.9 : 0.5);
  const isSub = [...t].some((w) => SUB_WORDS.has(w)) || /sub|ks28|sb18|ks21|lfc|xlf/.test(flat) && score > 0;
  const fill = Object.keys(FILL_WORDS).find((w) => t.has(w));
  const array = [...t].some((w) => ARRAY_WORDS.has(w)) && score > 0;
  const role = !score ? null : isSub ? 'sub' : fill ? FILL_WORDS[fill] : null;
  return { score: Math.min(1, score), typeId, role, array, words };
}

// ---------------------------------------------------------------- geometry helpers
const size = (o) => [o.hi[0] - o.lo[0], o.hi[1] - o.lo[1], o.hi[2] - o.lo[2]];
const centre = (o) => [(o.lo[0] + o.hi[0]) / 2, (o.lo[1] + o.hi[1]) / 2, (o.lo[2] + o.hi[2]) / 2];

/** The audience point a speaker at (x, y, z) facing `yaw` aims at: the back third of the seats, at ear height. */
function aimPoint(audience, x) {
  const a = audience;
  const tx = Math.min(a.x1, Math.max(a.x0, (x + (a.x0 + a.x1) / 2) / 2));
  const ty = a.y0 + 0.65 * (a.y1 - a.y0);
  return [tx, ty, a.ear_height ?? 1.2];
}

/** Yaw (deg) for a horizontal direction (dx, dy). */
const yawOf = (dx, dy) => Math.atan2(dx, dy) * DEG;

/** Tilt (deg) from (x, y, z) toward the audience along the speaker's facing. */
function pitchToward(pos, audience) {
  const [tx, ty, tz] = aimPoint(audience, pos[0]);
  return Math.atan2(tz - pos[2], Math.hypot(tx - pos[0], ty - pos[1])) * DEG;
}

/** The wall a box hangs on (within `gap` m), as an inward normal, or null. */
function wallOf(o, room, gap = 0.35) {
  const opts = [
    [o.lo[0], [1, 0]], [room.width - o.hi[0], [-1, 0]],
    [o.lo[1], [0, 1]], [room.length - o.hi[1], [0, -1]],
  ].filter(([d]) => d < gap).sort((a, b) => a[0] - b[0]);
  return opts.length ? opts[0][1] : null;
}

const round = (v, nd = 2) => Math.round(v * 10 ** nd) / 10 ** nd;

/**
 * Score a box-shaped part from its size and where it sits.
 * @returns {{score: number, kind: 'point'|'sub'|null, why: string}}
 */
export function shapeEvidence(o, room) {
  const [sx, sy, sz] = o.boxSize || size(o); // a line array is judged by one of its boxes
  const dims = [sx, sy, sz].sort((a, b) => a - b);
  const boxArea = 2 * (sx * sy + sy * sz + sx * sz) * (o.stacked || 1);
  const fill = o.area ? o.area / boxArea : 1; // a closed box covers its bounding box's surface once
  if (dims[2] > 1.3 || dims[0] < 0.12) return { score: 0, kind: null, why: 'not speaker-sized' };
  if (fill < 0.6 || fill > 1.6) return { score: 0, kind: null, why: 'not box-shaped' };
  const wide = Math.max(sx, sy); // along the wall it hangs on, roughly
  const onFloor = o.lo[2] < 0.15;
  if (onFloor) {
    // subs: a low, deep box near the stage; a tall or shallow one is furniture
    if (sz < 0.3 || sz > 1.0 || dims[1] < 0.4) return { score: 0, kind: null, why: 'floor box of the wrong size' };
    const nearStage = o.lo[1] < Math.max(3, room.length * 0.2);
    const score = 0.3 + (nearStage ? 0.3 : 0) + (dims[0] >= 0.45 ? 0.05 : 0);
    return { score, kind: 'sub', why: nearStage ? 'a floor box at the stage, sized like a subwoofer' : 'a floor box sized like a subwoofer, away from the stage' };
  }
  const high = o.lo[2] > 1.8;
  const wall = wallOf(o, room);
  const ceiling = o.hi[2] > room.height - 0.5;
  if (!high) return { score: 0.1, kind: 'point', why: 'a box at furniture height' };
  // speakers are about as tall as they are wide or taller; an air-conditioning unit or a light box is wide and flat
  const aspect = wide / Math.max(sz, 1e-3);
  if (aspect > 2.2 && !o.stacked) return { score: 0.05, kind: null, why: 'wide and flat (more like an air-conditioning unit)' };
  let score = 0.35 + (wall ? 0.15 : 0) + (ceiling && !wall ? 0.1 : 0) + (aspect < 1.2 ? 0.1 : 0);
  if (o.stacked) score = 0.8;
  return { score, kind: 'point', why: o.stacked ? `a stack of ${o.stacked} boxes, hung like a line array` : wall ? 'a speaker-sized box high on a wall' : 'a speaker-sized box hung high' };
}

/** Boxes of one column (same footprint, touching vertically) merged into one line-array candidate. */
export function mergeColumns(objs) {
  const used = new Set();
  const out = [];
  const sorted = [...objs].sort((a, b) => b.hi[2] - a.hi[2]);
  for (let i = 0; i < sorted.length; i++) {
    if (used.has(i)) continue;
    const col = [sorted[i]];
    used.add(i);
    for (let j = i + 1; j < sorted.length; j++) {
      if (used.has(j)) continue;
      const a = col.at(-1), b = sorted[j];
      const [ax, ay] = centre(a), [bx, by] = centre(b);
      const sa = size(a), sb = size(b);
      const sameFoot = Math.hypot(ax - bx, ay - by) < 0.25 && Math.abs(Math.max(sa[0], sa[1]) - Math.max(sb[0], sb[1])) < 0.2;
      const touching = a.lo[2] - b.hi[2] < 0.12 && a.lo[2] - b.hi[2] > -0.1;
      if (sameFoot && touching) { col.push(b); used.add(j); }
    }
    if (col.length < 2) { out.push(sorted[i]); continue; }
    const lo = [0, 1, 2].map((k) => Math.min(...col.map((c) => c.lo[k])));
    const hi = [0, 1, 2].map((k) => Math.max(...col.map((c) => c.hi[k])));
    out.push({ ...col[0], lo, hi, area: col.reduce((s, c) => s + (c.area || 0), 0), stacked: col.length,
      boxSize: size(col[0]), parts: col });
  }
  return out;
}

/**
 * Find candidate speakers.
 * @param {{objects?: Array<{name?: string, lo: number[], hi: number[], area?: number}>,
 *          fixtures?: Array<{name: string, spec?: string, pos: number[], fwd?: number[]}>}} found
 * @param {{room: {width: number, length: number, height: number}, audience: object, types?: object[]}} ctx
 * @returns {{candidates: object[], skippedFixtures: number}}
 */
export function findSpeakers(found, ctx) {
  const { room, audience, types = [] } = ctx;
  const cands = [];
  let skippedFixtures = 0;

  for (const f of found.fixtures || []) {
    const ev = [nameEvidence(f.name, types), nameEvidence(f.spec, types)].sort((a, b) => b.score - a.score)[0];
    const spec = nameEvidence(f.spec, types);
    const typeId = spec.typeId || ev.typeId;
    if (Math.max(ev.score, spec.score) < 0.5) { skippedFixtures++; continue; }
    const pos = f.pos;
    const role = ev.role || spec.role;
    const fwd = f.fwd || [0, 0, -1];
    const h = Math.hypot(fwd[0], fwd[1]);
    // GDTF puts a fixture's beam along its -Z; a hanging speaker whose -Z points down gets aimed at the seats
    const [ax, ay] = aimPoint(audience, pos[0]);
    const yaw = h > 0.5 ? yawOf(fwd[0], fwd[1]) : yawOf(ax - pos[0], ay - pos[1]);
    cands.push({ source: 'fixture', name: f.name || f.spec, why: `MVR fixture${f.spec ? ` (${f.spec})` : ''}`,
      confidence: Math.max(0.9, ev.score, spec.score), pos, yaw, pitch: role === 'sub' ? 0 : pitchToward(pos, audience),
      role: role || 'main', typeId, boxes: 0, lo: pos.map((v) => v - 0.2), hi: pos.map((v) => v + 0.2) });
  }

  const named = [], shapes = [];
  for (const o of found.objects || []) {
    const ev = nameEvidence(o.name, types);
    if (ev.score >= 0.5 && Math.max(...size(o)) <= 4) named.push({ o, ev }); // a named wall or room is not a speaker
    else shapes.push(o);
  }
  for (const { o, ev } of named) {
    const c = centre(o);
    const sub = ev.role === 'sub';
    const wall = wallOf(o, room);
    const [ax, ay] = aimPoint(audience, c[0]);
    const yaw = wall && !sub ? yawOf(wall[0], wall[1]) : yawOf(ax - c[0], ay - c[1]);
    const n = ev.array ? Math.max(2, Math.round(size(o)[2] / 0.3)) : 0;
    cands.push({ source: 'name', name: o.name, why: `named "${o.name}"`, confidence: ev.score,
      pos: sub ? [c[0], c[1], o.lo[2] + size(o)[2] / 2] : c, yaw, pitch: sub ? 0 : pitchToward(c, audience),
      role: ev.role || (sub ? 'sub' : 'main'), typeId: ev.typeId, boxes: n, lo: o.lo, hi: o.hi });
  }
  for (const o of mergeColumns(shapes)) {
    const ev = shapeEvidence(o, room);
    if (ev.score < 0.3 || !ev.kind) continue;
    const c = centre(o);
    const sub = ev.kind === 'sub';
    const wall = wallOf(o, room);
    const [ax, ay] = aimPoint(audience, c[0]);
    let yaw = wall && !sub ? yawOf(wall[0], wall[1]) : yawOf(ax - c[0], ay - c[1]);
    // facing away from the audience (a box on the back wall seen from behind the seats): keep, but trust it less
    const toward = Math.cos((yaw - yawOf(ax - c[0], ay - c[1])) / DEG);
    const confidence = ev.score - (toward < 0 ? 0.15 : 0);
    if (sub) yaw = 0;
    cands.push({ source: 'shape', name: null, why: ev.why, confidence, pos: c, yaw, pitch: sub ? 0 : pitchToward(c, audience),
      role: sub ? 'sub' : 'main', typeId: null, boxes: o.stacked || 0, boxH: o.boxSize ? o.boxSize[2] : 0, lo: o.lo, hi: o.hi });
  }

  // one candidate per place: a named mesh is also a box-shaped part, and a fixture may carry a named model
  cands.sort((a, b) => b.confidence - a.confidence);
  const kept = [];
  for (const c of cands) {
    const dup = kept.find((k) => Math.hypot(k.pos[0] - c.pos[0], k.pos[1] - c.pos[1], k.pos[2] - c.pos[2]) < 0.5
      || (c.lo[0] <= k.pos[0] && k.pos[0] <= c.hi[0] && c.lo[1] <= k.pos[1] && k.pos[1] <= c.hi[1] && c.lo[2] <= k.pos[2] && k.pos[2] <= c.hi[2]));
    if (dup) { if (!dup.typeId && c.typeId) dup.typeId = c.typeId; continue; }
    kept.push(c);
  }
  const inside = (c) => c.pos[0] > -0.5 && c.pos[0] < room.width + 0.5 && c.pos[1] > -0.5 && c.pos[1] < room.length + 0.5
    && c.pos[2] > -0.5 && c.pos[2] < room.height + 0.5;
  const candidates = kept.filter(inside).slice(0, 64).map((c, i) => ({
    ...c, id: `cand${i + 1}`, checked: c.confidence >= 0.5,
    x: round(Math.min(room.width - 0.1, Math.max(0.1, c.pos[0]))), y: round(Math.min(room.length - 0.1, Math.max(0.1, c.pos[1]))),
    z: round(Math.min(room.height - 0.1, Math.max(0.1, c.pos[2]))), yaw: round(c.yaw, 1), pitch: round(Math.max(-60, Math.min(30, c.pitch)), 1),
    confidence: round(Math.max(0, Math.min(1, c.confidence)), 2),
  }));
  return { candidates, skippedFixtures };
}
