// Aim a speaker at its audience: a deterministic rule a system tech would use, not an optimizer pass.
//
// Point sources aim at the back third of the seats they serve. A line-array hang gets its top box on the
// last row and its bottom box on the first row it serves, by solving the tilt and the splay between boxes
// (constant-curvature boxes keep their fixed splay). Ear heights follow raked seating.

import { earAt, hangFields } from './store.js';
import { clamp, fmt } from './util.js';

const DEG = 180 / Math.PI;
const r1 = (v, step) => Math.round(v / step) * step;

/**
 * @returns {{patch: object|null, note: string}} the speaker fields to set, and what was done (or why not).
 */
export function aimPatch(s, project, type) {
  const a = project.audience;
  if (s.role === 'sub' || s.coverage_h >= 300) return { patch: null, note: `${s.name} radiates in every direction at low frequencies; there is nothing to aim.` };
  const near = Math.max(a.y0, s.y + 0.5);
  if (near >= a.y1 - 0.2) return { patch: null, note: `${s.name} is behind the audience area.` };
  const far = s.role === 'front_fill' ? Math.min(a.y1, near + 6) : a.y1; // front fills cover the first rows
  const cx = (a.x0 + a.x1) / 2;
  const tx = clamp((s.x + cx) / 2, a.x0, a.x1); // toe in halfway toward the centre line
  const ty = near + 0.65 * (far - near);
  const yaw = r1(Math.atan2(tx - s.x, ty - s.y) * DEG, 0.5);
  const elev = (y) => Math.atan2(earAt(a, y) - s.z, Math.hypot(tx - s.x, y - s.y)) * DEG;

  if (s.array_boxes >= 2 && type?.spec?.kind === 'element') {
    const n = s.array_boxes;
    const top = elev(far), bottom = elev(near);
    const max = type.spec.splay_max ?? 10;
    let splay = type.spec.constant_curvature ? s.array_splay : clamp((top - bottom) / (n - 1), 0, max);
    splay = r1(splay, 0.5);
    const pitch = r1(top - ((n - 1) / 2) * splay, 0.5); // top box on the last row
    const reach = pitch - ((n - 1) / 2) * splay; // where the bottom box points
    let note = `Aimed ${s.name}: turn ${fmt(yaw, 1)}°, tilt ${fmt(pitch, 1)}°, ${fmt(splay, 1)}° between boxes. The top box points at the last row`;
    if (reach > bottom + 1) {
      note += type.spec.constant_curvature || splay >= max
        ? `; the bottom box falls short of the first rows by ${fmt(reach - bottom, 0)}°: add boxes or front fills.`
        : '.';
    } else note += ', the bottom box at the first row.';
    if (!type.spec.constant_curvature && (top - bottom) / (n - 1) > max) note = note.replace(' between boxes', ` between boxes (the most this box allows)`);
    return { patch: { yaw, pitch, ...hangFields(type, n, splay) }, note };
  }
  const pitch = r1(elev(ty), 0.5);
  return { patch: { yaw, pitch }, note: `Aimed ${s.name}: turn ${fmt(yaw, 1)}°, tilt ${fmt(pitch, 1)}°, at the back third of its seats.` };
}
