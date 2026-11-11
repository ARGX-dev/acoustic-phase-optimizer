// Display units. The model is always in metres; only what is shown and typed changes.

import { state } from './store.js';
import { fmt } from './util.js';

export const FT = 0.3048;
export const imperial = () => state.ui.units === 'imperial';

/** 'm' or 'ft'. */
export const lenUnit = () => (imperial() ? 'ft' : 'm');
/** Metres to the display unit. */
export const toDisp = (m) => (m == null ? m : imperial() ? m / FT : m);
/** Display unit to metres. */
export const fromDisp = (v) => (v == null ? v : imperial() ? v * FT : v);

/** A length with its unit, e.g. "12.5 m" or "41.0 ft". */
export const len = (m, nd = 1) => `${fmt(toDisp(m), nd)} ${lenUnit()}`;
/** Just the number in display units. */
export const lenNum = (m, nd = 1) => fmt(toDisp(m), nd);
export const area = (m2) => (imperial()
  ? `${Math.round(m2 / (FT * FT)).toLocaleString()} ft²` : `${Math.round(m2).toLocaleString()} m²`);
export const volume = (m3) => (imperial()
  ? `${Math.round(m3 / (FT * FT * FT)).toLocaleString()} ft³` : `${Math.round(m3).toLocaleString()} m³`);
export const speed = (ms) => (imperial() ? `${fmt(ms / FT, 0)} ft/s` : `${fmt(ms, 1)} m/s`);

const NICE = [0.1, 0.25, 0.5, 1, 2, 5, 10, 25, 50, 100, 200, 500];
/** A field step in display units: a metric step converted, then snapped to a round number of feet. */
export function dispStep(stepM) {
  if (!imperial()) return stepM;
  const ft = stepM / FT;
  return NICE.reduce((a, b) => (Math.abs(b - ft) < Math.abs(a - ft) ? b : a));
}

/** Arrow-key nudge in metres: 0.1 m / 1 m, or 6 in / 3 ft. */
export const nudge = (big) => (imperial() ? (big ? 3 : 0.5) * FT : big ? 1 : 0.1);
export const nudgeLabel = () => (imperial() ? 'Nudge 6 in (Shift: 3 ft)' : 'Nudge 0.1 m (Shift: 1 m)');

/** Axis tick spacing in metres for a scale in pixels per metre, on round display numbers. */
export function tickStep(pxPerMetre, minPx = 46) {
  const per = imperial() ? pxPerMetre * FT : pxPerMetre; // pixels per display unit
  for (const s of [1, 2, 5, 10, 20, 50, 100, 200]) if (s * per >= minPx) return fromDisp(s);
  return fromDisp(500);
}
