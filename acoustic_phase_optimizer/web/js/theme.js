// Colour system: one validated palette, selected per theme (never an automatic flip).
// Sequential = one hue, diverging = two hues around a neutral grey, categorical = fixed slots.

export const isDark = () => document.documentElement.dataset.theme !== 'light';
export const cssVar = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();

const SERIES = {
  dark: ['#3987e5', '#d95926', '#199e70', '#c98500', '#d55181', '#008300', '#9085e9', '#e66767'],
  light: ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#008300', '#6250d6', '#e34948'],
};
/** Categorical colour for a speaker's fixed slot. */
export const seriesColor = (slot) => SERIES[isDark() ? 'dark' : 'light'][((slot ?? 0) % 8 + 8) % 8];

// Blue ramp, steps 100 -> 700. On a dark surface low values recede into the surface, so the ramp runs dark -> light.
const BLUE = ['#cde2fb', '#9ec5f4', '#6da7ec', '#3987e5', '#256abf', '#184f95', '#0d366b'];
const STOPS = {
  sequential: { light: BLUE, dark: ['#10233d', '#0d366b', '#184f95', '#256abf', '#3987e5', '#6da7ec', '#cde2fb'] },
  diverging: {
    light: ['#a32626', '#d03b3b', '#e99a94', '#f0efec', '#9ec5f4', '#2a78d6', '#104281'],
    dark: ['#f2a4a0', '#e66767', '#8a4a48', '#383835', '#2f5a8f', '#3987e5', '#9ec5f4'],
  },
};

const hex = (c) => [1, 3, 5].map((i) => parseInt(c.slice(i, i + 2), 16));

/** A colour scale over [lo, hi] with a 256-entry lookup table. */
export function makeScale(kind, lo, hi) {
  const stops = STOPS[kind][isDark() ? 'dark' : 'light'];
  const rgb = stops.map(hex);
  const lut = new Uint8ClampedArray(256 * 3);
  for (let i = 0; i < 256; i++) {
    const t = (i / 255) * (rgb.length - 1);
    const k = Math.min(Math.floor(t), rgb.length - 2);
    const f = t - k;
    for (let c = 0; c < 3; c++) lut[i * 3 + c] = rgb[k][c] * (1 - f) + rgb[k + 1][c] * f;
  }
  const span = Math.max(hi - lo, 1e-9);
  return {
    kind, lo, hi, lut,
    css: `linear-gradient(to right, ${stops.join(', ')})`,
    index: (v) => Math.max(0, Math.min(255, Math.round(((v - lo) / span) * 255))),
  };
}

export function toggleTheme() {
  const next = isDark() ? 'light' : 'dark';
  document.documentElement.dataset.theme = next;
  try { localStorage.setItem('apo-theme', next); } catch (e) { /* private mode */ }
  return next;
}
