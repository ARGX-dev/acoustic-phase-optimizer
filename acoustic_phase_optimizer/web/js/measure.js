// Measured responses: import a REW / Smaart / generic text export and overlay it on the prediction.
// Files are parsed here; the project keeps a decimated copy (1/48 octave), so a venue file stays small.

const MAX_POINTS = 1200;

/**
 * Parse a frequency-response text export. Accepts REW "Export measurement as text" (lines starting with
 * '*' are comments), Smaart ASCII exports (a header row, then tab-separated numbers) and any CSV/TSV whose
 * first columns are frequency (Hz), level (dB) and optionally phase (degrees).
 */
export function parseResponse(text) {
  const freqs = [], spl = [], phase = [];
  let anyPhase = false;
  for (const raw of String(text).split(/\r?\n/)) {
    const line = raw.trim();
    if (!line || /^[*#;/"]/.test(line) || !/^[-+]?\d/.test(line)) continue;
    // A semicolon or tab separated file may use decimal commas; otherwise commas separate fields.
    const usesComma = /[;\t]/.test(line);
    const parts = line.split(usesComma ? /[;\t]+/ : /[\s,]+/).map((x) => x.trim()).filter(Boolean)
      .map((x) => Number(usesComma ? x.replace(',', '.') : x));
    if (parts.length < 2 || !parts.slice(0, 2).every(Number.isFinite)) continue;
    const [f, db, ph] = parts;
    if (f <= 0 || f > 48000 || Math.abs(db) > 250) continue;
    freqs.push(f); spl.push(db);
    const okPh = Number.isFinite(ph) && Math.abs(ph) <= 3600;
    phase.push(okPh ? ph : null);
    anyPhase ||= okPh;
  }
  if (freqs.length < 8) throw new Error('No frequency response found. Expected rows of frequency (Hz), level (dB) and optionally phase (°).');
  // sort, drop duplicates
  const idx = freqs.map((_, i) => i).sort((a, b) => freqs[a] - freqs[b]);
  const out = { freqs: [], spl: [], phase: anyPhase ? [] : null };
  let last = 0;
  for (const i of idx) {
    if (freqs[i] <= last) continue;
    last = freqs[i];
    out.freqs.push(freqs[i]); out.spl.push(spl[i]);
    if (anyPhase) out.phase.push(phase[i]);
  }
  return decimate(out);
}

/** Thin a dense (often linear-spaced) response to at most 1/48 octave, averaging power within each bin. */
function decimate(m) {
  if (m.freqs.length <= MAX_POINTS) return round(m);
  const out = { freqs: [], spl: [], phase: m.phase ? [] : null };
  const step = Math.log2(m.freqs.at(-1) / m.freqs[0]) / Math.min(MAX_POINTS, 48 * Math.log2(m.freqs.at(-1) / m.freqs[0]));
  let i = 0;
  while (i < m.freqs.length) {
    const edge = m.freqs[i] * 2 ** step;
    let p = 0, n = 0, lf = 0, re = 0, im = 0;
    const j0 = i;
    for (; i < m.freqs.length && (m.freqs[i] < edge || i === j0); i++) {
      p += 10 ** (m.spl[i] / 10); lf += Math.log(m.freqs[i]); n++;
      if (m.phase && m.phase[i] != null) { re += Math.cos((m.phase[i] * Math.PI) / 180); im += Math.sin((m.phase[i] * Math.PI) / 180); }
    }
    out.freqs.push(Math.exp(lf / n)); out.spl.push(10 * Math.log10(p / n));
    if (m.phase) out.phase.push(re || im ? (Math.atan2(im, re) * 180) / Math.PI : null);
  }
  return round(out);
}

function round(m) {
  return { freqs: m.freqs.map((f) => Math.round(f * 100) / 100), spl: m.spl.map((v) => Math.round(v * 100) / 100),
    phase: m.phase ? m.phase.map((v) => (v == null ? null : Math.round(v * 10) / 10)) : null };
}

/**
 * Values of a measurement on another frequency grid (the chart's), smoothed to 1/`fraction` octave in power
 * so it compares like for like with the predicted curve. Outside the measured range: null.
 */
export function resample(m, grid, { fraction = 6, key = 'spl' } = {}) {
  const f = m.freqs, v = m[key];
  if (!v) return grid.map(() => null);
  const lf = f.map(Math.log2);
  const half = 0.5 / fraction;
  return grid.map((x) => {
    if (x < f[0] || x > f.at(-1)) return null;
    const c = Math.log2(x);
    if (key === 'phase') { // nearest point: phase is not averaged across a band
      let lo = 0, hi = f.length - 1;
      while (hi - lo > 1) { const mid = (lo + hi) >> 1; if (f[mid] < x) lo = mid; else hi = mid; }
      return v[x - f[lo] < f[hi] - x ? lo : hi];
    }
    let p = 0, n = 0;
    for (let i = 0; i < f.length; i++) {
      if (lf[i] < c - half) continue;
      if (lf[i] > c + half) break;
      if (v[i] != null) { p += 10 ** (v[i] / 10); n++; }
    }
    if (!n) { // sparser than the window: interpolate
      let i = f.findIndex((q) => q >= x);
      if (i <= 0) return v[0];
      const t = (c - lf[i - 1]) / (lf[i] - lf[i - 1]);
      return v[i - 1] + t * (v[i] - v[i - 1]);
    }
    return 10 * Math.log10(p / n);
  });
}

/** Offset (dB) that lines a measurement up with a predicted curve over 200 Hz - 2 kHz (median difference). */
export function matchOffset(m, grid, predicted) {
  const meas = resample(m, grid);
  const d = grid.map((x, i) => (x >= 200 && x <= 2000 && meas[i] != null && predicted[i] != null ? predicted[i] - meas[i] : null))
    .filter((x) => x != null).sort((a, b) => a - b);
  return d.length ? Math.round(d[Math.floor(d.length / 2)] * 10) / 10 : 0;
}
