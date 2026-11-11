// Module worker: repairs one mesh off the page thread (big models). Message: { positions, index, opts } -> { index, triSource, report }.
import { repair } from './meshfix.js';

self.onmessage = (e) => {
  try {
    const { positions, index, opts } = e.data;
    const r = repair(positions, index, opts);
    self.postMessage({ ok: true, ...r }, [r.index.buffer, r.triSource.buffer]);
  } catch (err) {
    self.postMessage({ ok: false, error: String(err?.message || err) });
  }
};
