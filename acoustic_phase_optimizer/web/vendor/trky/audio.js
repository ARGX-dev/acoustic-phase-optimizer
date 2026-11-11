/* TRKY audio module: lossless PCM coder (adaptive NLMS cascade + modelled residuals) and WAV support.
 * Decoding uses only + - * / Math.round and integer ops, so it is bit-exact in every browser. */
(function (T) {
  'use strict';
  const { Encoder, Decoder, squash, STR, DT, INIT, upd, tick } = T.util;
  const PAD = 320, LIM = 1 << 26;
  // Cascade: each stage predicts its input signal from its own past (taps) and the neighbouring
  // channel (cross), and hands the leftover to the next stage.
  let STAGES = [
    { taps: 256, cross: 0, mu: 0.004, eps: 16 },
    { taps: 16, cross: 8, mu: 0.04, eps: 16 },
    { taps: 16, cross: 0, mu: 0.05, eps: 16 },
  ];
  function tune(o) { if (o.STAGES) STAGES = o.STAGES; }

  /* ---------------------------------------------------------- predictor */
  class Predictor {
    constructor(nch, F) {
      this.nch = nch; this.K = STAGES.length;
      this.S = []; this.W = []; this.o = [];
      for (let k = 0; k < this.K; k++) {
        this.S.push([]); this.W.push([]);
        for (let c = 0; c < nch; c++) {
          this.S[k].push(new Int32Array(F + PAD));
          this.W[k].push(new Float64Array(STAGES[k].taps + STAGES[k].cross));
        }
      }
      for (let c = 0; c < nch; c++) this.o.push(nch === 1 ? 0 : (c === 0 ? 1 : c - 1));
      this.p = new Int32Array(this.K); this.en = new Float64Array(this.K);
    }
    predict(c, n) {
      const base = n + PAD;
      for (let k = 0; k < this.K; k++) {
        const st = STAGES[k], Sc = this.S[k][c], w = this.W[k][c], taps = st.taps;
        let s = 0, en = 0;
        for (let i = 0; i < taps; i++) { const h = Sc[base - 1 - i]; s += w[i] * h; en += h * h; }
        if (st.cross > 0 && this.nch > 1) {
          const o = this.o[c], So = this.S[k][o], off = o < c ? 0 : 1;
          for (let i = 0; i < st.cross; i++) { const h = So[base - off - i]; s += w[taps + i] * h; en += h * h; }
        }
        this.en[k] = en;
        let pk = Math.round(s); if (pk > LIM) pk = LIM; else if (pk < -LIM) pk = -LIM;
        this.p[k] = pk;
      }
    }
    // Encoder: feed the true diff d, fill every stage input, return final residual.
    forward(c, n, d) {
      const base = n + PAD; let v = d;
      for (let k = 0; k < this.K; k++) { this.S[k][c][base] = v; v -= this.p[k]; }
      return v;
    }
    // Decoder: rebuild stage inputs from the final residual r, return d.
    backward(c, n, r) {
      const base = n + PAD; let v = r;
      for (let k = this.K - 1; k >= 0; k--) { v += this.p[k]; this.S[k][c][base] = v; }
      return v;
    }
    adapt(c, n) {
      const base = n + PAD;
      for (let k = 0; k < this.K; k++) {
        const st = STAGES[k], Sc = this.S[k][c], w = this.W[k][c], taps = st.taps;
        const err = Sc[base] - this.p[k];
        const f = st.mu * err / (st.eps + this.en[k]);
        for (let i = 0; i < taps; i++) w[i] += f * Sc[base - 1 - i];
        if (st.cross > 0 && this.nch > 1) {
          const o = this.o[c], So = this.S[k][o], off = o < c ? 0 : 1;
          for (let i = 0; i < st.cross; i++) w[taps + i] += f * So[base - off - i];
        }
      }
    }
  }

  /* ----------------------------------------------- residual symbol model */
  const NI = 4;
  const nbits = (v) => (v === 0 ? 0 : 32 - Math.clz32(v));
  class ResCoder {
    constructor(nch, enc, dec) {
      this.e = enc; this.d = dec; this.nch = nch;
      this.s = [];
      for (let c = 0; c < nch; c++) {
        this.s.push({
          p1: 0, p2: 0, avg: 0,
          tA: new Uint32Array(32 * 32).fill(INIT),
          tB: new Uint32Array(64 * 32).fill(INIT),
          tC: new Uint32Array(256 * 32).fill(INIT),
          w: new Int32Array(32 * NI).fill(21845),
          m1: new Uint32Array(32 * 4).fill(INIT),
          m2: new Uint32Array(32 * 8).fill(INIT),
        });
        const s = this.s[c]; for (let j = 0; j < 32; j++) s.w[j * NI + NI - 1] = 0;
      }
      this.st = new Int32Array(NI);
    }
    bit(p, y) { if (this.d) return this.d.dec(p); this.e.enc(y, p); return y; }
    // z = zigzag residual (encoder) or ignored (decoder); returns z
    code(c, z) {
      const s = this.s[c], st = this.st;
      let nb = this.e ? nbits(z) : 0;
      const q = nbits(s.avg), ql = q * 2 + (q > 1 ? ((s.avg >> (q - 2)) & 1) : 0);
      const iA = (s.p1 > 31 ? 31 : s.p1) * 32, iB = (ql > 63 ? 63 : ql) * 32;
      const iC = ((s.p1 > 15 ? 15 : s.p1) * 16 + (s.p2 > 15 ? 15 : s.p2)) * 32;
      let j = 1;
      for (let k = 4; k >= 0; k--) {
        st[0] = STR[s.tA[iA + j] >>> 20]; st[1] = STR[s.tB[iB + j] >>> 20]; st[2] = STR[s.tC[iC + j] >>> 20];
        st[3] = 256;
        const wb = j * NI;
        let dot = 0; for (let i = 0; i < NI; i++) dot += st[i] * s.w[wb + i];
        dot = (dot / 65536) | 0; if (dot > 2047) dot = 2047; else if (dot < -2047) dot = -2047;
        let pr = squash(dot); if (pr < 1) pr = 1; else if (pr > 4095) pr = 4095;
        const y = this.bit(pr, (nb >> k) & 1);
        upd(s.tA, iA + j, y, 127); upd(s.tB, iB + j, y, 127); upd(s.tC, iC + j, y, 127);
        const err = (y << 12) - pr;
        for (let i = 0; i < NI; i++) s.w[wb + i] += (st[i] * err * 5 + 4096) >> 13;
        j = (j << 1) | y;
      }
      nb = j & 31;
      let out = 0;
      if (nb > 0) {
        out = 1;
        for (let i = nb - 2; i >= 0; i--) {
          const pos = nb - 2 - i; let p, ci = -1, tbl = null;
          if (pos === 0) { tbl = s.m1; ci = nb * 4 + (ql >> 4 > 3 ? 3 : ql >> 4); }
          else if (pos === 1) { tbl = s.m2; ci = nb * 8 + ((out & 1) << 1) + (ql & 1); }
          p = tbl ? (tbl[ci] >>> 20) : 2048; if (p < 1) p = 1; else if (p > 4095) p = 4095;
          const y = this.bit(p, (z >> i) & 1);
          if (tbl) upd(tbl, ci, y, 255);
          out = (out << 1) | y;
        }
      }
      s.p2 = s.p1; s.p1 = nb;
      s.avg += ((out > 134217727 ? 134217727 : out) * 16 - s.avg) >> 4;
      return out;
    }
  }

  /* ------------------------------------------- PCM stream (encode/decode) */
  // chans: array of Int32Array (already shifted right by `sh`)
  async function encodePCM(chans, F, progress) {
    const nch = chans.length, P = new Predictor(nch, F);
    const e = new Encoder((F * nch) + 4096), rc = new ResCoder(nch, e, null);
    const prev = new Int32Array(nch);
    for (let s = 0; s < F; s += 16384) {
      const end = Math.min(F, s + 16384);
      for (let n = s; n < end; n++) {
        for (let c = 0; c < nch; c++) {
          const x = chans[c][n], d = x - prev[c]; prev[c] = x;
          P.predict(c, n);
          const r = P.forward(c, n, d);
          rc.code(c, ((r << 1) ^ (r >> 31)) >>> 0);
          P.adapt(c, n);
        }
      }
      progress && progress(end / F);
      if (end < F) await tick();
    }
    return e.finish();
  }
  async function decodePCM(bytes, pos, nch, F, progress) {
    const P = new Predictor(nch, F), dec = new Decoder(bytes, pos), rc = new ResCoder(nch, null, dec);
    const out = []; for (let c = 0; c < nch; c++) out.push(new Int32Array(F));
    const prev = new Int32Array(nch);
    for (let s = 0; s < F; s += 16384) {
      const end = Math.min(F, s + 16384);
      for (let n = s; n < end; n++) {
        for (let c = 0; c < nch; c++) {
          P.predict(c, n);
          const z = rc.code(c, 0), r = (z >>> 1) ^ -(z & 1);
          const d = P.backward(c, n, r);
          const x = prev[c] + d; prev[c] = x; out[c][n] = x;
          P.adapt(c, n);
        }
      }
      progress && progress(end / F);
      if (end < F) await tick();
    }
    return out;
  }

  /* ----------------------------------------------------------------- WAV */
  function parseWav(d) {
    if (d.length < 44 || d[0] !== 0x52 || d[1] !== 0x49 || d[2] !== 0x46 || d[3] !== 0x46 || d[8] !== 0x57 || d[9] !== 0x41 || d[10] !== 0x56 || d[11] !== 0x45) return null;
    const dv = new DataView(d.buffer, d.byteOffset, d.length);
    let o = 12, fmt = null;
    while (o + 8 <= d.length) {
      const id = String.fromCharCode(d[o], d[o + 1], d[o + 2], d[o + 3]);
      let sz = dv.getUint32(o + 4, true);
      if (id === 'fmt ' && o + 8 + 16 <= d.length) {
        let tag = dv.getUint16(o + 8, true);
        if (tag === 0xfffe && sz >= 26) tag = dv.getUint16(o + 8 + 24, true);
        fmt = { tag, ch: dv.getUint16(o + 10, true), bits: dv.getUint16(o + 22, true), align: dv.getUint16(o + 20, true) };
      } else if (id === 'data') {
        if (!fmt || fmt.tag !== 1) return null;
        const start = o + 8; if (sz > d.length - start) sz = d.length - start;
        const bytes = fmt.bits >> 3;
        if (![8, 16, 24].includes(fmt.bits) || fmt.ch < 1 || fmt.ch > 8 || fmt.align !== bytes * fmt.ch) return null;
        const F = Math.floor(sz / fmt.align);
        return { ch: fmt.ch, bits: fmt.bits, start, F, end: start + F * fmt.align };
      }
      o += 8 + sz + (sz & 1);
    }
    return null;
  }
  function wavToChans(d, w) {
    const bytes = w.bits >> 3, chans = [];
    for (let c = 0; c < w.ch; c++) chans.push(new Int32Array(w.F));
    let p = w.start;
    for (let n = 0; n < w.F; n++) for (let c = 0; c < w.ch; c++) {
      let v;
      if (bytes === 1) v = d[p] - 128;
      else if (bytes === 2) v = ((d[p] | (d[p + 1] << 8)) << 16) >> 16;
      else v = ((d[p] | (d[p + 1] << 8) | (d[p + 2] << 16)) << 8) >> 8;
      chans[c][n] = v; p += bytes;
    }
    return chans;
  }
  function chansToBytes(chans, F, bits, dst, p) {
    const bytes = bits >> 3, nch = chans.length;
    for (let n = 0; n < F; n++) for (let c = 0; c < nch; c++) {
      const v = chans[c][n];
      if (bytes === 1) dst[p] = v + 128;
      else if (bytes === 2) { dst[p] = v & 255; dst[p + 1] = (v >> 8) & 255; }
      else { dst[p] = v & 255; dst[p + 1] = (v >> 8) & 255; dst[p + 2] = (v >> 16) & 255; }
      p += bytes;
    }
  }
  function trailingShift(chans, F) {
    let m = 0;
    for (const ch of chans) for (let i = 0; i < F; i++) m |= ch[i];
    if (m === 0) return 0;
    return 31 - Math.clz32(m & -m);
  }
  function applyShift(chans, F, sh, dir) {
    if (!sh) return;
    for (const ch of chans) for (let i = 0; i < F; i++) ch[i] = dir > 0 ? ch[i] << sh : ch[i] >> sh;
  }

  T.register('pcm', {
    async enc(d, ctx) {
      const w = parseWav(d); if (!w || w.F < 64) return null;
      const chans = wavToChans(d, w), sh = trailingShift(chans, w.F);
      applyShift(chans, w.F, sh, -1);
      const hdr = d.subarray(0, w.start), tail = d.subarray(w.end);
      const stream = await encodePCM(chans, w.F, ctx.sub('Compressing audio'));
      const blob = T.util.concat([hdr, tail, stream]);
      return { blob, meta: { ch: w.ch, bits: w.bits, F: w.F, hl: hdr.length, tl: tail.length, sh } };
    },
    async dec(blob, e, ctx) {
      const m = e.x, hl = m.hl, tl = m.tl;
      const chans = await decodePCM(blob, hl + tl, m.ch, m.F, ctx.sub('Restoring audio'));
      applyShift(chans, m.F, m.sh, 1);
      const out = new Uint8Array(hl + m.F * m.ch * (m.bits >> 3) + tl);
      out.set(blob.subarray(0, hl), 0);
      chansToBytes(chans, m.F, m.bits, out, hl);
      out.set(blob.subarray(hl, hl + tl), out.length - tl);
      return out;
    },
  });

  const isWav = (d) => d.length > 44 && d[0] === 0x52 && d[1] === 0x49 && d[2] === 0x46 && d[3] === 0x46 && d[8] === 0x57;
  T.addSniffer((d) => (isWav(d) && parseWav(d) ? ['pcm'] : null));

  T.audio = { tune, encodePCM, decodePCM, parseWav, wavToChans, trailingShift, applyShift, PAD };
})(typeof module === 'object' && module.exports ? require('./core.js') : TRKY);

