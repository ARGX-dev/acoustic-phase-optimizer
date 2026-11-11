/* TRKY core: runs unchanged in browsers and Node. No dependencies, no network.
 *
 * .trky v4 container
 *   8 bytes   "TRKY" 04 00 00 00
 *   1 byte    index flag (0 = plain JSON, 1 = deflate-raw JSON)
 *   4 bytes   index length (u32 LE)
 *   N bytes   index
 *   ...       entry blobs, back to back
 * Index entry: { n:name, m:method, z:original size, h:sha256, o:offset, l:length, x:meta }
 *
 * Determinism rule: everything that affects decoding uses integer maths or
 * IEEE + - * / only (never Math.exp/log/pow), so any browser decodes any file.
 */
(function (root, factory) {
  if (typeof module === 'object' && module.exports) module.exports = factory();
  else root.TRKY = factory();
})(typeof self !== 'undefined' ? self : this, function () {
  'use strict';

  const MAGIC = [0x54, 0x52, 0x4b, 0x59, 4, 0, 0, 0];
  const te = new TextEncoder(), td = new TextDecoder();
  const state = { cancel: false };
  const tick = () => new Promise((r) => setTimeout(r, 0)).then(() => { if (state.cancel) throw new Error('Cancelled'); });
  const CHUNK = 1 << 16;

  /* ------------------------------------------------------------------ SHA-256 */
  const K256 = new Uint32Array([
    0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5,
    0xd807aa98, 0x12835b01, 0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174,
    0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f, 0x4a7484aa, 0x5cb0a9dc, 0x76f988da,
    0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
    0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85,
    0xa2bfe8a1, 0xa81a664b, 0xc24b8b70, 0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070,
    0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a, 0x5b9cca4f, 0x682e6ff3,
    0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2]);

  function sha256js(data) {
    const H = new Uint32Array([0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19]);
    const W = new Uint32Array(64);
    const len = data.length, total = (((len + 9 + 63) >> 6) << 6);
    const tail = new Uint8Array(total - (len & ~63));
    tail.set(data.subarray(len & ~63));
    tail[len & 63] = 0x80;
    const dv = new DataView(tail.buffer);
    dv.setUint32(tail.length - 8, Math.floor(len / 0x20000000), false);
    dv.setUint32(tail.length - 4, (len << 3) >>> 0, false);
    const block = (u8, o) => {
      for (let i = 0; i < 16; i++) W[i] = (u8[o + 4 * i] << 24) | (u8[o + 4 * i + 1] << 16) | (u8[o + 4 * i + 2] << 8) | u8[o + 4 * i + 3];
      for (let i = 16; i < 64; i++) {
        const a = W[i - 15], b = W[i - 2];
        const s0 = ((a >>> 7) | (a << 25)) ^ ((a >>> 18) | (a << 14)) ^ (a >>> 3);
        const s1 = ((b >>> 17) | (b << 15)) ^ ((b >>> 19) | (b << 13)) ^ (b >>> 10);
        W[i] = (W[i - 16] + s0 + W[i - 7] + s1) | 0;
      }
      let a = H[0], b = H[1], c = H[2], d = H[3], e = H[4], f = H[5], g = H[6], h = H[7];
      for (let i = 0; i < 64; i++) {
        const S1 = ((e >>> 6) | (e << 26)) ^ ((e >>> 11) | (e << 21)) ^ ((e >>> 25) | (e << 7));
        const ch = (e & f) ^ (~e & g);
        const t1 = (h + S1 + ch + K256[i] + W[i]) | 0;
        const S0 = ((a >>> 2) | (a << 30)) ^ ((a >>> 13) | (a << 19)) ^ ((a >>> 22) | (a << 10));
        const mj = (a & b) ^ (a & c) ^ (b & c);
        const t2 = (S0 + mj) | 0;
        h = g; g = f; f = e; e = (d + t1) | 0; d = c; c = b; b = a; a = (t1 + t2) | 0;
      }
      H[0] += a; H[1] += b; H[2] += c; H[3] += d; H[4] += e; H[5] += f; H[6] += g; H[7] += h;
    };
    const full = len & ~63;
    for (let o = 0; o < full; o += 64) block(data, o);
    for (let o = 0; o < tail.length; o += 64) block(tail, o);
    let s = '';
    for (let i = 0; i < 8; i++) s += H[i].toString(16).padStart(8, '0');
    return s;
  }
  async function sha256(data) {
    try {
      const c = (typeof crypto !== 'undefined' && crypto.subtle) ? crypto : null;
      if (c) {
        const d = new Uint8Array(await c.subtle.digest('SHA-256', data));
        let s = ''; for (let i = 0; i < 32; i++) s += d[i].toString(16).padStart(2, '0');
        return s;
      }
    } catch (_) { /* fall back */ }
    return sha256js(data);
  }

  /* ------------------------------------------------------- shared CM tables */
  const SQ_T = [1, 2, 3, 6, 10, 16, 27, 45, 73, 120, 194, 310, 488, 747, 1101, 1546, 2047, 2549, 2994, 3348,
    3607, 3785, 3901, 3975, 4022, 4050, 4068, 4079, 4085, 4089, 4092, 4093, 4094];
  function squash(d) {
    if (d > 2047) return 4095;
    if (d < -2047) return 0;
    const w = d & 127; d = (d >> 7) + 16;
    return (SQ_T[d] * (128 - w) + SQ_T[d + 1] * w + 64) >> 7;
  }
  const STR = new Int16Array(4096);
  (function () {
    let pi = 0;
    for (let x = -2047; x <= 2047; x++) {
      const v = squash(x);
      for (let i = pi; i <= v; i++) STR[i] = x;
      pi = v + 1;
    }
    for (let i = pi; i < 4096; i++) STR[i] = 2047;
  })();
  const DT = new Int32Array(1024);
  for (let i = 0; i < 1024; i++) DT[i] = Math.floor(65536 / (i + 1.5));
  const INIT = 0x80000000;

  // adaptive probability counter: top 22 bits = P(1), low 10 bits = hit count
  function upd(t, i, y, lim) {
    const v = t[i], n = v & 1023, p = v >>> 10;
    const np = p + ((((y ? 4194303 : 0) - p) * DT[n]) / 65536 | 0);
    t[i] = ((np << 10) | (n < lim ? n + 1 : n)) >>> 0;
  }

  /* --------------------------------------------------------- binary arith coder */
  class Encoder {
    constructor(cap) { this.x1 = 0; this.x2 = 0xffffffff; this.out = new Uint8Array(Math.max(cap, 1024)); this.n = 0; }
    put(b) {
      if (this.n >= this.out.length) { const o = new Uint8Array(this.out.length * 2); o.set(this.out); this.out = o; }
      this.out[this.n++] = b;
    }
    enc(y, p) { // p = P(y=1) in 1..4095
      const xmid = (this.x1 + (((this.x2 - this.x1) >>> 12) * p)) >>> 0;
      if (y) this.x2 = xmid; else this.x1 = (xmid + 1) >>> 0;
      while (((this.x1 ^ this.x2) & 0xff000000) === 0) {
        this.put(this.x2 >>> 24);
        this.x1 = (this.x1 << 8) >>> 0;
        this.x2 = ((this.x2 << 8) | 255) >>> 0;
      }
    }
    finish() {
      this.put(this.x1 >>> 24); this.put((this.x1 >>> 16) & 255); this.put((this.x1 >>> 8) & 255); this.put(this.x1 & 255);
      return this.out.subarray(0, this.n);
    }
  }
  class Decoder {
    constructor(buf, pos) {
      this.buf = buf; this.pos = pos || 0; this.x1 = 0; this.x2 = 0xffffffff; this.x = 0;
      for (let i = 0; i < 4; i++) this.x = ((this.x << 8) | this.next()) >>> 0;
    }
    next() { return this.pos < this.buf.length ? this.buf[this.pos++] : 0; }
    dec(p) {
      const xmid = (this.x1 + (((this.x2 - this.x1) >>> 12) * p)) >>> 0;
      let y;
      if (this.x <= xmid) { y = 1; this.x2 = xmid; } else { y = 0; this.x1 = (xmid + 1) >>> 0; }
      while (((this.x1 ^ this.x2) & 0xff000000) === 0) {
        this.x1 = (this.x1 << 8) >>> 0;
        this.x2 = ((this.x2 << 8) | 255) >>> 0;
        this.x = ((this.x << 8) | this.next()) >>> 0;
      }
      return y;
    }
  }

  /* ----------------------------------------------- general byte model (CM) */
  const NIN = 8;
  function H(a, b, k) {
    let h = Math.imul(a, 0x9e3779b1) ^ Math.imul(b, 0x85ebca6b) ^ Math.imul(k, 0xc2b2ae35);
    h ^= h >>> 15; h = Math.imul(h, 0x2c1b3c6d); h ^= h >>> 12;
    return h;
  }
  class ByteModel {
    constructor(len, buf) {
      let tb = 16; while ((1 << tb) < len * 2 && tb < 22) tb++;
      this.tb = tb; this.buf = buf; this.pos = 0;
      const N = 1 << tb;
      this.t0 = new Uint32Array(256).fill(INIT);
      this.t1 = new Uint32Array(65536).fill(INIT);
      this.t2 = new Uint32Array(N).fill(INIT);
      this.t3 = new Uint32Array(N).fill(INIT);
      this.t4 = new Uint32Array(N).fill(INIT);
      this.t6 = new Uint32Array(N).fill(INIT);
      this.mm = new Uint32Array(64).fill(INIT);
      this.mtb = Math.min(tb, 20);
      this.mtab = new Int32Array(1 << this.mtb);
      this.w = new Int32Array(768 * NIN).fill(16384);
      this.st = new Int32Array(NIN);
      this.c0 = 1; this.c4 = 0; this.c8 = 0; this.bp = 0;
      this.h2 = this.h3 = this.h4 = this.h6 = 0;
      this.i2 = this.i3 = this.i4 = this.i6 = 0;
      this.mptr = 0; this.mlen = 0; this.mctx = -1;
      this.ws = 0; this.pr = 2048;
    }
    p() {
      const c0 = this.c0, sh = 32 - this.tb, st = this.st;
      const m = Math.imul(c0, 0x7feb352d);
      this.i2 = Math.imul(this.h2 ^ m, 0x846ca68b) >>> sh;
      this.i3 = Math.imul(this.h3 ^ m, 0x846ca68b) >>> sh;
      this.i4 = Math.imul(this.h4 ^ m, 0x846ca68b) >>> sh;
      this.i6 = Math.imul(this.h6 ^ m, 0x846ca68b) >>> sh;
      st[0] = STR[this.t0[c0] >>> 20];
      st[1] = STR[this.t1[((this.c4 & 255) << 8) | c0] >>> 20];
      st[2] = STR[this.t2[this.i2] >>> 20];
      st[3] = STR[this.t3[this.i3] >>> 20];
      st[4] = STR[this.t4[this.i4] >>> 20];
      st[5] = STR[this.t6[this.i6] >>> 20];
      this.mctx = -1; st[6] = 0;
      if (this.mlen > 0) {
        const e = this.buf[this.mptr];
        if (((e | 256) >> (8 - this.bp)) === c0) {
          this.mctx = (this.mlen > 31 ? 31 : this.mlen) * 2 + ((e >> (7 - this.bp)) & 1);
          st[6] = STR[this.mm[this.mctx] >>> 20];
        } else this.mlen = 0;
      }
      st[7] = 256;
      const ws = (c0 + 256 * (this.mctx < 0 ? 0 : (this.mlen < 16 ? 1 : 2))) * NIN;
      this.ws = ws;
      let dot = 0;
      for (let i = 0; i < NIN; i++) dot += st[i] * this.w[ws + i];
      dot = (dot / 65536) | 0;
      if (dot > 2047) dot = 2047; else if (dot < -2047) dot = -2047;
      let pr = squash(dot);
      if (pr < 1) pr = 1; else if (pr > 4095) pr = 4095;
      return (this.pr = pr);
    }
    update(y) {
      upd(this.t0, this.c0, y, 255);
      upd(this.t1, ((this.c4 & 255) << 8) | this.c0, y, 255);
      upd(this.t2, this.i2, y, 255); upd(this.t3, this.i3, y, 255);
      upd(this.t4, this.i4, y, 255); upd(this.t6, this.i6, y, 255);
      if (this.mctx >= 0) upd(this.mm, this.mctx, y, 255);
      const err = (y << 12) - this.pr, ws = this.ws, st = this.st, w = this.w;
      for (let i = 0; i < NIN; i++) w[ws + i] += (st[i] * err * 5 + 4096) >> 13;
      this.c0 = (this.c0 << 1) | y; this.bp++;
      if (this.bp === 8) this.byteDone(this.c0 & 255);
    }
    byteDone(c) {
      const buf = this.buf; buf[this.pos++] = c;
      this.c8 = ((this.c8 << 8) | (this.c4 >>> 24)) >>> 0;
      this.c4 = ((this.c4 << 8) | c) >>> 0;
      this.c0 = 1; this.bp = 0;
      const c4 = this.c4, c8 = this.c8 & 0xffff;
      this.h2 = H(c4 & 0xffff, 0, 2); this.h3 = H(c4 & 0xffffff, 0, 3);
      this.h4 = H(c4, 0, 4); this.h6 = H(c4, c8, 6);
      if (this.mlen > 0 && buf[this.mptr] === c) { this.mlen++; this.mptr++; if (this.mlen > 65535) this.mlen = 65535; }
      else this.mlen = 0;
      if (this.pos >= 6) {
        const hh = H(c4, c8, 9) >>> (32 - this.mtb);
        if (this.mlen === 0) {
          const cand = this.mtab[hh];
          if (cand > 0) {
            let l = 0;
            while (l < 32 && cand - 1 - l >= 0 && buf[cand - 1 - l] === buf[this.pos - 1 - l]) l++;
            if (l >= 4) { this.mlen = l; this.mptr = cand; }
          }
        }
        this.mtab[hh] = this.pos;
      }
    }
  }

  async function cmCompress(data, progress) {
    const len = data.length;
    const work = new Uint8Array(len);
    const m = new ByteModel(len, work);
    const e = new Encoder((len >> 1) + 4096);
    for (let s = 0; s < len; s += CHUNK) {
      const end = Math.min(len, s + CHUNK);
      for (let i = s; i < end; i++) {
        const c = data[i];
        for (let b = 7; b >= 0; b--) { const y = (c >> b) & 1; e.enc(y, m.p()); m.update(y); }
      }
      if (progress) progress(end / len);
      if (end < len) await tick();
    }
    return e.finish();
  }
  async function cmDecompress(comp, len, progress) {
    const out = new Uint8Array(len);
    const m = new ByteModel(len, out);
    const d = new Decoder(comp, 0);
    for (let s = 0; s < len; s += CHUNK) {
      const end = Math.min(len, s + CHUNK);
      while (m.pos < end) { for (let b = 0; b < 8; b++) m.update(d.dec(m.p())); }
      if (progress) progress(end / len);
      if (end < len) await tick();
    }
    return out;
  }

  /* ------------------------------------------------------------ small helpers */
  function eq(a, b) {
    if (a.length !== b.length) return false;
    for (let i = 0; i < a.length; i++) if (a[i] !== b[i]) return false;
    return true;
  }
  function concat(parts) {
    let n = 0; for (const p of parts) n += p.length;
    const o = new Uint8Array(n); let k = 0;
    for (const p of parts) { o.set(p, k); k += p.length; }
    return o;
  }
  async function streamBytes(stream, data) {
    const w = stream.writable.getWriter(); w.write(data); w.close();
    const parts = []; const r = stream.readable.getReader();
    for (;;) { const { done, value } = await r.read(); if (done) break; parts.push(value); }
    return concat(parts);
  }
  const hasCS = () => typeof CompressionStream !== 'undefined';

  /* ------------------------------------------------------ method registry */
  // Each method: { enc(data, ctx) -> {blob, meta} | null, dec(blob, entry, ctx) -> Uint8Array }
  const methods = {
    store: {
      enc: async (d) => ({ blob: d, meta: null }),
      dec: async (blob) => blob,
    },
    cm: {
      enc: async (d, ctx) => ({ blob: await cmCompress(d, ctx.sub('Compressing')), meta: null }),
      dec: async (blob, e, ctx) => cmDecompress(blob, e.z, ctx.sub('Restoring')),
    },
  };
  // Other modules (audio) register themselves here.
  function register(name, impl) { methods[name] = impl; }

  const sniffers = []; // [(data, name, opts) => [methodNames to try first]]
  function addSniffer(fn) { sniffers.push(fn); }

  /* ---------------------------------------------------------- pack / unpack */
  function mkCtx(onProgress, label, k, count) {
    return {
      sub(stage) { return (f) => onProgress && onProgress({ stage, file: label, f, k, count, frac: (k + f) / count }); },
    };
  }

  async function encodeEntry(name, data, opts, onProgress, k, count) {
    const verify = opts.verify !== false;
    const ctx = mkCtx(onProgress, name, k, count);
    const h = await sha256(data);
    let best = { method: 'store', blob: data, meta: null };
    const tryMethod = async (mname) => {
      const impl = methods[mname];
      if (!impl) return;
      let r;
      try { r = await impl.enc(data, Object.assign({ opts }, ctx)); } catch (e) { if (e && e.message === 'Cancelled') throw e; return; }
      if (!r || r.blob.length >= best.blob.length) return;
      if (verify) {
        const entry = { z: data.length, x: r.meta, m: mname };
        let ok = false;
        try {
          const back = await impl.dec(r.blob, entry, Object.assign({ opts }, mkCtx(onProgress, name, k, count)));
          ok = r.contentOnly ? ((await impl.contentHash(back)) === r.contentHash) : (back.length === data.length && (await sha256(back)) === h);
        } catch (e) { if (e && e.message === 'Cancelled') throw e; ok = false; }
        if (!ok) return; // never accept an entry that does not restore correctly
      }
      best = { method: mname, blob: r.blob, meta: r.meta, contentHash: r.contentHash };
    };
    let tried = false;
    for (const s of sniffers) {
      const list = s(data, name, opts);
      if (list && list.length) { for (const m of list) await tryMethod(m); tried = true; break; }
    }
    if (!tried || best.method === 'store') {
      // general-purpose path, with a cheap probe so incompressible media is not ground for minutes
      if (data.length > 0) {
        const probe = data.subarray(0, Math.min(data.length, 1 << 18));
        const pc = await cmCompress(probe);
        if (pc.length < probe.length * 0.985 || data.length <= (1 << 18)) await tryMethod('cm');
      }
    }
    return { name, method: best.method, size: data.length, hash: h, blob: best.blob, meta: best.meta, chash: best.contentHash };
  }

  async function pack(files, opts, onProgress) {
    opts = opts || {};
    const index = [], blobs = [];
    let off = 0, totalIn = 0;
    for (let k = 0; k < files.length; k++) {
      const f = files[k];
      onProgress && onProgress({ stage: 'Reading', file: f.name, f: 0, k, count: files.length, frac: k / files.length });
      const data = f.data || await f.get();
      const r = await encodeEntry(f.name, data, opts, onProgress, k, files.length);
      index.push({ n: r.name, m: r.method, z: r.size, h: r.hash, o: off, l: r.blob.length, x: r.meta || undefined, c: r.chash || undefined });
      blobs.push(r.blob); off += r.blob.length; totalIn += r.size;
      onProgress && onProgress({ stage: 'done', file: f.name, f: 1, k, count: files.length, frac: (k + 1) / files.length, entry: index[index.length - 1] });
      await tick();
    }
    let idx = te.encode(JSON.stringify({ v: 4, files: index })), flag = 0;
    if (hasCS()) {
      try {
        const z = await streamBytes(new CompressionStream('deflate-raw'), idx);
        if (z.length < idx.length) { idx = z; flag = 1; }
      } catch (_) { /* keep plain */ }
    }
    const head = new Uint8Array(13);
    head.set(MAGIC, 0); head[8] = flag;
    new DataView(head.buffer).setUint32(9, idx.length, true);
    const parts = [head, idx, ...blobs];
    let size = 0; for (const p of parts) size += p.length;
    return { parts, size, totalIn, index };
  }

  async function open(u8) {
    for (let i = 0; i < 4; i++) if (u8[i] !== MAGIC[i]) throw new Error('This is not a .trky file.');
    const ver = u8[4];
    if (ver === 4) {
      const flag = u8[8], len = new DataView(u8.buffer, u8.byteOffset).getUint32(9, true);
      let idx = u8.subarray(13, 13 + len);
      if (flag === 1) idx = await streamBytes(new DecompressionStream('deflate-raw'), idx);
      const j = JSON.parse(td.decode(idx));
      return { version: 4, files: j.files, base: 13 + len, u8 };
    }
    if (ver === 3) return openV3(u8);
    throw new Error('Unsupported .trky version ' + ver + '.');
  }

  // Older files made by the first desktop prototype (Brotli index). Readable only where the
  // browser can decompress Brotli, and only for methods that keep a plain blob.
  async function openV3(u8) {
    let ds;
    try { ds = new DecompressionStream('brotli'); } catch (_) {
      throw new Error('This .trky was made by the old desktop prototype (v3) and this browser cannot read its Brotli index. Re-create it with this page.');
    }
    const dv = new DataView(u8.buffer, u8.byteOffset);
    const n = Number(dv.getBigUint64(8, true));
    const idx = await streamBytes(ds, u8.subarray(16, 16 + n));
    const j = JSON.parse(td.decode(idx));
    const files = j.files.map((e) => ({ n: e.name, m: 'v3-' + e.method, z: e.size, h: e.sha, o: e.off, l: e.len, x: e.meta, c: e.chash, v3ext: e.method === 'ffv1' }));
    return { version: 3, files, base: 16 + n, u8 };
  }
  methods['v3-store'] = { dec: async (b) => b };
  methods['v3-flac-direct'] = { dec: async (b) => b };
  methods['v3-ffv1'] = { dec: async (b) => b };
  methods['v3-brotli'] = { dec: async (b) => streamBytes(new DecompressionStream('brotli'), b) };

  async function extract(arc, entry, onProgress, verify) {
    const impl = methods[entry.m];
    if (!impl || !impl.dec) throw new Error('Cannot restore "' + entry.n + '": method "' + entry.m + '" is not supported by this page.');
    const blob = arc.u8.subarray(arc.base + entry.o, arc.base + entry.o + entry.l);
    const ctx = mkCtx(onProgress, entry.n, 0, 1);
    const data = await impl.dec(blob, entry, ctx);
    let status = 'ok';
    if (verify !== false) {
      const hh = await sha256(data);
      if (hh === entry.h) status = 'identical';
      else if (entry.c && impl.contentHash && (await impl.contentHash(data)) === entry.c) status = 'content-identical';
      else status = 'MISMATCH';
    }
    return { data, status };
  }

  /* ----------------------------------------------------------- zip (stored) */
  const CRCT = new Uint32Array(256);
  for (let n = 0; n < 256; n++) { let c = n; for (let k = 0; k < 8; k++) c = c & 1 ? 0xedb88320 ^ (c >>> 1) : c >>> 1; CRCT[n] = c >>> 0; }
  function crc32(d) { let c = 0xffffffff; for (let i = 0; i < d.length; i++) c = CRCT[(c ^ d[i]) & 255] ^ (c >>> 8); return (c ^ 0xffffffff) >>> 0; }
  function zipStore(files) { // files: [{name, data}] -> Uint8Array parts (no compression, zip64 not supported: < 4 GB total)
    const parts = [], central = []; let off = 0;
    for (const f of files) {
      const nm = te.encode(f.name), crc = crc32(f.data);
      const lh = new Uint8Array(30 + nm.length), v = new DataView(lh.buffer);
      v.setUint32(0, 0x04034b50, true); v.setUint16(4, 20, true); v.setUint16(6, 0x0800, true);
      v.setUint32(14, crc, true); v.setUint32(18, f.data.length, true); v.setUint32(22, f.data.length, true);
      v.setUint16(26, nm.length, true); lh.set(nm, 30);
      parts.push(lh, f.data);
      const ch = new Uint8Array(46 + nm.length), c = new DataView(ch.buffer);
      c.setUint32(0, 0x02014b50, true); c.setUint16(4, 20, true); c.setUint16(6, 20, true); c.setUint16(8, 0x0800, true);
      c.setUint32(16, crc, true); c.setUint32(20, f.data.length, true); c.setUint32(24, f.data.length, true);
      c.setUint16(28, nm.length, true); c.setUint32(42, off, true); ch.set(nm, 46);
      central.push(ch); off += lh.length + f.data.length;
    }
    let csz = 0; for (const c of central) csz += c.length;
    const end = new Uint8Array(22), e = new DataView(end.buffer);
    e.setUint32(0, 0x06054b50, true); e.setUint16(8, files.length, true); e.setUint16(10, files.length, true);
    e.setUint32(12, csz, true); e.setUint32(16, off, true);
    return [...parts, ...central, end];
  }

  return {
    pack, open, extract, zipStore, sha256, sha256js, register, addSniffer, methods, state,
    util: { Encoder, Decoder, squash, STR, DT, INIT, upd, tick, eq, concat, streamBytes, te, td, CHUNK },
    cmCompress, cmDecompress,
  };
});

