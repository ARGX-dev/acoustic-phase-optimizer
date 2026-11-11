/* TRKY FLAC module: FLAC decoder + encoder written from the spec (RFC 9639), no dependencies.
 * Pack side: decode FLAC -> PCM -> audio coder.  Restore side: PCM -> new FLAC (valid, any decoder plays it).
 * Samples and all tag / cover-art / cue blocks come back identical; frame bytes differ from the original encoder's. */
(function (T) {
  'use strict';
  const { concat, tick } = T.util;
  const POW2 = []; for (let i = 0; i <= 40; i++) POW2.push(Math.pow(2, i));

  /* ------------------------------------------------------------- decoder */
  class BR {
    constructor(b, byteOff) { this.b = b; this.p = byteOff * 8; }
    bits(n) { // n <= 32, returns non-negative number
      let v = 0;
      while (n > 0) {
        const byte = this.b[this.p >> 3]; if (byte === undefined) throw new Error('eof');
        const avail = 8 - (this.p & 7), take = n < avail ? n : avail;
        v = v * POW2[take] + ((byte >> (avail - take)) & ((1 << take) - 1));
        this.p += take; n -= take;
      }
      return v;
    }
    signed(n) { const v = this.bits(n); return v >= POW2[n - 1] ? v - POW2[n] : v; }
    unary() { let q = 0; while (this.bits(1) === 0) q++; return q; }
    align() { this.p = (this.p + 7) & ~7; }
    get byte() { return this.p >> 3; }
  }
  const BS_TABLE = [0, 192, 576, 1152, 2304, 4608, 0, 0, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768];
  const BPS_TABLE = [0, 8, 12, 0, 16, 20, 24, 32];

  function readResidual(br, N, order, out, off) {
    const method = br.bits(2); if (method > 1) throw new Error('bad residual method');
    const po = br.bits(4), parts = 1 << po, pbits = method ? 5 : 4, esc = method ? 31 : 15;
    let i = off;
    for (let p = 0; p < parts; p++) {
      const n = p === 0 ? (N >> po) - order : (N >> po);
      const k = br.bits(pbits);
      if (k === esc) {
        const rb = br.bits(5);
        for (let j = 0; j < n; j++) out[i++] = rb ? br.signed(rb) : 0;
      } else {
        for (let j = 0; j < n; j++) {
          const q = br.unary(), low = k ? br.bits(k) : 0, u = q * POW2[k] + low;
          out[i++] = (u % 2 === 0) ? u / 2 : -(u + 1) / 2;
        }
      }
    }
  }
  function subframe(br, N, bps) {
    if (br.bits(1) !== 0) throw new Error('bad subframe');
    const type = br.bits(6);
    let wasted = 0;
    if (br.bits(1)) wasted = br.unary() + 1;
    const b = bps - wasted, x = new Int32Array(N);
    if (type === 0) { const v = br.signed(b); x.fill(v); }
    else if (type === 1) { for (let i = 0; i < N; i++) x[i] = br.signed(b); }
    else if (type >= 8 && type <= 12) {
      const order = type - 8;
      for (let i = 0; i < order; i++) x[i] = br.signed(b);
      readResidual(br, N, order, x, order);
      for (let i = order; i < N; i++) {
        const r = x[i];
        switch (order) {
          case 0: break;
          case 1: x[i] = r + x[i - 1]; break;
          case 2: x[i] = r + 2 * x[i - 1] - x[i - 2]; break;
          case 3: x[i] = r + 3 * x[i - 1] - 3 * x[i - 2] + x[i - 3]; break;
          default: x[i] = r + 4 * x[i - 1] - 6 * x[i - 2] + 4 * x[i - 3] - x[i - 4];
        }
      }
    } else if (type >= 32) {
      const order = (type & 31) + 1;
      for (let i = 0; i < order; i++) x[i] = br.signed(b);
      const prec = br.bits(4) + 1, shift = br.signed(5);
      if (shift < 0) throw new Error('negative lpc shift');
      const co = new Array(order); for (let j = 0; j < order; j++) co[j] = br.signed(prec);
      readResidual(br, N, order, x, order);
      const div = POW2[shift];
      for (let i = order; i < N; i++) {
        let s = 0; for (let j = 0; j < order; j++) s += co[j] * x[i - 1 - j];
        x[i] = x[i] + Math.floor(s / div);
      }
    } else throw new Error('bad subframe type');
    if (wasted) { const m = POW2[wasted]; for (let i = 0; i < N; i++) x[i] = x[i] * m; }
    return x;
  }

  function parseMeta(d) {
    if (d.length < 42 || d[0] !== 0x66 || d[1] !== 0x4c || d[2] !== 0x61 || d[3] !== 0x43) return null;
    const blocks = []; let o = 4, last = false;
    while (!last) {
      if (o + 4 > d.length) return null;
      last = !!(d[o] & 0x80); const type = d[o] & 0x7f, len = (d[o + 1] << 16) | (d[o + 2] << 8) | d[o + 3];
      if (o + 4 + len > d.length) return null;
      blocks.push({ type, data: d.subarray(o + 4, o + 4 + len) }); o += 4 + len;
    }
    if (!blocks.length || blocks[0].type !== 0 || blocks[0].data.length !== 34) return null;
    const si = blocks[0].data;
    const rate = (si[10] << 12) | (si[11] << 4) | (si[12] >> 4);
    const nch = ((si[12] >> 1) & 7) + 1, bps = (((si[12] & 1) << 4) | (si[13] >> 4)) + 1;
    const total = (si[13] & 15) * 4294967296 + (((si[14] << 24) | (si[15] << 16) | (si[16] << 8) | si[17]) >>> 0);
    return { blocks, si, rate, nch, bps, total, frames: o };
  }

  function decodeFlac(d) {
    const m = parseMeta(d); if (!m) return null;
    const { nch, bps } = m;
    if (bps > 24 || bps < 4 || nch > 8) return null;
    let cap = m.total > 0 ? m.total : 1 << 20;
    let chans = []; for (let c = 0; c < nch; c++) chans.push(new Int32Array(cap));
    let n = 0, pos = m.frames;
    while (pos + 6 < d.length) {
      if (d[pos] !== 0xff || (d[pos + 1] & 0xfe) !== 0xf8) break;
      const br = new BR(d, pos);
      br.bits(16);
      const bsc = br.bits(4), src = br.bits(4), ca = br.bits(4), szc = br.bits(3); br.bits(1);
      const u0 = d[br.byte]; let ulen = 1;
      if (u0 >= 0xc0) { ulen = 2; if (u0 >= 0xe0) ulen = 3; if (u0 >= 0xf0) ulen = 4; if (u0 >= 0xf8) ulen = 5; if (u0 >= 0xfc) ulen = 6; if (u0 >= 0xfe) ulen = 7; }
      br.p += ulen * 8;
      let N = BS_TABLE[bsc];
      if (bsc === 6) N = br.bits(8) + 1; else if (bsc === 7) N = br.bits(16) + 1;
      if (src === 12) br.bits(8); else if (src === 13 || src === 14) br.bits(16);
      br.bits(8); // crc8
      let fb = szc === 0 ? bps : BPS_TABLE[szc];
      if (!N || !fb) throw new Error('bad frame');
      if (fb !== bps) throw new Error('mixed sample size');
      const nsub = ca < 8 ? ca + 1 : 2;
      const sub = [];
      for (let c = 0; c < nsub; c++) {
        let b = fb; if (ca === 8 && c === 1) b++; else if (ca === 9 && c === 0) b++; else if (ca === 10 && c === 1) b++;
        sub.push(subframe(br, N, b));
      }
      br.align(); br.bits(16);
      if (nch !== nsub) throw new Error('channel mismatch');
      if (n + N > cap) { cap = Math.max(cap * 2, n + N); chans = chans.map((a) => { const b = new Int32Array(cap); b.set(a); return b; }); }
      if (ca < 8) { for (let c = 0; c < nch; c++) chans[c].set(sub[c], n); }
      else if (ca === 8) { for (let i = 0; i < N; i++) { chans[0][n + i] = sub[0][i]; chans[1][n + i] = sub[0][i] - sub[1][i]; } }
      else if (ca === 9) { for (let i = 0; i < N; i++) { chans[1][n + i] = sub[1][i]; chans[0][n + i] = sub[1][i] + sub[0][i]; } }
      else if (ca === 10) {
        for (let i = 0; i < N; i++) {
          const s = sub[1][i], mid = sub[0][i] * 2 + (s & 1);
          chans[0][n + i] = (mid + s) >> 1; chans[1][n + i] = (mid - s) >> 1;
        }
      } else throw new Error('bad channel assignment');
      n += N; pos = br.byte;
    }
    if (n === 0) return null;
    if (m.total > 0 && n !== m.total) return null;
    return { meta: m, chans: chans.map((a) => (a.length === n ? a : a.subarray(0, n))), F: n };
  }

  /* ------------------------------------------------------------- encoder */
  const CRC8 = new Uint8Array(256), CRC16 = new Uint16Array(256);
  for (let i = 0; i < 256; i++) {
    let c = i; for (let k = 0; k < 8; k++) c = c & 0x80 ? ((c << 1) ^ 0x07) & 255 : (c << 1) & 255; CRC8[i] = c;
    let d = i << 8; for (let k = 0; k < 8; k++) d = d & 0x8000 ? ((d << 1) ^ 0x8005) & 0xffff : (d << 1) & 0xffff; CRC16[i] = d;
  }
  class BW {
    constructor(cap) { this.b = new Uint8Array(cap || 1 << 16); this.n = 0; this.acc = 0; this.cnt = 0; }
    byte(v) { if (this.n >= this.b.length) { const o = new Uint8Array(this.b.length * 2); o.set(this.b); this.b = o; } this.b[this.n++] = v; }
    write(v, n) { // v in [0, 2^n), n <= 32
      while (n > 0) {
        const take = Math.min(8 - this.cnt, n), sh = n - take;
        const chunk = Math.floor(v / POW2[sh]); v -= chunk * POW2[sh];
        this.acc = (this.acc << take) | chunk; this.cnt += take; n -= take;
        if (this.cnt === 8) { this.byte(this.acc & 255); this.acc = 0; this.cnt = 0; }
      }
    }
    signed(v, n) { this.write(v < 0 ? v + POW2[n] : v, n); }
    unary(q) { while (q >= 24) { this.write(0, 24); q -= 24; } this.write(1, q + 1); }
    align() { if (this.cnt) this.write(0, 8 - this.cnt); }
  }
  const zz = (r) => (r >= 0 ? r * 2 : -r * 2 - 1);
  function riceChoose(res, N, order) {
    // returns { po, ks[], bits, method } using exact per-partition k search
    const cnt = res.length;
    const u = new Float64Array(cnt); for (let i = 0; i < cnt; i++) u[i] = zz(res[i]);
    const pre = new Float64Array(cnt + 1); for (let i = 0; i < cnt; i++) pre[i + 1] = pre[i] + u[i];
    let best = null, maxPo = 0;
    while (maxPo < 8 && N % (1 << (maxPo + 1)) === 0 && (N >> (maxPo + 1)) > order) maxPo++;
    for (let po = 0; po <= maxPo; po++) {
      const parts = 1 << po; let bits = 6, st = 0; const ks = [];
      for (let p = 0; p < parts; p++) {
        const n = p === 0 ? (N >> po) - order : N >> po, S = pre[st + n] - pre[st];
        let k = 0; if (n > 0 && S > 0) { k = Math.max(0, Math.floor(Math.log2(S / n))); if (k > 30) k = 30; }
        let bk = Infinity, kk = k;
        for (let t = Math.max(0, k - 1); t <= Math.min(30, k + 1); t++) {
          const b = n * (t + 1) + Math.floor(S / POW2[t]);
          if (b < bk) { bk = b; kk = t; }
        }
        ks.push(kk); bits += 4 + bk; st += n;
      }
      if (!best || bits < best.bits) best = { po, ks, bits };
    }
    best.method = Math.max(...best.ks) > 14 ? 1 : 0;
    if (best.method) best.bits += (1 << best.po);
    best.u = u;
    return best;
  }
  function writeResidual(w, best, N, order) {
    w.write(best.method, 2); w.write(best.po, 4);
    const parts = 1 << best.po, pbits = best.method ? 5 : 4; let i = 0;
    for (let p = 0; p < parts; p++) {
      const n = p === 0 ? (N >> best.po) - order : N >> best.po, k = best.ks[p], div = POW2[k];
      w.write(k, pbits);
      for (let j = 0; j < n; j++, i++) {
        const u = best.u[i], q = Math.floor(u / div);
        w.unary(q); if (k) w.write(u - q * div, k);
      }
    }
  }
  function fixedRes(x, N, order) {
    const r = new Int32Array(N - order);
    for (let i = order; i < N; i++) {
      let p;
      switch (order) {
        case 0: p = 0; break;
        case 1: p = x[i - 1]; break;
        case 2: p = 2 * x[i - 1] - x[i - 2]; break;
        case 3: p = 3 * x[i - 1] - 3 * x[i - 2] + x[i - 3]; break;
        default: p = 4 * x[i - 1] - 6 * x[i - 2] + 4 * x[i - 3] - x[i - 4];
      }
      r[i - order] = x[i] - p;
    }
    return r;
  }
  function lpcAnalyse(x, N, maxOrder, bps) {
    const win = new Float64Array(N), a = 0.5, edge = Math.floor(a * N / 2);
    for (let i = 0; i < N; i++) {
      let w = 1; if (i < edge) w = 0.5 - 0.5 * Math.cos(Math.PI * i / edge); else if (i >= N - edge) w = 0.5 - 0.5 * Math.cos(Math.PI * (N - 1 - i) / edge);
      win[i] = x[i] * w;
    }
    const ac = new Float64Array(maxOrder + 1);
    for (let l = 0; l <= maxOrder; l++) { let s = 0; for (let i = l; i < N; i++) s += win[i] * win[i - l]; ac[l] = s; }
    if (ac[0] === 0) return null;
    ac[0] *= 1.0000001;
    let err = ac[0], lpc = new Float64Array(maxOrder + 1), tmp = new Float64Array(maxOrder + 1);
    let bestOrder = 0, bestEst = N * 0.5 * Math.log2(ac[0] / N + 1), bestCoefs = null;
    for (let m = 1; m <= maxOrder; m++) {
      let acc = ac[m]; for (let j = 1; j < m; j++) acc -= lpc[j] * ac[m - j];
      const k = acc / err; tmp.set(lpc);
      lpc[m] = k; for (let j = 1; j < m; j++) lpc[j] = tmp[j] - k * tmp[m - j];
      err *= (1 - k * k); if (err <= 0) break;
      const est = N * 0.5 * Math.log2(err / N + 1e-9) + m * (bps > 16 ? 15 : 13) * 0.5;
      if (est < bestEst) { bestEst = est; bestOrder = m; bestCoefs = Array.from(lpc.subarray(1, m + 1)); }
    }
    return bestOrder ? { order: bestOrder, coefs: bestCoefs } : null;
  }
  function lpcQuant(coefs, bps) {
    const prec = bps > 16 ? 14 : 12; let cmax = 0;
    for (const c of coefs) cmax = Math.max(cmax, Math.abs(c));
    if (cmax === 0) return null;
    let shift = prec - 1 - Math.ceil(Math.log2(cmax)); if (shift > 15) shift = 15; if (shift < 0) return null;
    const lim = POW2[prec - 1] - 1, q = []; let carry = 0;
    for (const c of coefs) {
      const v = c * POW2[shift] + carry; let r = Math.round(v); if (r > lim) r = lim; if (r < -lim) r = -lim;
      carry = v - r; q.push(r);
    }
    return { q, shift, prec };
  }
  function lpcRes(x, N, q, shift) {
    const order = q.length, r = new Int32Array(N - order), div = POW2[shift];
    for (let i = order; i < N; i++) {
      let s = 0; for (let j = 0; j < order; j++) s += q[j] * x[i - 1 - j];
      r[i - order] = x[i] - Math.floor(s / div);
    }
    return r;
  }
  function encodeSubframe(w, x, N, bps) {
    // constant?
    let same = true; for (let i = 1; i < N; i++) if (x[i] !== x[0]) { same = false; break; }
    if (same) { w.write(0, 8); w.signed(x[0], bps); return; }
    let best = { bits: 8 + N * bps, kind: 'verbatim' };
    for (let o = 0; o <= 4 && o < N; o++) {
      const res = fixedRes(x, N, o), rc = riceChoose(res, N, o), bits = 8 + o * bps + rc.bits;
      if (bits < best.bits) best = { bits, kind: 'fixed', order: o, rc };
    }
    if (N >= 32) {
      const an = lpcAnalyse(x, N, bps > 16 ? 16 : 12, bps);
      if (an) {
        const qz = lpcQuant(an.coefs, bps);
        if (qz) {
          const res = lpcRes(x, N, qz.q, qz.shift), rc = riceChoose(res, N, qz.q.length);
          const bits = 8 + qz.q.length * bps + 9 + qz.q.length * qz.prec + rc.bits;
          if (bits < best.bits) best = { bits, kind: 'lpc', order: qz.q.length, qz, rc };
        }
      }
    }
    if (best.kind === 'verbatim') { w.write(2, 8); for (let i = 0; i < N; i++) w.signed(x[i], bps); }
    else if (best.kind === 'fixed') {
      w.write((8 + best.order) << 1, 8);
      for (let i = 0; i < best.order; i++) w.signed(x[i], bps);
      writeResidual(w, best.rc, N, best.order);
    } else {
      w.write((32 + best.order - 1) << 1, 8);
      for (let i = 0; i < best.order; i++) w.signed(x[i], bps);
      w.write(best.qz.prec - 1, 4); w.signed(best.qz.shift, 5);
      for (const c of best.qz.q) w.signed(c, best.qz.prec);
      writeResidual(w, best.rc, N, best.order);
    }
  }
  function utf8Num(v) { // UTF-8 style variable-length number, up to 36 bits
    if (v < 0x80) return [v];
    let len = 2;
    for (const l of [0x800, 0x10000, 0x200000, 0x4000000, 0x80000000]) { if (v >= l) len++; else break; }
    const first = [0, 0, 0xc0, 0xe0, 0xf0, 0xf8, 0xfc, 0xfe][len], out = [];
    for (let i = len - 1; i > 0; i--) { out.unshift(0x80 | (v % 64)); v = Math.floor(v / 64); }
    out.unshift(first | v);
    return out;
  }
  const SZC = { 8: 1, 12: 2, 16: 4, 20: 5, 24: 6, 32: 7 };
  async function encodeFlac(chans, F, bps, rate, mdBlocks, si, progress) {
    const nch = chans.length, BS = rate > 48000 ? 8192 : 4096;
    const bsc = BS === 4096 ? 12 : 13;
    // STREAMINFO: rewrite block sizes, unknown frame sizes, keep rate/ch/bps/total/md5
    const sinfo = new Uint8Array(si);
    sinfo[0] = sinfo[2] = BS >> 8; sinfo[1] = sinfo[3] = BS & 255; for (let i = 4; i < 10; i++) sinfo[i] = 0;
    if (F < BS) { sinfo[0] = sinfo[2] = F >> 8; sinfo[1] = sinfo[3] = F & 255; }
    const tot = (sinfo[13] & 15) * 4294967296 + (((sinfo[14] << 24) | (sinfo[15] << 16) | (sinfo[16] << 8) | sinfo[17]) >>> 0);
    if (tot === 0) { sinfo[13] = (sinfo[13] & 0xf0) | (Math.floor(F / 4294967296) & 15); const lo = F >>> 0; sinfo[14] = lo >>> 24; sinfo[15] = (lo >>> 16) & 255; sinfo[16] = (lo >>> 8) & 255; sinfo[17] = lo & 255; }
    const head = new BW(1 << 12);
    for (const ch of [0x66, 0x4c, 0x61, 0x43]) head.byte(ch);
    const all = [{ type: 0, data: sinfo }, ...mdBlocks];
    all.forEach((b, i) => {
      head.byte(b.type | (i === all.length - 1 ? 0x80 : 0));
      head.byte((b.data.length >> 16) & 255); head.byte((b.data.length >> 8) & 255); head.byte(b.data.length & 255);
      for (let k = 0; k < b.data.length; k++) head.byte(b.data[k]);
    });
    const out = new BW(F * nch + (1 << 16));
    for (let k = 0; k < head.n; k++) out.byte(head.b[k]);
    let fno = 0;
    for (let s = 0; s < F; s += BS, fno++) {
      const N = Math.min(BS, F - s);
      const sig = chans.map((a) => a.subarray(s, s + N));
      let ca = nch - 1, parts = sig, bpsList = sig.map(() => bps);
      if (nch === 2) {
        const L = sig[0], R = sig[1], M = new Int32Array(N), S = new Int32Array(N);
        for (let i = 0; i < N; i++) { S[i] = L[i] - R[i]; M[i] = (L[i] + R[i]) >> 1; }
        const cost = (x) => { let t = 0; for (let i = 2; i < N; i++) t += Math.abs(x[i] - 2 * x[i - 1] + x[i - 2]); return t; };
        const cL = cost(L), cR = cost(R), cM = cost(M), cS = cost(S);
        const opts = [[cL + cR, 1], [cL + cS, 8], [cS + cR, 9], [cM + cS, 10]];
        opts.sort((p, q) => p[0] - q[0]); const pick = opts[0][1];
        if (pick === 1) { ca = 1; }
        else if (pick === 8) { ca = 8; parts = [L, S]; bpsList = [bps, bps + 1]; }
        else if (pick === 9) { ca = 9; parts = [S, R]; bpsList = [bps + 1, bps]; }
        else { ca = 10; parts = [M, S]; bpsList = [bps, bps + 1]; }
      }
      const w = new BW(1 << 16), hb = [0xff, 0xf8];
      let nsz = bsc;
      let hdr = hb.concat([(nsz << 4) | 0, (ca << 4) | ((SZC[bps] || 0) << 1)], utf8Num(fno));
      if (N !== BS) { // short last frame
        const code = N <= 256 ? 6 : 7; hdr = hb.concat([(code << 4) | 0, (ca << 4) | ((SZC[bps] || 0) << 1)], utf8Num(fno));
        if (code === 6) hdr.push(N - 1); else hdr.push((N - 1) >> 8, (N - 1) & 255);
      }
      let c8 = 0; for (const v of hdr) c8 = CRC8[c8 ^ v]; hdr.push(c8);
      for (const v of hdr) w.byte(v);
      for (let c = 0; c < parts.length; c++) encodeSubframe(w, parts[c], N, bpsList[c]);
      w.align();
      let c16 = 0; for (let i = 0; i < w.n; i++) c16 = ((c16 << 8) ^ CRC16[(c16 >> 8) ^ w.b[i]]) & 0xffff;
      w.byte(c16 >> 8); w.byte(c16 & 255);
      for (let i = 0; i < w.n; i++) out.byte(w.b[i]);
      if ((fno & 15) === 0) { progress && progress(s / F); await tick(); }
    }
    return out.b.subarray(0, out.n);
  }

  /* --------------------------------------------------------- the method */
  async function contentHashOf(d) {
    const f = decodeFlac(d); if (!f) return null;
    const { chans, F, meta } = f, nch = chans.length, nb = (meta.bps + 7) >> 3;
    const md = []; for (const b of meta.blocks) if ([2, 4, 5, 6].includes(b.type)) md.push(new Uint8Array([b.type, b.data.length >> 16 & 255, b.data.length >> 8 & 255, b.data.length & 255]), b.data);
    const pcm = new Uint8Array(F * nch * nb + 12);
    new DataView(pcm.buffer).setUint32(0, F, true); pcm[4] = nch; pcm[5] = meta.bps; pcm[6] = meta.rate & 255; pcm[7] = (meta.rate >> 8) & 255; pcm[8] = (meta.rate >> 16) & 255;
    let p = 12; for (let n = 0; n < F; n++) for (let c = 0; c < nch; c++) { const v = chans[c][n]; for (let b = 0; b < nb; b++) pcm[p++] = (v >> (8 * b)) & 255; }
    return T.sha256(concat([...md, pcm]));
  }
  function serMeta(blocks) {
    const parts = [];
    for (const b of blocks) if ([2, 4, 5, 6].includes(b.type)) parts.push(new Uint8Array([b.type, (b.data.length >> 16) & 255, (b.data.length >> 8) & 255, b.data.length & 255]), b.data);
    return concat(parts);
  }
  function deserMeta(u8) {
    const out = []; let o = 0;
    while (o + 4 <= u8.length) { const len = (u8[o + 1] << 16) | (u8[o + 2] << 8) | u8[o + 3]; out.push({ type: u8[o], data: u8.subarray(o + 4, o + 4 + len) }); o += 4 + len; }
    return out;
  }

  T.register('flac', {
    async enc(d, ctx) {
      if (!ctx.opts || ctx.opts.reencode === false) return null;
      const f = decodeFlac(d); if (!f || f.F < 64) return null;
      const A = T.audio, chans = f.chans, F = f.F;
      const ch = chans.map((a) => new Int32Array(a));
      const sh = A.trailingShift(ch, F); A.applyShift(ch, F, sh, -1);
      const stream = await A.encodePCM(ch, F, ctx.sub('Compressing audio'));
      const md = serMeta(f.meta.blocks), len = new Uint8Array(4); new DataView(len.buffer).setUint32(0, md.length, true);
      const blob = concat([len, md, f.meta.si, stream]);
      return { blob, meta: { ch: chans.length, bps: f.meta.bps, rate: f.meta.rate, F, sh }, contentOnly: true, contentHash: await contentHashOf(d) };
    },
    async dec(blob, e, ctx) {
      const m = e.x, mdLen = new DataView(blob.buffer, blob.byteOffset).getUint32(0, true);
      const md = deserMeta(blob.subarray(4, 4 + mdLen)), si = blob.subarray(4 + mdLen, 4 + mdLen + 34);
      const chans = await T.audio.decodePCM(blob, 4 + mdLen + 34, m.ch, m.F, ctx.sub('Restoring audio'));
      T.audio.applyShift(chans, m.F, m.sh, 1);
      return encodeFlac(chans, m.F, m.bps, m.rate, md, si, ctx.sub('Writing FLAC'));
    },
    contentHash: contentHashOf,
  });
  T.addSniffer((d, name, opts) => (d.length > 64 && d[0] === 0x66 && d[1] === 0x4c && d[2] === 0x61 && d[3] === 0x43 && opts.reencode !== false ? ['flac'] : null));
  T.flac = { decodeFlac, encodeFlac, parseMeta, contentHashOf };
})(typeof module === 'object' && module.exports ? require('./core.js') : TRKY);

