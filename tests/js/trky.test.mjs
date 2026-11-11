// The vendored .trky code (web/vendor/trky, copied unchanged from the converter page). Run: node --test tests/js
import test from 'node:test';
import assert from 'node:assert/strict';
import { createRequire } from 'node:module';

const require = createRequire(import.meta.url);
const T = require('../../acoustic_phase_optimizer/web/vendor/trky/core.js');
require('../../acoustic_phase_optimizer/web/vendor/trky/audio.js');
require('../../acoustic_phase_optimizer/web/vendor/trky/flac.js');

const join = (parts) => {
  const out = new Uint8Array(parts.reduce((n, p) => n + p.length, 0));
  let o = 0;
  for (const p of parts) { out.set(p, o); o += p.length; }
  return out;
};

/** 16-bit stereo WAV with a decaying tone (compresses well, like an impulse response). */
function wav(frames = 4000) {
  const b = new Uint8Array(44 + frames * 4), v = new DataView(b.buffer);
  const s = (o, t) => [...t].forEach((c, i) => { b[o + i] = c.charCodeAt(0); });
  s(0, 'RIFF'); v.setUint32(4, 36 + frames * 4, true); s(8, 'WAVEfmt '); v.setUint32(16, 16, true);
  v.setUint16(20, 1, true); v.setUint16(22, 2, true); v.setUint32(24, 48000, true); v.setUint32(28, 192000, true);
  v.setUint16(32, 4, true); v.setUint16(34, 16, true); s(36, 'data'); v.setUint32(40, frames * 4, true);
  for (let i = 0; i < frames; i++) {
    const x = Math.round(12000 * Math.sin(i * 0.07) * (1 - i / frames));
    v.setInt16(44 + 4 * i, x, true); v.setInt16(46 + 4 * i, x >> 1, true);
  }
  return b;
}

const enc = new TextEncoder();
const files = () => [
  { name: 'project.apo.json', data: enc.encode(JSON.stringify({ version: 2, speakers: Array.from({ length: 40 }, (_, i) => ({ id: i, x: i * 0.5 })) })) },
  { name: 'scan/hall.glb', data: Uint8Array.from({ length: 3000 }, (_, i) => (i * 7919) & 255) },
  { name: 'ir/front.wav', data: wav() },
];

test('pack, open and extract give the same bytes back', async () => {
  const input = files();
  const packed = await T.pack(input, { verify: true });
  const arc = await T.open(join(packed.parts));
  assert.equal(arc.version, 4);
  assert.deepEqual(arc.files.map((e) => e.n), input.map((f) => f.name));
  for (const [k, e] of arc.files.entries()) {
    const { data, status } = await T.extract(arc, e);
    assert.equal(status, 'identical');
    assert.deepEqual(data, input[k].data);
  }
  assert.equal(arc.files.find((e) => e.n === 'ir/front.wav').m, 'pcm');
  assert.ok(packed.size < packed.totalIn);
});

test('a damaged entry is caught', async () => {
  const input = [{ name: 'a.bin', data: Uint8Array.from({ length: 500 }, (_, i) => i & 255) }];
  const packed = await T.pack(input, { verify: true });
  const bytes = join(packed.parts);
  const arc = await T.open(bytes);
  const e = arc.files[0];
  bytes[arc.base + e.o + e.l - 1] ^= 0x55;
  const r = await T.extract(arc, e).catch(() => ({ status: 'MISMATCH' })); // a coder may also refuse outright
  assert.equal(r.status, 'MISMATCH');
});

test('files that are not .trky are refused', async () => {
  await assert.rejects(T.open(enc.encode('PK\x03\x04 not an archive')), /not a \.trky/);
});
