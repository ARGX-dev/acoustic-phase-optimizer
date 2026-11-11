// Pack files into a .trky with the vendored TRKY code, for the UI tests.
// Usage: node tests/fixtures/make_trky.cjs out.trky name=path [name=path ...]
const fs = require('fs');
const path = require('path');
const vendor = path.join(__dirname, '..', '..', 'acoustic_phase_optimizer', 'web', 'vendor', 'trky');
const T = require(path.join(vendor, 'core.js'));
require(path.join(vendor, 'audio.js'));
require(path.join(vendor, 'flac.js'));

(async () => {
  const [out, ...pairs] = process.argv.slice(2);
  const files = pairs.map((p) => {
    const i = p.indexOf('=');
    return { name: p.slice(0, i), data: new Uint8Array(fs.readFileSync(p.slice(i + 1))) };
  });
  const r = await T.pack(files, { verify: true });
  fs.writeFileSync(out, Buffer.concat(r.parts.map((x) => Buffer.from(x.buffer, x.byteOffset, x.length))));
})().catch((e) => { console.error(e); process.exit(1); });
