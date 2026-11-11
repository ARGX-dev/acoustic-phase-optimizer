# occt-import-js 0.0.23, as vendored here

- `occt-import-js.wasm`, `license.occt-import-js.txt`, `license.occt.txt`: unmodified copies from the published
  npm package (`dist/`). Source: https://github.com/kovacsv/occt-import-js (LGPL-2.1), built on OpenCascade.
- `occt-import-js.js`: **modified on 2026-11-11**. The published file builds two kinds of call wrapper (embind's
  `craftInvokerFunction` and `__emval_get_method_caller`) by generating JavaScript source and evaluating it with
  `Function()`. A Content-Security-Policy without `unsafe-eval`, which this app uses, blocks that. Both places
  now create a closure that performs the same steps as the generated source. Nothing else is changed.
- `apply_local_patch.py` makes the modified file from the published one, so the change can be checked or redone
  for another version: `python apply_local_patch.py PUBLISHED.js OUT.js`.

You may replace these files with your own build of the library (LGPL-2.1, section 6); the app loads them from
this folder at run time.
