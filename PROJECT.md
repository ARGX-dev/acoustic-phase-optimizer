# Acoustic Phase Optimizer: project notes

Version 2 is a rewrite. Version 1 (PyQt6 desktop app) is preserved at the git tag `v1-pyqt-final`.

## What it is

A planning and alignment tool for live sound in reflective venues (gyms, theatres, churches, conference
rooms). You model the room and the speakers. It predicts coverage, clarity, interference and echo across
the audience. Then it sets delay, gain, polarity and EQ, and you copy those settings into the processor.

## Architecture

```
acoustic_phase_optimizer/
  engine/     pure numpy/scipy, no I/O
    model.py      pydantic Project / Room / Speaker / Mic / Settings (the file format)
    air.py        speed of sound, ISO 9613-1 air absorption
    materials.py  18 surface materials, octave-band absorption, Sabine/Eyring, critical distance
    eq.py         RBJ biquads for the PEQ
    field.py      per-speaker transfer functions on the audience grid (image sources, directivity),
                  cached by geometry only, so DSP edits never recompute the field
    metrics.py    coverage, interference, clarity, timing, tonal -> score 0..100
    optimize.py   quick_align, deep_optimize (differential evolution + Powell), auto_eq
    simulate.py   maps, metrics, point response, room report for the UI
    presets.py    speaker types and example venues
    advice.py     plain-language findings
    measure.py    sweeps, deconvolution, impulse response analysis (ISO 3382, Lundeby, Schroeder)
  server/     FastAPI: stateless API (the project travels with each request), background jobs, JSON storage
  web/        vanilla ES modules + canvas, no build step
    js/store.js    state, undo/redo, request pumps, autosave
    js/stage.js    plan view and heat map
    js/chart.js    magnitude / phase / arrivals
    js/panels.js   venue editor (left)
    js/results.js  score, optimize, findings (right)
    js/app.js      top bar, start-up, shortcuts
    js/scan.js     scan and model import, room fitting, the 3D scan in the view
    js/formats.js  glTF/USD/CAD readers and converters (dxf.js, mvr.js, occt-worker.js beside it)
    js/meshfix.js  mesh repair (pure, typed arrays; meshfix-worker.js for big meshes,
                   meshfix-three.js applies it to a three.js model and can switch it off again)
    js/trky.js     .trky archives: open (project bundle or any archive) and save; trky-worker.js
                   runs the vendored coders (web/vendor/trky)
    js/drop.js     drag and drop onto the window
    js/measure-ui.js  measurements: import, level match, the impulse response viewer, delay suggestions
    js/live.js     live sweep capture (getUserMedia + capture-worklet.js, which plays and records in one
                   audio callback); js/rta.js the real-time analyser and pink noise
    vendor/        three.js, web-ifc, occt-import-js, TRKY (see THIRD_PARTY.md)
  export.py   DSP settings as CSV / TXT / JSON, the VENU360 sheet, REW / Equalizer APO filter text and
              miniDSP biquads (zipped, one file per output)
  report.py   self-contained client report (cover, contents, plots; A4 print CSS); js/report-ui.js is
              its dialog. Branding is app-wide (data folder, branding.json, GET/PUT /api/branding); the
              job's client and reference live in the project (Project.report)
  __main__.py launcher (app window via Edge/Chrome --app, or --browser) and the `optimize` CLI
```

## Conventions

- A project bundle (.trky) holds `apo-manifest.json` (`kind: "apo-project"`, schema version, the scan's
  path), `project.apo.json` and the scan's original bytes under `scan/<id>.<ext>`. Opening one validates the
  project with `POST /api/projects/check` and stores the scan again (its id is the content hash).
- Mesh repair never moves or deletes vertices; it only rewrites the triangle list, and it is deterministic,
  so a saved venue re-applies it on load (`ScanRef.repair`).

- Processor exports are written from the same values as the model: LR24 crossovers are two Butterworth
  biquads at Q 0.7071, PEQ bands use the RBJ cookbook (engine/eq.py), and flat bells or shelves are left out.
  miniDSP's a1 and a2 are the negated denominator terms (checked against a published REW export).
- Everything that reaches the report's HTML is escaped; the accent colour and logo are checked by pattern
  (#rrggbb, a PNG or JPEG data URL), and the footer text is written as an escaped CSS string.

- Coordinates: x across the room, y away from the stage wall, z up. On the plan the stage is on the left.
- Yaw 0 faces the audience; pitch is negative downward.
- Mains are the time reference. Mirror-image L/R pairs are linked. Optimizer gains are capped at 0 dB.
  Locked values are never changed. Polarity is only searched for subs.
- Colour: one validated categorical palette per theme; each speaker keeps its colour slot.

## Not carried over from v1

Direct control of the dbx VENU360 / Dante devices and the weather lookup. (LiDAR scans came back, see
js/scan.js, and so did live measurement: sweeps, impulse responses, RT60 and the live analyser, see
engine/measure.py and js/live.js.) Settings still leave the app as exports, not over the network.
