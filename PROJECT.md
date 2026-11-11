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
    vendor/        three.js, web-ifc, occt-import-js, TRKY (see THIRD_PARTY.md)
  export.py   DSP settings as CSV / TXT / JSON
  report.py   self-contained printable HTML report
  __main__.py launcher (app window via Edge/Chrome --app, or --browser) and the `optimize` CLI
```

## Conventions

- A project bundle (.trky) holds `apo-manifest.json` (`kind: "apo-project"`, schema version, the scan's
  path), `project.apo.json` and the scan's original bytes under `scan/<id>.<ext>`. Opening one validates the
  project with `POST /api/projects/check` and stores the scan again (its id is the content hash).
- Mesh repair never moves or deletes vertices; it only rewrites the triangle list, and it is deterministic,
  so a saved venue re-applies it on load (`ScanRef.repair`).

- Coordinates: x across the room, y away from the stage wall, z up. On the plan the stage is on the left.
- Yaw 0 faces the audience; pitch is negative downward.
- Mains are the time reference. Mirror-image L/R pairs are linked. Optimizer gains are capped at 0 dB.
  Locked values are never changed. Polarity is only searched for subs.
- Colour: one validated categorical palette per theme; each speaker keeps its colour slot.

## Not carried over from v1

Live measurement with an audio interface (sweeps, impulse response, RT60), the live FFT monitor, direct
control of the dbx VENU360 / Dante devices and the weather lookup. (LiDAR scans came back: see js/scan.js.) v2 is prediction and
settings export only. These could come back as separate modules on top of the engine.
