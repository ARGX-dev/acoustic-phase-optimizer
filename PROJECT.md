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
  export.py   DSP settings as CSV / TXT / JSON
  report.py   self-contained printable HTML report
  __main__.py launcher (app window via Edge/Chrome --app, or --browser) and the `optimize` CLI
```

## Conventions

- Coordinates: x across the room, y away from the stage wall, z up. On the plan the stage is on the left.
- Yaw 0 faces the audience; pitch is negative downward.
- Mains are the time reference. Mirror-image L/R pairs are linked. Optimizer gains are capped at 0 dB.
  Locked values are never changed. Polarity is only searched for subs.
- Colour: one validated categorical palette per theme; each speaker keeps its colour slot.

## Not carried over from v1

Live measurement with an audio interface (sweeps, impulse response, RT60), the live FFT monitor, direct
control of the dbx VENU360 / Dante devices, LIDAR import and the weather lookup. v2 is prediction and
settings export only. These could come back as separate modules on top of the engine.
