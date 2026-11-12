# Acoustic Phase Optimizer 2

Plan, align and optimize a live sound system for a reflective venue, before you get there.

Draw the room, drop in the speakers, and the app predicts what every seat hears: coverage, clarity, where
speakers add up or cancel, and where late arrivals turn into echoes. One click time-aligns the system the
way a system tech would. A deeper search and an automatic EQ take it further. The result is a sheet of
delay, gain, polarity and EQ settings for your processor, plus a printable report.

## Run it

Windows: double-click **`install.bat`** once, then **`Acoustic Phase Optimizer.bat`** whenever you want the app.

Anywhere (Python 3.10 or newer):

```bash
pip install -e .
python -m acoustic_phase_optimizer            # opens in its own window (Edge or Chrome)
python -m acoustic_phase_optimizer --browser  # or in your normal browser
```

Everything runs on your own computer. The window talks to a small local engine on `127.0.0.1`; nothing goes
online. The engine refuses requests from other web sites and other host names, and the page runs under a
strict Content-Security-Policy.

## What you can do

- **Model the venue.** An indoor room, a stadium (open roof) or open air. Room size, a material for each
  surface (or your own, with absorption per octave), how full the seats are, temperature, humidity, and
  pressure or altitude. Six example venues (gymnasium, theatre, house of worship, conference room, festival
  field, stadium) or an empty room. The Room tab shows reverb time per octave, critical distance, Schroeder
  frequency and air loss; for open air it shows free-field facts instead.
- **Import a 3D scan or a venue model.** A scan as GLB, glTF, USDZ, OBJ or PLY from a LiDAR app (Polycam,
  Scaniverse, 3d Scanner App, iPhone RoomPlan) or a laser scanner, or a CAD model as MVR, 3DS, FBX, COLLADA (.dae) or STL, for example
  from Vectorworks (File > Export). MVR files are read in full: scene objects, trusses, screens, nested
  groups and symbols, in millimetres; fixtures are counted but not drawn, since MVR keeps their shapes in
  GDTF files. USDZ is read in binary (crate) and text form, with its own units and up axis; .usd, .usda and
  .usdc layers work too. A .gltf is chosen together with its .bin and texture files (or import the folder it
  is in) and is packed into one GLB. If Windows has unpacked an .mvr into a folder, use **Import a folder
  (unpacked MVR or glTF)…**. The app measures the room from it (floor, ceiling, wall direction), you pick the stage wall
  and confirm the size, and the model is shown in the 3D view. The sound prediction still uses the fitted
  box, not the modelled shape.
  CAD drawings are read too: DXF (the app's own reader), IFC (web-ifc), STEP, IGES and BREP (OpenCascade,
  in WebAssembly), and DWG when ODA File Converter or LibreDWG is installed. Draco- and meshopt-compressed
  GLBs and KTX2 textures work. Not readable: Vectorworks .vwx, Revit, DWF, Cinema 4D and SAT; the app says
  which export to use instead.
- **Broken models are repaired on import.** Faces wound the wrong way, small holes, duplicated and broken
  faces and unjoined seams are fixed before the room is measured. Doorways and other large openings are
  left open. The import dialog lists what was fixed, and **Use the repaired model** switches back to the
  file exactly as it is. The file itself is never changed; the choice is saved with the venue.
- **Speakers found in the scan.** **Room > Find speakers in the scan…** looks for loudspeakers: rigged MVR
  fixtures (lights are left out), objects named like speakers or like a library model (KARA, KS28…), and
  separate speaker-sized boxes on walls, under the ceiling, stacked as a line array or on the floor as subs.
  A plan shows each one with how sure the app is; tick the ones to keep and choose a model for each. Nothing
  is added until you say so, and one undo takes them all back out. Rigs and named speakers are offered right
  after an import. A speaker that is part of one continuous scanned surface cannot be told apart.
- **Share a whole venue as one .trky file.** **Export > Project archive** packs the venue and its scan into
  a lossless .trky archive (the TRKY format, which shrinks audio and text well and checks every file with
  SHA-256 when it is unpacked). **Open > Open a .trky archive** brings it back on any computer. Other .trky
  archives list the models, measurements and speaker lists inside so you can pick one.
- **Drop files anywhere on the window.** A .trky, a scan or model (or a folder holding an unpacked MVR or a
  glTF), a saved project (.apo.json) or a measurement export each go where their own button would take them.
- **Place speakers on the plan, in the section or in 3D.** Drag to move, drag the handle to aim. Mains,
  fills, front fills, delays and subs. Left/right pairs are linked so they stay matched. Arrow keys nudge,
  R turns, Ctrl+D duplicates, M mutes; drag a field's label to scrub its value; press `?` for every shortcut.
- **Choose real loudspeakers.** A searchable library of 47 models from QSC, JBL, Yamaha, Electro-Voice, RCF,
  L-Acoustics, d&b audiotechnik, Meyer Sound and Bose, plus generic types. Coverage, max SPL and
  low-frequency limit come from the manufacturer datasheets (each entry links to its source); figures that
  could not be confirmed are marked *approx.* Pattern-control frequency and the high-pass corner are
  estimated by fixed rules, and the app says so. Line-array hangs take a box count and splay, and each box is modelled.
  There is no open, checked dataset of thousands of PA loudspeakers, so the library grows by hand; add your
  own models with **Import list…** (CSV or JSON; download the template from the Speakers tab) or
  **Save as my model**.
- **Build sub arrays.** Cardioid (gradient pairs), end-fire, or a line with arc delay. The builder shows each
  box's delay and polarity, the optimizer treats the array as one unit, and the exported settings include
  the pattern offsets.
- **Three views.** The plan (scroll or pinch to zoom, right-drag or Space-drag to pan); a section, a vertical
  slice of level through the probe along or across the room, where you drag speakers in height and tilt them
  by their aim handle; and 3D, where you orbit the room and drag speakers and mics across the floor.
- **See the result live.** Maps of direct level (all frequencies or one octave, Z, A or C weighted), total
  level, clarity, speech intelligibility (an STI estimate), summation and cancellation, and arrival-time
  spread. Click anywhere for the magnitude, phase and arrivals at that seat, or add measurement-mic positions.
- **Aim at the audience.** One click points a speaker at the seats it serves: a single box at the back
  third; a line-array hang with its top box on the last row and its bottom box on the first, solving the
  tilt and the splay (capped at what the box allows, and the app says when it falls short). Aim every
  speaker at once, in one undoable step. Linked left/right pairs follow, including their hang.
- **Raked seating.** Set how much higher the back row sits; the maps, the probe, the section and the 3D
  view follow the slope.
- **Metric or imperial.** Choose metres or feet in the Setup tab; every field, axis and readout follows.
  Venue files are always saved in metres.
- **Compare with a measurement.** Import a REW "Export measurement as text", a Smaart ASCII export or any
  CSV of frequency, level and phase. It is drawn dashed over the prediction, smoothed the same way, with its
  level matched to the prediction.
- **Optimize.**
  - *Quick align* (under a second): time-aligns fills and delays to the mains, matches their levels, and
    sets sub delay and polarity.
  - *Deep optimize* (about 15 s, adjustable): a global search over every delay and gain that is free to move,
    weighted by your priorities. It is never worse than where it started.
  - *Auto EQ*: fits a few cut-biased parametric bands per speaker toward the target curve.
  - Locked values are never touched. Every run shows a before/after comparison and can be undone.
- **Take it with you.** Export the processor settings as CSV, text or JSON, or open a printable report.

Projects are saved as plain JSON in `%LOCALAPPDATA%\AcousticPhaseOptimizer` (set `APO_DATA_DIR` to change it).
The last session is restored automatically.

## Command line

```bash
python -m acoustic_phase_optimizer optimize venue.apo.json --deep --eq --report report.html
```

This aligns a saved project without the window. It writes `venue.optimized.apo.json` and prints the settings sheet.

## How the prediction works

- Each speaker is modelled as a point source with a frequency-dependent directivity pattern, its
  high-pass and low-pass filters, and its processor delay, gain, polarity and EQ.
- Reflections come from the image-source method in a box-shaped room, with absorption per octave band for
  each surface and ISO 9613-1 air absorption.
- Sound arriving within 50 ms adds coherently, with phase, so comb filtering shows up. Later reflections and
  the statistical reverberant tail add as energy.
- The score (0 to 100) combines coverage evenness, clarity, timing, summation and tonal balance across the
  audience.
- STI is estimated from the early and late energy and the reverberation time per octave (IEC 60268-16
  weights). It leaves out background noise, and treats late arrivals from delay speakers as useful sound, so
  read it as an upper bound.

Limits worth knowing:

- The room is a box. An imported scan sets its size and is shown in 3D; it does not change the acoustics.
- In a line-array hang every box is computed for the direct sound (each a short line source on the hang's
  curve), so the beam narrowing with frequency, the long throw and the low-frequency build-up come out of
  the model. Reflections from a hang still use one combined source, scaled per frequency to the power the
  boxes actually radiate. Box heights come from the datasheets
  (d&b V8 and Y8: the architect's-specification maximum; Meyer LEOPARD: estimated from its woofer size).
- Raked seating lifts the ears evenly from the front row to the back row (one straight slope). The floor
  under it is still flat for reflections.
- Outdoors, wind and temperature gradients are not modelled.

It is a prediction from an idealised room and datasheet loudspeakers. Use it to plan and to get close, then
confirm delays and EQ with a measurement on site.

## Development

```bash
pip install -e ".[dev]"
pytest                    # the browser tests need: playwright install chromium
node --test tests/js      # mesh repair, .trky and the speaker finder (Node 18 or newer)
ruff check acoustic_phase_optimizer tests tools
```

| Path | What lives there |
|---|---|
| `acoustic_phase_optimizer/engine/` | Physics, metrics, optimizer, presets (numpy, scipy, pydantic) |
| `acoustic_phase_optimizer/server/` | FastAPI app, background optimization jobs, project storage |
| `acoustic_phase_optimizer/web/` | The interface: plain HTML, CSS and ES modules, no build step |
| `acoustic_phase_optimizer/export.py`, `report.py` | Settings sheets and the HTML report |
| `acoustic_phase_optimizer/web/vendor/` | Third-party code, unchanged unless noted (see `THIRD_PARTY.md`, `sha256.txt`) |
| `tests/` | Engine, API and browser (Playwright) tests; `tests/js/` holds the Node tests |
| `tools/vendor_trky.py` | Copies the TRKY code out of the converter page, or `--check`s the vendored copy |

Version 1 (the PyQt desktop app with hardware measurement) is preserved at the git tag `v1-pyqt-final`.

## Licence

MIT
