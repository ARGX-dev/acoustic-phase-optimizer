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

Everything runs on your own computer. The window talks to a small local engine on `127.0.0.1`; nothing goes online.

## What you can do

- **Model the venue.** Room size, a material for each surface, the audience area, temperature and humidity.
  Four example venues (gymnasium, theatre, house of worship, conference room) or an empty room.
- **Place speakers on the plan.** Drag to move, drag the handle to aim. Mains, fills, front fills, delays and
  subs, from eight generic loudspeaker types or your own coverage figures. Left/right pairs are linked so
  they stay matched.
- **See the result live.** Maps of direct level (all frequencies or one octave), total level, clarity,
  summation and cancellation, and arrival-time spread. Click anywhere for the magnitude, phase and arrivals
  at that seat, or add measurement-mic positions.
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

It is a prediction from an idealised room and generic loudspeakers. Use it to plan and to get close, then
confirm delays and EQ with a measurement on site.

## Development

```bash
pip install -e ".[dev]"
pytest
ruff check acoustic_phase_optimizer tests
```

| Path | What lives there |
|---|---|
| `acoustic_phase_optimizer/engine/` | Physics, metrics, optimizer, presets (numpy, scipy, pydantic) |
| `acoustic_phase_optimizer/server/` | FastAPI app, background optimization jobs, project storage |
| `acoustic_phase_optimizer/web/` | The interface: plain HTML, CSS and ES modules, no build step |
| `acoustic_phase_optimizer/export.py`, `report.py` | Settings sheets and the HTML report |
| `tests/` | Engine and API tests |

Version 1 (the PyQt desktop app with hardware measurement) is preserved at the git tag `v1-pyqt-final`.

## Licence

MIT
