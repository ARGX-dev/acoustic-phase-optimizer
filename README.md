# Acoustic Phase Optimizer

**Professional live sound DSP optimization for highly reflective venues.**

Optimizes phase response, frequency response, delay, and speaker interaction across multi-speaker arrays using AI-powered algorithms. Runs on Windows (Linux support planned).

---

## Quick Install (Fresh Windows Machine)

### One-Click Setup

Double-click `install.bat` — it handles everything:

1. Checks if Python is installed (prompts to download if missing)
2. Creates a virtual environment
3. Installs all dependencies (NumPy, SciPy, PyQt6, etc.)
4. Installs the Acoustic Phase Optimizer package

After it finishes, run:

```
venv\Scripts\python -m acoustic_phase_optimizer --gui
```

### Manual Setup

If the automated script doesn't work for your environment:

```bash
# 1. Get the code
git clone https://github.com/ARGX-dev/acoustic-phase-optimizer.git
cd acoustic-phase-optimizer

# 2. Create and activate a virtual environment
python -m venv venv
venv\Scripts\activate

# 3. Install with GUI and dev extras
pip install -e ".[gui,dev]"
```

### Verify

```bash
python -c "from acoustic_phase_optimizer import AcousticPhaseOptimizer; print('OK')"
```

---

## Running the Program

### GUI Mode (recommended for first use)

```bash
python -m acoustic_phase_optimizer --gui
```

This opens an interactive window showing:
- **Room 2D / Room 3D** — Venue layout with speaker and mic positions
- **Heatmap** — SPL distribution across the listening area
- **Frequency Response** — Magnitude and phase plots per speaker-mic pair
- **Group Delay** — Delay vs frequency analysis
- **Spectrogram** — Time-frequency representation

**Workflow in the GUI:**

| Step | What to Do |
|------|------------|
| 1 | The default venue loads automatically (30×20×8m room) |
| 2 | Go to the **Measurement** tab → click **Start Measurement** to run a virtual sweep |
| 3 | Go to the **Optimization** tab → select an algorithm → click **Run Optimization** |
| 4 | View results in the visualisation panels (heatmap, frequency response, etc.) |
| 5 | Go to the **Speakers** tab → adjust individual delay/gain/polarity manually |

### Headless Mode (scripting / automation)

```bash
python -m acoustic_phase_optimizer --headless
```

Outputs JSON with the best-found parameters. Useful for batch processing or CI pipelines.

### Generate Test Signals

```bash
python -m acoustic_phase_optimizer --measure
```

Creates `measurement_signal.wav` with a log-sweep for use with external measurement systems.

### Run the Examples

```bash
# Full measurement pipeline demo
python examples/run_measurement.py

# Venue optimization with algorithm comparison
python examples/optimize_venue.py
```

---

## Configuration

Copy and edit `config/default_config.yaml` to match your venue:

```yaml
acoustic:
  speed_of_sound: 343.0          # Adjust for temperature/humidity
  temperature: 20.0

optimization:
  default_algorithm: genetic      # genetic, gradient, annealing, bayesian
  max_iterations: 1000
  population_size: 100

zones:
  left_main:
    x: -10.0
    y: 0.0
    z: 2.0
  right_main:
    x: 10.0
    y: 0.0
    z: 2.0
  subwoofer:
    x: 0.0
    y: 2.0
    z: 0.0
```

Use your config:

```bash
python -m acoustic_phase_optimizer --gui -c my_venue.yaml
```

---

## Operational Workflow

```
┌─────────────────────────────────────────────────────┐
│  1. SETUP VENUE                                     │
│     Define room dimensions, speaker positions,       │
│     microphone measurement points in YAML config     │
├─────────────────────────────────────────────────────┤
│  2. MEASURE                                         │
│     Generate log-sweep or MLS test signals           │
│     Record impulse responses at each mic position    │
│     Extract phase, magnitude, group delay, RT60      │
├─────────────────────────────────────────────────────┤
│  3. MODEL                                           │
│     Build 3D room model with reflective surfaces     │
│     Compute image-source reflections                 │
│     Detect comb filtering and phase cancellation     │
├─────────────────────────────────────────────────────┤
│  4. OPTIMIZE                                        │
│     Choose algorithm (genetic recommended first)     │
│     Set objective weights (phase, magnitude, delay)  │
│     Run optimization                                 │
│     Compare algorithms if needed (--compare)         │
├─────────────────────────────────────────────────────┤
│  5. APPLY                                           │
│     Export optimised DSP parameters                  │
│     Apply delays, gains, FIR/IIR, crossover to DSP   │
│     Verify with post-optimisation measurement        │
└─────────────────────────────────────────────────────┘
```

---

## Command Reference

| Command | Description |
|---------|-------------|
| `-g, --gui` | Launch the interactive GUI |
| `--headless` | Run optimisation from the command line |
| `--measure` | Generate WAV test signals |
| `-c FILE, --config FILE` | Path to YAML configuration file |
| `-v, --verbose` | Show debug-level logging |
| `--log-file FILE` | Write logs to a file |
| `-h, --help` | Show all options |

---

## Project Structure

```
acoustic_phase_optimizer/
├── acoustic_phase_optimizer/    # Main Python package
│   ├── measurement/             # Sweep gen, IR extraction, RT60
│   ├── acoustic/                # Room model, speakers, reflections
│   ├── optimization/            # 4 optimisation algorithms
│   ├── dsp/                     # DSP hardware abstraction layer
│   ├── simulation/              # Virtual room + virtual DSP
│   ├── visualization/           # PyQt6 GUI panels
│   └── utils/                   # Math, audio I/O, logging
├── config/                      # YAML config files
├── examples/                    # Runnable example scripts
├── tests/                       # 210 unit & integration tests
│   └── test_*.py                # pytest test suites
└── docs/                        # Architecture, dev, install guides
```

---

## Running Tests

```bash
pytest tests/ -v
```

Requires the `dev` extras: `pip install -e ".[dev]"`

All 210 tests should pass.

---

## Troubleshooting

| Problem | Solution |
|---------|----------|
| `ModuleNotFoundError: No module named 'PyQt6'` | `pip install -e ".[gui]"`  |
| GUI won't open | You installed without `[gui]`; re-run `pip install -e ".[gui]"` or use `--headless` |
| `No audio device found` | Plug in a USB audio interface or microphone |
| Test failures on `gradient` tests | Gradient optimizer is sensitive to learning rate; use `genetic` or `annealing` for real work |
| Test failures on `gradient` tests | Gradient optimizer is sensitive to learning rate; use `genetic` or `annealing` for real work |

---

## Features

- Room measurement (log sweep, MLS, IR, phase, magnitude, RT60)
- 3D acoustic modelling with image-source reflections
- 4 optimisation algorithms: gradient, genetic, annealing, Bayesian
- Interactive PyQt6 visualisation (room layout, heatmaps, frequency plots)
- DSP abstraction layer (Venue360, Dante, AES67 — stubs ready)
- Virtual room simulation for hardware-free testing
- 210 passing unit and integration tests

---

## License

MIT License — see [LICENSE](LICENSE).
