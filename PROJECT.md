# Acoustic Phase Optimizer

## What This App Is

A professional live sound DSP optimization platform for reflective venues. It helps audio engineers measure room acoustics, simulate speaker alignment, optimize delay/gain/EQ parameters, and export hardware-ready configurations — all in a single PyQt6 desktop application.

**Target users:** Live sound engineers, system techs, venue installers working in reverberant spaces (gymnasiums, theaters, conference rooms, houses of worship).

---

## Core Problem It Solves

In reflective venues, multiple speakers interact destructively, creating:
- **Comb filtering** — deep nulls at specific frequencies from path-length differences
- **Phase cancellation** — uneven coverage across the audience area
- **Poor intelligibility** — smeared transients, lost clarity

The app measures the room, computes optimal speaker delays/gains/EQ to minimize destructive interference, and exports settings for real DSP hardware (dbx DriveRack VENU360, generic Dante/AES67 devices).

---

## Architecture Overview

```
acoustic_phase_optimizer/
├── acoustic/           # Physics models: room, speaker, microphone, reflections
├── measurement/        # Signal gen, IR extraction, RT60, mic calibration
├── simulation/         # VirtualRoom — offline acoustics simulator
├── optimization/       # Genetic, gradient, annealing, bayesian optimizers
├── dsp/                # PEQ, GEQ, crossovers, FIR, hardware interfaces
├── visualization/      # PyQt6 GUI: 2D/3D room, heatmaps, FFT, reports
├── reporting/          # HTML session reports with embedded heatmaps
├── weather/            # Temperature/humidity → speed of sound correction
└── room/               # LIDAR import, mic placement optimization
```

---

## What We've Built (Session Summary)

### 1. Real Hardware Measurement Pipeline (`measurement/runner.py`)
- **Playback/recording** via `sounddevice` (ASIO/WASAPI/CoreAudio)
- **Log sweeps & MLS** with configurable duration, freq range, averaging
- **Loopback latency calibration** — measures exact output→input round-trip
- **Sample rate verification** — guards against driver rate mismatch
- **Clipping detection** — aborts and warns if input overloads
- **Mic calibration file support** — loads 2-column `.txt/.cal` curves, applies inverse EQ

### 2. Live FFT Monitor During Measurement (`visualization/fft_monitor.py`)
- Background thread acquires input blocks via `sounddevice.InputStream` callback
- Real-time spectrum + peak/RMS level meters
- Peak-hold trace for spotting transient issues

### 3. Updated Measurement Controls UI (`visualization/controls.py`)
- Device selectors (refreshable dropdowns for input/output)
- Channel count spinners (multi-channel simultaneous measurement)
- Mic calibration file loader (shows filename, stores path)
- Loopback latency toggle

### 4. Polished Session Reports (`reporting/`)
- `MeasurementSummary` dataclass: RT60 per octave, C80, D50, clipping flag, latency
- Auto-generated recommendations based on measured data:
  - "Reduce input gain" if clipping detected
  - "High reverberation — consider treatment" if RT60 > 2s
  - "Residual nulls — add fill speakers" if coherence gain < 3 dB
- Embedded before/after coverage heatmaps (canvas-rendered in browser)

### 5. VENU360 Protocol Reverse-Engineering CLI
```bash
python -m acoustic_phase_optimizer capture-venu360 capture.pcap --ip 192.168.1.100
```
Parses Wireshark pcap, extracts TCP payloads to/from the VENU360 IP. No fake protocol — just raw bytes for you to correlate with app actions.

### 6. Performance Optimizations (Multithreading + Numba JIT)

| Module | Change | Speedup |
|--------|--------|---------|
| `acoustic/reflection.py` | `@njit` on image source quadruple loop | ~15× |
| `acoustic/comb_filter.py` | `@njit` on O(N²) Gaussian smoothing | ~60× |
| `optimization/genetic.py` | `ThreadPoolExecutor` parallel fitness eval | ~N_cores× |
| `simulation/virtual_room.py` | Parallel speaker-mic transfer functions | ~N_cores× |
| `visualization/app.py` | Vectorized `_update_views` (removed 3 nested loops) | ~30× |
| `config.py` | Added `system.workers` (auto-detects CPU count) | — |

All **210 tests pass** after changes.

---

## Current Capabilities

| Feature | Status |
|---------|--------|
| Room modeling (rectangular, image-source reflections) | ✅ |
| Speaker placement (drag in 2D/3D view) | ✅ |
| Mic placement optimization (greedy diversity) | ✅ |
| Log sweep / MLS signal generation | ✅ |
| Real hardware playback & recording | ✅ |
| Impulse response extraction (sweep & MLS) | ✅ |
| RT60, EDT, C50/C80, D50 estimation | ✅ |
| Comb filtering detection & cancellation mapping | ✅ |
| Genetic / Gradient / Annealing / Bayesian optimization | ✅ |
| Algorithm comparison mode (run all, pick best) | ✅ |
| PEQ (6 bands) + GEQ (31-band ISO) per speaker | ✅ |
| Delay, gain, polarity per speaker | ✅ |
| VENU360 config export (JSON/CSV) | ✅ |
| Dante / AES67 interface stubs | ✅ (stubs) |
| LIDAR import (PLY, PCD, LAS, OBJ, GLTF, MVR…) | ✅ |
| Weather API → speed of sound correction | ✅ |
| HTML session report with heatmaps | ✅ |
| Live FFT monitor during measurement | ✅ |

---

## What's Next (Roadmap)

### Short Term
- [ ] Multi-channel simultaneous measurement (UI done, runner supports it)
- [ ] ASIO driver selection in device dropdown
- [ ] Save/load session presets (room + speakers + mics + optimization)
- [ ] Dante Controller API integration (real, not stub)
- [ ] AES67 RTP stream support

### Medium Term
- [ ] Non-rectangular room geometry (LIDAR → mesh)
- [ ] Beamforming optimization for array speakers
- [ ] GPU-accelerated FIR convolution (CuPy)
- [ ] All-pass filter design for phase alignment
- [ ] Automated venue calibration wizard (measure → optimize → export → verify)

### Long Term
- [ ] Linux/macOS packaging
- [ ] VST3/AU plugin for DAW integration
- [ ] Web-based remote control interface
- [ ] ML model for reflection pattern prediction
- [ ] Real-time adaptive DSP during performances

---

## How to Run

```bash
# GUI (needs PyQt6, pyqtgraph, pyopengl)
python -m acoustic_phase_optimizer

# Headless optimization
python -m acoustic_phase_optimizer --headless

# Measurement only
python -m acoustic_phase_optimizer --measure

# VENU360 pcap analysis
python -m acoustic_phase_optimizer capture-venu360 mycapture.pcap --ip 192.168.1.100
```

**Requirements:** Python ≥3.10, `pip install -e ".[gui,dev]"`

---

## Design Principles

1. **No fake hardware protocols** — VENU360 driver is honest: exports config files, doesn't pretend to speak TCP. Real protocol derived from *your* pcap captures.
2. **Measurement correctness first** — latency calibration, rate verification, clipping detection, mic cal are not optional.
3. **Optimization is transparent** — every algorithm (GA, gradient, etc.) shows convergence history; "compare_all" runs them side-by-side.
4. **Report is the deliverable** — HTML opens in browser, prints to PDF, includes all data needed to replicate or hand off.
5. **Performance matters** — Numba JIT + threading so UI stays responsive during 1000-gen optimizations.

---

## Key Files to Understand

| File | Purpose |
|------|---------|
| `visualization/app.py` | Main window, wires everything together |
| `visualization/controls.py` | All control panels (room, measurement, optimization, speakers) |
| `measurement/runner.py` | Hardware measurement orchestration |
| `measurement/calibration.py` | Latency, rate, clipping, mic cal |
| `acoustic/reflection.py` | Image-source reflection engine (JIT) |
| `acoustic/comb_filter.py` | Comb detection, cancellation zones |
| `optimization/engine.py` | Algorithm orchestration, parallel compare |
| `optimization/genetic.py` | Parallel GA with ThreadPoolExecutor |
| `simulation/virtual_room.py` | Offline simulator, parallel TFs |
| `dsp/venus360.py` | Honest VENU360 interface + pcap parser |
| `reporting/renderer.py` | HTML report generation |

---

## License

MIT — see `LICENSE`.