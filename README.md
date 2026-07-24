# Acoustic Phase Optimizer

**Professional live sound DSP optimization for highly reflective venues.**

A software platform for measuring, modelling, and optimizing live sound systems by analyzing phase response, frequency response, delay, and speaker interaction across multi-speaker arrays.

## Features

- **Room Measurement** — Log sweep and MLS signal generation, impulse response extraction, phase/magnitude analysis, group delay, RT60 estimation
- **Acoustic Modelling** — 3D room model with surfaces, image-source reflection estimation, comb filtering detection, phase cancellation mapping
- **Optimization Engine** — Gradient descent, genetic algorithm, simulated annealing, and Bayesian optimization; computes speaker delays, sub alignment, crossover optimization, FIR/IIR filters, EQ, gain, polarity
- **Visualization** — Interactive PyQt6 GUI with 2D/3D room layout, SPL/phase/delay heatmaps, frequency response, group delay, spectrograms, cancellation zone mapping
- **DSP Abstraction** — Extensible interface for dbx DriveRack Venue360, generic DSP, Dante, and AES67 devices
- **Simulation** — Virtual room acoustics for hardware-free optimization testing
- **AI Optimization** — Multiple optimization algorithms with automatic comparison

## System Architecture

```
acoustic_phase_optimizer/
├── measurement/       # Signal generation, IR extraction, RT60
├── acoustic/          # Room model, speakers, reflections, comb filtering
├── optimization/      # Gradient, genetic, annealing, Bayesian optimizers
├── dsp/               # DSP interface abstraction, filters, crossover
├── simulation/        # Virtual room and virtual DSP
├── visualization/     # PyQt6 GUI with multiple view panels
└── utils/             # Math, audio, logging utilities
```

## Mathematical Background

### Room Acoustic Parameters

| Parameter | Formula | Description |
|-----------|---------|-------------|
| RT60 (Sabine) | `T = 0.161 V / (S α)` | Reverberation time |
| RT60 (Eyring) | `T = 0.161 V / (-S ln(1-α))` | More accurate for absorptive rooms |
| Schroeder Frequency | `f_s = 2000 √(T_60 / V)` | Transition frequency |
| Clarity | `C_t = 10 log₁₀(E_early / E_late)` | Speech/music clarity |
| Definition | `D_50 = E_0-50ms / E_total` | Speech definition |

### Optimization Algorithms

- **Gradient Descent**: Numerical gradient with adaptive learning rate and momentum
- **Genetic Algorithm**: Tournament selection, uniform crossover, adaptive mutation
- **Simulated Annealing**: Adaptive cooling with re-annealing and Metropolis acceptance
- **Bayesian Optimization**: Gaussian process surrogate with expected improvement

### Objective Function

The optimization maximizes:
```
J = w₁·C_phase + w₂·M_flat + w₃·I_cancel + w₄·D_align + w₅·R_dev
```

Where C_phase is phase coherence, M_flat is magnitude flatness, I_cancel penalizes destructive interference, D_align rewards delay alignment, and R_dev targets desired RT60.

## Installation

### Requirements

- Python 3.10+
- NumPy, SciPy, sounddevice, soundfile, PyYAML, matplotlib
- PyQt6 and pyqtgraph (optional, for GUI)

### Install

```bash
# Clone the repository
git clone https://github.com/anomalyco/acoustic-phase-optimizer.git
cd acoustic-phase-optimizer

# Basic installation
pip install -e .

# With GUI support
pip install -e .[gui]

# Full installation (all features)
pip install -e .[full]
```

### Quick Start

```bash
# Launch GUI
acoustic-phase-optimizer --gui

# Headless optimization
acoustic-phase-optimizer --headless

# Generate measurement signal
acoustic-phase-optimizer --measure

# With custom config
acoustic-phase-optimizer --gui -c config/my_venue.yaml

# Verbose logging
acoustic-phase-optimizer --gui -v --log-file optimizer.log
```

## Usage

### GUI Mode

1. Launch with `acoustic-phase-optimizer --gui`
2. Configure venue dimensions and speaker positions in controls
3. Run measurements or use virtual room simulation
4. Select optimization algorithm and parameters
5. View results in real-time across all visualization panels

### Headless Mode

```python
from acoustic_phase_optimizer import AcousticPhaseOptimizer
from acoustic_phase_optimizer.config import Config

config = Config("config/my_venue.yaml")
app = AcousticPhaseOptimizer(config)
app.initialize()

# Run optimization
result = app.run_headless()
print(f"Best value: {result['best_value']}")
```

### Programmatic API

```python
import numpy as np
from acoustic_phase_optimizer.acoustic.room_model import RoomModel
from acoustic_phase_optimizer.acoustic.speaker import Speaker, SpeakerType
from acoustic_phase_optimizer.simulation.virtual_room import VirtualRoom
from acoustic_phase_optimizer.optimization.engine import OptimizationEngine

room = RoomModel()
vr = VirtualRoom(room)

speaker = Speaker("Main", SpeakerType.MAIN_LEFT, np.array([-5.0, 0.0, 2.0]))
vr.add_speaker(speaker)

engine = OptimizationEngine({"max_iterations": 100})

def objective(params):
    speaker.delay_ms = float(params[0])
    return -abs(params[0] - 50.0) / 50.0 + 1.0

result = engine.optimize("genetic", objective, np.array([10.0]))
print(f"Optimal delay: {result.best_params[0]:.1f}ms")
```

## Folder Structure

```
acoustic_phase_optimizer/
├── acoustic_phase_optimizer/   # Main package
│   ├── measurement/            # Room measurement signals and analysis
│   ├── acoustic/               # Room and speaker modelling
│   ├── optimization/           # Optimization algorithms
│   ├── dsp/                    # DSP hardware abstraction
│   ├── simulation/             # Virtual room and DSP
│   ├── visualization/          # PyQt6 GUI
│   └── utils/                  # Support utilities
├── config/                     # YAML configuration files
├── docs/                       # Documentation
├── tests/                      # Unit and integration tests
├── examples/                   # Usage examples
└── data/                       # Sample data and virtual rooms
```

## Development Roadmap

| Phase | Feature | Status |
|-------|---------|--------|
| 1 | Core measurement and IR extraction | Done |
| 1 | Acoustic modelling and room simulation | Done |
| 1 | Optimization algorithms | Done |
| 1 | Virtual room for testing | Done |
| 1 | PyQt6 visualization | Done |
| 2 | Real hardware recording/playback | Planned |
| 2 | Venue360 control integration | Stub |
| 2 | Dante/AES67 support | Stub |
| 3 | Real-time monitoring | Planned |
| 3 | Multi-channel simultaneous measurement | Planned |
| 3 | Automated venue calibration wizard | Planned |
| 4 | Linux support | Planned |
| 4 | VST3/AU plugin integration | Future |

## Known Limitations

- Venue360, Dante, and AES67 interfaces are stubs awaiting hardware integration
- Image-source reflection model uses simplified rectangular rooms
- Gaussian Process implementation is basic (full scikit-learn integration available with `[full]` install)
- Real-time measurement requires audio interface configuration
- GUI requires PyQt6 (optional dependency)

## License

MIT License — see LICENSE for details.
