# Installation Guide

## Prerequisites

- **Python 3.10 or higher**
- **pip** (Python package installer)
- **Audio interface** (for real hardware measurements)
- **Measurement microphone** (calibrated, for real measurements)

## Standard Installation

### From Source

```bash
# Clone the repository
git clone https://github.com/anomalyco/acoustic-phase-optimizer.git
cd acoustic-phase-optimizer

# Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or: venv\Scripts\activate  # Windows

# Install base package
pip install -e .

# Install with GUI support
pip install -e .[gui]

# Full installation with all optional features
pip install -e .[full]
```

### Using pip (future)

```bash
pip install acoustic-phase-optimizer
```

## Dependencies

### Core (Required)
- `numpy>=1.24` — Numerical computing
- `scipy>=1.10` — Signal processing, optimization
- `sounddevice>=0.4` — Audio playback/recording
- `soundfile>=0.12` — WAV file I/O
- `pyyaml>=6.0` — Configuration files
- `matplotlib>=3.7` — Plotting
- `numba>=0.57` — JIT compilation for performance

### GUI (Optional)
- `PyQt6>=6.5` — GUI framework
- `pyqtgraph>=0.13` — Real-time plotting
- `pyopengl>=3.1` — 3D acceleration

### Full (Optional)
- `librosa>=0.10` — Audio analysis
- `scikit-learn>=1.3` — Gaussian processes

## Platform Support

### Windows
- Tested on Windows 10/11
- ASIO drivers recommended for low-latency audio
- Use `sounddevice` with ASIO backend

### Linux (Future)
- ALSA or JACK audio backend
- PulseAudio support via sounddevice

### macOS (Future)
- CoreAudio support via sounddevice

## Verification

After installation, verify with:

```bash
python -c "from acoustic_phase_optimizer import AcousticPhaseOptimizer; print('OK')"
```

Run tests:

```bash
pip install pytest
pytest tests/ -v
```

## Audio Setup

For real measurements:

1. Connect measurement microphone to audio interface
2. Configure audio device in config.yaml:
   ```yaml
   system:
     sample_rate: 48000
     buffer_size: 1024
   ```
3. Verify input/output with:
   ```bash
   python -c "import sounddevice; print(sounddevice.query_devices())"
   ```
