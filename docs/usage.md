# Usage Guide

## Command Line Interface

```
acoustic-phase-optimizer [-h] [-c CONFIG] [-g] [--headless] [--measure] [-v] [--log-file LOG_FILE]
```

### Options

| Option | Description |
|--------|-------------|
| `-h, --help` | Show help message |
| `-c, --config CONFIG` | Path to YAML configuration file |
| `-g, --gui` | Launch graphical user interface (default) |
| `--headless` | Run optimization in headless mode |
| `--measure` | Generate measurement signals |
| `-v, --verbose` | Enable debug logging |
| `--log-file FILE` | Write logs to file |

## GUI Mode

### Getting Started

1. **Launch**: `acoustic-phase-optimizer --gui`
2. The main window shows a 2D room layout with default speaker/microphone positions
3. Use the control panel on the left to configure parameters

### Tabs

- **Room 2D** — Top-down view of venue with speakers and microphones
- **Room 3D** — Three-dimensional view with room bounds
- **Heatmap** — SPL distribution contour map
- **All Maps** — Multi-panel view (SPL, phase, delay, cancellation, coherence)
- **Frequency Response** — Magnitude and phase plots per speaker-mic pair
- **Group Delay** — Group delay analysis
- **Spectrogram** — Time-frequency representation

### Measurement Panel

Configure:
- Sweep type (log or MLS)
- Duration
- Frequency range
- Averaging count

### Optimization Panel

Configure:
- Algorithm (genetic, gradient, annealing, bayesian, compare_all)
- Max iterations
- Population size
- Mutation rate

### Speaker Panel

Adjust individual speaker parameters:
- Delay (ms)
- Gain (dB)
- Polarity inversion

## Headless Mode

Optimize without GUI:

```bash
acoustic-phase-optimizer --headless
```

Returns JSON with optimization results including best parameters and objective values.

## Measurement Mode

Generate test signals:

```bash
acoustic-phase-optimizer --measure
```

Creates `measurement_signal.wav` in the current directory.

## Configuration Files

Configuration uses YAML format. See `config/default_config.yaml` for all options.

### Custom Venue Setup

```yaml
# my_venue.yaml
acoustic:
  speed_of_sound: 345.0
  temperature: 25.0

optimization:
  default_algorithm: genetic
  max_iterations: 500
  population_size: 100

zones:
  left_main:
    x: -12.0
    y: 0.0
    z: 3.0
  right_main:
    x: 12.0
    y: 0.0
    z: 3.0
```

Usage:
```bash
acoustic-phase-optimizer --gui -c my_venue.yaml
```

## Python API

### Basic Usage

```python
from acoustic_phase_optimizer import AcousticPhaseOptimizer, Config

config = Config("config/my_venue.yaml")
app = AcousticPhaseOptimizer(config)
app.initialize()

# Run headless optimization
result = app.run_headless()
```

### Custom Objective Function

```python
import numpy as np
from acoustic_phase_optimizer.optimization.engine import OptimizationEngine

engine = OptimizationEngine({"max_iterations": 200})

def my_objective(params):
    delay = params[0]
    gain = params[1]
    # Compute acoustic performance metric
    return -((delay - 50) ** 2 + (gain + 3) ** 2) / 1000 + 1

result = engine.optimize("genetic", my_objective, np.array([10.0, 0.0]))
```

### Algorithm Comparison

```python
results = engine.compare_algorithms(
    my_objective,
    np.array([10.0, 0.0]),
    algorithms=["genetic", "bayesian", "annealing"]
)

best_algo, best_result = engine.get_best_result(results)
print(f"Best: {best_algo} = {best_result.best_value}")
```
