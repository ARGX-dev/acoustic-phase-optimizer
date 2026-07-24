# Developer Documentation

## Getting Started

### Setup Development Environment

```bash
git clone https://github.com/anomalyco/acoustic-phase-optimizer.git
cd acoustic-phase-optimizer
python -m venv venv
source venv/bin/activate
pip install -e .[full]
pip install pytest pytest-cov mypy
```

### Running Tests

```bash
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=acoustic_phase_optimizer --cov-report=html

# Run specific test file
pytest tests/test_optimization.py -v

# Run specific test class
pytest tests/test_measurement.py::TestSignalGenerator -v
```

### Code Style

- Type hints required for all function signatures
- NumPy docstrings for public APIs
- Follow PEP 8
- Use `numpy.typing.NDArray` for array types
- Use dataclasses for data containers
- Use abstract base classes for interfaces

## Project Structure

```
acoustic_phase_optimizer/
├── acoustic_phase_optimizer/    # Main package
│   ├── __init__.py              # Package exports
│   ├── main.py                  # Entry point and CLI
│   ├── config.py                # Configuration management
│   ├── measurement/             # Room measurement
│   │   ├── signal_generator.py  # Log sweep, MLS generation
│   │   ├── impulse_response.py  # IR extraction, phase/magnitude
│   │   ├── room_analysis.py     # Acoustic parameter analysis
│   │   └── rt60.py              # Reverberation time estimation
│   ├── acoustic/                # Acoustic modelling
│   │   ├── room_model.py        # 3D room with surfaces
│   │   ├── speaker.py           # Speaker definition
│   │   ├── microphone.py        # Microphone definition
│   │   ├── reflection.py        # Image-source reflection engine
│   │   └── comb_filter.py       # Comb filtering detection
│   ├── optimization/            # Optimization algorithms
│   │   ├── engine.py            # Optimization orchestrator
│   │   ├── objectives.py        # Objective functions
│   │   ├── constraints.py       # Parameter constraints
│   │   ├── gradient.py          # Gradient descent
│   │   ├── genetic.py           # Genetic algorithm
│   │   ├── annealing.py         # Simulated annealing
│   │   └── bayesian.py          # Bayesian optimization
│   ├── dsp/                     # DSP interface
│   │   ├── interface.py         # Abstract base class
│   │   ├── filters.py           # FIR/IIR filter design
│   │   ├── crossover.py         # Crossover network design
│   │   ├── venus360.py          # dbx DriveRack Venue360
│   │   ├── generic.py           # Generic DSP
│   │   ├── dante.py             # Dante interface
│   │   └── aes67.py             # AES67 interface
│   ├── simulation/              # Simulation
│   │   ├── virtual_room.py      # Virtual room acoustics
│   │   └── virtual_dsp.py       # Software DSP processor
│   ├── visualization/           # GUI
│   │   ├── app.py               # Main application window
│   │   ├── room_view.py         # Room layout views
│   │   ├── heatmap.py           # Heat map visualizations
│   │   ├── frequency_view.py    # Frequency response plots
│   │   └── controls.py          # Control panel widgets
│   └── utils/                   # Utilities
│       ├── math_utils.py        # Math helpers
│       ├── audio_utils.py       # Audio I/O
│       └── logging.py           # Logging setup
├── tests/                       # Test suite
├── config/                      # Configuration files
├── docs/                        # Documentation
├── examples/                    # Usage examples
└── data/                        # Sample data
```

## Adding New Features

### New Optimization Algorithm

1. Create class in `optimization/` implementing the algorithm
2. Implement `optimize(objective_fn, initial_params) -> (best_params, best_value, history)`
3. Register in `OptimizationEngine._algorithms`
4. Add factory method in `OptimizationEngine`
5. Add to `ComparisonAlgorithm`

### New DSP Hardware

1. Create class in `dsp/` extending `DSPInterface`
2. Implement all abstract methods
3. Register in `DSPInterface.create()` factory
4. Add tests

### New Measurement Signal

1. Add generation method to `SignalGenerator`
2. Add extraction method to `ImpulseResponse`
3. Add to `generate()` dispatch

## Testing Guidelines

- Every module must have corresponding tests in `tests/`
- Tests should cover normal operation and edge cases
- Integration tests in `test_integration.py` verify end-to-end workflows
- Use `pytest` fixtures for shared setup
- Achieve >80% code coverage

## Performance Considerations

- Use `numba` JIT for hot loops if needed
- FFT operations should use power-of-two lengths
- Cache computed results where appropriate
- Profile before optimizing: `python -m cProfile -o profile.out script.py`

## Building Documentation

Documentation is written in Markdown. Generate HTML docs:

```bash
pip install mkdocs mkdocs-material
mkdocs build
```
