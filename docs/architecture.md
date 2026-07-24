# Architecture Documentation

## Overview

The Acoustic Phase Optimizer follows a modular, layered architecture designed for extensibility and testability. Each module has a single responsibility and communicates through well-defined interfaces.

## System Layers

### 1. Measurement Layer
```
SignalGenerator → (play/record) → ImpulseResponse → RoomAnalysis
```
- Generates test signals (log sweep, MLS)
- Extracts impulse responses via deconvolution
- Analyzes phase, magnitude, group delay, RT60

### 2. Acoustic Model Layer
```
RoomModel + Speaker + Microphone → ReflectionEngine → CombFilterDetector
```
- 3D room geometry with surfaces and materials
- Speaker and microphone definitions with positions
- Image-source reflection estimation
- Comb filtering and phase cancellation detection

### 3. Optimization Layer
```
ObjectiveFunction + Constraints → Engine → [Gradient|Genetic|Annealing|Bayesian]
```
- Multiple optimization algorithms with common interface
- Configurable objective weights
- Parameter bounds and constraint handling
- Algorithm comparison and voting

### 4. Simulation Layer
```
VirtualRoom + VirtualDSP
```
- Software-based room acoustics simulation
- Software DSP processing (delays, gains, filters, EQ)
- Enables optimization without real hardware

### 5. DSP Interface Layer
```
DSPInterface (abstract)
├── Venus360 (dbx DriveRack)
├── GenericDSP (manual/software)
├── DanteInterface
└── AES67Interface
```
- Abstract base class for all hardware
- Common API for delays, gains, filters, crossover, EQ
- Factory method for instantiation

### 6. Visualization Layer
```
VisualizationApp
├── RoomViewWidget / Room3DViewWidget
├── HeatmapWidget / MultiHeatmapWidget
├── FrequencyResponseWidget
├── GroupDelayWidget / SpectrogramWidget
└── ControlPanel
```
- PyQt6-based interactive GUI
- Real-time data updates
- Multiple synchronized view panels

## Design Patterns

- **Abstract Factory**: `DSPInterface.create()` instantiates the correct hardware interface
- **Strategy**: Optimization algorithms are interchangeable strategies
- **Observer**: GUI panels observe data changes via signals
- **Facade**: `AcousticPhaseOptimizer` provides a unified entry point
- **Repository**: `Config` manages hierarchical configuration

## Data Flow

```
Measurement Data
    │
    ▼
Objective Function ◄── Acoustic Model
    │
    ▼
Optimization Engine ──► DSP Parameters
    │                        │
    ▼                        ▼
Simulation ───► Visualization    DSP Hardware
```

## Module Dependencies

```
utils (no deps)
    ├── measurement
    ├── acoustic
    ├── dsp
    ├── simulation
    │       ├── acoustic
    │       └── dsp
    ├── optimization
    │       ├── measurement
    │       └── acoustic
    ├── visualization
    │       ├── acoustic
    │       └── optimization
    └── main
            └── all modules
```
