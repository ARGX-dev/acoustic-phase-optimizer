"""
Acoustic Phase Optimizer

Professional live sound DSP optimization platform for highly reflective venues.
Measures, models, and optimizes phase response, frequency response, delays,
and speaker interactions across multi-speaker sound systems.
"""

__version__ = "0.1.0"
__author__ = "Acoustic Phase Optimizer Team"
__license__ = "MIT"

from acoustic_phase_optimizer.config import Config
from acoustic_phase_optimizer.main import AcousticPhaseOptimizer


__all__ = [
    "Config",
    "AcousticPhaseOptimizer",
]
