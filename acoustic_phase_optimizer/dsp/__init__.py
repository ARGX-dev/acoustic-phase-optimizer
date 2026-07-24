"""
DSP interface module for hardware communication and filter design.

Provides an abstraction layer for DSP processors including dbx DriveRack Venue360,
generic DSP devices, Dante devices, and AES67 devices. Includes filter design
(FIR/IIR), crossover optimization, and signal processing utilities.
"""

from acoustic_phase_optimizer.dsp.interface import DSPInterface
from acoustic_phase_optimizer.dsp.filters import FilterDesign
from acoustic_phase_optimizer.dsp.crossover import CrossoverDesigner

__all__ = [
    "DSPInterface",
    "FilterDesign",
    "CrossoverDesigner",
]
