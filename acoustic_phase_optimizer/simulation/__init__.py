"""
Simulation module for virtual room acoustics.

Provides virtual room simulation for optimization without requiring real hardware.
Generates synthetic impulse responses based on room geometry and speaker positions.
"""

from acoustic_phase_optimizer.simulation.virtual_room import VirtualRoom
from acoustic_phase_optimizer.simulation.virtual_dsp import VirtualDSP

__all__ = [
    "VirtualRoom",
    "VirtualDSP",
]
