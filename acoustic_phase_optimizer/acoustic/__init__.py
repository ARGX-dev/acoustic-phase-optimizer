"""
Acoustic modeling module for room geometry and speaker interaction.

Provides 3D room models, speaker and microphone definitions,
reflection estimation, comb filtering detection, and phase cancellation mapping.
"""

from acoustic_phase_optimizer.acoustic.room_model import RoomModel
from acoustic_phase_optimizer.acoustic.speaker import Speaker
from acoustic_phase_optimizer.acoustic.microphone import Microphone
from acoustic_phase_optimizer.acoustic.reflection import ReflectionEngine
from acoustic_phase_optimizer.acoustic.comb_filter import CombFilterDetector

__all__ = [
    "RoomModel",
    "Speaker",
    "Microphone",
    "ReflectionEngine",
    "CombFilterDetector",
]
