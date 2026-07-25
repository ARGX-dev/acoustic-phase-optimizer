"""
Measurement module for room acoustics acquisition.

Provides signal generation (log sweep, MLS), impulse response extraction,
phase/magnitude analysis, group delay, and RT60 estimation.
"""

from acoustic_phase_optimizer.measurement.signal_generator import SignalGenerator
from acoustic_phase_optimizer.measurement.impulse_response import ImpulseResponse
from acoustic_phase_optimizer.measurement.room_analysis import RoomAnalysis
from acoustic_phase_optimizer.measurement.rt60 import RT60Estimator
from acoustic_phase_optimizer.measurement.calibration import (
    measure_loopback_latency,
    open_verified_stream,
    check_clipping,
    ClippingReport,
    load_mic_calibration,
    apply_mic_calibration,
)

__all__ = [
    "SignalGenerator",
    "ImpulseResponse",
    "RoomAnalysis",
    "RT60Estimator",
    "measure_loopback_latency",
    "open_verified_stream",
    "check_clipping",
    "ClippingReport",
    "load_mic_calibration",
    "apply_mic_calibration",
]
