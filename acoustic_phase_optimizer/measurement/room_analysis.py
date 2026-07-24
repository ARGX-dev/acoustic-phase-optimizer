"""Room acoustic analysis from impulse response measurements."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy import signal
from typing import Optional, Tuple
from acoustic_phase_optimizer.measurement.impulse_response import ImpulseResponse
from acoustic_phase_optimizer.measurement.rt60 import RT60Estimator
from acoustic_phase_optimizer.utils.logging import get_logger

logger = get_logger(__name__)


class RoomAnalysis:
    """Comprehensive room acoustic analysis from measurements."""

    def __init__(self, sample_rate: int = 48000):
        self.sample_rate = sample_rate
        self.ir_module = ImpulseResponse(sample_rate)
        self.rt60_module = RT60Estimator(sample_rate)

    def analyze_impulse_response(
        self,
        ir: NDArray[np.float64],
    ) -> dict:
        freqs_phase, phase = self.ir_module.extract_phase(ir)
        freqs_mag, magnitude_db = self.ir_module.extract_magnitude(ir)
        freqs_gd, group_delay = self.ir_module.compute_group_delay(ir)

        rt60 = self.rt60_module.estimate_rt60(ir)
        edt = self.rt60_module.estimate_edt(ir)
        clarity = self.rt60_module.clarity(ir)
        definition = self.rt60_module.definition(ir)
        center_time = self.rt60_module.center_time(ir)

        return {
            "magnitude_db": magnitude_db,
            "magnitude_freqs": freqs_mag,
            "phase": phase,
            "phase_freqs": freqs_phase,
            "group_delay": group_delay,
            "group_delay_freqs": freqs_gd,
            "rt60": rt60,
            "rt60_freqs": self.rt60_module.freqs if hasattr(self.rt60_module, 'freqs') else None,
            "edt": edt,
            "clarity_50": clarity.get("c50", 0.0),
            "clarity_80": clarity.get("c80", 0.0),
            "definition_50": definition.get("d50", 0.0),
            "center_time": center_time,
        }

    def detect_reflections(
        self,
        ir: NDArray[np.float64],
        threshold_db: float = -20.0,
        min_distance_samples: int = 10,
    ) -> list[dict]:
        envelope = np.abs(ir)
        envelope_db = 20.0 * np.log10(envelope + 1e-12)

        reflections = []
        i = min_distance_samples
        while i < len(envelope_db) - 1:
            if (envelope_db[i] > envelope_db[i - 1] and
                envelope_db[i] > envelope_db[i + 1] and
                envelope_db[i] > threshold_db):

                peak_start = max(0, i - 2)
                peak_end = min(len(ir), i + 3)
                peak_idx = peak_start + np.argmax(np.abs(ir[peak_start:peak_end]))

                reflections.append({
                    "sample": int(peak_idx),
                    "time_ms": peak_idx * 1000.0 / self.sample_rate,
                    "amplitude": float(np.abs(ir[peak_idx])),
                    "amplitude_db": float(20.0 * np.log10(np.abs(ir[peak_idx]) + 1e-12)),
                    "relative_db": float(
                        20.0 * np.log10(np.abs(ir[peak_idx]) / (np.max(np.abs(ir)) + 1e-12) + 1e-12)
                    ),
                })
                i += min_distance_samples
            i += 1

        return reflections

    def compute_energy_decay_curve(
        self,
        ir: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        squared = ir ** 2
        decay = np.cumsum(squared[::-1])[::-1]
        decay_db = 10.0 * np.log10(decay + 1e-12)
        decay_db -= decay_db[0]
        return decay_db

    def estimate_noise_floor(
        self,
        ir: NDArray[np.float64],
        tail_percent: float = 0.2,
    ) -> float:
        if len(ir) < 10:
            return float(20.0 * np.log10(np.max(np.abs(ir)) + 1e-12))
        tail_start = int(len(ir) * (1.0 - tail_percent))
        tail = ir[tail_start:]
        return float(20.0 * np.log10(np.sqrt(np.mean(tail ** 2)) + 1e-12))
