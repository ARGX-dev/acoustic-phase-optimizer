"""RT60 and related reverberation time estimation."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy import signal
from typing import Optional, Tuple
from acoustic_phase_optimizer.utils.logging import get_logger

logger = get_logger(__name__)


class RT60Estimator:
    """Estimates reverberation times (RT60, EDT, etc.) from impulse responses."""

    def __init__(self, sample_rate: int = 48000):
        self.sample_rate = sample_rate
        self.freqs: Optional[NDArray[np.float64]] = None

    def estimate_rt60(
        self,
        ir: NDArray[np.float64],
        bands: Optional[list[float]] = None,
    ) -> dict:
        if bands is None:
            bands = [63, 125, 250, 500, 1000, 2000, 4000, 8000]

        rt60_values = {}
        for band in bands:
            filtered = self._bandpass_filter(ir, band)
            rt60 = self._schroeder_rt60(filtered)
            rt60_values[f"rt60_{band}hz"] = rt60

        broadband_rt60 = self._schroeder_rt60(ir)
        rt60_values["rt60_broadband"] = broadband_rt60

        return rt60_values

    def estimate_edt(self, ir: NDArray[np.float64]) -> float:
        decay_db = self._energy_decay_curve(ir)
        return self._slope_to_rt60(decay_db, 0.0, -10.0)

    def clarity(self, ir: NDArray[np.float64]) -> dict:
        squared = ir ** 2
        total_energy = np.sum(squared)
        if total_energy <= 0:
            return {"c50": 0.0, "c80": 0.0}

        early_50_samples = int(0.050 * self.sample_rate)
        early_80_samples = int(0.080 * self.sample_rate)

        early_50 = np.sum(squared[:early_50_samples])
        late_50 = total_energy - early_50
        early_80 = np.sum(squared[:early_80_samples])
        late_80 = total_energy - early_80

        c50 = 10.0 * np.log10(early_50 / (late_50 + 1e-12))
        c80 = 10.0 * np.log10(early_80 / (late_80 + 1e-12))

        return {"c50": c50, "c80": c80}

    def definition(self, ir: NDArray[np.float64]) -> dict:
        squared = ir ** 2
        total_energy = np.sum(squared)
        if total_energy <= 0:
            return {"d50": 0.0}

        early_50_samples = int(0.050 * self.sample_rate)
        early_energy = np.sum(squared[:early_50_samples])
        d50 = early_energy / (total_energy + 1e-12)

        return {"d50": d50}

    def center_time(self, ir: NDArray[np.float64]) -> float:
        squared = ir ** 2
        total_energy = np.sum(squared)
        if total_energy <= 0:
            return 0.0

        times = np.arange(len(ir)) / self.sample_rate
        ts = np.sum(times * squared) / total_energy
        return float(ts)

    def _schroeder_rt60(self, ir: NDArray[np.float64]) -> float:
        decay_db = self._energy_decay_curve(ir)
        return self._slope_to_rt60(decay_db, -5.0, -35.0)

    def _energy_decay_curve(self, ir: NDArray[np.float64]) -> NDArray[np.float64]:
        squared = ir ** 2
        decay = np.cumsum(squared[::-1])[::-1]
        decay_db = 10.0 * np.log10(decay + 1e-12)
        decay_db -= decay_db[0]
        return decay_db

    def _slope_to_rt60(
        self,
        decay_db: NDArray[np.float64],
        start_db: float = -5.0,
        end_db: float = -35.0,
    ) -> float:
        db_range = end_db - start_db
        if db_range >= 0:
            return 0.0

        start_idx = np.argmin(np.abs(decay_db - start_db))
        end_idx = np.argmin(np.abs(decay_db - end_db))

        if end_idx <= start_idx or start_idx >= len(decay_db) - 1:
            return 0.0

        x = np.arange(start_idx, end_idx)
        y = decay_db[start_idx:end_idx]

        if len(x) < 2:
            return 0.0

        A = np.vstack([x, np.ones_like(x)]).T
        try:
            slope, _ = np.linalg.lstsq(A, y, rcond=None)[0]
        except np.linalg.LinAlgError:
            return 0.0

        if slope >= 0:
            return 0.0

        rt60 = -60.0 / slope
        return float(np.clip(rt60, 0.0, 20.0))

    def _bandpass_filter(
        self,
        ir: NDArray[np.float64],
        center_freq: float,
    ) -> NDArray[np.float64]:
        nyquist = self.sample_rate / 2.0
        if center_freq >= nyquist:
            return ir

        q = np.sqrt(2.0)
        f_low = center_freq / np.sqrt(2.0)
        f_high = center_freq * np.sqrt(2.0)

        sos = signal.butter(
            4,
            [f_low / nyquist, f_high / nyquist],
            btype="band",
            output="sos",
        )
        filtered = signal.sosfilt(sos, ir)
        return filtered
