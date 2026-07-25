from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import List

ISO_BANDS_31 = [
    20, 25, 31.5, 40, 50, 63, 80, 100, 125, 160, 200, 250, 315, 400, 500,
    630, 800, 1000, 1250, 1600, 2000, 2500, 3150, 4000, 5000, 6300, 8000,
    10000, 12500, 16000, 20000,
]


@dataclass
class GEQConfig:
    gains_db: List[float] = field(default_factory=lambda: [0.0] * 31)
    sample_rate: float = 48000.0


class GEQFilter:
    def __init__(self, config: GEQConfig):
        self.config = config

    def apply(self, signal: np.ndarray) -> np.ndarray:
        n = len(signal)
        spectrum = np.fft.rfft(signal)
        freqs = np.fft.rfftfreq(n, 1.0 / self.config.sample_rate)

        magnitude = np.abs(spectrum)
        phase = np.angle(spectrum)

        for i, (band_freq, gain_db) in enumerate(zip(ISO_BANDS_31, self.config.gains_db)):
            gain_linear = 10.0 ** (gain_db / 20.0)
            if gain_linear == 1.0:
                continue
            bw = band_freq * (2.0 ** (1.0 / 6.0) - 2.0 ** (-1.0 / 6.0))
            lo = max(band_freq - bw / 2, 0.0)
            hi = band_freq + bw / 2
            mask = (freqs >= lo) & (freqs < hi)
            magnitude[mask] *= gain_linear

        new_spec = magnitude * np.exp(1j * phase)
        filtered = np.fft.irfft(new_spec, n=n)
        return filtered.astype(np.float64)

    @staticmethod
    def magnitude_response(gains_db: List[float], freqs: np.ndarray) -> np.ndarray:
        result = np.ones(len(freqs), dtype=np.float64)
        for band_freq, gain_db in zip(ISO_BANDS_31, gains_db):
            gain_linear = 10.0 ** (gain_db / 20.0)
            if abs(gain_linear - 1.0) < 1e-9:
                continue
            bw = band_freq * (2.0 ** (1.0 / 6.0) - 2.0 ** (-1.0 / 6.0))
            lo = max(band_freq - bw / 2, 0.0)
            hi = band_freq + bw / 2
            mask = (freqs >= lo) & (freqs < hi)
            result[mask] *= gain_linear
        return result
