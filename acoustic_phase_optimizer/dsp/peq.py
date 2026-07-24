from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class PEQBand:
    freq_hz: float = 1000.0
    gain_db: float = 0.0
    q: float = 0.707

    def copy(self) -> PEQBand:
        return PEQBand(freq_hz=self.freq_hz, gain_db=self.gain_db, q=self.q)


@dataclass
class PEQConfig:
    bands: List[PEQBand] = field(default_factory=lambda: [PEQBand() for _ in range(6)])

    def copy(self) -> PEQConfig:
        return PEQBands(bands=[b.copy() for b in self.bands])


def biquad_coefficients(peq: PEQBand, sample_rate: float) -> np.ndarray:
    w0 = 2.0 * np.pi * peq.freq_hz / sample_rate
    alpha = np.sin(w0) / (2.0 * peq.q)
    A = 10.0 ** (peq.gain_db / 40.0)

    cos_w0 = np.cos(w0)

    b0 = 1.0 + alpha * A
    b1 = -2.0 * cos_w0
    b2 = 1.0 - alpha * A
    a0 = 1.0 + alpha / A
    a1 = -2.0 * cos_w0
    a2 = 1.0 - alpha / A

    norm = 1.0 / a0
    return np.array([b0 * norm, b1 * norm, b2 * norm, a1 * norm, a2 * norm])


class PEQFilter:
    def __init__(self, config: PEQConfig, sample_rate: float = 48000.0):
        self.config = config
        self.sample_rate = sample_rate

    def apply(self, signal: np.ndarray) -> np.ndarray:
        out = signal.astype(np.float64)
        for band in self.config.bands:
            coeffs = biquad_coefficients(band, self.sample_rate)
            b0, b1, b2, a1, a2 = coeffs
            z1 = z2 = 0.0
            for i in range(len(out)):
                x = out[i]
                y = b0 * x + z1
                z1 = b1 * x - a1 * y + z2
                z2 = b2 * x - a2 * y
                out[i] = y
        return out


def peq_from_bands(bands: List[dict], sample_rate: float = 48000.0) -> PEQFilter:
    peq_bands = []
    for b in bands:
        peq_bands.append(PEQBand(
            freq_hz=b.get("freq", 1000.0),
            gain_db=b.get("gain", 0.0),
            q=b.get("q", 0.707),
        ))
    config = PEQConfig(bands=peq_bands)
    return PEQFilter(config, sample_rate)
