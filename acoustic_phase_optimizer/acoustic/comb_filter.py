"""Comb filtering detection and phase cancellation analysis."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy import signal, fft
from typing import List, Optional, Tuple
from dataclasses import dataclass
from acoustic_phase_optimizer.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class CombFilterResult:
    detected: bool
    notch_frequencies: NDArray[np.float64]
    peak_frequencies: NDArray[np.float64]
    depth_db: float
    fundamental_freq: float
    delay_ms: float
    severity: float


@dataclass
class PhaseCancellationZone:
    position: NDArray[np.float64]
    frequency: float
    depth_db: float
    speakers_involved: List[str]


class CombFilterDetector:
    """Detects comb filtering effects from speaker interaction."""

    def __init__(self, sample_rate: int = 48000):
        self.sample_rate = sample_rate

    def detect_comb_filtering(
        self,
        magnitude_db: NDArray[np.float64],
        freqs: NDArray[np.float64],
        threshold_db: float = 3.0,
    ) -> CombFilterResult:
        smooth = self._smooth_spectrum(magnitude_db, freqs)
        difference = magnitude_db - smooth

        notches = self._find_peaks(-difference, freqs)
        peaks = self._find_peaks(difference, freqs)

        notch_freqs = np.array([f for _, f in notches])
        peak_freqs = np.array([f for _, f in peaks])

        depth = float(np.max(difference) - np.min(difference)) if len(difference) > 0 else 0.0

        fundamental = self._estimate_fundamental(notch_freqs)

        delay_ms = 0.0
        if fundamental > 0:
            delay_ms = 1000.0 / (2.0 * fundamental)

        severity = self._compute_severity(depth, len(notch_freqs))

        return CombFilterResult(
            detected=bool(severity > threshold_db),
            notch_frequencies=notch_freqs,
            peak_frequencies=peak_freqs,
            depth_db=float(depth),
            fundamental_freq=float(fundamental),
            delay_ms=float(delay_ms),
            severity=float(severity),
        )

    def detect_phase_cancellation(
        self,
        ir1: NDArray[np.float64],
        ir2: NDArray[np.float64],
        sample_rate: int = 48000,
    ) -> Tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
        n_fft = len(ir1) + len(ir2) - 1
        n_fft = int(2 ** np.ceil(np.log2(n_fft)))

        fft1 = fft.fft(ir1, n=n_fft)
        fft2 = fft.fft(ir2, n=n_fft)

        sum_fft = fft1 + fft2
        sum_mag = np.abs(sum_fft)
        individual_mag = np.abs(fft1) + np.abs(fft2)

        freqs = fft.fftfreq(n_fft, 1.0 / sample_rate)
        positive = freqs >= 0

        cancellation_db = 20.0 * np.log10(
            sum_mag[positive] / (individual_mag[positive] + 1e-12) + 1e-12
        )
        phase_diff = np.angle(fft1[positive]) - np.angle(fft2[positive])

        return freqs[positive], cancellation_db, phase_diff

    def map_cancellation_zones(
        self,
        speaker1_pos: NDArray[np.float64],
        speaker2_pos: NDArray[np.float64],
        frequency: float,
        room_bounds: Tuple[float, float, float, float],
        resolution: int = 50,
    ) -> Tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
        x_min, x_max, y_min, y_max = room_bounds
        x = np.linspace(x_min, x_max, resolution)
        y = np.linspace(y_min, y_max, resolution)
        X, Y = np.meshgrid(x, y)

        speed = 343.0
        wavelength = speed / frequency

        d1 = np.sqrt((X - speaker1_pos[0]) ** 2 + (Y - speaker1_pos[1]) ** 2)
        d2 = np.sqrt((X - speaker2_pos[0]) ** 2 + (Y - speaker2_pos[1]) ** 2)

        path_diff = d1 - d2
        phase_diff = 2.0 * np.pi * path_diff / wavelength
        cancellation = np.cos(phase_diff)

        return X, Y, cancellation

    def _smooth_spectrum(
        self,
        magnitude_db: NDArray[np.float64],
        freqs: NDArray[np.float64],
        octave_fraction: float = 6.0,
    ) -> NDArray[np.float64]:
        log_freqs = np.log10(np.maximum(freqs, 1.0))
        sigma = (log_freqs[-1] - log_freqs[0]) / (len(freqs) * octave_fraction)
        smoothed = np.zeros_like(magnitude_db)
        for i in range(len(magnitude_db)):
            weights = np.exp(-0.5 * ((log_freqs - log_freqs[i]) / sigma) ** 2)
            weights /= np.sum(weights) + 1e-12
            smoothed[i] = np.sum(weights * magnitude_db)
        return smoothed

    def _find_peaks(
        self,
        data: NDArray[np.float64],
        freqs: NDArray[np.float64],
        min_distance: int = 5,
    ) -> List[Tuple[float, float]]:
        peaks = []
        for i in range(min_distance, len(data) - min_distance):
            if data[i] > data[i - min_distance] and data[i] > data[i + min_distance]:
                peaks.append((data[i], freqs[i]))
        peaks.sort(reverse=True)
        return peaks

    def _estimate_fundamental(self, freqs: NDArray[np.float64]) -> float:
        if len(freqs) < 2:
            return 0.0
        spacings = np.diff(freqs)
        if len(spacings) == 0:
            return 0.0
        median_spacing = np.median(spacings)
        return float(median_spacing) if median_spacing > 0 else 0.0

    @staticmethod
    def _compute_severity(depth_db: float, num_notches: int) -> float:
        return depth_db * np.log1p(num_notches)
