"""Mathematical utilities for audio DSP and acoustic analysis."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy import signal, interpolate
from typing import Tuple, Optional


class MathUtils:
    """Static math utilities for DSP and acoustic computations."""

    SAMPLE_RATES = [44100, 48000, 96000, 192000]

    @staticmethod
    def db_to_linear(db: NDArray[np.float64]) -> NDArray[np.float64]:
        return 10.0 ** (np.asarray(db, dtype=np.float64) / 20.0)

    @staticmethod
    def linear_to_db(linear: NDArray[np.float64], floor: float = 1e-12) -> NDArray[np.float64]:
        linear = np.maximum(np.abs(np.asarray(linear, dtype=np.float64)), floor)
        return 20.0 * np.log10(linear)

    @staticmethod
    def magnitude_to_db(magnitude: NDArray[np.float64], floor: float = 1e-12) -> NDArray[np.float64]:
        return MathUtils.linear_to_db(magnitude, floor)

    @staticmethod
    def phase_unwrap(phase: NDArray[np.float64]) -> NDArray[np.float64]:
        return np.unwrap(phase)

    @staticmethod
    def phase_to_samples(phase_rad: NDArray[np.float64], freqs: NDArray[np.float64]) -> NDArray[np.float64]:
        safe_freqs = np.where(np.abs(freqs) < 1e-12, 1e-12, freqs)
        return -phase_rad / (2.0 * np.pi * safe_freqs)

    @staticmethod
    def group_delay(freq_response: NDArray[np.complex128], freqs: NDArray[np.float64]) -> NDArray[np.float64]:
        phase = np.angle(freq_response)
        unwrapped = np.unwrap(phase)
        return -np.gradient(unwrapped) / np.gradient(2.0 * np.pi * freqs)

    @staticmethod
    def next_power_of_two(n: int) -> int:
        return int(2 ** np.ceil(np.log2(n)))

    @staticmethod
    def pad_to_power_of_two(data: NDArray[np.float64]) -> NDArray[np.float64]:
        n = MathUtils.next_power_of_two(len(data))
        return np.pad(data, (0, n - len(data)), mode="constant")

    @staticmethod
    def smooth_frequency_response(
        magnitude_db: NDArray[np.float64],
        freqs: NDArray[np.float64],
        octave_fraction: float = 3.0,
    ) -> NDArray[np.float64]:
        log_freqs = np.log10(np.maximum(freqs, 1.0))
        log_spacing = np.linspace(log_freqs[0], log_freqs[-1], len(magnitude_db) // 4)
        spacing = log_spacing[1] - log_spacing[0] if len(log_spacing) > 1 else 1.0
        sigma = spacing * octave_fraction
        smoothed = np.zeros_like(magnitude_db)
        for i in range(len(magnitude_db)):
            weights = np.exp(-0.5 * ((log_freqs - log_freqs[i]) / sigma) ** 2)
            weights /= np.sum(weights)
            smoothed[i] = np.sum(weights * magnitude_db)
        return smoothed

    @staticmethod
    def interpolate_ir(
        ir: NDArray[np.float64],
        original_rate: int,
        target_rate: int,
    ) -> NDArray[np.float64]:
        if original_rate == target_rate:
            return ir
        n_samples = int(len(ir) * target_rate / original_rate)
        return signal.resample(ir, n_samples)

    @staticmethod
    def cosine_window(n: int) -> NDArray[np.float64]:
        return np.hanning(n)

    @staticmethod
    def apply_window(data: NDArray[np.float64], window: Optional[str] = "hann") -> NDArray[np.float64]:
        if window is None:
            return data
        if window == "hann":
            w = np.hanning(len(data))
        elif window == "hamming":
            w = np.hamming(len(data))
        elif window == "blackman":
            w = np.blackman(len(data))
        elif window == "bartlett":
            w = np.bartlett(len(data))
        else:
            raise ValueError(f"Unknown window type: {window}")
        return data * w

    @staticmethod
    def find_nearest_frequency(freqs: NDArray[np.float64], target: float) -> Tuple[int, float]:
        idx = np.argmin(np.abs(freqs - target))
        return idx, freqs[idx]

    @staticmethod
    def compute_coherence(
        x: NDArray[np.float64],
        y: NDArray[np.float64],
        nperseg: Optional[int] = None,
        fs: float = 48000.0,
    ) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
        if nperseg is None:
            nperseg = min(2048, len(x) // 4)
        f, cxy = signal.coherence(x, y, fs=fs, nperseg=nperseg)
        return f, cxy

    @staticmethod
    def ms_to_samples(ms: float, sample_rate: int) -> int:
        return int(ms * sample_rate / 1000.0)

    @staticmethod
    def samples_to_ms(samples: int, sample_rate: int) -> float:
        return samples * 1000.0 / sample_rate

    @staticmethod
    def freq_to_bark(freq_hz: NDArray[np.float64]) -> NDArray[np.float64]:
        f = np.asarray(freq_hz, dtype=np.float64)
        return 13.0 * np.arctan(0.00076 * f) + 3.5 * np.arctan((f / 7500.0) ** 2)

    @staticmethod
    def freq_to_mel(freq_hz: NDArray[np.float64]) -> NDArray[np.float64]:
        return 2595.0 * np.log10(1.0 + np.asarray(freq_hz, dtype=np.float64) / 700.0)
