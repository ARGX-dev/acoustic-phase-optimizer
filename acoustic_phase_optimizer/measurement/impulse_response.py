"""Impulse response extraction and analysis from measured signals."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy import signal, fft
from typing import Optional, Tuple
from acoustic_phase_optimizer.utils.logging import get_logger

logger = get_logger(__name__)


class ImpulseResponse:
    """Extracts and analyzes impulse responses from room measurements."""

    def __init__(self, sample_rate: int = 48000):
        self.sample_rate = sample_rate

    def extract_from_sweep(
        self,
        recorded: NDArray[np.float64],
        sweep: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        if recorded.ndim > 1:
            recorded = recorded[:, 0]
        if sweep.ndim > 1:
            sweep = sweep[:, 0]

        n_fft = len(sweep) + len(recorded) - 1
        n_fft = int(2 ** np.ceil(np.log2(n_fft)))

        sweep_fft = fft.fft(sweep, n=n_fft)
        recorded_fft = fft.fft(recorded, n=n_fft)

        epsilon = np.max(np.abs(sweep_fft)) * 1e-12
        inverse_filter = np.conj(sweep_fft) / (np.abs(sweep_fft) ** 2 + epsilon)

        ir_fft = recorded_fft * inverse_filter
        ir = fft.ifft(ir_fft, n=n_fft).real

        ir = ir[:len(recorded)]

        return ir.astype(np.float64)

    def extract_from_mls(
        self,
        recorded: NDArray[np.float64],
        mls_signal: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        if recorded.ndim > 1:
            recorded = recorded[:, 0]
        if mls_signal.ndim > 1:
            mls_signal = mls_signal[:, 0]

        n = len(mls_signal)
        if len(recorded) < n:
            recorded = np.pad(recorded, (0, n - len(recorded)))

        recorded = recorded[:n]

        mls_fft = fft.fft(mls_signal)
        recorded_fft = fft.fft(recorded)

        epsilon = np.max(np.abs(mls_fft)) * 1e-12
        ir_fft = recorded_fft * np.conj(mls_fft) / (np.abs(mls_fft) ** 2 + epsilon)
        ir = fft.ifft(ir_fft).real

        return ir.astype(np.float64)

    def extract_phase(
        self,
        ir: NDArray[np.float64],
    ) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
        n_fft = len(ir)
        freq_response = fft.fft(ir, n=n_fft)
        freqs = fft.fftfreq(n_fft, 1.0 / self.sample_rate)

        positive = freqs >= 0
        freqs = freqs[positive]
        magnitude = np.abs(freq_response[positive])
        phase = np.angle(freq_response[positive])

        return freqs, phase

    def extract_magnitude(
        self,
        ir: NDArray[np.float64],
    ) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
        n_fft = len(ir)
        freq_response = fft.fft(ir, n=n_fft)
        freqs = fft.fftfreq(n_fft, 1.0 / self.sample_rate)

        positive = freqs >= 0
        magnitude_db = 20.0 * np.log10(np.abs(freq_response[positive]) + 1e-12)

        return freqs[positive], magnitude_db

    def compute_group_delay(
        self,
        ir: NDArray[np.float64],
    ) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
        freqs, phase = self.extract_phase(ir)
        unwrapped = np.unwrap(phase)
        group_delay = -np.gradient(unwrapped, freqs[1] - freqs[0])
        return freqs, group_delay

    def window_ir(
        self,
        ir: NDArray[np.float64],
        direct_start: int = 0,
        direct_end: Optional[int] = None,
        window_type: str = "hann",
    ) -> NDArray[np.float64]:
        result = np.zeros_like(ir)
        if direct_end is None:
            direct_end = len(ir)

        direct_end = min(direct_end, len(ir))

        if direct_end <= direct_start:
            return result

        segment = ir[direct_start:direct_end].copy()

        if window_type == "hann":
            w = np.hanning(len(segment))
        elif window_type == "hamming":
            w = np.hamming(len(segment))
        elif window_type == "blackman":
            w = np.blackman(len(segment))
        elif window_type == "rectangular":
            w = np.ones(len(segment))
        else:
            w = np.hanning(len(segment))

        result[direct_start:direct_end] = segment * w
        return result.astype(np.float64)

    def minimum_phase(self, ir: NDArray[np.float64]) -> NDArray[np.float64]:
        n_fft = len(ir)
        n_fft_pow2 = int(2 ** np.ceil(np.log2(n_fft)))

        freq_response = fft.fft(ir, n=n_fft_pow2)

        magnitude = np.abs(freq_response)
        log_magnitude = np.log(magnitude + 1e-12)

        cepstrum = fft.ifft(log_magnitude).real

        cepstrum[1:n_fft_pow2 // 2] *= 2.0
        cepstrum[n_fft_pow2 // 2 + 1:] = 0.0

        min_phase_fft = fft.fft(cepstrum)
        min_phase_fft = magnitude * np.exp(1j * min_phase_fft.real)

        min_phase_ir = fft.ifft(min_phase_fft).real[:n_fft]

        return min_phase_ir.astype(np.float64)
