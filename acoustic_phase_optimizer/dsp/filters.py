"""Digital filter design for FIR, IIR, and equalization filters."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy import signal, fft
from typing import Dict, List, Optional, Tuple, Union
from acoustic_phase_optimizer.utils.logging import get_logger

logger = get_logger(__name__)


class FilterDesign:
    """Designs FIR and IIR filters for loudspeaker DSP."""

    def __init__(self, sample_rate: int = 48000):
        self.sample_rate = sample_rate

    def fir_lowpass(
        self,
        cutoff_hz: float,
        num_taps: int = 512,
        window: str = "hann",
    ) -> NDArray[np.float64]:
        nyquist = self.sample_rate / 2.0
        normalized_cutoff = cutoff_hz / nyquist

        taps = signal.firwin(
            num_taps,
            normalized_cutoff,
            window=window,
            pass_zero="lowpass",
        )
        return taps.astype(np.float64)

    def fir_highpass(
        self,
        cutoff_hz: float,
        num_taps: int = 511,
        window: str = "hann",
    ) -> NDArray[np.float64]:
        if num_taps % 2 == 0:
            num_taps = num_taps + 1
        nyquist = self.sample_rate / 2.0
        normalized_cutoff = cutoff_hz / nyquist

        taps = signal.firwin(
            num_taps,
            normalized_cutoff,
            window=window,
            pass_zero="highpass",
        )
        return taps.astype(np.float64)

    def fir_bandpass(
        self,
        low_hz: float,
        high_hz: float,
        num_taps: int = 512,
        window: str = "hann",
    ) -> NDArray[np.float64]:
        nyquist = self.sample_rate / 2.0
        taps = signal.firwin(
            num_taps,
            [low_hz / nyquist, high_hz / nyquist],
            window=window,
            pass_zero="bandpass",
        )
        return taps.astype(np.float64)

    def fir_bandstop(
        self,
        low_hz: float,
        high_hz: float,
        num_taps: int = 511,
        window: str = "hann",
    ) -> NDArray[np.float64]:
        if num_taps % 2 == 0:
            num_taps = num_taps + 1
        nyquist = self.sample_rate / 2.0
        taps = signal.firwin(
            num_taps,
            [low_hz / nyquist, high_hz / nyquist],
            window=window,
            pass_zero="bandstop",
        )
        return taps.astype(np.float64)

    def fir_from_target(
        self,
        target_magnitude: NDArray[np.float64],
        target_freqs: NDArray[np.float64],
        num_taps: int = 511,
    ) -> NDArray[np.float64]:
        if num_taps % 2 == 0:
            num_taps = num_taps + 1
        nyquist = self.sample_rate / 2.0
        normalized_freqs = np.clip(target_freqs / nyquist, 0.0, 1.0)

        mag = np.abs(target_magnitude).copy()
        if mag[-1] != 0.0 and (num_taps % 2 == 0):
            mag[-1] = 0.0

        taps = signal.firwin2(
            num_taps,
            normalized_freqs,
            mag,
            window="hann",
        )
        return taps.astype(np.float64)

    def fir_minimum_phase(self, taps: NDArray[np.float64]) -> NDArray[np.float64]:
        n_fft = len(taps)
        n_fft_pow2 = int(2 ** np.ceil(np.log2(n_fft)))

        H = fft.fft(taps, n=n_fft_pow2)
        magnitude = np.abs(H)

        log_mag = np.log(magnitude + 1e-12)
        cepstrum = fft.ifft(log_mag).real

        cepstrum[0] *= 1.0
        cepstrum[1:n_fft_pow2 // 2] *= 2.0
        cepstrum[n_fft_pow2 // 2 + 1:] = 0.0

        min_phase_fft = fft.fft(cepstrum)
        min_phase_fft = magnitude * np.exp(1j * min_phase_fft.real)
        min_phase_ir = fft.ifft(min_phase_fft).real[:n_fft]

        return min_phase_ir.astype(np.float64)

    def iir_parametric_eq(
        self,
        frequency_hz: float,
        gain_db: float,
        q: float,
        filter_type: str = "peaking",
    ) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
        fs = self.sample_rate
        f0 = frequency_hz
        A = 10.0 ** (gain_db / 40.0)
        w0 = 2.0 * np.pi * f0 / fs
        alpha = np.sin(w0) / (2.0 * q)

        if filter_type == "peaking":
            b = np.array([
                1.0 + alpha * A,
                -2.0 * np.cos(w0),
                1.0 - alpha * A,
            ])
            a = np.array([
                1.0 + alpha / A,
                -2.0 * np.cos(w0),
                1.0 - alpha / A,
            ])
        elif filter_type == "lowshelf":
            sqrt_a = np.sqrt(A)
            b = np.array([
                A * ((A + 1.0) - (A - 1.0) * np.cos(w0) + 2.0 * sqrt_a * alpha),
                2.0 * A * ((A - 1.0) - (A + 1.0) * np.cos(w0)),
                A * ((A + 1.0) - (A - 1.0) * np.cos(w0) - 2.0 * sqrt_a * alpha),
            ])
            a = np.array([
                (A + 1.0) + (A - 1.0) * np.cos(w0) + 2.0 * sqrt_a * alpha,
                -2.0 * ((A - 1.0) + (A + 1.0) * np.cos(w0)),
                (A + 1.0) + (A - 1.0) * np.cos(w0) - 2.0 * sqrt_a * alpha,
            ])
        elif filter_type == "highshelf":
            sqrt_a = np.sqrt(A)
            b = np.array([
                A * ((A + 1.0) + (A - 1.0) * np.cos(w0) + 2.0 * sqrt_a * alpha),
                -2.0 * A * ((A - 1.0) + (A + 1.0) * np.cos(w0)),
                A * ((A + 1.0) + (A - 1.0) * np.cos(w0) - 2.0 * sqrt_a * alpha),
            ])
            a = np.array([
                (A + 1.0) - (A - 1.0) * np.cos(w0) + 2.0 * sqrt_a * alpha,
                2.0 * ((A - 1.0) - (A + 1.0) * np.cos(w0)),
                (A + 1.0) - (A - 1.0) * np.cos(w0) - 2.0 * sqrt_a * alpha,
            ])
        else:
            raise ValueError(f"Unknown filter type: {filter_type}")

        return (b.astype(np.float64), a.astype(np.float64))

    def iir_butterworth(
        self,
        order: int,
        cutoff_hz: float,
        filter_type: str = "lowpass",
    ) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
        nyquist = self.sample_rate / 2.0
        normalized = cutoff_hz / nyquist

        sos = signal.butter(order, normalized, btype=filter_type, output="sos")
        return self._sos_to_ba(sos)

    def iir_linkwitz_riley(
        self,
        order: int,
        crossover_hz: float,
        filter_type: str = "lowpass",
    ) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
        nyquist = self.sample_rate / 2.0
        normalized = crossover_hz / nyquist

        sos = signal.butter(order, normalized, btype=filter_type, output="sos")
        sos = signal.sosfreqz(sos)[0]

        b, a = signal.butter(order, normalized, btype=filter_type)
        return (b.astype(np.float64), a.astype(np.float64))

    def apply_fir(
        self,
        signal_data: NDArray[np.float64],
        fir_taps: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        return signal.convolve(signal_data, fir_taps, mode="same").astype(np.float64)

    def apply_iir(
        self,
        signal_data: NDArray[np.float64],
        b: NDArray[np.float64],
        a: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        return signal.lfilter(b, a, signal_data).astype(np.float64)

    def apply_sos(
        self,
        signal_data: NDArray[np.float64],
        sos: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        return signal.sosfilt(sos, signal_data).astype(np.float64)

    @staticmethod
    def _sos_to_ba(
        sos: NDArray[np.float64],
    ) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
        b = np.array([1.0])
        a = np.array([1.0])
        for section in sos:
            b = np.polymul(b, section[:3])
            a = np.polymul(a, section[3:])
        return (b.astype(np.float64), a.astype(np.float64))

    def frequency_response(
        self,
        b: NDArray[np.float64],
        a: NDArray[np.float64],
        n_points: int = 512,
    ) -> Tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
        w, h = signal.freqz(b, a, worN=n_points, fs=self.sample_rate)
        return w, np.abs(h), np.angle(h)

    def fir_frequency_response(
        self,
        taps: NDArray[np.float64],
        n_points: int = 512,
    ) -> Tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
        w, h = signal.freqz(taps, worN=n_points, fs=self.sample_rate)
        return w, np.abs(h), np.angle(h)

    def design_corrective_eq(
        self,
        measured_magnitude: NDArray[np.float64],
        freqs: NDArray[np.float64],
        target_magnitude: Optional[NDArray[np.float64]] = None,
        num_bands: int = 31,
    ) -> List[Dict[str, float]]:
        if target_magnitude is None:
            target_magnitude = np.ones_like(measured_magnitude)
            mean_level = np.mean(measured_magnitude[measured_magnitude > -100])
            target_magnitude *= mean_level

        correction_db = target_magnitude - measured_magnitude
        correction_db = np.clip(correction_db, -12.0, 12.0)

        bands = np.logspace(np.log10(20), np.log10(20000), num_bands)
        eq_settings = []

        for center_freq in bands:
            if center_freq > self.sample_rate / 2.0:
                break

            idx = np.argmin(np.abs(freqs - center_freq))
            if idx < len(correction_db):
                gain = correction_db[idx]
                if abs(gain) > 0.5:
                    q = 4.0
                    eq_settings.append({
                        "frequency": float(center_freq),
                        "gain_db": float(gain),
                        "q": q,
                        "type": "peaking",
                    })

        return eq_settings
