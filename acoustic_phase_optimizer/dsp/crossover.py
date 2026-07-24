"""Crossover network designer for multi-way loudspeaker systems."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy import signal
from typing import Dict, List, Optional, Tuple
from acoustic_phase_optimizer.utils.logging import get_logger

logger = get_logger(__name__)


class CrossoverDesigner:
    """Designs crossover filters for multi-way speaker systems."""

    def __init__(self, sample_rate: int = 48000):
        self.sample_rate = sample_rate
        from acoustic_phase_optimizer.dsp.filters import FilterDesign
        self.filter_design = FilterDesign(sample_rate)

    def linkwitz_riley(
        self,
        crossover_freq: float,
        order: int = 4,
        filter_type: str = "lowpass",
    ) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
        nyquist = self.sample_rate / 2.0
        normalized = crossover_freq / nyquist

        b, a = signal.butter(order, normalized, btype=filter_type)
        return (b.astype(np.float64), a.astype(np.float64))

    def butterworth(
        self,
        crossover_freq: float,
        order: int = 4,
        filter_type: str = "lowpass",
    ) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
        return self.linkwitz_riley(crossover_freq, order, filter_type)

    def bessel(
        self,
        crossover_freq: float,
        order: int = 4,
        filter_type: str = "lowpass",
    ) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
        nyquist = self.sample_rate / 2.0
        normalized = crossover_freq / nyquist

        b, a = signal.bessel(order, normalized, btype=filter_type, norm="delay")
        return (b.astype(np.float64), a.astype(np.float64))

    def fir_crossover(
        self,
        crossover_freq: float,
        num_taps: int = 512,
        filter_type: str = "lowpass",
    ) -> NDArray[np.float64]:
        nyquist = self.sample_rate / 2.0
        normalized = crossover_freq / nyquist

        if filter_type == "lowpass":
            taps = signal.firwin(num_taps, normalized, window="hann", pass_zero="lowpass")
        elif filter_type == "highpass":
            taps = signal.firwin(num_taps, normalized, window="hann", pass_zero="highpass")
        else:
            raise ValueError(f"Unknown filter type: {filter_type}")

        return taps.astype(np.float64)

    def design_2way(
        self,
        crossover_freq: float,
        slope_db_per_octave: float = 24.0,
        filter_type: str = "linkwitz_riley",
    ) -> dict:
        order = int(slope_db_per_octave / 6.0)

        if filter_type == "linkwitz_riley":
            low_b, low_a = self.linkwitz_riley(crossover_freq, order, "lowpass")
            high_b, high_a = self.linkwitz_riley(crossover_freq, order, "highpass")
        elif filter_type == "butterworth":
            low_b, low_a = self.butterworth(crossover_freq, order, "lowpass")
            high_b, high_a = self.butterworth(crossover_freq, order, "highpass")
        elif filter_type == "bessel":
            low_b, low_a = self.bessel(crossover_freq, order, "lowpass")
            high_b, high_a = self.bessel(crossover_freq, order, "highpass")
        else:
            raise ValueError(f"Unknown filter type: {filter_type}")

        return {
            "type": "2_way",
            "crossover_freq": crossover_freq,
            "slope_db_per_octave": slope_db_per_octave,
            "filter_type": filter_type,
            "lowpass_b": low_b,
            "lowpass_a": low_a,
            "highpass_b": high_b,
            "highpass_a": high_a,
        }

    def design_3way(
        self,
        crossover_freq_low: float,
        crossover_freq_high: float,
        slope_db_per_octave: float = 24.0,
    ) -> dict:
        order = int(slope_db_per_octave / 6.0)

        low_b, low_a = self.linkwitz_riley(crossover_freq_low, order, "lowpass")
        mid_bp_b, mid_bp_a = self.linkwitz_riley(crossover_freq_high, order, "highpass")
        mid_lp_b, mid_lp_a = self.linkwitz_riley(crossover_freq_low, order, "lowpass")
        high_b, high_a = self.linkwitz_riley(crossover_freq_high, order, "highpass")

        return {
            "type": "3_way",
            "crossover_freq_low": crossover_freq_low,
            "crossover_freq_high": crossover_freq_high,
            "slope_db_per_octave": slope_db_per_octave,
            "lowpass_b": low_b,
            "lowpass_a": low_a,
            "mid_highpass_b": mid_bp_b,
            "mid_highpass_a": mid_bp_a,
            "mid_lowpass_b": mid_lp_b,
            "mid_lowpass_a": mid_lp_a,
            "highpass_b": high_b,
            "highpass_a": high_a,
        }

    def subwoofer_crossover(
        self,
        crossover_freq: float,
        slope_db_per_octave: float = 24.0,
    ) -> dict:
        return self.design_2way(crossover_freq, slope_db_per_octave)

    def optimize_crossover_freq(
        self,
        woofer_response: NDArray[np.float64],
        tweeter_response: NDArray[np.float64],
        freqs: NDArray[np.float64],
        search_range: Tuple[float, float] = (500.0, 4000.0),
    ) -> float:
        low_idx = np.searchsorted(freqs, search_range[0])
        high_idx = np.searchsorted(freqs, search_range[1])

        search_freqs = freqs[low_idx:high_idx]
        woofer_segment = woofer_response[low_idx:high_idx]
        tweeter_segment = tweeter_response[low_idx:high_idx]

        combined = woofer_segment + tweeter_segment
        smoothness = -np.std(np.diff(combined))

        best_idx = np.argmax(smoothness)
        return float(search_freqs[best_idx])
