"""Signal generation for acoustic measurement: log sweeps and MLS sequences."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from typing import Tuple, Optional
from acoustic_phase_optimizer.utils.logging import get_logger

logger = get_logger(__name__)


class SignalGenerator:
    """Generates test signals for room acoustic measurement."""

    def __init__(self, sample_rate: int = 48000):
        self.sample_rate = sample_rate

    def log_sweep(
        self,
        duration: float = 5.0,
        start_freq: float = 20.0,
        end_freq: float = 20000.0,
        fade_samples: int = 100,
    ) -> NDArray[np.float64]:
        n_samples = int(duration * self.sample_rate)
        t = np.linspace(0, duration, n_samples, endpoint=False)
        rate = np.log(end_freq / start_freq) / duration
        signal = np.sin(2.0 * np.pi * start_freq * (np.exp(rate * t) - 1.0) / rate)

        if fade_samples > 0 and n_samples > 2 * fade_samples:
            fade_in = np.linspace(0, 1, fade_samples)
            fade_out = np.linspace(1, 0, fade_samples)
            signal[:fade_samples] *= fade_in
            signal[-fade_samples:] *= fade_out

        return signal.astype(np.float64)

    def inverse_sweep(
        self,
        duration: float = 5.0,
        start_freq: float = 20.0,
        end_freq: float = 20000.0,
    ) -> NDArray[np.float64]:
        n_samples = int(duration * self.sample_rate)
        t = np.linspace(0, duration, n_samples, endpoint=False)
        rate = np.log(end_freq / start_freq) / duration

        envelope = np.exp(-rate * t)
        signal = envelope * np.sin(
            2.0 * np.pi * start_freq * (np.exp(rate * t) - 1.0) / rate
        )

        amplitude = np.max(np.abs(signal))
        if amplitude > 0:
            signal /= amplitude

        return signal.astype(np.float64)

    def mls_sequence(
        self,
        order: int = 15,
        repetitions: int = 2,
    ) -> NDArray[np.float64]:
        n = 2 ** order - 1
        reg = np.ones(order, dtype=np.int64)
        seq = np.zeros(n, dtype=np.int64)

        taps = self._mls_taps(order)
        for i in range(n):
            seq[i] = reg[-1]
            feedback = 0
            for tap in taps:
                feedback ^= reg[tap - 1]
            reg = np.roll(reg, 1)
            reg[0] = feedback

        signal = (2.0 * seq - 1.0).astype(np.float64)
        signal = np.tile(signal, repetitions)
        return signal

    @staticmethod
    def _mls_taps(order: int) -> list[int]:
        taps_map = {
            2: [1, 2],
            3: [1, 3],
            4: [1, 4],
            5: [2, 5],
            6: [1, 6],
            7: [3, 7],
            8: [2, 3, 4, 8],
            9: [4, 9],
            10: [3, 10],
            11: [2, 11],
            12: [6, 8, 11, 12],
            13: [4, 8, 9, 13],
            14: [4, 8, 11, 14],
            15: [1, 15],
            16: [4, 13, 15, 16],
            17: [3, 17],
            18: [5, 18],
            19: [5, 9, 14, 19],
            20: [3, 20],
        }
        if order not in taps_map:
            raise ValueError(f"MLS order {order} not supported. Use 2-20.")
        return taps_map[order]

    def generate(
        self,
        sweep_type: str = "log",
        duration: float = 5.0,
        start_freq: float = 20.0,
        end_freq: float = 20000.0,
        mls_order: int = 15,
        averaging: int = 1,
    ) -> NDArray[np.float64]:
        if sweep_type == "log":
            signal = self.log_sweep(duration, start_freq, end_freq)
        elif sweep_type == "mls":
            signal = self.mls_sequence(order=mls_order)
        else:
            raise ValueError(f"Unsupported sweep type: {sweep_type}")

        if averaging > 1:
            signal = np.tile(signal, averaging)

        return signal.astype(np.float64)

    def generate_test_signal(
        self,
        frequency: float = 1000.0,
        duration: float = 1.0,
    ) -> NDArray[np.float64]:
        n_samples = int(duration * self.sample_rate)
        t = np.linspace(0, duration, n_samples, endpoint=False)
        return np.sin(2.0 * np.pi * frequency * t).astype(np.float64)
