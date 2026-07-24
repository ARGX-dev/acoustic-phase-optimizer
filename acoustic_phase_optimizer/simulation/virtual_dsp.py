"""Virtual DSP processor for simulation without hardware."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from typing import Dict, List, Optional, Tuple
from scipy import signal
from acoustic_phase_optimizer.dsp.interface import DSPInterface
from acoustic_phase_optimizer.dsp.filters import FilterDesign
from acoustic_phase_optimizer.utils.logging import get_logger

logger = get_logger(__name__)


class VirtualDSP(DSPInterface):
    """Software DSP processor for simulation.

    Applies all DSP processing (delays, gains, filters, EQ) in software,
    allowing the optimization engine to compute results without hardware.
    """

    def __init__(self, sample_rate: int = 48000):
        super().__init__("127.0.0.1", 0, sample_rate)
        self.filter_design = FilterDesign(sample_rate)
        self._config: Dict[int, dict] = {}

    def connect(self) -> bool:
        self._connected = True
        return True

    def disconnect(self) -> bool:
        self._connected = False
        return True

    def set_delay(self, channel: int, delay_ms: float) -> bool:
        self._ensure_channel(channel)
        self._config[channel]["delay_ms"] = max(0.0, min(delay_ms, 2000.0))
        return True

    def set_gain(self, channel: int, gain_db: float) -> bool:
        self._ensure_channel(channel)
        self._config[channel]["gain_db"] = max(-100.0, min(gain_db, 20.0))
        return True

    def set_polarity(self, channel: int, inverted: bool) -> bool:
        self._ensure_channel(channel)
        self._config[channel]["polarity"] = -1 if inverted else 1
        return True

    def set_crossover(
        self,
        channel: int,
        frequency_hz: float,
        slope_db_per_octave: float = 24.0,
    ) -> bool:
        self._ensure_channel(channel)
        self._config[channel]["crossover_freq"] = frequency_hz
        self._config[channel]["crossover_slope"] = slope_db_per_octave

        nyquist = self.sample_rate / 2.0
        normalized = frequency_hz / nyquist
        order = int(slope_db_per_octave / 6.0)

        low_b, low_a = signal.butter(order, normalized, btype="lowpass")
        high_b, high_a = signal.butter(order, normalized, btype="highpass")

        self._config[channel]["lowpass_b"] = low_b
        self._config[channel]["lowpass_a"] = low_a
        self._config[channel]["highpass_b"] = high_b
        self._config[channel]["highpass_a"] = high_a
        return True

    def set_eq_parametric(
        self,
        channel: int,
        frequency_hz: float,
        gain_db: float,
        q: float,
    ) -> bool:
        self._ensure_channel(channel)
        b, a = self.filter_design.iir_parametric_eq(frequency_hz, gain_db, q, "peaking")
        if "eq_filters" not in self._config[channel]:
            self._config[channel]["eq_filters"] = []
        self._config[channel]["eq_filters"].append({
            "freq": frequency_hz,
            "gain_db": gain_db,
            "q": q,
            "b": b,
            "a": a,
        })
        return True

    def set_fir_coefficients(
        self,
        channel: int,
        coefficients: NDArray[np.float64],
    ) -> bool:
        self._ensure_channel(channel)
        self._config[channel]["fir"] = coefficients.copy()
        return True

    def get_delay(self, channel: int) -> Optional[float]:
        if channel not in self._config:
            return None
        return self._config[channel].get("delay_ms", 0.0)

    def get_gain(self, channel: int) -> Optional[float]:
        if channel not in self._config:
            return None
        return self._config[channel].get("gain_db", 0.0)

    def get_polarity(self, channel: int) -> Optional[bool]:
        if channel not in self._config:
            return None
        return self._config[channel].get("polarity", 1) == -1

    def get_crossover(self, channel: int) -> Optional[Tuple[float, float]]:
        if channel not in self._config:
            return None
        freq = self._config[channel].get("crossover_freq")
        slope = self._config[channel].get("crossover_slope", 24.0)
        if freq is None:
            return None
        return (freq, slope)

    def mute_channel(self, channel: int, muted: bool) -> bool:
        self._ensure_channel(channel)
        self._config[channel]["muted"] = muted
        return True

    def apply_configuration(self, config: dict) -> bool:
        for ch_str, settings in config.items():
            ch = int(ch_str)
            self._ensure_channel(ch)
            for key, value in settings.items():
                self._config[ch][key] = value
        return True

    def read_configuration(self) -> dict:
        result = {}
        for ch, config in self._config.items():
            ch_dict = {}
            for key, value in config.items():
                if isinstance(value, np.ndarray):
                    ch_dict[key] = value.tolist()
                elif isinstance(value, list):
                    ch_dict[key] = [
                        {k: (v.tolist() if isinstance(v, np.ndarray) else v)
                         for k, v in item.items()}
                        if isinstance(item, dict) else item
                        for item in value
                    ]
                else:
                    ch_dict[key] = value
            result[str(ch)] = ch_dict
        return result

    def reset_to_defaults(self) -> bool:
        self._config.clear()
        return True

    def process_signal(
        self,
        channel: int,
        signal_data: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        if channel not in self._config:
            return signal_data

        config = self._config[channel]
        result = signal_data.copy()

        if config.get("muted", False):
            return np.zeros_like(result)

        gain_linear = 10.0 ** (config.get("gain_db", 0.0) / 20.0)
        result *= gain_linear

        polarity = config.get("polarity", 1)
        result *= polarity

        for eq in config.get("eq_filters", []):
            result = signal.lfilter(eq["b"], eq["a"], result)

        fir = config.get("fir")
        if fir is not None and len(fir) > 0:
            result = signal.convolve(result, fir, mode="same")

        delay_ms = config.get("delay_ms", 0.0)
        if delay_ms > 0:
            delay_samples = int(delay_ms * self.sample_rate / 1000.0)
            if delay_samples > 0:
                result = np.roll(result, delay_samples)
                result[:delay_samples] = 0.0

        return result.astype(np.float64)

    def _ensure_channel(self, channel: int) -> None:
        if channel not in self._config:
            self._config[channel] = {
                "delay_ms": 0.0,
                "gain_db": 0.0,
                "polarity": 1,
                "crossover_freq": None,
                "crossover_slope": 24.0,
                "eq_filters": [],
                "fir": None,
                "muted": False,
            }
