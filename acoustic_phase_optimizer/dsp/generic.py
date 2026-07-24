"""Generic DSP interface for systems without dedicated control APIs."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from typing import Dict, List, Optional, Tuple
from acoustic_phase_optimizer.dsp.interface import DSPInterface
from acoustic_phase_optimizer.utils.logging import get_logger

logger = get_logger(__name__)


class GenericDSP(DSPInterface):
    """Generic DSP interface for processors without native control.

    Stores configuration locally for manual transfer or future control protocol.
    """

    def __init__(self, address: str = "127.0.0.1", port: int = 0, sample_rate: int = 48000):
        super().__init__(address, port, sample_rate)
        self._config: Dict[int, dict] = {}

    def connect(self) -> bool:
        logger.info(f"GenericDSP: interface initialized (no hardware control)")
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
        self._config[channel]["polarity_inverted"] = inverted
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
        return True

    def set_eq_parametric(
        self,
        channel: int,
        frequency_hz: float,
        gain_db: float,
        q: float,
    ) -> bool:
        self._ensure_channel(channel)
        if "eq" not in self._config[channel]:
            self._config[channel]["eq"] = []
        self._config[channel]["eq"].append({
            "freq": frequency_hz,
            "gain_db": gain_db,
            "q": q,
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
        return self._config[channel].get("polarity_inverted", False)

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
        self._config = {}
        for ch_str, settings in config.items():
            ch = int(ch_str)
            self._ensure_channel(ch)
            for key, value in settings.items():
                if key in ("fir",) and isinstance(value, (list, np.ndarray)):
                    self._config[ch][key] = np.array(value, dtype=np.float64)
                else:
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
                    ch_dict[key] = value
                else:
                    ch_dict[key] = value
            result[str(ch)] = ch_dict
        return result

    def reset_to_defaults(self) -> bool:
        self._config = {}
        return True

    def export_as_yaml(self) -> str:
        import yaml
        from io import StringIO
        buf = StringIO()
        yaml.dump(self.read_configuration(), buf, default_flow_style=False)
        return buf.getvalue()

    def _ensure_channel(self, channel: int) -> None:
        if channel not in self._config:
            self._config[channel] = {
                "delay_ms": 0.0,
                "gain_db": 0.0,
                "polarity_inverted": False,
                "crossover_freq": None,
                "crossover_slope": 24.0,
                "eq": [],
                "fir": None,
                "muted": False,
            }
