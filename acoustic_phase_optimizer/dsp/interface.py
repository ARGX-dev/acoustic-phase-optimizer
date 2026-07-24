"""Abstract DSP interface for hardware communication abstraction layer."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple
from acoustic_phase_optimizer.utils.logging import get_logger

logger = get_logger(__name__)


class DSPInterface(ABC):
    """Abstract base class for all DSP processor interfaces.

    Provides a unified API for communicating with different DSP hardware
    including dbx DriveRack Venue360, generic DSPs, Dante, and AES67 devices.
    """

    def __init__(self, address: str, port: int, sample_rate: int = 48000):
        self.address = address
        self.port = port
        self.sample_rate = sample_rate
        self._connected = False

    @property
    def is_connected(self) -> bool:
        return self._connected

    @abstractmethod
    def connect(self) -> bool:
        ...

    @abstractmethod
    def disconnect(self) -> bool:
        ...

    @abstractmethod
    def set_delay(self, channel: int, delay_ms: float) -> bool:
        ...

    @abstractmethod
    def set_gain(self, channel: int, gain_db: float) -> bool:
        ...

    @abstractmethod
    def set_polarity(self, channel: int, inverted: bool) -> bool:
        ...

    @abstractmethod
    def set_crossover(
        self,
        channel: int,
        frequency_hz: float,
        slope_db_per_octave: float = 24.0,
    ) -> bool:
        ...

    @abstractmethod
    def set_eq_parametric(
        self,
        channel: int,
        frequency_hz: float,
        gain_db: float,
        q: float,
    ) -> bool:
        ...

    @abstractmethod
    def set_fir_coefficients(
        self,
        channel: int,
        coefficients: NDArray[np.float64],
    ) -> bool:
        ...

    @abstractmethod
    def get_delay(self, channel: int) -> Optional[float]:
        ...

    @abstractmethod
    def get_gain(self, channel: int) -> Optional[float]:
        ...

    @abstractmethod
    def get_polarity(self, channel: int) -> Optional[bool]:
        ...

    @abstractmethod
    def get_crossover(self, channel: int) -> Optional[Tuple[float, float]]:
        ...

    @abstractmethod
    def mute_channel(self, channel: int, muted: bool) -> bool:
        ...

    @abstractmethod
    def apply_configuration(self, config: dict) -> bool:
        ...

    @abstractmethod
    def read_configuration(self) -> dict:
        ...

    @abstractmethod
    def reset_to_defaults(self) -> bool:
        ...

    def validate_sample_rate(self, rate: int) -> bool:
        valid_rates = [44100, 48000, 96000, 192000]
        return rate in valid_rates

    @staticmethod
    def create(config: dict) -> "DSPInterface":
        dsp_type = config.get("type", "generic").lower()
        address = config.get("address", "127.0.0.1")
        port = config.get("port", 23)
        sample_rate = config.get("sample_rate", 48000)

        if dsp_type == "dbx_venus360":
            from acoustic_phase_optimizer.dsp.venus360 import Venus360
            return Venus360(address, port, sample_rate)
        elif dsp_type == "generic":
            from acoustic_phase_optimizer.dsp.generic import GenericDSP
            return GenericDSP(address, port, sample_rate)
        elif dsp_type == "dante":
            from acoustic_phase_optimizer.dsp.dante import DanteInterface
            return DanteInterface(address, port, sample_rate)
        elif dsp_type == "aes67":
            from acoustic_phase_optimizer.dsp.aes67 import AES67Interface
            return AES67Interface(address, port, sample_rate)
        else:
            logger.warning(f"Unknown DSP type '{dsp_type}', using generic interface")
            from acoustic_phase_optimizer.dsp.generic import GenericDSP
            return GenericDSP(address, port, sample_rate)
