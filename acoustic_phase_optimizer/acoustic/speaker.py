"""Speaker (loudspeaker) definitions and acoustic properties."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from enum import Enum
from acoustic_phase_optimizer.dsp.peq import PEQConfig
from acoustic_phase_optimizer.dsp.geq import GEQConfig


class SpeakerType(Enum):
    MAIN_LEFT = "left_main"
    MAIN_RIGHT = "right_main"
    CENTER = "center"
    SUBWOOFER = "subwoofer"
    FRONT_FILL = "front_fill"
    MONITOR = "monitor"
    DELAY = "delay"


class SpeakerPolarity(Enum):
    NORMAL = 1
    INVERTED = -1


@dataclass
class Speaker:
    """A loudspeaker in the sound system."""

    name: str
    speaker_type: SpeakerType
    position: NDArray[np.float64]
    polarity: SpeakerPolarity = SpeakerPolarity.NORMAL

    delay_ms: float = 0.0
    gain_db: float = 0.0

    crossover_freq_low: Optional[float] = None
    crossover_freq_high: Optional[float] = None

    horizontal_coverage: float = 90.0
    vertical_coverage: float = 60.0
    max_spl: float = 130.0

    sensitivity_db: float = 95.0
    impedance_ohms: float = 8.0

    peq: PEQConfig = field(default_factory=PEQConfig)
    geq: GEQConfig = field(default_factory=GEQConfig)

    fir_coefficients: Optional[NDArray[np.float64]] = None
    iir_eq: List[dict] = field(default_factory=list)

    enabled: bool = True

    def __post_init__(self):
        if isinstance(self.position, list):
            self.position = np.array(self.position, dtype=np.float64)

    @property
    def x(self) -> float:
        return float(self.position[0])

    @property
    def y(self) -> float:
        return float(self.position[1]) if len(self.position) > 1 else 0.0

    @property
    def z(self) -> float:
        return float(self.position[2]) if len(self.position) > 2 else 0.0

    def distance_to(self, point: NDArray[np.float64]) -> float:
        return float(np.linalg.norm(self.position - point))

    def delay_from_distance(self, point: NDArray[np.float64], speed_of_sound: float = 343.0) -> float:
        return self.distance_to(point) / speed_of_sound * 1000.0

    def spl_at_distance(
        self,
        point: NDArray[np.float64],
        input_power_db: float = 0.0,
    ) -> float:
        distance = self.distance_to(point)
        if distance < 0.1:
            distance = 0.1
        spl = self.sensitivity_db + input_power_db - 20.0 * np.log10(distance)
        return float(spl)

    def is_in_coverage(
        self,
        point: NDArray[np.float64],
    ) -> bool:
        direction = point - self.position
        dist = np.linalg.norm(direction)
        if dist < 0.01:
            return True

        direction = direction / dist

        angle_h = np.degrees(np.arctan2(direction[1], direction[0]))
        angle_v = np.degrees(np.arcsin(direction[2] / dist))

        horizontal_tolerance = self.horizontal_coverage / 2.0
        on_axis_h = np.array([0.0, 1.0, 0.0])
        on_axis_v = np.array([0.0, 0.0, 1.0])

        dot_h = np.abs(np.dot(direction, on_axis_h))
        dot_v = np.abs(np.dot(direction, on_axis_v))

        return dot_h >= np.cos(np.radians(horizontal_tolerance)) and \
               dot_v >= np.cos(np.radians(self.vertical_coverage / 2.0))

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "type": self.speaker_type.value,
            "position": self.position.tolist(),
            "polarity": self.polarity.value,
            "delay_ms": self.delay_ms,
            "gain_db": self.gain_db,
            "crossover_freq_low": self.crossover_freq_low,
            "crossover_freq_high": self.crossover_freq_high,
            "horizontal_coverage": self.horizontal_coverage,
            "vertical_coverage": self.vertical_coverage,
            "peq_bands": [(b.freq_hz, b.gain_db, b.q) for b in self.peq.bands],
            "geq_gains": list(self.geq.gains_db),
            "enabled": self.enabled,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Speaker":
        peq_bands_data = data.get("peq_bands", [])
        from acoustic_phase_optimizer.dsp.peq import PEQBand
        peq = PEQConfig(bands=[PEQBand(freq_hz=f, gain_db=g, q=q) for f, g, q in peq_bands_data]) if peq_bands_data else PEQConfig()

        geq_gains = data.get("geq_gains", None)
        geq = GEQConfig(gains_db=list(geq_gains)) if geq_gains else GEQConfig()

        return cls(
            name=data["name"],
            speaker_type=SpeakerType(data["type"]),
            position=np.array(data["position"], dtype=np.float64),
            polarity=SpeakerPolarity(data.get("polarity", 1)),
            delay_ms=data.get("delay_ms", 0.0),
            gain_db=data.get("gain_db", 0.0),
            peq=peq,
            geq=geq,
            crossover_freq_low=data.get("crossover_freq_low"),
            crossover_freq_high=data.get("crossover_freq_high"),
            horizontal_coverage=data.get("horizontal_coverage", 90.0),
            vertical_coverage=data.get("vertical_coverage", 60.0),
            enabled=data.get("enabled", True),
        )
