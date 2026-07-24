"""Measurement microphone definitions and positions."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class Microphone:
    """A measurement microphone position in the venue."""

    name: str
    position: NDArray[np.float64]
    measurement_data: Optional[dict] = None

    zone: str = "general"
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

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "position": self.position.tolist(),
            "zone": self.zone,
            "enabled": self.enabled,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "Microphone":
        return cls(
            name=data["name"],
            position=np.array(data["position"], dtype=np.float64),
            zone=data.get("zone", "general"),
            enabled=data.get("enabled", True),
        )
