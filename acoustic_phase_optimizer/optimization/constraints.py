"""Constraint handling for optimization parameter bounds."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from acoustic_phase_optimizer.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class Bounds:
    lower: NDArray[np.float64]
    upper: NDArray[np.float64]

    def __post_init__(self):
        self.lower = np.asarray(self.lower, dtype=np.float64)
        self.upper = np.asarray(self.upper, dtype=np.float64)

    def is_valid(self, params: NDArray[np.float64]) -> bool:
        return bool(np.all(params >= self.lower) and np.all(params <= self.upper))

    def clip(self, params: NDArray[np.float64]) -> NDArray[np.float64]:
        return np.clip(params, self.lower, self.upper)

    def normalize(self, params: NDArray[np.float64]) -> NDArray[np.float64]:
        return (params - self.lower) / (self.upper - self.lower + 1e-12)

    def denormalize(self, normalized: NDArray[np.float64]) -> NDArray[np.float64]:
        return normalized * (self.upper - self.lower) + self.lower


@dataclass
class Constraint:
    name: str
    min_value: float
    max_value: float
    current_value: float = 0.0
    penalty_weight: float = 1.0

    def violation(self) -> float:
        if self.current_value < self.min_value:
            return (self.min_value - self.current_value) * self.penalty_weight
        elif self.current_value > self.max_value:
            return (self.current_value - self.max_value) * self.penalty_weight
        return 0.0

    def is_satisfied(self) -> bool:
        return self.min_value <= self.current_value <= self.max_value


class Constraints:
    """Manages optimization constraints and penalties."""

    DEFAULT_BOUNDS = {
        "delay_ms": (0.0, 500.0),
        "gain_db": (-20.0, 12.0),
        "crossover_freq": (20.0, 20000.0),
        "crossover_slope": (6.0, 48.0),
        "eq_freq": (20.0, 20000.0),
        "eq_gain": (-15.0, 15.0),
        "eq_q": (0.1, 20.0),
        "fir_taps": (1, 1024),
    }

    def __init__(self):
        self._bounds: Dict[str, Bounds] = {}
        self._constraints: List[Constraint] = []
        self._initialize_defaults()

    def _initialize_defaults(self) -> None:
        for name, (lo, hi) in self.DEFAULT_BOUNDS.items():
            self._bounds[name] = Bounds(
                lower=np.array([lo], dtype=np.float64),
                upper=np.array([hi], dtype=np.float64),
            )

    def add_bound(self, name: str, lower: float, upper: float) -> None:
        self._bounds[name] = Bounds(
            lower=np.array([lower], dtype=np.float64),
            upper=np.array([upper], dtype=np.float64),
        )

    def add_constraint(
        self,
        name: str,
        min_value: float,
        max_value: float,
        penalty_weight: float = 1.0,
    ) -> None:
        self._constraints.append(
            Constraint(name, min_value, max_value, penalty_weight=penalty_weight)
        )

    def get_bound(self, name: str) -> Optional[Bounds]:
        return self._bounds.get(name)

    def clip_to_bounds(self, params: NDArray[np.float64], bound_name: str = "delay_ms") -> NDArray[np.float64]:
        bounds = self._bounds.get(bound_name)
        if bounds is None:
            return params
        return bounds.clip(params)

    def total_penalty(self) -> float:
        return sum(c.violation() for c in self._constraints)

    def check_all(self) -> List[str]:
        violated = []
        for constraint in self._constraints:
            if not constraint.is_satisfied():
                violated.append(constraint.name)
        return violated

    def speaker_parameter_bounds(self, n_speakers: int) -> Bounds:
        n_params = n_speakers * 3
        lower = np.zeros(n_params)
        upper = np.zeros(n_params)

        for i in range(n_speakers):
            lower[i * 3] = 0.0
            upper[i * 3] = 500.0

            lower[i * 3 + 1] = -20.0
            upper[i * 3 + 1] = 12.0

            lower[i * 3 + 2] = 0.0
            upper[i * 3 + 2] = 1.0

        return Bounds(lower=lower, upper=upper)

    def dsp_parameter_bounds(self, n_speakers: int) -> Bounds:
        n_peq = n_speakers * 6
        n_geq = n_speakers * 31
        n_common = n_speakers * 2
        total = n_peq * 3 + n_geq + n_common

        lower = np.zeros(total)
        upper = np.zeros(total)
        idx = 0

        for _ in range(n_speakers):
            for _ in range(6):
                lower[idx] = 20.0
                upper[idx] = 20000.0
                idx += 1
                lower[idx] = -15.0
                upper[idx] = 15.0
                idx += 1
                lower[idx] = 0.1
                upper[idx] = 20.0
                idx += 1

        for _ in range(n_speakers):
            for _ in range(31):
                lower[idx] = -15.0
                upper[idx] = 15.0
                idx += 1

        for _ in range(n_speakers):
            lower[idx] = 0.0
            upper[idx] = 500.0
            idx += 1

        for _ in range(n_speakers):
            lower[idx] = -1.0
            upper[idx] = 1.0
            idx += 1

        return Bounds(lower=lower, upper=upper)
