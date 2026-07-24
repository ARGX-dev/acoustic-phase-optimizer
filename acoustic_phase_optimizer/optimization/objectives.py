"""Objective functions for acoustic optimization."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from typing import Dict, List, Optional, Tuple, Callable
from dataclasses import dataclass
from acoustic_phase_optimizer.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class ObjectiveWeights:
    phase_coherence: float = 1.0
    magnitude_flatness: float = 0.5
    destructive_interference: float = 1.5
    delay_alignment: float = 1.0
    rt60_deviation: float = 0.3


class ObjectiveFunction:
    """Defines and computes optimization objective functions.

    The goal is to maximize average coherence across the listening area
    while minimizing destructive interference.
    """

    def __init__(self, weights: Optional[ObjectiveWeights] = None):
        self.weights = weights or ObjectiveWeights()

    def set_weights(self, weights: ObjectiveWeights) -> None:
        self.weights = weights

    def compute(
        self,
        params: NDArray[np.float64],
        measurement_data: dict,
    ) -> float:
        phase_coherence = self.phase_coherence_objective(params, measurement_data)
        magnitude_flatness = self.magnitude_flatness_objective(params, measurement_data)
        interference = self.destructive_interference_objective(params, measurement_data)
        delay_align = self.delay_alignment_objective(params, measurement_data)
        rt60_dev = self.rt60_deviation_objective(params, measurement_data)

        total = (
            self.weights.phase_coherence * phase_coherence +
            self.weights.magnitude_flatness * magnitude_flatness +
            self.weights.destructive_interference * interference +
            self.weights.delay_alignment * delay_align +
            self.weights.rt60_deviation * rt60_dev
        )

        return float(total)

    def phase_coherence_objective(
        self,
        params: NDArray[np.float64],
        data: dict,
    ) -> float:
        phases = data.get("phase", None)
        if phases is None or len(phases) == 0:
            return 0.0

        if isinstance(phases, list) and len(phases) > 1:
            phase_array = np.array(phases)
            phase_variance = np.var(np.unwrap(phase_array, axis=0), axis=0)
        else:
            return 0.0

        coherence = np.exp(-np.mean(phase_variance))
        return float(coherence)

    def magnitude_flatness_objective(
        self,
        params: NDArray[np.float64],
        data: dict,
    ) -> float:
        magnitudes = data.get("magnitude_db", None)
        if magnitudes is None or len(magnitudes) == 0:
            return 0.0

        if isinstance(magnitudes, list) and len(magnitudes) > 1:
            mag_array = np.array(magnitudes)
            mean_mag = np.mean(mag_array, axis=0)
        else:
            mean_mag = np.array(magnitudes).flatten()

        variance = np.var(mean_mag)
        flatness = np.exp(-variance / 100.0)
        return float(flatness)

    def destructive_interference_objective(
        self,
        params: NDArray[np.float64],
        data: dict,
    ) -> float:
        cancellations = data.get("cancellation_zones", None)
        if cancellations is not None and len(cancellations) > 0:
            mean_cancellation = float(np.mean(np.abs(cancellations)))
            if mean_cancellation > 0:
                return float(np.exp(-mean_cancellation))
        return 0.5

    def delay_alignment_objective(
        self,
        params: NDArray[np.float64],
        data: dict,
    ) -> float:
        delays = data.get("delays_ms", None)
        if delays is None or len(delays) == 0:
            return 0.0

        if isinstance(delays, list) and len(delays) > 1:
            delay_array = np.array(delays)
            delay_variance = np.var(delay_array)
        else:
            delay_array = np.array(delays).flatten()
            delay_variance = np.var(delay_array)

        alignment = np.exp(-delay_variance / 1000.0)
        return float(alignment)

    def rt60_deviation_objective(
        self,
        params: NDArray[np.float64],
        data: dict,
    ) -> float:
        rt60_values = data.get("rt60", None)
        if rt60_values is None or len(rt60_values) == 0:
            return 0.5

        if isinstance(rt60_values, dict):
            values = np.array([v for v in rt60_values.values() if isinstance(v, (int, float))])
        else:
            values = np.array(rt60_values).flatten()

        if len(values) == 0:
            return 0.5

        target_rt60 = 0.8
        deviation = np.mean(np.abs(values - target_rt60))
        return float(np.exp(-deviation * 2.0))

    def combined_objective(
        self,
        params: NDArray[np.float64],
        measurement_data: dict,
    ) -> float:
        return self.compute(params, measurement_data)

    def compute_array(
        self,
        params_batch: NDArray[np.float64],
        measurement_data: dict,
    ) -> NDArray[np.float64]:
        return np.array([
            self.compute(params, measurement_data) for params in params_batch
        ])

    def get_objective_names(self) -> List[str]:
        return [
            "phase_coherence",
            "magnitude_flatness",
            "destructive_interference",
            "delay_alignment",
            "rt60_deviation",
        ]
