"""Simulated annealing optimizer for acoustic parameter optimization."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from typing import Callable, Dict, List, Optional, Tuple
from acoustic_phase_optimizer.optimization.constraints import Bounds
from acoustic_phase_optimizer.utils.logging import get_logger

logger = get_logger(__name__)


class AnnealingOptimizer:
    """Simulated annealing optimization with adaptive cooling schedule."""

    def __init__(
        self,
        bounds: Optional[Bounds] = None,
        initial_temperature: float = 100.0,
        cooling_rate: float = 0.95,
        min_temperature: float = 1e-6,
        max_iterations: int = 1000,
        convergence_threshold: float = 1e-6,
        reanneal_interval: int = 200,
    ):
        self.bounds = bounds
        self.initial_temperature = initial_temperature
        self.cooling_rate = cooling_rate
        self.min_temperature = min_temperature
        self.max_iterations = max_iterations
        self.convergence_threshold = convergence_threshold
        self.reanneal_interval = reanneal_interval

    def optimize(
        self,
        objective_fn: Callable[[NDArray[np.float64]], float],
        initial_params: NDArray[np.float64],
    ) -> Tuple[NDArray[np.float64], float, Dict]:
        current = initial_params.copy().astype(np.float64)
        if self.bounds is not None:
            current = self.bounds.clip(current)

        current_value = objective_fn(current)

        best_params = current.copy()
        best_value = current_value

        temperature = self.initial_temperature
        step_sizes = np.abs(self.bounds.upper - self.bounds.lower) * 0.1 if self.bounds is not None else \
                     np.ones_like(current) * 0.1

        history = {
            "values": [current_value],
            "temperature": [temperature],
            "acceptance_rate": [],
            "best_values": [best_value],
        }

        accepts = 0
        total_attempts = 0

        for iteration in range(self.max_iterations):
            candidate = self._propose_candidate(current, step_sizes)
            if self.bounds is not None:
                candidate = self.bounds.clip(candidate)

            candidate_value = objective_fn(candidate)
            delta = candidate_value - current_value

            total_attempts += 1

            if delta > 0:
                current = candidate
                current_value = candidate_value
                accepts += 1

                if candidate_value > best_value:
                    best_params = candidate.copy()
                    best_value = candidate_value
            else:
                acceptance_prob = np.exp(delta / (temperature + 1e-12))
                if np.random.random() < acceptance_prob:
                    current = candidate
                    current_value = candidate_value
                    accepts += 1

            temperature = max(
                self.min_temperature,
                temperature * self.cooling_rate,
            )

            history["values"].append(current_value)
            history["temperature"].append(temperature)
            history["best_values"].append(best_value)

            if total_attempts > 0:
                accept_rate = accepts / total_attempts
            else:
                accept_rate = 0.0
            history["acceptance_rate"] = history.get("acceptance_rate", []) + [accept_rate]

            if iteration > 20 and self._check_convergence(history["best_values"], 20):
                logger.info(f"Annealing converged at iteration {iteration}")
                break

            if iteration % self.reanneal_interval == 0 and iteration > 0:
                if accept_rate < 0.1:
                    temperature *= 2.0
                    logger.debug(f"Re-annealing at iteration {iteration}")

            if iteration % 100 == 0:
                logger.debug(
                    f"Annealing iter {iteration}: T = {temperature:.4f}, "
                    f"best = {best_value:.6f}, accept = {accept_rate:.2%}"
                )

        history["iterations"] = iteration + 1
        history["final_temperature"] = temperature

        return best_params, best_value, history

    def _propose_candidate(
        self,
        current: NDArray[np.float64],
        step_sizes: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        perturbation = np.random.normal(0, step_sizes, size=len(current))
        return current + perturbation

    def _check_convergence(
        self,
        values: List[float],
        window: int = 20,
    ) -> bool:
        if len(values) < window:
            return False
        recent = values[-window:]
        return (max(recent) - min(recent)) < self.convergence_threshold

    def reset(self) -> None:
        pass
