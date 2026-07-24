"""Gradient descent optimizer for acoustic parameter optimization."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from typing import Callable, Dict, List, Optional, Tuple
from acoustic_phase_optimizer.optimization.constraints import Bounds, Constraints
from acoustic_phase_optimizer.utils.logging import get_logger

logger = get_logger(__name__)


class GradientOptimizer:
    """Gradient descent optimization with adaptive learning rate and momentum."""

    def __init__(
        self,
        bounds: Optional[Bounds] = None,
        learning_rate: float = 0.01,
        momentum: float = 0.9,
        max_iterations: int = 1000,
        convergence_threshold: float = 1e-6,
    ):
        self.bounds = bounds
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.max_iterations = max_iterations
        self.convergence_threshold = convergence_threshold

    def optimize(
        self,
        objective_fn: Callable[[NDArray[np.float64]], float],
        initial_params: NDArray[np.float64],
    ) -> Tuple[NDArray[np.float64], float, Dict]:
        params = initial_params.copy().astype(np.float64)
        if self.bounds is not None:
            params = self.bounds.clip(params)

        velocity = np.zeros_like(params)
        best_params = params.copy()
        best_value = objective_fn(params)

        history = {
            "values": [best_value],
            "params": [params.copy()],
            "gradients": [],
        }

        for iteration in range(self.max_iterations):
            grad = self._compute_gradient(objective_fn, params)

            grad_norm = np.linalg.norm(grad)
            history["gradients"].append(grad_norm)

            velocity = self.momentum * velocity - self.learning_rate * grad
            params = params + velocity

            if self.bounds is not None:
                params = self.bounds.clip(params)

            current_value = objective_fn(params)
            history["values"].append(current_value)
            history["params"].append(params.copy())

            if current_value > best_value:
                best_params = params.copy()
                best_value = current_value

            if iteration > 0 and abs(history["values"][-1] - history["values"][-2]) < self.convergence_threshold:
                logger.info(f"Gradient descent converged at iteration {iteration}")
                break

            if iteration % 100 == 0:
                logger.debug(f"Gradient iteration {iteration}: value = {current_value:.6f}")

        history["best_value"] = best_value
        history["best_params"] = best_params
        history["iterations"] = iteration + 1

        return best_params, best_value, history

    def _compute_gradient(
        self,
        objective_fn: Callable[[NDArray[np.float64]], float],
        params: NDArray[np.float64],
        epsilon: float = 1e-6,
    ) -> NDArray[np.float64]:
        grad = np.zeros_like(params)
        base_value = objective_fn(params)

        for i in range(len(params)):
            params_step = params.copy()
            params_step[i] += epsilon

            if self.bounds is not None:
                params_step = self.bounds.clip(params_step)

            step_value = objective_fn(params_step)
            grad[i] = (base_value - step_value) / epsilon

        return grad

    def reset(self) -> None:
        pass
