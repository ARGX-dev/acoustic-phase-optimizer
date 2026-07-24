"""Main optimization engine orchestrating all optimization algorithms."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from typing import Callable, Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from acoustic_phase_optimizer.optimization.objectives import (
    ObjectiveFunction, ObjectiveWeights,
)
from acoustic_phase_optimizer.optimization.constraints import Constraints, Bounds
from acoustic_phase_optimizer.optimization.gradient import GradientOptimizer
from acoustic_phase_optimizer.optimization.genetic import GeneticOptimizer
from acoustic_phase_optimizer.optimization.annealing import AnnealingOptimizer
from acoustic_phase_optimizer.optimization.bayesian import BayesianOptimizer
from acoustic_phase_optimizer.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class OptimizationResult:
    algorithm: str
    best_params: NDArray[np.float64]
    best_value: float
    history: dict
    success: bool
    iterations: int
    computation_time: float = 0.0


class OptimizationEngine:
    """Orchestrates optimization across multiple algorithms and compares results."""

    def __init__(self, config: Optional[dict] = None):
        self.config = config or {}
        self.objective_fn = ObjectiveFunction()
        self.constraints = Constraints()

        self._algorithms: Dict[str, str] = {
            "gradient": "gradient",
            "genetic": "genetic",
            "annealing": "annealing",
            "bayesian": "bayesian",
        }

    def optimize(
        self,
        algorithm: str,
        objective_fn: Callable[[NDArray[np.float64]], float],
        initial_params: NDArray[np.float64],
    ) -> OptimizationResult:
        algorithm_key = algorithm.lower()

        if algorithm_key == "gradient":
            optimizer = self._create_gradient_optimizer()
        elif algorithm_key == "genetic":
            optimizer = self._create_genetic_optimizer()
        elif algorithm_key == "annealing":
            optimizer = self._create_annealing_optimizer()
        elif algorithm_key == "bayesian":
            optimizer = self._create_bayesian_optimizer()
        else:
            raise ValueError(f"Unknown algorithm: {algorithm}. "
                           f"Choose from: {list(self._algorithms.keys())}")

        import time
        start_time = time.time()

        try:
            best_params, best_value, history = optimizer.optimize(
                objective_fn, initial_params
            )
            success = True
        except Exception as e:
            logger.error(f"Optimization failed with {algorithm}: {e}")
            best_params = initial_params
            best_value = objective_fn(initial_params)
            history = {"error": str(e)}
            success = False

        elapsed = time.time() - start_time

        return OptimizationResult(
            algorithm=algorithm,
            best_params=best_params,
            best_value=best_value,
            history=history,
            success=success,
            iterations=history.get("iterations", 1),
            computation_time=elapsed,
        )

    def compare_algorithms(
        self,
        objective_fn: Callable[[NDArray[np.float64]], float],
        initial_params: NDArray[np.float64],
        algorithms: Optional[List[str]] = None,
    ) -> Dict[str, OptimizationResult]:
        if algorithms is None:
            algorithms = list(self._algorithms.keys())

        results = {}
        for algo in algorithms:
            logger.info(f"Running {algo} optimization...")
            result = self.optimize(algo, objective_fn, initial_params)
            results[algo] = result
            logger.info(
                f"  {algo}: best = {result.best_value:.6f}, "
                f"iterations = {result.iterations}, "
                f"time = {result.computation_time:.2f}s"
            )

        return results

    def get_best_result(
        self,
        results: Dict[str, OptimizationResult],
    ) -> Tuple[str, OptimizationResult]:
        best_algo = max(results, key=lambda k: results[k].best_value)
        return best_algo, results[best_algo]

    def set_objective_weights(self, weights: ObjectiveWeights) -> None:
        self.objective_fn.set_weights(weights)

    def setup_speaker_optimization(
        self,
        n_speakers: int,
    ) -> Tuple[Bounds, NDArray[np.float64]]:
        bounds = self.constraints.speaker_parameter_bounds(n_speakers)

        initial_params = np.zeros(n_speakers * 3)
        for i in range(n_speakers):
            initial_params[i * 3] = 10.0
            initial_params[i * 3 + 1] = 0.0
            initial_params[i * 3 + 2] = 0.0

        return bounds, initial_params

    def _create_gradient_optimizer(self) -> GradientOptimizer:
        return GradientOptimizer(
            learning_rate=self.config.get("learning_rate", 0.01),
            max_iterations=self.config.get("max_iterations", 1000),
            convergence_threshold=self.config.get("convergence_threshold", 1e-6),
        )

    def _create_genetic_optimizer(self) -> GeneticOptimizer:
        return GeneticOptimizer(
            population_size=self.config.get("population_size", 100),
            mutation_rate=self.config.get("mutation_rate", 0.15),
            crossover_rate=self.config.get("crossover_rate", 0.8),
            max_iterations=self.config.get("max_iterations", 1000),
            convergence_threshold=self.config.get("convergence_threshold", 1e-6),
        )

    def _create_annealing_optimizer(self) -> AnnealingOptimizer:
        return AnnealingOptimizer(
            initial_temperature=self.config.get("initial_temperature", 100.0),
            cooling_rate=self.config.get("cooling_rate", 0.95),
            max_iterations=self.config.get("max_iterations", 1000),
            convergence_threshold=self.config.get("convergence_threshold", 1e-6),
        )

    def _create_bayesian_optimizer(self) -> BayesianOptimizer:
        return BayesianOptimizer(
            n_initial_points=10,
            max_iterations=self.config.get("max_iterations", 500),
            convergence_threshold=self.config.get("convergence_threshold", 1e-6),
        )
