"""Genetic algorithm optimizer for acoustic parameter optimization."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from typing import Callable, Dict, List, Optional, Tuple
from acoustic_phase_optimizer.optimization.constraints import Bounds
from acoustic_phase_optimizer.utils.logging import get_logger

logger = get_logger(__name__)


class GeneticOptimizer:
    """Genetic algorithm optimization for multi-parameter acoustic tuning."""

    def __init__(
        self,
        bounds: Optional[Bounds] = None,
        population_size: int = 100,
        mutation_rate: float = 0.15,
        crossover_rate: float = 0.8,
        elite_ratio: float = 0.1,
        max_iterations: int = 1000,
        convergence_threshold: float = 1e-6,
    ):
        self.bounds = bounds
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.elite_count = max(1, int(population_size * elite_ratio))
        self.max_iterations = max_iterations
        self.convergence_threshold = convergence_threshold

    def optimize(
        self,
        objective_fn: Callable[[NDArray[np.float64]], float],
        initial_params: NDArray[np.float64],
    ) -> Tuple[NDArray[np.float64], float, Dict]:
        n_params = len(initial_params)
        population = self._initialize_population(n_params, initial_params)
        fitness = np.array([objective_fn(ind) for ind in population])

        best_idx = np.argmax(fitness)
        best_params = population[best_idx].copy()
        best_value = float(fitness[best_idx])

        history = {
            "values": [best_value],
            "mean_fitness": [float(np.mean(fitness))],
            "std_fitness": [float(np.std(fitness))],
            "best_params": [best_params.copy()],
        }

        stall_count = 0
        for generation in range(self.max_iterations):
            new_population = self._elite_selection(population, fitness)

            while len(new_population) < self.population_size:
                parent1 = self._tournament_select(population, fitness)
                parent2 = self._tournament_select(population, fitness)

                if np.random.random() < self.crossover_rate:
                    child1, child2 = self._crossover(parent1, parent2)
                else:
                    child1, child2 = parent1.copy(), parent2.copy()

                child1 = self._mutate(child1)
                child2 = self._mutate(child2)

                new_population.append(child1)
                if len(new_population) < self.population_size:
                    new_population.append(child2)

            population = np.array(new_population[:self.population_size])
            fitness = np.array([objective_fn(ind) for ind in population])

            current_best_idx = np.argmax(fitness)
            current_best_value = float(fitness[current_best_idx])

            history["values"].append(current_best_value)
            history["mean_fitness"].append(float(np.mean(fitness)))
            history["std_fitness"].append(float(np.std(fitness)))

            if current_best_value > best_value:
                best_value = current_best_value
                best_params = population[current_best_idx].copy()
                stall_count = 0
            else:
                stall_count += 1

            history["best_params"].append(best_params.copy())

            if stall_count > 50:
                logger.info(f"Genetic algorithm stalled at generation {generation}")
                break

            if generation > 20:
                recent = history["values"][-20:]
                if max(recent) - min(recent) < self.convergence_threshold:
                    logger.info(f"Genetic algorithm converged at generation {generation}")
                    break

            if generation % 50 == 0:
                logger.debug(
                    f"Gen {generation}: best = {best_value:.6f}, "
                    f"mean = {history['mean_fitness'][-1]:.6f}"
                )

        history["generations"] = generation + 1
        return best_params, best_value, history

    def _initialize_population(
        self,
        n_params: int,
        initial_params: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        population = np.zeros((self.population_size, n_params))

        population[0] = initial_params

        for i in range(1, self.population_size):
            if self.bounds is not None:
                population[i] = np.random.uniform(
                    self.bounds.lower, self.bounds.upper, size=n_params
                )
            else:
                population[i] = initial_params + np.random.randn(n_params) * 0.1

        return population

    def _tournament_select(
        self,
        population: NDArray[np.float64],
        fitness: NDArray[np.float64],
        tournament_size: int = 3,
    ) -> NDArray[np.float64]:
        indices = np.random.choice(len(population), tournament_size, replace=False)
        winner_idx = indices[np.argmax(fitness[indices])]
        return population[winner_idx].copy()

    def _crossover(
        self,
        parent1: NDArray[np.float64],
        parent2: NDArray[np.float64],
    ) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
        n = len(parent1)
        mask = np.random.random(n) < 0.5

        child1 = np.where(mask, parent1, parent2)
        child2 = np.where(mask, parent2, parent1)
        return child1, child2

    def _mutate(self, individual: NDArray[np.float64]) -> NDArray[np.float64]:
        mutation_mask = np.random.random(len(individual)) < self.mutation_rate

        if np.any(mutation_mask):
            mutation = np.random.randn(len(individual)) * 0.1
            individual = individual + mutation * mutation_mask

            if self.bounds is not None:
                individual = self.bounds.clip(individual)

        return individual

    def _elite_selection(
        self,
        population: NDArray[np.float64],
        fitness: NDArray[np.float64],
    ) -> List[NDArray[np.float64]]:
        elite_indices = np.argsort(fitness)[-self.elite_count:]
        return [population[idx].copy() for idx in elite_indices[::-1]]
