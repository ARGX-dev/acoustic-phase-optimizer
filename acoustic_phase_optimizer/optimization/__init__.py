"""
Optimization engine module for DSP parameter calculation.

Provides multiple optimization algorithms: gradient descent, genetic algorithms,
simulated annealing, and Bayesian optimization. Computes speaker delays,
crossover settings, FIR/IIR filters, gain optimization, and polarity inversion.
"""

from acoustic_phase_optimizer.optimization.engine import OptimizationEngine
from acoustic_phase_optimizer.optimization.objectives import ObjectiveFunction
from acoustic_phase_optimizer.optimization.constraints import Constraints

__all__ = [
    "OptimizationEngine",
    "ObjectiveFunction",
    "Constraints",
]
