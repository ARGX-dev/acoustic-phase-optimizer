"""Tests for the optimization module."""

import numpy as np
import pytest
from acoustic_phase_optimizer.optimization.objectives import ObjectiveFunction, ObjectiveWeights
from acoustic_phase_optimizer.optimization.constraints import Constraints, Bounds, Constraint
from acoustic_phase_optimizer.optimization.gradient import GradientOptimizer
from acoustic_phase_optimizer.optimization.genetic import GeneticOptimizer
from acoustic_phase_optimizer.optimization.annealing import AnnealingOptimizer
from acoustic_phase_optimizer.optimization.bayesian import BayesianOptimizer
from acoustic_phase_optimizer.optimization.engine import OptimizationEngine


class TestObjectiveFunction:
    def test_phase_coherence(self):
        obj = ObjectiveFunction()
        data = {"phase": [[0.1, 0.2], [0.15, 0.25]]}
        v = obj.phase_coherence_objective(np.array([1.0]), data)
        assert 0 <= v <= 1

    def test_magnitude_flatness(self):
        obj = ObjectiveFunction()
        data = {"magnitude_db": [[-20, -18], [-22, -19]]}
        v = obj.magnitude_flatness_objective(np.array([1.0]), data)
        assert 0 <= v <= 1

    def test_destructive_interference(self):
        obj = ObjectiveFunction()
        data = {"cancellation_zones": np.array([0.1, 0.2])}
        v = obj.destructive_interference_objective(np.array([1.0]), data)
        assert 0 <= v <= 1

    def test_delay_alignment(self):
        obj = ObjectiveFunction()
        data = {"delays_ms": [10, 12, 11]}
        v = obj.delay_alignment_objective(np.array([1.0]), data)
        assert 0 <= v <= 1

    def test_rt60_deviation(self):
        obj = ObjectiveFunction()
        data = {"rt60": {"rt60_500hz": 0.8, "rt60_1000hz": 0.9}}
        v = obj.rt60_deviation_objective(np.array([1.0]), data)
        assert 0 <= v <= 1

    def test_compute(self):
        obj = ObjectiveFunction()
        data = {
            "phase": [[0.1]],
            "magnitude_db": [[-20]],
            "cancellation_zones": np.array([0.1]),
            "delays_ms": [10],
            "rt60": {"broadband": 0.8},
        }
        v = obj.compute(np.array([1.0]), data)
        assert isinstance(v, float)

    def test_empty_data(self):
        obj = ObjectiveFunction()
        data = {}
        v = obj.compute(np.array([1.0]), data)
        assert isinstance(v, float)

    def test_set_weights(self):
        obj = ObjectiveFunction()
        weights = ObjectiveWeights(1.0, 2.0, 3.0, 4.0, 5.0)
        obj.set_weights(weights)
        assert obj.weights.phase_coherence == 1.0
        assert obj.weights.magnitude_flatness == 2.0

    def test_objective_names(self):
        obj = ObjectiveFunction()
        names = obj.get_objective_names()
        assert len(names) == 5

    def test_compute_array(self):
        obj = ObjectiveFunction()
        batch = np.array([[1.0], [2.0]])
        data = {
            "phase": [[0.1]],
            "magnitude_db": [[-20]],
            "cancellation_zones": np.array([0.1]),
            "delays_ms": [10],
            "rt60": {"broadband": 0.8},
        }
        values = obj.compute_array(batch, data)
        assert len(values) == 2


class TestConstraints:
    def test_default_bounds(self):
        c = Constraints()
        bound = c.get_bound("delay_ms")
        assert bound is not None
        assert bound.lower[0] == 0.0

    def test_add_bound(self):
        c = Constraints()
        c.add_bound("test", -1.0, 1.0)
        bound = c.get_bound("test")
        assert bound is not None

    def test_add_constraint(self):
        c = Constraints()
        c.add_constraint("test", 0.0, 10.0)
        violations = c.check_all()
        assert len(violations) == 0

    def test_clip_to_bounds(self):
        c = Constraints()
        clipped = c.clip_to_bounds(np.array([1000.0]), "delay_ms")
        assert clipped[0] == 500.0

    def test_speaker_parameter_bounds(self):
        c = Constraints()
        bounds = c.speaker_parameter_bounds(3)
        assert len(bounds.lower) == 9
        assert len(bounds.upper) == 9

    def test_bounds_is_valid(self):
        bounds = Bounds(np.array([0.0]), np.array([10.0]))
        assert bounds.is_valid(np.array([5.0]))
        assert not bounds.is_valid(np.array([-1.0]))

    def test_bounds_clip(self):
        bounds = Bounds(np.array([0.0]), np.array([10.0]))
        assert bounds.clip(np.array([-5.0]))[0] == 0.0
        assert bounds.clip(np.array([15.0]))[0] == 10.0

    def test_bounds_normalize(self):
        bounds = Bounds(np.array([0.0]), np.array([10.0]))
        n = bounds.normalize(np.array([5.0]))
        assert abs(n[0] - 0.5) < 1e-10

    def test_constraint_violation(self):
        c = Constraint("test", 0.0, 10.0, current_value=15.0)
        assert c.violation() > 0
        assert not c.is_satisfied()

    def test_constraint_satisfied(self):
        c = Constraint("test", 0.0, 10.0, current_value=5.0)
        assert c.violation() == 0
        assert c.is_satisfied()

    def test_total_penalty(self):
        c = Constraints()
        c.add_constraint("test", 0.0, 10.0, penalty_weight=2.0)
        total = c.total_penalty()
        assert total >= 0


class TestGradientOptimizer:
    def test_simple_optimization(self):
        bounds = Bounds(np.array([-5.0, -5.0]), np.array([5.0, 5.0]))
        def fn(x):
            return -(x[0] ** 2 + x[1] ** 2) + 10
        opt = GradientOptimizer(
            bounds=bounds, max_iterations=500,
            learning_rate=0.1, momentum=0.0,
        )
        best_params, best_value, history = opt.optimize(fn, np.array([2.0, -1.0]))
        assert best_value > 6.0
        assert "values" in history

    def test_bounded_optimization(self):
        bounds = Bounds(np.array([-5.0]), np.array([5.0]))
        def fn(x):
            return -(x[0] - 2.0) ** 2 + 10.0
        opt = GradientOptimizer(bounds=bounds, max_iterations=500, learning_rate=0.1)
        best_params, best_value, history = opt.optimize(fn, np.array([0.0]))
        assert best_value > 8.0

    def test_convergence(self):
        bounds = Bounds(np.array([-5.0, -5.0]), np.array([5.0, 5.0]))
        def fn(x):
            return -(x[0] ** 2 + x[1] ** 2)
        opt = GradientOptimizer(
            bounds=bounds, max_iterations=500,
            convergence_threshold=1e-3, learning_rate=0.1,
        )
        best_params, best_value, history = opt.optimize(fn, np.array([3.0, 2.0]))
        assert len(history["values"]) < 500


class TestGeneticOptimizer:
    def test_simple_optimization(self):
        bounds = Bounds(np.array([-10.0, -10.0]), np.array([10.0, 10.0]))
        def fn(x):
            return -np.sum(x ** 2) + 100
        opt = GeneticOptimizer(
            bounds=bounds, population_size=50,
            max_iterations=100, mutation_rate=0.2,
        )
        best_params, best_value, history = opt.optimize(fn, np.array([0.0, 0.0]))
        assert best_value > 90

    def test_elite_preservation(self):
        bounds = Bounds(np.array([0.0]), np.array([1.0]))
        def fn(x):
            return x[0]
        opt = GeneticOptimizer(bounds=bounds, population_size=20, elite_ratio=0.2)
        best_params, best_value, history = opt.optimize(fn, np.array([0.5]))
        assert best_value > 0.5

    def test_population_diversity(self):
        bounds = Bounds(np.array([-5.0]), np.array([5.0]))
        def fn(x):
            return -abs(x[0]) + 5
        opt = GeneticOptimizer(bounds=bounds, population_size=30, max_iterations=50)
        best_params, best_value, history = opt.optimize(fn, np.array([0.0]))
        assert "mean_fitness" in history
        assert "std_fitness" in history


class TestAnnealingOptimizer:
    def test_simple_optimization(self):
        bounds = Bounds(np.array([-10.0]), np.array([10.0]))
        def fn(x):
            return -(x[0] - 3) ** 2 + 50
        opt = AnnealingOptimizer(
            bounds=bounds, initial_temperature=50.0,
            cooling_rate=0.95, max_iterations=200,
        )
        best_params, best_value, history = opt.optimize(fn, np.array([0.0]))
        assert best_value > 45

    def test_temperature_schedule(self):
        bounds = Bounds(np.array([-10.0]), np.array([10.0]))
        def fn(x):
            return -x[0] ** 2
        opt = AnnealingOptimizer(
            bounds=bounds, initial_temperature=100.0,
            cooling_rate=0.9, max_iterations=50,
        )
        best_params, best_value, history = opt.optimize(fn, np.array([5.0]))
        assert "temperature" in history
        assert history["temperature"][-1] < history["temperature"][0]


class TestBayesianOptimizer:
    def test_simple_optimization(self):
        bounds = Bounds(np.array([-5.0]), np.array([5.0]))
        def fn(x):
            return -(x[0] - 2) ** 2 + 20
        opt = BayesianOptimizer(
            bounds=bounds, n_initial_points=5,
            max_iterations=30,
        )
        best_params, best_value, history = opt.optimize(fn, np.array([0.0]))
        assert best_value > 16

    def test_gp_predict(self):
        from acoustic_phase_optimizer.optimization.bayesian import GaussianProcess
        gp = GaussianProcess()
        X = np.array([[0.0], [1.0], [2.0], [3.0]])
        y = np.array([0.0, 1.0, 0.0, -1.0])
        gp.fit(X, y)
        mu, sigma = gp.predict(np.array([[1.5]]))
        assert len(mu) == 1
        assert np.isfinite(mu[0])
        assert sigma[0] >= 0

    def test_acquisition_functions(self):
        mu = np.array([0.0, 1.0, 2.0])
        sigma = np.array([1.0, 1.0, 1.0])
        ei = BayesianOptimizer._expected_improvement(mu, sigma, 1.0)
        assert len(ei) == 3
        ucb = BayesianOptimizer._upper_confidence_bound(mu, sigma)
        assert len(ucb) == 3
        pi = BayesianOptimizer._probability_of_improvement(mu, sigma, 1.0)
        assert len(pi) == 3


class TestOptimizationEngine:
    def test_optimize_gradient(self):
        engine = OptimizationEngine()
        def fn(x):
            return -np.sum(x ** 2)
        result = engine.optimize("gradient", fn, np.array([2.0, -1.0]))
        assert result.success
        assert "values" in result.history

    def test_optimize_genetic(self):
        engine = OptimizationEngine({"max_iterations": 50, "population_size": 30})
        def fn(x):
            return -np.sum(x ** 2) + 10
        result = engine.optimize("genetic", fn, np.array([1.0, -1.0]))
        assert result.success
        assert result.best_value > 5

    def test_optimize_annealing(self):
        engine = OptimizationEngine({"max_iterations": 100})
        def fn(x):
            return -(x[0] - 3) ** 2 + 10
        result = engine.optimize("annealing", fn, np.array([0.0]))
        assert result.success

    def test_optimize_bayesian(self):
        engine = OptimizationEngine({"max_iterations": 30})
        def fn(x):
            return -(x[0] - 2) ** 2 + 10
        result = engine.optimize("bayesian", fn, np.array([0.0]))
        assert result.success

    def test_compare_algorithms(self):
        engine = OptimizationEngine({"max_iterations": 30, "population_size": 20})
        def fn(x):
            return -np.sum(x ** 2) + 10
        results = engine.compare_algorithms(
            fn, np.array([1.0, -1.0]),
            algorithms=["gradient", "genetic"],
        )
        assert len(results) == 2

    def test_get_best_result(self):
        from acoustic_phase_optimizer.optimization.engine import OptimizationResult
        r1 = OptimizationResult("a", np.array([1.0]), 10.0, {}, True, 10)
        r2 = OptimizationResult("b", np.array([2.0]), 20.0, {}, True, 10)
        engine = OptimizationEngine()
        algo, result = engine.get_best_result({"a": r1, "b": r2})
        assert algo == "b"

    def test_invalid_algorithm(self):
        engine = OptimizationEngine()
        def fn(x):
            return 0.0
        with pytest.raises(ValueError):
            engine.optimize("invalid", fn, np.array([0.0]))

    def test_speaker_setup(self):
        engine = OptimizationEngine()
        bounds, initial = engine.setup_speaker_optimization(4)
        assert len(initial) == 12
        assert len(bounds.lower) == 12
