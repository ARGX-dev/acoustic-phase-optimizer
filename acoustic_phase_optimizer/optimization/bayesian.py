"""Bayesian optimization for acoustic parameter optimization using Gaussian Processes."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from scipy.special import erf
from typing import Callable, Dict, List, Optional, Tuple
from acoustic_phase_optimizer.optimization.constraints import Bounds
from acoustic_phase_optimizer.utils.logging import get_logger

logger = get_logger(__name__)


class GaussianProcess:
    """Simple Gaussian Process regression for Bayesian optimization."""

    def __init__(self, length_scale: float = 1.0, sigma_f: float = 1.0, sigma_n: float = 1e-6):
        self.length_scale = length_scale
        self.sigma_f = sigma_f
        self.sigma_n = sigma_n
        self.X_train: Optional[NDArray[np.float64]] = None
        self.y_train: Optional[NDArray[np.float64]] = None

    def fit(self, X: NDArray[np.float64], y: NDArray[np.float64]) -> None:
        self.X_train = np.asarray(X, dtype=np.float64)
        self.y_train = np.asarray(y, dtype=np.float64)

    def predict(
        self,
        X_test: NDArray[np.float64],
    ) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
        if self.X_train is None or len(self.X_train) == 0:
            return np.zeros(len(X_test)), np.ones(len(X_test)) * self.sigma_f

        K = self._rbf_kernel(self.X_train, self.X_train)
        K += self.sigma_n ** 2 * np.eye(len(self.X_train))

        K_s = self._rbf_kernel(self.X_train, X_test)
        K_ss = self._rbf_kernel(X_test, X_test)

        try:
            L = np.linalg.cholesky(K)
            alpha = np.linalg.solve(L.T, np.linalg.solve(L, self.y_train))
        except np.linalg.LinAlgError:
            K_reg = K + 1e-6 * np.eye(len(K))
            L = np.linalg.cholesky(K_reg)
            alpha = np.linalg.solve(L.T, np.linalg.solve(L, self.y_train))

        mu = K_s.T @ alpha

        v = np.linalg.solve(L, K_s)
        var = np.diag(K_ss) - np.sum(v ** 2, axis=0)

        var = np.maximum(var, 0.0)
        return mu, np.sqrt(var)

    def _rbf_kernel(
        self,
        X1: NDArray[np.float64],
        X2: NDArray[np.float64],
    ) -> NDArray[np.float64]:
        sqdist = np.sum(X1 ** 2, axis=1, keepdims=True) + \
                 np.sum(X2 ** 2, axis=1) - 2.0 * X1 @ X2.T
        return self.sigma_f ** 2 * np.exp(-0.5 / self.length_scale ** 2 * sqdist)


class BayesianOptimizer:
    """Bayesian optimization using Gaussian Process surrogate model."""

    def __init__(
        self,
        bounds: Optional[Bounds] = None,
        n_initial_points: int = 10,
        max_iterations: int = 500,
        acquisition: str = "expected_improvement",
        exploration_weight: float = 0.01,
        convergence_threshold: float = 1e-6,
    ):
        self.bounds = bounds
        self.n_initial_points = n_initial_points
        self.max_iterations = max_iterations
        self.acquisition = acquisition
        self.exploration_weight = exploration_weight
        self.convergence_threshold = convergence_threshold
        self.gp = GaussianProcess()

    def optimize(
        self,
        objective_fn: Callable[[NDArray[np.float64]], float],
        initial_params: NDArray[np.float64],
    ) -> Tuple[NDArray[np.float64], float, Dict]:
        n_params = len(initial_params)
        X_sample = []
        y_sample = []

        X_sample.append(initial_params.copy())
        y_sample.append(objective_fn(initial_params))

        if self.bounds is not None:
            random_points = np.random.uniform(
                self.bounds.lower,
                self.bounds.upper,
                size=(self.n_initial_points - 1, n_params),
            )
        else:
            random_points = initial_params + np.random.randn(self.n_initial_points - 1, n_params) * 0.5

        for x in random_points:
            if self.bounds is not None:
                x = self.bounds.clip(x)
            X_sample.append(x)
            y_sample.append(objective_fn(x))

        X_sample = np.array(X_sample)
        y_sample = np.array(y_sample)

        best_idx = np.argmax(y_sample)
        best_params = X_sample[best_idx].copy()
        best_value = float(y_sample[best_idx])

        history = {
            "values": [best_value],
            "n_samples": len(X_sample),
            "best_params": [best_params.copy()],
        }

        for iteration in range(self.max_iterations):
            self.gp.fit(X_sample, y_sample)

            x_candidate = self._optimize_acquisition(n_params)
            if self.bounds is not None:
                x_candidate = self.bounds.clip(x_candidate)

            y_candidate = objective_fn(x_candidate)

            X_sample = np.vstack([X_sample, x_candidate.reshape(1, -1)])
            y_sample = np.append(y_sample, y_candidate)

            if y_candidate > best_value:
                best_value = y_candidate
                best_params = x_candidate.copy()

            history["values"].append(best_value)
            history["n_samples"] = len(X_sample)
            history["best_params"].append(best_params.copy())

            if iteration > 10 and self._check_convergence(y_sample):
                logger.info(f"Bayesian optimization converged at iteration {iteration}")
                break

            if iteration % 50 == 0:
                logger.debug(
                    f"Bayesian iter {iteration}: best = {best_value:.6f}, "
                    f"samples = {len(X_sample)}"
                )

        history["iterations"] = iteration + 1
        history["total_samples"] = len(X_sample)
        history["X_samples"] = X_sample
        history["y_samples"] = y_sample

        return best_params, best_value, history

    def _optimize_acquisition(self, n_params: int) -> NDArray[np.float64]:
        if self.bounds is not None:
            candidates = np.random.uniform(
                self.bounds.lower,
                self.bounds.upper,
                size=(1000, n_params),
            )
        else:
            candidates = np.random.randn(1000, n_params)

        mu, sigma = self.gp.predict(candidates)
        y_best = np.max(self.gp.y_train) if self.gp.y_train is not None and len(self.gp.y_train) > 0 else 0.0

        if self.acquisition == "expected_improvement":
            acq = self._expected_improvement(mu, sigma, y_best)
        elif self.acquisition == "upper_confidence_bound":
            acq = self._upper_confidence_bound(mu, sigma)
        elif self.acquisition == "probability_of_improvement":
            acq = self._probability_of_improvement(mu, sigma, y_best)
        else:
            acq = self._expected_improvement(mu, sigma, y_best)

        best_idx = np.argmax(acq)
        return candidates[best_idx]

    @staticmethod
    def _expected_improvement(
        mu: NDArray[np.float64],
        sigma: NDArray[np.float64],
        y_best: float,
    ) -> NDArray[np.float64]:
        imp = mu - y_best
        z = imp / (sigma + 1e-12)
        ei = imp * _norm_cdf(z) + sigma * _norm_pdf(z)
        return ei

    @staticmethod
    def _upper_confidence_bound(
        mu: NDArray[np.float64],
        sigma: NDArray[np.float64],
        kappa: float = 2.0,
    ) -> NDArray[np.float64]:
        return mu + kappa * sigma

    @staticmethod
    def _probability_of_improvement(
        mu: NDArray[np.float64],
        sigma: NDArray[np.float64],
        y_best: float,
    ) -> NDArray[np.float64]:
        z = (mu - y_best) / (sigma + 1e-12)
        return _norm_cdf(z)

    @staticmethod
    def _check_convergence(y_sample: NDArray[np.float64], window: int = 10) -> bool:
        if len(y_sample) < window:
            return False
        recent = y_sample[-window:]
        return np.std(recent) < 1e-6


def _norm_cdf(x: NDArray[np.float64]) -> NDArray[np.float64]:
    return 0.5 * (1.0 + erf(x / np.sqrt(2.0)))


def _norm_pdf(x: NDArray[np.float64]) -> NDArray[np.float64]:
    return np.exp(-0.5 * x ** 2) / np.sqrt(2.0 * np.pi)
