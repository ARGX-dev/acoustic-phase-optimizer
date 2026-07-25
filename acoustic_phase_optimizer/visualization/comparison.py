"""Before/after comparison widget and algorithm convergence viewer."""

from __future__ import annotations

import numpy as np
from typing import Optional, Dict, List

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure


class BeforeAfterWidget(FigureCanvas):
    """Side-by-side before/after heatmap with shared colorbar and difference panel."""

    def __init__(self, parent=None, width: int = 10, height: int = 5, dpi: int = 100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        super().__init__(self.fig)
        self.setParent(parent)

        self.ax_before = self.fig.add_subplot(1, 3, 1)
        self.ax_after = self.fig.add_subplot(1, 3, 2, sharey=self.ax_before)
        self.ax_diff = self.fig.add_subplot(1, 3, 3)

        self._colorbar = None

    def update_data(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        Z_before: np.ndarray,
        Z_after: np.ndarray,
        vmin: Optional[float] = None,
        vmax: Optional[float] = None,
    ) -> None:
        if self._colorbar is not None:
            self._colorbar.remove()
            self._colorbar = None
        for ax in [self.ax_before, self.ax_after, self.ax_diff]:
            ax.clear()

        common = dict(cmap="plasma", levels=50)
        if vmin is not None and vmax is not None:
            common["vmin"] = vmin
            common["vmax"] = vmax

        c1 = self.ax_before.contourf(X, Y, Z_before, **common)
        self.ax_before.set_title("Before Alignment", fontsize=10)
        self.ax_before.set_xlabel("Width (m)")
        self.ax_before.set_ylabel("Depth (m)")
        self.ax_before.set_aspect("equal")

        self.ax_after.contourf(X, Y, Z_after, **common)
        self.ax_after.set_title("After Optimization", fontsize=10)
        self.ax_after.set_xlabel("Width (m)")
        self.ax_after.set_aspect("equal")

        Z_diff = Z_after - Z_before
        diff_max = max(abs(np.min(Z_diff)), abs(np.max(Z_diff))) or 1.0
        self.ax_diff.contourf(X, Y, Z_diff, levels=50, cmap="RdBu_r", vmin=-diff_max, vmax=diff_max)
        self.ax_diff.set_title("Improvement (dB)", fontsize=10)
        self.ax_diff.set_xlabel("Width (m)")
        self.ax_diff.set_aspect("equal")

        self._colorbar = self.fig.colorbar(c1, ax=[self.ax_before, self.ax_after, self.ax_diff],
                                           shrink=0.6, pad=0.05)
        self._colorbar.set_label("SPL (dB)")
        self.fig.subplots_adjust(left=0.06, right=0.88, wspace=0.25)
        self.draw()


class AlgorithmComparisonWidget(FigureCanvas):
    """Convergence curves + final score bar chart for all algorithms."""

    def __init__(self, parent=None, width: int = 8, height: int = 5, dpi: int = 100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        super().__init__(self.fig)
        self.setParent(parent)

        self.ax_curve = self.fig.add_subplot(1, 2, 1)
        self.ax_bar = self.fig.add_subplot(1, 2, 2)

    def update_results(self, results: Dict[str, dict]) -> None:
        self.ax_curve.clear()
        self.ax_bar.clear()

        colors = {"genetic": "#34d6c0", "gradient": "#ff5462",
                  "annealing": "#ffb020", "bayesian": "#4488ff"}

        names = []
        scores = []
        for algo_key, r in sorted(results.items(), key=lambda kv: -kv[1].get("best_value", 0)):
            history = r.get("history", {})
            values = history.get("best_values", history.get("values", []))
            if values:
                self.ax_curve.plot(values, label=algo_key.capitalize(),
                                   color=colors.get(algo_key, "#888888"), linewidth=1.5)
            names.append(algo_key.capitalize())
            scores.append(r.get("best_value", 0))

        self.ax_curve.set_title("Convergence", fontsize=10)
        self.ax_curve.set_xlabel("Iteration")
        self.ax_curve.set_ylabel("Objective Score")
        self.ax_curve.legend(fontsize=8)
        self.ax_curve.grid(True, alpha=0.3)

        bar_colors = [colors.get(k.lower(), "#888888") for k in names]
        self.ax_bar.barh(range(len(names)), scores, color=bar_colors, height=0.6)
        self.ax_bar.set_yticks(range(len(names)))
        self.ax_bar.set_yticklabels(names, fontsize=9)
        self.ax_bar.set_title("Final Score", fontsize=10)
        self.ax_bar.invert_yaxis()
        for i, v in enumerate(scores):
            self.ax_bar.text(v + 0.01 * max(scores or [1]), i, f"{v:.4f}",
                             va="center", fontsize=8)

        self.fig.subplots_adjust(left=0.08, right=0.95, wspace=0.35)
        self.draw()
