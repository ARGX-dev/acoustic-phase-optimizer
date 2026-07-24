"""Heat map visualization for SPL, phase, delay, and cancellation zones."""

from __future__ import annotations

import numpy as np
from typing import Optional, Tuple
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt


class HeatmapWidget(FigureCanvas):
    """Contour/heatmap display for acoustic spatial data."""

    def __init__(self, parent=None, width: int = 5, height: int = 4, dpi: int = 100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        super().__init__(self.fig)
        self.setParent(parent)
        self.ax = self.fig.add_subplot(111)

        self._data: Optional[np.ndarray] = None
        self._X: Optional[np.ndarray] = None
        self._Y: Optional[np.ndarray] = None
        self._title: str = "Heatmap"
        self._colormap: str = "inferno"

    def update_data(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        data: np.ndarray,
        title: str = "Heatmap",
        colormap: str = "inferno",
        vmin: Optional[float] = None,
        vmax: Optional[float] = None,
    ) -> None:
        self._X = X
        self._Y = Y
        self._data = data
        self._title = title
        self._colormap = colormap
        self._draw(vmin, vmax)

    def _draw(
        self,
        vmin: Optional[float] = None,
        vmax: Optional[float] = None,
    ) -> None:
        self.ax.clear()

        if self._data is not None and self._X is not None and self._Y is not None:
            contour = self.ax.contourf(
                self._X, self._Y, self._data,
                levels=50, cmap=self._colormap,
                vmin=vmin, vmax=vmax,
            )
            self.fig.colorbar(contour, ax=self.ax, shrink=0.8)

        self.ax.set_xlabel("Width (m)")
        self.ax.set_ylabel("Depth (m)")
        self.ax.set_title(self._title)
        self.ax.set_aspect("equal")
        self.ax.grid(True, alpha=0.2)

        self.fig.tight_layout()
        self.draw()

    def set_spl_heatmap(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        spl_db: np.ndarray,
    ) -> None:
        self.update_data(
            X, Y, spl_db,
            title="SPL Distribution (dB)",
            colormap="plasma",
        )

    def set_phase_heatmap(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        phase_rad: np.ndarray,
    ) -> None:
        self.update_data(
            X, Y, phase_rad,
            title="Phase Distribution (rad)",
            colormap="RdBu_r",
            vmin=-np.pi, vmax=np.pi,
        )

    def set_delay_heatmap(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        delay_ms: np.ndarray,
    ) -> None:
        self.update_data(
            X, Y, delay_ms,
            title="Delay Distribution (ms)",
            colormap="viridis",
        )

    def set_cancellation_heatmap(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        cancellation: np.ndarray,
    ) -> None:
        self.update_data(
            X, Y, cancellation,
            title="Phase Cancellation Zones",
            colormap="RdBu_r",
            vmin=-1, vmax=1,
        )


class MultiHeatmapWidget(FigureCanvas):
    """Multi-panel heatmap display for side-by-side comparison."""

    def __init__(self, parent=None, width: int = 10, height: int = 6, dpi: int = 100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        super().__init__(self.fig)
        self.setParent(parent)

        self.axes = [
            self.fig.add_subplot(2, 3, 1),
            self.fig.add_subplot(2, 3, 2),
            self.fig.add_subplot(2, 3, 3),
            self.fig.add_subplot(2, 3, 4),
            self.fig.add_subplot(2, 3, 5),
            self.fig.add_subplot(2, 3, 6),
        ]

        self._clear_axes()

    def _clear_axes(self) -> None:
        for ax in self.axes:
            ax.clear()

    def update_all(
        self,
        spl_data: Optional[Tuple] = None,
        phase_data: Optional[Tuple] = None,
        delay_data: Optional[Tuple] = None,
        cancellation_data: Optional[Tuple] = None,
        coherence_data: Optional[Tuple] = None,
        freq_response_data: Optional[Tuple] = None,
    ) -> None:
        self._clear_axes()

        titles = ["SPL (dB)", "Phase (rad)", "Delay (ms)",
                  "Cancellation", "Coherence", "Freq Response"]
        datasets = [spl_data, phase_data, delay_data,
                    cancellation_data, coherence_data, freq_response_data]
        cmaps = ["plasma", "RdBu_r", "viridis", "RdBu_r", "Greens", "magma"]

        for ax, title, data, cmap in zip(self.axes, titles, datasets, cmaps):
            if data is not None:
                X, Y, Z = data
                contour = ax.contourf(X, Y, Z, levels=50, cmap=cmap)
                self.fig.colorbar(contour, ax=ax, shrink=0.7)
            ax.set_title(title)
            ax.set_xlabel("Width (m)")
            ax.set_ylabel("Depth (m)")
            ax.set_aspect("equal")

        self.fig.tight_layout()
        self.draw()
