"""Frequency response and phase visualization widgets."""

from __future__ import annotations

import numpy as np
from typing import Dict, List, Optional, Tuple
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import matplotlib.pyplot as plt


class FrequencyResponseWidget(FigureCanvas):
    """Single-axis frequency response with phase overlay and toggleable legend."""

    def __init__(self, parent=None, width: int = 6, height: int = 4, dpi: int = 100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        super().__init__(self.fig)
        self.setParent(parent)

        self.ax = self.fig.add_subplot(111)
        self.ax.set_xlim(20, 20000)
        self.ax.set_ylim(-60, 10)
        self._phase_ax = None
        self._traces: Dict[str, dict] = {}
        self._show_legend = False

    def toggle_legend(self) -> None:
        self._show_legend = not self._show_legend
        self._draw()

    def update_data(
        self,
        trace_name: str,
        freqs: np.ndarray,
        magnitude_db: np.ndarray,
        phase_rad: Optional[np.ndarray] = None,
        color: Optional[str] = None,
        linestyle: str = "-",
    ) -> None:
        self._traces[trace_name] = {
            "freqs": freqs,
            "magnitude_db": magnitude_db,
            "phase_rad": phase_rad,
            "color": color,
            "linestyle": linestyle,
        }
        self._draw()

    def remove_trace(self, trace_name: str) -> None:
        self._traces.pop(trace_name, None)
        self._draw()

    def clear_traces(self) -> None:
        self._traces.clear()
        self._draw()

    def _draw(self) -> None:
        self.ax.clear()

        colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
                  "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"]

        has_phase = any(t["phase_rad"] is not None for t in self._traces.values())

        if has_phase and self._phase_ax is None:
            self._phase_ax = self.ax.twinx()
        elif not has_phase and self._phase_ax is not None:
            self._phase_ax.remove()
            self._phase_ax = None
        elif has_phase and self._phase_ax is not None:
            self._phase_ax.clear()

        for i, (name, trace) in enumerate(self._traces.items()):
            color = trace["color"] or colors[i % len(colors)]
            freqs = trace["freqs"]
            magnitude_db = trace["magnitude_db"]

            valid = (freqs >= 20) & np.isfinite(magnitude_db)
            if not np.any(valid):
                continue
            self.ax.semilogx(
                freqs[valid], magnitude_db[valid],
                color=color, linestyle=trace["linestyle"],
                label=name, linewidth=1.5,
            )

            if trace["phase_rad"] is not None and self._phase_ax is not None:
                phase = trace["phase_rad"]
                valid_p = (freqs >= 20) & np.isfinite(phase)
                if np.any(valid_p):
                    self._phase_ax.semilogx(
                        freqs[valid_p], phase[valid_p],
                        color=color, linestyle="--", linewidth=0.8, alpha=0.5,
                    )

        self.ax.set_xlim(20, 20000)
        self.ax.set_ylabel("Magnitude (dB)")
        self.ax.set_title("Frequency Response")
        self.ax.grid(True, alpha=0.3, which="both")
        self.ax.set_ylim(-60, 10)
        self.ax.set_xlabel("Frequency (Hz)")
        self.ax.tick_params(labelsize=7)

        if self._phase_ax is not None:
            self._phase_ax.set_ylabel("Phase (rad)", fontsize=8)
            self._phase_ax.set_ylim(-np.pi, np.pi)
            self._phase_ax.tick_params(labelsize=7)

        if self._show_legend and self._traces:
            self.ax.legend(loc="best", fontsize=6)

        self.fig.subplots_adjust(left=0.12, right=0.88, top=0.92, bottom=0.12)
        self.draw()


class GroupDelayWidget(FigureCanvas):
    """Group delay visualization."""

    def __init__(self, parent=None, width: int = 6, height: int = 3, dpi: int = 100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        super().__init__(self.fig)
        self.setParent(parent)
        self.ax = self.fig.add_subplot(111)

    def update_data(
        self,
        freqs: np.ndarray,
        group_delay: np.ndarray,
        label: str = "Group Delay",
    ) -> None:
        self.ax.clear()

        valid = freqs > 0
        self.ax.semilogx(freqs[valid], group_delay[valid] * 1000, linewidth=1.5)

        self.ax.set_xlabel("Frequency (Hz)")
        self.ax.set_ylabel("Group Delay (ms)")
        self.ax.set_title(label)
        self.ax.grid(True, alpha=0.3, which="both")
        self.ax.set_xlim(20, 20000)

        self.fig.tight_layout()
        self.draw()


class SpectrogramWidget(FigureCanvas):
    """Spectrogram display for time-frequency analysis."""

    def __init__(self, parent=None, width: int = 6, height: int = 4, dpi: int = 100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        super().__init__(self.fig)
        self.setParent(parent)
        self.ax = self.fig.add_subplot(111)

    def update_data(
        self,
        times: np.ndarray,
        freqs: np.ndarray,
        spectrogram: np.ndarray,
        title: str = "Spectrogram",
    ) -> None:
        self.ax.clear()

        spec_db = 20.0 * np.log10(spectrogram + 1e-12)
        self.ax.pcolormesh(times, freqs, spec_db, shading="gouraud", cmap="inferno")

        self.ax.set_ylim(20, 20000)
        self.ax.set_yscale("log")
        self.ax.set_xlabel("Time (s)")
        self.ax.set_ylabel("Frequency (Hz)")
        self.ax.set_title(title)

        self.fig.tight_layout()
        self.draw()
