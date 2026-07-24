"""Room layout visualization widget."""

from __future__ import annotations

import numpy as np
from typing import List, Optional, Tuple
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle, Circle
import matplotlib.pyplot as plt

from acoustic_phase_optimizer.acoustic.room_model import RoomModel
from acoustic_phase_optimizer.acoustic.speaker import Speaker
from acoustic_phase_optimizer.acoustic.microphone import Microphone


class RoomViewWidget(FigureCanvas):
    """2D top-down view of the room with speakers and microphones."""

    def __init__(self, parent=None, width: int = 6, height: int = 5, dpi: int = 100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        super().__init__(self.fig)
        self.setParent(parent)

        self.ax = self.fig.add_subplot(111)
        self.room_model: Optional[RoomModel] = None
        self.speakers: List[Speaker] = []
        self.microphones: List[Microphone] = []
        self.cancellation_data: Optional[Tuple] = None

        self.fig.tight_layout()

    def update_data(
        self,
        room_model: Optional[RoomModel] = None,
        speakers: Optional[List[Speaker]] = None,
        microphones: Optional[List[Microphone]] = None,
    ) -> None:
        if room_model is not None:
            self.room_model = room_model
        if speakers is not None:
            self.speakers = speakers
        if microphones is not None:
            self.microphones = microphones
        self._draw()

    def set_cancellation_data(
        self,
        X: np.ndarray,
        Y: np.ndarray,
        Z: np.ndarray,
    ) -> None:
        self.cancellation_data = (X, Y, Z)
        self._draw()

    def _draw(self) -> None:
        self.ax.clear()

        if self.room_model is not None:
            L, W, _ = self.room_model.get_dimensions_array()
            rect = Rectangle(
                (-L / 2, -W / 2), L, W,
                linewidth=2, edgecolor="black", facecolor="#f0f0f0", alpha=0.3,
            )
            self.ax.add_patch(rect)
            self.ax.set_xlim(-L / 2 - 1, L / 2 + 1)
            self.ax.set_ylim(-W / 2 - 1, W / 2 + 1)
        else:
            self.ax.set_xlim(-15, 15)
            self.ax.set_ylim(-10, 10)

        if self.cancellation_data is not None:
            X, Y, Z = self.cancellation_data
            self.ax.contourf(X, Y, Z, levels=20, cmap="RdBu_r", alpha=0.5)
            self.ax.contour(X, Y, Z, levels=[-0.5, 0.5], colors="black", linewidths=0.5, alpha=0.3)

        colors = {
            "left_main": "#ff4444",
            "right_main": "#4488ff",
            "center": "#44ff44",
            "subwoofer": "#8844ff",
            "front_fill": "#ff8844",
            "monitor": "#ff44ff",
            "delay": "#44ffff",
        }

        for speaker in self.speakers:
            if not speaker.enabled:
                continue
            color = colors.get(speaker.speaker_type.value, "#888888")
            self.ax.plot(
                speaker.x, speaker.y,
                marker="s", markersize=12,
                color=color, markeredgecolor="black", markeredgewidth=1.5,
                zorder=5,
            )
            self.ax.annotate(
                speaker.name,
                (speaker.x, speaker.y),
                xytext=(5, 5), textcoords="offset points",
                fontsize=8, fontweight="bold",
            )

        for mic in self.microphones:
            if not mic.enabled:
                continue
            self.ax.plot(
                mic.x, mic.y,
                marker="o", markersize=8,
                color="green", markeredgecolor="black", markeredgewidth=1,
                zorder=5,
            )
            self.ax.annotate(
                mic.name,
                (mic.x, mic.y),
                xytext=(5, -10), textcoords="offset points",
                fontsize=7,
            )

        stage_rect = Rectangle(
            (-12, -2), 24, 4,
            linewidth=1, edgecolor="#666666", facecolor="#dddddd", alpha=0.5,
            linestyle="--",
        )
        self.ax.add_patch(stage_rect)
        self.ax.text(0, 0, "STAGE", ha="center", va="center", fontsize=10,
                    color="#666666", alpha=0.7)

        self.ax.set_xlabel("Width (m)")
        self.ax.set_ylabel("Depth (m)")
        self.ax.set_title("Room Layout")
        self.ax.grid(True, alpha=0.3)
        self.ax.set_aspect("equal")

        self.fig.tight_layout()
        self.draw()


class Room3DViewWidget(FigureCanvas):
    """3D view of the room with speaker coverage cones."""

    def __init__(self, parent=None, width: int = 6, height: int = 5, dpi: int = 100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        super().__init__(self.fig)
        self.setParent(parent)
        self.ax = self.fig.add_subplot(111, projection="3d")
        self.room_model: Optional[RoomModel] = None
        self.speakers: List[Speaker] = []
        self.microphones: List[Microphone] = []

    def update_data(
        self,
        room_model: Optional[RoomModel] = None,
        speakers: Optional[List[Speaker]] = None,
        microphones: Optional[List[Microphone]] = None,
    ) -> None:
        if room_model is not None:
            self.room_model = room_model
        if speakers is not None:
            self.speakers = speakers
        if microphones is not None:
            self.microphones = microphones
        self._draw()

    def _draw(self) -> None:
        self.ax.clear()

        if self.room_model is not None:
            L, W, H = self.room_model.get_dimensions_array()
            self._draw_room_box(L, W, H)
            self.ax.set_xlim(-L / 2 - 1, L / 2 + 1)
            self.ax.set_ylim(-W / 2 - 1, W / 2 + 1)
            self.ax.set_zlim(0, H + 1)

        colors = {
            "left_main": "#ff4444",
            "right_main": "#4488ff",
            "center": "#44ff44",
            "subwoofer": "#8844ff",
            "front_fill": "#ff8844",
            "monitor": "#ff44ff",
            "delay": "#44ffff",
        }

        for speaker in self.speakers:
            if not speaker.enabled:
                continue
            color = colors.get(speaker.speaker_type.value, "#888888")
            pos = speaker.position
            self.ax.scatter(
                pos[0], pos[1], pos[2],
                marker="s", s=100, color=color, edgecolors="black",
                linewidths=1.5, zorder=5,
            )
            self._draw_coverage_cone(speaker, color)

        for mic in self.microphones:
            if not mic.enabled:
                continue
            pos = mic.position
            self.ax.scatter(
                pos[0], pos[1], pos[2],
                marker="o", s=60, color="green", edgecolors="black",
                linewidths=1, zorder=5,
            )

        self.ax.set_xlabel("X (m)")
        self.ax.set_ylabel("Y (m)")
        self.ax.set_zlabel("Z (m)")
        self.ax.set_title("3D Room View")
        self.ax.view_init(elev=30, azim=45)

        self.fig.tight_layout()
        self.draw()

    def _draw_room_box(self, L: float, W: float, H: float) -> None:
        x = [-L / 2, L / 2, L / 2, -L / 2, -L / 2]
        y = [-W / 2, -W / 2, W / 2, W / 2, -W / 2]
        z_bottom = [0, 0, 0, 0, 0]
        z_top = [H, H, H, H, H]

        self.ax.plot(x, y, z_bottom, color="black", alpha=0.5)
        self.ax.plot(x, y, z_top, color="black", alpha=0.5)

        for i in range(4):
            self.ax.plot(
                [x[i], x[i]], [y[i], y[i]], [0, H],
                color="black", alpha=0.5,
            )

    def _draw_coverage_cone(self, speaker: Speaker, color: str) -> None:
        pass
