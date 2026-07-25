from __future__ import annotations

import numpy as np
from typing import Callable, List, Optional, Tuple
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle, Circle
import matplotlib.pyplot as plt

from acoustic_phase_optimizer.acoustic.room_model import RoomModel
from acoustic_phase_optimizer.acoustic.speaker import Speaker, SpeakerType
from acoustic_phase_optimizer.acoustic.microphone import Microphone

SPEAKER_COLORS = {
    SpeakerType.MAIN_LEFT: "#ff4444",
    SpeakerType.MAIN_RIGHT: "#4488ff",
    SpeakerType.CENTER: "#44ff44",
    SpeakerType.SUBWOOFER: "#8844ff",
    SpeakerType.FRONT_FILL: "#ff8844",
    SpeakerType.MONITOR: "#ff44ff",
    SpeakerType.DELAY: "#44ffff",
}

SPEAKER_MARKERS = {
    SpeakerType.MAIN_LEFT: "s",
    SpeakerType.MAIN_RIGHT: "s",
    SpeakerType.CENTER: "D",
    SpeakerType.SUBWOOFER: "^",
    SpeakerType.FRONT_FILL: "o",
    SpeakerType.MONITOR: "h",
    SpeakerType.DELAY: "P",
}


class RoomViewWidget(FigureCanvas):
    """2D top-down view with interactive stage, speaker placement, and context menus."""

    def __init__(self, parent=None, width: int = 6, height: int = 5, dpi: int = 100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        super().__init__(self.fig)
        self.setParent(parent)

        self.ax = self.fig.add_subplot(111)
        self.room_model: Optional[RoomModel] = None
        self.speakers: List[Speaker] = []
        self.microphones: List[Microphone] = []
        self.cancellation_data: Optional[Tuple] = None
        self.lidar_points: Optional[np.ndarray] = None

        self.stage_x = 0.0
        self.stage_y = 0.0
        self.stage_w = 24.0
        self.stage_d = 4.0
        self.stage_height = 1.0
        self.stage_visible = False

        self._pending_type: Optional[SpeakerType] = None
        self._drag_data: dict = {}
        self._on_speaker_placed: Optional[Callable[[str, SpeakerType, float, float, float], None]] = None
        self._on_speaker_right_click: Optional[Callable[[str, float, float], None]] = None
        self._on_speaker_moved: Optional[Callable[[str, float, float], None]] = None
        self._on_stage_changed: Optional[Callable[[float, float, float, float, float], None]] = None

        self.fig.tight_layout()

        self.mpl_connect("button_press_event", self._on_mouse_down)
        self.mpl_connect("button_release_event", self._on_mouse_up)
        self.mpl_connect("motion_notify_event", self._on_mouse_move)

    def set_stage_visible(self, visible: bool) -> None:
        self.stage_visible = visible
        self._draw()

    def set_on_speaker_placed(self, callback: Callable) -> None:
        self._on_speaker_placed = callback

    def set_on_speaker_right_click(self, callback: Callable) -> None:
        self._on_speaker_right_click = callback

    def set_on_speaker_moved(self, callback: Callable) -> None:
        self._on_speaker_moved = callback

    def set_on_stage_changed(self, callback: Callable) -> None:
        self._on_stage_changed = callback

    def set_pending_type(self, speaker_type: Optional[SpeakerType]) -> None:
        self._pending_type = speaker_type

    def set_lidar_points(self, points: Optional[np.ndarray]) -> None:
        self.lidar_points = points
        self._draw()

    def update_data(self, room_model=None, speakers=None, microphones=None) -> None:
        if room_model is not None:
            self.room_model = room_model
        if speakers is not None:
            self.speakers = speakers
        if microphones is not None:
            self.microphones = microphones
        self._draw()

    def set_cancellation_data(self, X, Y, Z) -> None:
        self.cancellation_data = (X, Y, Z)
        self._draw()

    def _is_on_stage_edge(self, x, y) -> Optional[str]:
        sx, sy = self.stage_x, self.stage_y
        sw, sd = self.stage_w, self.stage_d
        tol = 1.0
        if abs(x - sx) < tol and sy - sd / 2 < y < sy + sd / 2:
            return "left"
        if abs(x - (sx + sw)) < tol and sy - sd / 2 < y < sy + sd / 2:
            return "right"
        if abs(y - (sy - sd / 2)) < tol and sx < x < sx + sw:
            return "front"
        if abs(y - (sy + sd / 2)) < tol and sx < x < sx + sw:
            return "back"
        if sx < x < sx + sw and sy - sd / 2 < y < sy + sd / 2:
            return "inside"
        return None

    def _find_speaker_at(self, x, y) -> Optional[Speaker]:
        for spk in self.speakers:
            if not spk.enabled:
                continue
            if abs(spk.x - x) < 1.0 and abs(spk.y - y) < 1.0:
                return spk
        return None

    def _on_mouse_down(self, event) -> None:
        if not event.inaxes or event.inaxes != self.ax:
            return

        if event.button == 3:
            spk = self._find_speaker_at(event.xdata, event.ydata)
            if spk and self._on_speaker_right_click:
                self._on_speaker_right_click(spk.name, event.xdata, event.ydata)
            return

        if event.button != 1:
            return

        spk = self._find_speaker_at(event.xdata, event.ydata)
        if spk:
            self._drag_data = {
                "type": "speaker",
                "speaker": spk,
                "start_x": event.xdata,
                "start_y": event.ydata,
            }
            return

        edge = self._is_on_stage_edge(event.xdata, event.ydata)
        if edge:
            self._drag_data = {
                "type": "stage_" + edge,
                "start_x": event.xdata,
                "start_y": event.ydata,
                "orig_x": self.stage_x,
                "orig_y": self.stage_y,
                "orig_w": self.stage_w,
                "orig_d": self.stage_d,
            }
            return

        if self._pending_type and self._on_speaker_placed:
            z = 2.0
            self._on_speaker_placed("", self._pending_type, event.xdata, event.ydata, z)
            self._pending_type = None
            return

    def _on_mouse_up(self, event) -> None:
        if not self._drag_data:
            return
        if self._drag_data["type"] == "speaker":
            spk = self._drag_data["speaker"]
            if self._on_speaker_moved:
                self._on_speaker_moved(spk.name, spk.x, spk.y)
        elif self._on_stage_changed:
            self._on_stage_changed(
                self.stage_x, self.stage_y, self.stage_w, self.stage_d, self.stage_height
            )
        self._drag_data = {}

    def _on_mouse_move(self, event) -> None:
        if not self._drag_data or not event.inaxes:
            return
        dx = event.xdata - self._drag_data["start_x"]
        dy = event.ydata - self._drag_data["start_y"]
        drag_type = self._drag_data["type"]

        if drag_type == "speaker":
            spk = self._drag_data["speaker"]
            spk.x = float(event.xdata)
            spk.y = float(event.ydata)
        elif drag_type == "stage_inside":
            self.stage_x = self._drag_data["orig_x"] + dx
            self.stage_y = self._drag_data["orig_y"] + dy
        elif drag_type == "stage_left":
            self.stage_x = self._drag_data["orig_x"] + dx
            self.stage_w = self._drag_data["orig_w"] - dx
        elif drag_type == "stage_right":
            self.stage_w = self._drag_data["orig_w"] + dx
        elif drag_type == "stage_front":
            self.stage_y = self._drag_data["orig_y"] + dy
            self.stage_d = self._drag_data["orig_d"] - dy
        elif drag_type == "stage_back":
            self.stage_d = self._drag_data["orig_d"] + dy

        self._draw()

    def _draw(self) -> None:
        self.ax.clear()

        if self.lidar_points is not None and len(self.lidar_points) > 0:
            self.ax.scatter(
                self.lidar_points[:, 0], self.lidar_points[:, 1],
                s=0.5, c="#888888", alpha=0.3, zorder=1,
            )

        if self.room_model is not None:
            L, W, _ = self.room_model.get_dimensions_array()
            rect = Rectangle(
                (-L / 2, -W / 2), L, W,
                linewidth=2, edgecolor="black", facecolor="#f0f0f0", alpha=0.3,
            )
            self.ax.add_patch(rect)
            self.ax.set_xlim(-L / 2 - 2, L / 2 + 2)
            self.ax.set_ylim(-W / 2 - 2, W / 2 + 2)
        else:
            self.ax.set_xlim(-17, 17)
            self.ax.set_ylim(-12, 12)

        if self.stage_visible:
            stage_rect = Rectangle(
                (self.stage_x, self.stage_y - self.stage_d / 2),
                self.stage_w, self.stage_d,
                linewidth=2, edgecolor="#8B4513", facecolor="#D2B48C", alpha=0.6,
                linestyle="-", zorder=2,
            )
            self.ax.add_patch(stage_rect)
            self.ax.text(
                self.stage_x + self.stage_w / 2, self.stage_y,
                f"STAGE\n{self.stage_height:.1f}m", ha="center", va="center",
                fontsize=9, color="#8B4513", alpha=0.8, fontweight="bold",
            )

        if self.cancellation_data is not None:
            X, Y, Z = self.cancellation_data
            self.ax.contourf(X, Y, Z, levels=20, cmap="RdBu_r", alpha=0.5)
            self.ax.contour(X, Y, Z, levels=[-0.5, 0.5], colors="black", linewidths=0.5, alpha=0.3)

        for spk in self.speakers:
            if not spk.enabled:
                continue
            color = SPEAKER_COLORS.get(spk.speaker_type, "#888888")
            marker = SPEAKER_MARKERS.get(spk.speaker_type, "s")
            self.ax.plot(
                spk.x, spk.y,
                marker=marker, markersize=14,
                color=color, markeredgecolor="black", markeredgewidth=1.5,
                zorder=5, pickradius=5,
            )
            self.ax.annotate(
                f"{spk.name}\n({spk.z:.1f}m)",
                (spk.x, spk.y),
                xytext=(8, 8), textcoords="offset points",
                fontsize=7, fontweight="bold",
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

        if self._pending_type:
            self.ax.set_title(
                f"Room Layout — Click to place {self._pending_type.value} | Right-click for options",
                fontsize=10,
            )
        else:
            self.ax.set_title("Room Layout — Double-click to place speaker | Drag stage edges", fontsize=10)

        self.ax.set_xlabel("Width (m)")
        self.ax.set_ylabel("Depth (m)")
        self.ax.grid(True, alpha=0.3)
        self.ax.set_aspect("equal")

        self.fig.tight_layout()
        self.draw()


class Room3DViewWidget(FigureCanvas):
    """3D view of the room with speaker coverage cones and LIDAR overlay."""

    def __init__(self, parent=None, width: int = 6, height: int = 5, dpi: int = 100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        super().__init__(self.fig)
        self.setParent(parent)
        self.ax = self.fig.add_subplot(111, projection="3d")
        self.room_model: Optional[RoomModel] = None
        self.speakers: List[Speaker] = []
        self.microphones: List[Microphone] = []
        self.lidar_points: Optional[np.ndarray] = None

    def set_lidar_points(self, points: Optional[np.ndarray]) -> None:
        self.lidar_points = points
        self._draw()

    def update_data(self, room_model=None, speakers=None, microphones=None) -> None:
        if room_model is not None:
            self.room_model = room_model
        if speakers is not None:
            self.speakers = speakers
        if microphones is not None:
            self.microphones = microphones
        self._draw()

    def _draw(self) -> None:
        self.ax.clear()

        if self.lidar_points is not None and len(self.lidar_points) > 0:
            step = max(1, len(self.lidar_points) // 10000)
            self.ax.scatter(
                self.lidar_points[::step, 0],
                self.lidar_points[::step, 1],
                self.lidar_points[::step, 2],
                s=0.5, c="#888888", alpha=0.2, zorder=1,
            )

        if self.room_model is not None:
            L, W, H = self.room_model.get_dimensions_array()
            self._draw_room_box(L, W, H)
            self.ax.set_xlim(-L / 2 - 1, L / 2 + 1)
            self.ax.set_ylim(-W / 2 - 1, W / 2 + 1)
            self.ax.set_zlim(0, H + 1)

        for spk in self.speakers:
            if not spk.enabled:
                continue
            color = SPEAKER_COLORS.get(spk.speaker_type, "#888888")
            pos = spk.position
            self.ax.scatter(
                pos[0], pos[1], pos[2],
                marker="s", s=100, color=color, edgecolors="black",
                linewidths=1.5, zorder=5,
            )

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
