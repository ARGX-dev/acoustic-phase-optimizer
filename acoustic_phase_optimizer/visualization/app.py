"""Main PyQt6 visualization application."""

from __future__ import annotations

import sys
import numpy as np
from typing import Dict, List, Optional, Tuple
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QSplitter,
    QVBoxLayout, QHBoxLayout, QTabWidget, QMessageBox,
)
from PyQt6.QtCore import Qt, QTimer

from acoustic_phase_optimizer.config import Config
from acoustic_phase_optimizer.acoustic.room_model import RoomModel
from acoustic_phase_optimizer.acoustic.speaker import Speaker, SpeakerType, SpeakerPolarity
from acoustic_phase_optimizer.acoustic.microphone import Microphone
from acoustic_phase_optimizer.acoustic.comb_filter import CombFilterDetector
from acoustic_phase_optimizer.simulation.virtual_room import VirtualRoom
from acoustic_phase_optimizer.optimization.engine import OptimizationEngine
from acoustic_phase_optimizer.optimization.objectives import ObjectiveFunction
from acoustic_phase_optimizer.visualization.room_view import RoomViewWidget, Room3DViewWidget
from acoustic_phase_optimizer.visualization.heatmap import HeatmapWidget, MultiHeatmapWidget
from acoustic_phase_optimizer.visualization.frequency_view import (
    FrequencyResponseWidget, GroupDelayWidget, SpectrogramWidget,
)
from acoustic_phase_optimizer.visualization.controls import ControlPanel
from acoustic_phase_optimizer.utils.logging import get_logger

logger = get_logger(__name__)


class VisualizationApp(QMainWindow):
    """Main application window for the Acoustic Phase Optimizer."""

    def __init__(self, config: Optional[Config] = None):
        super().__init__()
        self.config = config or Config()
        self.setWindowTitle("Acoustic Phase Optimizer")
        self.setMinimumSize(1400, 900)

        self.room_model = RoomModel()
        self.speakers: List[Speaker] = []
        self.microphones: List[Microphone] = []
        self.virtual_room: Optional[VirtualRoom] = None

        self._init_ui()
        self._setup_default_data()
        self._connect_signals()

        self._refresh_timer = QTimer()
        self._refresh_timer.timeout.connect(self._periodic_refresh)
        self._refresh_timer.start(100)

    def _init_ui(self) -> None:
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QHBoxLayout()

        self.control_panel = ControlPanel()
        main_layout.addWidget(self.control_panel, 1)

        view_splitter = QSplitter(Qt.Orientation.Vertical)

        top_tabs = QTabWidget()
        self.room_view = RoomViewWidget()
        self.room_3d_view = Room3DViewWidget()
        self.heatmap_view = HeatmapWidget()
        self.multi_heatmap = MultiHeatmapWidget()

        top_tabs.addTab(self.room_view, "Room 2D")
        top_tabs.addTab(self.room_3d_view, "Room 3D")
        top_tabs.addTab(self.heatmap_view, "Heatmap")
        top_tabs.addTab(self.multi_heatmap, "All Maps")

        view_splitter.addWidget(top_tabs)

        bottom_tabs = QTabWidget()
        self.freq_view = FrequencyResponseWidget()
        self.group_delay_view = GroupDelayWidget()
        self.spectrogram_view = SpectrogramWidget()

        bottom_tabs.addTab(self.freq_view, "Frequency Response")
        bottom_tabs.addTab(self.group_delay_view, "Group Delay")
        bottom_tabs.addTab(self.spectrogram_view, "Spectrogram")

        view_splitter.addWidget(bottom_tabs)
        main_layout.addWidget(view_splitter, 3)

        central.setLayout(main_layout)

    def _setup_default_data(self) -> None:
        self.speakers = [
            Speaker("Left Main", SpeakerType.MAIN_LEFT, np.array([-8.0, 1.0, 2.0])),
            Speaker("Right Main", SpeakerType.MAIN_RIGHT, np.array([8.0, 1.0, 2.0])),
            Speaker("Subwoofer", SpeakerType.SUBWOOFER, np.array([0.0, 2.5, 0.0])),
            Speaker("Delay Left", SpeakerType.DELAY, np.array([-12.0, 15.0, 3.0])),
            Speaker("Delay Right", SpeakerType.DELAY, np.array([12.0, 15.0, 3.0])),
        ]
        for s in self.speakers:
            s.enabled = True

        self.microphones = [
            Microphone("Mic FOH", np.array([0.0, 12.0, 1.2]), zone="foh"),
            Microphone("Mic Left", np.array([-6.0, 8.0, 1.2]), zone="left"),
            Microphone("Mic Right", np.array([6.0, 8.0, 1.2]), zone="right"),
            Microphone("Mic Center", np.array([0.0, 6.0, 1.2]), zone="center"),
            Microphone("Mic Balcony", np.array([0.0, 18.0, 3.0]), zone="balcony"),
        ]

        self.virtual_room = VirtualRoom(self.room_model)
        for s in self.speakers:
            self.virtual_room.add_speaker(s)
        for m in self.microphones:
            self.virtual_room.add_microphone(m)

        self._update_views()
        self.control_panel.set_speaker_names([s.name for s in self.speakers])
        self.control_panel.log("Application initialized with default venue")

    def _connect_signals(self) -> None:
        self.control_panel.measurement_started.connect(self._on_measurement_start)
        self.control_panel.optimization_started.connect(self._on_optimization_start)
        self.control_panel.speaker_updated.connect(self._on_speaker_update)

    def _on_measurement_start(self, params: dict) -> None:
        self.control_panel.log(f"Starting measurement: {params}")
        if self.virtual_room:
            for mic in self.microphones:
                for spk in self.speakers:
                    freqs, mag_db = self.virtual_room.compute_transfer_function(spk, mic)
                    self.freq_view.update_data(
                        f"{spk.name} -> {mic.name}",
                        freqs, mag_db,
                        phase_rad=np.zeros_like(freqs),
                    )

    def _on_optimization_start(self, params: dict) -> None:
        self.control_panel.log(f"Starting optimization: {params['algorithm']}")
        engine = OptimizationEngine(params)
        objective = ObjectiveFunction()

        def objective_wrapper(p: np.ndarray) -> float:
            for i, spk in enumerate(self.speakers):
                if i * 3 < len(p):
                    spk.delay_ms = float(p[i * 3])
                    spk.gain_db = float(p[i * 3 + 1])
            data = self._gather_measurement_data()
            return objective.compute(p, data)

        bounds, initial = engine.setup_speaker_optimization(len(self.speakers))

        if params["algorithm"] == "compare_all":
            results = engine.compare_algorithms(objective_wrapper, initial)
            best_algo, best_result = engine.get_best_result(results)
            self.control_panel.log(f"Best algorithm: {best_algo} ({best_result.best_value:.4f})")
            for name, result in results.items():
                self.control_panel.log(f"  {name}: {result.best_value:.4f} ({result.iterations} it, {result.computation_time:.1f}s)")
        else:
            result = engine.optimize(params["algorithm"], objective_wrapper, initial)
            self.control_panel.log(f"Optimization complete: {result.best_value:.4f}")

        self._update_views()

    def _on_speaker_update(self, speaker_name: str, params: dict) -> None:
        for spk in self.speakers:
            if spk.name == speaker_name:
                spk.delay_ms = params.get("delay_ms", spk.delay_ms)
                spk.gain_db = params.get("gain_db", spk.gain_db)
                if params.get("polarity_inverted"):
                    spk.polarity = SpeakerPolarity.INVERTED
                else:
                    spk.polarity = SpeakerPolarity.NORMAL
                self.control_panel.log(f"Updated {speaker_name}: delay={spk.delay_ms}ms, gain={spk.gain_db}dB")
                self._update_views()
                break

    def _gather_measurement_data(self) -> dict:
        return {
            "phase": [],
            "magnitude_db": [],
            "cancellation_zones": [],
            "delays_ms": [s.delay_ms for s in self.speakers],
            "rt60": {"broadband": self.room_model.estimate_rt60_eyring()},
        }

    def _update_views(self) -> None:
        self.room_view.update_data(self.room_model, self.speakers, self.microphones)
        self.room_3d_view.update_data(self.room_model, self.speakers, self.microphones)

        if self.virtual_room:
            X, Y = self.virtual_room.get_listening_area_mesh(30)
            Z_spl = np.zeros_like(X)
            for spk in self.speakers:
                if not spk.enabled:
                    continue
                for i in range(X.shape[0]):
                    for j in range(X.shape[1]):
                        pt = np.array([X[i, j], Y[i, j], 1.2])
                        Z_spl[i, j] += spk.spl_at_distance(pt)
            self.heatmap_view.set_spl_heatmap(X, Y, Z_spl)

        for i, spk in enumerate(self.speakers):
            if not spk.enabled:
                continue
            color = ["#ff4444", "#4488ff", "#8844ff", "#44ffff", "#44ffff"][i % 5]
            test_freqs = np.logspace(np.log10(20), np.log10(20000), 500)
            test_mag = -10 * np.log10(1 + (test_freqs / 1000) ** 2) + spk.gain_db
            self.freq_view.update_data(
                spk.name, test_freqs, test_mag,
                color=color,
            )

    def _periodic_refresh(self) -> None:
        pass

    def closeEvent(self, event) -> None:
        self._refresh_timer.stop()
        event.accept()

    def run(self) -> None:
        self.show()
        sys.exit(QApplication.instance().exec())


def launch_gui(config: Optional[Config] = None) -> None:
    app = QApplication.instance()
    if app is None:
        app = QApplication(sys.argv)
    app.setApplicationName("Acoustic Phase Optimizer")
    app.setOrganizationName("APO")

    window = VisualizationApp(config)
    window.show()
    sys.exit(app.exec())
