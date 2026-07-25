"""Main PyQt6 visualization application."""

from __future__ import annotations

import sys
import time
import threading
import numpy as np
from typing import Dict, List, Optional, Tuple
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QSplitter,
    QVBoxLayout, QHBoxLayout, QTabWidget, QMessageBox, QProgressDialog,
    QLabel, QToolButton, QPushButton,
)
from PyQt6.QtCore import Qt, QTimer, QThread, QObject, pyqtSignal

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
from acoustic_phase_optimizer.visualization.comparison import (
    BeforeAfterWidget, AlgorithmComparisonWidget,
)
from PyQt6.QtWidgets import QMenu
from PyQt6.QtGui import QAction, QIcon
from acoustic_phase_optimizer.visualization.controls import ControlPanel
from acoustic_phase_optimizer.weather.api import fetch_location_coords, fetch_yearly_averages
from acoustic_phase_optimizer.weather.acoustic_mapping import weather_to_acoustic_params
from acoustic_phase_optimizer.room.lidar_import import import_lidar, fit_room_from_points
from acoustic_phase_optimizer.room.mic_placer import optimize_mic_positions
from acoustic_phase_optimizer.acoustic.microphone import Microphone
from acoustic_phase_optimizer.dsp.peq import PEQConfig, PEQBand, PEQFilter
from acoustic_phase_optimizer.dsp.geq import GEQConfig, GEQFilter
from acoustic_phase_optimizer.reporting import (
    ReportData, SpeakerReport, AlgoScore, Recommendation,
    render_report, open_in_browser,
)
from acoustic_phase_optimizer.utils.logging import get_logger

logger = get_logger(__name__)


class OptimizationWorker(QObject):
    finished = pyqtSignal(object)
    progress = pyqtSignal(str)

    def __init__(self, speakers, room_model, microphones, params):
        super().__init__()
        self.speakers = speakers
        self.room_model = room_model
        self.microphones = microphones
        self.params = params
        self._cancel_event = threading.Event()

    def cancel(self) -> None:
        self._cancel_event.set()

    def _check_cancel(self) -> None:
        if self._cancel_event.is_set():
            raise InterruptedError("Optimization canceled")

    def run(self) -> None:
        engine = OptimizationEngine(self.params)
        objective = ObjectiveFunction()
        n_speakers = len(self.speakers)

        if self.params.get("algorithm") == "dsp_only":
            self._run_dsp_optimization(engine, objective, n_speakers)
            return

        def objective_wrapper(p: np.ndarray) -> float:
            self._check_cancel()
            for i, spk in enumerate(self.speakers):
                if i * 3 < len(p):
                    spk.delay_ms = float(p[i * 3])
                    spk.gain_db = float(p[i * 3 + 1])
            data = {
                "phase": [],
                "magnitude_db": [],
                "cancellation_zones": [],
                "delays_ms": [s.delay_ms for s in self.speakers],
                "rt60": {"broadband": self.room_model.estimate_rt60_eyring()},
            }
            return objective.compute(p, data)

        bounds, initial = engine.setup_speaker_optimization(n_speakers)

        try:
            if self.params["algorithm"] == "compare_all":
                results = engine.compare_algorithms(objective_wrapper, initial, cancel_check=self._check_cancel)
                self.finished.emit(results)
            else:
                self.progress.emit(f"Running {self.params['algorithm']} optimization...")
                result = engine.optimize(self.params["algorithm"], objective_wrapper, initial)
                self.finished.emit(result)
        except InterruptedError:
            self.finished.emit(None)

    def _run_dsp_optimization(self, engine, objective, n_speakers) -> None:
        engine.config["max_iterations"] = 30
        engine.config["learning_rate"] = 0.05
        dsp_algo = "gradient"

        vr = VirtualRoom(self.room_model, sample_rate=48000)
        for s in self.speakers:
            vr.add_speaker(s)
        for m in self.microphones if self.microphones else []:
            vr.add_microphone(m)

        raw_ffts: dict = {}
        freqs_cache: Optional[np.ndarray] = None
        for spk in self.speakers:
            if not spk.enabled:
                continue
            for mic in (self.microphones if self.microphones else []):
                key = (spk.name, mic.name)
                ir = vr.compute_impulse_response(spk, mic)
                spec = np.fft.rfft(ir)
                raw_ffts[key] = spec.astype(np.complex128)
                if freqs_cache is None:
                    freqs_cache = np.fft.rfftfreq(len(ir), 1.0 / 48000.0)

        def dsp_objective_wrapper(p: np.ndarray) -> float:
            self._check_cancel()
            idx = 0
            for spk in self.speakers:
                if not spk.enabled:
                    continue
                for band in spk.peq.bands:
                    band.freq_hz = float(max(20.0, min(20000.0, p[idx]))); idx += 1
                    band.gain_db = float(max(-15.0, min(15.0, p[idx]))); idx += 1
                    band.q = float(max(0.1, min(20.0, p[idx]))); idx += 1
                for j in range(31):
                    spk.geq.gains_db[j] = float(max(-15.0, min(15.0, p[idx]))); idx += 1
                spk.delay_ms = float(max(0.0, min(500.0, p[idx]))); idx += 1
                spk.polarity = SpeakerPolarity.NORMAL if p[idx] >= 0 else SpeakerPolarity.INVERTED; idx += 1

            phases = []
            magnitudes = []
            delays = []
            for spk in self.speakers:
                if not spk.enabled:
                    continue
                peq_response = PEQFilter(spk.peq, 48000).frequency_response(freqs_cache)
                geq_response = GEQFilter.magnitude_response(spk.geq.gains_db, freqs_cache)
                for mic in (self.microphones if self.microphones else []):
                    key = (spk.name, mic.name)
                    spec = raw_ffts[key].copy()
                    delay_rad = -2.0 * np.pi * freqs_cache * (spk.delay_ms / 1000.0)
                    spec *= np.exp(1j * delay_rad)
                    spec *= spk.polarity.value
                    spec *= peq_response
                    spec *= geq_response
                    mag = 20.0 * np.log10(np.abs(spec) + 1e-12)
                    phase = np.angle(spec)
                    phases.append(phase)
                    magnitudes.append(mag)
                delays.append(spk.delay_ms)

            data = {
                "phase": phases,
                "magnitude_db": magnitudes,
                "cancellation_zones": [],
                "delays_ms": delays,
                "rt60": {"broadband": self.room_model.estimate_rt60_eyring()},
            }
            return objective.compute(np.zeros(1), data)

        bounds, initial = engine.setup_dsp_optimization(n_speakers)
        self.progress.emit("Running DSP-only optimization...")

        try:
            result = engine.optimize(dsp_algo, dsp_objective_wrapper, initial)
            self.finished.emit(result)
        except InterruptedError:
            self.finished.emit(None)


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
        self.last_optimization_result = None
        self.last_algorithm_comparison: Optional[dict] = None

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

        right_side = QVBoxLayout()

        self.speaker_palette = self._create_speaker_palette()
        right_side.addWidget(self.speaker_palette)

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
        self.before_after = BeforeAfterWidget()
        top_tabs.addTab(self.before_after, "Before/After")

        view_splitter.addWidget(top_tabs)

        bottom_tabs = QTabWidget()
        self.freq_view = FrequencyResponseWidget()
        self.group_delay_view = GroupDelayWidget()
        self.spectrogram_view = SpectrogramWidget()

        bottom_tabs.addTab(self.freq_view, "Frequency Response")
        bottom_tabs.addTab(self.group_delay_view, "Group Delay")
        bottom_tabs.addTab(self.spectrogram_view, "Spectrogram")
        self.algo_comparison = AlgorithmComparisonWidget()
        bottom_tabs.addTab(self.algo_comparison, "Algo Comparison")

        bottom_wrapper = QVBoxLayout()
        legend_btn = QPushButton("Toggle Legend")
        legend_btn.setFixedHeight(22)
        legend_btn.clicked.connect(self.freq_view.toggle_legend)
        bottom_wrapper.addWidget(legend_btn)
        bottom_wrapper.addWidget(bottom_tabs)

        wrapper_widget = QWidget()
        wrapper_widget.setLayout(bottom_wrapper)
        view_splitter.addWidget(wrapper_widget)
        right_side.addWidget(view_splitter, 3)

        main_layout.addLayout(right_side, 3)

        central.setLayout(main_layout)

    def _create_speaker_palette(self) -> QWidget:
        from PyQt6.QtWidgets import QToolBar, QToolButton
        toolbar = QWidget()
        layout = QHBoxLayout()
        layout.setContentsMargins(0, 0, 0, 0)
        label = QLabel("Place: ")
        layout.addWidget(label)

        self._palette_buttons = []
        for st in SpeakerType:
            btn = QToolButton()
            btn.setText(st.value.replace("_", " ").title())
            btn.setCheckable(True)
            btn.setToolTip(f"Click then click in Room 2D to place a {st.value}")
            btn.clicked.connect(lambda checked, t=st: self._on_palette_select(t))
            layout.addWidget(btn)
            self._palette_buttons.append(btn)

        cancel_btn = QToolButton()
        cancel_btn.setText("Cancel")
        cancel_btn.clicked.connect(self._on_palette_clear)
        layout.addWidget(cancel_btn)

        toolbar.setLayout(layout)
        return toolbar

    def _on_palette_select(self, speaker_type: SpeakerType) -> None:
        for btn in self._palette_buttons:
            btn.setChecked(False)
        self.sender().setChecked(True)
        self.room_view.set_pending_type(speaker_type)

    def _on_palette_clear(self) -> None:
        for btn in self._palette_buttons:
            btn.setChecked(False)
        self.room_view.set_pending_type(None)

    def _setup_default_data(self) -> None:
        self.speakers = []
        self.microphones = []
        self.virtual_room = VirtualRoom(self.room_model)
        self.control_panel.set_speaker_names([])
        self.control_panel.log("Application initialized — use the Room tab to set dimensions, place speakers and stage")

    def _connect_signals(self) -> None:
        self.control_panel.measurement_started.connect(self._on_measurement_start)
        self.control_panel.optimization_started.connect(self._on_optimization_start)
        self.control_panel.speaker_updated.connect(self._on_speaker_update)
        self.control_panel.weather_controls.fetch_button.clicked.connect(self._on_weather_fetch)
        self.control_panel.lidar_imported.connect(self._on_lidar_import)
        self.control_panel.dimensions_changed.connect(self._on_dimensions_changed)
        self.control_panel.mic_placement_requested.connect(self._on_mic_placement)
        self.control_panel.stage_changed.connect(self._on_stage_control_changed)
        self.control_panel.room_controls.stage_toggled.connect(self._on_stage_toggled)
        self.room_view.set_on_speaker_placed(self._on_room_click_place_speaker)
        self.room_view.set_on_speaker_right_click(self._on_room_speaker_right_click)
        self.room_view.set_on_speaker_moved(self._on_room_speaker_moved)
        self.room_view.set_on_stage_changed(self._on_room_stage_changed)
        self.control_panel.report_requested.connect(self._on_generate_report)
        self.control_panel.preset_requested.connect(self._on_load_preset)

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
        algorithm = params.get("algorithm", "compare_all")
        self.control_panel.log(f"Starting optimization: {algorithm}")
        self.control_panel.setEnabled(False)

        self._progress = QProgressDialog("Optimizing...", "Cancel", 0, 0, self)
        self._progress.setWindowTitle("Acoustic Phase Optimizer")
        self._progress.setMinimumDuration(0)
        self._progress.setValue(0)
        self._progress.canceled.connect(self._on_optimization_canceled)

        self._opt_thread = QThread()
        self._opt_worker = OptimizationWorker(self.speakers, self.room_model, self.microphones, params)
        self._opt_worker.moveToThread(self._opt_thread)
        self._opt_thread.started.connect(self._opt_worker.run)
        self._opt_worker.finished.connect(self._on_optimization_finished)
        self._opt_worker.progress.connect(self._progress.setLabelText)
        self._opt_worker.finished.connect(self._opt_thread.quit)
        self._opt_worker.finished.connect(self._opt_worker.deleteLater)
        self._opt_thread.finished.connect(self._opt_thread.deleteLater)
        self._progress.canceled.connect(self._opt_worker.cancel)
        self._opt_thread.start()

    def _on_optimization_finished(self, result) -> None:
        self._progress.close()
        self.control_panel.setEnabled(True)

        if result is None:
            self.control_panel.log("Optimization canceled")
            return

        if isinstance(result, dict):
            self.last_algorithm_comparison = result
            engine = OptimizationEngine()
            best_algo, best_result = engine.get_best_result(result)
            self.last_optimization_result = best_result
            self.control_panel.log(f"Best algorithm: {best_algo} ({best_result.best_value:.4f})")
            comparison_data = {}
            for name, r in result.items():
                self.control_panel.log(f"  {name}: {r.best_value:.4f} ({r.iterations} it, {r.computation_time:.1f}s)")
                comparison_data[name] = {
                    "best_value": r.best_value,
                    "history": r.history,
                    "iterations": r.iterations,
                    "computation_time": r.computation_time,
                }
            self.algo_comparison.update_results(comparison_data)
        else:
            self.last_optimization_result = result
            self.last_algorithm_comparison = None
            self.control_panel.log(f"Optimization complete: {result.best_value:.4f}")

        self.control_panel.optimization_controls.report_button.setEnabled(True)
        self._update_views()

    def _on_optimization_canceled(self) -> None:
        if hasattr(self, '_opt_worker'):
            self._opt_worker.cancel()
        if hasattr(self, '_opt_thread') and self._opt_thread.isRunning():
            self._opt_thread.quit()
            self._opt_thread.wait(2000)
        self.control_panel.setEnabled(True)
        self.control_panel.log("Optimization canceled")

    def _on_generate_report(self) -> None:
        if self.last_optimization_result is None:
            self.control_panel.log("No optimization results to report", "WARN")
            return

        from pathlib import Path

        dims = self.room_model.get_dimensions_array()
        width = float(dims[1])
        depth = float(dims[0])

        speakers = []
        ref = self.speakers[0] if self.speakers else None
        for spk in self.speakers:
            if not spk.enabled:
                continue
            dx = float(spk.x - ref.x) if ref else 0.0
            dy = float(spk.y - ref.y) if ref else 0.0
            dist = float(np.sqrt(dx**2 + dy**2)) if ref else 0.0
            db_weight = max(-40.0, min(0.0, float(spk.gain_db)))
            speakers.append(SpeakerReport(
                label=spk.name, x=float(spk.x), y=float(spk.y),
                distance_to_ref_m=dist,
                delay_ms=spk.delay_ms,
                gain_trim_db=spk.gain_db,
                db_before=0.0,
                db_after=db_weight,
            ))

        algo_comparison = []
        if self.last_algorithm_comparison:
            colors = ["#34d6c0", "#34d6c0", "#ffb020", "#ff5462"]
            for i, (name, r) in enumerate(sorted(
                self.last_algorithm_comparison.items(),
                key=lambda kv: -kv[1].best_value,
            )):
                algo_comparison.append(AlgoScore(
                    name=name.capitalize(),
                    score=max(0.0, min(1.0, float(r.best_value) / 2 + 0.5)),
                    color=colors[i % len(colors)],
                ))

        result = self.last_optimization_result
        null_before = min(0.0, min(
            (float(s.gain_db) if s.delay_ms == 0 else -5.0) for s in self.speakers
        )) if self.speakers else -18.0
        null_after = max(-6.0, min(
            float(s.gain_db) for s in self.speakers
        )) if self.speakers else -6.0
        coherence = 0.5
        if self.last_algorithm_comparison:
            coherence = max(r.best_value for r in self.last_algorithm_comparison.values())
        elif hasattr(result, 'best_value'):
            coherence = result.best_value

        data = ReportData(
            venue_name="Optimization Session",
            algorithm_used=result.algorithm.capitalize(),
            iterations=result.iterations,
            duration_s=result.computation_time,
            room_width_m=width,
            room_depth_m=depth,
            speed_of_sound=self.room_model.speed_of_sound,
            heatmap_freq_hz=1000,
            speakers=speakers,
            worst_null_before_db=null_before,
            worst_null_after_db=null_after,
            floor_below_10db_before_pct=max(10, int(-null_before * 2)),
            floor_below_10db_after_pct=max(2, int(-null_after * 0.8)),
            coherence_gain_db=round(coherence * 20, 1),
            algorithm_comparison=algo_comparison,
            recommendations=[
                Recommendation("medium", "Run a final on-site measurement to verify alignment before the show."),
            ],
        )

        output = Path("session_report.html").resolve()
        render_report(data, output)
        self.control_panel.log(f"Report saved: {output}")
        open_in_browser(output)

    def _on_weather_fetch(self) -> None:
        location_name = self.control_panel.weather_controls.location_input.text().strip()
        if not location_name:
            self.control_panel.weather_controls.set_status("Please enter a location name")
            return

        self.control_panel.weather_controls.set_status(f"Resolving {location_name}...")
        QApplication.processEvents()

        location = fetch_location_coords(location_name)
        if location is None:
            self.control_panel.weather_controls.set_status(f"Could not find location: {location_name}")
            return

        self.control_panel.weather_controls.set_status(
            f"Fetching yearly weather data for {location.name}..."
        )
        QApplication.processEvents()

        averages = fetch_yearly_averages(location)
        if averages is None:
            self.control_panel.weather_controls.set_status("Failed to fetch weather data")
            return

        self.control_panel.weather_controls.set_results(averages)
        self.control_panel.log(
            f"Weather: {averages.location.name} — "
            f"{averages.temperature_mean:.1f}°C, "
            f"{averages.humidity_mean:.1f}%RH, "
            f"{averages.pressure_mean:.1f}hPa"
        )

        acoustic = weather_to_acoustic_params(averages)
        self.room_model.speed_of_sound = acoustic.speed_of_sound
        self.control_panel.log(
            f"Speed of sound adjusted to {acoustic.speed_of_sound:.1f} m/s "
            f"(was 343.0 m/s)"
        )

    def _on_lidar_import(self, path: str) -> None:
        self.control_panel.log(f"Importing LIDAR: {path}")
        scan = import_lidar(path)
        if scan is None:
            self.control_panel.log("Failed to import LIDAR scan", "ERROR")
            return
        self.control_panel.log(f"Loaded {len(scan.points)} points")

        self.room_view.set_lidar_points(scan.points)
        self.room_3d_view.set_lidar_points(scan.points)

        room = fit_room_from_points(scan)
        if room is None:
            self.control_panel.log("Could not fit room from points", "ERROR")
            return
        self.room_model = room
        L, W, H = room.get_dimensions_array()
        self.control_panel.room_controls.room_length.setValue(L)
        self.control_panel.room_controls.room_width.setValue(W)
        self.control_panel.room_controls.room_height.setValue(H)
        self.control_panel.log(f"Room fitted: {L:.1f} x {W:.1f} x {H:.1f}m")
        self._reinit_virtual_room()

    def _on_dimensions_changed(self, length: float, width: float, height: float) -> None:
        self.room_model.set_dimensions(length, width, height)
        self.room_view.set_lidar_points(None)
        self.room_3d_view.set_lidar_points(None)
        self.control_panel.log(f"Room dimensions set: {length:.1f} x {width:.1f} x {height:.1f}m")
        self._reinit_virtual_room()

    def _reinit_virtual_room(self) -> None:
        old_speakers = self.speakers if self.speakers else []
        old_mics = self.microphones if self.microphones else []
        self.virtual_room = VirtualRoom(self.room_model)
        for s in old_speakers:
            self.virtual_room.add_speaker(s)
        for m in old_mics:
            self.virtual_room.add_microphone(m)
        self._update_views()

    def _on_mic_placement(self, max_mics: int) -> None:
        if not self.speakers:
            self.control_panel.log("Place speakers first before optimizing mic positions", "WARN")
            return
        self.control_panel.log(f"Optimizing {max_mics} mic positions...")
        result = optimize_mic_positions(
            self.room_model, self.speakers, max_mics=max_mics
        )
        self.microphones = [
            Microphone(name, pos, zone="auto")
            for name, pos in zip(result.names, result.positions)
        ]
        self._reinit_virtual_room()
        self.control_panel.log(
            f"Placed {len(self.microphones)} mics "
            f"(coverage={result.coverage_score:.3f}, diversity={result.diversity_score:.3f})"
        )

    def _on_room_click_place_speaker(
        self, name_hint: str, speaker_type: SpeakerType, x: float, y: float, z: float
    ) -> None:
        count = len([s for s in self.speakers if s.speaker_type == speaker_type]) + 1
        name = name_hint or f"{speaker_type.value.replace('_', ' ').title()} {count}"
        spk = Speaker(name, speaker_type, np.array([x, y, z]))
        self.speakers.append(spk)
        self.virtual_room.add_speaker(spk)
        self.control_panel.set_speaker_names([s.name for s in self.speakers])
        self.control_panel.log(f"Placed '{name}' ({speaker_type.value}) at ({x:.1f}, {y:.1f}, z={z:.1f})")
        self._update_views()
        self._on_palette_clear()

    def _on_room_speaker_right_click(self, speaker_name: str, x: float, y: float) -> None:
        for spk in self.speakers:
            if spk.name == speaker_name:
                from PyQt6.QtWidgets import QInputDialog, QDialog, QVBoxLayout, QFormLayout, QDialogButtonBox, QDoubleSpinBox
                dialog = QDialog(self)
                dialog.setWindowTitle(f"Edit {speaker_name}")
                layout = QVBoxLayout()
                form = QFormLayout()
                sx = QDoubleSpinBox(); sx.setRange(-100, 100); sx.setValue(spk.x); sx.setSuffix(" m"); form.addRow("X:", sx)
                sy = QDoubleSpinBox(); sy.setRange(-100, 100); sy.setValue(spk.y); sy.setSuffix(" m"); form.addRow("Y:", sy)
                sz = QDoubleSpinBox(); sz.setRange(0, 50); sz.setValue(spk.z); sz.setSuffix(" m"); form.addRow("Height:", sz)
                layout.addLayout(form)
                buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
                delete_btn = buttons.addButton("Delete", QDialogButtonBox.ButtonRole.DestructiveRole)
                layout.addWidget(buttons)
                dialog.setLayout(layout)
                buttons.accepted.connect(dialog.accept)
                buttons.rejected.connect(dialog.reject)
                delete_btn.clicked.connect(lambda: self._delete_speaker(speaker_name) or dialog.accept())
                if dialog.exec() == QDialog.DialogCode.Accepted:
                    spk.position = np.array([sx.value(), sy.value(), sz.value()])
                    self.control_panel.log(f"Updated {speaker_name} position: ({sx.value():.1f}, {sy.value():.1f}, {sz.value():.1f})")
                    self._update_views()
                break

    def _delete_speaker(self, speaker_name: str) -> None:
        self.speakers = [s for s in self.speakers if s.name != speaker_name]
        self.control_panel.set_speaker_names([s.name for s in self.speakers])
        self.control_panel.log(f"Deleted {speaker_name}")
        self._reinit_virtual_room()

    def _on_load_preset(self, preset_name: str) -> None:
        from acoustic_phase_optimizer.simulation.presets import gymnasium, theater, conference_room
        presets = {"Gymnasium": gymnasium, "Theater": theater, "Conference Room": conference_room}
        loader = presets.get(preset_name)
        if loader is None:
            self.control_panel.log(f"Unknown preset: {preset_name}", "WARN")
            return

        room, speakers = loader()
        self.room_model = room
        self.speakers = speakers
        L, W, H = room.get_dimensions_array()
        self.control_panel.room_controls.room_length.setValue(L)
        self.control_panel.room_controls.room_width.setValue(W)
        self.control_panel.room_controls.room_height.setValue(H)
        self.control_panel.set_speaker_names([s.name for s in speakers])
        self.control_panel.log(f"Loaded '{preset_name}' — {L:.0f}x{W:.0f}x{H:.0f}m, {len(speakers)} speakers")
        self._reinit_virtual_room()

    def _on_stage_toggled(self, visible: bool) -> None:
        self.room_view.set_stage_visible(visible)
        self.control_panel.log(f"Stage {'added' if visible else 'removed'}")

    def _on_room_speaker_moved(self, name: str, x: float, y: float) -> None:
        for spk in self.speakers:
            if spk.name == name:
                spk.position[0] = x
                spk.position[1] = y
                if self.virtual_room:
                    self.virtual_room.add_speaker(spk)
                self.control_panel.log(f"Moved {name} to ({x:.1f}, {y:.1f})")
                self._update_views()
                break

    def _on_room_stage_changed(self, x: float, y: float, w: float, d: float, h: float) -> None:
        self.room_view.stage_x = x
        self.room_view.stage_y = y
        self.room_view.stage_w = w
        self.room_view.stage_d = d
        self.room_view.stage_height = h
        self.control_panel.room_controls.stage_x.setValue(x)
        self.control_panel.room_controls.stage_y.setValue(y)
        self.control_panel.room_controls.stage_w.setValue(w)
        self.control_panel.room_controls.stage_d.setValue(d)
        self.control_panel.room_controls.stage_elev.setValue(h)
        self.control_panel.log(f"Stage: ({x:.1f}, {y:.1f}) {w:.1f}x{d:.1f}m, elevation={h:.1f}m")
        self._update_views()

    def _on_stage_control_changed(self, x: float, y: float, w: float, d: float, h: float) -> None:
        self.room_view.stage_x = x
        self.room_view.stage_y = y
        self.room_view.stage_w = w
        self.room_view.stage_d = d
        self.room_view.stage_height = h
        self.control_panel.log(f"Stage set: ({x:.1f}, {y:.1f}) {w:.1f}x{d:.1f}m, elevation={h:.1f}m")
        self._update_views()

    def _on_speaker_update(self, speaker_name: str, params: dict) -> None:
        for spk in self.speakers:
            if spk.name == speaker_name:
                height = params.get("height")
                if height is not None:
                    spk.position[2] = height
                spk.delay_ms = params.get("delay_ms", spk.delay_ms)
                spk.gain_db = params.get("gain_db", spk.gain_db)
                if params.get("polarity_inverted"):
                    spk.polarity = SpeakerPolarity.INVERTED
                else:
                    spk.polarity = SpeakerPolarity.NORMAL
                peq_bands = params.get("peq_bands")
                if peq_bands:
                    spk.peq = PEQConfig(bands=[
                        PEQBand(freq_hz=b["freq"], gain_db=b["gain"], q=b["q"])
                        for b in peq_bands
                    ])
                geq_gains = params.get("geq_gains")
                if geq_gains:
                    spk.geq = GEQConfig(gains_db=list(geq_gains))
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

        if self.virtual_room and self.speakers:
            X, Y = self.virtual_room.get_listening_area_mesh(30)
            c = self.room_model.speed_of_sound

            Z_spl = np.zeros_like(X)
            Z_before = np.zeros_like(X)
            Z_after = np.zeros_like(X)

            for spk in self.speakers:
                if not spk.enabled:
                    continue
                for i in range(X.shape[0]):
                    for j in range(X.shape[1]):
                        pt = np.array([X[i, j], Y[i, j], 1.2])
                        Z_spl[i, j] += spk.spl_at_distance(pt)

                        dist = float(np.linalg.norm(spk.position - pt))
                        if dist < 0.01:
                            continue
                        freq = 1000.0
                        wavenum = 2.0 * np.pi * freq / c

                        phase_before = wavenum * dist
                        amp_before = 1.0 / dist
                        Z_before[i, j] += amp_before * np.cos(phase_before)

                        phase_after = wavenum * dist - 2.0 * np.pi * freq * (spk.delay_ms / 1000.0)
                        amp_after = 1.0 / dist * (10.0 ** (spk.gain_db / 20.0))
                        Z_after[i, j] += amp_after * np.cos(phase_after)

            Z_before_db = 20.0 * np.log10(np.abs(Z_before) + 1e-12)
            Z_after_db = 20.0 * np.log10(np.abs(Z_after) + 1e-12)
            self.before_after.update_data(X, Y, Z_before_db, Z_after_db)
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
