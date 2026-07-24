"""Control panel widgets for the visualization application."""

from __future__ import annotations

from typing import Callable, Dict, List, Optional, Any
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QSlider, QComboBox, QDoubleSpinBox, QSpinBox, QGroupBox,
    QCheckBox, QTabWidget, QTextEdit, QGridLayout, QFrame,
)
from PyQt6.QtCore import Qt, pyqtSignal
from acoustic_phase_optimizer.weather.weather_controls import WeatherControls
from acoustic_phase_optimizer.visualization.room_controls import RoomControls


class MeasurementControls(QGroupBox):
    """Controls for room measurement parameters."""

    def __init__(self, parent=None):
        super().__init__("Measurement Controls", parent)
        layout = QGridLayout()

        layout.addWidget(QLabel("Sweep Type:"), 0, 0)
        self.sweep_type = QComboBox()
        self.sweep_type.addItems(["log", "mls"])
        layout.addWidget(self.sweep_type, 0, 1)

        layout.addWidget(QLabel("Duration (s):"), 1, 0)
        self.duration = QDoubleSpinBox()
        self.duration.setRange(0.5, 30.0)
        self.duration.setValue(5.0)
        self.duration.setSingleStep(0.5)
        layout.addWidget(self.duration, 1, 1)

        layout.addWidget(QLabel("Start Freq (Hz):"), 2, 0)
        self.start_freq = QDoubleSpinBox()
        self.start_freq.setRange(10.0, 1000.0)
        self.start_freq.setValue(20.0)
        layout.addWidget(self.start_freq, 2, 1)

        layout.addWidget(QLabel("End Freq (Hz):"), 3, 0)
        self.end_freq = QDoubleSpinBox()
        self.end_freq.setRange(1000.0, 48000.0)
        self.end_freq.setValue(20000.0)
        layout.addWidget(self.end_freq, 3, 1)

        layout.addWidget(QLabel("Averaging:"), 4, 0)
        self.averaging = QSpinBox()
        self.averaging.setRange(1, 10)
        self.averaging.setValue(3)
        layout.addWidget(self.averaging, 4, 1)

        self.start_button = QPushButton("Start Measurement")
        self.start_button.setStyleSheet(
            "background-color: #4CAF50; color: white; font-weight: bold; padding: 8px;"
        )
        layout.addWidget(self.start_button, 5, 0, 1, 2)

        self.setLayout(layout)

    def get_parameters(self) -> dict:
        return {
            "sweep_type": self.sweep_type.currentText(),
            "duration": self.duration.value(),
            "start_freq": self.start_freq.value(),
            "end_freq": self.end_freq.value(),
            "averaging": self.averaging.value(),
        }


class OptimizationControls(QGroupBox):
    """Controls for optimization parameters."""

    def __init__(self, parent=None):
        super().__init__("Optimization Controls", parent)
        layout = QGridLayout()

        layout.addWidget(QLabel("Algorithm:"), 0, 0)
        self.algorithm = QComboBox()
        self.algorithm.addItems([
            "genetic", "gradient", "annealing", "bayesian",
            "compare_all", "dsp_only",
        ])
        layout.addWidget(self.algorithm, 0, 1)

        layout.addWidget(QLabel("Max Iterations:"), 1, 0)
        self.max_iterations = QSpinBox()
        self.max_iterations.setRange(10, 10000)
        self.max_iterations.setValue(200)
        self.max_iterations.setSingleStep(100)
        layout.addWidget(self.max_iterations, 1, 1)

        layout.addWidget(QLabel("Population:"), 2, 0)
        self.population = QSpinBox()
        self.population.setRange(10, 500)
        self.population.setValue(100)
        layout.addWidget(self.population, 2, 1)

        layout.addWidget(QLabel("Mutation Rate:"), 3, 0)
        self.mutation_rate = QDoubleSpinBox()
        self.mutation_rate.setRange(0.0, 1.0)
        self.mutation_rate.setValue(0.15)
        self.mutation_rate.setSingleStep(0.05)
        layout.addWidget(self.mutation_rate, 3, 1)

        self.run_button = QPushButton("Run Optimization")
        self.run_button.setStyleSheet(
            "background-color: #2196F3; color: white; font-weight: bold; padding: 8px;"
        )
        layout.addWidget(self.run_button, 4, 0, 1, 2)

        self.setLayout(layout)

    def get_parameters(self) -> dict:
        return {
            "algorithm": self.algorithm.currentText(),
            "max_iterations": self.max_iterations.value(),
            "population_size": self.population.value(),
            "mutation_rate": self.mutation_rate.value(),
        }


class PEQBandWidget(QGroupBox):
    def __init__(self, band_index: int, parent=None):
        super().__init__(f"Band {band_index + 1}", parent)
        layout = QHBoxLayout()
        self.freq = QDoubleSpinBox()
        self.freq.setRange(20.0, 20000.0)
        self.freq.setValue(1000.0)
        self.freq.setSuffix(" Hz")
        layout.addWidget(QLabel("F:"))
        layout.addWidget(self.freq)
        self.gain = QDoubleSpinBox()
        self.gain.setRange(-15.0, 15.0)
        self.gain.setValue(0.0)
        self.gain.setSuffix(" dB")
        layout.addWidget(QLabel("G:"))
        layout.addWidget(self.gain)
        self.q = QDoubleSpinBox()
        self.q.setRange(0.1, 20.0)
        self.q.setValue(0.707)
        layout.addWidget(QLabel("Q:"))
        layout.addWidget(self.q)
        self.setLayout(layout)


class SpeakerControls(QGroupBox):
    """Individual speaker parameter adjustment with PEQ/GEQ."""

    def __init__(self, parent=None):
        super().__init__("Speaker Controls", parent)
        layout = QVBoxLayout()

        top_grid = QGridLayout()
        top_grid.addWidget(QLabel("Speaker:"), 0, 0)
        self.speaker_selector = QComboBox()
        top_grid.addWidget(self.speaker_selector, 0, 1)
        top_grid.addWidget(QLabel("Delay (ms):"), 1, 0)
        self.delay = QDoubleSpinBox()
        self.delay.setRange(0.0, 500.0)
        self.delay.setValue(0.0)
        self.delay.setSuffix(" ms")
        top_grid.addWidget(self.delay, 1, 1)
        top_grid.addWidget(QLabel("Gain (dB):"), 2, 0)
        self.gain = QDoubleSpinBox()
        self.gain.setRange(-60.0, 20.0)
        self.gain.setValue(0.0)
        self.gain.setSuffix(" dB")
        top_grid.addWidget(self.gain, 2, 1)
        self.polarity_invert = QCheckBox("Invert Polarity")
        top_grid.addWidget(self.polarity_invert, 3, 0, 1, 2)
        self.apply_button = QPushButton("Apply")
        self.apply_button.setStyleSheet("background-color: #FF9800; color: white; padding: 6px;")
        top_grid.addWidget(self.apply_button, 4, 0, 1, 2)
        layout.addLayout(top_grid)

        self.peq_group = QGroupBox("PEQ (6 Bands)")
        peq_layout = QVBoxLayout()
        self.peq_bands: List[PEQBandWidget] = []
        for i in range(6):
            bw = PEQBandWidget(i)
            self.peq_bands.append(bw)
            peq_layout.addWidget(bw)
        self.peq_group.setLayout(peq_layout)
        self.peq_group.setCheckable(True)
        self.peq_group.setChecked(False)
        layout.addWidget(self.peq_group)

        self.geq_group = QGroupBox("GEQ (31-Band ISO)")
        geq_layout = QVBoxLayout()
        geq_scroll = QWidget()
        geq_scroll_layout = QVBoxLayout(geq_scroll)
        self.geq_sliders: List[QDoubleSpinBox] = []
        iso_bands = [20, 25, 31.5, 40, 50, 63, 80, 100, 125, 160, 200, 250, 315, 400, 500,
                     630, 800, 1000, 1250, 1600, 2000, 2500, 3150, 4000, 5000, 6300, 8000,
                     10000, 12500, 16000, 20000]
        for freq in iso_bands:
            row = QHBoxLayout()
            row.addWidget(QLabel(f"{freq} Hz"))
            s = QDoubleSpinBox()
            s.setRange(-15.0, 15.0)
            s.setValue(0.0)
            s.setSuffix(" dB")
            row.addWidget(s)
            geq_scroll_layout.addLayout(row)
            self.geq_sliders.append(s)
        geq_layout.addWidget(geq_scroll)
        self.geq_group.setLayout(geq_layout)
        self.geq_group.setCheckable(True)
        self.geq_group.setChecked(False)
        layout.addWidget(self.geq_group)

        self.setLayout(layout)


class StatusPanel(QTextEdit):
    """Status and log output display."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setReadOnly(True)
        self.setMaximumHeight(150)
        self.setStyleSheet("background-color: #1e1e1e; color: #00ff00; font-family: monospace;")

    def append_message(self, message: str, level: str = "INFO") -> None:
        self.append(f"[{level}] {message}")


class ControlPanel(QWidget):
    """Main control panel combining all control groups."""

    measurement_started = pyqtSignal(dict)
    optimization_started = pyqtSignal(dict)
    speaker_updated = pyqtSignal(str, dict)
    lidar_imported = pyqtSignal(str)
    dimensions_changed = pyqtSignal(float, float, float)
    mic_placement_requested = pyqtSignal(int)

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout()

        self.tabs = QTabWidget()
        self.room_controls = RoomControls()
        self.measurement_controls = MeasurementControls()
        self.optimization_controls = OptimizationControls()
        self.speaker_controls = SpeakerControls()
        self.weather_controls = WeatherControls()

        self.tabs.addTab(self.room_controls, "Room")
        self.tabs.addTab(self.measurement_controls, "Measurement")
        self.tabs.addTab(self.optimization_controls, "Optimization")
        self.tabs.addTab(self.speaker_controls, "Speakers")
        self.tabs.addTab(self.weather_controls, "Weather")

        layout.addWidget(self.tabs)

        self.status_panel = StatusPanel()
        layout.addWidget(self.status_panel)

        self.setLayout(layout)

        self._connect_signals()

    def _connect_signals(self) -> None:
        self.measurement_controls.start_button.clicked.connect(
            lambda: self.measurement_started.emit(
                self.measurement_controls.get_parameters()
            )
        )
        self.optimization_controls.run_button.clicked.connect(
            lambda: self.optimization_started.emit(
                self.optimization_controls.get_parameters()
            )
        )
        self.speaker_controls.apply_button.clicked.connect(
            self._on_speaker_update
        )
        self.room_controls.import_button.clicked.connect(self._on_lidar_import)
        self.room_controls.apply_dims_button.clicked.connect(self._on_apply_dimensions)
        self.room_controls.place_mics_button.clicked.connect(
            lambda: self.mic_placement_requested.emit(self.room_controls.max_mics.value())
        )

    def _on_speaker_update(self) -> None:
        speaker_name = self.speaker_controls.speaker_selector.currentText()
        peq_bands = []
        for bw in self.speaker_controls.peq_bands:
            peq_bands.append({"freq": bw.freq.value(), "gain": bw.gain.value(), "q": bw.q.value()})
        geq_gains = [s.value() for s in self.speaker_controls.geq_sliders]
        params = {
            "delay_ms": self.speaker_controls.delay.value(),
            "gain_db": self.speaker_controls.gain.value(),
            "polarity_inverted": self.speaker_controls.polarity_invert.isChecked(),
            "peq_bands": peq_bands,
            "geq_gains": geq_gains,
        }
        self.speaker_updated.emit(speaker_name, params)

    def _on_lidar_import(self) -> None:
        from PyQt6.QtWidgets import QFileDialog
        path, _ = QFileDialog.getOpenFileName(
            self, "Import LIDAR Scan", "",
            "Point Cloud Files (*.ply *.pcd *.las *.laz);;All Files (*.*)"
        )
        if path:
            self.room_controls.lidar_path_label.setText(path.split("/")[-1].split("\\")[-1])
            self.lidar_imported.emit(path)

    def _on_apply_dimensions(self) -> None:
        L = self.room_controls.room_length.value()
        W = self.room_controls.room_width.value()
        H = self.room_controls.room_height.value()
        self.dimensions_changed.emit(L, W, H)

    def set_speaker_names(self, names: List[str]) -> None:
        self.speaker_controls.speaker_selector.clear()
        self.speaker_controls.speaker_selector.addItems(names)

    def log(self, message: str, level: str = "INFO") -> None:
        self.status_panel.append_message(message, level)
