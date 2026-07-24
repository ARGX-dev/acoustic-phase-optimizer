from __future__ import annotations

from typing import Optional
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QLineEdit, QGroupBox, QFormLayout, QDoubleSpinBox,
)
from PyQt6.QtCore import Qt, pyqtSignal

from acoustic_phase_optimizer.weather.models import Location, YearlyAverages


class WeatherControls(QGroupBox):
    """Controls for location-based weather data acquisition."""

    weather_fetched = pyqtSignal(object)

    def __init__(self, parent=None):
        super().__init__("Weather & Climate", parent)
        layout = QVBoxLayout()

        location_layout = QHBoxLayout()
        location_layout.addWidget(QLabel("Location:"))
        self.location_input = QLineEdit()
        self.location_input.setPlaceholderText("e.g. London, New York, Paris")
        location_layout.addWidget(self.location_input)
        self.fetch_button = QPushButton("Fetch Weather")
        location_layout.addWidget(self.fetch_button)
        layout.addLayout(location_layout)

        self.status_label = QLabel("Enter a location and click Fetch Weather")
        self.status_label.setWordWrap(True)
        layout.addWidget(self.status_label)

        params_group = QGroupBox("Yearly Averages")
        form = QFormLayout()

        self.temp_label = QLabel("-- °C")
        form.addRow("Temperature:", self.temp_label)
        self.humidity_label = QLabel("-- %")
        form.addRow("Humidity:", self.humidity_label)
        self.pressure_label = QLabel("-- hPa")
        form.addRow("Pressure:", self.pressure_label)
        self.speed_label = QLabel("-- m/s")
        form.addRow("Speed of Sound:", self.speed_label)

        params_group.setLayout(form)
        layout.addWidget(params_group)

        self._yearly_averages: Optional[YearlyAverages] = None

        self.setLayout(layout)

    def get_yearly_averages(self) -> Optional[YearlyAverages]:
        return self._yearly_averages

    def set_results(self, averages: YearlyAverages) -> None:
        self._yearly_averages = averages
        self.status_label.setText(
            f"Location: {averages.location.name}, {averages.location.country} "
            f"({averages.sample_count} days)"
        )
        self.temp_label.setText(f"{averages.temperature_mean:.1f} °C")
        self.humidity_label.setText(f"{averages.humidity_mean:.1f} %")
        self.pressure_label.setText(f"{averages.pressure_mean:.1f} hPa")

        from acoustic_phase_optimizer.weather.acoustic_mapping import weather_to_acoustic_params
        params = weather_to_acoustic_params(averages)
        self.speed_label.setText(f"{params.speed_of_sound:.1f} m/s")

    def set_status(self, message: str) -> None:
        self.status_label.setText(message)
