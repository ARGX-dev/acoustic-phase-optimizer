from __future__ import annotations

from typing import Optional
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QLineEdit, QGroupBox, QFormLayout, QDoubleSpinBox, QSpinBox,
    QFileDialog,
)
from PyQt6.QtCore import Qt, pyqtSignal


class RoomControls(QGroupBox):
    """Room capture controls: LIDAR import or manual dimensions."""

    lidar_imported = pyqtSignal(str)
    dimensions_changed = pyqtSignal(float, float, float)
    mic_placement_requested = pyqtSignal(int)

    def __init__(self, parent=None):
        super().__init__("Room Setup", parent)
        layout = QVBoxLayout()

        lidar_group = QGroupBox("LIDAR Import")
        lidar_layout = QVBoxLayout()
        self.lidar_path_label = QLabel("No file loaded")
        lidar_layout.addWidget(self.lidar_path_label)
        self.import_button = QPushButton("Import LIDAR Scan...")
        lidar_layout.addWidget(self.import_button)
        lidar_group.setLayout(lidar_layout)
        layout.addWidget(lidar_group)

        manual_group = QGroupBox("Manual Dimensions")
        form = QFormLayout()
        self.room_length = QDoubleSpinBox()
        self.room_length.setRange(1.0, 200.0)
        self.room_length.setValue(30.0)
        self.room_length.setSuffix(" m")
        form.addRow("Length:", self.room_length)

        self.room_width = QDoubleSpinBox()
        self.room_width.setRange(1.0, 200.0)
        self.room_width.setValue(20.0)
        self.room_width.setSuffix(" m")
        form.addRow("Width:", self.room_width)

        self.room_height = QDoubleSpinBox()
        self.room_height.setRange(1.0, 50.0)
        self.room_height.setValue(8.0)
        self.room_height.setSuffix(" m")
        form.addRow("Height:", self.room_height)

        self.apply_dims_button = QPushButton("Apply Dimensions")
        form.addRow(self.apply_dims_button)
        manual_group.setLayout(form)
        layout.addWidget(manual_group)

        mic_group = QGroupBox("Mic Placement")
        mic_layout = QFormLayout()
        self.max_mics = QSpinBox()
        self.max_mics.setRange(1, 32)
        self.max_mics.setValue(8)
        mic_layout.addRow("Max Mics:", self.max_mics)
        self.place_mics_button = QPushButton("Optimize Mic Positions")
        mic_layout.addRow(self.place_mics_button)
        mic_group.setLayout(mic_layout)
        layout.addWidget(mic_group)

        self.setLayout(layout)
