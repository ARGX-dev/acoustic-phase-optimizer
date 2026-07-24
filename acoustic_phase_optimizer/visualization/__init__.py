"""
Visualization module for interactive GUI.

Provides PyQt6-based interactive visualization of room layout, speaker/microphone
positions, heat maps (SPL, phase, delay), frequency response plots, and cancellation zones.
"""

from acoustic_phase_optimizer.visualization.app import VisualizationApp

__all__ = [
    "VisualizationApp",
]
