"""Session report generation for presenting optimization results."""

from acoustic_phase_optimizer.reporting.renderer import (
    ReportData, SpeakerReport, AlgoScore, Recommendation,
    render_report, open_in_browser,
)

__all__ = [
    "ReportData", "SpeakerReport", "AlgoScore", "Recommendation",
    "render_report", "open_in_browser",
]
