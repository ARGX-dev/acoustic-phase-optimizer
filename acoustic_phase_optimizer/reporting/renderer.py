"""HTML session report generator.

Turns optimizer output into a standalone HTML report with before/after
coverage heatmaps, algorithm comparison, delay table, and recommendations.
"""

from __future__ import annotations

import json
import webbrowser
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from string import Template
from typing import Optional


@dataclass
class SpeakerReport:
    label: str
    x: float
    y: float
    distance_to_ref_m: float
    delay_ms: float
    gain_trim_db: float
    db_before: float = 0.0
    db_after: float = 0.0


@dataclass
class AlgoScore:
    name: str
    score: float
    color: str = "#34d6c0"


@dataclass
class Recommendation:
    severity: str  # "high" | "medium" | "low"
    text: str


@dataclass
class MeasurementSummary:
    rt60_broadband_s: float = 0.0
    rt60_125hz: float = 0.0
    rt60_1khz: float = 0.0
    rt60_4khz: float = 0.0
    clarity_c80_db: float = 0.0
    definition_d50: float = 0.0
    clipping_detected: bool = False
    channels_measured: int = 0
    loopback_latency_ms: float = 0.0
    mic_cal_applied: bool = False


@dataclass
class ReportData:
    venue_name: str
    algorithm_used: str
    iterations: int
    duration_s: float
    room_width_m: float
    room_depth_m: float
    speed_of_sound: float
    heatmap_freq_hz: float
    speakers: list[SpeakerReport]
    worst_null_before_db: float
    worst_null_after_db: float
    floor_below_10db_before_pct: float
    floor_below_10db_after_pct: float
    coherence_gain_db: float
    algorithm_comparison: list[AlgoScore]
    recommendations: list[Recommendation] = field(default_factory=list)
    measurement: Optional[MeasurementSummary] = None


def render_report(
    data: ReportData,
    output_path: str | Path,
    template_path: Optional[str | Path] = None,
) -> Path:
    """Render a ReportData into a standalone HTML report file.

    Args:
        data: The session data to render.
        output_path: Where to write the HTML file.
        template_path: Path to the HTML template. Defaults to the bundled
                       templates/report_template.html.

    Returns:
        The absolute path to the generated report.
    """
    if template_path is None:
        template_path = Path(__file__).parent / "templates" / "report_template.html"

    template_text = Path(template_path).read_text(encoding="utf-8")

    def _spk_json(spk: SpeakerReport, use_after: bool) -> dict:
        hw = data.room_width_m / 2.0
        hd = data.room_depth_m / 2.0
        return {
            "label": spk.label,
            "x": round(spk.y + hw, 2),
            "y": round(spk.x + hd, 2),
            "db": spk.db_after if use_after else spk.db_before,
            "delayMs": spk.delay_ms if use_after else 0.0,
        }

    speakers_before = json.dumps([_spk_json(s, False) for s in data.speakers])
    speakers_after = json.dumps([_spk_json(s, True) for s in data.speakers])
    algos_json = json.dumps([
        {"name": a.name, "score": round(a.score, 2), "color": a.color}
        for a in sorted(data.algorithm_comparison, key=lambda a: -a.score)
    ])
    delay_rows = "\n".join(
        f"<tr><td>{s.label}</td><td>{s.distance_to_ref_m:.1f}</td>"
        f"<td>{s.delay_ms:.2f}</td><td>{s.gain_trim_db:+.1f}</td></tr>"
        for s in data.speakers
    )

    measurement_rows = ""
    if data.measurement:
        m = data.measurement
        measurement_rows = (
            f"<tr><td>RT60 (broadband)</td><td>{m.rt60_broadband_s:.2f}s</td></tr>"
            f"<tr><td>RT60 (125 Hz)</td><td>{m.rt60_125hz:.2f}s</td></tr>"
            f"<tr><td>RT60 (1 kHz)</td><td>{m.rt60_1khz:.2f}s</td></tr>"
            f"<tr><td>RT60 (4 kHz)</td><td>{m.rt60_4khz:.2f}s</td></tr>"
            f"<tr><td>Clarity C80</td><td>{m.clarity_c80_db:.1f} dB</td></tr>"
            f"<tr><td>Definition D50</td><td>{m.definition_d50:.2f}</td></tr>"
            f"<tr><td>Channels measured</td><td>{m.channels_measured}</td></tr>"
            f"<tr><td>Loopback latency</td><td>{m.loopback_latency_ms:.2f} ms</td></tr>"
            f"<tr><td>Mic calibration</td><td>{'Applied' if m.mic_cal_applied else 'None'}</td></tr>"
            f"<tr><td>Clipping detected</td><td>{'Yes' if m.clipping_detected else 'No'}</td></tr>"
        )

    recs = list(data.recommendations)
    if data.measurement and data.measurement.clipping_detected:
        recs.insert(0, Recommendation("high", "Reduce input gain and re-measure — clipping was detected during acquisition."))
    if data.measurement and data.measurement.rt60_broadband_s > 2.0:
        recs.insert(0, Recommendation("high", f"High reverberation (RT60={data.measurement.rt60_broadband_s:.1f}s). Consider additional acoustic treatment before optimization."))
    if data.measurement and data.measurement.rt60_broadband_s < 0.3:
        recs.insert(0, Recommendation("low", "Very dry room — optimization gains may be limited. Consider adding reflections for natural ambience."))
    if data.worst_null_after_db > -6 and data.speakers:
        recs.append(Recommendation("medium", "Residual nulls remain after optimization. Consider adding fill speakers or subwoofers for the affected zones."))
    if data.coherence_gain_db < 3.0:
        recs.append(Recommendation("medium", "Coherence gain is low — verify speaker placement and room geometry before making final adjustments."))
    if not any("on-site" in r.text.lower() for r in recs):
        recs.append(Recommendation("medium", "Run a final on-site measurement to verify alignment before the show."))
    recs.append(Recommendation("low", "Save the session configuration for future recall and comparison."))

    tag_class = {"high": "high", "medium": "med", "low": "med"}
    rec_html = "\n".join(
        f'<div class="rec"><span class="tag {tag_class.get(r.severity, "med")}">'
        f'{r.severity.title()}</span><span>{r.text}</span></div>'
        for r in recs
    ) or '<p style="color:var(--dim);font-size:13px;">No recommendations generated.</p>'

    subs = {
        "VENUE_NAME": data.venue_name,
        "ROOM_HEIGHT": "8.0",
        "SESSION_META": (
            f"Session {datetime.now():%Y-%m-%d_%H-%M} &middot; "
            f"{data.algorithm_used} &middot; {data.iterations} iterations "
            f"&middot; {data.duration_s / 60:.0f}m {data.duration_s % 60:.0f}s"
        ),
        "COHERENCE_GAIN": f"+{data.coherence_gain_db:.1f} dB",
        "WORST_NULL_BEFORE": f"{data.worst_null_before_db:.1f} dB",
        "WORST_NULL_AFTER": f"{data.worst_null_after_db:.1f} dB",
        "FLOOR_RISK_BEFORE": f"{data.floor_below_10db_before_pct:.0f}%",
        "FLOOR_RISK_AFTER": f"{data.floor_below_10db_after_pct:.0f}%",
        "ROOM_W": str(data.room_width_m),
        "ROOM_D": str(data.room_depth_m),
        "SPEED_OF_SOUND": str(data.speed_of_sound),
        "HEATMAP_FREQ": str(int(data.heatmap_freq_hz)),
        "SPEAKERS_BEFORE_JSON": speakers_before,
        "SPEAKERS_AFTER_JSON": speakers_after,
        "ALGOS_JSON": algos_json,
        "ALGO_COUNT": str(len(data.algorithm_comparison)),
        "DELAY_ROWS": delay_rows,
        "MEASUREMENT_ROWS": measurement_rows,
        "RECOMMENDATIONS_HTML": rec_html,
        "HAS_MEASUREMENT": "true" if data.measurement else "false",
    }

    rendered = Template(template_text).safe_substitute(subs)
    output = Path(output_path)
    output.write_text(rendered, encoding="utf-8")
    return output


def open_in_browser(path: str | Path) -> None:
    """Open a generated report in the default browser."""
    webbrowser.open(str(Path(path).resolve()))
