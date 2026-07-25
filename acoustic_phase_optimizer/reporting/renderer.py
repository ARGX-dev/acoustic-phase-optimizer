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
        return {
            "label": spk.label,
            "x": spk.x,
            "y": spk.y,
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
    tag_class = {"high": "high", "medium": "med", "low": "med"}
    rec_html = "\n".join(
        f'<div class="rec"><span class="tag {tag_class.get(r.severity, "med")}">'
        f'{r.severity.title()}</span><span>{r.text}</span></div>'
        for r in data.recommendations
    ) or '<p style="color:var(--dim);font-size:13px;">No recommendations generated.</p>'

    subs = {
        "VENUE_NAME": data.venue_name,
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
        "RECOMMENDATIONS_HTML": rec_html,
    }

    rendered = Template(template_text).safe_substitute(subs)
    output = Path(output_path)
    output.write_text(rendered, encoding="utf-8")
    return output


def open_in_browser(path: str | Path) -> None:
    """Open a generated report in the default browser."""
    webbrowser.open(str(Path(path).resolve()))
