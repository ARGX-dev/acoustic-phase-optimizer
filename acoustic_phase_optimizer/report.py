"""Self-contained, printable HTML session report (no external assets)."""

from __future__ import annotations

import base64
import struct
import zlib
from datetime import datetime
from html import escape

import numpy as np

from . import __version__
from .engine.advice import advise
from .engine.materials import MATERIALS, SURFACES
from .engine.model import Project
from .engine.simulate import room_report, simulate
from .export import outputs

# Sequential: one hue, light (low) -> dark (high). Diverging: red <- neutral grey -> blue.
SEQUENTIAL = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
DIVERGING = ["#a32626", "#d03b3b", "#e99a94", "#f0efec", "#9ec5f4", "#2a78d6", "#104281"]
STATUS = {"good": ("#0ca30c", "✓"), "warning": ("#fab219", "!"), "serious": ("#ec835a", "▲"), "critical": ("#d03b3b", "✕")}
WALL_NAMES = {"floor": "Floor", "ceiling": "Ceiling", "front": "Stage wall", "back": "Back wall",
              "left": "Side wall (top of plan)", "right": "Side wall (bottom of plan)"}

METRICS = [  # key, label, unit, decimals, better
    ("mean_spl", "Average level", "dB SPL", 1, None),
    ("coverage", "Coverage evenness (seat-to-seat)", "± dB", 1, "low"),
    ("summation", "Summation between speakers", "dB", 2, "high"),
    ("cancel_pct", "Cancellation (> 3 dB lost)", "%", 1, "low"),
    ("clarity", "Clarity (early vs late)", "dB", 1, "high"),
    ("echo_pct", "Seats with late arrivals (> 30 ms)", "%", 1, "low"),
    ("tonal", "Deviation from target curve", "dB RMS", 2, "low"),
]


def _hex(c: str) -> np.ndarray:
    return np.array([int(c[i:i + 2], 16) for i in (1, 3, 5)], dtype=float)


def colorize(values: np.ndarray, lo: float, hi: float, stops: list[str]) -> np.ndarray:
    t = np.clip((np.nan_to_num(values, nan=lo) - lo) / max(hi - lo, 1e-9), 0.0, 1.0) * (len(stops) - 1)
    table = np.stack([_hex(c) for c in stops])
    i = np.clip(np.floor(t).astype(int), 0, len(stops) - 2)
    f = (t - i)[..., None]
    return np.round(table[i] * (1 - f) + table[i + 1] * f).astype(np.uint8)


def png(rgb: np.ndarray) -> bytes:
    h, w, _ = rgb.shape
    raw = b"".join(b"\x00" + rgb[y].tobytes() for y in range(h))

    def chunk(tag: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data) & 0xFFFFFFFF)

    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(raw, 9)) + chunk(b"IEND", b""))


def _upsample(a: np.ndarray, k: int) -> np.ndarray:
    h, w = a.shape
    xi, yi = np.linspace(0, w - 1, w * k), np.linspace(0, h - 1, h * k)
    tmp = np.array([np.interp(xi, np.arange(w), row) for row in a])
    return np.array([np.interp(yi, np.arange(h), col) for col in tmp.T]).T


def _map_svg(project: Project, sim: dict, key: str, lo: float, hi: float, stops: list[str]) -> str:
    """Plan view with the stage on the left: svg x = distance from stage, svg y = across the room."""
    g, room = sim["grid"], project.room
    vals = np.array([np.nan if v is None else v for v in sim["maps"][key]], dtype=float).reshape(g["ny"], g["nx"])
    img = colorize(_upsample(np.nan_to_num(vals, nan=lo).T, max(1, round(360 / max(g["ny"], 1)))), lo, hi, stops)
    data = base64.b64encode(png(img)).decode()
    sw = max(room.length, room.width) / 260
    parts = [
        f'<svg viewBox="{-sw} {-sw} {room.length + 2 * sw} {room.width + 2 * sw}" role="img" class="plan">',
        f'<rect x="0" y="0" width="{room.length}" height="{room.width}" fill="#f4f3ef" stroke="#898781" stroke-width="{sw}"/>',
        f'<image href="data:image/png;base64,{data}" x="{g["y0"]}" y="{g["x0"]}" width="{g["y1"] - g["y0"]}" '
        f'height="{g["x1"] - g["x0"]}" preserveAspectRatio="none"/>',
    ]
    r = max(room.length, room.width) / 70
    for s in project.speakers:
        if s.mute:
            continue
        parts.append(f'<circle cx="{s.y:.2f}" cy="{s.x:.2f}" r="{r:.2f}" fill="#0b0b0b" stroke="#fcfcfb" stroke-width="{sw}"/>')
    parts.append("</svg>")
    return "".join(parts)


def _legend(lo: float, hi: float, unit: str, stops: list[str], mid: str | None = None) -> str:
    grad = ", ".join(stops)
    centre = f"<span>{escape(mid)}</span>" if mid else "<span></span>"
    return (f'<div class="legend"><div class="bar" style="background:linear-gradient(to right, {grad})"></div>'
            f'<div class="ends"><span>{lo:g} {unit}</span>{centre}<span>{hi:g} {unit}</span></div></div>')


def _fmt(v, nd: int) -> str:
    return "–" if v is None else f"{v:.{nd}f}"


def _delta(before, after, better) -> str:
    if before is None or after is None or better is None or abs(after - before) < 0.05:
        return ""
    improved = (after < before) == (better == "low")
    return f'<span class="{"up" if improved else "down"}">{"▲ better" if improved else "▼ worse"}</span>'


def render_report(project: Project, baseline: Project | None = None) -> str:
    sim = simulate(project)
    base = simulate(baseline) if baseline is not None else None
    room = room_report(project)
    m, mb = sim["metrics"] or {}, (base["metrics"] or {}) if base else None
    e = escape
    out: list[str] = []
    add = out.append

    add(f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{e(project.name)} - system report</title>
<style>
:root{{--ink:#0b0b0b;--ink2:#52514e;--muted:#898781;--line:#e1e0d9;--surface:#fcfcfb;--page:#f9f9f7}}
*{{box-sizing:border-box}}
body{{margin:0;background:var(--page);color:var(--ink);font:14px/1.5 system-ui,-apple-system,"Segoe UI",sans-serif}}
main{{max-width:980px;margin:0 auto;padding:32px 24px 64px}}
h1{{font-size:28px;margin:0 0 4px;letter-spacing:-.01em}} h2{{font-size:17px;margin:36px 0 12px;padding-top:20px;border-top:1px solid var(--line)}}
.sub{{color:var(--ink2);margin:0 0 20px}}
.score{{display:flex;gap:28px;align-items:flex-end;flex-wrap:wrap;margin:8px 0 4px}}
.score .n{{font-size:56px;font-weight:650;line-height:1;letter-spacing:-.02em}} .score .l{{color:var(--ink2);font-size:13px}}
.score .was{{font-size:28px;font-weight:600;color:var(--muted);line-height:1.2}}
table{{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums}}
th,td{{text-align:left;padding:7px 10px;border-bottom:1px solid var(--line);vertical-align:top}}
th{{font-weight:600;color:var(--ink2);font-size:12px}} td.r,th.r{{text-align:right}}
.up{{color:#006300;font-size:12px;margin-left:8px}} .down{{color:#b3261e;font-size:12px;margin-left:8px}}
.maps{{display:grid;grid-template-columns:repeat(auto-fit,minmax(280px,1fr));gap:20px}}
figure{{margin:0;background:var(--surface);border:1px solid var(--line);border-radius:8px;padding:12px}}
figcaption{{font-weight:600;margin-bottom:8px}} figcaption small{{display:block;font-weight:400;color:var(--ink2)}}
svg.plan{{width:100%;height:auto;display:block}}
.legend .bar{{height:8px;border-radius:4px;margin-top:10px}} .legend .ends{{display:flex;justify-content:space-between;font-size:12px;color:var(--ink2);margin-top:3px}}
.find{{display:grid;grid-template-columns:22px 1fr;gap:4px 10px;padding:10px 0;border-bottom:1px solid var(--line)}}
.find .ic{{width:20px;height:20px;border-radius:50%;color:#fff;font-size:12px;font-weight:700;display:grid;place-items:center}}
.find b{{display:block}} .find span{{color:var(--ink2)}}
.note{{color:var(--ink2);font-size:12.5px}} .peq{{color:var(--ink2);font-size:12.5px}}
@media print{{body{{background:#fff}} main{{padding:0}} h2{{break-after:avoid}} figure,tr,.find{{break-inside:avoid}}}}
</style></head><body><main>""")

    add(f"<h1>{e(project.name)}</h1>")
    add(f'<p class="sub">System alignment report · {datetime.now():%d %B %Y, %H:%M} · '
        f'{len(project.speakers)} speakers · {project.room.width:g} × {project.room.length:g} × {project.room.height:g} m</p>')

    add('<div class="score">')
    add(f'<div><div class="n">{_fmt(m.get("score"), 0)}</div><div class="l">System score, out of 100</div></div>')
    if mb:
        add(f'<div><div class="was">{_fmt(mb.get("score"), 0)}</div><div class="l">Before optimization</div></div>')
    add("</div>")

    add("<h2>Findings</h2>")
    for f in advise(sim["metrics"], room):
        colour, icon = STATUS[f["level"]]
        add(f'<div class="find"><div class="ic" style="background:{colour}">{icon}</div>'
            f'<div><b>{e(f["title"])}</b><span>{e(f["text"])}</span></div></div>')

    add("<h2>Results</h2><table><tr><th>Measure</th>" + ('<th class="r">Before</th>' if mb else "")
        + '<th class="r">' + ("After" if mb else "Value") + "</th><th>Unit</th></tr>")
    for key, label, unit, nd, better in METRICS:
        add(f"<tr><td>{label}</td>" + (f'<td class="r">{_fmt(mb.get(key), nd)}</td>' if mb else "")
            + f'<td class="r">{_fmt(m.get(key), nd)}{_delta(mb.get(key), m.get(key), better) if mb else ""}</td><td>{unit}</td></tr>')
    add("</table>")

    if sim["maps"]:
        add("<h2>Coverage maps</h2><p class='note'>Plan view at ear height, stage on the left. Dots are speakers.</p>")
        direct = np.array([v for v in sim["maps"]["direct"] if v is not None] or [0.0])
        hi = float(np.ceil(np.percentile(direct, 99)))
        lo = hi - 18.0
        specs = [("direct", "Direct sound level", "Direct sound and early reflections, 100 Hz – 10 kHz", lo, hi, "dB", SEQUENTIAL, None),
                 ("clarity", "Clarity", "Early energy relative to late reverberant energy", -6.0, 12.0, "dB", SEQUENTIAL, None),
                 ("interference", "Summation and cancellation", "How speakers add (blue) or cancel (red), 1/3-octave smoothed",
                  -6.0, 6.0, "dB", DIVERGING, "0")]
        for key, title, desc, a, b, unit, stops, mid in specs:
            add('<div class="maps">')
            pairs = [("Before", baseline, base), ("After", project, sim)] if base and base["maps"] else [("", project, sim)]
            for tag, proj, s in pairs:
                add(f"<figure><figcaption>{title}{' - ' + tag.lower() if tag else ''}<small>{desc}</small></figcaption>"
                    f"{_map_svg(proj, s, key, a, b, stops)}{_legend(a, b, unit, stops, mid)}</figure>")
            add("</div><br>")

    add("<h2>DSP settings</h2><table><tr><th>#</th><th>Output</th><th>Role</th><th class='r'>Delay (ms)</th>"
        "<th class='r'>Gain (dB)</th><th>Polarity</th><th>Parametric EQ</th></tr>")
    for r in outputs(project):
        peq = "<br>".join(f"{e(b['type'])} {b['freq_hz']:g} Hz, {b['gain_db']:+.1f} dB, Q {b['q']:g}" for b in r["peq"]) or "Flat"
        name = e(r["name"]) + (" <span class='note'>(muted)</span>" if r["mute"] else "")
        add(f"<tr><td>{r['output']}</td><td>{name}</td><td>{e(r['role'])}</td><td class='r'>{r['delay_ms']:.2f}</td>"
            f"<td class='r'>{r['gain_db']:+.1f}</td><td>{r['polarity']}</td><td class='peq'>{peq}</td></tr>")
    add("</table>")

    env = project.environment
    add("<h2>Room</h2><table>")
    add(f"<tr><td>Volume</td><td class='r'>{room['volume_m3']:,.0f} m³</td><td>Speed of sound</td>"
        f"<td class='r'>{room['speed_of_sound']:.1f} m/s at {env.temperature_c:g} °C, {env.humidity_pct:g}% RH</td></tr>")
    add(f"<tr><td>Predicted RT60 (500 Hz – 1 kHz)</td><td class='r'>{room['rt60_mid']:.2f} s</td>"
        f"<td>Critical distance at 1 kHz (omni source)</td><td class='r'>{room['critical_distance_m'][3]:.1f} m</td></tr></table>")
    add("<br><table><tr><th>Octave band</th>" + "".join(f"<th class='r'>{int(b)} Hz</th>" for b in room["bands"]) + "</tr>")
    add("<tr><td>RT60, Eyring (s)</td>" + "".join(f"<td class='r'>{v:.2f}</td>" for v in room["rt60_eyring"]) + "</tr></table>")
    mats = project.room.materials.model_dump()
    add("<br><table><tr><th>Surface</th><th>Material</th></tr>" + "".join(
        f"<tr><td>{WALL_NAMES[s]}</td><td>{e(MATERIALS[mats[s]].name if mats[s] in MATERIALS else mats[s])}</td></tr>"
        for s in SURFACES) + "</table>")

    if project.notes.strip():
        add(f"<h2>Notes</h2><p>{e(project.notes).replace(chr(10), '<br>')}</p>")

    add(f"""<h2>How these numbers were made</h2>
<p class="note">This is a prediction, not a measurement. The room is modelled as a rectangular box using image-source
reflections up to order {project.settings.reflection_order}, a statistical reverberant tail, ISO 9613-1 air absorption and
generic loudspeaker directivity derived from each speaker's nominal coverage angles. Real loudspeakers, audiences, balconies
and furniture will differ. Always verify delay times and equalisation by measurement before a show.</p>
<p class="note">Acoustic Phase Optimizer {__version__}</p></main></body></html>""")
    return "".join(out)
