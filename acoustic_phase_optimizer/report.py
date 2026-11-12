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
from .engine.model import Branding, Project
from .engine.simulate import point_response, room_report, section, simulate
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


LINE_COLOURS = ["#2a78d6", "#d95926", "#0c8a0c", "#8a4fd8", "#c4339a", "#a87400", "#00897b", "#5f6b7a"]


def _resample(freqs: list[float], values: list[float | None], grid: np.ndarray, fraction: float = 6.0) -> list[float | None]:
    """A measurement on the plot's frequencies, power-averaged over 1/6 octave as the response chart draws it."""
    f = np.asarray(freqs, float)
    v = np.array([np.nan if x is None else x for x in values], float)
    lf, half = np.log2(f), 0.5 / fraction
    out: list[float | None] = []
    for x in grid:
        if x < f[0] or x > f[-1]:
            out.append(None)
            continue
        c = np.log2(x)
        sel = (lf >= c - half) & (lf <= c + half) & np.isfinite(v)
        if sel.any():
            out.append(float(10 * np.log10(np.mean(10 ** (v[sel] / 10)))))
        else:
            ok = np.isfinite(v)
            out.append(float(np.interp(c, lf[ok], v[ok])) if ok.any() else None)
    return out


def _response_svg(freqs: np.ndarray, series: list[dict], f_lo: float, f_hi: float) -> str:
    """Magnitude plot: log frequency, 36 dB tall. series: {name, values, colour, dash, width}."""
    W, H, L, R, T, B = 640, 250, 42, 12, 10, 28
    keep = (freqs >= f_lo) & (freqs <= f_hi)
    vals = [v for s in series for v, k in zip(s["values"], keep) if k and v is not None and np.isfinite(v)]
    top = float(np.ceil((max(vals) if vals else 0) / 6) * 6 + 3)
    lo = top - 36

    def X(f):
        return L + (np.log10(f) - np.log10(f_lo)) / (np.log10(f_hi) - np.log10(f_lo)) * (W - L - R)

    def Y(db):
        return T + (top - min(max(db, lo), top)) / (top - lo) * (H - T - B)

    p = [f'<svg viewBox="0 0 {W} {H}" class="resp" role="img" aria-label="Frequency response">',
         f'<rect x="{L}" y="{T}" width="{W - L - R}" height="{H - T - B}" fill="#fff" stroke="#e1e0d9"/>']
    for f, lab in [(31.5, "31.5"), (63, "63"), (125, "125"), (250, "250"), (500, "500"), (1000, "1k"), (2000, "2k"),
                   (4000, "4k"), (8000, "8k"), (16000, "16k")]:
        if f_lo <= f <= f_hi:
            x = X(f)
            p.append(f'<line x1="{x:.1f}" y1="{T}" x2="{x:.1f}" y2="{H - B}" stroke="#eeede8"/>'
                     f'<text x="{x:.1f}" y="{H - B + 16}" text-anchor="middle">{lab}</text>')
    db = lo
    while db <= top + 1e-9:
        y = Y(db)
        p.append(f'<line x1="{L}" y1="{y:.1f}" x2="{W - R}" y2="{y:.1f}" stroke="#eeede8"/>'
                 f'<text x="{L - 6}" y="{y + 4:.1f}" text-anchor="end">{db:g}</text>')
        db += 6
    for s in series:
        pts, seg = [], []
        for f, v, k in zip(freqs, s["values"], keep):
            if not k or v is None or not np.isfinite(v):
                if seg:
                    pts.append(seg)
                    seg = []
                continue
            seg.append(f"{X(f):.1f},{Y(v):.1f}")
        if seg:
            pts.append(seg)
        dash = ' stroke-dasharray="6 4"' if s.get("dash") else ""
        for sg in pts:
            p.append(f'<polyline points="{" ".join(sg)}" fill="none" stroke="{s["colour"]}" stroke-width="{s.get("width", 1.6)}"'
                     f'{dash} stroke-linejoin="round"/>')
    p.append("</svg>")
    keys = "".join(f'<span><i style="background:{s["colour"]}{";opacity:.7" if s.get("dash") else ""}"></i>{escape(s["name"])}</span>'
                   for s in series)
    return "".join(p) + f'<div class="keys">{keys}</div>'


def _section_svg(project: Project, sec: dict, lo: float, hi: float) -> str:
    """Long section through the middle of the room: stage on the left, floor at the bottom."""
    vals = np.array([np.nan if v is None else v for v in sec["values"]], float).reshape(sec["nz"], sec["nu"])[::-1]
    img = colorize(_upsample(np.nan_to_num(vals, nan=lo), max(1, round(240 / max(sec["nz"], 1)))), lo, hi, SEQUENTIAL)
    data = base64.b64encode(png(img)).decode()
    span, height = sec["span"], sec["height"]
    sw = max(span, height) / 300
    r = max(span, height) / 90
    parts = [f'<svg viewBox="{-sw} {-sw} {span + 2 * sw} {height + 2 * sw}" role="img" class="plan">',
             f'<rect x="0" y="0" width="{span}" height="{height}" fill="#f4f3ef" stroke="#898781" stroke-width="{sw}"/>',
             f'<image href="data:image/png;base64,{data}" x="{sec["u0"]}" y="{height - sec["z1"]}" width="{sec["u1"] - sec["u0"]}" '
             f'height="{sec["z1"] - sec["z0"]}" preserveAspectRatio="none"/>']
    for s in project.speakers:
        if not s.mute:
            parts.append(f'<circle cx="{s.y:.2f}" cy="{height - s.z:.2f}" r="{r:.2f}" fill="#0b0b0b" stroke="#fcfcfb" '
                         f'stroke-width="{sw}"/>')
    parts.append("</svg>")
    return "".join(parts)


def _listening_positions(project: Project) -> list[tuple[str, float, float, float | None, str | None]]:
    if project.mics:
        return [(m.name, m.x, m.y, m.z, m.id) for m in project.mics[:8]]
    a = project.audience
    return [("Middle of the audience", project.room.width / 2, (a.y0 + a.y1) / 2, None, None)]


def _css_string(text: str) -> str:
    """Text as a CSS string literal: anything but plain characters escaped, so it cannot end the string or the
    style element."""
    return '"' + "".join(c if c.isalnum() or c in " .,-·()&'" else f"\\{ord(c):x} " for c in text) + '"'


def render_report(project: Project, baseline: Project | None = None, branding: Branding | None = None) -> str:
    sim = simulate(project)
    base = simulate(baseline) if baseline is not None else None
    room = room_report(project)
    m, mb = sim["metrics"] or {}, (base["metrics"] or {}) if base else None
    br = branding or Branding()
    e = escape
    now = datetime.now()
    sections: list[tuple[str, str, str]] = []  # (anchor, title, html)

    # ------------------------------------------------------------------ summary
    out = ['<div class="score">',
           f'<div><div class="n">{_fmt(m.get("score"), 0)}</div><div class="l">System score, out of 100</div></div>']
    if mb:
        out.append(f'<div><div class="was">{_fmt(mb.get("score"), 0)}</div><div class="l">Before optimization</div></div>')
    out.append("</div>")
    for f in advise(sim["metrics"], room):
        colour, icon = STATUS[f["level"]]
        out.append(f'<div class="find"><div class="ic" style="background:{colour}">{icon}</div>'
                   f'<div><b>{e(f["title"])}</b><span>{e(f["text"])}</span></div></div>')
    sections.append(("summary", "Summary and findings", "".join(out)))

    out = ["<table><tr><th>Measure</th>" + ('<th class="r">Before</th>' if mb else "")
           + '<th class="r">' + ("After" if mb else "Value") + "</th><th>Unit</th></tr>"]
    for key, label, unit, nd, better in METRICS:
        out.append(f"<tr><td>{label}</td>" + (f'<td class="r">{_fmt(mb.get(key), nd)}</td>' if mb else "")
                   + f'<td class="r">{_fmt(m.get(key), nd)}{_delta(mb.get(key), m.get(key), better) if mb else ""}</td><td>{unit}</td></tr>')
    out.append("</table>")
    sections.append(("results", "Results", "".join(out)))

    # ------------------------------------------------------------------ maps and section
    if sim["maps"]:
        out = ["<p class='note'>Plan view at ear height, stage on the left. Dots are speakers.</p>"]
        direct = np.array([v for v in sim["maps"]["direct"] if v is not None] or [0.0])
        hi = float(np.ceil(np.percentile(direct, 99)))
        lo = hi - 18.0
        specs = [("direct", "Direct sound level", "Direct sound and early reflections, 100 Hz – 10 kHz", lo, hi, "dB", SEQUENTIAL, None),
                 ("clarity", "Clarity", "Early energy relative to late reverberant energy", -6.0, 12.0, "dB", SEQUENTIAL, None),
                 ("interference", "Summation and cancellation", "How speakers add (blue) or cancel (red), 1/3-octave smoothed",
                  -6.0, 6.0, "dB", DIVERGING, "0")]
        for key, title, desc, a, b, unit, stops, mid in specs:
            out.append('<div class="maps">')
            pairs = [("Before", baseline, base), ("After", project, sim)] if base and base["maps"] else [("", project, sim)]
            for tag, proj, s in pairs:
                out.append(f"<figure><figcaption>{title}{' - ' + tag.lower() if tag else ''}<small>{desc}</small></figcaption>"
                           f"{_map_svg(proj, s, key, a, b, stops)}{_legend(a, b, unit, stops, mid)}</figure>")
            out.append("</div>")
        sections.append(("maps", "Coverage maps", "".join(out)))

        if project.speakers:
            sec = section(project, "length")
            sv = np.array([v for v in sec["values"] if v is not None] or [0.0])
            shi = float(np.ceil(np.percentile(sv, 99)))
            sections.append(("section", "Long section", (
                f"<figure><figcaption>Direct sound level through the middle of the room<small>Stage on the left, floor at "
                f"the bottom, {sec['at']:g} m from the left wall. Dots are speakers, drawn at their height.</small></figcaption>"
                f"{_section_svg(project, sec, shi - 18, shi)}{_legend(shi - 18, shi, 'dB', SEQUENTIAL)}</figure>")))

    # ------------------------------------------------------------------ responses at the listening positions
    if project.speakers:
        st = project.settings
        out = ["<p class='note'>Predicted response of all speakers together, 1/6-octave smoothed. "
               + ("Measurements taken at a position are drawn dashed, with the level they were matched to on import."
                  if project.measurements else "") + "</p><div class='resps'>"]
        for name, x, y, z, mic_id in _listening_positions(project):
            pr = point_response(project, x, y, z)
            freqs = np.asarray(pr["freqs"], float)
            series = [{"name": "Predicted", "values": pr["spl_smooth"], "colour": "#0b0b0b", "width": 2}]
            for ms in project.measurements:
                if mic_id and ms.mic_id == mic_id:
                    vals = _resample(ms.freqs, ms.spl, freqs)
                    series.append({"name": f"{ms.name} (measured)", "dash": True, "colour": LINE_COLOURS[ms.color % 8],
                                   "values": [None if v is None else v + ms.offset_db for v in vals]})
            out.append(f"<figure><figcaption>{e(name)}<small>{pr['x']:.1f} m across, {pr['y']:.1f} m from the stage, "
                       f"first sound at {pr['first_arrival_ms']:.1f} ms</small></figcaption>"
                       f"{_response_svg(freqs, series, max(20.0, st.f_min), min(20000.0, st.f_max))}</figure>")
        out.append("</div>")
        sections.append(("responses", "Frequency response at the listening positions", "".join(out)))

    # ------------------------------------------------------------------ DSP
    out = ["<table><tr><th>#</th><th>Output</th><th>Role</th><th class='r'>Delay (ms)</th>"
           "<th class='r'>Gain (dB)</th><th>Polarity</th><th>Parametric EQ</th></tr>"]
    for r in outputs(project):
        peq = "<br>".join(f"{e(b['type'])} {b['freq_hz']:g} Hz, {b['gain_db']:+.1f} dB, Q {b['q']:g}" for b in r["peq"]) or "Flat"
        name = e(r["name"]) + (" <span class='note'>(muted)</span>" if r["mute"] else "")
        out.append(f"<tr><td>{r['output']}</td><td>{name}</td><td>{e(r['role'])}</td><td class='r'>{r['delay_ms']:.2f}</td>"
                   f"<td class='r'>{r['gain_db']:+.1f}</td><td>{r['polarity']}</td><td class='peq'>{peq}</td></tr>")
    out.append("</table>")
    sections.append(("dsp", "DSP settings", "".join(out)))

    # ------------------------------------------------------------------ room
    env = project.environment
    out = ["<table>",
           f"<tr><td>Volume</td><td class='r'>{room['volume_m3']:,.0f} m³</td><td>Speed of sound</td>"
           f"<td class='r'>{room['speed_of_sound']:.1f} m/s at {env.temperature_c:g} °C, {env.humidity_pct:g}% RH</td></tr>",
           f"<tr><td>Predicted RT60 (500 Hz – 1 kHz)</td><td class='r'>{room['rt60_mid']:.2f} s</td>"
           f"<td>Critical distance at 1 kHz (omni source)</td><td class='r'>{room['critical_distance_m'][3]:.1f} m</td></tr></table>",
           "<br><table><tr><th>Octave band</th>" + "".join(f"<th class='r'>{int(b)} Hz</th>" for b in room["bands"]) + "</tr>"]
    shaped = room.get("model") == "shape"
    rt_label = "RT60, ray-traced in the scanned shape (s)" if shaped else "RT60, Eyring (s)"
    out.append(f"<tr><td>{rt_label}</td>" + "".join(f"<td class='r'>{v:.2f}</td>" for v in room.get("rt60", room["rt60_eyring"]))
               + "</tr></table>")
    mats = project.room.materials.model_dump()

    def mat_name(k: str) -> str:
        return MATERIALS[k].name if k in MATERIALS else k

    if shaped and project.room.shape is not None:
        if room.get("escaped", 0) > 0.05:
            out.append(f"<p>{room['escaped'] * 100:.0f}% of the traced sound left the model through openings.</p>")
        rows = []
        for s in project.room.shape.surfaces:
            b = mats["left"] if s.role == "other" else mats[s.role]
            rows.append(f"<tr><td>{e(s.name)}</td><td class='r'>{s.area:,.0f} m²</td><td>{e(mat_name(s.material or b))}</td></tr>")
        out.append("<br><table><tr><th>Surface (scanned shape)</th><th class='r'>Area</th><th>Material</th></tr>" + "".join(rows)
                   + "</table>")
    else:
        out.append("<br><table><tr><th>Surface</th><th>Material</th></tr>" + "".join(
            f"<tr><td>{WALL_NAMES[s]}</td><td>{e(mat_name(mats[s]))}</td></tr>" for s in SURFACES) + "</table>")
    sections.append(("room", "Room", "".join(out)))

    if project.notes.strip():
        sections.append(("notes", "Notes", f"<p>{e(project.notes).replace(chr(10), '<br>')}</p>"))

    sections.append(("method", "How these numbers were made", f"""<p class="note">This is a prediction, not a measurement. The
room is modelled using image-source reflections up to order {project.settings.reflection_order}, a statistical reverberant
tail, ISO 9613-1 air absorption and generic loudspeaker directivity derived from each speaker's nominal coverage angles. Real
loudspeakers, audiences, balconies and furniture will differ. Always verify delay times and equalisation by measurement before
a show.</p><p class="note">Acoustic Phase Optimizer {__version__}</p>"""))

    # ------------------------------------------------------------------ assemble
    info = project.report
    company = br.company.strip()
    by = " · ".join(e(x) for x in (company, br.engineer.strip()) if x)
    runner = _css_string(" · ".join(x for x in (company, project.name, f"{now:%d %B %Y}") if x))
    cover = ['<section class="cover">']
    if br.logo:
        cover.append(f'<img class="logo" src="{br.logo}" alt="{e(company or "Logo")}">')
    elif company:
        cover.append(f'<div class="company">{e(company)}</div>')
    cover.append('<div class="kicker">Sound system design report</div>')
    cover.append(f"<h1>{e(project.name)}</h1>")
    facts = []
    if info.client.strip():
        facts.append(("Prepared for", e(info.client.strip())))
    if info.reference.strip():
        facts.append(("Reference", e(info.reference.strip())))
    facts.append(("Date", f"{now:%d %B %Y}"))
    if by:
        facts.append(("Prepared by", by))
    if br.contact.strip():
        facts.append(("Contact", e(br.contact.strip())))
    facts.append(("System", f"{len(project.speakers)} speakers · {project.room.width:g} × {project.room.length:g} × "
                            f"{project.room.height:g} m room"))
    cover.append('<dl class="facts">' + "".join(f"<dt>{k}</dt><dd>{v}</dd>" for k, v in facts) + "</dl>")
    cover.append(f'<div class="cover-score"><span class="n">{_fmt(m.get("score"), 0)}</span><span class="l">System score, '
                 f'out of 100{" (was " + _fmt(mb.get("score"), 0) + " before optimization)" if mb else ""}</span></div>')
    cover.append("<nav><h2 class='toc-h'>Contents</h2><ol>"
                 + "".join(f'<li><a href="#{a}">{t}</a></li>' for a, t, _ in sections) + "</ol></nav>")
    cover.append("</section>")

    accent = br.accent
    head = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{e(project.name)} - system report</title>
<style>
:root{{--ink:#0b0b0b;--ink2:#52514e;--muted:#898781;--line:#e1e0d9;--surface:#fcfcfb;--page:#f9f9f7;--accent:{accent}}}
*{{box-sizing:border-box}}
html{{-webkit-print-color-adjust:exact;print-color-adjust:exact}}
body{{margin:0;background:var(--page);color:var(--ink);font:14px/1.5 system-ui,-apple-system,"Segoe UI",sans-serif}}
main{{max-width:980px;margin:0 auto;padding:32px 24px 64px}}
h1{{font-size:34px;margin:0 0 20px;letter-spacing:-.01em;line-height:1.15}}
h2{{font-size:18px;margin:36px 0 12px;padding-top:14px;border-top:3px solid var(--accent)}}
h2 .num{{color:var(--accent);margin-right:10px}}
.cover{{min-height:88vh;display:flex;flex-direction:column;gap:18px;border-left:8px solid var(--accent);padding:8px 0 8px 28px}}
.cover .logo{{max-width:260px;max-height:110px;object-fit:contain;align-self:flex-start}}
.cover .company{{font-size:20px;font-weight:650;color:var(--accent)}}
.cover .kicker{{text-transform:uppercase;letter-spacing:.12em;font-size:12px;color:var(--ink2);margin-top:auto}}
.facts{{display:grid;grid-template-columns:max-content 1fr;gap:6px 24px;margin:0}}
.facts dt{{color:var(--ink2)}} .facts dd{{margin:0;font-weight:550}}
.cover-score{{display:flex;align-items:baseline;gap:14px}} .cover-score .n{{font-size:56px;font-weight:650;color:var(--accent);line-height:1}}
.cover-score .l{{color:var(--ink2)}}
.cover nav{{margin-top:auto}} .cover nav ol{{margin:0;padding-left:22px;columns:2;column-gap:32px}}
.cover nav a{{color:var(--ink);text-decoration:none}} .toc-h{{border:0;padding:0;margin:0 0 6px;font-size:13px;color:var(--ink2)}}
.score{{display:flex;gap:28px;align-items:flex-end;flex-wrap:wrap;margin:8px 0 4px}}
.score .n{{font-size:56px;font-weight:650;line-height:1;letter-spacing:-.02em}} .score .l{{color:var(--ink2);font-size:13px}}
.score .was{{font-size:28px;font-weight:600;color:var(--muted);line-height:1.2}}
table{{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums}}
th,td{{text-align:left;padding:7px 10px;border-bottom:1px solid var(--line);vertical-align:top}}
th{{font-weight:600;color:var(--ink2);font-size:12px}} td.r,th.r{{text-align:right}}
.up{{color:#006300;font-size:12px;margin-left:8px}} .down{{color:#b3261e;font-size:12px;margin-left:8px}}
.maps,.resps{{display:grid;grid-template-columns:repeat(auto-fit,minmax(280px,1fr));gap:20px;margin-bottom:20px}}
.resps{{grid-template-columns:repeat(auto-fit,minmax(420px,1fr))}}
figure{{margin:0;background:var(--surface);border:1px solid var(--line);border-radius:8px;padding:12px}}
figcaption{{font-weight:600;margin-bottom:8px}} figcaption small{{display:block;font-weight:400;color:var(--ink2)}}
svg.plan,svg.resp{{width:100%;height:auto;display:block}} svg.resp text{{font-size:13px;fill:#52514e}}
.keys{{display:flex;flex-wrap:wrap;gap:4px 14px;font-size:12px;color:var(--ink2);margin-top:6px}}
.keys i{{display:inline-block;width:14px;height:3px;border-radius:2px;margin-right:6px;vertical-align:middle}}
.legend .bar{{height:8px;border-radius:4px;margin-top:10px}} .legend .ends{{display:flex;justify-content:space-between;font-size:12px;color:var(--ink2);margin-top:3px}}
.find{{display:grid;grid-template-columns:22px 1fr;gap:4px 10px;padding:10px 0;border-bottom:1px solid var(--line)}}
.find .ic{{width:20px;height:20px;border-radius:50%;color:#fff;font-size:12px;font-weight:700;display:grid;place-items:center}}
.find b{{display:block}} .find span{{color:var(--ink2)}}
.note{{color:var(--ink2);font-size:12.5px}} .peq{{color:var(--ink2);font-size:12.5px}}
@page{{size:A4;margin:16mm 14mm 18mm}}
@page{{@bottom-right{{content:counter(page) " / " counter(pages);font:9px system-ui,sans-serif;color:#898781}}
  @bottom-left{{content:{runner};font:9px system-ui,sans-serif;color:#898781}}}}
@page:first{{@bottom-left{{content:none}}@bottom-right{{content:none}}}}
@media print{{body{{background:#fff;font-size:12.5px}} main{{padding:0;max-width:none}}
  .cover{{min-height:255mm;break-after:page}} h2{{break-after:avoid}} figure,tr,.find{{break-inside:avoid}}
  .resps{{grid-template-columns:1fr}} .maps{{grid-template-columns:1fr 1fr}}
  #maps,#responses,#dsp{{break-before:page}}}}
</style></head><body><main>"""
    body = [head, "".join(cover)]
    for n, (anchor, title, html) in enumerate(sections, start=1):
        body.append(f'<section class="part" id="{anchor}"><h2><span class="num">{n}</span>{title}</h2>{html}</section>')
    body.append("</main></body></html>")
    return "".join(body)
