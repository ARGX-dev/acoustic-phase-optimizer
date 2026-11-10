"""DSP setting sheets: the values an engineer types into (or imports to) a processor.

Nothing here pretends to speak a vendor's network protocol. The output is a
plain, complete settings sheet per processor output, in CSV, JSON and text.
"""

from __future__ import annotations

import csv
import io
import json
from datetime import datetime

from . import __version__
from .engine.air import speed_of_sound
from .engine.model import Project

FILTER_NAMES = {"bell": "Bell", "low_shelf": "Low shelf", "high_shelf": "High shelf",
                "highpass": "High-pass", "lowpass": "Low-pass"}
ROLE_NAMES = {"main": "Main", "fill": "Fill", "front_fill": "Front fill", "delay": "Delay", "sub": "Sub"}


def outputs(project: Project) -> list[dict]:
    env = project.environment
    c = speed_of_sound(env.temperature_c, env.humidity_pct, env.pressure_kpa)
    rows = []
    for n, s in enumerate(project.speakers, start=1):
        rows.append({
            "output": n,
            "name": s.name,
            "role": ROLE_NAMES.get(s.role, s.role),
            "delay_ms": round(s.delay_ms, 2),
            "delay_m": round(s.delay_ms * c / 1000.0, 2),
            "delay_ft": round(s.delay_ms * c / 1000.0 / 0.3048, 2),
            "gain_db": round(s.gain_db, 1),
            "polarity": "Inverted" if s.invert else "Normal",
            "mute": bool(s.mute),
            "hpf_hz": s.hp_hz or None,
            "lpf_hz": s.lp_hz or None,
            "peq": [{"type": FILTER_NAMES.get(b.type, b.type), "freq_hz": round(b.freq, 1),
                     "gain_db": round(b.gain_db, 1), "q": round(b.q, 2)}
                    for b in s.peq if b.enabled],
        })
    return rows


def to_json(project: Project) -> str:
    env = project.environment
    doc = {
        "venue": project.name,
        "generated": datetime.now().isoformat(timespec="seconds"),
        "generator": f"Acoustic Phase Optimizer {__version__}",
        "conditions": {"temperature_c": env.temperature_c, "humidity_pct": env.humidity_pct,
                       "speed_of_sound_m_s": round(speed_of_sound(env.temperature_c, env.humidity_pct, env.pressure_kpa), 2)},
        "outputs": outputs(project),
    }
    return json.dumps(doc, indent=2)


def to_csv(project: Project) -> str:
    rows = outputs(project)
    n_peq = max((len(r["peq"]) for r in rows), default=0)
    head = ["Output", "Name", "Role", "Delay (ms)", "Delay (m)", "Delay (ft)", "Gain (dB)", "Polarity", "Mute",
            "HPF (Hz)", "LPF (Hz)"]
    for i in range(1, n_peq + 1):
        head += [f"PEQ{i} Type", f"PEQ{i} Freq (Hz)", f"PEQ{i} Gain (dB)", f"PEQ{i} Q"]
    buf = io.StringIO()
    w = csv.writer(buf, lineterminator="\n")
    w.writerow(head)
    for r in rows:
        line = [r["output"], r["name"], r["role"], f"{r['delay_ms']:.2f}", f"{r['delay_m']:.2f}", f"{r['delay_ft']:.2f}",
                f"{r['gain_db']:.1f}", r["polarity"], "Yes" if r["mute"] else "No", r["hpf_hz"] or "", r["lpf_hz"] or ""]
        for b in r["peq"]:
            line += [b["type"], b["freq_hz"], f"{b['gain_db']:.1f}", b["q"]]
        w.writerow(line + [""] * (len(head) - len(line)))
    return buf.getvalue()


def to_text(project: Project) -> str:
    env = project.environment
    c = speed_of_sound(env.temperature_c, env.humidity_pct, env.pressure_kpa)
    out = [
        f"DSP SETTINGS - {project.name}",
        f"Generated {datetime.now():%Y-%m-%d %H:%M} by Acoustic Phase Optimizer {__version__}",
        f"Conditions: {env.temperature_c:g} degC, {env.humidity_pct:g} % RH, speed of sound {c:.1f} m/s",
        "",
    ]
    for r in outputs(project):
        out.append(f"OUTPUT {r['output']}: {r['name']}  [{r['role']}]" + ("  ** MUTED **" if r["mute"] else ""))
        out.append(f"  Delay     {r['delay_ms']:8.2f} ms   ({r['delay_m']:.2f} m / {r['delay_ft']:.2f} ft)")
        out.append(f"  Gain      {r['gain_db']:+8.1f} dB")
        out.append(f"  Polarity  {r['polarity']}")
        xo = []
        if r["hpf_hz"]:
            xo.append(f"HPF {r['hpf_hz']:g} Hz LR24")
        if r["lpf_hz"]:
            xo.append(f"LPF {r['lpf_hz']:g} Hz LR24")
        if xo:
            out.append("  Crossover " + ", ".join(xo) + "   (as assumed by the model)")
        if r["peq"]:
            out.append("  PEQ")
            for i, b in enumerate(r["peq"], start=1):
                gain = "" if b["type"] in ("High-pass", "Low-pass") else f"{b['gain_db']:+6.1f} dB  "
                out.append(f"    {i}. {b['type']:<10} {b['freq_hz']:>8.1f} Hz  {gain}Q {b['q']:.2f}")
        else:
            out.append("  PEQ       flat")
        out.append("")
    out.append("Delays are relative: the earliest output sits at 0 ms. Add your processor's own")
    out.append("latency offsets if outputs run through different signal paths.")
    return "\n".join(out) + "\n"


FORMATS = {
    "csv": (to_csv, "text/csv", "csv"),
    "json": (to_json, "application/json", "json"),
    "txt": (to_text, "text/plain", "txt"),
}
