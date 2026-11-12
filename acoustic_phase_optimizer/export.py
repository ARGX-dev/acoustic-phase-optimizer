"""DSP setting sheets: the values an engineer types into (or imports to) a processor.

Nothing here pretends to speak a vendor's network protocol. The output is a
plain, complete settings sheet per processor output (CSV, JSON, text), a sheet
laid out for the dbx DriveRack VENU360 (dbx publishes no file it can import, so
it is typed in), and filter files in two documented text formats that tools do
import: REW "Generic" / Equalizer APO filter lines and miniDSP biquad
coefficients.
"""

from __future__ import annotations

import csv
import io
import json
import re
import zipfile
from datetime import datetime

from . import __version__
from .engine.air import speed_of_sound
from .engine.eq import biquad_coeffs
from .engine.model import Project

FILTER_NAMES = {"bell": "Bell", "low_shelf": "Low shelf", "high_shelf": "High shelf",
                "highpass": "High-pass", "lowpass": "Low-pass"}
ROLE_NAMES = {"main": "Main", "fill": "Fill", "front_fill": "Front fill", "delay": "Delay", "sub": "Sub"}


def outputs(project: Project) -> list[dict]:
    env = project.environment
    c = speed_of_sound(env.temperature_c, env.humidity_pct, env.pressure_kpa)
    rows = []
    for n, s in enumerate(project.speakers, start=1):
        # What the processor output must actually do: the alignment delay plus any sub-array pattern offset,
        # and the polarity after both flips. A rear cardioid box exported as "0 ms, normal" would be wrong.
        d = s.delay_ms + s.array_delay_ms
        rows.append({
            "output": n,
            "name": s.name,
            "role": ROLE_NAMES.get(s.role, s.role),
            "delay_ms": round(d, 2),
            "delay_m": round(d * c / 1000.0, 2),
            "delay_ft": round(d * c / 1000.0 / 0.3048, 2),
            "gain_db": round(s.gain_db, 1),
            "polarity": "Inverted" if s.invert != s.array_invert else "Normal",
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


# ------------------------------------------------------------------------------- dbx DriveRack VENU360
# From dbx's published specifications: 3 inputs x 6 outputs, alignment delay up to 1000 ms per output, output
# PEQ Q up to 16. Anything else (how many output PEQ bands are free, which filter types) is left to check.
VENU360_OUTPUTS = 6
VENU360_DELAY_MS = 1000.0
VENU360_PEQ_Q = 16.0
VENU_TYPES = {"bell": "Bell", "low_shelf": "Low Shelf", "high_shelf": "High Shelf",
              "highpass": "High Pass (2nd order)", "lowpass": "Low Pass (2nd order)"}


def venu360_checks(project: Project) -> list[str]:
    out = []
    n = len(project.speakers)
    if n > VENU360_OUTPUTS:
        out.append(f"{n} outputs: one VENU360 has {VENU360_OUTPUTS} (3 in x 6 out). Outputs "
                   f"{VENU360_OUTPUTS + 1}-{n} need a second unit.")
    for r, s in zip(outputs(project), project.speakers):
        if r["delay_ms"] > VENU360_DELAY_MS:
            out.append(f"Output {r['output']} ({r['name']}): {r['delay_ms']:.2f} ms is over the {VENU360_DELAY_MS:g} ms "
                       "alignment delay of one output. Use an input or fill delay as well.")
        for i, b in enumerate((b for b in s.peq if b.enabled), start=1):
            if b.q > VENU360_PEQ_Q:
                out.append(f"Output {r['output']} ({r['name']}): PEQ band {i} has Q {b.q:g}; output PEQ goes up to "
                           f"Q {VENU360_PEQ_Q:g}.")
    return out


def to_venu360(project: Project) -> str:
    env = project.environment
    c = speed_of_sound(env.temperature_c, env.humidity_pct, env.pressure_kpa)
    rows = outputs(project)
    many = len(rows) > VENU360_OUTPUTS
    out = [
        f"dbx DriveRack VENU360 - {project.name}",
        f"Generated {datetime.now():%Y-%m-%d %H:%M} by Acoustic Phase Optimizer {__version__}",
        f"Conditions: {env.temperature_c:g} degC, {env.humidity_pct:g} % RH, speed of sound {c:.1f} m/s",
        "",
        "Enter these in DriveRack VENU360 Control (or on the unit) by hand: dbx publishes no settings file",
        "the VENU360 can import. Each block follows the output's processing chain.",
        "",
    ]
    for r, s in zip(rows, project.speakers):
        unit = (r["output"] - 1) // VENU360_OUTPUTS + 1
        out.append(f"OUTPUT {(r['output'] - 1) % VENU360_OUTPUTS + 1}" + (f" (unit {unit})" if many else "")
                   + f'  "{r["name"]}"  [{r["role"]}]' + ("  ** MUTE **" if r["mute"] else ""))
        xo = []
        if s.hp_hz:
            xo.append(f"High Pass {s.hp_hz:g} Hz, Linkwitz-Riley 24 dB/oct")
        if s.lp_hz:
            xo.append(f"Low Pass {s.lp_hz:g} Hz, Linkwitz-Riley 24 dB/oct")
        out.append("  Crossover        " + ("; ".join(xo) if xo else "Full range (no filters)"))
        bands = [b for b in s.peq if b.enabled]
        if bands:
            out.append(f"  Output PEQ       {len(bands)} band{'s' if len(bands) > 1 else ''}")
            for i, b in enumerate(bands, start=1):
                gain = "" if b.type in ("highpass", "lowpass") else f"  {b.gain_db:+.1f} dB"
                out.append(f"    Band {i}  {VENU_TYPES[b.type]:<22} {b.freq:8.1f} Hz{gain}  Q {b.q:.2f}")
        else:
            out.append("  Output PEQ       Flat (bypass)")
        out.append(f"  Alignment Delay  {r['delay_ms']:.2f} ms   ({r['delay_m']:.2f} m / {r['delay_ft']:.2f} ft)")
        out.append(f"  Gain             {r['gain_db']:+.1f} dB")
        out.append(f"  Polarity         {r['polarity']}")
        out.append("")
    out.append("CHECKS")
    out += [f"  ! {x}" for x in venu360_checks(project)] or ["  Within the VENU360's published limits."]
    out.append("  Check on your unit: that each output has as many free PEQ bands as listed, and that shelf")
    out.append("  filters are offered there. Delays are relative: the earliest output sits at 0 ms.")
    return "\n".join(out) + "\n"


# ----------------------------------------------------------------- filter files (REW / Equalizer APO, miniDSP)
def _file_name(n: int, name: str) -> str:
    """'03 Main L_R.txt': numbered, so two outputs with the same name never collide, and no path characters."""
    safe = re.sub(r"[^\w .()+-]", "_", name).strip(" .") or "Output"
    return f"{n:02d} {safe[:60]}.txt"


def _zip(files: list[tuple[str, str]]) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        for name, text in files:
            z.writestr(name, text.replace("\n", "\r\n"))
    return buf.getvalue()


def _active_bands(spk) -> list:
    return [b for b in spk.peq
            if b.enabled and not (b.type in ("bell", "low_shelf", "high_shelf") and abs(b.gain_db) < 1e-3)]


def crossover_biquads(spk, fs: float) -> list[tuple]:
    """Linkwitz-Riley 24 dB/oct as two identical 2nd-order Butterworth sections per filter, as the model assumes."""
    q = 2 ** -0.5
    out = []
    if spk.hp_hz:
        out += [biquad_coeffs("highpass", spk.hp_hz, 0.0, q, fs)] * 2
    if spk.lp_hz:
        out += [biquad_coeffs("lowpass", spk.lp_hz, 0.0, q, fs)] * 2
    return out


def peq_biquads(spk, fs: float) -> list[tuple]:
    return [biquad_coeffs(b.type, b.freq, b.gain_db, b.q, fs) for b in _active_bands(spk)]


APO_TYPES = {"bell": "PK", "low_shelf": "LSC", "high_shelf": "HSC", "highpass": "HPQ", "lowpass": "LPQ"}


def apo_lines(spk) -> list[str]:
    """The output's filters as REW "Generic" / Equalizer APO lines: crossover first, then the PEQ."""
    q = 2 ** -0.5
    lines = []
    for kind, f in (("HPQ", spk.hp_hz), ("LPQ", spk.lp_hz)):
        if f:
            lines += [f"ON {kind} Fc {f:g} Hz Q {q:.4f}"] * 2
    for b in _active_bands(spk):
        t = APO_TYPES[b.type]
        gain = "" if t in ("HPQ", "LPQ") else f" Gain {b.gain_db:.1f} dB"
        lines.append(f"ON {t} Fc {b.freq:g} Hz{gain} Q {b.q:.4f}")
    return [f"Filter {i}: {x}" for i, x in enumerate(lines, start=1)]


def to_rew_apo(project: Project) -> bytes:
    files = []
    for r, s in zip(outputs(project), project.speakers):
        head = [f"# {project.name} - output {r['output']}: {r['name']} [{r['role']}]",
                f"# Acoustic Phase Optimizer {__version__}. Room EQ Wizard 'Generic' / Equalizer APO filter text.",
                "# Crossover: Linkwitz-Riley 24 dB/oct, written as two Butterworth sections (Q 0.7071).",
                "# Bells (PK) and high and low pass (HPQ, LPQ) are the standard 2nd-order filters. Shelves",
                "# (LSC, HSC) use the RBJ cookbook Q; check your equaliser reads a shelf's Q the same way.",
                f"# Polarity: {r['polarity']}." + (" Invert this output in the processor." if r["polarity"] == "Inverted" else ""),
                f"Preamp: {r['gain_db']:.1f} dB",
                f"Delay: {r['delay_ms']:.2f} ms"]
        body = apo_lines(s) or ["# No filters: flat."]
        files.append((_file_name(r["output"], r["name"]), "\n".join(head + body) + "\n"))
    return _zip(files)


def minidsp_text(sections: list[tuple]) -> str:
    """miniDSP 'advanced biquad' text. miniDSP's a1 and a2 are the negated denominator terms:
    y[n] = b0 x[n] + b1 x[n-1] + b2 x[n-2] + a1 y[n-1] + a2 y[n-2]."""
    out = []
    for i, (b, a) in enumerate(sections, start=1):
        out.append(f"biquad{i},\nb0={b[0]:.15g},\nb1={b[1]:.15g},\nb2={b[2]:.15g},\na1={-a[1]:.15g},\na2={-a[2]:.15g},")
    return "\n".join(out) + ("\n" if out else "")


def to_minidsp(project: Project) -> bytes:
    rows = outputs(project)
    files = []
    for fs in (48000, 96000):
        d = f"{fs // 1000} kHz/"
        for r, s in zip(rows, project.speakers):
            name = _file_name(r["output"], r["name"])
            if s.hp_hz or s.lp_hz:
                files.append((d + "crossover/" + name, minidsp_text(crossover_biquads(s, fs))))
            if _active_bands(s):
                files.append((d + "peq/" + name, minidsp_text(peq_biquads(s, fs))))
    readme = [f"miniDSP biquads - {project.name}", f"Acoustic Phase Optimizer {__version__}", "",
              "Paste a file into the 'Advanced' biquad box of the matching block: crossover/ for the output's",
              "crossover, peq/ for its PEQ. Use the folder for your plugin's sample rate (48 or 96 kHz).",
              "Coefficients follow miniDSP's convention: a1 and a2 are negated, so",
              "  y[n] = b0 x[n] + b1 x[n-1] + b2 x[n-2] + a1 y[n-1] + a2 y[n-2].",
              "If the block has more biquads than a file lists, set the rest to bypass (b0=1, the others 0).",
              "", "Set these by hand in the plugin:"]
    for r in rows:
        readme.append(f"  Output {r['output']} {r['name']}: delay {r['delay_ms']:.2f} ms, gain {r['gain_db']:+.1f} dB, "
                      f"polarity {r['polarity'].lower()}" + (", muted" if r["mute"] else ""))
    files.append(("README.txt", "\n".join(readme) + "\n"))
    return _zip(files)


# format: (writer, media type, file name ending)
FORMATS = {
    "csv": (to_csv, "text/csv", "dsp-settings.csv"),
    "json": (to_json, "application/json", "dsp-settings.json"),
    "txt": (to_text, "text/plain", "dsp-settings.txt"),
    "venu360": (to_venu360, "text/plain", "venu360.txt"),
    "rew": (to_rew_apo, "application/zip", "filters-rew-eqapo.zip"),
    "minidsp": (to_minidsp, "application/zip", "minidsp-biquads.zip"),
}
