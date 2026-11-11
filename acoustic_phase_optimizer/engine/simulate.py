"""Turn a project into everything the UI draws: maps, metrics, responses."""

from __future__ import annotations

import time

import numpy as np

from .air import air_absorption_db_per_m, speed_of_sound
from .eq import total_delay_ms
from .field import (
    MIN_DISTANCE,
    Context,
    box_response,
    combine,
    dsp_weights,
    freq_grid,
    get_field,
    image_sources,
    log_directivity,
    make_context,
    source_position,
    speaker_amplitude,
    speaker_field,
)
from .materials import OCTAVE_BANDS, room_stats, surface_alphas
from .metrics import BANDS, analyse, arrivals, band_levels, db, smooth, spl, summarise, tone_floor
from .model import Project

F_LO, F_HI = 100.0, 10000.0


def _r(a, nd: int = 1) -> list:
    a = np.asarray(a, dtype=float)
    return [None if not np.isfinite(v) else round(float(v), nd) for v in a.ravel()]


def simulate(project: Project, preview: bool = False) -> dict:
    """Coverage maps, metrics and mic responses for the project's current DSP state."""
    t0 = time.perf_counter()
    field = get_field(project, preview)
    t_field = time.perf_counter() - t0
    freqs, g = field.freqs, field.grid_slice
    out: dict = {
        "grid": {"nx": field.grid.nx, "ny": field.grid.ny,
                 "x0": float(field.grid.xs[0]), "x1": float(field.grid.xs[-1]),
                 "y0": float(field.grid.ys[0]), "y1": float(field.grid.ys[-1])},
        "bands": BANDS,
        "speed_of_sound": round(field.c, 2),
        "preview": preview,
        "freqs": _r(freqs, 2),
        "maps": {}, "metrics": None, "mics": [],
    }
    if not any(not s.mute for s in project.speakers):
        out["timing"] = {"field_ms": round(t_field * 1000, 1), "total_ms": round((time.perf_counter() - t0) * 1000, 1)}
        return out

    w = dsp_weights(project.speakers, freqs)
    full = combine(field, w)
    c = full.rows(g)
    delays = np.array([total_delay_ms(s) for s in project.speakers])
    _, _, spread, dominant = arrivals(field, w, delays, g)
    a = analyse(c.early, c.early_psum, c.late, spread, freqs,
                project.settings.target_tilt_db_per_oct, tone_floor(project.speakers))
    bb = (freqs >= F_LO) & (freqs <= F_HI)
    direct_bands = band_levels(c.early, freqs)
    out["maps"] = {
        "direct": _r(spl(c.early[:, bb].mean(1))),
        **{f"direct_{int(b)}": _r(direct_bands[:, i]) for i, b in enumerate(BANDS)},
        "total": _r(spl(c.total[:, bb].mean(1))),
        "interference": _r(a["_rip_point"], 2),
        "clarity": _r(a["_clarity_point"], 1),
        "spread": _r(spread, 1),
        "dominant": [int(v) for v in dominant],
    }
    out["metrics"] = summarise(a)
    out["average"] = {  # audience-average direct response and the target it is judged against
        "spl": _r(spl(smooth(c.early.mean(0), freqs)), 2),
        "total": _r(spl(smooth(c.total.mean(0), freqs)), 2),
    }
    total = full.total
    for j, m in enumerate(project.mics):
        p = field.n_grid + j
        out["mics"].append({"id": m.id, "spl": _r(spl(smooth(total[p], freqs, 6.0)), 2)})
    out["timing"] = {"field_ms": round(t_field * 1000, 1), "total_ms": round((time.perf_counter() - t0) * 1000, 1)}
    return out


SECTION_MAX_POINTS = 3200


def section(project: Project, axis: str = "length", at: float | None = None, band: str = "all") -> dict:
    """Direct level (direct sound + early reflections, as in the plan's Direct map) on a vertical slice.

    axis="length": the slice runs away from the stage at `at` metres across the room.
    axis="width": the slice runs across the room at `at` metres from the stage.
    Columns run along the slice, rows run from the floor up.
    """
    t0 = time.perf_counter()
    room = project.room
    if axis == "length":
        span = room.length
        at = float(np.clip(room.width / 2 if at is None else at, 0.0, room.width))
    else:
        span = room.width
        at = float(np.clip(room.length / 2 if at is None else at, 0.0, room.length))
    step = max(0.1, float(np.sqrt(span * room.height / SECTION_MAX_POINTS)))
    nu = max(2, int(span / step) + 1)
    nz = max(2, int(room.height / step) + 1)
    us = np.linspace(0.05, span - 0.05, nu)
    zs = np.linspace(0.05, room.height - 0.05, nz)
    U, Z = np.meshgrid(us, zs)  # (nz, nu): row = height
    if axis == "length":
        pts = np.column_stack([np.full(U.size, at), U.ravel(), Z.ravel()])
    else:
        pts = np.column_stack([U.ravel(), np.full(U.size, at), Z.ravel()])
    ctx = make_context(project)
    freqs = ctx.freqs
    w = dsp_weights(project.speakers, freqs)
    coh = np.zeros((len(pts), len(freqs)), np.complex64)
    live = [i for i, s in enumerate(project.speakers) if not s.mute]
    chunks = np.array_split(np.arange(len(pts)), min(8, max(1, len(pts) // 300)))

    def work(job):
        i, rows = job
        return rows, speaker_field(project.speakers[i], pts[rows], ctx).H * w[i].astype(np.complex64)[None, :]

    from concurrent.futures import ThreadPoolExecutor

    with ThreadPoolExecutor(max_workers=8) as ex:
        for rows, part in ex.map(work, [(i, r) for i in live for r in chunks]):
            coh[rows] += part
    early = coh.real ** 2 + coh.imag ** 2
    if band == "all":
        sel = (freqs >= F_LO) & (freqs <= F_HI)
    else:
        fc = float(band)
        sel = (freqs >= fc / np.sqrt(2)) & (freqs < fc * np.sqrt(2))
    level = spl(early[:, sel].mean(1)) if sel.any() else np.full(len(pts), np.nan)
    return {
        "axis": axis, "at": round(at, 2), "band": band,
        "nu": nu, "nz": nz, "u0": float(us[0]), "u1": float(us[-1]), "z0": float(zs[0]), "z1": float(zs[-1]),
        "span": span, "height": room.height,
        "values": _r(level, 1),
        "timing_ms": round((time.perf_counter() - t0) * 1000, 1),
    }


def point_response(project: Project, x: float, y: float, z: float | None = None, ppo: int = 48) -> dict:
    """High-resolution response, per-speaker traces and arrival list at one point."""
    room = project.room
    z = project.audience.ear_height if z is None else z
    x = float(np.clip(x, 0.0, room.width))
    y = float(np.clip(y, 0.0, room.length))
    z = float(np.clip(z, 0.0, room.height))
    st = project.settings
    freqs = freq_grid(st.f_min, min(st.f_max, 20000.0), ppo)
    ctx = make_context(project, freqs)
    pt = np.array([[x, y, z]])
    w = dsp_weights(project.speakers, freqs)
    coh = np.zeros(len(freqs), complex)
    psum = np.zeros(len(freqs))
    late = np.zeros(len(freqs))
    traces, events = [], []
    f_ref = int(np.argmin(np.abs(freqs - 1000.0)))
    for i, s in enumerate(project.speakers):
        sf = speaker_field(s, pt, ctx, early_ms=np.inf)  # all modelled reflections, coherent
        h = sf.H[0].astype(complex) * w[i]
        e = sf.E * np.abs(w[i]) ** 2
        coh += h
        psum += np.abs(h) ** 2
        late += e
        dist = float(sf.dist[0])
        arrival = dist / ctx.c * 1000.0 + total_delay_ms(s)
        traces.append({"id": s.id, "h": h, "late": e, "arrival_ms": round(arrival, 2), "distance_m": round(dist, 2),
                       "muted": s.mute})
        if not s.mute:
            events += _events(s, i, pt[0], ctx, f_ref, total_delay_ms(s), 10 ** (s.gain_db / 20))
    active = [t["arrival_ms"] for t in traces if not t["muted"]]
    t_first = min(active) if active else 0.0
    unwind = np.exp(2j * np.pi * freqs * t_first * 1e-3)  # remove the bulk propagation delay
    total = np.abs(coh) ** 2 + late
    speakers = []
    for t in traces:
        speakers.append({
            "id": t["id"], "arrival_ms": t["arrival_ms"], "distance_m": t["distance_m"],
            "spl": None if t["muted"] else _r(spl(smooth(np.abs(t["h"]) ** 2, freqs, 12.0)), 1),
            "phase": None if t["muted"] else _r(np.degrees(np.angle(t["h"] * unwind)), 1),
        })
    events.sort(key=lambda ev: ev["t"])
    top = max((ev["db"] for ev in events), default=0.0)
    events = [ev for ev in events if ev["db"] > top - 45.0][:400]
    return {
        "x": x, "y": y, "z": z,
        "freqs": _r(freqs, 2),
        "spl": _r(spl(total), 2),
        "spl_smooth": _r(spl(smooth(total, freqs, 6.0)), 2),
        "power_sum": _r(spl(smooth(psum + late, freqs, 6.0)), 2),
        "phase": _r(np.degrees(np.angle(coh * unwind)), 1),
        "speakers": speakers,
        "arrivals": events,
        "first_arrival_ms": round(t_first, 2),
    }


def _events(spk, index: int, point: np.ndarray, ctx: Context, f_ref: int, delay_ms: float, gain: float) -> list[dict]:
    """Discrete arrivals (direct + reflections) of one speaker at a point, level at ~1 kHz."""
    src = source_position(spk, ctx.dims)
    one = ctx.freqs[f_ref:f_ref + 1]
    amp = speaker_amplitude(spk) * gain * abs(box_response(spk, one)[0])
    out = []
    for im in image_sources(src, ctx.dims, ctx.order):
        vec = point - im.pos
        d = max(float(np.linalg.norm(vec)), MIN_DISTANCE)
        la = float(log_directivity(spk, (vec * im.flip / d)[None, :], one)[0, 0])
        for surf, cnt in im.hits.items():
            la += cnt * float(ctx.log_beta[surf][f_ref])
        la -= d * float(ctx.air_np[f_ref])
        p = amp * np.exp(la) / d
        out.append({"t": round(d / ctx.c * 1000.0 + delay_ms, 2), "db": round(float(spl(p * p)), 1),
                    "speaker": index, "order": im.order})
    return out


def room_report(project: Project) -> dict:
    """Statistical room acoustics per octave band."""
    env, r = project.environment, project.room
    air = air_absorption_db_per_m(OCTAVE_BANDS, env.temperature_c, env.humidity_pct, env.pressure_kpa)
    st = room_stats(r.width, r.length, r.height, surface_alphas(project, OCTAVE_BANDS), OCTAVE_BANDS, air)
    rt_mid = float(np.mean(st.rt60_eyring[2:4]))
    return {
        "bands": OCTAVE_BANDS.tolist(),
        "volume_m3": round(st.volume, 1),
        "surface_m2": round(st.surface, 1),
        "rt60_sabine": _r(st.rt60_sabine, 2),
        "rt60_eyring": _r(st.rt60_eyring, 2),
        "rt60_mid": round(rt_mid, 2),
        "mean_alpha": _r(st.mean_alpha, 3),
        "critical_distance_m": _r(st.critical_distance, 2),
        "speed_of_sound": round(speed_of_sound(env.temperature_c, env.humidity_pct, env.pressure_kpa), 2),
        "schroeder_hz": round(float(2000 * np.sqrt(rt_mid / st.volume)), 1),
        # what the air itself takes away over 30 m, per band (ISO 9613-1, this temperature and humidity)
        "air_loss_30m_db": _r(air * 30.0, 1),
        "air_loss_8k_30m_db": round(float(air_absorption_db_per_m(np.array([8000.0]), env.temperature_c, env.humidity_pct,
                                                                  env.pressure_kpa)[0] * 30.0), 1),
    }


__all__ = ["simulate", "point_response", "room_report", "db"]
