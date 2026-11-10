"""System alignment and optimization.

Three tools, all working on a precomputed FieldModel (no geometry is re-run):

  quick_align    deterministic, the way a system tech does it: level-match
                 fills, time-align them to the mains, phase-align the subs
                 through the crossover, set the sub level to the target curve
  deep_optimize  differential evolution around that alignment, scoring whole
                 populations with the same metrics shown on screen
  auto_eq        cut-biased parametric EQ per speaker toward the target tilt

Conventions that keep results sane on a real rig:
  * the mains are the time reference and are never re-timed against each other
  * mirror-image pairs (L/R) are linked and always receive identical settings
  * polarity is only ever a variable for subwoofers
  * locked speakers are never touched
"""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from scipy.optimize import least_squares, minimize

from .eq import bell_db, peq_response
from .field import FieldModel, combine, dsp_weights
from .metrics import EPS, Weights, _spread, analyse, arrivals, db, penalty, smooth, summarise, tone_floor
from .model import PEQBand, Project, Speaker

AUTO_PREFIX = "auto"
GAIN_MIN, GAIN_MAX = -18.0, 0.0  # level_db is the box's maximum, so the optimizer only trims down


# ---------------------------------------------------------------- helpers

def _band(freqs: np.ndarray, lo: float, hi: float) -> np.ndarray:
    return (freqs >= lo) & (freqs <= hi)


def speaker_band(spk: Speaker, freqs: np.ndarray) -> np.ndarray:
    """Frequencies comfortably inside the speaker's passband."""
    lo = max(spk.hp_hz * 1.3, freqs[0]) if spk.hp_hz > 0 else freqs[0]
    hi = min(spk.lp_hz * 0.8, freqs[-1]) if spk.lp_hz > 0 else min(freqs[-1], 12000.0)
    return _band(freqs, lo, hi)


def mains_of(speakers: list[Speaker]) -> list[int]:
    """Indices of the reference speakers: role 'main', else the loudest full-range box."""
    idx = [i for i, s in enumerate(speakers) if s.role == "main" and not s.mute]
    if idx:
        return idx
    cands = [i for i, s in enumerate(speakers) if s.role != "sub" and not s.mute]
    if not cands:
        cands = [i for i, s in enumerate(speakers) if not s.mute]
    return [max(cands, key=lambda i: speakers[i].level_db)] if cands else []


def _is_mirror(a: Speaker, b: Speaker, width: float) -> bool:
    same = ("role", "coverage_h", "coverage_v", "level_db", "hp_hz", "lp_hz", "directivity_hz",
            "lock_delay", "lock_gain")
    if any(getattr(a, k) != getattr(b, k) for k in same):
        return False
    yaw_sum = (a.yaw + b.yaw + 180.0) % 360.0 - 180.0
    return (abs(a.x + b.x - width) < 0.06 and abs(a.x - b.x) > 0.1 and abs(a.y - b.y) < 0.06
            and abs(a.z - b.z) < 0.06 and abs(yaw_sum) < 1.5 and abs(a.pitch - b.pitch) < 1.5)


def link_units(project: Project, speakers: list[Speaker] | None = None) -> list[list[int]]:
    """Group active speakers into units that share DSP settings (mirror pairs or singles)."""
    spks = speakers if speakers is not None else project.speakers
    used: set[int] = set()
    units: list[list[int]] = []
    for i, s in enumerate(spks):
        if i in used or s.mute:
            continue
        mate = None
        if project.settings.link_symmetric:
            for j in range(i + 1, len(spks)):
                if j not in used and not spks[j].mute and _is_mirror(s, spks[j], project.room.width):
                    mate = j
                    break
        units.append([i] if mate is None else [i, mate])
        used.update(units[-1])
    return units


def footprint(field: FieldModel, i: int, spk: Speaker) -> np.ndarray:
    """Audience points inside the speaker's own -6 dB coverage (everything, for a sub)."""
    if spk.role == "sub":
        return np.ones(field.n_grid, bool)
    m = _band(field.freqs, 500, 4000) & speaker_band(spk, field.freqs)
    if not m.any():
        m = speaker_band(spk, field.freqs)
    p = field.H2[i, field.grid_slice][:, m].mean(1)
    return p >= p.max() * 10 ** (-6 / 10)


def _weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
    if len(values) == 0:
        return 0.0
    order = np.argsort(values)
    cw = np.cumsum(weights[order])
    return float(values[order][min(np.searchsorted(cw, cw[-1] / 2), len(order) - 1)])


def _normalise_delays(spks: list[Speaker]) -> float:
    """Shift unlocked speakers so no delay is negative; returns the shift applied."""
    free = [s for s in spks if not s.lock_delay and not s.mute]
    lo = min((s.delay_ms for s in free), default=0.0)
    shift = -lo if lo < 0 else 0.0
    for s in free:
        s.delay_ms += shift
    for s in spks:
        s.delay_ms = round(min(max(0.0, s.delay_ms), 500.0), 2)
    return shift


def _names(spks: list[Speaker], unit: list[int]) -> str:
    return " + ".join(spks[i].name for i in unit)


def analyse_project(project: Project, field: FieldModel, speakers: list[Speaker] | None = None) -> dict:
    spks = speakers if speakers is not None else project.speakers
    w = dsp_weights(spks, field.freqs)
    g = field.grid_slice
    c = combine(field, w).rows(g)
    delays = np.array([s.delay_ms for s in spks])
    _, _, spread, _ = arrivals(field, w, delays, g)
    return analyse(c.early, c.early_psum, c.late, spread, field.freqs,
                   project.settings.target_tilt_db_per_oct, tone_floor(spks))


def evaluate(project: Project, field: FieldModel, speakers: list[Speaker] | None = None) -> dict:
    """Full-resolution metrics (JSON-ready) for a DSP state."""
    if not any(not s.mute for s in (speakers if speakers is not None else project.speakers)):
        return {}
    return summarise(analyse_project(project, field, speakers))


def diff_speakers(before: list[Speaker], after: list[Speaker]) -> list[dict]:
    """Human-readable list of DSP changes."""
    old = {s.id: s for s in before}
    out: list[dict] = []
    for s in after:
        o = old.get(s.id)
        if o is None:
            continue
        if abs(s.delay_ms - o.delay_ms) >= 0.01:
            out.append({"id": s.id, "name": s.name, "field": "delay_ms", "before": round(o.delay_ms, 2), "after": round(s.delay_ms, 2)})
        if abs(s.gain_db - o.gain_db) >= 0.05:
            out.append({"id": s.id, "name": s.name, "field": "gain_db", "before": round(o.gain_db, 1), "after": round(s.gain_db, 1)})
        if s.invert != o.invert:
            out.append({"id": s.id, "name": s.name, "field": "invert", "before": o.invert, "after": s.invert})
        eq_o = [(b.type, b.freq, b.gain_db, b.q, b.enabled) for b in o.peq]
        eq_n = [(b.type, b.freq, b.gain_db, b.q, b.enabled) for b in s.peq]
        if eq_o != eq_n:
            out.append({"id": s.id, "name": s.name, "field": "peq", "before": len(eq_o), "after": len(eq_n)})
    return out


# ------------------------------------------------------------- quick align

def quick_align(project: Project, field: FieldModel) -> tuple[list[Speaker], list[str]]:
    """Deterministic level + time + phase alignment. Returns (speakers, notes)."""
    spks = [s.model_copy(deep=True) for s in project.speakers]
    notes: list[str] = []
    mains = mains_of(spks)
    if not mains:
        return spks, ["Nothing to align: every speaker is muted."]
    freqs, g, c = field.freqs, field.grid_slice, field.c
    Hg, H2g, dist = field.H[:, g], field.H2[:, g], field.dist[:, g]
    mid = _band(freqs, 500, 4000)
    prec = project.settings.precedence_ms
    units = link_units(project, spks)
    others = [u for u in units if not set(u) & set(mains)]
    fill_units = [u for u in others if spks[u[0]].role != "sub"]
    sub_units = [u for u in others if spks[u[0]].role == "sub"]
    if not others:
        return spks, ["Only main speakers are active, so there is nothing to align them to."]

    def weights() -> np.ndarray:
        return dsp_weights(spks, freqs)

    def mid_power(indices: list[int], w: np.ndarray) -> np.ndarray:
        w2 = (np.abs(w) ** 2).astype(np.float32)
        return sum(H2g[i][:, mid] @ w2[i, mid] for i in indices) / max(int(mid.sum()), 1)

    def main_time(w: np.ndarray) -> np.ndarray:
        """Arrival time of the first audible main at each seat, ms."""
        w2 = (np.abs(w) ** 2).astype(np.float32)
        pw = np.stack([H2g[m][:, mid] @ w2[m, mid] for m in mains])
        t = np.stack([dist[m] / c * 1000.0 + spks[m].delay_ms for m in mains])
        first, _, ok = _spread(pw, t)
        return np.where(ok, first, t.min(0))

    def overlap(unit: list[int], base_idx: list[int], zone: np.ndarray, fmask: np.ndarray, w: np.ndarray):
        f = freqs[fmask]
        base = sum(w[m, fmask][None, :] * Hg[m][zone][:, fmask] for m in base_idx)
        own = sum((10 ** (spks[i].gain_db / 20) * peq_response(spks[i].peq, f))[None, :] * Hg[i][zone][:, fmask]
                  for i in unit)
        pb, pu = np.abs(base) ** 2, np.abs(own) ** 2
        ovl = np.minimum(pb, pu) / np.maximum(pb + pu, EPS)
        return f, base, own, pb + pu, ovl

    def phase_search(f, base, own, psum, ovl, centre: float, window: float, step: float, allow_flip: bool):
        tot = float(ovl.sum())
        if tot <= 0:
            return centre, False
        wt = ovl / tot
        best = (-np.inf, centre, False)
        for pol in ((False, True) if allow_flip else (False,)):
            sign = -1.0 if pol else 1.0
            for d in np.arange(centre - window, centre + window + 1e-9, step):
                rot = sign * np.exp(-2j * np.pi * f * d * 1e-3)
                summed = np.abs(base + own * rot[None, :]) ** 2
                score = float((wt * db(summed / np.maximum(psum, EPS))).sum())
                score -= 0.02 * abs(d - centre) + (0.05 if pol else 0.0)  # prefer time alignment, normal polarity
                if score > best[0]:
                    best = (score, float(d), pol)
        return best[1], best[2]

    # ---- reference level: what the mains deliver in their own coverage
    w = weights()
    pm = mid_power(mains, w)
    main_zone = np.zeros(field.n_grid, bool)
    for m in mains:
        main_zone |= footprint(field, m, spks[m])
    target = float(np.median(db(pm[main_zone])))
    aligned: list[int] = list(mains)

    # ---- fills and delays: level, then time
    for u in fill_units:
        zone = np.zeros(field.n_grid, bool)
        for i in u:
            zone |= footprint(field, i, spks[i])
        name = _names(spks, u)
        if not any(spks[i].lock_gain for i in u):
            eq2 = [np.abs(peq_response(spks[i].peq, freqs[mid])) ** 2 for i in u]
            ps = sum(H2g[i][:, mid] @ e.astype(np.float32) for i, e in zip(u, eq2)) / max(int(mid.sum()), 1)
            psz, pmz = ps[zone], pm[zone]
            g_match = -float(np.median(db(psz) - db(pmz)))  # fill as loud as the mains in its zone

            def zone_level(gain: float) -> float:
                return float(np.median(db(pmz + psz * 10 ** (gain / 10))))  # noqa: B023 - called within this iteration only

            gain = g_match
            if float(np.median(db(pmz))) < target - 0.25:  # mains are weak here: fill up to the target
                reach = [x for x in np.arange(GAIN_MIN, GAIN_MAX + 0.01, 0.25) if zone_level(x) >= target - 0.25]
                gain = max(g_match, reach[0] if reach else GAIN_MAX)
            while gain > g_match - 6.0 and zone_level(gain) > target + 3.0:
                gain -= 0.25
            short = gain - GAIN_MAX
            gain = round(float(np.clip(gain, GAIN_MIN, GAIN_MAX)), 1)
            for i in u:
                spks[i].gain_db = gain
            if short > 1.0:
                notes.append(f"{name}: at full output it is still {short:.0f} dB short of matching the mains "
                             "in its zone - a more powerful or closer box would help.")
            else:
                notes.append(f"{name}: level set to {gain:+.1f} dB to match the mains across its coverage.")
        if not any(spks[i].lock_delay for i in u):
            w = weights()
            tref = main_time(w)
            ests = []
            for i in u:
                fp = footprint(field, i, spks[i])
                ests.append(_weighted_median(tref[fp] - dist[i][fp] / c * 1000.0, H2g[i][fp][:, mid].mean(1)))
            centre = float(np.mean(ests)) + prec
            fmask = speaker_band(spks[u[0]], freqs) & _band(freqs, 80, 8000)
            d, _ = phase_search(*overlap(u, mains, zone, fmask, w), centre, 1.5, 0.05, False)
            for i in u:
                spks[i].delay_ms = d
            extra = f" (includes {prec:g} ms precedence offset)" if prec else ""
            notes.append(f"{name}: time-aligned to the mains{extra}.")
        aligned += u

    # ---- subs: phase through the crossover, then level to the target curve
    for u in sub_units:
        if any(spks[i].lock_delay for i in u):
            aligned += u
            continue
        w = weights()
        tref = main_time(w)
        zone = np.ones(field.n_grid, bool)
        fmask = _band(freqs, 30, 250)
        lf = H2g[u[0]][:, fmask].mean(1)
        centre = float(np.mean([_weighted_median(tref - dist[i] / c * 1000.0, lf) for i in u]))
        base_idx = [i for i in aligned if not spks[i].mute]
        f, base, own, psum, ovl = overlap(u, base_idx, zone, fmask, w)
        if ovl.sum() > 0:
            fx = float((ovl.sum(0) * f).sum() / ovl.sum())
            window = float(np.clip(500.0 / fx, 3.0, 10.0))  # half a period at the crossover
            d, pol = phase_search(f, base, own, psum, ovl, centre, window, 0.1, True)
        else:
            fx, d, pol = 0.0, centre, spks[u[0]].invert
        for i in u:
            spks[i].delay_ms, spks[i].invert = d, pol
        notes.append(f"{_names(spks, u)}: phase-aligned to the mains through the crossover"
                     + (f" (~{fx:.0f} Hz)" if fx else "") + (", polarity inverted." if pol else "."))
        aligned += u

    free_subs = [i for u in sub_units for i in u if not spks[i].lock_gain]
    non_subs = [i for i, s in enumerate(spks) if s.role != "sub" and not s.mute]
    if free_subs and non_subs:
        tilt = project.settings.target_tilt_db_per_oct
        shape = tilt * np.log2(freqs / 1000.0)
        ref_band = _band(freqs, 250, 4000)
        moved = 0.0
        for _ in range(3):
            w = weights()
            w2 = (np.abs(w) ** 2).astype(np.float32)
            resp = db(smooth(combine(field, w).early[g].mean(0), freqs))
            sub_pow = sum((H2g[i] * w2[i][None, :]).mean(0) for u in sub_units for i in u)
            oth_pow = sum((H2g[i] * w2[i][None, :]).mean(0) for i in non_subs)
            sb = (sub_pow > oth_pow) & (freqs >= 35)
            if sb.sum() < 2:
                break
            offset = float(np.median(resp[ref_band] - shape[ref_band]))
            err = float(np.mean(resp[sb] - shape[sb] - offset))
            if abs(err) < 0.25:
                break
            for i in free_subs:
                new = float(np.clip(spks[i].gain_db - err, -24.0, max(GAIN_MAX, spks[i].gain_db)))
                moved += new - spks[i].gain_db if i == free_subs[0] else 0.0
                spks[i].gain_db = new
        for i in free_subs:
            spks[i].gain_db = round(spks[i].gain_db, 1)
        if abs(moved) >= 0.05:
            notes.append(f"Subs: level trimmed {moved:+.1f} dB to sit on the target curve ({tilt:+.1f} dB/oct).")

    shift = _normalise_delays(spks)
    if shift > 0.005:
        notes.append(f"All outputs: +{shift:.2f} ms so that no delay is negative (relative timing unchanged).")
    return spks, notes


# ---------------------------------------------------------- deep optimize

@dataclass
class Progress:
    generation: int
    max_generations: int
    score: float
    start_score: float
    elapsed: float


class BatchObjective:
    """Scores many DSP states at once on a subsample of the audience."""

    def __init__(self, field: FieldModel, speakers: list[Speaker], weights: Weights, tilt: float,
                 max_points: int = 300, seed: int = 0):
        rng = np.random.default_rng(seed)
        P = field.n_grid
        idx = np.arange(P) if P <= max_points else np.sort(rng.choice(P, max_points, replace=False))
        self.freqs = field.freqs
        self.Hf = np.ascontiguousarray(field.H[:, idx].transpose(2, 0, 1))  # (F, S, P)
        self.H2f = np.ascontiguousarray(field.H2[:, idx].transpose(2, 0, 1))
        self.LEf = np.ascontiguousarray(field.LE[:, idx].transpose(2, 0, 1))
        self.t_geo = field.dist[:, idx] / field.c * 1000.0  # (S, P)
        self.mid = _band(self.freqs, 500, 4000)
        self.eq = np.stack([peq_response(s.peq, self.freqs) for s in speakers])
        self.active = np.array([0.0 if s.mute else 1.0 for s in speakers])
        self.weights, self.tilt, self.tone_lo = weights, tilt, tone_floor(speakers)
        self.ref_level: float | None = None
        self.n_points = len(idx)

    def evaluate(self, delays: np.ndarray, gains: np.ndarray, invert: np.ndarray) -> np.ndarray:
        """delays / gains / invert shaped (B, S) -> penalty (B,). Lower is better."""
        B = delays.shape[0]
        chunk = max(1, int(4e6 // max(self.n_points * len(self.freqs), 1)))
        out = np.empty(B)
        for a in range(0, B, chunk):
            b = min(B, a + chunk)
            amp = 10 ** (gains[a:b] / 20) * np.where(invert[a:b], -1.0, 1.0) * self.active
            rot = np.exp(-2j * np.pi * delays[a:b, :, None] * 1e-3 * self.freqs[None, None, :])
            Wf = np.ascontiguousarray((amp[:, :, None] * rot * self.eq[None]).transpose(2, 0, 1)).astype(np.complex64)
            coh = Wf @ self.Hf  # (F, b, P)
            W2 = (Wf.real ** 2 + Wf.imag ** 2).astype(np.float32)
            early = coh.real ** 2 + coh.imag ** 2
            psum = W2 @ self.H2f
            late = W2 @ self.LEf
            pw = np.einsum("fbs,fsp->bsp", W2[self.mid], self.H2f[self.mid])
            t = self.t_geo[None] + delays[a:b, :, None]
            first, last, ok = _spread(pw, t, axis=1)
            spread = np.where(ok, last - first, 0.0)
            res = analyse(early.transpose(1, 2, 0), psum.transpose(1, 2, 0), late.transpose(1, 2, 0),
                          spread, self.freqs, self.tilt, self.tone_lo)
            if self.ref_level is None:
                self.ref_level = float(res["mean_spl"][0])
            out[a:b] = penalty(res, self.weights) + 2.0 * np.maximum(0.0, self.ref_level - res["mean_spl"])
        return out


def deep_optimize(
    project: Project,
    field: FieldModel,
    weights: Weights | None = None,
    start: str = "align",
    max_generations: int = 150,
    time_budget_s: float = 20.0,
    progress: Callable[[Progress], None] | None = None,
    cancelled: Callable[[], bool] = lambda: False,
    rng_seed: int = 1,
) -> tuple[list[Speaker], dict]:
    """Global search over delay / gain (and sub polarity) of every unlocked, non-main unit.

    Returns (speakers, info). The result is never worse than the starting point:
    the original, the quick alignment and the search result are compared at full
    resolution and the best one is returned.
    """
    weights = weights or Weights()
    original = [s.model_copy(deep=True) for s in project.speakers]
    seed = quick_align(project, field)[0] if start == "align" else [s.model_copy(deep=True) for s in original]
    tilt = project.settings.target_tilt_db_per_oct
    info: dict = {"history": [], "generations": 0, "picked": "start"}
    mains = set(mains_of(seed))
    units = [u for u in link_units(project, seed) if not set(u) & mains]
    any_lock = any(s.lock_delay for s in seed)

    d0 = np.array([s.delay_ms for s in seed])
    g0 = np.array([s.gain_db for s in seed])
    p0 = np.array([s.invert for s in seed])
    variables: list[tuple[str, list[int]]] = []
    lo: list[float] = []
    hi: list[float] = []
    for u in units:
        s = seed[u[0]]
        is_sub = s.role == "sub"
        if not any(seed[i].lock_delay for i in u):
            win = float(np.clip(500.0 / s.lp_hz, 3.0, 10.0)) if is_sub and s.lp_hz > 0 else (6.0 if is_sub else 3.0)
            variables.append(("d", u))
            lo.append(max(-win, -float(d0[u].min())) if any_lock else -win)
            hi.append(win)
            if is_sub:
                variables.append(("p", u))
                lo.append(0.0)
                hi.append(1.0)
        if not any(seed[i].lock_gain for i in u):
            variables.append(("g", u))
            lo.append(max(-6.0, -24.0 - float(g0[u].min())))
            hi.append(min(6.0, max(GAIN_MAX, float(g0[u].max())) - float(g0[u].max())))
    D = len(variables)
    lo_a, hi_a = np.array(lo), np.maximum(np.array(hi), np.array(lo))

    def decode(X: np.ndarray):
        B = X.shape[0]
        dd, gg, pp = np.repeat(d0[None], B, 0), np.repeat(g0[None], B, 0), np.repeat(p0[None], B, 0)
        for j, (kind, u) in enumerate(variables):
            if kind == "d":
                dd[:, u] += X[:, j:j + 1]
            elif kind == "g":
                gg[:, u] += X[:, j:j + 1]
            else:
                pp[:, u] ^= X[:, j:j + 1] > 0.5
        return dd, gg, pp

    def build(x: np.ndarray) -> list[Speaker]:
        dd, gg, pp = decode(x[None])
        out = [s.model_copy(deep=True) for s in seed]
        for i, s in enumerate(out):
            if not s.mute:
                s.delay_ms, s.gain_db, s.invert = float(dd[0, i]), round(float(gg[0, i]), 1), bool(pp[0, i])
        _normalise_delays(out)
        return out

    def full_penalty(spks: list[Speaker]) -> float:
        return float(penalty(analyse_project(project, field, spks), weights))

    candidates = {"original": original, "aligned": seed} if start == "align" else {"original": original}
    if D > 0 and not cancelled():
        obj = BatchObjective(field, seed, weights, tilt)

        def f(X: np.ndarray) -> np.ndarray:
            return obj.evaluate(*decode(X))

        x0 = np.array([0.25 if kind == "p" else 0.0 for kind, _ in variables])
        x0 = np.clip(x0, lo_a, hi_a)
        start_pen = float(f(x0[None])[0])
        info["history"].append({"generation": 0, "score": round(100.0 - start_pen, 2)})
        rng = np.random.default_rng(rng_seed)
        NP = int(np.clip(8 * D, 24, 80))
        X = lo_a + rng.random((NP, D)) * (hi_a - lo_a)
        X[0] = x0
        X[1:NP // 3] = np.clip(x0 + rng.normal(0, 0.15, (NP // 3 - 1, D)) * (hi_a - lo_a), lo_a, hi_a)
        fit = f(X)
        mu_f, mu_cr = 0.6, 0.85
        t0 = time.perf_counter()
        best, stall, gen = int(np.argmin(fit)), 0, 0
        for gen in range(1, max_generations + 1):
            if cancelled():
                break
            Fs = np.clip(rng.standard_cauchy(NP) * 0.1 + mu_f, 0.05, 1.0)
            CRs = np.clip(rng.normal(mu_cr, 0.1, NP), 0.0, 1.0)
            order = np.argsort(fit)
            pbest = order[rng.integers(0, max(2, NP // 8), NP)]
            r1, r2 = rng.integers(0, NP, NP), rng.integers(0, NP, NP)
            V = X + Fs[:, None] * (X[pbest] - X) + Fs[:, None] * (X[r1] - X[r2])
            V = np.where(V < lo_a, (lo_a + X) / 2, V)
            V = np.where(V > hi_a, (hi_a + X) / 2, V)
            cross = rng.random((NP, D)) < CRs[:, None]
            cross[np.arange(NP), rng.integers(0, D, NP)] = True
            U = np.where(cross, V, X)
            fu = f(U)
            better = fu < fit
            if better.any():
                mu_f = 0.9 * mu_f + 0.1 * float(np.sum(Fs[better] ** 2) / max(np.sum(Fs[better]), 1e-9))
                mu_cr = 0.9 * mu_cr + 0.1 * float(np.mean(CRs[better]))
            X[better], fit[better] = U[better], fu[better]
            nb = int(np.argmin(fit))
            stall = stall + 1 if fit[nb] >= fit[best] - 1e-3 else 0
            best = nb
            elapsed = time.perf_counter() - t0
            info["history"].append({"generation": gen, "score": round(100.0 - float(fit[best]), 2)})
            if progress:
                progress(Progress(gen, max_generations, 100.0 - float(fit[best]), 100.0 - start_pen, elapsed))
            if elapsed > time_budget_s or stall > 30:
                break
        xb = X[best].copy()
        cont = [j for j, (kind, _) in enumerate(variables) if kind != "p"]
        if cont and not cancelled():
            def f1(z: np.ndarray) -> float:
                x = xb.copy()
                x[cont] = z
                return float(f(x[None])[0])

            res = minimize(f1, xb[cont], method="Powell", bounds=list(zip(lo_a[cont], hi_a[cont])),
                           options={"maxfev": 60 * len(cont), "xtol": 1e-2, "ftol": 1e-4})
            if res.fun < fit[best]:
                xb[cont] = np.clip(res.x, lo_a[cont], hi_a[cont])
                info["history"].append({"generation": gen + 1, "score": round(100.0 - float(res.fun), 2)})
        info["generations"] = gen
        candidates["optimized"] = build(xb)

    scores = {k: full_penalty(v) for k, v in candidates.items()}
    picked = min(scores, key=lambda k: scores[k] + (0.0 if k == "optimized" else 0.05 if k == "aligned" else 0.1))
    info["picked"] = picked
    info["variables"] = D
    return candidates[picked], info


# ------------------------------------------------------------------- auto EQ

def _fit_bells(f: np.ndarray, err: np.ndarray, max_bands: int, max_boost: float, max_cut: float,
               tolerance: float) -> list[tuple[float, float, float]]:
    """Greedy bell placement on the error curve (positive = too loud), then joint refinement."""
    bands: list[tuple[float, float, float]] = []
    resid = err.copy()
    for _ in range(max_bands):
        # cuts count fully; dips only matter beyond 3 dB and at half weight (don't chase holes)
        pri = np.where(resid > 0, resid, np.minimum(0, resid + 3.0) * 0.5)
        k = int(np.argmax(np.abs(pri)))
        if abs(pri[k]) < tolerance:
            break
        gain = float(np.clip(-resid[k], -max_cut, max_boost))
        half = np.abs(resid) >= abs(resid[k]) / 2
        lft = rgt = k
        while lft > 0 and half[lft - 1] and np.sign(resid[lft - 1]) == np.sign(resid[k]):
            lft -= 1
        while rgt < len(f) - 1 and half[rgt + 1] and np.sign(resid[rgt + 1]) == np.sign(resid[k]):
            rgt += 1
        bw_oct = max(float(np.log2(f[rgt] / f[lft])), 0.15)
        bands.append((float(f[k]), gain, float(np.clip(1.2 / bw_oct, 0.5, 8.0))))
        resid = err + sum(bell_db(f, *b) for b in bands)
    if not bands:
        return bands
    x0 = np.array([v for b in bands for v in (np.log2(b[0]), b[1], b[2])])
    lb = np.array([v for _ in bands for v in (np.log2(f[0]), -max_cut, 0.4)])
    ub = np.array([v for _ in bands for v in (np.log2(f[-1]), max_boost, 10.0)])
    x0 = np.clip(x0, lb + 1e-6, ub - 1e-6)

    def residual(x: np.ndarray) -> np.ndarray:
        return err + sum(bell_db(f, 2 ** x[j], x[j + 1], x[j + 2]) for j in range(0, len(x), 3))

    try:
        x = least_squares(residual, x0, bounds=(lb, ub), max_nfev=150).x
    except Exception:  # noqa: BLE001 - fall back to the greedy solution
        x = x0
    return [(float(2 ** x[j]), float(x[j + 1]), float(x[j + 2])) for j in range(0, len(x), 3)]


def auto_eq(project: Project, field: FieldModel, max_bands: int = 5, max_boost: float = 3.0,
            max_cut: float = 12.0, tolerance: float = 1.0) -> tuple[list[Speaker], list[dict]]:
    """Fit PEQ bells per speaker unit so its zone-averaged direct response follows the target tilt.

    Existing auto bands are replaced; bands the user added by hand are kept.
    """
    spks = [s.model_copy(deep=True) for s in project.speakers]
    for s in spks:
        s.peq = [b for b in s.peq if not b.id.startswith(AUTO_PREFIX)]
    freqs, g = field.freqs, field.grid_slice
    tilt = project.settings.target_tilt_db_per_oct
    mains = set(mains_of(spks))
    units = link_units(project, spks)
    rank = {"main": 0, "sub": 1}
    units.sort(key=lambda u: 0 if set(u) & mains else rank.get(spks[u[0]].role, 2))
    jobs = []
    for u in units:
        s0 = spks[u[0]]
        zone = np.zeros(field.n_grid, bool)
        for i in u:
            zone |= footprint(field, i, spks[i])
        fm = speaker_band(s0, freqs) & _band(freqs, 40 if s0.role == "sub" else 80, 12000)
        if fm.sum() < 6 or not zone.any():
            continue
        f = freqs[fm]
        shape = tilt * np.log2(f / 1000.0)

        def error(zone=zone, fm=fm, shape=shape) -> np.ndarray:
            early = combine(field, dsp_weights(spks, freqs)).early[g]
            resp = db(smooth(early[zone].mean(0), freqs))[fm]
            return resp - shape - np.median(resp - shape)

        err = error()
        bands = [b for b in _fit_bells(f, err, max_bands, max_boost, max_cut, tolerance) if abs(b[1]) >= 0.5]
        for i in u:
            for n, (fc, gain, q) in enumerate(bands):
                spks[i].peq.append(PEQBand(id=f"{AUTO_PREFIX}-{n}-{spks[i].id}", type="bell",
                                           freq=round(fc, 1), gain_db=round(gain, 1), q=round(q, 2)))
        jobs.append((u, len(bands), float(np.sqrt(np.mean(err ** 2))), error))
    report = []
    for u, n, before, error in jobs:
        report.append({"ids": [spks[i].id for i in u], "name": _names(spks, u), "bands": n,
                       "rms_before_db": round(before, 2), "rms_after_db": round(float(np.sqrt(np.mean(error() ** 2))), 2)})
    return spks, report


__all__ = ["quick_align", "deep_optimize", "auto_eq", "evaluate", "analyse_project", "diff_speakers",
           "link_units", "mains_of", "Weights", "Progress"]
