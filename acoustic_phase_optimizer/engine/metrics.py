"""Coverage, interference, clarity, timing and tonal metrics.

Everything is judged the way a system tech would judge it:

  coverage      evenness of the *direct/early* sound per octave band (total SPL
                is always even in a reverberant room, so it tells you nothing)
  interference  how speakers add to or cancel each other, 1/3-octave smoothed,
                so only cancellations wide enough to be heard count
  clarity       early-to-late energy ratio (C50-like), per seat
  timing        spread of arrival times of the audible sources at each seat
  tonal         deviation of the audience-average response from the target tilt

`analyse` is batch-capable (leading batch axis), so the optimizer scores whole
populations with exactly the same code that produces the numbers on screen.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .field import P_REF, Combined, FieldModel

BANDS = [63.0, 125.0, 250.0, 500.0, 1000.0, 2000.0, 4000.0, 8000.0]
EPS = 1e-30
AUDIBLE_DB = 10.0  # sources within this many dB of the loudest count as audible
SPREAD_OK_MS = 15.0  # arrival spread below this fuses with the direct sound
ECHO_MS = 30.0  # beyond the Haas zone: heard as smear or a distinct echo
CLARITY_TARGET_DB = 6.0
CANCEL_DB = -3.0
F_LO, F_HI = 80.0, 10000.0


def db(x):
    return 10.0 * np.log10(np.maximum(x, EPS))


def spl(power) -> np.ndarray:
    return db(np.asarray(power) / P_REF**2)


def band_masks(freqs: np.ndarray, centres: list[float] = BANDS) -> list[np.ndarray]:
    return [(freqs >= fc / np.sqrt(2)) & (freqs < fc * np.sqrt(2)) for fc in centres]


# ------------------------------------------------------------------ STI (IEC 60268-16 rev. 4, male speech)
STI_BANDS = [125.0, 250.0, 500.0, 1000.0, 2000.0, 4000.0, 8000.0]
STI_ALPHA = np.array([0.085, 0.127, 0.230, 0.233, 0.309, 0.224, 0.173])
STI_BETA = np.array([0.085, 0.078, 0.065, 0.011, 0.047, 0.095])
STI_MOD = 0.63 * 2.0 ** (np.arange(14) / 3.0)  # 0.63 ... 12.5 Hz


def sti_from_mtf(m: np.ndarray) -> np.ndarray:
    """STI from modulation transfer values shaped (..., 7 bands, 14 modulation frequencies)."""
    m = np.clip(m, 1e-6, 1 - 1e-6)
    snr = np.clip(10 * np.log10(m / (1 - m)), -15.0, 15.0)
    ti = (snr + 15.0) / 30.0
    mti = ti.mean(-1)  # (..., 7)
    sti = (mti * STI_ALPHA).sum(-1) - (STI_BETA * np.sqrt(np.maximum(mti[..., :-1] * mti[..., 1:], 0))).sum(-1)
    return np.clip(sti, 0.0, 1.0)


def sti_estimate(early: np.ndarray, late: np.ndarray, freqs: np.ndarray, rt60: np.ndarray) -> np.ndarray:
    """Statistical STI per point from early (direct + early reflections) and late (reverberant) energy.

    Early energy is treated as fully modulated; late energy as an exponential decay with the band's RT,
    whose modulation transfer is 1 / (1 + j 2 pi F T / 13.8). The two add as energy-weighted complex
    MTFs. Background noise and masking are not included, so this is an upper estimate.
    """
    P = early.shape[0]
    m = np.ones((P, len(STI_BANDS), len(STI_MOD)))
    for i, mask in enumerate(band_masks(freqs, STI_BANDS)):
        if not mask.any():
            continue
        e, r = early[:, mask].mean(1), late[:, mask].mean(1)
        t = float(np.nan_to_num(np.mean(rt60[mask]), nan=0.0))
        mr = 1.0 / (1.0 + 2j * np.pi * STI_MOD * max(t, 0.0) / 13.8)  # (14,)
        tot = np.maximum(e + r, 1e-30)
        m[:, i, :] = np.abs((e[:, None] + r[:, None] * mr[None, :]) / tot[:, None])
    return sti_from_mtf(m)


def sti_rating(v: float) -> str:
    return "excellent" if v >= 0.75 else "good" if v >= 0.6 else "fair" if v >= 0.45 else "poor" if v >= 0.3 else "bad"


def weighting_db(freqs: np.ndarray, kind: str) -> np.ndarray:
    """IEC 61672 A or C weighting in dB (Z = flat)."""
    f2 = np.asarray(freqs, float) ** 2
    if kind == "A":
        ra = 12194.0 ** 2 * f2 ** 2 / ((f2 + 20.6 ** 2) * np.sqrt((f2 + 107.7 ** 2) * (f2 + 737.9 ** 2)) * (f2 + 12194.0 ** 2))
        return 20 * np.log10(ra) + 2.00
    if kind == "C":
        rc = 12194.0 ** 2 * f2 / ((f2 + 20.6 ** 2) * (f2 + 12194.0 ** 2))
        return 20 * np.log10(rc) + 0.06
    return np.zeros_like(f2)


def weighting_gain(freqs: np.ndarray, kind: str) -> np.ndarray:
    return 10 ** (weighting_db(freqs, kind) / 10)


def band_levels(power: np.ndarray, freqs: np.ndarray, centres: list[float] = BANDS) -> np.ndarray:
    """Band SPL per point, shape (P, n_bands). Bands with no grid frequencies get NaN."""
    out = np.full((power.shape[0], len(centres)), np.nan)
    for i, m in enumerate(band_masks(freqs, centres)):
        if m.any():
            out[:, i] = spl(power[:, m].mean(1))
    return out


def smooth_bounds(freqs: np.ndarray, fraction: float) -> tuple[np.ndarray, np.ndarray]:
    lf = np.log2(freqs)
    half = 0.5 / fraction
    return np.searchsorted(lf, lf - half - 1e-9, side="left"), np.searchsorted(lf, lf + half + 1e-9, side="right")


def smooth(power: np.ndarray, freqs: np.ndarray, fraction: float = 3.0) -> np.ndarray:
    """Power-domain fractional-octave smoothing along the last (frequency) axis."""
    lo, hi = smooth_bounds(freqs, fraction)
    p = np.asarray(power, dtype=np.float64)
    csum = np.concatenate([np.zeros(p.shape[:-1] + (1,)), np.cumsum(p, -1)], -1)
    return (csum[..., hi] - csum[..., lo]) / np.maximum(hi - lo, 1)


def arrivals(field: FieldModel, weights: np.ndarray, delays_ms: np.ndarray, sl=slice(None)):
    """(first_ms, loudest_ms, spread_ms, dominant_index) per point among audible sources."""
    f = field.freqs
    mid = (f >= 500) & (f <= 4000)
    w2 = np.abs(weights[:, mid]) ** 2
    pw = np.einsum("sf,spf->sp", w2.astype(np.float32), field.H2[:, sl][:, :, mid])
    t = field.dist[:, sl] / field.c * 1000.0 + np.asarray(delays_ms)[:, None]
    first, last, any_audible = _spread(pw, t)
    dominant = pw.argmax(0)
    loudest = t[dominant, np.arange(t.shape[1])]
    spread = np.where(any_audible, last - first, 0.0)
    return np.where(any_audible, first, 0.0), loudest, spread, dominant


def _spread(pw: np.ndarray, t: np.ndarray, axis: int = 0):
    audible = (pw >= pw.max(axis, keepdims=True) * 10 ** (-AUDIBLE_DB / 10)) & (pw > 0)
    first = np.where(audible, t, np.inf).min(axis)
    last = np.where(audible, t, -np.inf).max(axis)
    return first, last, audible.any(axis)


def analyse(early: np.ndarray, psum: np.ndarray, late: np.ndarray, spread: np.ndarray,
            freqs: np.ndarray, tilt: float = 0.0, tone_lo: float = 100.0) -> dict[str, np.ndarray]:
    """Metrics for arrays shaped (..., P, F) (spread: (..., P)). Returns arrays shaped (...)."""
    fm = (freqs >= F_LO) & (freqs <= F_HI)
    out: dict[str, np.ndarray] = {}

    stds = []
    for i, m in enumerate(band_masks(freqs)):
        if m.any() and F_LO / 1.5 <= BANDS[i] <= F_HI:
            stds.append(db(early[..., m].mean(-1)).std(-1))
    out["coverage"] = np.mean(stds, axis=0) if stds else np.zeros(early.shape[:-2])
    bb = db(early[..., fm].mean(-1))
    out["range"] = np.percentile(bb, 95, axis=-1) - np.percentile(bb, 5, axis=-1)

    rip = db(smooth(early, freqs)[..., fm]) - db(smooth(psum, freqs)[..., fm])
    out["summation"] = rip.mean((-1, -2))
    out["cancel"] = np.maximum(0.0, -rip).mean((-1, -2))
    out["gain"] = np.maximum(0.0, rip).mean((-1, -2))
    out["cancel_pct"] = (rip < CANCEL_DB).mean((-1, -2)) * 100.0
    out["_rip_point"] = rip.mean(-1)

    mid = (freqs >= 500) & (freqs <= 4000)
    clar = db(early[..., mid].mean(-1)) - db(late[..., mid].mean(-1))
    out["_clarity_point"] = clar
    out["clarity"] = clar.mean(-1)
    out["clarity_deficit"] = np.maximum(0.0, CLARITY_TARGET_DB - clar).mean(-1)

    out["timing"] = (np.maximum(0.0, spread - SPREAD_OK_MS) / 10.0).mean(-1)
    out["echo_pct"] = (spread > ECHO_MS).mean(-1) * 100.0

    tm = (freqs >= tone_lo) & (freqs <= F_HI)
    resp = db(smooth(early.mean(-2), freqs)[..., tm])
    dev = resp - tilt * np.log2(freqs[tm] / 1000.0)
    dev = dev - np.median(dev, axis=-1, keepdims=True)
    out["tonal"] = np.sqrt((dev * dev).mean(-1))

    out["mean_spl"] = spl((early + late)[..., fm].mean((-1, -2)))
    return out


@dataclass
class Weights:
    """Relative priorities for the optimizer (1 = default)."""

    coverage: float = 1.0
    interference: float = 1.0
    clarity: float = 1.0
    timing: float = 1.0
    tonal: float = 1.0


def penalty(a: dict[str, np.ndarray], w: Weights | None = None) -> np.ndarray:
    """Penalty points; the on-screen score is 100 minus this with default weights."""
    w = w or Weights()
    return (
        w.coverage * 4.0 * a["coverage"]
        + w.interference * (5.0 * a["cancel"] - 1.5 * a["gain"])
        + w.clarity * 1.2 * a["clarity_deficit"]
        + w.timing * 5.0 * a["timing"]
        + w.tonal * 2.0 * a["tonal"]
    )


def breakdown(a: dict[str, np.ndarray]) -> dict[str, float]:
    """Points lost per category (default weights), for the score explanation."""
    return {
        "coverage": float(4.0 * a["coverage"]),
        "interference": float(5.0 * a["cancel"] - 1.5 * a["gain"]),
        "clarity": float(1.2 * a["clarity_deficit"]),
        "timing": float(5.0 * a["timing"]),
        "tonal": float(2.0 * a["tonal"]),
    }


METRIC_KEYS = ("mean_spl", "coverage", "range", "summation", "cancel_pct", "clarity", "echo_pct", "tonal")


def summarise(a: dict[str, np.ndarray]) -> dict:
    """JSON-ready metrics for a single (non-batch) analysis."""
    def num(v) -> float | None:
        v = float(v)
        return round(v, 3) if np.isfinite(v) else None

    m = {k: num(a[k]) for k in METRIC_KEYS}
    score = float(np.clip(100.0 - penalty(a), 0.0, 100.0))
    m["score"] = round(score, 1) if np.isfinite(score) else None
    m["lost"] = {k: round(v, 2) if np.isfinite(v) else None for k, v in breakdown(a).items()}
    return m


def tone_floor(speakers) -> float:
    """Lowest frequency the system is expected to reproduce (for the tonal metric)."""
    hps = [s.hp_hz for s in speakers if not s.mute]
    return float(np.clip(1.5 * min(hps), 40.0, 400.0)) if hps else 100.0


def evaluate_combined(c: Combined, spread: np.ndarray, freqs: np.ndarray, tilt: float, tone_lo: float) -> dict[str, np.ndarray]:
    return analyse(c.early, c.early_psum, c.late, spread, freqs, tilt, tone_lo)
