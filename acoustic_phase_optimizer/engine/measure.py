"""Room measurement: impulse responses from sweeps, and the ISO 3382 room parameters from those.

    sweep(fs, ...)              the exponential sine sweep the browser plays (Farina)
    deconvolve(rec, ref, ...)   the impulse response from a recording and the signal that was played
    analyse_ir(ir, fs, ...)     onset, arrival, polarity, per-octave EDT / T20 / T30 / C50 / C80 / D50 / Ts,
                                the frequency response and curves to draw (ETC, energy decay)
    read_wav / write_wav        WAV files in and out (PCM 16/24/32 bit, float)

ISO 3382-1 / -2 choices, in short. The impulse response starts where it first comes within 20 dB of its peak.
Each octave is filtered on the time-reversed response (so filter ringing does not lengthen short decays), the
noise floor is found with Lundeby's iteration, the backward (Schroeder) integral stops where the decay meets
the noise and is compensated for the energy that would have followed. A decay time is only given when the
measurement has the range for it (T30 needs the decay to reach 45 dB above the noise, T20 35, EDT 20);
otherwise it is None, never a made-up number.
"""

from __future__ import annotations

import io

import numpy as np
from scipy import signal

BANDS = [63.0, 125.0, 250.0, 500.0, 1000.0, 2000.0, 4000.0, 8000.0]
MID = (3, 4)  # 500 Hz and 1 kHz: the single-number values average these
EPS = 1e-30


class MeasureError(ValueError):
    """The input cannot give an impulse response (silence, no sweep heard, not an impulse response)."""


# --------------------------------------------------------------------------------------------- signals


def sweep(fs: float, seconds: float = 3.0, f1: float = 20.0, f2: float = 20000.0, fade_s: float = 0.02) -> np.ndarray:
    """Exponential sine sweep, peak 1.0, with short raised-cosine fades (the low one a little longer)."""
    f2 = min(f2, 0.48 * fs)
    n = int(round(seconds * fs))
    t = np.arange(n) / fs
    k = np.log(f2 / f1)
    x = np.sin(2 * np.pi * f1 * seconds / k * (np.exp(t * k / seconds) - 1.0))
    a, b = min(n // 4, int(fade_s * 2 * fs)), min(n // 4, int(fade_s * fs))
    if a:
        x[:a] *= 0.5 - 0.5 * np.cos(np.pi * np.arange(a) / a)
    if b:
        x[n - b:] *= 0.5 + 0.5 * np.cos(np.pi * np.arange(b) / b)
    return x.astype(np.float64)


def deconvolve(rec: np.ndarray, ref: np.ndarray, fs: float, f1: float = 20.0, f2: float = 20000.0) -> np.ndarray:
    """Impulse response of `rec` against `ref` (what was played, sample-aligned with the recording).

    Regularised spectral division, band-limited to the sweep's range so noise outside it is not blown up.
    For an exponential sweep the harmonic distortion lands at negative time, which wraps to the end of the
    circular result; the returned part (time 0 to the length of the recorded tail) holds the linear response.
    """
    rec = np.asarray(rec, dtype=np.float64)
    ref = np.asarray(ref, dtype=np.float64)
    n = 1 << int(np.ceil(np.log2(len(rec) + len(ref))))
    R = np.fft.rfft(rec, n)
    S = np.fft.rfft(ref, n)
    f = np.fft.rfftfreq(n, 1 / fs)
    p = np.abs(S) ** 2
    lo, hi = max(f1, 1.0), min(f2, 0.49 * fs)
    inband = (f >= lo) & (f <= hi)
    if not inband.any() or p[inband].max() <= 0:
        raise MeasureError("The played signal is silent.")
    floor = p[inband].max()
    # in band a tiny epsilon; a half-octave raised-cosine skirt out to a large one beyond the sweep
    oct_out = np.maximum(np.log2(lo / np.maximum(f, 1e-9)), np.log2(np.maximum(f, 1e-9) / hi))
    w = np.clip(oct_out / 0.5, 0, 1)
    eps = floor * (1e-8 + (0.5 - 0.5 * np.cos(np.pi * w)) * 1.0)
    H = R * np.conj(S) / (p + eps)
    H *= 1.0 - w  # fade the response itself out beyond the sweep
    ir = np.fft.irfft(H, n)
    # the reference may carry silence around the sweep (it is sample-aligned with the recording): the
    # usable response is as long as the recording runs on after the sweep has finished
    nz = np.nonzero(np.abs(ref) > 1e-9 * np.abs(ref).max())[0]
    keep = max(len(rec) - (int(nz[-1]) + 1 if len(nz) else len(ref)), int(0.25 * fs))
    return ir[: min(keep, len(rec))]


# --------------------------------------------------------------------------------------------- WAV


def read_wav(data: bytes) -> tuple[np.ndarray, int]:
    """(samples, fs) from WAV bytes: first channel, float in [-1, 1]."""
    from scipy.io import wavfile

    try:
        fs, x = wavfile.read(io.BytesIO(data))
    except Exception as exc:  # noqa: BLE001 - scipy raises several kinds for a bad file
        raise MeasureError(f"Not a WAV file this app can read ({exc}).") from None
    if x.ndim > 1:
        x = x[:, 0]
    if x.dtype.kind == "f":
        y = x.astype(np.float64)
    elif x.dtype == np.uint8:
        y = (x.astype(np.float64) - 128.0) / 128.0
    else:
        y = x.astype(np.float64) / float(np.iinfo(x.dtype).max)
    if not 8000 <= fs <= 384000:
        raise MeasureError(f"Sample rate {fs} Hz is outside 8-384 kHz.")
    return y, int(fs)


def write_wav(x: np.ndarray, fs: int) -> bytes:
    """32-bit float WAV, one channel."""
    from scipy.io import wavfile

    buf = io.BytesIO()
    wavfile.write(buf, int(fs), np.asarray(x, dtype=np.float32))
    return buf.getvalue()


# --------------------------------------------------------------------------------------------- analysis


def _octave_sos(fc: float, fs: float):
    lo, hi = fc / np.sqrt(2), fc * np.sqrt(2)
    return signal.butter(3, [lo, hi], btype="bandpass", fs=fs, output="sos")


def onset(ir: np.ndarray, fs: float) -> int:
    """ISO 3382-1 A.3.4: where the response first rises to within 20 dB of its peak (searched up to 50 ms
    before the peak, so noise ahead of a late recording does not count)."""
    e = ir * ir
    k = int(np.argmax(e))
    thr = e[k] * 10 ** (-20 / 10)
    j0 = max(0, k - int(0.05 * fs))
    above = np.nonzero(e[j0:k + 1] >= thr)[0]
    return j0 + int(above[0]) if len(above) else k


def _blocks(e: np.ndarray, m: int) -> tuple[np.ndarray, np.ndarray]:
    """Mean of e over consecutive blocks of m samples, and each block's centre (in samples)."""
    nb = len(e) // m
    if nb < 2:
        return e.copy(), np.arange(len(e), dtype=float)
    b = e[: nb * m].reshape(nb, m).mean(axis=1)
    return b, (np.arange(nb) + 0.5) * m


def _fit(x: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    A = np.vstack([x, np.ones_like(x)]).T
    (s, c), *_ = np.linalg.lstsq(A, y, rcond=None)
    return float(s), float(c)


def lundeby(e: np.ndarray, fs: float) -> dict:
    """Lundeby et al. (1995): the point where the decay meets the noise floor, the late decay slope and the
    noise level, from the squared (band) impulse response e starting at the onset.
    Returns {cross (sample), slope (dB per sample), noise_db, peak_db}; cross = len(e) when no noise is found."""
    n = len(e)
    noise = float(np.mean(e[int(0.9 * n):])) + EPS
    m = max(1, int(0.01 * fs))  # 10 ms blocks to start
    b, tb = _blocks(e, m)
    bdb = 10 * np.log10(b + EPS)
    peak_db = float(bdb.max())
    ndb = 10 * np.log10(noise)
    res = {"cross": n, "slope": None, "noise_db": ndb, "peak_db": peak_db}
    if peak_db - ndb < 10:
        return res
    # first regression: from the peak to 10 dB above the noise
    i0 = int(np.argmax(bdb))
    stop = np.nonzero(bdb[i0:] <= ndb + 10)[0]
    i1 = i0 + (int(stop[0]) if len(stop) else len(bdb) - i0)
    if i1 - i0 < 2:
        return res
    s, c = _fit(tb[i0:i1], bdb[i0:i1])
    if s >= 0:
        return res
    cross = int(np.clip((ndb - c) / s, 0, n))
    for _ in range(5):
        # new block length: 5 blocks per 10 dB of decay
        m = int(np.clip(-10.0 / s / 5, max(1, 0.002 * fs), 0.05 * fs))
        b, tb = _blocks(e, m)
        bdb = 10 * np.log10(b + EPS)
        # noise from 5 dB below the crossing (where the decay has died away) to the end, at least 10 %
        start = int(min(cross + (-5.0 / s), 0.9 * n))
        noise = float(np.mean(e[start:])) + EPS if start < n else noise
        ndb = 10 * np.log10(noise)
        # regression over the late decay: 5 dB above the noise ... 25 dB above it (or the peak)
        top = min(peak_db - 5, ndb + 25) if peak_db - ndb > 30 else peak_db - 5
        sel = (bdb <= top) & (bdb >= ndb + 5) & (tb < cross + 1)
        idx = np.nonzero(sel)[0]
        if len(idx) < 3:
            break
        # use the first contiguous run below `top` so a late bump does not bend the line
        s2, c2 = _fit(tb[idx], bdb[idx])
        if s2 >= 0:
            break
        new = int(np.clip((ndb - c2) / s2, 0, n))
        s, c = s2, c2
        if abs(new - cross) < m:
            cross = new
            break
        cross = new
    res.update(cross=int(min(cross, n)), slope=s, noise_db=ndb)
    return res


def schroeder(e: np.ndarray, fs: float, lb: dict | None = None) -> np.ndarray:
    """Backward-integrated energy decay in dB (0 at the start), truncated at the noise crossing and
    compensated for the energy after it (ISO 3382-1 5.3.3 / Lundeby). NaN beyond the crossing."""
    lb = lb or lundeby(e, fs)
    tc = max(1, int(lb["cross"]))
    tail = 0.0
    if lb["slope"] is not None and tc < len(e):
        a = -lb["slope"] * np.log(10) / 10  # per-sample energy decay rate
        level = 10 ** ((lb["noise_db"]) / 10)  # the line meets the noise at tc: energy density there
        tail = level / (1 - np.exp(-a)) if a > 0 else 0.0
    c = np.cumsum(e[:tc][::-1])[::-1] + tail
    out = np.full(len(e), np.nan)
    out[:tc] = 10 * np.log10(c / c[0] + EPS)
    return out


def _decay_time(edc: np.ndarray, fs: float, hi: float, lo: float) -> tuple[float | None, float]:
    """Time for 60 dB from a line fitted to the EDC between hi and lo dB (e.g. -5 and -35). (T, r)."""
    ok = np.isfinite(edc)
    sel = ok & (edc <= hi) & (edc >= lo)
    idx = np.nonzero(sel)[0]
    if len(idx) < 4 or not np.any(ok & (edc <= lo)):
        return None, 0.0
    i0, i1 = idx[0], idx[-1]
    t = np.arange(i0, i1 + 1) / fs
    y = edc[i0:i1 + 1]
    good = np.isfinite(y)
    s, c = _fit(t[good], y[good])
    if s >= 0:
        return None, 0.0
    r = float(np.corrcoef(t[good], y[good])[0, 1]) if good.sum() > 2 else 0.0
    return -60.0 / s, r


def _energy_params(e: np.ndarray, fs: float, cross: int, tail: float) -> dict:
    t = np.arange(len(e)) / fs
    tc = max(1, min(cross, len(e)))
    total = float(e[:tc].sum()) + tail
    if total <= 0:
        return {"c50": None, "c80": None, "d50": None, "ts": None}
    k50, k80 = int(0.05 * fs), int(0.08 * fs)
    e50, e80 = float(e[: min(k50, tc)].sum()), float(e[: min(k80, tc)].sum())
    return {
        "c50": 10 * np.log10(e50 / max(total - e50, EPS)),
        "c80": 10 * np.log10(e80 / max(total - e80, EPS)),
        "d50": e50 / total,
        "ts": float((t[:tc] * e[:tc]).sum() / total) * 1000.0,  # ms
    }


def _r(v, nd=2):
    return None if v is None or not np.isfinite(v) else round(float(v), nd)


def _decimate_curve(y: np.ndarray, n: int, how: str = "max") -> list[float | None]:
    """At most n points: max (ETC, so peaks show) or first value per bucket. NaN -> None."""
    if len(y) <= n:
        return [_r(v, 1) for v in y]
    edges = np.linspace(0, len(y), n + 1).astype(int)
    out = []
    for a, b in zip(edges[:-1], edges[1:], strict=True):
        seg = y[a:max(b, a + 1)]
        seg = seg[np.isfinite(seg)]
        out.append(_r(seg.max() if how == "max" else seg[0], 1) if len(seg) else None)
    return out


def response(ir: np.ndarray, fs: float, arrival_s: float, f_lo: float, f_hi: float, per_octave: int = 48) -> dict:
    """Frequency response of the whole response (fade-out at its end), on a 1/48-octave grid, with the bulk
    delay to the direct sound removed from the phase so it can sit over the predicted phase."""
    n = len(ir)
    w = np.ones(n)
    k = min(n // 4, int(0.01 * fs))
    if k:
        w[n - k:] = 0.5 + 0.5 * np.cos(np.pi * np.arange(k) / k)
    nf = 1 << int(np.ceil(np.log2(max(n, int(fs)))))  # at least 1 Hz resolution
    H = np.fft.rfft(ir * w, nf)
    f = np.fft.rfftfreq(nf, 1 / fs)
    H = H * np.exp(2j * np.pi * f * arrival_s)
    lo, hi = max(f_lo, 10.0), min(f_hi, 0.45 * fs, 22000.0)
    edges = lo * 2 ** (np.arange(0, np.log2(hi / lo) * per_octave + 1) / per_octave)
    freqs, spl, phase = [], [], []
    for a, b in zip(edges[:-1], edges[1:], strict=True):
        sel = (f >= a) & (f < b)
        if not sel.any():
            j = int(np.argmin(np.abs(f - np.sqrt(a * b))))
            sel = np.zeros_like(f, dtype=bool)
            sel[j] = True
        h = H[sel]
        freqs.append(round(float(np.sqrt(a * b)), 2))
        spl.append(round(float(10 * np.log10(np.mean(np.abs(h) ** 2) + EPS)), 2))
        phase.append(round(float(np.degrees(np.angle(h.sum()))), 1))
    return {"freqs": freqs, "spl": spl, "phase": phase}


def analyse_ir(ir: np.ndarray, fs: float, f_lo: float = 20.0, f_hi: float = 20000.0, start_s: float = 0.0) -> dict:
    """Everything the app shows about one impulse response. start_s: when sample 0 happened, in seconds
    after the sound left the output; every time reported (ms) counts from that moment."""
    ir = np.asarray(ir, dtype=np.float64)
    if len(ir) < int(0.05 * fs) or not np.any(ir):
        raise MeasureError("The impulse response is empty.")
    e_full = ir * ir
    peak = float(e_full.max())
    noise_all = float(np.median(e_full)) + EPS
    if 10 * np.log10(peak / noise_all) < 20:
        raise MeasureError("No clear impulse in the response: the sweep was not heard above the noise. "
                           "Check the output, the mic and the level.")
    k0 = onset(ir, fs)
    # arrival: the strongest point in the first 3 ms after the onset (sub-sample), and its sign
    env = np.abs(signal.hilbert(ir[k0: k0 + int(0.02 * fs)]))
    w = max(2, int(0.003 * fs))
    j = int(np.argmax(env[:w]))
    if 0 < j < len(env) - 1:
        y0, y1, y2 = env[j - 1], env[j], env[j + 1]
        den = y0 - 2 * y1 + y2
        j = j + (0.5 * (y0 - y2) / den if den else 0.0)
    arrival = (k0 + j) / fs
    jj = k0 + int(np.argmax(np.abs(ir[k0: k0 + w])))
    polarity = 1 if ir[jj] >= 0 else -1

    x = ir[k0:]
    bb = lundeby(x * x, fs)
    bands = []
    for fc in BANDS:
        row = {"f": fc, "edt": None, "t20": None, "t30": None, "c50": None, "c80": None, "d50": None, "ts": None,
               "inr": None, "note": None}
        if fc * np.sqrt(2) > 0.45 * fs or fc / np.sqrt(2) < f_lo * 0.9 or fc * np.sqrt(2) > f_hi * 1.1:
            row["note"] = "outside the measured range"
            bands.append(row)
            continue
        sos = _octave_sos(fc, fs)
        y = signal.sosfilt(sos, ir[::-1])[::-1][k0:]  # time-reversed filtering
        e = y * y
        lb = lundeby(e, fs)
        inr = lb["peak_db"] - lb["noise_db"]
        edc = schroeder(e, fs, lb)
        tail = 0.0
        if lb["slope"] is not None and lb["cross"] < len(e):
            a = -lb["slope"] * np.log(10) / 10
            tail = 10 ** (lb["noise_db"] / 10) / (1 - np.exp(-a)) if a > 0 else 0.0
        edt, _ = _decay_time(edc, fs, 0.0, -10.0) if inr >= 20 else (None, 0)
        t20, _ = _decay_time(edc, fs, -5.0, -25.0) if inr >= 35 else (None, 0)
        t30, _ = _decay_time(edc, fs, -5.0, -35.0) if inr >= 45 else (None, 0)
        row.update(edt=_r(edt), t20=_r(t20), t30=_r(t30), inr=_r(inr, 1),
                   **{k: _r(v, 3 if k == "d50" else 2) for k, v in _energy_params(e, fs, lb["cross"], tail).items()})
        t_any = t30 or t20 or edt
        bw = fc * (np.sqrt(2) - 1 / np.sqrt(2))
        if t_any and bw * t_any < 16:
            row["note"] = "decay too short for this band's filter: read it with care"
        elif inr < 35:
            row["note"] = "too little range above the noise for T20/T30"
        row["_edc"] = edc
        bands.append(row)

    def mid(key):
        v = [bands[i][key] for i in MID if bands[i][key] is not None]
        return round(float(np.mean(v)), 3 if key == "d50" else 2) if len(v) == len(MID) else None

    # curves for the viewer: ETC from 5 ms before the onset, energy decay per band
    span = int(min(len(ir), k0 + max(bb["cross"] * 1.3, 0.2 * fs)))
    a0 = max(0, k0 - int(0.005 * fs))
    envf = np.abs(signal.hilbert(ir[a0:span])) ** 2
    etc = 10 * np.log10(envf / (envf.max() + EPS) + EPS)
    npts = 1200
    curves = {
        "t0_ms": round((a0 / fs + start_s) * 1000, 3),
        "dt_ms": round((span - a0) / fs * 1000 / min(npts, span - a0), 4),
        "etc": _decimate_curve(etc, npts, "max"),
        "edc": {str(int(b["f"])): _decimate_curve(b["_edc"][: span - k0], 300, "first")
                for b in bands if "_edc" in b},
        "edc_dt_ms": round((span - k0) / fs * 1000 / min(300, span - k0), 4),
        "edc_t0_ms": round((k0 / fs + start_s) * 1000, 3),
    }
    for b in bands:
        b.pop("_edc", None)
    return {
        "fs": int(fs),
        "length_s": round(len(ir) / fs, 3),
        "onset_ms": round((k0 / fs + start_s) * 1000, 3),
        "arrival_ms": round((arrival + start_s) * 1000, 3),
        "polarity": polarity,
        "peak_db": round(10 * np.log10(peak + EPS), 1),
        "inr_db": round(bb["peak_db"] - bb["noise_db"], 1),
        "bands": bands,
        "single": {"edt": mid("edt"), "t20": mid("t20"), "t30": mid("t30"), "c50": mid("c50"),
                   "c80": mid("c80"), "d50": mid("d50"), "ts": mid("ts")},
        "response": response(ir[k0 - min(k0, int(0.002 * fs)):], fs, arrival - (k0 - min(k0, int(0.002 * fs))) / fs,
                             f_lo, f_hi),
        "curves": curves,
    }


def trim_ir(ir: np.ndarray, fs: float, max_s: float = 10.0) -> tuple[np.ndarray, int]:
    """The part of an impulse response worth keeping: from 50 ms before the onset to the broadband noise
    crossing plus a quarter of that (at most max_s), with a short fade at the end. Returns it and the index
    of its first sample in `ir`, so times stay counted from the original start."""
    k0 = onset(ir, fs)
    a = max(0, k0 - int(0.05 * fs))
    x = ir[k0:]
    lb = lundeby(x * x, fs)
    end = min(len(ir), k0 + int(lb["cross"] * 1.25) + int(0.05 * fs), a + int(max_s * fs))
    y = ir[a:end].copy()
    k = min(len(y) // 8, int(0.01 * fs))
    if k:
        y[len(y) - k:] *= 0.5 + 0.5 * np.cos(np.pi * np.arange(k) / k)
    return y, a


def capture(rec: np.ndarray, ref: np.ndarray, fs: float, f1: float, f2: float, loop: np.ndarray | None = None) -> dict:
    """A sweep capture: mic recording, what was played (sample-aligned), optional loopback of the output.
    Returns {ir (trimmed), start_s (when its sample 0 happened), analysis, warnings}. With a loopback, times are from when the sound left the
    interface; without one they include the fixed latency of the audio chain (fine for comparing takes)."""
    rec = np.asarray(rec, dtype=np.float64)
    warnings = []
    clip = int(np.sum(np.abs(rec) >= 0.999))
    if clip > 3:
        raise MeasureError(f"The recording clipped ({clip} samples at full scale). Turn the level or the "
                           "mic gain down and measure again.")
    peak_dbfs = 20 * np.log10(np.max(np.abs(rec)) + EPS)
    if peak_dbfs < -60:
        raise MeasureError(f"The mic heard almost nothing (peak {peak_dbfs:.0f} dBFS). Check the input and the level.")
    if peak_dbfs < -40:
        warnings.append(f"The recording is quiet (peak {peak_dbfs:.0f} dBFS); turn the level up for a cleaner result.")
    ir = deconvolve(rec, ref, fs, f1, f2)
    t0 = 0.0
    if loop is not None:
        li = deconvolve(np.asarray(loop, dtype=np.float64), ref, fs, f1, f2)
        t0 = int(np.argmax(np.abs(li))) / fs
    ir_t, a = trim_ir(ir, fs)
    start = a / fs - t0  # when the trimmed response's sample 0 happened, from when the sound left
    an = analyse_ir(ir_t, fs, f1, f2, start_s=start)
    if an["inr_db"] < 35:
        warnings.append(f"Only {an['inr_db']:.0f} dB above the noise: T30 needs 45 dB. Use a longer sweep, a higher "
                        "level, or average several.")
    an["peak_dbfs"] = round(float(peak_dbfs), 1)
    an["loopback"] = loop is not None
    return {"ir": ir_t, "start_s": start, "analysis": an, "warnings": warnings}
