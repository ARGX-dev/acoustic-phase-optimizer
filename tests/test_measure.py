"""Room measurement (engine/measure.py) against signals whose answers are known."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.signal import fftconvolve

from acoustic_phase_optimizer.engine import measure as M

FS = 48000


def decay(T: float, seconds: float = 3.0, delay_s: float = 0.0, noise_db: float = -110.0, direct: float = 0.0,
          seed: int = 1) -> np.ndarray:
    """Exponentially decaying white noise (a diffuse field with reverberation time T), optionally after a
    direct-sound spike, plus steady noise noise_db below the peak."""
    rng = np.random.default_rng(seed)
    n = int(seconds * FS)
    t = np.arange(n) / FS
    tail = rng.standard_normal(n) * np.exp(-6.9078 * t / T)
    k = int(round(delay_s * FS))
    ir = np.zeros(n)
    ir[k:] = tail[: n - k]
    if direct:
        ir[k] = direct
    ir /= np.abs(ir).max()
    return ir + rng.standard_normal(n) * 10 ** (noise_db / 20)


@pytest.mark.parametrize("T", [0.4, 1.0, 2.0])
def test_decay_times_match_the_room(T):
    a = M.analyse_ir(decay(T), FS)
    # a noise decay scatters more in narrow (low) bands: ISO 3382-2 puts T20's spread at ~0.88 T / sqrt(B T)
    for b in a["bands"][2:]:
        tol = 0.06 if b["f"] >= 1000 else 0.15
        assert b["t30"] == pytest.approx(T, rel=tol), b
        assert b["t20"] == pytest.approx(T, rel=tol + 0.02), b
        assert b["edt"] == pytest.approx(T, rel=0.2 if b["f"] >= 1000 else 0.3), b
    assert a["single"]["t30"] == pytest.approx(T, rel=0.05)


def test_clarity_and_definition_of_a_pure_decay():
    # energy exp(-13.8 t / T): C80 = 10 log10(exp(13.8 * 0.08 / T) - 1), D50 = 1 - exp(-13.8 * 0.05 / T)
    T = 1.0
    a = M.analyse_ir(decay(T, seed=3), FS)
    k = 6.9078 * 2 / T
    assert a["single"]["c80"] == pytest.approx(10 * np.log10(np.exp(k * 0.08) - 1), abs=0.6)
    assert a["single"]["d50"] == pytest.approx(1 - np.exp(-k * 0.05), abs=0.04)
    assert a["single"]["ts"] == pytest.approx(1000 / k, rel=0.08)  # centre time of an exponential: 1/k


def test_a_click_is_all_early_sound():
    ir = np.zeros(FS)
    ir[1000] = 1.0
    a = M.analyse_ir(ir, FS)
    assert a["arrival_ms"] == pytest.approx(1000 / FS * 1000, abs=0.03)
    mids = [a["bands"][i] for i in M.MID]
    assert all(b["d50"] > 0.99 and b["c80"] > 20 for b in mids), mids


def test_noise_limits_what_is_reported():
    # 40 dB of range: EDT yes, T30 no - never a made-up number
    a = M.analyse_ir(decay(1.0, noise_db=-42), FS)
    mids = [a["bands"][i] for i in M.MID]
    assert all(b["t30"] is None and b["edt"] is not None for b in mids), mids
    assert all(b["note"] for b in mids)
    assert a["single"]["t30"] is None


def test_sweep_round_trip_finds_delay_polarity_and_decay():
    rng = np.random.default_rng(7)
    s = M.sweep(FS, 2.0, 20, 20000)
    room = decay(1.0, 1.5, delay_s=0.0123, direct=20)
    pre = int(0.1 * FS)
    ref = np.concatenate([np.zeros(pre), s, np.zeros(int(1.5 * FS))])
    rec = fftconvolve(ref, room)[: len(ref)]
    rec = rec / np.abs(rec).max() * 0.5 + rng.standard_normal(len(ref)) * 1e-5
    c = M.capture(rec, ref, FS, 20, 20000)
    an = c["analysis"]
    assert an["arrival_ms"] == pytest.approx(12.3, abs=0.05)
    assert an["polarity"] == 1 and not c["warnings"]
    assert an["single"]["t30"] == pytest.approx(1.0, rel=0.06)
    assert M.capture(-rec, ref, FS, 20, 20000)["analysis"]["polarity"] == -1
    # a loopback 2.5 ms late (the interface's own latency) is taken off the arrival
    loop = np.concatenate([np.zeros(int(0.0025 * FS)), ref])[: len(ref)]
    assert M.capture(rec, ref, FS, 20, 20000, loop=loop)["analysis"]["arrival_ms"] == pytest.approx(9.8, abs=0.05)


def test_harmonic_distortion_stays_out_of_the_response():
    s = M.sweep(FS, 2.0, 20, 20000)
    pre = int(0.1 * FS)
    ref = np.concatenate([np.zeros(pre), s, np.zeros(FS)])
    lin = np.concatenate([np.zeros(240), ref])[: len(ref)]  # 5 ms of flight, no room
    rec = 0.4 * lin + 0.04 * lin ** 2 + 0.02 * lin ** 3  # 2nd and 3rd harmonic distortion
    ir = M.deconvolve(rec, ref, FS, 20, 20000)
    e = ir ** 2
    k = int(np.argmax(e))
    assert k == 240
    # nothing within 40 dB of the peak anywhere else in the kept response
    others = np.delete(e, np.arange(k - 48, k + 48))
    assert 10 * np.log10(others.max() / e[k]) < -40


def test_bad_takes_are_rejected():
    rng = np.random.default_rng(2)
    s = M.sweep(FS, 1.0)
    ref = np.concatenate([s, np.zeros(FS)])
    with pytest.raises(M.MeasureError, match="clipped"):
        M.capture(np.clip(np.concatenate([s, np.zeros(FS)]) * 2, -1, 1), ref, FS, 20, 20000)
    with pytest.raises(M.MeasureError, match="almost nothing"):
        M.capture(rng.standard_normal(len(ref)) * 1e-5, ref, FS, 20, 20000)
    with pytest.raises(M.MeasureError, match="not heard"):
        M.capture(rng.standard_normal(len(ref)) * 0.1, ref, FS, 20, 20000)


def test_response_is_flat_with_the_delay_removed():
    ir = np.zeros(FS // 2)
    ir[480] = 0.5  # 10 ms
    a = M.analyse_ir(ir, FS)
    r = a["response"]
    band = [(f, s, p) for f, s, p in zip(r["freqs"], r["spl"], r["phase"], strict=True) if 50 < f < 15000]
    assert max(abs(s - band[0][1]) for _, s, _ in band) < 0.5
    assert max(abs(p) for _, _, p in band) < 5, band[:5]
    assert len(r["freqs"]) <= 2000


def test_wav_round_trip_and_pcm():
    import io

    from scipy.io import wavfile

    x = np.sin(np.arange(4800) / 10) * 0.5
    y, fs = M.read_wav(M.write_wav(x, 44100))
    assert fs == 44100 and np.allclose(y, x, atol=1e-6)
    buf = io.BytesIO()
    wavfile.write(buf, 48000, np.stack([(x * 32767).astype(np.int16)] * 2, axis=1))
    y, fs = M.read_wav(buf.getvalue())
    assert fs == 48000 and np.allclose(y, x, atol=1e-4)
    with pytest.raises(M.MeasureError):
        M.read_wav(b"RIFF....not a wav")
