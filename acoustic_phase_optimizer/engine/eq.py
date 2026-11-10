"""Parametric EQ: RBJ cookbook biquads evaluated as complex frequency responses."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np

from .model import PEQBand

DESIGN_FS = 96000.0  # digital design rate; well above audio band, so curves match analog


def biquad_coeffs(kind: str, f0: float, gain_db: float, q: float, fs: float = DESIGN_FS):
    """Return (b, a) normalized biquad coefficients (Robert Bristow-Johnson cookbook)."""
    f0 = min(max(f0, 1.0), fs * 0.49)
    A = 10.0 ** (gain_db / 40.0)
    w0 = 2.0 * np.pi * f0 / fs
    cw, sw = np.cos(w0), np.sin(w0)
    alpha = sw / (2.0 * max(q, 1e-3))
    if kind == "bell":
        b = [1 + alpha * A, -2 * cw, 1 - alpha * A]
        a = [1 + alpha / A, -2 * cw, 1 - alpha / A]
    elif kind == "low_shelf":
        sa = 2 * np.sqrt(A) * alpha
        b = [A * ((A + 1) - (A - 1) * cw + sa), 2 * A * ((A - 1) - (A + 1) * cw), A * ((A + 1) - (A - 1) * cw - sa)]
        a = [(A + 1) + (A - 1) * cw + sa, -2 * ((A - 1) + (A + 1) * cw), (A + 1) + (A - 1) * cw - sa]
    elif kind == "high_shelf":
        sa = 2 * np.sqrt(A) * alpha
        b = [A * ((A + 1) + (A - 1) * cw + sa), -2 * A * ((A - 1) + (A + 1) * cw), A * ((A + 1) + (A - 1) * cw - sa)]
        a = [(A + 1) - (A - 1) * cw + sa, 2 * ((A - 1) - (A + 1) * cw), (A + 1) - (A - 1) * cw - sa]
    elif kind == "highpass":
        b = [(1 + cw) / 2, -(1 + cw), (1 + cw) / 2]
        a = [1 + alpha, -2 * cw, 1 - alpha]
    elif kind == "lowpass":
        b = [(1 - cw) / 2, 1 - cw, (1 - cw) / 2]
        a = [1 + alpha, -2 * cw, 1 - alpha]
    else:
        raise ValueError(f"unknown filter type {kind!r}")
    b = np.array(b, float) / a[0]
    a = np.array(a, float) / a[0]
    return b, a


def biquad_response(b: np.ndarray, a: np.ndarray, freqs: np.ndarray, fs: float = DESIGN_FS) -> np.ndarray:
    z1 = np.exp(-2j * np.pi * np.asarray(freqs, float) / fs)
    z2 = z1 * z1
    return (b[0] + b[1] * z1 + b[2] * z2) / (a[0] + a[1] * z1 + a[2] * z2)


def peq_response(bands: Iterable[PEQBand], freqs: np.ndarray) -> np.ndarray:
    """Complex response of a chain of PEQ bands."""
    h = np.ones(len(freqs), dtype=complex)
    for band in bands:
        if not band.enabled:
            continue
        if band.type in ("bell", "low_shelf", "high_shelf") and abs(band.gain_db) < 1e-3:
            continue
        b, a = biquad_coeffs(band.type, band.freq, band.gain_db, band.q)
        h *= biquad_response(b, a, freqs)
    return h


def bell_db(freqs: np.ndarray, f0: float, gain_db: float, q: float) -> np.ndarray:
    """Magnitude (dB) of a single bell; used by the auto-EQ fitter."""
    b, a = biquad_coeffs("bell", f0, gain_db, q)
    return 20.0 * np.log10(np.abs(biquad_response(b, a, freqs)) + 1e-12)
