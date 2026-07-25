"""Measurement pipeline correctness layer.

Fixes three classes of bug that silently corrupt phase/delay measurements:

  1. Unknown output->input latency (driver buffering, USB round-trip)
  2. Requested vs actual sample rate mismatch
  3. Input clipping during a sweep
  4. Missing mic calibration curve
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import sounddevice as sd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# 1. Loopback latency calibration
# ---------------------------------------------------------------------------

def measure_loopback_latency(
    samplerate: int,
    device: tuple[int, int] | None = None,
    sweep_duration_s: float = 1.0,
) -> float:
    """
    Play a short sweep and record it via a loopback path (physically patch
    an output back into a spare input, or digital loopback if available),
    then cross-correlate to find the exact sample offset.

    Subtract this latency (in seconds) from every subsequent time-of-flight
    measurement before converting to distance or delay.
    """
    t = np.linspace(0, sweep_duration_s, int(samplerate * sweep_duration_s), endpoint=False)
    f0, f1 = 100, 10_000
    sweep = np.sin(2 * np.pi * (f0 * t + (f1 - f0) / (2 * sweep_duration_s) * t**2))
    sweep = sweep.astype(np.float32) * 0.5

    recording = sd.playrec(sweep, samplerate=samplerate, channels=1, device=device)
    sd.wait()
    recorded = recording[:, 0]

    correlation = np.correlate(recorded, sweep, mode="full")
    lag_samples = np.argmax(correlation) - (len(sweep) - 1)

    if lag_samples < 0:
        logger.warning(
            "Loopback correlation returned negative lag (%d samples) — "
            "check loopback routing before trusting this offset.",
            lag_samples,
        )
        lag_samples = 0

    latency_s = lag_samples / samplerate
    logger.info("Loopback latency: %.2f ms (%d samples)", latency_s * 1000, lag_samples)
    return latency_s


# ---------------------------------------------------------------------------
# 2. Sample rate verification
# ---------------------------------------------------------------------------

def open_verified_stream(
    requested_samplerate: int,
    **stream_kwargs,
) -> tuple[sd.Stream, int]:
    """
    Open a stream and return the ACTUAL negotiated sample rate alongside it.
    Always use the returned rate for FFT bin sizing and time-of-flight math.
    """
    stream = sd.Stream(samplerate=requested_samplerate, **stream_kwargs)
    actual_rate = int(stream.samplerate)
    if actual_rate != requested_samplerate:
        logger.warning(
            "Requested %d Hz but device opened at %d Hz — using actual rate.",
            requested_samplerate,
            actual_rate,
        )
    return stream, actual_rate


# ---------------------------------------------------------------------------
# 3. Clipping detection
# ---------------------------------------------------------------------------

@dataclass
class ClippingReport:
    clipped: bool
    peak_level: float
    clipped_sample_count: int
    clipped_fraction: float


def check_clipping(recording: np.ndarray, threshold: float = 0.999) -> ClippingReport:
    """
    Run immediately after every capture, before any analysis. A clipped sweep
    produces phase data that looks plausible on a magnitude plot but is wrong.
    """
    peak = float(np.max(np.abs(recording)))
    clipped_mask = np.abs(recording) >= threshold
    clipped_count = int(np.sum(clipped_mask))
    fraction = clipped_count / len(recording) if len(recording) else 0.0

    report = ClippingReport(
        clipped=clipped_count > 0,
        peak_level=peak,
        clipped_sample_count=clipped_count,
        clipped_fraction=fraction,
    )
    if report.clipped:
        logger.error(
            "Clipping: peak=%.3f, %d samples (%.3f%%). Reduce input gain and re-measure.",
            peak,
            clipped_count,
            fraction * 100,
        )
    return report


# ---------------------------------------------------------------------------
# 4. Microphone calibration curve
# ---------------------------------------------------------------------------

def load_mic_calibration(cal_file_path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """
    Load a mic calibration file in the common two-column format:
        freq_hz  correction_db

    Most measurement mic manufacturers ship this format. Returns (frequencies,
    corrections_db) sorted by frequency.
    """
    data = np.loadtxt(cal_file_path, comments=("*", "#", ";"))
    freqs, corrections = data[:, 0], data[:, 1]
    order = np.argsort(freqs)
    return freqs[order], corrections[order]


def apply_mic_calibration(
    magnitude_db: np.ndarray,
    freq_bins_hz: np.ndarray,
    cal_freqs: np.ndarray,
    cal_corrections_db: np.ndarray,
) -> np.ndarray:
    """
    Subtract the mic's own frequency response from a measured magnitude
    spectrum. Without this, measurements above ~10 kHz and below ~40 Hz
    are partly mic coloration.
    """
    correction = np.interp(freq_bins_hz, cal_freqs, cal_corrections_db)
    return magnitude_db - correction
