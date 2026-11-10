from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from threading import Event
from typing import Optional

import numpy as np
from numpy.typing import NDArray

from acoustic_phase_optimizer.measurement.signal_generator import SignalGenerator
from acoustic_phase_optimizer.measurement.impulse_response import ImpulseResponse
from acoustic_phase_optimizer.measurement.room_analysis import RoomAnalysis
from acoustic_phase_optimizer.measurement.rt60 import RT60Estimator
from acoustic_phase_optimizer.measurement.calibration import (
    measure_loopback_latency,
    check_clipping,
    ClippingReport,
    load_mic_calibration,
    apply_mic_calibration,
)
from acoustic_phase_optimizer.utils.audio_utils import AudioUtils
from acoustic_phase_optimizer.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class MeasurementResult:
    impulse_responses: dict[str, NDArray[np.float64]]
    raw_recordings: dict[str, NDArray[np.float64]]
    sample_rate: int
    sweep_type: str
    duration_s: float
    clipping_reports: dict[str, ClippingReport]
    loopback_latency_s: float = 0.0
    mic_calibration_applied: bool = False
    analysis: dict = field(default_factory=dict)
    rt60: dict = field(default_factory=dict)
    phase_data: dict = field(default_factory=dict)
    magnitude_data: dict = field(default_factory=dict)
    transfer_functions: dict[str, tuple[NDArray, NDArray]] = field(default_factory=dict)
    cancelled: bool = False


class MeasurementRunner:
    def __init__(
        self,
        sample_rate: int = 48000,
        output_device: Optional[int] = None,
        input_device: Optional[int] = None,
    ):
        self.sample_rate = sample_rate
        self.output_device = output_device
        self.input_device = input_device
        self._cancel_event = Event()
        self.signal_gen = SignalGenerator(sample_rate)
        self.ir_module = ImpulseResponse(sample_rate)
        self.room_analysis = RoomAnalysis(sample_rate)
        self.rt60_estimator = RT60Estimator(sample_rate)

    def cancel(self) -> None:
        self._cancel_event.set()

    @property
    def cancelled(self) -> bool:
        return self._cancel_event.is_set()

    def set_devices(
        self,
        output_device: Optional[int] = None,
        input_device: Optional[int] = None,
    ) -> None:
        self.output_device = output_device
        self.input_device = input_device

    def run_measurement(
        self,
        sweep_type: str = "log",
        duration: float = 5.0,
        start_freq: float = 20.0,
        end_freq: float = 20000.0,
        averaging: int = 3,
        mls_order: int = 15,
        output_channels: list[int] | None = None,
        input_channels: list[int] | None = None,
        mic_cal_path: str | Path | None = None,
        measure_latency: bool = False,
        channel_labels: list[str] | None = None,
    ) -> MeasurementResult:
        if self.cancelled:
            return MeasurementResult(
                impulse_responses={}, raw_recordings={},
                sample_rate=self.sample_rate, sweep_type=sweep_type,
                duration_s=duration, clipping_reports={}, cancelled=True,
            )

        logger.info(
            f"Starting measurement: {sweep_type}, {duration}s, "
            f"{start_freq}-{end_freq}Hz, avg={averaging}"
        )

        signal = self.signal_gen.generate(
            sweep_type=sweep_type,
            duration=duration,
            start_freq=start_freq,
            end_freq=end_freq,
            mls_order=mls_order,
            averaging=averaging,
        )

        signal = AudioUtils.normalize_signal(signal, peak_level=0.85)
        total_channels = max(len(output_channels) if output_channels else 1,
                             len(input_channels) if input_channels else 1)

        loopback_latency_s = 0.0
        if measure_latency:
            loopback_latency_s = measure_loopback_latency(
                self.sample_rate,
                device=self.input_device,
            )
            logger.info(f"Loopback latency: {loopback_latency_s * 1000:.2f} ms")

        cal_freqs: NDArray | None = None
        cal_corrections: NDArray | None = None
        if mic_cal_path:
            try:
                cal_freqs, cal_corrections = load_mic_calibration(str(mic_cal_path))
                logger.info(f"Loaded mic calibration: {len(cal_freqs)} points")
            except Exception as e:
                logger.error(f"Failed to load mic calibration: {e}")

        raw_recordings: dict[str, NDArray] = {}
        impulse_responses: dict[str, NDArray] = {}
        clipping_reports: dict[str, ClippingReport] = {}
        transfer_functions: dict[str, tuple[NDArray, NDArray]] = {}

        import sounddevice as sd

        ch_count = max(total_channels, 1)
        n_samples = len(signal)

        if output_channels:
            out_signal = np.zeros((n_samples, max(output_channels) + 1))
            for ch in output_channels:
                out_signal[:, ch] = signal
        else:
            out_signal = signal.reshape(-1, 1)

        recorded = sd.playrec(
            out_signal,
            samplerate=self.sample_rate,
            output_device=self.output_device,
            input_device=self.input_device,
            channels=ch_count,
            blocking=False,
        )

        import time
        poll_interval = 0.05
        elapsed = 0.0
        expected = len(out_signal) / self.sample_rate
        while sd.get_status() is None and elapsed < expected + 1.0:
            if self.cancelled:
                sd.stop()
                logger.info("Measurement cancelled")
                return MeasurementResult(
                    impulse_responses={}, raw_recordings={},
                    sample_rate=self.sample_rate, sweep_type=sweep_type,
                    duration_s=duration, clipping_reports={}, cancelled=True,
                )
            time.sleep(poll_interval)
            elapsed += poll_interval

        sd.wait()

        if recorded.ndim == 1:
            recorded = recorded.reshape(-1, 1)

        if input_channels:
            for i, ch in enumerate(input_channels):
                label = channel_labels[i] if channel_labels and i < len(channel_labels) else f"input_{ch}"
                ch_data = recorded[:, ch] if recorded.shape[1] > ch else recorded[:, 0]
                raw_recordings[label] = ch_data.astype(np.float64)
        else:
            for ch in range(recorded.shape[1]):
                raw_recordings[f"input_{ch}"] = recorded[:, ch].astype(np.float64)

        for label, rec in raw_recordings.items():
            clip = check_clipping(rec)
            clipping_reports[label] = clip

            if sweep_type == "log":
                inv_sweep = self.signal_gen.inverse_sweep(
                    duration=duration, start_freq=start_freq, end_freq=end_freq,
                )
                ir = self.ir_module.extract_from_sweep(rec, signal)
            else:
                ir = self.ir_module.extract_from_mls(rec, signal)

            impulse_responses[label] = ir

            freqs_mag, magnitude_db = self.ir_module.extract_magnitude(ir)
            freqs_phase, phase = self.ir_module.extract_phase(ir)

            if cal_freqs is not None and cal_corrections is not None:
                magnitude_db = apply_mic_calibration(
                    magnitude_db, freqs_mag, cal_freqs, cal_corrections,
                )

            transfer_functions[label] = (freqs_mag, magnitude_db)

        analysis: dict = {}
        for label, ir in impulse_responses.items():
            analysis[label] = self.room_analysis.analyze_impulse_response(ir)

        rt60_bands = [63, 125, 250, 500, 1000, 2000, 4000, 8000]
        rt60: dict = {}
        for label, ir in impulse_responses.items():
            rt60[label] = self.rt60_estimator.estimate_rt60(ir, bands=rt60_bands)

        logger.info(f"Measurement complete: {len(impulse_responses)} channels")

        return MeasurementResult(
            impulse_responses=impulse_responses,
            raw_recordings=raw_recordings,
            sample_rate=self.sample_rate,
            sweep_type=sweep_type,
            duration_s=duration,
            clipping_reports=clipping_reports,
            loopback_latency_s=loopback_latency_s,
            mic_calibration_applied=cal_freqs is not None,
            analysis=analysis,
            rt60=rt60,
            transfer_functions=transfer_functions,
        )
