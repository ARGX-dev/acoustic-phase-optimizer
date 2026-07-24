"""Audio utility functions for playback, recording, and file I/O."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
import sounddevice as sd
import soundfile as sf
from typing import Optional, Tuple, List
from acoustic_phase_optimizer.utils.logging import get_logger

logger = get_logger(__name__)


class AudioUtils:
    """Utilities for audio playback, recording, and file operations."""

    @staticmethod
    def list_devices() -> List[dict]:
        return list(sd.query_devices())

    @staticmethod
    def list_input_devices() -> List[dict]:
        devices = sd.query_devices()
        return [d for d in devices if d["max_input_channels"] > 0]

    @staticmethod
    def list_output_devices() -> List[dict]:
        devices = sd.query_devices()
        return [d for d in devices if d["max_output_channels"] > 0]

    @staticmethod
    def play_signal(
        signal: NDArray[np.float64],
        sample_rate: int = 48000,
        device: Optional[int] = None,
        blocking: bool = True,
    ) -> None:
        sd.play(signal, samplerate=sample_rate, device=device, blocking=blocking)

    @staticmethod
    def record_audio(
        duration: float,
        sample_rate: int = 48000,
        channels: int = 1,
        device: Optional[int] = None,
        dtype: str = "float64",
    ) -> NDArray[np.float64]:
        recording = sd.rec(
            int(duration * sample_rate),
            samplerate=sample_rate,
            channels=channels,
            device=device,
            dtype=dtype,
            blocking=True,
        )
        return recording

    @staticmethod
    def play_and_record(
        signal: NDArray[np.float64],
        sample_rate: int = 48000,
        output_device: Optional[int] = None,
        input_device: Optional[int] = None,
        channels: int = 1,
    ) -> NDArray[np.float64]:
        recording = sd.playrec(
            signal,
            samplerate=sample_rate,
            output_device=output_device,
            input_device=input_device,
            channels=channels,
            blocking=True,
        )
        return recording

    @staticmethod
    def write_wav(
        filepath: str,
        data: NDArray[np.float64],
        sample_rate: int = 48000,
    ) -> None:
        sf.write(filepath, data, sample_rate)
        logger.info(f"Wrote {filepath} ({sample_rate} Hz, {len(data)} samples)")

    @staticmethod
    def read_wav(filepath: str) -> Tuple[NDArray[np.float64], int]:
        data, sr = sf.read(filepath, dtype="float64")
        if data.ndim == 1:
            data = data[:, np.newaxis]
        logger.info(f"Read {filepath} ({sr} Hz, {data.shape[0]} samples, {data.shape[1]} channels)")
        return data, sr

    @staticmethod
    def normalize_signal(
        signal: NDArray[np.float64],
        peak_level: float = 0.95,
    ) -> NDArray[np.float64]:
        max_val = np.max(np.abs(signal))
        if max_val > 0:
            return signal * (peak_level / max_val)
        return signal

    @staticmethod
    def apply_fade(
        signal: NDArray[np.float64],
        fade_samples: int = 100,
    ) -> NDArray[np.float64]:
        result = signal.copy()
        if fade_samples > 0 and len(result) > 2 * fade_samples:
            fade_in = np.linspace(0, 1, fade_samples)
            fade_out = np.linspace(1, 0, fade_samples)
            result[:fade_samples] *= fade_in
            result[-fade_samples:] *= fade_out
        return result

    @staticmethod
    def silence_detection(
        signal: NDArray[np.float64],
        threshold_db: float = -60.0,
    ) -> bool:
        level = 20 * np.log10(np.max(np.abs(signal)) + 1e-12)
        return level < threshold_db

    @staticmethod
    def compute_rms(signal: NDArray[np.float64]) -> float:
        return float(np.sqrt(np.mean(signal ** 2)))

    @staticmethod
    def compute_peak(signal: NDArray[np.float64]) -> float:
        return float(np.max(np.abs(signal)))
