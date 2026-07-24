"""Virtual room simulation for optimization without real hardware."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from typing import Dict, List, Optional, Tuple
from acoustic_phase_optimizer.acoustic.room_model import RoomModel
from acoustic_phase_optimizer.acoustic.speaker import Speaker
from acoustic_phase_optimizer.acoustic.microphone import Microphone
from acoustic_phase_optimizer.acoustic.reflection import ReflectionEngine
from acoustic_phase_optimizer.utils.logging import get_logger

logger = get_logger(__name__)


class VirtualRoom:
    """Simulates room acoustics for testing optimization algorithms."""

    def __init__(
        self,
        room_model: Optional[RoomModel] = None,
        sample_rate: int = 48000,
    ):
        self.room_model = room_model or RoomModel()
        self.sample_rate = sample_rate
        self.speakers: List[Speaker] = []
        self.microphones: List[Microphone] = []
        self.reflection_engine = ReflectionEngine(self.room_model)

    def add_speaker(self, speaker: Speaker) -> None:
        self.speakers.append(speaker)

    def add_microphone(self, microphone: Microphone) -> None:
        self.microphones.append(microphone)

    def compute_transfer_function(
        self,
        speaker: Speaker,
        microphone: Microphone,
        include_reflections: bool = True,
        max_order: int = 3,
    ) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
        direct_distance = speaker.distance_to(microphone.position)
        direct_delay = direct_distance / self.room_model.speed_of_sound
        direct_attenuation = 1.0 / (direct_distance + 0.01)

        n_fft = 2 ** 14
        freqs = np.fft.rfftfreq(n_fft, 1.0 / self.sample_rate)

        transfer = np.zeros(len(freqs), dtype=np.complex128)

        phase_shift = -2.0 * np.pi * freqs * direct_delay
        transfer += direct_attenuation * np.exp(1j * phase_shift)

        if include_reflections:
            reflections = self.reflection_engine.compute_reflections(
                speaker.position, microphone.position
            )

            for ref in reflections:
                if ref["order"] == 0 or ref["order"] > max_order:
                    continue
                delay = ref["delay_s"]
                amplitude = ref["amplitude"]
                phase_shift = -2.0 * np.pi * freqs * delay
                transfer += amplitude * np.exp(1j * phase_shift)

        magnitude = np.abs(transfer)
        phase = np.angle(transfer)

        polar = speaker.polarity.value if hasattr(speaker, 'polarity') else 1
        magnitude *= polar

        magnitude_db = 20.0 * np.log10(magnitude + 1e-12)

        return freqs, magnitude_db

    def simulate_measurement(
        self,
        signal: NDArray[np.float64],
        speaker: Speaker,
        microphone: Microphone,
        add_noise: bool = True,
        noise_level_db: float = -60.0,
    ) -> NDArray[np.float64]:
        freq_response = self.compute_impulse_response(speaker, microphone)

        signal_fft = np.fft.fft(signal, n=len(freq_response))
        ir_fft = np.fft.fft(freq_response)

        response = np.fft.ifft(signal_fft * ir_fft).real

        response = response[:len(signal)]

        if add_noise:
            noise_level = 10.0 ** (noise_level_db / 20.0)
            noise = np.random.normal(0, noise_level, len(response))
            response += noise

        return response.astype(np.float64)

    def compute_impulse_response(
        self,
        speaker: Speaker,
        microphone: Microphone,
    ) -> NDArray[np.float64]:
        ir, _ = self.reflection_engine.compute_impulse_response(
            speaker.position,
            microphone.position,
            self.sample_rate,
            max_delay_s=1.0,
        )
        return ir

    def compute_multiple_transfer_functions(
        self,
    ) -> Dict[str, Dict[str, NDArray[np.float64]]]:
        results = {}
        for mic in self.microphones:
            mic_key = mic.name
            results[mic_key] = {}
            for spk in self.speakers:
                if not spk.enabled:
                    continue
                freqs, mag_db = self.compute_transfer_function(spk, mic)
                results[mic_key][spk.name] = {
                    "freqs": freqs,
                    "magnitude_db": mag_db,
                }
        return results

    def get_listening_area_mesh(
        self,
        resolution: int = 50,
    ) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
        L, W, _ = self.room_model.get_dimensions_array()
        x = np.linspace(-L / 2 + 0.5, L / 2 - 0.5, resolution)
        y = np.linspace(-W / 2 + 0.5, W / 2 - 0.5, resolution)
        return np.meshgrid(x, y)
