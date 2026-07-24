"""Reflection estimation using ray tracing and image source methods."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from typing import List, Optional, Tuple
from acoustic_phase_optimizer.acoustic.room_model import RoomModel
from acoustic_phase_optimizer.utils.logging import get_logger

logger = get_logger(__name__)


class ReflectionEngine:
    """Estimates early reflections using image source method."""

    def __init__(
        self,
        room_model: RoomModel,
        max_order: int = 3,
    ):
        self.room_model = room_model
        self.max_order = max_order

    def compute_image_sources(
        self,
        source_pos: NDArray[np.float64],
    ) -> list[dict]:
        L, W, H = self.room_model.get_dimensions_array()
        images = []

        for order in range(1, self.max_order + 1):
            for ref_x in range(-order, order + 1):
                for ref_y in range(-order, order + 1):
                    for ref_z in range(0, order + 1):
                        if abs(ref_x) + abs(ref_y) + abs(ref_z) != order:
                            continue

                        sign_x = 1 if ref_x % 2 == 0 else -1
                        sign_y = 1 if ref_y % 2 == 0 else -1
                        sign_z = 1 if ref_z % 2 == 0 else -1

                        img_x = sign_x * source_pos[0] + 2 * ref_x * source_pos[0]
                        img_y = sign_y * source_pos[1] + 2 * ref_y * source_pos[1]
                        img_z = sign_z * source_pos[2] + 2 * ref_z * source_pos[2]

                        img_pos = np.array([img_x, img_y, img_z])
                        distance = np.linalg.norm(img_pos - source_pos)

                        absorption = self.room_model.average_absorption()
                        reflection_coeff = (1.0 - absorption) ** order

                        images.append({
                            "order": order,
                            "position": img_pos,
                            "distance": distance,
                            "reflection_coefficient": reflection_coeff,
                            "image_indices": (ref_x, ref_y, ref_z),
                        })

        images.sort(key=lambda x: x["distance"])
        return images

    def compute_reflections(
        self,
        source_pos: NDArray[np.float64],
        receiver_pos: NDArray[np.float64],
    ) -> list[dict]:
        images = self.compute_image_sources(source_pos)
        reflections = []

        direct_distance = np.linalg.norm(source_pos - receiver_pos)
        direct_delay = direct_distance / self.room_model.speed_of_sound

        reflections.append({
            "type": "direct",
            "order": 0,
            "delay_s": direct_delay,
            "distance": direct_distance,
            "amplitude": 1.0,
            "source_position": source_pos,
            "receiver_position": receiver_pos,
        })

        for img in images:
            img_pos = img["position"]
            total_distance = np.linalg.norm(img_pos - receiver_pos)
            delay = total_distance / self.room_model.speed_of_sound
            attenuation = 1.0 / (total_distance + 0.01)
            amplitude = img["reflection_coefficient"] * attenuation

            reflections.append({
                "type": f"reflection_order_{img['order']}",
                "order": img["order"],
                "delay_s": delay,
                "distance": total_distance,
                "amplitude": amplitude,
                "reflection_coefficient": img["reflection_coefficient"],
                "image_position": img_pos,
            })

        reflections.sort(key=lambda x: x["delay_s"])
        return reflections

    def compute_impulse_response(
        self,
        source_pos: NDArray[np.float64],
        receiver_pos: NDArray[np.float64],
        sample_rate: int = 48000,
        max_delay_s: float = 2.0,
    ) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
        reflections = self.compute_reflections(source_pos, receiver_pos)
        max_samples = int(max_delay_s * sample_rate)

        ir = np.zeros(max_samples)
        times = np.arange(max_samples) / sample_rate

        for ref in reflections:
            sample_idx = int(ref["delay_s"] * sample_rate)
            if sample_idx < max_samples:
                ir[sample_idx] += ref["amplitude"]

        ir = ir.astype(np.float64)
        return ir, times

    def compute_reverb_time_estimate(
        self,
        source_pos: NDArray[np.float64],
        receiver_pos: NDArray[np.float64],
    ) -> float:
        reflections = self.compute_reflections(source_pos, receiver_pos)
        late_reflections = [r for r in reflections if r["order"] >= 2]

        if not late_reflections:
            return 0.0

        delays = np.array([r["delay_s"] for r in late_reflections])
        amplitudes = np.array([abs(r["amplitude"]) for r in late_reflections])

        if len(delays) < 2:
            return 0.0

        decay_rate = np.polyfit(delays, 20 * np.log10(amplitudes + 1e-12), 1)[0]
        if decay_rate >= 0:
            return 0.0

        rt60 = -60.0 / decay_rate
        return float(np.clip(rt60, 0.0, 10.0))
