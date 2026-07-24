from __future__ import annotations

import numpy as np
from dataclasses import dataclass
from typing import List, Optional

from acoustic_phase_optimizer.acoustic.room_model import RoomModel
from acoustic_phase_optimizer.acoustic.speaker import Speaker
from acoustic_phase_optimizer.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class MicPlacementResult:
    positions: List[np.ndarray]
    names: List[str]
    coverage_score: float
    diversity_score: float


def optimize_mic_positions(
    room: RoomModel,
    speakers: List[Speaker],
    max_mics: int = 8,
    grid_resolution: int = 20,
    min_spacing: float = 1.5,
    ear_height: float = 1.2,
) -> MicPlacementResult:
    L, W, H = room.get_dimensions_array()

    xs = np.linspace(-L / 2 + 1, L / 2 - 1, grid_resolution)
    ys = np.linspace(-W / 2 + 1, W / 2 - 1, grid_resolution)
    X, Y = np.meshgrid(xs, ys)
    candidates = np.column_stack([X.ravel(), Y.ravel()])

    n_zone_cols = min(3, max(1, max_mics // 2))
    n_zone_rows = min(2, max(1, max_mics // 3))
    zones_x = np.array_split(np.linspace(-L / 2, L / 2, grid_resolution), n_zone_cols)
    zones_y = np.array_split(np.linspace(-W / 2, W / 2, grid_resolution), n_zone_rows)

    selected = []
    selected_names = []
    zone_counts = np.zeros((len(zones_y), len(zones_x)))

    available = list(range(len(candidates)))

    while len(selected) < max_mics and available:
        best_idx = None
        best_score = -np.inf

        for ci in available:
            pt = candidates[ci]

            too_close = False
            for s in selected:
                if np.linalg.norm(pt - s) < min_spacing:
                    too_close = True
                    break
            if too_close:
                continue

            speaker_dist = 0.0
            for spk in speakers:
                if not spk.enabled:
                    continue
                d = np.linalg.norm(pt - spk.position[:2])
                if d < 1.0:
                    d = 1.0
                speaker_dist += d
            speaker_term = speaker_dist / max(len([s for s in speakers if s.enabled]), 1)

            zone_x = np.searchsorted([z[-1] for z in zones_x], pt[0])
            zone_y = np.searchsorted([z[-1] for z in zones_y], pt[1])
            zone_x = min(zone_x, n_zone_cols - 1)
            zone_y = min(zone_y, n_zone_rows - 1)
            zone_penalty = zone_counts[zone_y, zone_x] * 2.0

            dist_to_selected = 0.0
            if selected:
                dists = [np.linalg.norm(pt - s) for s in selected]
                dist_to_selected = min(dists)

            diversity_term = dist_to_selected if selected else L

            cx = pt[0] / (L / 2)
            cy = pt[1] / (W / 2)
            center_term = 1.0 - min(np.sqrt(cx**2 + cy**2), 1.0) * 0.3

            score = (
                speaker_term * 0.3
                + diversity_term * 10.0
                + center_term * 5.0
                - zone_penalty * 2.0
            )

            if score > best_score:
                best_score = score
                best_idx = ci

        if best_idx is None:
            break

        pt = candidates[best_idx]
        zone_x = np.searchsorted([z[-1] for z in zones_x], pt[0])
        zone_y = np.searchsorted([z[-1] for z in zones_y], pt[1])
        zone_x = min(zone_x, n_zone_cols - 1)
        zone_y = min(zone_y, n_zone_rows - 1)
        zone_counts[zone_y, zone_x] += 1

        selected.append(pt.copy())
        selected_names.append(f"Mic {len(selected) + 1}")
        available.remove(best_idx)

    if len(selected) < max_mics:
        remaining = max_mics - len(selected)
        for ci in available[:remaining]:
            pt = candidates[ci]
            selected.append(pt.copy())
            selected_names.append(f"Mic {len(selected) + 1}")

    positions = [np.array([p[0], p[1], ear_height]) for p in selected]

    if len(positions) >= 2:
        dists = []
        for i in range(len(positions)):
            for j in range(i + 1, len(positions)):
                dists.append(np.linalg.norm(positions[i] - positions[j]))
        diversity = float(min(np.mean(dists) / 3.0, 1.0)) if dists else 1.0
    else:
        diversity = 1.0

    coverage = min(len(positions) / max_mics, 1.0)

    logger.info(
        f"Placed {len(positions)} mics (coverage={coverage:.3f}, diversity={diversity:.3f})"
    )
    return MicPlacementResult(
        positions=positions,
        names=selected_names,
        coverage_score=coverage,
        diversity_score=diversity,
    )
