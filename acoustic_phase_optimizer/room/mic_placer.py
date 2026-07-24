from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

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
    grid_resolution: int = 15,
    min_spacing: float = 1.0,
    ear_height: float = 1.2,
) -> MicPlacementResult:
    L, W, H = room.get_dimensions_array()
    xs = np.linspace(-L / 2 + 1, L / 2 - 1, grid_resolution)
    ys = np.linspace(-W / 2 + 1, W / 2 - 1, grid_resolution)
    X, Y = np.meshgrid(xs, ys)
    candidates = np.column_stack([X.ravel(), Y.ravel()])

    if len(candidates) > 500:
        idx = np.random.choice(len(candidates), 500, replace=False)
        candidates = candidates[idx]

    scores = []
    for cand in candidates:
        pt = np.array([cand[0], cand[1], ear_height])
        score = _score_candidate(pt, speakers, room)
        scores.append(score)

    scores = np.array(scores)
    order = np.argsort(-scores)

    selected = []
    selected_names = []
    for idx in order:
        pt = candidates[idx]
        if len(selected) >= max_mics:
            break
        too_close = False
        for s in selected:
            dist = np.linalg.norm(pt - s)
            if dist < min_spacing:
                too_close = True
                break
        if too_close:
            continue
        selected.append(pt.copy())
        selected_names.append(f"Mic {len(selected)}")

    positions = [np.array([p[0], p[1], ear_height]) for p in selected]

    coverage = float(np.mean(scores[order[:len(selected)]])) if len(selected) > 0 else 0.0
    diversity = _diversity_score(positions)

    logger.info(
        f"Placed {len(positions)} mics (coverage={coverage:.3f}, diversity={diversity:.3f})"
    )
    return MicPlacementResult(
        positions=positions,
        names=selected_names,
        coverage_score=coverage,
        diversity_score=diversity,
    )


def _score_candidate(pt: np.ndarray, speakers: List[Speaker], room: RoomModel) -> float:
    score = 0.0
    for spk in speakers:
        if not spk.enabled:
            continue
        d = np.linalg.norm(pt - spk.position)
        if d < 0.5:
            d = 0.5
        score += 1.0 / d

    cx = pt[0] / (room.dimensions.length / 2)
    cy = pt[1] / (room.dimensions.width / 2)
    center_penalty = np.sqrt(cx**2 + cy**2) * 0.3
    score -= center_penalty

    return score


def _diversity_score(positions: List[np.ndarray]) -> float:
    if len(positions) < 2:
        return 1.0
    dists = []
    for i in range(len(positions)):
        for j in range(i + 1, len(positions)):
            dists.append(np.linalg.norm(positions[i] - positions[j]))
    mean_dist = float(np.mean(dists)) if dists else 1.0
    return float(min(mean_dist / 2.0, 1.0))
