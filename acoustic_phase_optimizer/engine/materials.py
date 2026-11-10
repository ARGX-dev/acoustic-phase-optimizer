"""Surface material absorption data and statistical room acoustics."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

OCTAVE_BANDS = np.array([125.0, 250.0, 500.0, 1000.0, 2000.0, 4000.0])


@dataclass(frozen=True)
class Material:
    id: str
    name: str
    alpha: tuple[float, float, float, float, float, float]  # 125 Hz .. 4 kHz


# Typical published random-incidence absorption coefficients.
MATERIALS: dict[str, Material] = {
    m.id: m
    for m in [
        Material("concrete", "Concrete, painted", (0.01, 0.01, 0.02, 0.02, 0.02, 0.03)),
        Material("concrete_block", "Concrete block, unpainted", (0.36, 0.44, 0.31, 0.29, 0.39, 0.25)),
        Material("brick", "Brick", (0.03, 0.03, 0.03, 0.04, 0.05, 0.07)),
        Material("plaster", "Plaster on masonry", (0.01, 0.02, 0.02, 0.03, 0.04, 0.05)),
        Material("gypsum", "Gypsum board on studs", (0.29, 0.10, 0.05, 0.04, 0.07, 0.09)),
        Material("glass", "Glass, large pane", (0.18, 0.06, 0.04, 0.03, 0.02, 0.02)),
        Material("wood_panel", "Wood paneling", (0.28, 0.22, 0.17, 0.09, 0.10, 0.11)),
        Material("wood_floor", "Wood floor (gym / stage)", (0.15, 0.11, 0.10, 0.07, 0.06, 0.07)),
        Material("vinyl_floor", "Vinyl / linoleum on concrete", (0.02, 0.03, 0.03, 0.03, 0.03, 0.02)),
        Material("carpet", "Carpet on pad", (0.08, 0.24, 0.57, 0.69, 0.71, 0.73)),
        Material("steel_deck", "Steel roof deck", (0.20, 0.15, 0.10, 0.10, 0.10, 0.10)),
        Material("acoustic_tile", "Acoustic ceiling tile", (0.70, 0.66, 0.72, 0.92, 0.88, 0.75)),
        Material("acoustic_panel", "Absorber panel, 50 mm", (0.18, 0.70, 0.99, 0.99, 0.99, 0.99)),
        Material("drape", "Heavy velour drape", (0.14, 0.35, 0.55, 0.72, 0.70, 0.65)),
        Material("audience", "Audience, seated", (0.39, 0.57, 0.80, 0.94, 0.92, 0.87)),
        Material("seats_empty", "Upholstered seats, empty", (0.19, 0.37, 0.56, 0.67, 0.61, 0.59)),
        Material("bleachers", "Wood bleachers, empty", (0.10, 0.12, 0.12, 0.10, 0.10, 0.10)),
        Material("open", "Open (fully absorptive)", (1.0, 1.0, 1.0, 1.0, 1.0, 1.0)),
    ]
}

SURFACES = ("floor", "ceiling", "front", "back", "left", "right")


def absorption(material_id: str, freqs: np.ndarray) -> np.ndarray:
    """Absorption coefficient interpolated on a log-frequency axis, clamped at the ends."""
    mat = MATERIALS.get(material_id) or MATERIALS["concrete"]
    a = np.interp(np.log2(np.asarray(freqs, float)), np.log2(OCTAVE_BANDS), np.array(mat.alpha))
    return np.clip(a, 0.005, 0.999)


def reflection_amplitude(material_id: str, freqs: np.ndarray) -> np.ndarray:
    """Pressure reflection magnitude sqrt(1 - alpha)."""
    return np.sqrt(1.0 - absorption(material_id, freqs))


@dataclass
class RoomStats:
    volume: float
    surface: float
    mean_alpha: np.ndarray  # per frequency
    rt60_sabine: np.ndarray
    rt60_eyring: np.ndarray
    room_constant: np.ndarray  # R = S*a / (1 - a), m^2
    critical_distance: np.ndarray  # for Q = 1, m


def room_stats(
    width: float,
    length: float,
    height: float,
    materials: dict[str, str],
    freqs: np.ndarray,
    air_db_per_m: np.ndarray,
) -> RoomStats:
    """Sabine/Eyring statistics for a shoebox room."""
    W, L, H = width, length, height
    areas = {
        "floor": W * L,
        "ceiling": W * L,
        "front": W * H,
        "back": W * H,
        "left": L * H,
        "right": L * H,
    }
    V = W * L * H
    S = sum(areas.values())
    A = sum(areas[s] * absorption(materials[s], freqs) for s in SURFACES)
    a_mean = np.clip(A / S, 1e-4, 0.999)
    m = air_db_per_m / (10.0 * np.log10(np.e))  # intensity attenuation, 1/m
    rt_sab = 0.161 * V / (A + 4.0 * m * V)
    rt_ey = 0.161 * V / (-S * np.log(1.0 - a_mean) + 4.0 * m * V)
    a_tot = (A + 4.0 * m * V) / S
    a_tot = np.clip(a_tot, 1e-4, 0.999)
    R = S * a_tot / (1.0 - a_tot)
    dc = 0.141 * np.sqrt(R)
    return RoomStats(V, S, a_mean, rt_sab, rt_ey, R, dc)
