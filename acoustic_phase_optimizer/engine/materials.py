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
        Material("tile", "Ceramic tile / marble", (0.01, 0.01, 0.01, 0.01, 0.02, 0.02)),
        Material("carpet_thin", "Thin carpet on concrete", (0.02, 0.06, 0.14, 0.37, 0.60, 0.65)),
        Material("curtain_light", "Light curtain, hung flat", (0.03, 0.04, 0.11, 0.17, 0.24, 0.35)),
        Material("window", "Window glass, ordinary", (0.35, 0.25, 0.18, 0.12, 0.07, 0.04)),
        Material("plywood_thin", "Plywood panel over air gap", (0.42, 0.21, 0.10, 0.08, 0.06, 0.06)),
        Material("mineral_wool", "Mineral wool 100 mm, faced", (0.65, 0.95, 0.99, 0.99, 0.99, 0.99)),
        Material("bass_trap", "Membrane bass absorber", (0.50, 0.35, 0.20, 0.10, 0.05, 0.05)),
        Material("orchestra", "Orchestra with instruments", (0.27, 0.53, 0.67, 0.93, 0.87, 0.80)),
        Material("grass", "Grass / earth (outdoor)", (0.11, 0.26, 0.60, 0.69, 0.92, 0.99)),
        Material("open", "Open (fully absorptive)", (1.0, 1.0, 1.0, 1.0, 1.0, 1.0)),
    ]
}

SURFACES = ("floor", "ceiling", "front", "back", "left", "right")


def _interp(alpha, freqs: np.ndarray) -> np.ndarray:
    a = np.interp(np.log2(np.asarray(freqs, float)), np.log2(OCTAVE_BANDS), np.asarray(alpha, float))
    return np.clip(a, 0.005, 0.999)


def absorption(material_id: str, freqs: np.ndarray, custom: dict | None = None) -> np.ndarray:
    """Absorption coefficient interpolated on a log-frequency axis, clamped at the ends."""
    if custom and material_id in custom:
        return _interp(custom[material_id], freqs)
    mat = MATERIALS.get(material_id) or MATERIALS["concrete"]
    return _interp(mat.alpha, freqs)


def reflection_amplitude(material_id: str, freqs: np.ndarray) -> np.ndarray:
    """Pressure reflection magnitude sqrt(1 - alpha)."""
    return np.sqrt(1.0 - absorption(material_id, freqs))


def surface_alphas(project, freqs: np.ndarray, occupancy: bool = True) -> dict[str, np.ndarray]:
    """Absorption per surface for a project: its own materials first, then the built-in list.

    Audience occupancy blends the seated-audience coefficients into the floor over the audience area, so a
    half-full hall sits between its empty-seat and full-house values.
    """
    custom = {m.id: m.alpha for m in getattr(project, "custom_materials", [])}
    mats = project.room.materials.model_dump()
    kind = getattr(project.room, "kind", "indoor")
    if kind in ("stadium", "outdoor"):
        mats["ceiling"] = "open"
    if kind == "outdoor":
        for s in ("front", "back", "left", "right"):
            mats[s] = "open"
    out = {s: absorption(mats[s], freqs, custom) for s in SURFACES}
    occ = getattr(project.audience, "occupancy", 0.0)
    if occ > 0 and occupancy:
        r, a = project.room, project.audience
        frac = max(0.0, min(1.0, (a.x1 - a.x0) * (a.y1 - a.y0) / (r.width * r.length)))
        full = absorption("audience", freqs)
        out["floor"] = np.clip(out["floor"] + frac * occ * np.maximum(full - out["floor"], 0.0), 0.005, 0.999)
    return out


def shape_alphas(project, freqs: np.ndarray, base: dict[str, np.ndarray] | None = None) -> np.ndarray:
    """Absorption per surface of the scanned shape, (surfaces, F).

    A surface without its own material takes the material of its role in the room's materials (furniture and small
    parts take the side walls'). Audience occupancy blends into the floor surfaces as for the box.
    """
    shape = project.room.shape
    custom = {m.id: m.alpha for m in getattr(project, "custom_materials", [])}
    base = base if base is not None else surface_alphas(project, freqs, occupancy=False)
    rows = []
    for s in shape.surfaces:
        if s.material:
            rows.append(absorption(s.material, freqs, custom))
        else:
            rows.append(base["left" if s.role == "other" else s.role])
    out = np.array(rows).reshape(len(rows), len(freqs))
    occ = getattr(project.audience, "occupancy", 0.0)
    floors = [i for i, s in enumerate(shape.surfaces) if s.role == "floor"]
    if occ > 0 and floors:
        a = project.audience
        floor_area = sum(shape.surfaces[i].area for i in floors) or 1.0
        frac = max(0.0, min(1.0, (a.x1 - a.x0) * (a.y1 - a.y0) / floor_area))
        full = absorption("audience", freqs)
        out[floors] = np.clip(out[floors] + frac * occ * np.maximum(full - out[floors], 0.0), 0.005, 0.999)
    return out


@dataclass
class RoomStats:
    volume: float
    surface: float
    mean_alpha: np.ndarray  # per frequency
    rt60_sabine: np.ndarray
    rt60_eyring: np.ndarray
    room_constant: np.ndarray  # R = S*a / (1 - a), m^2
    critical_distance: np.ndarray  # for Q = 1, m
    rt60_rays: np.ndarray | None = None  # ray-traced, scanned shape only; NaN where the decay could not be read
    escaped: float = 0.0  # share of traced rays that left the scanned shape through an opening

    @property
    def rt60(self) -> np.ndarray:
        """The best reverberation time available: ray-traced where it could be read, Eyring elsewhere."""
        if self.rt60_rays is None:
            return self.rt60_eyring
        return np.where(np.isfinite(self.rt60_rays), self.rt60_rays, self.rt60_eyring)


def stats_from(V: float, S: float, A: np.ndarray, air_db_per_m: np.ndarray, rt60_rays: np.ndarray | None = None,
               escaped: float = 0.0) -> RoomStats:
    """Sabine/Eyring statistics from volume, surface and total absorption area A (m^2 per frequency).

    With a ray-traced RT60 the room constant follows the traced decay, so the diffuse level agrees with it.
    """
    a_mean = np.clip(A / S, 1e-4, 0.999)
    m = air_db_per_m / (10.0 * np.log10(np.e))  # intensity attenuation, 1/m
    rt_sab = 0.161 * V / (A + 4.0 * m * V)
    rt_ey = 0.161 * V / (-S * np.log(1.0 - a_mean) + 4.0 * m * V)
    a_tot = (A + 4.0 * m * V) / S
    if rt60_rays is not None:
        good = np.isfinite(rt60_rays) & (rt60_rays > 0)
        a_ray = 1.0 - np.exp(-0.161 * V / (S * np.where(good, rt60_rays, 1.0)))  # Eyring inverted
        a_tot = np.where(good, a_ray, a_tot)
    a_tot = np.clip(a_tot, 1e-4, 0.999)
    R = S * a_tot / (1.0 - a_tot)
    dc = 0.141 * np.sqrt(R)
    return RoomStats(V, S, a_mean, rt_sab, rt_ey, R, dc, rt60_rays, escaped)


def room_stats(
    width: float,
    length: float,
    height: float,
    alphas: dict[str, np.ndarray],
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
    A = sum(areas[s] * alphas[s] for s in SURFACES)
    return stats_from(W * L * H, sum(areas.values()), A, air_db_per_m)
