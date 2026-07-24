"""3D room model for acoustic simulation and visualization."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray
from dataclasses import dataclass, field
from typing import List, Optional, Tuple
from acoustic_phase_optimizer.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class Surface:
    vertices: NDArray[np.float64]
    absorption_coefficient: float = 0.1
    scattering_coefficient: float = 0.1
    material: str = "default"
    normal: Optional[NDArray[np.float64]] = None

    def __post_init__(self):
        if self.normal is None and len(self.vertices) >= 3:
            v1 = self.vertices[1] - self.vertices[0]
            v2 = self.vertices[2] - self.vertices[0]
            self.normal = np.cross(v1, v2)
            norm = np.linalg.norm(self.normal)
            if norm > 0:
                self.normal = self.normal / norm


@dataclass
class RoomDimensions:
    length: float = 30.0
    width: float = 20.0
    height: float = 8.0


class RoomModel:
    """3D room model with surfaces, materials, and acoustic properties."""

    def __init__(
        self,
        dimensions: Optional[RoomDimensions] = None,
        speed_of_sound: float = 343.0,
    ):
        self.dimensions = dimensions or RoomDimensions()
        self.speed_of_sound = speed_of_sound
        self.surfaces: List[Surface] = []
        self._build_default_room()

    def _build_default_room(self) -> None:
        L, W, H = self.dimensions.length, self.dimensions.width, self.dimensions.height

        floor = Surface(
            vertices=np.array([[-L/2, -W/2, 0], [L/2, -W/2, 0], [L/2, W/2, 0], [-L/2, W/2, 0]]),
            absorption_coefficient=0.4, material="stage_floor",
        )
        ceiling = Surface(
            vertices=np.array([[-L/2, -W/2, H], [L/2, -W/2, H], [L/2, W/2, H], [-L/2, W/2, H]]),
            absorption_coefficient=0.3, material="acoustic_tile",
        )
        front_wall = Surface(
            vertices=np.array([[-L/2, -W/2, 0], [-L/2, W/2, 0], [-L/2, W/2, H], [-L/2, -W/2, H]]),
            absorption_coefficient=0.05, material="concrete",
        )
        rear_wall = Surface(
            vertices=np.array([[L/2, -W/2, 0], [L/2, W/2, 0], [L/2, W/2, H], [L/2, -W/2, H]]),
            absorption_coefficient=0.2, material="treated_wall",
        )
        left_wall = Surface(
            vertices=np.array([[-L/2, -W/2, 0], [L/2, -W/2, 0], [L/2, -W/2, H], [-L/2, -W/2, H]]),
            absorption_coefficient=0.1, material="drywall",
        )
        right_wall = Surface(
            vertices=np.array([[-L/2, W/2, 0], [L/2, W/2, 0], [L/2, W/2, H], [-L/2, W/2, H]]),
            absorption_coefficient=0.1, material="drywall",
        )

        self.surfaces = [floor, ceiling, front_wall, rear_wall, left_wall, right_wall]

    def add_surface(self, surface: Surface) -> None:
        self.surfaces.append(surface)

    def set_dimensions(self, length: float, width: float, height: float) -> None:
        self.dimensions = RoomDimensions(length, width, height)
        self.surfaces.clear()
        self._build_default_room()

    def ray_intersect(
        self,
        origin: NDArray[np.float64],
        direction: NDArray[np.float64],
    ) -> List[Tuple[Surface, float, NDArray[np.float64]]]:
        intersections = []
        for surface in self.surfaces:
            point, distance = self._intersect_triangle_mesh(origin, direction, surface)
            if point is not None:
                intersections.append((surface, distance, point))
        intersections.sort(key=lambda x: x[1])
        return intersections

    def _intersect_triangle_mesh(
        self,
        origin: NDArray[np.float64],
        direction: NDArray[np.float64],
        surface: Surface,
    ) -> Tuple[Optional[NDArray[np.float64]], Optional[float]]:
        verts = surface.vertices
        for i in range(len(verts) - 2):
            v0, v1, v2 = verts[0], verts[i + 1], verts[i + 2]
            point, distance = self._ray_triangle_intersect(origin, direction, v0, v1, v2)
            if point is not None:
                return point, distance
        return None, None

    @staticmethod
    def _ray_triangle_intersect(
        origin: NDArray[np.float64],
        direction: NDArray[np.float64],
        v0: NDArray[np.float64],
        v1: NDArray[np.float64],
        v2: NDArray[np.float64],
    ) -> Tuple[Optional[NDArray[np.float64]], Optional[float]]:
        epsilon = 1e-12
        edge1 = v1 - v0
        edge2 = v2 - v0
        h = np.cross(direction, edge2)
        a = np.dot(edge1, h)

        if -epsilon < a < epsilon:
            return None, None

        f = 1.0 / a
        s = origin - v0
        u = f * np.dot(s, h)

        if u < 0.0 or u > 1.0:
            return None, None

        q = np.cross(s, edge1)
        v = f * np.dot(direction, q)

        if v < 0.0 or u + v > 1.0:
            return None, None

        t = f * np.dot(edge2, q)
        if t > epsilon:
            point = origin + t * direction
            return point, t

        return None, None

    def get_dimensions_array(self) -> Tuple[float, float, float]:
        return (self.dimensions.length, self.dimensions.width, self.dimensions.height)

    def get_volume(self) -> float:
        return self.dimensions.length * self.dimensions.width * self.dimensions.height

    def get_surface_area(self) -> float:
        area = 0.0
        for surface in self.surfaces:
            verts = surface.vertices
            if len(verts) >= 3:
                v0, v1, v2 = verts[0], verts[1], verts[2 % len(verts)]
                area += 0.5 * np.linalg.norm(np.cross(v1 - v0, v2 - v0))
        return area

    def average_absorption(self) -> float:
        if not self.surfaces:
            return 0.0
        return float(np.mean([s.absorption_coefficient for s in self.surfaces]))

    def estimate_rt60_sabine(self) -> float:
        volume = self.get_volume()
        area = self.get_surface_area()
        alpha = self.average_absorption()
        if alpha <= 0 or area <= 0:
            return 0.0
        return 0.161 * volume / (area * alpha)

    def estimate_rt60_eyring(self) -> float:
        volume = self.get_volume()
        area = self.get_surface_area()
        alpha = self.average_absorption()
        if alpha <= 0 or area <= 0:
            return 0.0
        return 0.161 * volume / (-area * np.log(1.0 - alpha + 1e-12))

    def schroeder_frequency(self) -> float:
        rt60 = self.estimate_rt60_sabine()
        volume = self.get_volume()
        if rt60 <= 0 or volume <= 0:
            return 1000.0
        return 2000.0 * np.sqrt(rt60 / volume)
