from __future__ import annotations

import re
import struct
import numpy as np
from typing import Optional, Tuple
from dataclasses import dataclass

from acoustic_phase_optimizer.acoustic.room_model import RoomModel, RoomDimensions
from acoustic_phase_optimizer.utils.logging import get_logger

logger = get_logger(__name__)


@dataclass
class LidarScan:
    points: np.ndarray
    colors: Optional[np.ndarray] = None
    normals: Optional[np.ndarray] = None


def import_lidar(path: str) -> Optional[LidarScan]:
    path_lower = path.lower()
    if path_lower.endswith(".ply"):
        return _read_ply(path)
    elif path_lower.endswith(".pcd"):
        return _read_pcd(path)
    elif path_lower.endswith(".las") or path_lower.endswith(".laz"):
        return _read_las(path)
    else:
        logger.error(f"Unsupported LIDAR format: {path}")
        return None


def _read_ply(path: str) -> Optional[LidarScan]:
    try:
        with open(path, "rb") as f:
            header = b""
            while True:
                line = f.readline()
                if not line:
                    return None
                header += line
                if line.strip() == b"end_header":
                    break

            header_str = header.decode("ascii", errors="replace")

            vertex_count = 0
            fmt = "ascii"
            has_color = False
            for line in header_str.split("\n"):
                if line.startswith("element vertex"):
                    vertex_count = int(line.split()[-1])
                elif line.startswith("property uchar") or line.startswith("property uint8"):
                    if "red" in line or "green" in line or "blue" in line:
                        has_color = True
                elif line.startswith("format"):
                    fmt = line.split()[1]

            if vertex_count == 0:
                return None

            if fmt == "ascii":
                data = f.read().decode("ascii", errors="replace")
                rows = []
                for line in data.strip().split("\n"):
                    parts = line.strip().split()
                    if len(parts) >= 3:
                        rows.append([float(parts[0]), float(parts[1]), float(parts[2])])
                points = np.array(rows, dtype=np.float64)
            else:
                points = np.frombuffer(f.read(vertex_count * 3 * 4), dtype=np.float32).reshape(-1, 3).astype(np.float64)

            logger.info(f"Loaded PLY: {len(points)} points")
            return LidarScan(points=points)
    except Exception as e:
        logger.error(f"Failed to read PLY: {e}")
        return None


def _read_pcd(path: str) -> Optional[LidarScan]:
    try:
        with open(path, "rb") as f:
            header_lines = []
            while True:
                line = f.readline().decode("ascii", errors="replace")
                header_lines.append(line)
                if line.startswith("DATA"):
                    fmt = line.strip().split()[-1].lower()
                    break

            header = "".join(header_lines)
            point_count = 0
            for line in header_lines:
                if line.startswith("POINTS"):
                    point_count = int(line.split()[-1])

            if point_count == 0:
                return None

            if fmt == "ascii":
                data = f.read().decode("ascii", errors="replace")
                rows = []
                for line in data.strip().split("\n"):
                    parts = line.strip().split()
                    if len(parts) >= 3:
                        rows.append([float(parts[0]), float(parts[1]), float(parts[2])])
                points = np.array(rows, dtype=np.float64)
            else:
                points = np.frombuffer(f.read(point_count * 4 * 4), dtype=np.float32).reshape(-1, 4).astype(np.float64)[:, :3]

            logger.info(f"Loaded PCD: {len(points)} points")
            return LidarScan(points=points)
    except Exception as e:
        logger.error(f"Failed to read PCD: {e}")
        return None


def _read_las(path: str) -> Optional[LidarScan]:
    try:
        import laspy
        las = laspy.read(path)
        points = np.vstack([las.x, las.y, las.z]).T.astype(np.float64)
        logger.info(f"Loaded LAS: {len(points)} points")
        return LidarScan(points=points)
    except ImportError:
        logger.error("laspy not installed. Install with: pip install laspy")
        return None
    except Exception as e:
        logger.error(f"Failed to read LAS: {e}")
        return None


def fit_room_from_points(scan: LidarScan) -> Optional[RoomModel]:
    pts = scan.points
    if len(pts) < 100:
        logger.error("Too few points for room fitting")
        return None

    z_vals = pts[:, 2]
    floor_z = np.percentile(z_vals, 5)
    ceil_z = np.percentile(z_vals, 95)

    floor_mask = pts[:, 2] < floor_z + (ceil_z - floor_z) * 0.1
    ceil_mask = pts[:, 2] > ceil_z - (ceil_z - floor_z) * 0.1

    wall_pts = pts[~(floor_mask | ceil_mask)]
    if len(wall_pts) < 50:
        wall_pts = pts

    x_min, x_max = np.percentile(wall_pts[:, 0], [2, 98])
    y_min, y_max = np.percentile(wall_pts[:, 1], [2, 98])

    length = float(x_max - x_min)
    width = float(y_max - y_min)
    height = float(ceil_z - floor_z)

    if length < 1 or width < 1 or height < 1:
        logger.error(f"Invalid room dimensions: {length:.1f}x{width:.1f}x{height:.1f}")
        return None

    dims = RoomDimensions(length=length, width=width, height=height)
    room = RoomModel(dimensions=dims)
    logger.info(f"Fitted room: {length:.1f} x {width:.1f} x {height:.1f}m")
    return room
