from __future__ import annotations

import os
import re
import struct
import zipfile
import numpy as np
import xml.etree.ElementTree as ET
from typing import Optional, Tuple
from dataclasses import dataclass
from pathlib import Path

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
    readers = _READERS
    ext = _resolve_extension(path_lower, readers)
    if ext is None:
        if path_lower.endswith(".mvr.zip") or path_lower.endswith(".zip"):
            result = _read_zip_auto(path)
            if result is not None:
                return result
        logger.error(f"Unsupported 3D format: {path}")
        return None
    result = readers[ext](path)
    if result is not None:
        logger.info(f"Loaded {ext}: {len(result.points)} points from {path}")
    return result


def _resolve_extension(path_lower: str, readers: dict) -> Optional[str]:
    longest = ""
    for ext in readers:
        if path_lower.endswith(ext) and len(ext) > len(longest):
            longest = ext
    return longest if longest else None


# ---------------------------------------------------------------------------
# PLY
# ---------------------------------------------------------------------------

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
            return LidarScan(points=points)
    except Exception as e:
        logger.error(f"Failed to read PLY: {e}")
        return None


# ---------------------------------------------------------------------------
# PCD
# ---------------------------------------------------------------------------

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
            return LidarScan(points=points)
    except Exception as e:
        logger.error(f"Failed to read PCD: {e}")
        return None


# ---------------------------------------------------------------------------
# LAS/LAZ
# ---------------------------------------------------------------------------

def _read_las(path: str) -> Optional[LidarScan]:
    try:
        import laspy
        las = laspy.read(path)
        points = np.vstack([las.x, las.y, las.z]).T.astype(np.float64)
        return LidarScan(points=points)
    except ImportError:
        logger.error("laspy not installed. Install with: pip install laspy")
        return None
    except Exception as e:
        logger.error(f"Failed to read LAS: {e}")
        return None


# ---------------------------------------------------------------------------
# OBJ (Wavefront)
# ---------------------------------------------------------------------------

def _read_obj(path: str) -> Optional[LidarScan]:
    try:
        vertices = []
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            for line in f:
                line = line.strip()
                if line.startswith("v ") or line.startswith("v\t"):
                    parts = line[1:].strip().split()
                    if len(parts) >= 3:
                        try:
                            vertices.append([float(parts[0]), float(parts[1]), float(parts[2])])
                        except ValueError:
                            continue
        if not vertices:
            logger.error(f"No vertices found in OBJ: {path}")
            return None
        return LidarScan(points=np.array(vertices, dtype=np.float64))
    except Exception as e:
        logger.error(f"Failed to read OBJ: {e}")
        return None


# ---------------------------------------------------------------------------
# STL (binary)
# ---------------------------------------------------------------------------

def _read_stl(path: str) -> Optional[LidarScan]:
    try:
        with open(path, "rb") as f:
            header = f.read(80)
            if len(header) < 80:
                return None
            tri_count = struct.unpack("<I", f.read(4))[0]
            if tri_count == 0 or tri_count > 1e8:
                return None
            points = []
            for _ in range(tri_count):
                data = f.read(50)
                if len(data) < 50:
                    break
                verts = struct.unpack("<12f", data[:48])
                points.append([verts[0], verts[1], verts[2]])
                points.append([verts[3], verts[4], verts[5]])
                points.append([verts[6], verts[7], verts[8]])
        if not points:
            return None
        return LidarScan(points=np.array(points, dtype=np.float64))
    except Exception as e:
        logger.error(f"Failed to read STL: {e}")
        return None


# ---------------------------------------------------------------------------
# XYZ
# ---------------------------------------------------------------------------

def _read_xyz(path: str) -> Optional[LidarScan]:
    try:
        rows = []
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or line.startswith("//"):
                    continue
                parts = line.split()
                if len(parts) >= 3:
                    try:
                        rows.append([float(parts[0]), float(parts[1]), float(parts[2])])
                    except ValueError:
                        continue
        if not rows:
            return None
        return LidarScan(points=np.array(rows, dtype=np.float64))
    except Exception as e:
        logger.error(f"Failed to read XYZ: {e}")
        return None


# ---------------------------------------------------------------------------
# PTS (Leica)
# ---------------------------------------------------------------------------

def _read_pts(path: str) -> Optional[LidarScan]:
    try:
        rows = []
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            lines = f.readlines()
        start = 0
        if lines and lines[0].strip().isdigit():
            start = 1
        for line in lines[start:]:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) >= 3:
                try:
                    rows.append([float(parts[0]), float(parts[1]), float(parts[2])])
                except ValueError:
                    continue
        if not rows:
            return None
        return LidarScan(points=np.array(rows, dtype=np.float64))
    except Exception as e:
        logger.error(f"Failed to read PTS: {e}")
        return None


# ---------------------------------------------------------------------------
# OFF (Object File Format)
# ---------------------------------------------------------------------------

def _read_off(path: str) -> Optional[LidarScan]:
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            content = f.read()
        lines = content.strip().split("\n")
        if not lines or lines[0].strip() != "OFF":
            return None
        header_parts = lines[1].strip().split() if len(lines) > 1 else []
        if len(header_parts) >= 1:
            vcount = int(header_parts[0])
        else:
            return None
        vertices = []
        for line in lines[2:]:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) >= 3:
                try:
                    vertices.append([float(parts[0]), float(parts[1]), float(parts[2])])
                except ValueError:
                    continue
            if len(vertices) >= vcount:
                break
        if not vertices:
            return None
        return LidarScan(points=np.array(vertices, dtype=np.float64))
    except Exception as e:
        logger.error(f"Failed to read OFF: {e}")
        return None


# ---------------------------------------------------------------------------
# MVR (My Virtual Rig)
# ---------------------------------------------------------------------------

def _read_mvr(path: str) -> Optional[LidarScan]:
    try:
        import pymvr
        scene = pymvr.parse(path)
        all_points = []
        for layer in scene.layers:
            for child in layer.child_list:
                try:
                    matrix = child.matrix
                    if child.geometry and child.geometry.mesh:
                        for prim in child.geometry.mesh.primitives:
                            verts = prim.get_vertices()
                            if verts is not None:
                                all_points.append(verts)
                except Exception:
                    continue
        if all_points:
            points = np.vstack(all_points)
            return LidarScan(points=points.astype(np.float64))
    except ImportError:
        pass
    except Exception as e:
        logger.warning(f"pymvr parse failed for {path}: {e}")

    return _read_mvr_fallback(path)


def _read_mvr_fallback(path: str) -> Optional[LidarScan]:
    try:
        with zipfile.ZipFile(path, "r") as z:
            names = z.namelist()
            gb_files = [n for n in names if n.lower().endswith(".glb")]
            gtlf_files = [n for n in names if n.lower().endswith(".gltf")]
            obj_files = [n for n in names if n.lower().endswith(".obj")]
            ds_files = [n for n in names if n.lower().endswith(".3ds")]

            candidates = gb_files + gtlf_files + obj_files + ds_files
            if not candidates:
                scene_xmls = [n for n in names if n.lower() == "generalscenedescription.xml"]
                if not scene_xmls:
                    logger.error(f"No 3D data found in MVR archive: {path}")
                    return None
                tree = ET.parse(z.open(scene_xmls[0]))
                root = tree.getroot()
                ns = re.sub(r"{.*}", "", root.tag) if "}" in root.tag else ""
                refs = set()
                for elem in root.iter():
                    if elem.text and elem.text.strip().endswith((".glb", ".gltf", ".obj", ".3ds")):
                        refs.add(elem.text.strip())
                    file_attr = elem.get("file") or elem.get("File") or ""
                    if file_attr.endswith((".glb", ".gltf", ".obj", ".3ds")):
                        refs.add(file_attr)
                candidates = [r for r in refs if r in names]
                if not candidates:
                    logger.error(f"No mesh files referenced in MVR XML: {path}")
                    return None

            all_points = []
            for cname in candidates:
                ext = Path(cname).suffix.lower()
                tmp = os.path.join(os.path.dirname(path), f"__mvr_extract_{os.path.basename(cname)}")
                try:
                    with z.open(cname) as src, open(tmp, "wb") as dst:
                        dst.write(src.read())
                    if ext == ".glb" or ext == ".gltf":
                        scan = _read_gltf(tmp)
                    elif ext == ".obj":
                        scan = _read_obj(tmp)
                    else:
                        continue
                    if scan is not None:
                        all_points.append(scan.points)
                finally:
                    if os.path.exists(tmp):
                        os.remove(tmp)

            if not all_points:
                return None
            return LidarScan(points=np.vstack(all_points))
    except Exception as e:
        logger.error(f"Failed to read MVR: {e}")
        return None


# ---------------------------------------------------------------------------
# ZIP auto-detect (tries MVR → 3MF → loose mesh files)
# ---------------------------------------------------------------------------

def _read_zip_auto(path: str) -> Optional[LidarScan]:
    with zipfile.ZipFile(path, "r") as z:
        names = z.namelist()
    has_mvr_xml = any(n.lower() == "generalscenedescription.xml" for n in names)
    has_3mf_model = any(n.endswith(".model") for n in names)
    if has_mvr_xml:
        return _read_mvr_fallback(path)
    if has_3mf_model:
        return _read_3mf(path)
    mesh_exts = {".glb", ".gltf", ".obj", ".stl", ".ply", ".pcd", ".off", ".xyz", ".pts"}
    candidates = [n for n in names if Path(n).suffix.lower() in mesh_exts]
    if not candidates:
        return None
    all_points = []
    tmp_dir = os.path.join(os.path.dirname(path), f"__zip_extract_{os.getpid()}")
    os.makedirs(tmp_dir, exist_ok=True)
    try:
        with zipfile.ZipFile(path, "r") as z:
            for cname in candidates:
                ext = Path(cname).suffix.lower()
                tmp = os.path.join(tmp_dir, os.path.basename(cname))
                with z.open(cname) as src, open(tmp, "wb") as dst:
                    dst.write(src.read())
                reader = _READERS.get(ext)
                if reader is None:
                    os.remove(tmp)
                    continue
                result = reader(tmp)
                if result is not None:
                    all_points.append(result.points)
                os.remove(tmp)
        if not all_points:
            return None
        return LidarScan(points=np.vstack(all_points))
    finally:
        if os.path.isdir(tmp_dir):
            for f in os.listdir(tmp_dir):
                try:
                    os.remove(os.path.join(tmp_dir, f))
                except OSError:
                    pass
            try:
                os.rmdir(tmp_dir)
            except OSError:
                pass


# ---------------------------------------------------------------------------
# glTF / GLB
# ---------------------------------------------------------------------------

def _read_gltf(path: str) -> Optional[LidarScan]:
    try:
        import pygltflib
        gltf = pygltflib.GLTF2().load(path)
        all_verts = []
        for mesh in gltf.meshes:
            for prim in mesh.primitives:
                if prim.attributes.POSITION is None:
                    continue
                acc = gltf.accessors[prim.attributes.POSITION]
                bv = gltf.bufferViews[acc.bufferView]
                blob = gltf.binary_blob() if bv.buffer == 0 else None
                if blob is None:
                    continue
                offset = (bv.byteOffset or 0) + (acc.byteOffset or 0)
                count = acc.count
                dtype = _gltf_component_type(acc.componentType)
                if dtype is None:
                    continue
                raw = blob[offset:offset + count * 3 * np.dtype(dtype).itemsize]
                verts = np.frombuffer(raw, dtype=dtype).reshape(-1, 3).astype(np.float64)
                all_verts.append(verts)
        if not all_verts:
            return None
        return LidarScan(points=np.vstack(all_verts))
    except ImportError:
        logger.error("pygltflib not installed. Install with: pip install pygltflib")
        return None
    except Exception as e:
        logger.error(f"Failed to read glTF: {e}")
        return None


def _gltf_component_type(ct: int) -> Optional[type]:
    return {5120: np.int8, 5121: np.uint8, 5122: np.int16,
            5123: np.uint16, 5125: np.uint32, 5126: np.float32}.get(ct)


# ---------------------------------------------------------------------------
# E57
# ---------------------------------------------------------------------------

def _read_e57(path: str) -> Optional[LidarScan]:
    try:
        import pye57
        with pye57.E57(path) as e57:
            data = e57.read_scan_raw(0)
        x = data.get("cartesianX", np.array([]))
        y = data.get("cartesianY", np.array([]))
        z = data.get("cartesianZ", np.array([]))
        if len(x) == 0 or len(y) == 0 or len(z) == 0:
            return None
        points = np.column_stack([x, y, z]).astype(np.float64)
        return LidarScan(points=points)
    except ImportError:
        logger.error("pye57 not installed. Install with: pip install pye57")
        return None
    except Exception as e:
        logger.error(f"Failed to read E57: {e}")
        return None


# ---------------------------------------------------------------------------
# 3MF
# ---------------------------------------------------------------------------

def _read_3mf(path: str) -> Optional[LidarScan]:
    try:
        with zipfile.ZipFile(path, "r") as z:
            names = z.namelist()
            model_files = [n for n in names if n.endswith(".model") and "3d/" in n.lower()]
            if not model_files:
                model_files = [n for n in names if n.lower().endswith(".model")]
            if not model_files:
                return None
            tree = ET.parse(z.open(model_files[0]))
            root = tree.getroot()
            ns = re.sub(r"{.*}", "", root.tag) if "}" in root.tag else ""
            vertices = []
            for mesh in root.iter(f"{{{ns}}}mesh" if ns else "mesh"):
                for vert in mesh.iter(f"{{{ns}}}vertex" if ns else "vertex"):
                    x = float(vert.get("x", 0))
                    y = float(vert.get("y", 0))
                    z = float(vert.get("z", 0))
                    vertices.append([x, y, z])
            if not vertices:
                return None
            return LidarScan(points=np.array(vertices, dtype=np.float64))
    except Exception as e:
        logger.error(f"Failed to read 3MF: {e}")
        return None


# ---------------------------------------------------------------------------
# FBX (limited ASCII subset)
# ---------------------------------------------------------------------------

def _read_fbx(path: str) -> Optional[LidarScan]:
    try:
        with open(path, "rb") as f:
            magic = f.read(5)
        is_binary = magic == b"Kaydara"
        if is_binary:
            logger.warning("Binary FBX not supported (too complex). Try converting to OBJ or glTF.")
            return None
        vertices = []
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            for line in f:
                stripped = line.strip()
                if stripped.startswith("Vertices:"):
                    nums_str = stripped.split(":", 1)[1].strip().strip("*").strip()
                    nums = []
                    for token in nums_str.replace(",", " ").split():
                        try:
                            nums.append(float(token))
                        except ValueError:
                            continue
                    for i in range(0, len(nums) - 2, 3):
                        vertices.append([nums[i], nums[i + 1], nums[i + 2]])
                    break
        if not vertices:
            return None
        return LidarScan(points=np.array(vertices, dtype=np.float64))
    except Exception as e:
        logger.error(f"Failed to read FBX: {e}")
        return None


# ---------------------------------------------------------------------------
# Room fitting
# ---------------------------------------------------------------------------

_READERS = {
    ".ply": _read_ply,
    ".pcd": _read_pcd,
    ".las": _read_las,
    ".laz": _read_las,
    ".obj": _read_obj,
    ".stl": _read_stl,
    ".xyz": _read_xyz,
    ".pts": _read_pts,
    ".off": _read_off,
    ".asc": _read_xyz,
    ".mvr": _read_mvr,
    ".glb": _read_gltf,
    ".gltf": _read_gltf,
    ".e57": _read_e57,
    ".3mf": _read_3mf,
    ".fbx": _read_fbx,
}


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
