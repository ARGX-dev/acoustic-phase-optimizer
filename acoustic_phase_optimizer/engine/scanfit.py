"""Fit the acoustic box to a 3D scan of a room.

The browser samples points (and, for meshes, surface normals) evenly over the scanned surfaces and sends
them here. We find the floor and ceiling, the direction the walls run, and the extent of the room in that
direction, ignoring stray furniture and scan noise by using percentiles rather than min/max.

Input points are already in metres and z-up (the browser applies the unit and up-axis choices).
The result offers the four ways to place the stage wall; each maps scan coordinates to room coordinates:

    room = R(yaw) @ (scan - origin)   with R a rotation about z, then x across, y from the stage, z up
"""

from __future__ import annotations

import numpy as np

SIDES = ("-y", "+x", "+y", "-x")  # in the fitted frame: which wall becomes the stage


def _dominant_angle_from_normals(normals: np.ndarray) -> float | None:
    """Wall direction (radians, modulo 90 degrees) from horizontal-ish surface normals."""
    if normals is None or len(normals) < 50:
        return None
    n = normals / np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-9)
    horiz = np.abs(n[:, 2]) < 0.35
    if horiz.sum() < 50:
        return None
    th = np.arctan2(n[horiz, 1], n[horiz, 0])
    w = np.hypot(n[horiz, 0], n[horiz, 1])
    # walls at right angles share one direction modulo 90 degrees: average on the 4x circle
    c, s = (w * np.cos(4 * th)).sum(), (w * np.sin(4 * th)).sum()
    if np.hypot(c, s) < 0.15 * w.sum():  # no clear dominant direction (round room, noise)
        return None
    return float(np.arctan2(s, c) / 4)


def _dominant_angle_by_area(xy: np.ndarray, lo: float, hi: float) -> float:
    """Fallback for bare point clouds: the rotation with the smallest percentile footprint."""
    best, best_area = 0.0, np.inf
    for deg in np.arange(0.0, 90.0, 0.5):
        a = np.radians(deg)
        r = xy @ np.array([[np.cos(a), -np.sin(a)], [np.sin(a), np.cos(a)]])  # rotates the points by -a
        ext = np.percentile(r, hi, axis=0) - np.percentile(r, lo, axis=0)
        area = ext[0] * ext[1]
        if area < best_area:
            best, best_area = a, area
    return best  # walls run at +best; fit_box rotates by -angle to square them up


def fit_box(points, normals=None, lo_pct: float = 1.0, hi_pct: float = 99.0) -> dict:
    p = np.asarray(points, dtype=float).reshape(-1, 3)
    p = p[np.all(np.isfinite(p), axis=1)]
    if len(p) < 100:
        raise ValueError("The scan has too few points to measure a room.")
    nrm = None if normals is None else np.asarray(normals, dtype=float).reshape(-1, 3)
    if nrm is not None and len(nrm) != len(p):  # normals must pair up with points; otherwise ignore them
        nrm = None

    floor = float(np.percentile(p[:, 2], lo_pct))
    ceiling = float(np.percentile(p[:, 2], hi_pct))
    height = ceiling - floor
    if height < 1.5:
        raise ValueError(f"The scan is only {height:.1f} m tall. Check the units and which way is up.")

    # walls: the middle of the height band, away from floor clutter and ceiling fixtures
    band = (p[:, 2] > floor + 0.15 * height) & (p[:, 2] < ceiling - 0.1 * height)
    wall_pts = p[band] if band.sum() > 200 else p
    angle = _dominant_angle_from_normals(None if nrm is None else nrm[band] if band.sum() > 200 else nrm)
    method = "surfaces"
    if angle is None:
        angle = _dominant_angle_by_area(wall_pts[:, :2], lo_pct, hi_pct)
        method = "footprint"

    c, s = np.cos(-angle), np.sin(-angle)
    rot = np.array([[c, -s], [s, c]])  # rotate scan by -angle so walls run along the axes
    xy = wall_pts[:, :2] @ rot.T
    x0, y0 = np.percentile(xy, lo_pct, axis=0)
    x1, y1 = np.percentile(xy, hi_pct, axis=0)
    ex, ey = float(x1 - x0), float(y1 - y0)

    options = {}
    for side in SIDES:
        # extra turn so the chosen wall lies along room y = 0 and the room extends into +y
        turn = {"-y": 0.0, "+x": -np.pi / 2, "+y": np.pi, "-x": np.pi / 2}[side]
        yaw = -angle + turn
        width, length = (ex, ey) if side in ("-y", "+y") else (ey, ex)
        ct, st = np.cos(turn), np.sin(turn)
        corners = np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]]) @ np.array([[ct, -st], [st, ct]]).T
        mn = corners.min(axis=0)
        options[side] = {
            "room": {"width": round(width, 2), "length": round(length, 2), "height": round(height, 2)},
            # room = Rz(yaw) @ scan + t
            "yaw_deg": round(float(np.degrees(yaw)), 3),
            "t": [round(float(-mn[0]), 4), round(float(-mn[1]), 4), round(-floor, 4)],
        }
    return {
        "height": round(height, 2), "extent": [round(ex, 2), round(ey, 2)], "angle_deg": round(float(np.degrees(angle)), 2),
        "method": method, "points": int(len(p)), "options": options,
        "suggested": "-y" if ey >= ex else "-x",  # the stage usually sits on a short wall
    }


def apply(option: dict, pts: np.ndarray) -> np.ndarray:
    """Map scan points (z-up, metres) into room coordinates with one fitted option."""
    a = np.radians(option["yaw_deg"])
    R = np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]])
    return pts @ R.T + np.asarray(option["t"])
