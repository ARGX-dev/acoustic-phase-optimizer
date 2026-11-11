"""Fitting the acoustic box to a scanned room."""

from __future__ import annotations

import numpy as np
import pytest

from acoustic_phase_optimizer.engine.scanfit import apply, fit_box


def scanned_room(w=14.0, l=22.0, h=6.5, yaw_deg=27.0, origin=(3.0, -5.0, 1.2), n=30000, seed=1, normals=True):
    """Points on the inside surfaces of a box, rotated and shifted like a raw scan, with noise and clutter."""
    rng = np.random.default_rng(seed)
    faces = [  # (axis fixed, value, normal)
        (0, 0.0, (1, 0, 0)), (0, w, (-1, 0, 0)), (1, 0.0, (0, 1, 0)), (1, l, (0, -1, 0)),
        (2, 0.0, (0, 0, 1)), (2, h, (0, 0, -1))]
    dims = np.array([w, l, h])
    areas = np.array([np.prod(np.delete(dims, ax)) for ax, _, _ in faces])
    pick = rng.choice(len(faces), size=n, p=areas / areas.sum())
    pts = rng.random((n, 3)) * dims
    nrm = np.zeros((n, 3))
    for i, (ax, v, nv) in enumerate(faces):
        m = pick == i
        pts[m, ax] = v
        nrm[m] = nv
    pts += rng.normal(0, 0.02, pts.shape)
    clutter = rng.random((n // 50, 3)) * [w, l, 1.0]  # chairs and people near the floor
    pts = np.vstack([pts, clutter])
    nrm = np.vstack([nrm, rng.normal(size=clutter.shape)])
    a = np.radians(yaw_deg)
    R = np.array([[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]])
    return pts @ R.T + origin, (nrm @ R.T if normals else None)


@pytest.mark.parametrize("use_normals", [True, False])
def test_fit_recovers_size_and_orientation(use_normals):
    pts, nrm = scanned_room(normals=use_normals)
    fit = fit_box(pts, nrm)
    assert fit["method"] == ("surfaces" if use_normals else "footprint")
    opt = fit["options"]["-y"] if fit["extent"][1] >= fit["extent"][0] else fit["options"]["-x"]
    room = opt["room"]
    assert sorted([room["width"], room["length"]]) == pytest.approx([14.0, 22.0], abs=0.35)
    assert room["height"] == pytest.approx(6.5, abs=0.15)


def test_every_option_puts_the_room_in_the_positive_box():
    pts, nrm = scanned_room()
    fit = fit_box(pts, nrm)
    for side, opt in fit["options"].items():
        r = apply(opt, pts)
        room = opt["room"]
        for k, ax in (("width", 0), ("length", 1), ("height", 2)):
            lo, hi = np.percentile(r[:, ax], [1, 99])
            assert lo == pytest.approx(0, abs=0.3), (side, k)
            assert hi == pytest.approx(room[k], abs=0.3), (side, k)


def test_stage_choice_picks_the_right_wall():
    # unrotated room, 10 wide (x) by 30 long (y): the -y wall is at scan y = 0, the +y wall at y = 30
    pts, nrm = scanned_room(w=10, l=30, h=5, yaw_deg=0, origin=(0, 0, 0))
    fit = fit_box(pts, nrm)
    near_minus_y = np.array([[5.0, 0.5, 2.0]])
    near_plus_y = np.array([[5.0, 29.5, 2.0]])
    assert apply(fit["options"]["-y"], near_minus_y)[0, 1] < 1.0
    assert apply(fit["options"]["+y"], near_plus_y)[0, 1] < 1.0
    near_plus_x = np.array([[9.5, 15.0, 2.0]])
    near_minus_x = np.array([[0.5, 15.0, 2.0]])
    assert apply(fit["options"]["+x"], near_plus_x)[0, 1] < 1.0
    assert apply(fit["options"]["-x"], near_minus_x)[0, 1] < 1.0
    assert fit["suggested"] == "-y"


def test_flat_or_tiny_scans_are_rejected_with_a_reason():
    with pytest.raises(ValueError, match="too few"):
        fit_box(np.zeros((10, 3)))
    flat = np.random.default_rng(0).random((5000, 3)) * [10, 10, 0.5]
    with pytest.raises(ValueError, match="units"):
        fit_box(flat)
