"""Scanned-shape acoustics: surface merging, image sources on real polygons, the ray trace, and agreement with the box."""

from __future__ import annotations

import numpy as np
import pytest
from pydantic import ValidationError

from acoustic_phase_optimizer.engine import geometry as g
from acoustic_phase_optimizer.engine.field import clear_cache, make_context, speaker_field
from acoustic_phase_optimizer.engine.model import Audience, Materials, Project, Room, RoomShape, Settings, Speaker
from acoustic_phase_optimizer.engine.simulate import point_response, room_report, simulate

from .roomshapes import box_mesh, grid_quad

W, L, H = 12.0, 20.0, 5.0
MATS = Materials(floor="wood_floor", ceiling="gypsum", front="brick", back="drape", left="plaster", right="glass")


@pytest.fixture(autouse=True)
def _fresh_cache():
    clear_cache()
    yield


def shape_of(V, T, **kw) -> RoomShape:
    sh = g.build_shape(V, T, (W, L, H), **kw)
    return RoomShape(**{k: v for k, v in sh.items() if k != "stats"})


def project(model="box", shape=None, order=2, materials=MATS) -> Project:
    spk = Speaker(name="S", x=4.0, y=1.5, z=3.0, pitch=-10, coverage_h=90, coverage_v=60, level_db=120,
                  hp_hz=0, lp_hz=0, directivity_hz=1000)
    return Project(room=Room(width=W, length=L, height=H, materials=materials, model=model, shape=shape),
                   audience=Audience(x0=1, y0=3, x1=11, y1=19, ear_height=1.2),
                   speakers=[spk], settings=Settings(reflection_order=order, grid_resolution=1.0))


@pytest.fixture(scope="module")
def box_shape() -> RoomShape:
    V, T = box_mesh(W, L, H, step=0.5)
    return shape_of(V, T)


# ------------------------------------------------------------------ surfaces

def test_a_box_mesh_merges_into_its_six_walls():
    for noise in (0.0, 0.01):
        V, T = box_mesh(W, L, H, step=0.25, noise=noise)
        sh = g.build_shape(V, T, (W, L, H))
        big = {s["role"]: s["area"] for s in sh["surfaces"] if s["area"] > 10}
        assert set(big) == {"floor", "ceiling", "front", "back", "left", "right"}
        assert big["floor"] == pytest.approx(W * L, rel=0.02)
        assert big["left"] == pytest.approx(L * H, rel=0.02)
        assert big["front"] == pytest.approx(W * H, rel=0.03)
        assert sh["stats"]["triangles"] <= g.MAX_TRIS


def test_surface_names_pick_materials_by_keyword():
    assert g.material_for("Window_Glass_02") == "window"
    assert g.material_for("Glazing panel") == "glass"
    assert g.material_for("Concrete wall") == "concrete"
    assert g.material_for("Mesh 17") is None


def test_quadric_clustering_keeps_corners_sharp(box_shape):
    m = g.mesh_room(box_shape)
    planes = {box_shape.surfaces[p.surface].role: (p.n, p.d) for p in m.plates}
    n, d = planes["right"]
    assert abs(abs(n[0]) - 1) < 1e-3 and abs(abs(d) - W) < 0.02
    n, d = planes["left"]
    assert abs(d) < 0.02


def test_room_shape_rejects_bad_indices():
    with pytest.raises(ValidationError):
        RoomShape(vertices=[0, 0, 0, 1, 0, 0, 0, 1, 0], triangles=[0, 1, 5], surface=[0],
                  surfaces=[{"name": "x", "role": "floor"}])


# ------------------------------------------------------------- image sources

def test_box_as_a_mesh_reproduces_the_shoebox_field(box_shape):
    pts = np.random.default_rng(3).uniform([0.5, 2, 0.8], [11.5, 19.5, 2.0], (400, 3))
    box, shp = project(), project("shape", box_shape)
    cb, cs = make_context(box), make_context(shp)
    assert cs.mesh is not None and cb.mesh is None
    fb, fs = speaker_field(box.speakers[0], pts, cb), speaker_field(shp.speakers[0], pts, cs)
    db = 10 * np.log10(np.abs(fs.H) ** 2 / np.abs(fb.H) ** 2)
    assert np.median(np.abs(db)) < 0.2
    assert np.percentile(np.abs(db), 95) < 1.0
    late = 10 * np.log10(fs.L.sum(axis=1) / fb.L.sum(axis=1))
    assert np.percentile(np.abs(late), 90) < 1.0
    # the diffuse tail comes from the ray trace, the box's from Eyring: close, not equal
    assert np.median(np.abs(10 * np.log10(fs.E / fb.E))) < 1.5


def test_a_partition_blocks_the_paths_behind_it():
    V, T = box_mesh(W, L, H, step=1.0)
    pv, pt = grid_quad((0, 10, 0), (8, 0, 0), (0, 0, 4), 8, 4)  # a wall across most of the room at y = 10
    V2, T2 = np.vstack([V, pv]), np.vstack([T, pt + len(V)])
    m = g.mesh_room(shape_of(V2, T2))
    src = np.array([4.0, 2.0, 2.0])
    behind = np.array([[4.0, 15.0, 2.0]])
    beside = np.array([[11.5, 12.0, 2.0]])  # passes the wall's open end
    assert m.blocked(src, behind)[0]
    assert not m.blocked(src, beside)[0]
    ims = g.mesh_images(src, m, 2)
    reach = sum(bool(g.path_valid(im, behind, m)[0]) for im in ims if im.order)
    free = g.mesh_room(shape_of(V, T))
    reach_free = sum(bool(g.path_valid(im, behind, free)[0]) for im in g.mesh_images(src, free, 2) if im.order)
    assert reach < reach_free


# ----------------------------------------------------------------- ray trace

def test_ray_volume_and_rt60_agree_with_theory(box_shape):
    m = g.mesh_room(box_shape)
    p = g.trace(m, np.array([6.0, 10.0, 2.5]))
    assert p.escaped < 0.02
    assert p.mfp == pytest.approx(4 * W * L * H / (2 * (W * L + W * H + L * H)), rel=0.05)
    a = np.array([0.06, 0.1, 0.2])
    rt = g.decay_rt60(p, np.tile(a, (len(m.roles), 1)), np.zeros(3), 343.0)
    S, V = 2 * (W * L + W * H + L * H), W * L * H
    eyring = 0.161 * V / (-S * np.log(1 - a))
    assert np.all(np.abs(rt / eyring - 1) < 0.15)


def test_an_open_side_lets_rays_escape():
    V, T = box_mesh(W, L, H, step=1.0)
    keep = ~np.all(V[T][:, :, 2] > H - 1e-6, axis=1)  # no ceiling
    m = g.mesh_room(shape_of(V, T[keep]))
    assert g.trace(m, np.array([6.0, 10.0, 2.5])).escaped > 0.3


def test_editing_a_material_reuses_the_trace(box_shape):
    p = project("shape", box_shape)
    r1 = room_report(p)
    m = g.mesh_room(box_shape)
    traces = dict(m._paths)
    p2 = p.model_copy(deep=True)
    for s in p2.room.shape.surfaces:
        if s.role == "back":
            s.material = "concrete"
    r2 = room_report(p2)
    assert r1["model"] == "shape" and r1["rt60_rays"] is not None
    assert r2["rt60_mid"] > r1["rt60_mid"]  # a hard back wall instead of drapes
    assert g.mesh_room(p2.room.shape)._paths.keys() == traces.keys()


def test_shape_mode_simulates_end_to_end(box_shape):
    p = project("shape", box_shape)
    out = simulate(p, preview=True)
    assert np.all(np.isfinite(np.asarray(out["maps"]["total"], float)))
    assert np.isfinite(out["metrics"]["sti"])
    pr = point_response(p, 6.0, 12.0)
    assert any(ev["order"] == 2 for ev in pr["arrivals"])
