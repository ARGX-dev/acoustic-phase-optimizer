"""Physics and optimizer tests for the engine."""

from __future__ import annotations

import json

import numpy as np
import pytest

from acoustic_phase_optimizer.engine import presets
from acoustic_phase_optimizer.engine.air import air_absorption_db_per_m, speed_of_sound
from acoustic_phase_optimizer.engine.eq import bell_db, biquad_coeffs, biquad_response, peq_response
from acoustic_phase_optimizer.engine.field import (
    clear_cache,
    combine,
    directivity,
    directivity_factor,
    dsp_weights,
    freq_grid,
    get_field,
    image_sources,
)
from acoustic_phase_optimizer.engine.materials import MATERIALS, OCTAVE_BANDS, absorption, room_stats
from acoustic_phase_optimizer.engine.metrics import penalty, smooth, spl
from acoustic_phase_optimizer.engine.model import Audience, Materials, Mic, PEQBand, Project, Room, Settings, Speaker
from acoustic_phase_optimizer.engine.optimize import (
    analyse_project,
    auto_eq,
    deep_optimize,
    diff_speakers,
    evaluate,
    link_units,
    mains_of,
    quick_align,
)
from acoustic_phase_optimizer.engine.simulate import point_response, room_report, simulate

OPEN = Materials(floor="open", ceiling="open", front="open", back="open", left="open", right="open")


def omni(name="S", x=10.0, y=2.0, z=2.0, **kw) -> Speaker:
    base = dict(coverage_h=360, coverage_v=360, level_db=120, hp_hz=0, lp_hz=0, directivity_hz=10000)
    base.update(kw)
    return Speaker(name=name, x=x, y=y, z=z, pitch=0, **base)


def free_field(speakers, mics=(), **settings) -> Project:
    st = dict(reflection_order=0, grid_resolution=2.0)
    st.update(settings)
    return Project(room=Room(width=20, length=40, height=10, materials=OPEN),
                   audience=Audience(x0=2, y0=6, x1=18, y1=38, ear_height=2.0),
                   speakers=list(speakers), mics=list(mics), settings=Settings(**st))


@pytest.fixture(autouse=True)
def _fresh_cache():
    clear_cache()
    yield


# ----------------------------------------------------------------------- air

def test_speed_of_sound_reference_values():
    assert speed_of_sound(20, 0) == pytest.approx(343.2, abs=0.3)
    assert speed_of_sound(0, 0) == pytest.approx(331.3, abs=0.3)
    assert 0.3 < speed_of_sound(20, 50) - speed_of_sound(20, 0) < 1.0  # humid air is faster
    assert speed_of_sound(35, 50) > speed_of_sound(5, 50)


def test_air_absorption_matches_iso_9613_table():
    # ISO 9613-2 table 2, 20 degC / 70 % RH, dB per km
    a = air_absorption_db_per_m(np.array([1000.0, 4000.0, 8000.0]), 20, 70) * 1000
    assert a == pytest.approx([5.0, 22.9, 76.6], rel=0.08)


# ----------------------------------------------------------------- materials

def test_absorption_interpolates_and_clamps():
    a = absorption("carpet", OCTAVE_BANDS)
    assert a == pytest.approx(MATERIALS["carpet"].alpha)
    assert absorption("carpet", np.array([31.5]))[0] == pytest.approx(MATERIALS["carpet"].alpha[0])
    mid = absorption("carpet", np.array([707.0]))[0]
    assert MATERIALS["carpet"].alpha[2] < mid < MATERIALS["carpet"].alpha[3]


def test_room_stats_sabine_by_hand():
    from acoustic_phase_optimizer.engine.materials import SURFACES, absorption

    mats = {s: absorption("concrete", np.array([1000.0])) for s in SURFACES}
    st = room_stats(10, 20, 5, mats, np.array([1000.0]), np.array([0.0]))
    surface = 2 * (10 * 20 + 10 * 5 + 20 * 5)
    assert st.surface == pytest.approx(surface)
    assert st.rt60_sabine[0] == pytest.approx(0.161 * 1000 / (surface * 0.02), rel=1e-6)
    assert st.rt60_eyring[0] < st.rt60_sabine[0]


def test_absorptive_room_is_drier():
    live = room_report(presets.gymnasium())
    p = presets.gymnasium()
    p.room.materials.ceiling = "acoustic_tile"
    p.room.materials.floor = "carpet"
    dead = room_report(p)
    assert dead["rt60_mid"] < 0.5 * live["rt60_mid"]
    assert dead["critical_distance_m"][3] > live["critical_distance_m"][3]


# ------------------------------------------------------------------------ eq

def test_bell_gain_at_centre_and_far_away():
    f = np.array([20.0, 1000.0, 16000.0])
    g = bell_db(f, 1000.0, -6.0, 2.0)
    assert g[1] == pytest.approx(-6.0, abs=0.01)
    assert abs(g[0]) < 0.05 and abs(g[2]) < 0.3


def test_highpass_is_3db_down_at_cutoff():
    b, a = biquad_coeffs("highpass", 100.0, 0.0, 0.7071)
    h = np.abs(biquad_response(b, a, np.array([100.0, 1000.0])))
    assert 20 * np.log10(h[0]) == pytest.approx(-3.01, abs=0.05)
    assert 20 * np.log10(h[1]) == pytest.approx(0.0, abs=0.01)


def test_disabled_and_zero_gain_bands_are_transparent():
    bands = [PEQBand(freq=500, gain_db=-9, q=2, enabled=False), PEQBand(freq=2000, gain_db=0, q=2)]
    assert np.allclose(peq_response(bands, freq_grid(50, 10000)), 1.0)


# -------------------------------------------------------------------- field

def test_image_source_counts_and_first_order_geometry():
    src = np.array([1.0, 2.0, 3.0])
    dims = (10.0, 20.0, 5.0)
    assert [len(image_sources(src, dims, n)) for n in range(4)] == [1, 7, 25, 63]
    first = {tuple(np.round(im.pos, 6)): im for im in image_sources(src, dims, 1) if im.order == 1}
    assert first[(-1.0, 2.0, 3.0)].hits == {"left": 1}
    assert first[(19.0, 2.0, 3.0)].hits == {"right": 1}
    assert first[(1.0, -2.0, 3.0)].hits == {"front": 1}
    assert first[(1.0, 38.0, 3.0)].hits == {"back": 1}
    assert first[(1.0, 2.0, -3.0)].hits == {"floor": 1}
    assert first[(1.0, 2.0, 7.0)].hits == {"ceiling": 1}
    assert list(first[(-1.0, 2.0, 3.0)].flip) == [-1, 1, 1]


def test_directivity_is_6db_down_at_the_coverage_edge():
    spk = Speaker(coverage_h=90, coverage_v=60, yaw=0, pitch=0, directivity_hz=500)
    f = np.array([16000.0])
    def at(h_deg, v_deg):
        h, v = np.radians(h_deg), np.radians(v_deg)
        d = np.array([[np.sin(h) * np.cos(v), np.cos(h) * np.cos(v), np.sin(v)]])
        return directivity(spk, d, f)[0, 0]
    assert at(0, 0) == pytest.approx(1.0)
    assert at(45, 0) == pytest.approx(0.5, rel=0.02)
    assert at(0, 30) == pytest.approx(0.5, rel=0.02)
    assert at(180, 0) == pytest.approx(10 ** (-24 / 20), rel=0.01)  # rear floor


def test_directivity_relaxes_to_omni_at_low_frequency():
    spk = Speaker(coverage_h=60, coverage_v=40, directivity_hz=1000)
    q = directivity_factor(spk, np.array([20.0, 16000.0]))
    assert q[0] == pytest.approx(1.0, abs=0.05)
    assert q[1] > 8


def test_free_field_level_follows_inverse_square_law():
    mics = [Mic(name="10 m", x=10, y=12, z=2), Mic(name="20 m", x=10, y=22, z=2)]
    p = free_field([omni()], mics)
    f = get_field(p)
    c = combine(f, dsp_weights(p.speakers, f.freqs))
    k = int(np.argmin(np.abs(f.freqs - 250.0)))  # low enough that air absorption is negligible
    near, far = spl(c.early[f.n_grid, k]), spl(c.early[f.n_grid + 1, k])
    assert near == pytest.approx(120 - 20, abs=0.1)
    assert near - far == pytest.approx(6.02, abs=0.1)
    assert np.all(c.late < c.early.max() * 1e-3)  # no room, no reverberant field


def test_two_sources_sum_and_cancel():
    mic = Mic(x=10, y=20, z=2)  # equidistant from both
    a, b = omni("A", x=6), omni("B", x=14)
    p = free_field([a, b], [mic])
    f = get_field(p)
    row = f.n_grid

    def level(speakers):
        return spl(combine(f, dsp_weights(speakers, f.freqs)).early[row])

    single = level([a, b.model_copy(update={"mute": True})])
    assert level([a, b]) - single == pytest.approx(6.02, abs=0.05)
    assert np.all(level([a, b.model_copy(update={"invert": True})]) < single - 40)
    # delaying one source by half a period nulls that frequency and leaves the octave below at +3 dB
    k = int(np.argmin(np.abs(f.freqs - 500.0)))
    delayed = level([a, b.model_copy(update={"delay_ms": 500.0 / f.freqs[k]})])
    assert delayed[k] < single[k] - 30
    k2 = int(np.argmin(np.abs(f.freqs - f.freqs[k] / 2)))
    assert delayed[k2] - single[k2] == pytest.approx(3.01, abs=0.3)


def test_reflections_add_energy_and_late_field_grows_with_liveness():
    dead = free_field([omni()])
    live = dead.model_copy(deep=True)
    live.room.materials = Materials(floor="concrete", ceiling="concrete", front="concrete",
                                    back="concrete", left="concrete", right="concrete")
    live.settings.reflection_order = 2
    fd, fl = get_field(dead), get_field(live)
    cd = combine(fd, dsp_weights(dead.speakers, fd.freqs))
    cl = combine(fl, dsp_weights(live.speakers, fl.freqs))
    assert cl.total.mean() > 2 * cd.total.mean()
    assert cl.late.mean() > 100 * cd.late.mean()


def test_field_cache_reuses_unchanged_speakers():
    p = presets.conference()
    p.settings.grid_resolution = 1.0
    f1 = get_field(p)
    assert get_field(p) is f1
    moved = p.model_copy(deep=True)
    moved.speakers[0].x += 0.5
    f2 = get_field(moved)
    assert f2 is not f1
    assert np.array_equal(f2.H[1], f1.H[1])  # untouched speaker comes straight from the cache
    assert not np.array_equal(f2.H[0], f1.H[0])
    dsp = p.model_copy(deep=True)
    dsp.speakers[0].delay_ms = 12.0  # DSP changes never invalidate the field
    assert get_field(dsp) is f1


def test_preview_grid_is_coarser():
    p = presets.gymnasium()
    assert get_field(p, preview=True).n_grid < get_field(p).n_grid / 2


# ------------------------------------------------------------------ metrics

def test_smoothing_preserves_flat_power_and_reduces_a_notch():
    f = freq_grid(31.5, 16000)
    flat = np.ones(len(f))
    assert np.allclose(smooth(flat, f), 1.0)
    notched = flat.copy()
    notched[50] = 1e-6
    assert smooth(notched, f)[50] > 0.5


# ---------------------------------------------------------------- optimizer

def delay_project() -> Project:
    main_l = Speaker(name="Main L", role="main", x=6, y=1, z=5, pitch=-8, coverage_h=70, coverage_v=50,
                     level_db=130, hp_hz=60, directivity_hz=800)
    main_r = main_l.model_copy(update={"id": "mainr", "name": "Main R", "x": 14})
    delay = Speaker(name="Delay", role="delay", x=10, y=24, z=5, pitch=-35, coverage_h=90, coverage_v=90,
                    level_db=122, hp_hz=90, directivity_hz=900)
    return Project(room=Room(width=20, length=40, height=7),
                   audience=Audience(x0=2, y0=5, x1=18, y1=38, ear_height=1.2),
                   speakers=[main_l, main_r, delay], mics=[Mic(x=10, y=30, z=1.2)],
                   settings=Settings(reflection_order=1, grid_resolution=1.5))


def test_mirror_pairs_are_linked():
    gym = presets.gymnasium()
    names = [[gym.speakers[i].name for i in u] for u in link_units(gym)]
    assert names == [["Main L", "Main R"], ["Delay L", "Delay R"], ["Sub L", "Sub R"]]
    gym.settings.link_symmetric = False
    assert all(len(u) == 1 for u in link_units(gym))
    assert [gym.speakers[i].name for i in mains_of(gym.speakers)] == ["Main L", "Main R"]


def test_quick_align_times_the_delay_speaker():
    p = delay_project()
    f = get_field(p)
    before = evaluate(p, f)
    spks, notes = quick_align(p, f)
    after = evaluate(p, f, spks)
    by = {s.name: s for s in spks}
    # geometric expectation at a seat under the delay's aim point
    seat = np.array([10.0, 27.0, 1.2])
    d_main = np.linalg.norm(seat - [6, 1, 5])
    d_delay = np.linalg.norm(seat - [10, 24, 5])
    expected = (d_main - d_delay) / f.c * 1000.0
    rel = by["Delay"].delay_ms - by["Main L"].delay_ms
    assert rel == pytest.approx(expected, abs=6.0)
    assert by["Main L"].delay_ms == by["Main R"].delay_ms
    assert by["Main L"].gain_db == 0 and not by["Main L"].invert and not by["Delay"].invert
    assert min(s.delay_ms for s in spks) == 0
    assert after["echo_pct"] < before["echo_pct"]
    assert after["score"] > before["score"] + 3
    assert any("time-aligned" in n for n in notes)
    assert {c["field"] for c in diff_speakers(p.speakers, spks)} >= {"delay_ms"}


def test_quick_align_keeps_subs_together():
    p = presets.gymnasium()
    p.settings.grid_resolution = 1.2
    spks, _ = quick_align(p, get_field(p))
    by = {s.name: s for s in spks}
    for a, b in (("Sub L", "Sub R"), ("Delay L", "Delay R"), ("Main L", "Main R")):
        assert (by[a].delay_ms, by[a].gain_db, by[a].invert) == (by[b].delay_ms, by[b].gain_db, by[b].invert)
    assert all(s.gain_db <= 0 for s in spks)  # never asks a box for more than its maximum


def test_locked_speakers_are_never_changed():
    p = delay_project()
    p.speakers[2].lock_delay = True
    p.speakers[2].lock_gain = True
    p.speakers[2].delay_ms = 7.0
    p.speakers[2].gain_db = -4.0
    f = get_field(p)
    for spks in (quick_align(p, f)[0], deep_optimize(p, f, max_generations=4, time_budget_s=2)[0]):
        d = next(s for s in spks if s.name == "Delay")
        assert (d.delay_ms, d.gain_db, d.invert) == (7.0, -4.0, False)


def test_deep_optimize_never_returns_something_worse():
    p = delay_project()
    f = get_field(p)
    seen = []
    spks, info = deep_optimize(p, f, max_generations=10, time_budget_s=5, progress=seen.append)
    assert info["picked"] in {"original", "aligned", "optimized"}
    assert seen and seen[-1].generation == info["generations"]
    start = float(penalty(analyse_project(p, f)))
    aligned = float(penalty(analyse_project(p, f, quick_align(p, f)[0])))
    result = float(penalty(analyse_project(p, f, spks)))
    assert result <= min(start, aligned) + 1e-6
    mains = [s for s in spks if s.role == "main"]
    assert mains[0].delay_ms == mains[1].delay_ms and all(not s.invert for s in spks)


def test_deep_optimize_can_be_cancelled():
    p = delay_project()
    spks, info = deep_optimize(p, get_field(p), cancelled=lambda: True)
    assert info["generations"] == 0 and len(spks) == 3


def test_auto_eq_flattens_a_coloured_system_and_is_idempotent():
    p = delay_project()
    for s in p.speakers[:2]:  # build a 6 dB bump into the mains
        s.peq = [PEQBand(id="user1" + s.id, freq=800, gain_db=6, q=1.5)]
    f = get_field(p)
    before = evaluate(p, f)
    spks, report = auto_eq(p, f, max_bands=4)
    after = evaluate(p, f, spks)
    assert after["tonal"] < before["tonal"] - 0.4
    mains = report[0]
    assert mains["name"] == "Main L + Main R" and mains["rms_after_db"] < mains["rms_before_db"]
    for s in spks:
        auto = [b for b in s.peq if b.id.startswith("auto")]
        assert len(auto) <= 4 and all(-12 <= b.gain_db <= 3 for b in auto)
    assert [b.id for b in spks[0].peq][0].startswith("user1")  # hand-made bands survive
    assert [(b.freq, b.gain_db, b.q) for b in spks[0].peq] == [(b.freq, b.gain_db, b.q) for b in spks[1].peq]
    again, _ = auto_eq(p.model_copy(update={"speakers": spks}), f, max_bands=4)
    assert len(again[0].peq) <= 1 + 4  # auto bands are replaced, not stacked


# ------------------------------------------------------------------ outputs

@pytest.mark.parametrize("name", list(presets.VENUES))
def test_presets_simulate_and_serialise(name):
    p = presets.VENUES[name][1]()
    p.settings.grid_resolution = 1.5
    r = simulate(p)
    json.dumps(r)
    n = r["grid"]["nx"] * r["grid"]["ny"]
    assert all(len(v) == n for v in r["maps"].values())
    assert set(r["maps"]) >= {"direct", "total", "interference", "clarity", "spread", "dominant", "direct_1000"}
    assert 0 <= r["metrics"]["score"] <= 100
    assert len(r["mics"]) == len(p.mics) and len(r["mics"][0]["spl"]) == len(r["freqs"])
    assert max(r["maps"]["dominant"]) < len(p.speakers)
    # the same project round-trips through JSON
    assert Project.model_validate_json(p.model_dump_json()) == p


def test_all_muted_and_empty_projects_do_not_crash():
    p = presets.conference()
    for s in p.speakers:
        s.mute = True
    r = simulate(p)
    assert r["metrics"] is None and r["maps"] == {}
    assert evaluate(p, get_field(p)) == {}
    assert quick_align(p, get_field(p))[0] == p.speakers
    empty = Project(speakers=[], mics=[])
    assert simulate(empty)["metrics"] is None
    json.dumps(point_response(empty, 5, 5))


def test_point_response_reports_arrivals_in_time_order():
    p = delay_project()
    r = point_response(p, 10, 30)
    json.dumps(r)
    assert len(r["freqs"]) == len(r["spl"]) == len(r["phase"]) > 300
    times = [a["t"] for a in r["arrivals"]]
    assert times == sorted(times)
    direct = [a for a in r["arrivals"] if a["order"] == 0]
    assert len(direct) == 3
    d = np.linalg.norm(np.array([10, 30, 1.2]) - [10, 24, 5])
    delay_arrival = next(a for a in direct if a["speaker"] == 2)
    assert delay_arrival["t"] == pytest.approx(d / speed_of_sound(20, 50) * 1000, abs=0.1)
    assert r["first_arrival_ms"] == pytest.approx(min(times), abs=0.01)


def test_audience_is_clamped_into_the_room():
    p = Project(room=Room(width=10, length=12, height=4), audience=Audience(x0=-5, y0=20, x1=50, y1=2, ear_height=9))
    a = p.audience
    assert (a.x0, a.x1, a.y0, a.y1) == (0, 10, 2, 12)
    assert a.ear_height < 4


def test_speaker_library_entries_are_valid_and_sourced():
    from acoustic_phase_optimizer.engine.library import LIBRARY
    from acoustic_phase_optimizer.engine.model import Speaker
    from acoustic_phase_optimizer.engine.presets import SPEAKER_TYPES

    ids = [t["id"] for t in LIBRARY]
    assert len(ids) == len(set(ids)) and not set(ids) & set(SPEAKER_TYPES)
    fields = ("coverage_h", "coverage_v", "level_db", "hp_hz", "lp_hz", "directivity_hz", "role")
    for t in LIBRARY:
        Speaker(model=t["id"], **{k: t[k] for k in fields})
        assert t["spec"]["source"].startswith("https://"), t["id"]
        if t["role"] == "sub":
            assert t["lp_hz"] > 0


def test_occupancy_and_custom_materials_change_the_room():
    from acoustic_phase_optimizer.engine.model import CustomMaterial

    empty = room_report(presets.theater())
    p = presets.theater()
    p.audience.occupancy = 1.0
    full = room_report(p)
    assert full["rt60_mid"] < empty["rt60_mid"]
    p = presets.theater()
    p.custom_materials = [CustomMaterial(id="custom_felt", name="Felt", alpha=(0.9,) * 6)]
    p.room.materials.ceiling = "custom_felt"
    assert room_report(p)["rt60_mid"] < empty["rt60_mid"]
    # unknown ids still fall back instead of failing
    p.room.materials.left = "custom_missing"
    assert room_report(p)["rt60_mid"] > 0


def test_air_loss_grows_with_dry_air():
    p = presets.theater()
    p.environment.humidity_pct = 70
    humid = room_report(p)["air_loss_8k_30m_db"]
    p.environment.humidity_pct = 15
    assert room_report(p)["air_loss_8k_30m_db"] > humid


def test_old_project_files_still_load_and_hangs_get_boxes():
    from acoustic_phase_optimizer.engine.model import Project

    old = presets.theater().model_dump()
    old["room"].pop("kind")
    old["audience"].pop("occupancy")
    old.pop("custom_materials")
    old.pop("scan")
    for s in old["speakers"]:
        s.pop("array_boxes")
        s.pop("array_splay")
    old["speakers"][0]["model"] = "lacoustics_k2"
    p = Project(**old)
    assert p.room.kind == "indoor" and p.audience.occupancy == 0 and p.custom_materials == []
    assert p.speakers[0].array_boxes == 8 and p.speakers[0].array_splay > 0
    assert p.speakers[1].array_boxes in (0, 8)


def test_presets_with_real_hangs_carry_box_counts():
    fest = presets.festival()
    assert fest.speakers[0].array_boxes == 8
    assert presets.stadium().speakers[0].array_boxes == 8


def test_section_matches_plan_at_ear_height():
    from acoustic_phase_optimizer.engine.simulate import section

    p = presets.theater()
    sim = simulate(p)
    g = sim["grid"]
    ix, iy = g["nx"] // 2, g["ny"] // 2
    x = g["x0"] + (g["x1"] - g["x0"]) * ix / (g["nx"] - 1)
    y = g["y0"] + (g["y1"] - g["y0"]) * iy / (g["ny"] - 1)
    plan = sim["maps"]["direct"][iy * g["nx"] + ix]
    sec = section(p, "length", x)
    us = np.linspace(sec["u0"], sec["u1"], sec["nu"])
    zs = np.linspace(sec["z0"], sec["z1"], sec["nz"])
    iu, iz = int(np.argmin(abs(us - y))), int(np.argmin(abs(zs - p.audience.ear_height)))
    assert abs(us[iu] - y) < 0.35 and abs(zs[iz] - p.audience.ear_height) < 0.35
    val = sec["values"][iz * sec["nu"] + iu]
    assert abs(val - plan) < 1.5, (val, plan)
    assert sec["nu"] * sec["nz"] <= 5200
    across = section(p, "width", y, "1000")
    assert across["axis"] == "width" and len(across["values"]) == across["nu"] * across["nz"]


def _lf_level(p, x, y):
    r = point_response(p, x, y, 1.2, ppo=24)
    f = np.array(r["freqs"])
    m = (f > 50) & (f < 80)
    return float(np.mean(np.array(r["spl"])[m]))


def _outdoor_with(speakers):
    p = presets.festival()
    p.speakers = speakers
    return p


def test_cardioid_sub_pair_cancels_behind_and_adds_in_front():
    from acoustic_phase_optimizer.engine import subarray

    p = presets.festival()
    x, y = p.room.width / 2, 12.0
    single = _outdoor_with(subarray.build(p, "line", "sub_18", 1, 1.0, x, y))
    card = _outdoor_with(subarray.build(p, "cardioid", "sub_18", 1, 1.0, x, y, depth=1.0))
    assert [s.array_invert for s in card.speakers] == [False, True]
    assert card.speakers[1].array_delay_ms == pytest.approx(1000 / 343.5, rel=0.03)
    front_gain = _lf_level(card, x, y + 15) - _lf_level(single, x, y + 15)
    rear_gain = _lf_level(card, x, y - 9) - _lf_level(single, x, y - 9)
    assert front_gain > 2.0, front_gain  # the pair still adds up in front at 63 Hz
    assert front_gain - rear_gain > 15.0, (front_gain, rear_gain)  # and leaves a deep null behind


def test_endfire_and_arc_delays():
    from acoustic_phase_optimizer.engine import subarray

    p = presets.festival()
    ef = subarray.build(p, "endfire", "sub_18", 4, 1.0, 30, 14)
    ys = [s.y for s in ef]
    assert ys == sorted(ys) and ef[0].array_delay_ms == 0
    assert ef[-1].array_delay_ms == pytest.approx(3 * 1000 / 343.5, rel=0.03)
    arc = subarray.build(p, "line", "sub_18", 6, 1.2, 30, 3, spread_deg=60)
    d = [s.array_delay_ms for s in arc]
    assert d == pytest.approx(d[::-1]) and d[0] > d[2] > 0  # symmetric, outer boxes delayed most
    assert len({s.array_group for s in arc}) == 1


def test_optimizer_keeps_a_sub_array_pattern():
    from acoustic_phase_optimizer.engine import subarray
    from acoustic_phase_optimizer.engine.optimize import link_units, quick_align

    p = presets.festival()
    p.speakers = [s for s in p.speakers if s.role != "sub"] + subarray.build(p, "cardioid", "sub_18", 2, 2.0, 30, 6)
    subs = [i for i, s in enumerate(p.speakers) if s.role == "sub"]
    assert subs in link_units(p)
    before = [(s.array_delay_ms, s.array_invert) for s in p.speakers if s.role == "sub"]
    from acoustic_phase_optimizer.engine.field import get_field

    spk, _ = quick_align(p, get_field(p))
    subs_after = [s for s in spk if s.role == "sub"]
    assert [(s.array_delay_ms, s.array_invert) for s in subs_after] == before
    assert len({(s.delay_ms, s.invert) for s in subs_after}) == 1  # moved together


def test_weighting_curves_and_maps():
    from acoustic_phase_optimizer.engine.metrics import weighting_db

    f = np.array([31.5, 63, 125, 1000, 4000, 10000])
    a = weighting_db(f, "A")
    assert a == pytest.approx([-39.5, -26.2, -16.1, 0.0, 1.0, -2.5], abs=0.2)
    assert weighting_db(np.array([1000.0, 63.0]), "C") == pytest.approx([0.0, -0.8], abs=0.2)
    p = presets.theater()
    z = simulate(p)
    p.settings.weighting = "A"
    w = simulate(p)
    assert w["weighting"] == "A"
    assert np.mean(w["maps"]["direct"]) < np.mean(z["maps"]["direct"])
    assert w["metrics"]["score"] == z["metrics"]["score"]  # weighting is reporting, not optimisation


def test_measurements_validate():
    from pydantic import ValidationError

    p = presets.theater().model_dump()
    f = [20 * 2 ** (i / 3) for i in range(30)]
    p["measurements"] = [{"name": "FOH", "freqs": f, "spl": [90.0] * 30, "phase": [0.0] * 30}]
    assert Project(**p).measurements[0].name == "FOH"
    p["measurements"][0]["spl"] = [90.0] * 29
    with pytest.raises(ValidationError):
        Project(**p)
