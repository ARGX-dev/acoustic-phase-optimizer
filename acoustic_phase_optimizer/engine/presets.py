"""Generic loudspeaker types and demo venues (no brand data; typical published values)."""

from __future__ import annotations

from .model import Audience, Environment, Materials, Mic, Project, Room, Settings, Speaker

SPEAKER_TYPES: dict[str, dict] = {
    "generic_12_2way": dict(label='12" 2-way point source (90x60)', coverage_h=90, coverage_v=60,
                            level_db=129, hp_hz=60, lp_hz=0, directivity_hz=1100, role="main"),
    "generic_15_2way": dict(label='15" 2-way point source (60x40)', coverage_h=60, coverage_v=40,
                            level_db=133, hp_hz=50, lp_hz=0, directivity_hz=800, role="main"),
    "line_array_hang": dict(label="Line array hang, 8 elements (100x20)", coverage_h=100, coverage_v=20,
                            level_db=140, hp_hz=55, lp_hz=0, directivity_hz=400, role="main"),
    "column": dict(label="Steerable column (150x15)", coverage_h=150, coverage_v=15,
                   level_db=120, hp_hz=120, lp_hz=0, directivity_hz=500, role="main"),
    "compact_8": dict(label='8" compact fill (100x100)', coverage_h=100, coverage_v=100,
                      level_db=118, hp_hz=90, lp_hz=0, directivity_hz=1800, role="fill"),
    "frontfill_5": dict(label='5" front fill (110x110)', coverage_h=110, coverage_v=110,
                        level_db=112, hp_hz=120, lp_hz=0, directivity_hz=2500, role="front_fill"),
    "sub_18": dict(label='18" subwoofer', coverage_h=360, coverage_v=360,
                   level_db=134, hp_hz=30, lp_hz=100, directivity_hz=10000, role="sub"),
    "sub_2x18": dict(label='2x18" subwoofer', coverage_h=360, coverage_v=360,
                     level_db=139, hp_hz=28, lp_hz=100, directivity_hz=10000, role="sub"),
}


def make_speaker(model: str, name: str, x: float, y: float, z: float, yaw: float = 0.0,
                 pitch: float = 0.0, **kw) -> Speaker:
    if model in SPEAKER_TYPES:
        t = {k: v for k, v in SPEAKER_TYPES[model].items() if k != "label"}
    else:  # a real model from the library
        from .library import BY_ID

        src = BY_ID[model]
        t = {k: src[k] for k in ("coverage_h", "coverage_v", "level_db", "hp_hz", "lp_hz", "directivity_hz", "role",
                                 "array_boxes", "array_splay") if k in src}
    t.update(kw)
    return Speaker(model=model, name=name, x=x, y=y, z=z, yaw=yaw, pitch=pitch, **t)


def gymnasium() -> Project:
    return Project(
        name="High school gymnasium",
        room=Room(width=30, length=40, height=9,
                  materials=Materials(floor="wood_floor", ceiling="steel_deck", front="concrete_block",
                                      back="concrete_block", left="bleachers", right="concrete_block")),
        environment=Environment(temperature_c=22, humidity_pct=45),
        audience=Audience(x0=3, y0=6, x1=27, y1=37, ear_height=1.2),
        speakers=[
            make_speaker("generic_15_2way", "Main L", 9, 1.5, 6.5, yaw=8, pitch=-12),
            make_speaker("generic_15_2way", "Main R", 21, 1.5, 6.5, yaw=-8, pitch=-12),
            make_speaker("compact_8", "Delay L", 9, 22, 6.0, yaw=0, pitch=-25, role="delay", delay_ms=0),
            make_speaker("compact_8", "Delay R", 21, 22, 6.0, yaw=0, pitch=-25, role="delay", delay_ms=0),
            make_speaker("sub_18", "Sub L", 11, 1.0, 0.5, gain_db=0),
            make_speaker("sub_18", "Sub R", 19, 1.0, 0.5, gain_db=0),
        ],
        mics=[Mic(name="FOH", x=15, y=20, z=1.2), Mic(name="Front", x=12, y=8, z=1.2),
              Mic(name="Rear", x=18, y=34, z=1.2)],
        settings=Settings(reflection_order=2, grid_resolution=0.6),
    )


def theater() -> Project:
    return Project(
        name="Proscenium theater",
        room=Room(width=22, length=28, height=12,
                  materials=Materials(floor="seats_empty", ceiling="plaster", front="drape",
                                      back="wood_panel", left="wood_panel", right="wood_panel")),
        audience=Audience(x0=2, y0=5, x1=20, y1=27, ear_height=1.1),
        speakers=[
            make_speaker("line_array_hang", "Hang L", 4, 2.0, 8.5, yaw=10, pitch=-14),
            make_speaker("line_array_hang", "Hang R", 18, 2.0, 8.5, yaw=-10, pitch=-14),
            make_speaker("frontfill_5", "Front fill L", 8, 3.6, 1.0, yaw=0, pitch=5),
            make_speaker("frontfill_5", "Front fill R", 14, 3.6, 1.0, yaw=0, pitch=5),
            make_speaker("sub_2x18", "Sub C", 11, 2.0, 0.5),
        ],
        mics=[Mic(name="Row C", x=9, y=7, z=1.1), Mic(name="FOH", x=11, y=17, z=1.1),
              Mic(name="Back row", x=14, y=26, z=1.1)],
    )


def worship() -> Project:
    return Project(
        name="House of worship",
        room=Room(width=18, length=36, height=11,
                  materials=Materials(floor="vinyl_floor", ceiling="wood_panel", front="brick",
                                      back="brick", left="glass", right="glass")),
        audience=Audience(x0=1.5, y0=7, x1=16.5, y1=34, ear_height=1.1),
        speakers=[
            make_speaker("column", "Column L", 2.5, 5.0, 3.5, yaw=15, pitch=-4),
            make_speaker("column", "Column R", 15.5, 5.0, 3.5, yaw=-15, pitch=-4),
            make_speaker("compact_8", "Delay ring L", 2, 20, 4.0, yaw=50, pitch=-15, role="delay"),
            make_speaker("compact_8", "Delay ring R", 16, 20, 4.0, yaw=-50, pitch=-15, role="delay"),
        ],
        mics=[Mic(name="Pew 3", x=6, y=10, z=1.1), Mic(name="Centre aisle", x=9, y=22, z=1.1),
              Mic(name="Rear", x=12, y=32, z=1.1)],
    )


def conference() -> Project:
    return Project(
        name="Conference room",
        room=Room(width=12, length=16, height=3.2,
                  materials=Materials(floor="carpet", ceiling="acoustic_tile", front="gypsum",
                                      back="gypsum", left="glass", right="gypsum")),
        audience=Audience(x0=1, y0=3, x1=11, y1=15, ear_height=1.2),
        speakers=[
            make_speaker("compact_8", "Front L", 3, 0.5, 2.6, yaw=10, pitch=-15, role="main"),
            make_speaker("compact_8", "Front R", 9, 0.5, 2.6, yaw=-10, pitch=-15, role="main"),
            make_speaker("frontfill_5", "Ceiling rear L", 4, 11, 3.1, pitch=-89, role="delay", coverage_h=120, coverage_v=120),
            make_speaker("frontfill_5", "Ceiling rear R", 8, 11, 3.1, pitch=-89, role="delay", coverage_h=120, coverage_v=120),
        ],
        mics=[Mic(name="Table", x=6, y=8, z=1.2)],
        settings=Settings(reflection_order=3, grid_resolution=0.3),
    )


def festival() -> Project:
    return Project(
        name="Festival field",
        room=Room(kind="outdoor", width=60, length=90, height=20,
                  materials=Materials(floor="grass")),
        environment=Environment(temperature_c=24, humidity_pct=35),
        audience=Audience(x0=4, y0=8, x1=56, y1=86, ear_height=1.7),
        speakers=[
            make_speaker("lacoustics_k2", "Main L", 18, 2.0, 13, yaw=6, pitch=-4),
            make_speaker("lacoustics_k2", "Main R", 42, 2.0, 13, yaw=-6, pitch=-4),
            make_speaker("lacoustics_ks28", "Sub L", 26, 1.5, 0.6),
            make_speaker("lacoustics_ks28", "Sub R", 34, 1.5, 0.6),
            make_speaker("lacoustics_kara2", "Delay L", 16, 48, 10, yaw=0, pitch=-6, role="delay"),
            make_speaker("lacoustics_kara2", "Delay R", 44, 48, 10, yaw=0, pitch=-6, role="delay"),
        ],
        mics=[Mic(name="FOH", x=30, y=40, z=1.7), Mic(name="Front", x=24, y=12, z=1.7),
              Mic(name="Behind delays", x=36, y=70, z=1.7)],
        settings=Settings(reflection_order=1, grid_resolution=1.5),
    )


def stadium() -> Project:
    return Project(
        name="Stadium",
        room=Room(kind="stadium", width=90, length=120, height=35,
                  materials=Materials(floor="grass", front="concrete", back="seats_empty",
                                      left="seats_empty", right="seats_empty")),
        environment=Environment(temperature_c=18, humidity_pct=60),
        audience=Audience(x0=8, y0=12, x1=82, y1=110, ear_height=1.7),
        speakers=[
            make_speaker("db_gsl8", "Main L", 30, 3, 20, yaw=8, pitch=-5),
            make_speaker("db_gsl8", "Main R", 60, 3, 20, yaw=-8, pitch=-5),
            make_speaker("db_sl_sub", "Sub L", 40, 2, 0.6),
            make_speaker("db_sl_sub", "Sub R", 50, 2, 0.6),
            make_speaker("db_ksl8", "Delay L", 25, 60, 15, pitch=-6, role="delay"),
            make_speaker("db_ksl8", "Delay R", 65, 60, 15, pitch=-6, role="delay"),
        ],
        mics=[Mic(name="FOH", x=45, y=50, z=1.7), Mic(name="Pitch rear", x=45, y=100, z=1.7)],
        settings=Settings(reflection_order=1, grid_resolution=2.0),
    )


VENUES = {
    "gymnasium": ("High school gymnasium", gymnasium),
    "theater": ("Proscenium theater", theater),
    "worship": ("House of worship", worship),
    "conference": ("Conference room", conference),
    "festival": ("Festival field (open air)", festival),
    "stadium": ("Stadium (open roof)", stadium),
}


def blank() -> Project:
    return Project(name="New venue", speakers=[
        make_speaker("generic_12_2way", "Main L", 6, 1, 5, yaw=5, pitch=-10),
        make_speaker("generic_12_2way", "Main R", 14, 1, 5, yaw=-5, pitch=-10),
    ], mics=[Mic(name="FOH", x=10, y=18, z=1.2)])
