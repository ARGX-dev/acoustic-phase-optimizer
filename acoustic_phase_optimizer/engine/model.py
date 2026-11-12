"""Project data model, shared by the engine, the API and saved project files.

Coordinate system (metres): x = across the room (0..width), y = from the stage
wall (y = 0) toward the back (0..length), z = up (0..height).
Speaker aim: yaw 0 deg points down +y (toward the audience), positive yaw turns
toward +x; pitch negative tilts down.
"""

from __future__ import annotations

import math
import uuid
from typing import Literal

from pydantic import BaseModel, Field, model_validator


def _id() -> str:
    return uuid.uuid4().hex[:8]


MaterialId = str
Role = Literal["main", "fill", "front_fill", "delay", "sub"]
FilterType = Literal["bell", "low_shelf", "high_shelf", "highpass", "lowpass"]


class Materials(BaseModel):
    floor: MaterialId = "wood_floor"
    ceiling: MaterialId = "gypsum"
    front: MaterialId = "concrete"
    back: MaterialId = "concrete"
    left: MaterialId = "concrete"
    right: MaterialId = "concrete"


SurfaceRole = Literal["floor", "ceiling", "front", "back", "left", "right", "other"]


class Surface(BaseModel):
    """One flat surface (or the group of small parts) of a scanned room."""

    name: str = Field("Surface", max_length=60)
    role: SurfaceRole = "other"  # which wall of the box it stands for; sets its material when none is chosen
    material: MaterialId | None = None  # None: the room's material for its role (side walls for small parts)
    area: float = Field(0.0, ge=0)


class RoomShape(BaseModel):
    """The scanned room's real shape for the sound prediction (engine/geometry.py:build_shape).

    Triangles in room coordinates, each tagged with the surface it belongs to. Made from the scan as it was
    placed; `source` records that placement so the app can tell when the scan has moved since.
    """

    source: str = Field("", max_length=200)
    vertices: list[float] = Field(max_length=3 * 3 * 4000)
    triangles: list[int] = Field(max_length=3 * 4000)
    surface: list[int] = Field(max_length=4000)
    surfaces: list[Surface] = Field(min_length=1, max_length=160)

    @model_validator(mode="after")
    def _consistent(self) -> RoomShape:
        nv = len(self.vertices)
        if nv % 3 or len(self.triangles) % 3 or len(self.surface) * 3 != len(self.triangles) or not self.surface:
            raise ValueError("room shape: vertices, triangles and surface tags do not match")
        if min(self.triangles) < 0 or max(self.triangles) >= nv // 3:
            raise ValueError("room shape: a triangle refers to a corner that does not exist")
        if min(self.surface) < 0 or max(self.surface) >= len(self.surfaces):
            raise ValueError("room shape: a triangle refers to a surface that does not exist")
        if not all(math.isfinite(v) and abs(v) < 1e5 for v in self.vertices):
            raise ValueError("room shape: corner coordinates must be finite")
        return self


class Room(BaseModel):
    # indoor: every surface reflects; stadium: no roof; outdoor: only the ground reflects
    kind: Literal["indoor", "stadium", "outdoor"] = "indoor"
    width: float = Field(20.0, gt=1, le=200)
    length: float = Field(30.0, gt=1, le=300)
    height: float = Field(8.0, gt=1.5, le=60)
    materials: Materials = Field(default_factory=Materials)
    # box: the fitted shoebox (fast); shape: the scanned room's real surfaces (needs `shape`)
    model: Literal["box", "shape"] = "box"
    shape: RoomShape | None = None


class Environment(BaseModel):
    temperature_c: float = Field(20.0, ge=-20, le=50)
    humidity_pct: float = Field(50.0, ge=0, le=100)
    pressure_kpa: float = Field(101.325, ge=60, le=110)


class CustomMaterial(BaseModel):
    """A material the user typed in: absorption at 125, 250, 500, 1k, 2k and 4k Hz."""

    id: str = Field(default_factory=lambda: "custom_" + _id(), pattern=r"^custom_[0-9a-z_]{1,40}$")
    name: str = Field("My material", max_length=60)
    alpha: tuple[float, float, float, float, float, float] = (0.1, 0.1, 0.1, 0.1, 0.1, 0.1)

    @model_validator(mode="after")
    def _clip(self):
        self.alpha = tuple(min(1.0, max(0.0, float(a))) for a in self.alpha)
        return self


class Audience(BaseModel):
    x0: float = 1.0
    y0: float = 4.0
    x1: float = 19.0
    y1: float = 28.0
    ear_height: float = Field(1.2, ge=0.3, le=20)  # at the front row
    rake: float = Field(0.0, ge=0, le=40)  # raked seating: how much higher the back row's ears are, m
    occupancy: float = Field(0.0, ge=0, le=1)  # share of seats filled; blends audience absorption into the floor

    def ear_at(self, y: float) -> float:
        """Ear height at distance y from the stage: rising evenly from the front row to the back row."""
        span = self.y1 - self.y0
        t = 0.0 if span <= 0 else min(1.0, max(0.0, (y - self.y0) / span))
        return self.ear_height + self.rake * t


class PEQBand(BaseModel):
    id: str = Field(default_factory=_id)
    type: FilterType = "bell"
    freq: float = Field(1000.0, ge=10, le=24000)
    gain_db: float = Field(0.0, ge=-24, le=24)
    q: float = Field(1.0, ge=0.1, le=30)
    enabled: bool = True


class Speaker(BaseModel):
    id: str = Field(default_factory=_id)
    name: str = "Speaker"
    role: Role = "main"
    model: str = "generic_12_2way"
    # Placement
    x: float = 10.0
    y: float = 1.0
    z: float = 4.0
    yaw: float = 0.0
    pitch: float = -10.0
    # Acoustic model
    coverage_h: float = Field(90.0, ge=5, le=360)
    coverage_v: float = Field(60.0, ge=5, le=360)
    level_db: float = Field(128.0, ge=60, le=160)  # max SPL @ 1 m
    hp_hz: float = Field(60.0, ge=0, le=2000)  # 0 = no high-pass
    lp_hz: float = Field(0.0, ge=0, le=24000)  # 0 = no low-pass
    directivity_hz: float = Field(900.0, ge=50, le=10000)  # pattern control above this
    array_boxes: int = Field(0, ge=0, le=32)  # line-array hang: number of boxes (0 = not an array)
    array_splay: float = Field(0.0, ge=0, le=30)  # degrees between boxes
    array_box_h: float = Field(0.0, ge=0, le=1.5)  # height of one box, m (0 = not known: a typical 0.3 m)
    array_box_curve: float = Field(0.0, ge=0, le=60)  # a constant-curvature box's own vertical angle, degrees
    # DSP (what the optimizer adjusts)
    delay_ms: float = Field(0.0, ge=0, le=500)
    gain_db: float = Field(0.0, ge=-40, le=20)
    invert: bool = False
    mute: bool = False
    peq: list[PEQBand] = Field(default_factory=list)
    # Optimizer locks
    lock_delay: bool = False
    lock_gain: bool = False
    # Sub arrays (cardioid, end-fire, arc): boxes sharing a group are processed as one unit by the optimizer,
    # and each keeps its own pattern offset - a delay and polarity on top of the unit's - that nothing alters.
    array_group: str | None = Field(None, max_length=40)
    array_delay_ms: float = Field(0.0, ge=0, le=100)
    array_invert: bool = False
    color: int | None = Field(None, ge=0, le=7)  # chart colour slot; follows the speaker, never its rank


class Mic(BaseModel):
    id: str = Field(default_factory=_id)
    name: str = "Mic"
    x: float = 10.0
    y: float = 15.0
    z: float = 1.2


class MeasuredBand(BaseModel):
    """ISO 3382 values of one octave band of a measured impulse response (None: not measurable)."""

    f: float
    edt: float | None = None
    t20: float | None = None
    t30: float | None = None
    c50: float | None = None
    c80: float | None = None
    d50: float | None = None
    ts: float | None = None


class Measurement(BaseModel):
    """A measured response shown over the prediction: imported (REW / Smaart export, an impulse response
    WAV) or measured live (a sweep, an RTA snapshot). Display only: it never changes the prediction."""

    id: str = Field(default_factory=_id)
    name: str = Field("Measurement", max_length=80)
    freqs: list[float] = Field(..., min_length=8, max_length=2000)
    spl: list[float] = Field(..., min_length=8, max_length=2000)
    phase: list[float | None] | None = Field(None, max_length=2000)
    offset_db: float = Field(0.0, ge=-80, le=80)
    color: int = Field(5, ge=0, le=7)
    source: Literal["file", "ir", "sweep", "rta"] = "file"
    # the impulse response, kept as a WAV in the data folder (and in a project .trky); start_ms is when its
    # first sample happened, counted from when the sound left the output
    ir_id: str | None = Field(None, pattern=r"^[0-9a-f]{16,64}$")
    start_ms: float = 0.0
    arrival_ms: float | None = None  # the direct sound; absolute only when loopback is set
    loopback: bool = False
    polarity: Literal[-1, 1] | None = None
    bands: list[MeasuredBand] | None = Field(None, max_length=8)
    mic_id: str | None = None  # the measurement position it was taken at
    speaker_ids: list[str] = Field(default_factory=list, max_length=64)  # what was playing
    taken: str | None = Field(None, max_length=40)  # ISO date and time

    @model_validator(mode="after")
    def _same_length(self) -> Measurement:
        if len(self.spl) != len(self.freqs) or (self.phase is not None and len(self.phase) != len(self.freqs)):
            raise ValueError("frequency, level and phase lists must be the same length")
        return self


class ReportInfo(BaseModel):
    """What the venue's report says about the job; the company's own details are app-wide (Branding)."""

    client: str = Field("", max_length=120)
    reference: str = Field("", max_length=60)  # job or quote number


class Branding(BaseModel):
    """The user's company on report covers. Kept in the data folder, not in projects, and checked here because
    the values go into the report's HTML and CSS."""

    company: str = Field("", max_length=120)
    engineer: str = Field("", max_length=120)
    contact: str = Field("", max_length=200)
    accent: str = Field("#256abf", pattern=r"^#[0-9a-fA-F]{6}$")
    logo: str | None = Field(None, max_length=300_000, pattern=r"^data:image/(png|jpeg);base64,[A-Za-z0-9+/]+={0,2}$")


class Settings(BaseModel):
    grid_resolution: float = Field(0.5, ge=0.1, le=5)
    reflection_order: int = Field(2, ge=0, le=4)
    f_min: float = Field(31.5, ge=10, le=500)
    f_max: float = Field(16000.0, ge=1000, le=22000)
    precedence_ms: float = Field(0.0, ge=0, le=30)  # extra delay on fills so the image stays on stage
    target_tilt_db_per_oct: float = Field(-1.0, ge=-6, le=3)
    link_symmetric: bool = True  # mirror-image pairs always receive identical DSP
    weighting: Literal["Z", "A", "C"] = "Z"  # frequency weighting of broadband levels (maps, section, average)


class ScanRef(BaseModel):
    """An imported 3D scan, shown in the 3D view. The file lives in the data folder; only this reference travels.

    To place it: convert to z-up metres (up axis, unit scale), then room = Rz(yaw_deg) @ p + t.
    The prediction uses the fitted box (Room) unless Room.model is "shape", which uses Room.shape, made
    from this scan.
    """

    id: str = Field(pattern=r"^[0-9a-f]{16,64}$")
    name: str = "scan"
    ext: Literal["glb", "obj", "ply", "mvr", "3ds", "dae", "fbx", "stl", "usdz", "usd", "usda", "usdc"]
    up: Literal["y", "z"] = "y"
    unit_scale: float = Field(1.0, gt=0, le=10)
    yaw_deg: float = 0.0
    t: tuple[float, float, float] = (0.0, 0.0, 0.0)
    visible: bool = True
    opacity: float = Field(1.0, ge=0.05, le=1)
    repair: bool = True  # fix flipped faces, small holes and broken triangles on load (web/js/meshfix.js); the file is kept as-is


class Project(BaseModel):
    version: int = 2
    name: str = "Untitled venue"
    room: Room = Field(default_factory=Room)
    environment: Environment = Field(default_factory=Environment)
    audience: Audience = Field(default_factory=Audience)
    speakers: list[Speaker] = Field(default_factory=list)
    mics: list[Mic] = Field(default_factory=list)
    settings: Settings = Field(default_factory=Settings)
    notes: str = ""
    scan: ScanRef | None = None
    custom_materials: list[CustomMaterial] = Field(default_factory=list, max_length=40)
    measurements: list[Measurement] = Field(default_factory=list, max_length=8)
    report: ReportInfo = Field(default_factory=ReportInfo)

    @model_validator(mode="after")
    def _hang_defaults(self):
        """Line-array speakers saved before hangs had a box count get the library default (level and coverage kept)."""
        from .library import BY_ID

        for s in self.speakers:
            t = BY_ID.get(s.model)
            if t and t["spec"]["kind"] == "element":
                if s.array_boxes == 0:
                    s.array_boxes, s.array_splay = t["array_boxes"], t["array_splay"]
                if s.array_box_h == 0:  # saved before boxes had a height
                    s.array_box_h, s.array_box_curve = t["array_box_h"], t["array_box_curve"]
        return self

    @model_validator(mode="after")
    def _clamp_audience(self) -> Project:
        a, r = self.audience, self.room
        x0, x1 = sorted((a.x0, a.x1))
        y0, y1 = sorted((a.y0, a.y1))
        a.x0, a.x1 = max(0.0, x0), min(r.width, x1)
        a.y0, a.y1 = max(0.0, y0), min(r.length, y1)
        if a.x1 - a.x0 < 0.2:
            a.x0, a.x1 = 0.0, r.width
        if a.y1 - a.y0 < 0.2:
            a.y0, a.y1 = 0.0, r.length
        a.ear_height = min(a.ear_height, r.height - 0.1)
        a.rake = max(0.0, min(a.rake, r.height - 0.1 - a.ear_height))  # the back row stays under the ceiling
        return self

    @model_validator(mode="after")
    def _assign_colors(self) -> Project:
        used = {s.color for s in self.speakers if s.color is not None}
        for s in self.speakers:
            if s.color is None:
                free = [c for c in range(8) if c not in used]
                s.color = free[0] if free else len(used) % 8
                used.add(s.color)
        return self

    @model_validator(mode="after")
    def _unique_ids(self) -> Project:
        for items in (self.speakers, self.mics):
            seen: set[str] = set()
            for it in items:
                if it.id in seen:
                    it.id = _id()
                seen.add(it.id)
        return self
