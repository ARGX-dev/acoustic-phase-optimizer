"""Real loudspeakers, from manufacturer datasheets (data/speakers.json), plus the user's own models.

Each row holds what the datasheet says: coverage, max SPL and how it was measured, the low-frequency
limit and the level it is quoted at, and where it came from. Rows whose figures could not all be
confirmed carry `approximate: true` and a note saying which figure.

The simulator needs a few numbers datasheets don't give directly. They are derived by the fixed rules
below and shown to the user as estimates:

- high-pass corner: the quoted low limit moved up to an approximate -3 dB point
- pattern-control frequency: from woofer size and horizontal coverage (a wavelength-vs-aperture rule)
- max SPL is normalised to a peak figure: continuous (RMS) ratings get +6 dB, the usual crest allowance
- a line-array box becomes a hang: level + 10*log10(boxes) (a power sum; a curved hang does not add
  fully in phase) and vertical coverage = boxes x splay
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, field_validator

C = 343.0
DEFAULT_BOXES = 8
_TO_3DB = {-10: 1.3, -6: 1.15, -5: 1.12, -3: 1.0, None: 1.2}
_TO_PEAK = {"peak": 0.0, "max": 0.0, "linear peak": 0.0, "calculated peak": 0.0, "continuous": 6.0}


class SpeakerRow(BaseModel):
    """One loudspeaker as a datasheet describes it. Also the format for importing your own."""

    id: str = Field(pattern=r"^[a-z0-9_]{2,60}$")
    brand: str = Field(min_length=1, max_length=60)
    model: str = Field(min_length=1, max_length=80)
    kind: Literal["point", "element", "column", "sub"] = "point"
    woofer_in: float = Field(12, ge=1, le=30)
    coverage_h: float = Field(90, ge=5, le=360)
    coverage_v: float = Field(60, ge=1, le=360)  # for a line-array box: its splay range / fixed angle
    max_spl: float = Field(ge=80, le=160)
    spl_basis: Literal["peak", "max", "linear peak", "calculated peak", "continuous"] = "peak"
    lf_hz: float = Field(ge=10, le=2000)
    lf_db: Literal[-10, -6, -5, -3] | None = None
    hf_hz: float | None = Field(None, ge=40, le=24000)
    pattern_hz: float | None = Field(None, ge=50, le=10000)  # overrides the estimate when known
    box_h_mm: float | None = Field(None, ge=80, le=1500)  # line-array box height (sets the hang's length)
    source: str = Field("", max_length=400)
    note: str = Field("", max_length=400)
    approximate: bool = False
    user: bool = False

    @field_validator("lf_db", mode="before")
    @classmethod
    def _db(cls, v):
        if isinstance(v, str):
            v = v.strip().replace("dB", "").strip()
            return int(float(v)) if v else None
        return int(v) if isinstance(v, float) else v

    @field_validator("approximate", "user", mode="before")
    @classmethod
    def _flag(cls, v):
        return v.strip().lower() in ("1", "true", "yes", "y") if isinstance(v, str) else v

    @field_validator("source")
    @classmethod
    def _url(cls, v: str) -> str:
        if v and not v.startswith(("https://", "http://")):
            raise ValueError("source must be a web address")
        return v


def _pattern_control_hz(row: SpeakerRow) -> float:
    if row.pattern_hz:
        return row.pattern_hz
    if row.kind == "sub":
        return 10000.0
    if row.kind == "column":
        return 500.0
    if row.kind == "element":
        return 400.0
    aperture = row.woofer_in * 0.0254
    f = 0.6 * C / (aperture * math.sin(math.radians(min(row.coverage_h, 170)) / 2))
    return float(round(min(3000.0, max(300.0, f)), -1))


def constant_curvature(row: SpeakerRow) -> bool:
    return row.kind == "element" and "constant-curvature" in row.note.lower()


def box_height(row: SpeakerRow) -> float:
    """Box height in metres: the datasheet figure, else a rule from the woofer size (8 in: 0.24 m, 15 in: 0.42 m)."""
    return round(row.box_h_mm / 1000 if row.box_h_mm else 0.03 + row.woofer_in * 0.026, 3)


def hang(row: SpeakerRow, boxes: int, splay: float) -> tuple[float, float]:
    """Level and vertical coverage of a hang of `boxes` elements at `splay` degrees between boxes."""
    peak = row.max_spl + _TO_PEAK[row.spl_basis]
    return round(min(160.0, peak + 10 * math.log10(max(1, boxes))), 1), round(max(5.0, min(180.0, boxes * splay)), 1)


def to_speaker_type(row: SpeakerRow) -> dict:
    peak = row.max_spl + _TO_PEAK[row.spl_basis]
    level, cov_v, label = peak, row.coverage_v, f"{row.brand} {row.model}"
    extra = {}
    estimated = ["pattern-control frequency" if not row.pattern_hz else None, "high-pass corner"]
    if row.spl_basis == "continuous":
        estimated.append("peak SPL = continuous + 6 dB")
    if row.kind == "element":
        splay = row.coverage_v if constant_curvature(row) else min(3.0, row.coverage_v)
        level, cov_v = hang(row, DEFAULT_BOXES, splay)
        extra = {"array_boxes": DEFAULT_BOXES, "array_splay": splay, "array_box_h": box_height(row),
                 "array_box_curve": row.coverage_v if constant_curvature(row) else 0.0}
        estimated.append("hang reflections: one source, level +10·log10(boxes), vertical = boxes × splay")
        if not row.box_h_mm:
            estimated.append("box height (from the woofer size)")
    return {
        "id": row.id, "label": label, "brand": row.brand, "group": "My speakers" if row.user else row.brand,
        "coverage_h": row.coverage_h, "coverage_v": cov_v, "level_db": min(160.0, level),
        "hp_hz": round(row.lf_hz * _TO_3DB[row.lf_db]),
        "lp_hz": round(row.hf_hz / _TO_3DB[row.lf_db]) if row.hf_hz else (100 if row.kind == "sub" else 0),
        "directivity_hz": _pattern_control_hz(row),
        "role": "sub" if row.kind == "sub" else "main",
        **extra,
        "spec": {
            "kind": row.kind, "coverage": "omnidirectional" if row.kind == "sub" else f"{row.coverage_h:g}° × {row.coverage_v:g}°",
            "max_spl": row.max_spl, "spl_basis": row.spl_basis, "low_limit": row.lf_hz, "low_limit_db": row.lf_db,
            "high_limit": row.hf_hz, "source": row.source, "note": row.note, "approximate": row.approximate, "user": row.user,
            "box_spl": peak if row.kind == "element" else None,
            "splay_max": row.coverage_v if row.kind == "element" else None,
            "constant_curvature": constant_curvature(row),
            "estimated": [e for e in estimated if e],
        },
    }


_DATA = Path(__file__).with_name("data") / "speakers.json"
ROWS: list[SpeakerRow] = [SpeakerRow(**r) for r in json.loads(_DATA.read_text(encoding="utf-8"))["speakers"]]
LIBRARY: list[dict] = [to_speaker_type(r) for r in ROWS]
BY_ID: dict[str, dict] = {t["id"]: t for t in LIBRARY}
ROW_BY_ID: dict[str, SpeakerRow] = {r.id: r for r in ROWS}
