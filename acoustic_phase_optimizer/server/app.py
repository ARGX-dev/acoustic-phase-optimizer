"""HTTP API and static web app.

The server is stateless about the venue: every request carries the project, so
there is nothing to get out of sync between the UI and the engine.
"""

from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .. import __version__
from ..engine import presets
from ..engine.advice import advise
from ..engine.field import get_field
from ..engine.materials import MATERIALS
from ..engine.metrics import BANDS, Weights
from ..engine.model import Project, Speaker
from ..engine.optimize import auto_eq, deep_optimize, diff_speakers, evaluate, link_units, quick_align
from ..engine.simulate import point_response, room_report, simulate
from ..export import FORMATS
from ..report import render_report
from . import storage
from .jobs import Job, JobManager

WEB_DIR = Path(__file__).resolve().parent.parent / "web"


class ProjectIn(BaseModel):
    project: Project


class SimulateIn(ProjectIn):
    preview: bool = False


class PointIn(ProjectIn):
    x: float
    y: float
    z: float | None = None


class AutoEqIn(ProjectIn):
    max_bands: int = Field(5, ge=1, le=10)


class OptimizeIn(ProjectIn):
    weights: dict[str, float] = Field(default_factory=dict)
    time_budget_s: float = Field(15.0, ge=2, le=120)
    start: Literal["align", "current"] = "align"


class ExportIn(ProjectIn):
    format: Literal["csv", "json", "txt"]


class ReportIn(ProjectIn):
    baseline: Project | None = None


class SessionIn(BaseModel):
    project: Project
    baseline: Project | None = None
    ui: dict = Field(default_factory=dict)


def _dump(speakers: list[Speaker]) -> list[dict]:
    return [s.model_dump() for s in speakers]


def _weights(raw: dict[str, float]) -> Weights:
    w = Weights()
    for k, v in raw.items():
        if hasattr(w, k):
            setattr(w, k, float(min(max(v, 0.0), 5.0)))
    return w


class Presence:
    """Whether a window is still showing the app, so the launcher can stop when the last one closes."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.last_seen: float | None = None
        self.bye_at: float | None = None

    def ping(self) -> None:
        with self._lock:
            self.last_seen = time.monotonic()
            self.bye_at = None

    def resumed(self) -> None:
        """The computer slept: give the page a fresh timeout instead of counting the sleep as silence."""
        with self._lock:
            if self.last_seen is not None:
                self.last_seen = time.monotonic()

    def bye(self) -> None:
        with self._lock:
            self.bye_at = time.monotonic()

    def gone(self, grace_s: float = 5.0, timeout_s: float = 180.0) -> bool:
        """True once a page said goodbye and none came back (a reload pings again), or all went silent."""
        with self._lock:
            if self.last_seen is None:
                return False
            now = time.monotonic()
            if self.bye_at is not None and now - self.bye_at > grace_s:
                return True
            return now - self.last_seen > timeout_s


def create_app() -> FastAPI:
    app = FastAPI(title="Acoustic Phase Optimizer", version=__version__, docs_url="/api/docs", redoc_url=None)
    jobs = JobManager()
    app.state.presence = presence = Presence()

    @app.middleware("http")
    async def no_cache(request: Request, call_next):
        response = await call_next(request)
        response.headers.setdefault("Cache-Control", "no-cache")
        return response

    @app.exception_handler(Exception)
    async def unhandled(_: Request, exc: Exception):
        return JSONResponse(status_code=500, content={"detail": f"{type(exc).__name__}: {exc}"})

    @app.post("/api/ping")
    def api_ping() -> dict:
        presence.ping()
        return {"ok": True}

    @app.post("/api/bye")
    def api_bye() -> dict:
        presence.bye()
        return {"ok": True}

    # ---------------------------------------------------------------- reference data
    @app.get("/api/meta")
    def meta() -> dict:
        return {
            "version": __version__,
            "bands": BANDS,
            "materials": [{"id": m.id, "name": m.name, "alpha": list(m.alpha)} for m in MATERIALS.values()],
            "speaker_types": [{"id": k, **v} for k, v in presets.SPEAKER_TYPES.items()],
            "venues": [{"id": k, "name": v[0]} for k, v in presets.VENUES.items()],
            "data_dir": str(storage.data_dir()),
        }

    @app.get("/api/venues/{venue_id}")
    def venue(venue_id: str) -> dict:
        if venue_id == "blank":
            return presets.blank().model_dump()
        if venue_id not in presets.VENUES:
            raise HTTPException(404, "Unknown venue")
        return presets.VENUES[venue_id][1]().model_dump()

    # ---------------------------------------------------------------------- engine
    @app.post("/api/simulate")
    def api_simulate(body: SimulateIn) -> dict:
        result = simulate(body.project, body.preview)
        result["room"] = room_report(body.project)
        result["advice"] = advise(result["metrics"], result["room"])
        result["units"] = [[body.project.speakers[i].id for i in u] for u in link_units(body.project)]
        return result

    @app.post("/api/point")
    def api_point(body: PointIn) -> dict:
        return point_response(body.project, body.x, body.y, body.z)

    @app.post("/api/align")
    def api_align(body: ProjectIn) -> dict:
        p = body.project
        field = get_field(p)
        speakers, notes = quick_align(p, field)
        return {"speakers": _dump(speakers), "notes": notes, "changes": diff_speakers(p.speakers, speakers),
                "before": evaluate(p, field), "after": evaluate(p, field, speakers)}

    @app.post("/api/autoeq")
    def api_autoeq(body: AutoEqIn) -> dict:
        p = body.project
        field = get_field(p)
        speakers, report = auto_eq(p, field, max_bands=body.max_bands)
        notes = [f"{r['name']}: {r['bands']} band{'s' if r['bands'] != 1 else ''}, deviation "
                 f"{r['rms_before_db']:.1f} → {r['rms_after_db']:.1f} dB" if r["bands"]
                 else f"{r['name']}: already within tolerance, no EQ needed" for r in report]
        return {"speakers": _dump(speakers), "notes": notes, "report": report,
                "changes": diff_speakers(p.speakers, speakers),
                "before": evaluate(p, field), "after": evaluate(p, field, speakers)}

    @app.post("/api/optimize")
    def api_optimize(body: OptimizeIn) -> dict:
        p = body.project

        def run(job: Job) -> dict:
            field = get_field(p)
            before = evaluate(p, field)
            job.progress = {"generation": 0, "score": before.get("score"), "start_score": before.get("score"),
                            "fraction": 0.0}

            def progress(pr) -> None:
                job.progress = {"generation": pr.generation, "score": round(pr.score, 2),
                                "start_score": round(pr.start_score, 2),
                                "fraction": round(min(1.0, pr.elapsed / body.time_budget_s), 3)}

            speakers, info = deep_optimize(p, field, _weights(body.weights), start=body.start,
                                           time_budget_s=body.time_budget_s, progress=progress, cancelled=job.cancelled)
            picked = {"optimized": "Search result is the best of the three candidates.",
                      "aligned": "The search found nothing better than the quick alignment, so that is what you get.",
                      "original": "Your current settings already score best; nothing was changed."}[info["picked"]]
            return {"speakers": _dump(speakers), "notes": [picked], "info": info,
                    "changes": diff_speakers(p.speakers, speakers),
                    "before": before, "after": evaluate(p, field, speakers)}

        return {"job_id": jobs.start(run).id}

    @app.get("/api/jobs/{job_id}")
    def api_job(job_id: str) -> dict:
        job = jobs.get(job_id)
        if job is None:
            raise HTTPException(404, "Unknown job")
        return job.view()

    @app.delete("/api/jobs/{job_id}")
    def api_job_cancel(job_id: str) -> dict:
        job = jobs.get(job_id)
        if job is None:
            raise HTTPException(404, "Unknown job")
        job.cancel()
        return {"ok": True}

    # ---------------------------------------------------------------- export / report
    @app.post("/api/export")
    def api_export(body: ExportIn) -> Response:
        fn, mime, ext = FORMATS[body.format]
        name = storage.slug(body.project.name)
        return Response(fn(body.project), media_type=mime,
                        headers={"Content-Disposition": f'attachment; filename="{name}-dsp-settings.{ext}"'})

    @app.post("/api/report", response_class=HTMLResponse)
    def api_report(body: ReportIn) -> str:
        return render_report(body.project, body.baseline)

    # ---------------------------------------------------------------------- storage
    @app.get("/api/projects")
    def api_projects() -> list[dict]:
        return storage.list_projects()

    @app.post("/api/projects")
    def api_project_save(body: ProjectIn) -> dict:
        return storage.save_project(body.project)

    @app.get("/api/projects/{key}")
    def api_project_load(key: str) -> dict:
        try:
            return storage.load_project(key).model_dump()
        except FileNotFoundError:
            raise HTTPException(404, "Project not found") from None
        except ValueError as exc:
            raise HTTPException(422, f"Project file is not valid: {exc}") from None

    @app.delete("/api/projects/{key}")
    def api_project_delete(key: str) -> dict:
        if not storage.delete_project(key):
            raise HTTPException(404, "Project not found")
        return {"ok": True}

    @app.get("/api/session")
    def api_session_get() -> dict:
        return storage.load_session() or {}

    @app.put("/api/session")
    def api_session_put(body: SessionIn) -> dict:
        storage.save_session(body.model_dump())
        return {"ok": True}

    app.mount("/", StaticFiles(directory=WEB_DIR, html=True), name="web")
    return app
