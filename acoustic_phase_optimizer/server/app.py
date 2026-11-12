"""HTTP API and static web app.

The server is stateless about the venue: every request carries the project, so
there is nothing to get out of sync between the UI and the engine.
"""

from __future__ import annotations

import json
import mimetypes
import threading
import time
from pathlib import Path
from typing import Literal

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from .. import __version__
from ..engine import presets
from ..engine.advice import advise
from ..engine.field import get_field
from ..engine.library import LIBRARY, ROW_BY_ID, SpeakerRow, to_speaker_type
from ..engine.materials import MATERIALS
from ..engine.metrics import BANDS, Weights
from ..engine.model import Project, Speaker
from ..engine.optimize import auto_eq, deep_optimize, diff_speakers, evaluate, link_units, quick_align
from ..engine.scanfit import fit_box
from ..engine.simulate import point_response, room_report, section, simulate
from ..export import FORMATS
from ..report import render_report
from . import storage
from .jobs import Job, JobManager

WEB_DIR = Path(__file__).resolve().parent.parent / "web"
MAX_SHAPE_BYTES = 64_000_000  # triangles sent to /api/shape (about 4M; the reduction takes ~10 us each)


class ProjectIn(BaseModel):
    project: Project


class SimulateIn(ProjectIn):
    preview: bool = False


class PointIn(ProjectIn):
    x: float
    y: float
    z: float | None = None


class SectionIn(ProjectIn):
    axis: Literal["length", "width"] = "length"
    at: float | None = None
    band: Literal["all", "63", "125", "250", "500", "1000", "2000", "4000", "8000"] = "all"
    coarse: bool = False  # while dragging: a quarter of the points


class SubArrayIn(ProjectIn):
    kind: Literal["line", "cardioid", "endfire"]
    model: str = "sub_18"
    count: int = Field(4, ge=1, le=24)
    spacing: float = Field(1.2, ge=0.3, le=10)
    depth: float = Field(1.0, ge=0.3, le=4)
    spread_deg: float = Field(0.0, ge=0, le=170)
    x: float
    y: float
    replace: bool = True


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


class ScanFitIn(BaseModel):
    points: list[float] = Field(max_length=3 * 120_000)  # flat x, y, z (metres, z up)
    normals: list[float] | None = Field(None, max_length=3 * 120_000)


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


LOOPBACK_HOSTS = {"127.0.0.1", "localhost", "::1"}


def content_security_policy() -> str:
    """Only this app's own files run. The import map is the one inline script; it is allowed by its hash."""
    import base64
    import hashlib
    import re as _re

    html = (WEB_DIR / "index.html").read_text(encoding="utf-8")
    hashes = [
        "'sha256-" + base64.b64encode(hashlib.sha256(m.encode("utf-8")).digest()).decode() + "'"
        for m in _re.findall(r"<script type=\"importmap\">(.*?)</script>", html, flags=_re.S)
    ]
    return "; ".join([
        "default-src 'self'",
        # wasm-unsafe-eval lets the decoders (Draco, KTX2, IFC, STEP...) compile WebAssembly; JavaScript eval stays blocked
        "script-src 'self' 'wasm-unsafe-eval' " + " ".join(hashes),
        "style-src 'self' 'unsafe-inline'",  # element.style from the UI code
        "img-src 'self' data: blob:",
        "connect-src 'self' blob:",  # GLTFLoader fetches embedded textures through blob: URLs
        "worker-src 'self' blob:",
        "object-src 'none'",
        "base-uri 'none'",
        "form-action 'none'",
        "frame-ancestors 'none'",
    ])


def create_app() -> FastAPI:
    # No interactive API docs: their page loads scripts from a CDN, which this app never does.
    app = FastAPI(title="Acoustic Phase Optimizer", version=__version__, docs_url=None, redoc_url=None, openapi_url=None)
    jobs = JobManager()
    app.state.presence = presence = Presence()

    csp = content_security_policy()

    @app.middleware("http")
    async def guard(request: Request, call_next):
        """The engine is for this computer's browser only.

        - Host must be a loopback name, which defeats DNS-rebinding pages that point a domain at 127.0.0.1.
        - A request that changes anything must not come from another site: a web page you visit can send
          POSTs to localhost, so Origin (when the browser sends one) must be this app, and fetch metadata
          must not say cross-site.
        """
        host = (request.headers.get("host") or "").rsplit(":", 1)[0].strip("[]").lower()
        if host not in LOOPBACK_HOSTS:
            return JSONResponse(status_code=403, content={"detail": "This engine only answers on 127.0.0.1."})
        if request.method not in ("GET", "HEAD", "OPTIONS"):
            origin = request.headers.get("origin")
            if origin and origin != "null":
                o_host = origin.split("://", 1)[-1].rsplit(":", 1)[0].strip("[]").lower()
                if o_host not in LOOPBACK_HOSTS or origin.split("://", 1)[-1] != request.headers.get("host"):
                    return JSONResponse(status_code=403, content={"detail": "Requests from other sites are refused."})
            elif origin == "null" or request.headers.get("sec-fetch-site") == "cross-site":
                return JSONResponse(status_code=403, content={"detail": "Requests from other sites are refused."})
        response = await call_next(request)
        response.headers.setdefault("Cache-Control", "no-cache")
        response.headers["X-Content-Type-Options"] = "nosniff"
        response.headers["Referrer-Policy"] = "no-referrer"
        response.headers["Content-Security-Policy"] = csp
        response.headers["X-Frame-Options"] = "DENY"
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
            "speaker_types": [{"id": k, "group": "Generic", **v} for k, v in presets.SPEAKER_TYPES.items()],
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

    @app.post("/api/section")
    def api_section(body: SectionIn) -> dict:
        return section(body.project, body.axis, body.at, body.band, body.coarse)

    @app.post("/api/subarray")
    def api_subarray(body: SubArrayIn) -> dict:
        from ..engine import subarray
        from ..engine.library import BY_ID
        from ..engine.presets import SPEAKER_TYPES

        if body.model not in SPEAKER_TYPES and body.model not in BY_ID:
            raise HTTPException(422, f"Unknown loudspeaker model {body.model!r}.")
        p = body.project
        try:
            new = subarray.build(p, body.kind, body.model, body.count, body.spacing, body.x, body.y,
                                 depth=body.depth, spread_deg=body.spread_deg)
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from None
        keep = [s for s in p.speakers if not (body.replace and s.role == "sub")]
        if len(keep) + len(new) > 64:
            raise HTTPException(422, "That would make more than 64 speakers.")
        return {"speakers": [s.model_dump() for s in keep + new], "added": len(new),
                "removed": len(p.speakers) - len(keep)}

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

    # --------------------------------------------------------------------- speakers
    def user_rows() -> list[SpeakerRow]:
        out = []
        for r in storage.load_user_speakers():
            try:
                out.append(SpeakerRow(**{**r, "user": True}))
            except ValueError:
                continue  # skip a row edited by hand into something invalid
        return out

    @app.get("/api/speakers")
    def api_speakers() -> dict:
        """Every real loudspeaker: the checked library, then the user's own models."""
        return {"library": LIBRARY, "user": [to_speaker_type(r) for r in user_rows()],
                "fields": list(SpeakerRow.model_fields)}

    @app.post("/api/speakers/user")
    def api_speaker_save(row: SpeakerRow) -> dict:
        if row.id in ROW_BY_ID:
            raise HTTPException(409, "That id belongs to a library model. Choose another name.")
        rows = [r for r in user_rows() if r.id != row.id] + [row.model_copy(update={"user": True})]
        storage.save_user_speakers([r.model_dump() for r in rows])
        return to_speaker_type(rows[-1])

    @app.delete("/api/speakers/user/{row_id}")
    def api_speaker_delete(row_id: str) -> dict:
        rows = user_rows()
        keep = [r for r in rows if r.id != row_id]
        storage.save_user_speakers([r.model_dump() for r in keep])
        return {"deleted": len(rows) - len(keep)}

    @app.post("/api/speakers/import")
    async def api_speaker_import(request: Request) -> dict:
        """CSV (with a header row) or JSON (a list of rows). Valid rows are added; each bad row is reported."""
        import csv
        import io
        import re as _re

        buf = bytearray()
        async for chunk in request.stream():
            buf += chunk
            if len(buf) > 5_000_000:
                raise HTTPException(413, "The file is larger than 5 MB.")
        text = bytes(buf).decode("utf-8-sig", errors="replace")
        try:
            raw = json.loads(text) if text.lstrip().startswith(("[", "{")) else list(csv.DictReader(io.StringIO(text)))
        except ValueError as exc:
            raise HTTPException(422, f"Could not read the file: {exc}") from None
        if isinstance(raw, dict):
            raw = raw.get("speakers", [])
        existing = {r.id: r for r in user_rows()}
        added, errors = 0, []
        for i, r in enumerate(raw, start=2 if not text.lstrip().startswith(("[", "{")) else 1):
            if not isinstance(r, dict):
                errors.append({"row": i, "error": "not a row"})
                continue
            r = {k.strip().lower(): (v.strip() if isinstance(v, str) else v) for k, v in r.items() if k}
            r = {k: v for k, v in r.items() if v not in ("", None)}
            if "id" not in r and r.get("brand") and r.get("model"):
                r["id"] = _re.sub(r"[^a-z0-9]+", "_", f"{r['brand']} {r['model']}".lower()).strip("_")[:60]
            try:
                row = SpeakerRow(**{**r, "user": True})
                if row.id in ROW_BY_ID:
                    raise ValueError("id is already used by a library model")
            except ValueError as exc:
                msg = str(exc).splitlines()
                errors.append({"row": i, "error": " ".join(m.strip() for m in msg[1:3]) or msg[0]})
                continue
            existing[row.id] = row
            added += 1
        storage.save_user_speakers([r.model_dump() for r in existing.values()])
        return {"added": added, "errors": errors[:50], "error_count": len(errors)}

    @app.get("/api/speakers/template.csv")
    def api_speaker_template() -> Response:
        import csv
        import io

        cols = [f for f in SpeakerRow.model_fields if f not in ("user", "approximate")]
        # Examples keyed by column name, so a new column can never shift them into the wrong place.
        examples = [
            {"id": "my_top_12", "brand": "MyBrand", "model": "Top 12", "kind": "point", "woofer_in": 12, "coverage_h": 90,
             "coverage_v": 60, "max_spl": 130, "spl_basis": "peak", "lf_hz": 50, "lf_db": -10,
             "source": "https://example.com/datasheet.pdf"},
            {"id": "my_line_10", "brand": "MyBrand", "model": "Line 10", "kind": "element", "woofer_in": 10, "coverage_h": 90,
             "coverage_v": 10, "max_spl": 136, "spl_basis": "peak", "lf_hz": 60, "lf_db": -10, "box_h_mm": 290,
             "source": "https://example.com/line10.pdf", "note": "coverage_v of a line-array box = its largest splay"},
        ]
        out = io.StringIO()
        w = csv.DictWriter(out, fieldnames=cols, lineterminator="\n")
        w.writeheader()
        w.writerows(examples)
        return Response(out.getvalue(), media_type="text/csv",
                        headers={"Content-Disposition": 'attachment; filename="speakers-template.csv"'})

    # ------------------------------------------------------------------------ DWG
    @app.get("/api/convert/dwg")
    def api_dwg_status() -> dict:
        from . import dwg

        found = dwg.find_converter()
        return {"available": bool(found), "kind": found[1] if found else None}

    @app.post("/api/convert/dwg")
    async def api_dwg_convert(request: Request) -> Response:
        """DWG in, ASCII DXF out, through a converter installed on this computer (see server/dwg.py)."""
        import asyncio

        from . import dwg

        buf = bytearray()
        async for chunk in request.stream():
            buf += chunk
            if len(buf) > storage.MAX_SCAN_BYTES:
                raise HTTPException(413, f"The drawing is larger than {storage.MAX_SCAN_BYTES // 1_000_000} MB.")
        try:
            dxf = await asyncio.to_thread(dwg.convert, bytes(buf))
        except dwg.NoConverter:
            raise HTTPException(501, "DWG needs a converter on this computer: install the free ODA File Converter "
                                     "(opendesign.com/guestfiles/oda_file_converter) or LibreDWG, then import again. "
                                     "Or save the drawing as DXF.") from None
        except dwg.ConvertFailed as exc:
            raise HTTPException(422, f"The DWG could not be converted: {exc}.") from None
        return Response(dxf, media_type="text/plain; charset=utf-8")

    # ------------------------------------------------------------------------ scans
    @app.post("/api/scans")
    async def api_scan_upload(request: Request, ext: str) -> dict:
        ext = ext.lower().lstrip(".")
        if ext not in storage.SCAN_TYPES:
            raise HTTPException(415, "Scans and models can be GLB, USDZ/USD, OBJ, PLY, MVR, 3DS, DAE, FBX or STL files.")
        size = int(request.headers.get("content-length") or 0)
        if size > storage.MAX_SCAN_BYTES:
            raise HTTPException(413, f"The scan is {size / 1e6:.0f} MB; the limit is {storage.MAX_SCAN_BYTES // 1_000_000} MB.")
        import hashlib
        import tempfile

        digest, total = hashlib.sha256(), 0
        tmp = tempfile.NamedTemporaryFile(delete=False, dir=storage.scans_dir(), suffix=".part")
        try:
            with tmp:
                async for chunk in request.stream():
                    total += len(chunk)
                    if total > storage.MAX_SCAN_BYTES:
                        raise HTTPException(413, f"The scan is larger than {storage.MAX_SCAN_BYTES // 1_000_000} MB.")
                    digest.update(chunk)
                    tmp.write(chunk)
            if total < 64:
                raise HTTPException(422, "The file is empty.")
            return storage.adopt_scan(Path(tmp.name), digest.hexdigest()[:32], ext, total)
        finally:
            Path(tmp.name).unlink(missing_ok=True)

    @app.get("/api/scans/{scan_id}.{ext}")
    def api_scan_file(scan_id: str, ext: str) -> FileResponse:
        try:
            path = storage.scan_file(scan_id, ext)
        except (FileNotFoundError, ValueError):
            raise HTTPException(404, "Scan not found. It may have been made on another computer.") from None
        return FileResponse(path, media_type=storage.SCAN_TYPES[ext], headers={"Cache-Control": "max-age=31536000, immutable"})

    @app.post("/api/scan-fit")
    def api_scan_fit(body: ScanFitIn) -> dict:
        import numpy as np

        pts = np.asarray(body.points, dtype=float)
        if len(pts) % 3:
            raise HTTPException(422, "points must be x, y, z triples")
        nrm = np.asarray(body.normals, dtype=float).reshape(-1, 3) if body.normals else None
        try:
            return fit_box(pts.reshape(-1, 3), nrm)
        except ValueError as exc:
            raise HTTPException(422, str(exc)) from None

    @app.post("/api/shape")
    async def api_shape(request: Request) -> dict:
        """The scanned room's acoustic shape (engine/geometry.py:build_shape) from the placed model's triangles.

        Binary body, little-endian: u32 header length, a JSON header {dims: [w, l, h], names: [...], nv, nt},
        zero padding to a multiple of 4, then float32 positions (nv x 3, room metres, z up), uint32 indices
        (nt x 3) and uint32 material-name indices per triangle (nt; 0xffffffff for none).
        """
        import asyncio
        import json
        import struct

        import numpy as np

        from ..engine import geometry

        buf = bytearray()
        async for chunk in request.stream():
            buf += chunk
            if len(buf) > MAX_SHAPE_BYTES:
                raise HTTPException(413, "The model is too detailed to use as the room's shape "
                                         f"(over {MAX_SHAPE_BYTES // 1_000_000} MB of triangles).")
        try:
            (hlen,) = struct.unpack_from("<I", buf, 0)
            head = json.loads(bytes(buf[4:4 + hlen]))
            nv, nt = int(head["nv"]), int(head["nt"])
            dims = tuple(float(d) for d in head["dims"])
            names = [str(n)[:120] for n in head.get("names", [])][:4096]
            off = (4 + hlen + 3) // 4 * 4
            if len(dims) != 3 or nv < 3 or nt < 1 or len(buf) != off + 12 * nv + 16 * nt:
                raise ValueError
        except (ValueError, KeyError, TypeError, struct.error):
            raise HTTPException(422, "The model data could not be read.") from None
        V = np.frombuffer(buf, "<f4", 3 * nv, off).reshape(-1, 3)
        T = np.frombuffer(buf, "<u4", 3 * nt, off + 12 * nv).reshape(-1, 3).astype(np.int64)
        tag = np.frombuffer(buf, "<u4", nt, off + 12 * nv + 12 * nt).astype(np.int64)
        tag[tag >= len(names)] = len(names)  # "no name" sorts past the names and picks no material
        try:
            return await asyncio.to_thread(geometry.build_shape, V, T, dims, tag, names)
        except ValueError as exc:
            raise HTTPException(422, f"The room's shape could not be made: {exc}.") from None

    # ---------------------------------------------------------------------- storage
    @app.get("/api/projects")
    def api_projects() -> list[dict]:
        return storage.list_projects()

    @app.post("/api/projects")
    def api_project_save(body: ProjectIn) -> dict:
        return storage.save_project(body.project)

    @app.post("/api/projects/check")
    def api_project_check(body: ProjectIn) -> dict:
        """Validate a project that arrived in a file (a .trky bundle) and return it in today's shape."""
        return body.project.model_dump()

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

    # ------------------------------------------------------------------ live measurement (engine/measure.py)
    async def _read_body(request: Request, limit: int, what: str) -> bytes:
        buf = bytearray()
        async for chunk in request.stream():
            buf += chunk
            if len(buf) > limit:
                raise HTTPException(413, f"The {what} is larger than {limit // 1_000_000} MB.")
        return bytes(buf)

    def _keep_ir(ir, fs: int, start_s: float, analysis: dict, warnings: list[str]) -> dict:
        from ..engine import measure

        ir_id = storage.save_ir(measure.write_wav(ir, fs))
        return {"ir_id": ir_id, "start_ms": round(start_s * 1000, 3), "analysis": analysis, "warnings": warnings}

    @app.get("/api/measure/sweep")
    def api_measure_sweep(fs: int, seconds: float = 3.0, f1: float = 20.0, f2: float = 20000.0) -> Response:
        """The sweep to play, as little-endian float32 at the audio device's rate (peak 1.0)."""
        from ..engine import measure

        if not 8000 <= fs <= 384000 or not 0.5 <= seconds <= 30 or not 5 <= f1 < f2:
            raise HTTPException(422, "fs 8-384 kHz, 0.5-30 s, f1 < f2")
        return Response(measure.sweep(fs, seconds, f1, f2).astype("<f4").tobytes(), media_type="application/octet-stream")

    @app.post("/api/measure/capture")
    async def api_measure_capture(request: Request) -> dict:
        """Analyse one sweep take. Binary body, little-endian: u32 header length, JSON header
        {fs, f1, f2, n, channels: ["mic", "ref"] or ["mic", "ref", "loop"]}, zero padding to a multiple of 4,
        then float32 samples, one channel after the other (n each). "ref" is what was played, sample-aligned
        with the mic; "loop" is the interface's output fed straight back into an input."""
        import asyncio
        import struct

        import numpy as np

        from ..engine import measure

        buf = await _read_body(request, storage.MAX_IR_BYTES * 3, "recording")
        try:
            (hlen,) = struct.unpack_from("<I", buf, 0)
            head = json.loads(buf[4:4 + hlen])
            fs, n = int(head["fs"]), int(head["n"])
            f1, f2 = float(head.get("f1", 20)), float(head.get("f2", 20000))
            names = list(head["channels"])
            off = (4 + hlen + 3) // 4 * 4
            if names[:2] != ["mic", "ref"] or len(names) > 3 or not 8000 <= fs <= 384000 or n < fs // 4:
                raise ValueError("bad header")
            if len(buf) - off != 4 * n * len(names):
                raise ValueError(f"expected {4 * n * len(names)} sample bytes, got {len(buf) - off}")
            x = np.frombuffer(buf, dtype="<f4", count=n * len(names), offset=off).reshape(len(names), n).astype(np.float64)
        except (ValueError, KeyError, TypeError, struct.error, json.JSONDecodeError) as exc:
            raise HTTPException(422, f"Not a recording this app understands ({exc}).") from None
        if not np.all(np.isfinite(x)):
            raise HTTPException(422, "The recording holds invalid samples.")
        try:
            res = await asyncio.to_thread(measure.capture, x[0], x[1], fs, f1, f2, x[2] if len(names) == 3 else None)
        except measure.MeasureError as exc:
            raise HTTPException(422, str(exc)) from None
        return _keep_ir(res["ir"], fs, res["start_s"], res["analysis"], res["warnings"])

    @app.post("/api/measure/ir")
    async def api_measure_ir(request: Request, f1: float = 20.0, f2: float = 20000.0) -> dict:
        """Analyse an impulse response WAV (from REW, Smaart, ARTA, a convolution reverb...). The body is the file."""
        import asyncio

        from ..engine import measure

        data = await _read_body(request, storage.MAX_IR_BYTES, "file")
        try:
            ir, fs = measure.read_wav(data)
            ir = measure.trim_ir(ir, fs)
            an = await asyncio.to_thread(measure.analyse_ir, ir, fs, f1, f2)
        except measure.MeasureError as exc:
            raise HTTPException(422, str(exc)) from None
        return _keep_ir(ir, fs, 0.0, an, [])

    @app.get("/api/irs/{ir_id}.wav")
    def api_ir_file(ir_id: str) -> FileResponse:
        try:
            path = storage.ir_file(ir_id)
        except (FileNotFoundError, ValueError):
            raise HTTPException(404, "Impulse response not found. It may have been measured on another computer.") from None
        return FileResponse(path, media_type="audio/wav", headers={"Cache-Control": "max-age=31536000, immutable"})

    @app.post("/api/irs")
    async def api_ir_upload(request: Request) -> dict:
        """Store an impulse response WAV as it is (a project bundle bringing its measurements back)."""
        from ..engine import measure

        data = await _read_body(request, storage.MAX_IR_BYTES, "file")
        try:
            measure.read_wav(data)
        except measure.MeasureError as exc:
            raise HTTPException(422, str(exc)) from None
        return {"ir_id": storage.save_ir(data)}

    @app.get("/api/irs/{ir_id}/analysis")
    async def api_ir_analysis(ir_id: str, start_ms: float = 0.0, f1: float = 20.0, f2: float = 20000.0) -> dict:
        """The analysis of a stored impulse response again (curves for the viewer); start_ms as it was measured."""
        import asyncio

        from ..engine import measure

        try:
            ir, fs = measure.read_wav(storage.ir_file(ir_id).read_bytes())
        except (FileNotFoundError, ValueError):
            raise HTTPException(404, "Impulse response not found. It may have been measured on another computer.") from None
        try:
            return await asyncio.to_thread(measure.analyse_ir, ir, fs, f1, f2, start_ms / 1000)
        except measure.MeasureError as exc:
            raise HTTPException(422, str(exc)) from None

    # WebAssembly must be served as application/wasm; a Windows registry entry can say otherwise
    mimetypes.add_type("application/wasm", ".wasm")
    app.mount("/", StaticFiles(directory=WEB_DIR, html=True), name="web")
    return app
