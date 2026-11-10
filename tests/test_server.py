"""End-to-end API tests against a real server on a local port."""

from __future__ import annotations

import json
import socket
import threading
import time
import urllib.error
import urllib.request

import pytest
import uvicorn

from acoustic_phase_optimizer.engine import presets
from acoustic_phase_optimizer.server import create_app, storage


@pytest.fixture(scope="module")
def api(tmp_path_factory):
    mp = pytest.MonkeyPatch()
    mp.setenv("APO_DATA_DIR", str(tmp_path_factory.mktemp("data")))
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    server = uvicorn.Server(uvicorn.Config(create_app(), host="127.0.0.1", port=port, log_level="error"))
    thread = threading.Thread(target=server.run, daemon=True)
    thread.start()
    deadline = time.time() + 15
    while not server.started and time.time() < deadline:
        time.sleep(0.02)
    assert server.started

    def call(method: str, path: str, body=None, raw: bool = False):
        data = None if body is None else json.dumps(body).encode()
        req = urllib.request.Request(f"http://127.0.0.1:{port}{path}", data=data, method=method,
                                     headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                payload = r.read()
                return (r.status, dict(r.headers), payload.decode()) if raw else json.loads(payload)
        except urllib.error.HTTPError as e:
            if raw:
                return e.code, dict(e.headers), e.read().decode()
            raise

    yield call
    server.should_exit = True
    thread.join(5)
    mp.undo()


def small(name: str = "conference") -> dict:
    p = presets.VENUES[name][1]()
    p.settings.grid_resolution = 1.2
    return p.model_dump()


def test_static_index_and_meta(api):
    status, headers, body = api("GET", "/", raw=True)
    assert status == 200 and "Acoustic Phase Optimizer" in body
    assert headers.get("cache-control") == "no-cache"
    meta = api("GET", "/api/meta")
    assert {v["id"] for v in meta["venues"]} == set(presets.VENUES)
    assert any(m["id"] == "carpet" for m in meta["materials"])
    assert all("coverage_h" in t and "label" in t for t in meta["speaker_types"])


def test_venue_simulate_and_point(api):
    project = api("GET", "/api/venues/gymnasium")
    assert len({s["color"] for s in project["speakers"]}) == len(project["speakers"])
    sim = api("POST", "/api/simulate", {"project": project, "preview": True})
    assert sim["preview"] and sim["metrics"]["score"] > 0
    assert sim["advice"] and sim["room"]["rt60_mid"] > 1
    assert sorted(len(u) for u in sim["units"]) == [2, 2, 2]
    pt = api("POST", "/api/point", {"project": project, "x": 15, "y": 20})
    assert len(pt["speakers"]) == 6 and pt["arrivals"]
    assert api("GET", "/api/venues/blank")["speakers"]
    assert api("GET", "/api/venues/nope", raw=True)[0] == 404


def test_align_and_autoeq(api):
    project = small("gymnasium")
    res = api("POST", "/api/align", {"project": project})
    assert res["after"]["score"] > res["before"]["score"]
    assert res["notes"] and any(c["field"] == "delay_ms" for c in res["changes"])
    project["speakers"] = res["speakers"]
    eq = api("POST", "/api/autoeq", {"project": project, "max_bands": 3})
    assert len(eq["notes"]) == len(eq["report"]) >= 1
    assert all(len([b for b in s["peq"] if b["id"].startswith("auto")]) <= 3 for s in eq["speakers"])


def test_optimize_job_runs_to_completion(api):
    job = api("POST", "/api/optimize", {"project": small("gymnasium"), "time_budget_s": 2})["job_id"]
    deadline = time.time() + 90
    view = api("GET", f"/api/jobs/{job}")
    while view["state"] == "running" and time.time() < deadline:
        time.sleep(0.2)
        view = api("GET", f"/api/jobs/{job}")
    assert view["state"] == "done", view
    r = view["result"]
    assert r["after"]["score"] >= r["before"]["score"] - 0.2
    assert r["info"]["picked"] in {"optimized", "aligned", "original"} and len(r["speakers"]) == 6
    assert api("GET", "/api/jobs/unknown", raw=True)[0] == 404


def test_optimize_job_can_be_cancelled(api):
    job = api("POST", "/api/optimize", {"project": small("gymnasium"), "time_budget_s": 60})["job_id"]
    assert api("DELETE", f"/api/jobs/{job}")["ok"]
    deadline = time.time() + 60
    view = api("GET", f"/api/jobs/{job}")
    while view["state"] == "running" and time.time() < deadline:
        time.sleep(0.2)
        view = api("GET", f"/api/jobs/{job}")
    assert view["state"] == "cancelled" and view["result"] is None


def test_exports(api):
    project = small()
    project["speakers"][0]["delay_ms"] = 12.5
    project["speakers"][0]["peq"] = [{"type": "bell", "freq": 250, "gain_db": -4.5, "q": 2}]
    status, headers, csv = api("POST", "/api/export", {"project": project, "format": "csv"}, raw=True)
    assert status == 200 and 'filename="conference-room-dsp-settings.csv"' in headers["content-disposition"]
    lines = csv.strip().split("\n")
    assert lines[0].startswith("Output,Name,Role,Delay (ms)") and "PEQ1 Freq (Hz)" in lines[0]
    assert lines[1].split(",")[3] == "12.50" and "-4.5" in lines[1]
    assert len(lines) == 1 + len(project["speakers"])
    doc = json.loads(api("POST", "/api/export", {"project": project, "format": "json"}, raw=True)[2])
    assert doc["outputs"][0]["delay_ms"] == 12.5 and doc["outputs"][0]["peq"][0]["gain_db"] == -4.5
    txt = api("POST", "/api/export", {"project": project, "format": "txt"}, raw=True)[2]
    assert "OUTPUT 1: Front L" in txt and "12.50 ms" in txt and "Bell" in txt
    assert api("POST", "/api/export", {"project": project, "format": "xml"}, raw=True)[0] == 422


def test_report_is_self_contained_and_escapes_names(api):
    project = small()
    project["name"] = "<script>alert(1)</script> Hall"
    baseline = json.loads(json.dumps(project))
    project["speakers"] = api("POST", "/api/align", {"project": project})["speakers"]
    status, _, html = api("POST", "/api/report", {"project": project, "baseline": baseline}, raw=True)
    assert status == 200
    assert "<script>alert(1)</script>" not in html and "&lt;script&gt;" in html
    assert html.count("data:image/png;base64,") == 6  # three maps, before and after
    assert "Before optimization" in html and "DSP settings" in html
    assert "http://" not in html and "https://" not in html  # no external assets


def test_project_storage_round_trip(api):
    project = small()
    project["name"] = "St. Mary's / Hall #2"
    saved = api("POST", "/api/projects", {"project": project})
    listing = api("GET", "/api/projects")
    assert [p["name"] for p in listing] == ["St. Mary's / Hall #2"]
    assert listing[0]["key"] == saved["key"] and listing[0]["speakers"] == 4
    assert api("GET", f"/api/projects/{saved['key']}")["name"] == project["name"]
    assert api("DELETE", f"/api/projects/{saved['key']}")["ok"]
    assert api("GET", "/api/projects") == []
    assert api("GET", f"/api/projects/{saved['key']}", raw=True)[0] == 404


def test_storage_keys_cannot_escape_the_projects_folder():
    for hostile in ("../../evil", "..\\..\\evil", "/etc/passwd", "C:\\Windows\\x", "", "..."):
        assert all(ch not in storage.slug(hostile) for ch in "/\\:.")


def test_session_autosave(api):
    assert api("GET", "/api/session") == {}
    project = small()
    assert api("PUT", "/api/session", {"project": project, "ui": {"map": "clarity"}})["ok"]
    back = api("GET", "/api/session")
    assert back["project"]["name"] == project["name"] and back["ui"] == {"map": "clarity"}
    (storage.data_dir() / "session.json").write_text("{not json", encoding="utf-8")
    assert api("GET", "/api/session") == {}


def test_invalid_projects_are_rejected_with_a_reason(api):
    project = small()
    project["room"]["width"] = -3
    status, _, body = api("POST", "/api/simulate", {"project": project}, raw=True)
    assert status == 422 and "width" in body
