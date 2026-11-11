"""End-to-end API tests against a real server on a local port."""

from __future__ import annotations

import json
import pathlib
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
                return (r.status, dict(r.headers), payload.decode(errors="replace")) if raw else json.loads(payload)
        except urllib.error.HTTPError as e:
            if raw:
                return e.code, dict(e.headers), e.read().decode()
            raise

    call.port = port
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


def test_project_check_validates_without_saving(api):
    project = small()
    project["name"] = "From a bundle"
    checked = api("POST", "/api/projects/check", {"project": project})
    assert checked["name"] == "From a bundle" and len(checked["speakers"]) == 4
    assert api("GET", "/api/projects") == []  # nothing stored
    project["speakers"][0]["x"] = "not a number"
    assert api("POST", "/api/projects/check", {"project": project}, raw=True)[0] == 422


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


def test_presence_stops_only_after_the_last_window_leaves(monkeypatch):
    from acoustic_phase_optimizer.server import app as app_module

    clock = [1000.0]
    monkeypatch.setattr(app_module.time, "monotonic", lambda: clock[0])
    p = app_module.Presence()
    assert not p.gone()  # nothing has connected yet: never stop
    p.ping()
    clock[0] += 60
    assert not p.gone()
    p.bye()
    clock[0] += 2
    p.ping()  # a reload comes straight back
    clock[0] += 10
    assert not p.gone()
    p.bye()
    clock[0] += 6
    assert p.gone()  # closed for good
    q = app_module.Presence()
    q.ping()
    clock[0] += 400  # laptop asleep
    q.resumed()
    assert not q.gone()
    clock[0] += 181
    assert q.gone()  # silent (crashed or killed) browser


def test_scan_upload_fetch_and_fit(api):
    from tests.test_scanfit import scanned_room

    body = b"ply\nformat binary_little_endian 1.0\n" + bytes(range(256)) * 8
    req = urllib.request.Request(f"http://127.0.0.1:{api.port}/api/scans?ext=ply", data=body, method="POST",
                                 headers={"Content-Type": "application/octet-stream"})
    with urllib.request.urlopen(req, timeout=30) as r:
        ref = json.loads(r.read())
    assert ref["ext"] == "ply" and ref["bytes"] == len(body)
    status, headers, _ = api("GET", f"/api/scans/{ref['id']}.ply", raw=True)
    assert status == 200
    assert api("GET", "/api/scans/0123456789abcdef0123.ply", raw=True)[0] == 404
    assert api("GET", "/api/scans/..%2F..%2Fsession.ply", raw=True)[0] == 404

    usdz = (pathlib.Path(__file__).parent / "fixtures" / "roomplan.usdz").read_bytes()
    req = urllib.request.Request(f"http://127.0.0.1:{api.port}/api/scans?ext=usdz", data=usdz, method="POST",
                                 headers={"Content-Type": "application/octet-stream"})
    with urllib.request.urlopen(req, timeout=30) as r:
        uref = json.loads(r.read())
    with urllib.request.urlopen(f"http://127.0.0.1:{api.port}/api/scans/{uref['id']}.usdz", timeout=30) as r:
        assert r.read() == usdz and r.headers["Content-Type"].startswith("model/vnd.usdz+zip")

    bad = urllib.request.Request(f"http://127.0.0.1:{api.port}/api/scans?ext=exe", data=body, method="POST")
    with pytest.raises(urllib.error.HTTPError) as e:
        urllib.request.urlopen(bad, timeout=30)
    assert e.value.code == 415

    pts, nrm = scanned_room(n=8000)
    fit = api("POST", "/api/scan-fit", {"points": pts.ravel().tolist(), "normals": nrm.ravel().tolist()})
    assert set(fit["options"]) == {"-y", "+x", "+y", "-x"}
    status, _, text = api("POST", "/api/scan-fit", {"points": [0.0] * 30}, raw=True)
    assert status == 422 and "too few" in text

    project = small()
    project["scan"] = {"id": ref["id"], "ext": "ply", "name": "hall", "yaw_deg": 12.5, "t": [1, 2, 0]}
    sim = api("POST", "/api/simulate", {"project": project, "preview": True})
    assert sim["metrics"]  # a scan reference rides along without affecting the engine


def test_speaker_library_and_user_import(api):
    lib = api("GET", "/api/speakers")
    assert len(lib["library"]) >= 40 and all(t["spec"]["source"].startswith("http") for t in lib["library"])
    csv_text = (
        "brand,model,kind,coverage_h,coverage_v,max_spl,spl_basis,lf_hz,lf_db,source\n"
        "Acme,Box 12,point,90,60,128,continuous,55,-10,https://example.com/a.pdf\n"
        "Acme,Broken,point,90,60,999,peak,55,-10,\n"
        "L,KS28 clash,sub,360,360,140,peak,30,-10,\n"
    )
    req = urllib.request.Request(f"http://127.0.0.1:{api.port}/api/speakers/import", data=csv_text.encode(), method="POST")
    with urllib.request.urlopen(req, timeout=30) as r:
        res = json.loads(r.read())
    assert res["added"] == 2 and res["error_count"] == 1 and res["errors"][0]["row"] == 3
    mine = {t["id"]: t for t in api("GET", "/api/speakers")["user"]}
    assert mine["acme_box_12"]["level_db"] == 134  # continuous + 6 dB
    assert mine["acme_box_12"]["group"] == "My speakers"
    status, _, _ = api("POST", "/api/speakers/user", {"id": "lacoustics_ks28", "brand": "x", "model": "y", "max_spl": 120, "lf_hz": 40}, raw=True)
    assert status == 409
    assert api("DELETE", "/api/speakers/user/acme_box_12")["deleted"] == 1


def test_outdoor_venue_has_no_reverb(api):
    project = small("theater")
    indoor = api("POST", "/api/simulate", {"project": project, "preview": True})["room"]["rt60_mid"]
    project["room"]["kind"] = "outdoor"
    outdoor = api("POST", "/api/simulate", {"project": project, "preview": True})["room"]["rt60_mid"]
    assert outdoor < 0.3 * indoor


def _req(port, method, path, headers, body=b"{}"):
    req = urllib.request.Request(f"http://127.0.0.1:{port}{path}", data=body if method != "GET" else None,
                                 method=method, headers={"Content-Type": "application/json", **headers})
    try:
        with urllib.request.urlopen(req, timeout=30) as r:
            return r.status, dict(r.headers)
    except urllib.error.HTTPError as e:
        return e.code, dict(e.headers)


def test_other_sites_cannot_drive_the_engine(api):
    p = api.port
    assert _req(p, "POST", "/api/bye", {"Origin": "https://evil.example"})[0] == 403
    assert _req(p, "POST", "/api/speakers/import", {"Origin": "http://localhost:1"})[0] == 403
    assert _req(p, "POST", "/api/bye", {"Origin": "null"})[0] == 403
    assert _req(p, "POST", "/api/bye", {"Sec-Fetch-Site": "cross-site"})[0] == 403
    assert _req(p, "GET", "/api/meta", {"Host": f"rebind.example:{p}"})[0] == 403
    assert _req(p, "POST", "/api/ping", {"Origin": f"http://127.0.0.1:{p}"})[0] == 200


def test_security_headers_and_csp_hash_matches_importmap(api):
    import base64
    import hashlib
    import re

    status, headers, html = api("GET", "/", raw=True)
    headers = {k.title(): v for k, v in headers.items()}
    assert status == 200
    assert headers["X-Content-Type-Options"] == "nosniff"
    csp = headers["Content-Security-Policy"]
    assert "unsafe-inline'" not in csp.split("script-src", 1)[1].split(";")[0]
    inline = re.findall(r"<script(?: type=\"importmap\")?>(.*?)</script>", html, flags=re.S)
    assert len(inline) == 1
    # browsers hash the script text after the HTML parser has turned CRLF into LF
    text = inline[0].replace(chr(13) + chr(10), chr(10)).replace(chr(13), chr(10))
    digest = base64.b64encode(hashlib.sha256(text.encode()).digest()).decode()
    assert f"'sha256-{digest}'" in csp


def test_import_over_the_cap_is_refused(api):
    # The server stops reading at the cap, so the client sees either the 413 or the connection being closed.
    try:
        status, _ = _req(api.port, "POST", "/api/speakers/import", {"Content-Type": "text/csv"}, body=b"a" * 5_100_000)
    except (ConnectionResetError, ConnectionAbortedError, urllib.error.URLError):
        status = 413
    assert status == 413
    assert "library" in api("GET", "/api/speakers")  # engine still answering



def test_export_includes_sub_array_offsets(api):
    from acoustic_phase_optimizer.engine import subarray

    p = presets.festival()
    p.speakers = [s for s in p.speakers if s.role != "sub"] + subarray.build(p, "cardioid", "sub_18", 1, 1.0, 30, 6)
    rear = p.speakers[-1]
    rear.delay_ms = 4.0  # an alignment delay on top of the pattern
    out = api("POST", "/api/export", {"project": p.model_dump(), "format": "json"}, raw=True)
    rows = json.loads(out[2])["outputs"]
    r = next(x for x in rows if x["name"] == rear.name)
    assert r["delay_ms"] == pytest.approx(4.0 + rear.array_delay_ms, abs=0.01)
    assert r["polarity"] == "Inverted"


def test_no_api_docs(api):
    status, _, body = api("GET", "/api/docs", raw=True)
    assert status == 404 and "swagger" not in body.lower()


def test_speaker_template_imports_cleanly(api):
    with urllib.request.urlopen(f"http://127.0.0.1:{api.port}/api/speakers/template.csv", timeout=30) as r:
        text = r.read()
    assert b"box_h_mm" in text.splitlines()[0]
    req = urllib.request.Request(f"http://127.0.0.1:{api.port}/api/speakers/import", data=text, method="POST")
    with urllib.request.urlopen(req, timeout=30) as r:
        res = json.loads(r.read())
    assert res["added"] == 2 and res["error_count"] == 0, res
    mine = {t["id"]: t for t in api("GET", "/api/speakers")["user"]}
    assert mine["my_line_10"]["array_box_h"] == 0.29
    for i in ("my_top_12", "my_line_10"):
        api("DELETE", f"/api/speakers/user/{i}")


def test_dwg_goes_through_a_local_converter(api, tmp_path, monkeypatch):
    import sys

    def post(data, headers=None):
        req = urllib.request.Request(f"http://127.0.0.1:{api.port}/api/convert/dwg", data=data, method="POST",
                                     headers={"Content-Type": "application/octet-stream", **(headers or {})})
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                return r.status, r.read()
        except urllib.error.HTTPError as e:
            return e.code, e.read()

    # nothing installed: a clear 501 naming what to install
    monkeypatch.delenv("APO_DWG_CONVERTER", raising=False)
    monkeypatch.setenv("PATH", str(tmp_path))
    monkeypatch.setenv("ProgramFiles", str(tmp_path))
    monkeypatch.setenv("ProgramFiles(x86)", str(tmp_path))
    assert api("GET", "/api/convert/dwg") == {"available": False, "kind": None}
    status, body = post(b"AC1032" + b"\0" * 100)
    assert status == 501 and b"ODA File Converter" in body

    # a converter called exactly as dwg2dxf is: -y -o OUT IN
    stub = pathlib.Path(__file__).parent / "fixtures" / "fake_dwg2dxf.py"
    monkeypatch.setenv("APO_DWG_CONVERTER", json.dumps([sys.executable, str(stub)]))
    monkeypatch.setenv("APO_DWG_CONVERTER_KIND", "libredwg")
    assert api("GET", "/api/convert/dwg")["available"] is True
    status, body = post(b"AC1032" + b"\0" * 100)
    assert status == 200 and b"SECTION" in body and b"3DSOLID" in body
    status, body = post(b"PK not a drawing")
    assert status == 422 and b"not a DWG" in body
    status, body = post(b"AC1032 BROKEN")
    assert status == 422 and b"ERROR 0x800" in body
    status, _ = post(b"AC1032" + b"\0" * 100, {"Origin": "http://evil.example"})
    assert status == 403
