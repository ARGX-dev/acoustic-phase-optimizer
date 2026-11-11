"""End-to-end checks of the web interface in a real browser.

Skipped when Playwright (or a Chromium/Edge it can drive) is not installed.
Set APO_UI_SHOTS to a folder to keep screenshots for a visual check.

The tests share one page and run in file order, like a person using the app.
"""

from __future__ import annotations

import os
import pathlib
import re
import socket
import threading
import time
from pathlib import Path

import pytest

sync_api = pytest.importorskip("playwright.sync_api")

import uvicorn  # noqa: E402

from acoustic_phase_optimizer.server import create_app  # noqa: E402

SHOTS = os.environ.get("APO_UI_SHOTS")


def shot(page, name: str) -> None:
    if SHOTS:
        Path(SHOTS).mkdir(parents=True, exist_ok=True)
        page.screenshot(path=str(Path(SHOTS) / f"{name}.png"))


@pytest.fixture(scope="module")
def ui(tmp_path_factory):
    os.environ["APO_DATA_DIR"] = str(tmp_path_factory.mktemp("apo-ui"))
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    server = uvicorn.Server(uvicorn.Config(create_app(), host="127.0.0.1", port=port, log_level="warning"))
    threading.Thread(target=server.run, daemon=True).start()
    deadline = time.time() + 15
    while not server.started and time.time() < deadline:
        time.sleep(0.05)

    pw = sync_api.sync_playwright().start()
    browser = None
    for kw in ({}, {"channel": "msedge"}, {"channel": "chrome"}):
        try:
            browser = pw.chromium.launch(**kw)
            break
        except Exception:
            continue
    if browser is None:
        pw.stop()
        server.should_exit = True
        pytest.skip("no browser Playwright can drive")
    # The app runs under a strict CSP; Playwright's own wait_for_function needs eval, so the main test
    # context bypasses it. test_strict_csp_blocks_nothing loads the app with the policy enforced.
    ctx = browser.new_context(viewport={"width": 1560, "height": 960}, accept_downloads=True, bypass_csp=True)
    page = ctx.new_page()
    state = {"errors": [], "dialogs": [], "page": page, "ctx": ctx, "url": f"http://127.0.0.1:{port}/", "browser": browser}
    page.on("console", lambda m: state["errors"].append(f"{m.type}: {m.text}") if m.type in ("error", "warning") else None)
    page.on("pageerror", lambda e: state["errors"].append(f"pageerror: {e}"))

    def on_dialog(d):
        state["dialogs"].append(d.message)
        d.accept()

    page.on("dialog", on_dialog)
    page.goto(state["url"])
    page.wait_for_selector(".score .num", timeout=30000)
    yield state
    browser.close()
    pw.stop()
    server.should_exit = True


def textured_glb(path):
    """A 4 x 6 m floor quad with an embedded 2 x 2 PNG texture: the shape of a Polycam GLB, in a few hundred bytes."""
    import json as _json
    import struct
    import zlib

    def png():
        raw = b"".join(b"\x00" + bytes([200, 60, 60, 255, 60, 200, 60, 255]) for _ in range(2))
        def chunk(t, d):
            return struct.pack(">I", len(d)) + t + d + struct.pack(">I", zlib.crc32(t + d) & 0xFFFFFFFF)
        return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", 2, 2, 8, 6, 0, 0, 0))
                + chunk(b"IDAT", zlib.compress(raw)) + chunk(b"IEND", b""))

    pos = struct.pack("<12f", 0, 0, 0, 4, 0, 0, 4, 0, 6, 0, 0, 6)
    uv = struct.pack("<8f", 0, 0, 1, 0, 1, 1, 0, 1)
    idx = struct.pack("<6H", 0, 2, 1, 0, 3, 2) + b"\x00\x00"
    img = png()
    img += b"\x00" * (-len(img) % 4)
    blob = pos + uv + idx + img
    views = [{"buffer": 0, "byteOffset": 0, "byteLength": len(pos)},
             {"buffer": 0, "byteOffset": len(pos), "byteLength": len(uv)},
             {"buffer": 0, "byteOffset": len(pos) + len(uv), "byteLength": 12},
             {"buffer": 0, "byteOffset": len(pos) + len(uv) + len(idx), "byteLength": len(png())}]
    gltf = {
        "asset": {"version": "2.0"}, "scene": 0, "scenes": [{"nodes": [0]}], "nodes": [{"mesh": 0}],
        "meshes": [{"primitives": [{"attributes": {"POSITION": 0, "TEXCOORD_0": 1}, "indices": 2, "material": 0}]}],
        "materials": [{"pbrMetallicRoughness": {"baseColorTexture": {"index": 0}}}],
        "textures": [{"source": 0}], "images": [{"bufferView": 3, "mimeType": "image/png"}],
        "accessors": [
            {"bufferView": 0, "componentType": 5126, "count": 4, "type": "VEC3", "min": [0, 0, 0], "max": [4, 0, 6]},
            {"bufferView": 1, "componentType": 5126, "count": 4, "type": "VEC2"},
            {"bufferView": 2, "componentType": 5123, "count": 6, "type": "SCALAR"}],
        "bufferViews": views, "buffers": [{"byteLength": len(blob)}],
    }
    js = _json.dumps(gltf).encode()
    js += b" " * (-len(js) % 4)
    body = struct.pack("<II", len(js), 0x4E4F534A) + js + struct.pack("<II", len(blob), 0x004E4942) + blob
    path.write_bytes(struct.pack("<III", 0x46546C67, 2, 12 + len(body)) + body)
    return path


def confirm(page) -> bool:
    """Accept the app's own confirmation dialog if one is open. Native dialogs are accepted by the fixture."""
    btn = page.locator(".dialog .btn.primary, .dialog .btn.danger-solid")
    try:
        btn.first.wait_for(timeout=1500)
    except Exception:
        return False
    btn.first.click()
    return True


def delays(page):
    return page.locator(".row-item .meta").all_inner_texts()


def test_boots_with_a_score_and_no_errors(ui):
    page = ui["page"]
    time.sleep(1)
    shot(page, "01-start")
    assert page.locator(".score .num").inner_text().isdigit()
    assert page.locator(".panel-title", has_text="Findings").count() == 1
    assert "[object" not in page.locator("body").inner_text()


def test_example_venue_loads_without_a_discard_prompt(ui):
    page = ui["page"]
    page.get_by_role("button", name="New").click()
    shot(page, "02-new-menu")
    page.get_by_role("menuitem").nth(1).click()
    time.sleep(0.4)
    asked = confirm(page) or ui["dialogs"]
    assert not asked
    page.wait_for_selector(".score .num", timeout=30000)


def test_save_open_and_export(ui):
    page, ctx = ui["page"], ui["ctx"]
    name = page.locator(".venue-name").input_value()
    page.keyboard.press("Control+s")
    page.wait_for_function("document.querySelector('.save-state').textContent === 'Saved'", timeout=10000)
    page.get_by_role("button", name="Open").click()
    time.sleep(0.4)
    shot(page, "03-open-menu")
    assert page.locator(".menu .menu-item", has_text=name).count() == 1
    page.keyboard.press("Escape")
    assert page.locator(".menu").count() == 0

    page.get_by_role("button", name="Export").click()
    box = page.locator(".menu").bounding_box()
    assert box and box["x"] + box["width"] <= 1560 and box["width"] < 400
    with page.expect_download() as dl:
        page.get_by_role("menuitem", name="Spreadsheet (CSV)").click()
    assert dl.value.suggested_filename.endswith(".csv")
    assert len(Path(dl.value.path()).read_text(encoding="utf-8")) > 50

    page.get_by_role("button", name="Export").click()
    with ctx.expect_page() as newp:
        page.get_by_role("menuitem", name="Printable report").click()
    rep = newp.value
    rep.wait_for_load_state()
    assert "<svg" in rep.content()
    rep.close()


def test_drag_delete_and_undo(ui):
    page = ui["page"]
    page.get_by_role("tab", name="Speakers").click()
    page.locator(".row-item").first.click()
    before = page.locator("input[aria-label='From stage']").first.input_value()
    canvas = page.locator("#stage canvas")
    box = canvas.bounding_box()
    hit = None
    for i in range(2, 30):
        for j in range(1, 40):
            x, y = box["x"] + box["width"] * i / 60, box["y"] + box["height"] * j / 40
            page.mouse.move(x, y)
            if page.evaluate("getComputedStyle(document.querySelector('#stage canvas')).cursor") == "grab":
                hit = (x, y)
                break
        if hit:
            break
    assert hit, "no speaker found on the plan"
    page.mouse.down()
    page.mouse.move(hit[0] + 60, hit[1] + 20, steps=8)
    page.mouse.up()
    time.sleep(1)
    shot(page, "04-dragged")
    assert page.locator("input[aria-label='From stage']").first.input_value() != before
    assert page.locator(".save-state").inner_text() == "Edited"

    n0 = page.locator(".row-item").count()
    canvas.focus()
    page.keyboard.press("Delete")
    time.sleep(0.5)
    assert page.locator(".row-item").count() == n0 - 1
    page.keyboard.press("Control+z")
    time.sleep(0.5)
    assert page.locator(".row-item").count() == n0


def test_quick_align_undo_and_restore_after_reload(ui):
    page = ui["page"]
    d0 = delays(page)
    page.get_by_role("button", name="Quick align").click()
    page.wait_for_selector(".changes", timeout=30000)
    time.sleep(0.8)
    assert delays(page) != d0
    page.locator("body").click(position={"x": 700, "y": 20})
    page.keyboard.press("Control+z")
    time.sleep(0.8)
    assert page.locator(".changes").count() == 0 and delays(page) == d0

    page.get_by_role("button", name="Quick align").click()
    page.wait_for_selector(".changes", timeout=30000)
    page.wait_for_timeout(2000)  # let the autosave land
    page.reload()
    page.wait_for_selector(".score .num", timeout=30000)
    time.sleep(2)
    shot(page, "05-reloaded")
    assert page.locator(".changes").count() == 1
    assert page.locator(".save-state").inner_text() == "Edited"
    page.get_by_role("tab", name="Speakers").click()
    page.get_by_role("button", name="Revert").click()
    time.sleep(1.2)
    assert page.locator(".changes").count() == 0 and delays(page) == d0


def test_theme_switch_and_chart_modes(ui):
    page = ui["page"]
    for mode in ["Phase", "Arrivals", "Magnitude"]:
        page.locator("#chart-toolbar .seg button", has_text=mode).click()
        time.sleep(0.3)
    page.get_by_role("button", name="Switch light or dark theme").click()
    time.sleep(0.6)
    shot(page, "06-light")
    assert page.evaluate("document.documentElement.dataset.theme") == "light"
    page.get_by_role("button", name="Switch light or dark theme").click()


def test_map_picker_and_3d_view(ui):
    page = ui["page"]
    page.locator("#stage-toolbar select").first.select_option("clarity")
    time.sleep(0.6)
    assert "Clarity" in page.locator("#stage-foot").inner_text() or page.locator("#stage-foot .scale").count() == 1
    page.locator("#stage-toolbar select").first.select_option("direct")
    page.locator("#stage-toolbar .seg button", has_text="3D").click()
    page.wait_for_selector(".view3d canvas", timeout=15000)
    time.sleep(2)
    shot(page, "07-3d")
    png = page.locator(".view3d canvas").screenshot()
    assert len(png) > 40_000, "3D canvas looks blank (WebGL did not render)"
    assert page.locator(".label3d").count() >= len(page.locator(".row-item").all()) or page.locator(".label3d").count() > 2
    page.locator(".view-buttons button", has_text="Top").click()
    time.sleep(1)
    shot(page, "08-3d-top")
    page.locator("#stage-toolbar .seg button", has_text="Plan").click()
    time.sleep(0.5)
    assert page.locator("#stage canvas.plan").is_visible()


def room_obj(path: Path, w=12.0, l=20.0, h=5.0, yaw_deg=30.0) -> Path:
    """An OBJ 'scan' of a plain room: six inward faces, rotated, y-up like a phone app export."""
    import math

    a = math.radians(yaw_deg)
    corners = [(x, y, z) for z in (0, h) for y in (0, l) for x in (0, w)]
    def place(p):  # rotate about the vertical, then convert z-up -> y-up (x, z, -y)
        x, y, z = p
        xr, yr = x * math.cos(a) - y * math.sin(a), x * math.sin(a) + y * math.cos(a)
        return xr + 2, z, -(yr - 3)
    verts = [("v {:.4f} {:.4f} {:.4f}".format(*place(c))) for c in corners]
    quads = [(1, 2, 4, 3), (5, 7, 8, 6), (1, 5, 6, 2), (3, 4, 8, 7), (1, 3, 7, 5), (2, 6, 8, 4)]
    faces = [line for a_, b, c, d in quads for line in (f"f {a_} {b} {c}", f"f {a_} {c} {d}")]
    path.write_text("\n".join(verts + faces) + "\n")
    return path


def room_3ds(w=12.0, l=20.0, h=5.0) -> bytes:
    """A 3DS room (six faces as 12 triangles) in millimetres, as MVR requires for 3DS."""
    import struct

    def chunk(cid, data):
        return struct.pack("<HI", cid, 6 + len(data)) + data

    corners = [(x * 1000, y * 1000, z * 1000) for z in (0, h) for y in (0, l) for x in (0, w)]
    quads = [(0, 1, 3, 2), (4, 6, 7, 5), (0, 4, 5, 1), (2, 3, 7, 6), (0, 2, 6, 4), (1, 5, 7, 3)]
    tris = [t for a, b, c, d in quads for t in ((a, b, c), (a, c, d))]
    verts = struct.pack("<H", len(corners)) + b"".join(struct.pack("<3f", *c) for c in corners)
    faces = struct.pack("<H", len(tris)) + b"".join(struct.pack("<4H", *t, 0) for t in tris)
    mesh = chunk(0x4100, chunk(0x4110, verts) + chunk(0x4120, faces))
    return chunk(0x4D4D, chunk(0x0002, struct.pack("<I", 3)) + chunk(0x3D3D, chunk(0x4000, b"room\x00" + mesh)))


def venue_mvr(path, tmp):
    """An MVR as Vectorworks writes one: a 3DS room inside a group, two GLB props through a symbol, a fixture."""
    import math
    import zipfile

    a = math.radians(30)
    c, s_ = math.cos(a), math.sin(a)
    rot = f"{{{c:.6f},{s_:.6f},0}}{{{-s_:.6f},{c:.6f},0}}{{0,0,1}}{{2000,3000,0}}"
    xml = f"""<?xml version="1.0" encoding="UTF-8" standalone="no" ?>
<GeneralSceneDescription verMajor="1" verMinor="6" provider="test">
  <Scene>
    <AUXData>
      <Symdef name="riser" uuid="S1"><ChildList><Geometry3D fileName="prop.glb"/></ChildList></Symdef>
    </AUXData>
    <Layers>
      <Layer name="Venue" uuid="L1">
        <ChildList>
          <GroupObject name="Building" uuid="G1">
            <Matrix>{{1,0,0}}{{0,1,0}}{{0,0,1}}{{1000,0,0}}</Matrix>
            <ChildList>
              <SceneObject name="Hall" uuid="O1">
                <Matrix>{rot}</Matrix>
                <Geometries><Geometry3D fileName="hall"/></Geometries>
              </SceneObject>
            </ChildList>
          </GroupObject>
          <SceneObject name="Riser A" uuid="O2">
            <Geometries><Symbol uuid="Y1" symdef="S1"><Matrix>{{1,0,0}}{{0,1,0}}{{0,0,1}}{{3000,4000,0}}</Matrix></Symbol></Geometries>
          </SceneObject>
          <SceneObject name="Riser B" uuid="O3">
            <Geometries><Symbol uuid="Y2" symdef="S1"><Matrix>{{1,0,0}}{{0,1,0}}{{0,0,1}}{{6000,9000,0}}</Matrix></Symbol></Geometries>
          </SceneObject>
          <Fixture name="Spot 1" uuid="F1"><GDTFSpec>Spot.gdtf</GDTFSpec><GDTFMode>Std</GDTFMode>
            <Matrix>{{1,0,0}}{{0,1,0}}{{0,0,1}}{{5000,5000,4000}}</Matrix></Fixture>
        </ChildList>
      </Layer>
    </Layers>
  </Scene>
</GeneralSceneDescription>"""
    glb = textured_glb(tmp / "prop.glb").read_bytes()
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("GeneralSceneDescription.xml", xml)
        z.writestr("hall.3ds", room_3ds())
        z.writestr("prop.glb", glb, compress_type=zipfile.ZIP_STORED)
    return path


def test_import_mvr_venue(ui, tmp_path):
    import base64

    page = ui["page"]
    path = venue_mvr(tmp_path / "venue.mvr", tmp_path)
    # the reader itself: metres, Z up, transforms compounded, symbols instanced, fixtures counted
    info = page.evaluate("""async (b64) => {
        const bytes = Uint8Array.from(atob(b64), (c) => c.charCodeAt(0));
        const THREE = await import('three');
        const g = await (await import('/js/mvr.js')).parseMvr(bytes.buffer);
        const hb = new THREE.Box3().setFromObject(g.children[0].children[0]);
        const hall = [...hb.min.toArray(), ...hb.max.toArray()].map((v) => +v.toFixed(3));
        const sizes = [];
        g.children[0].children.forEach((o) => { const b = new THREE.Box3().setFromObject(o); const s = b.getSize(new THREE.Vector3()); sizes.push([s.x, s.y, s.z].map((v) => +v.toFixed(2))); });
        let bad = null;
        try { await (await import('/js/mvr.js')).parseMvr(new Uint8Array(300).buffer); } catch (e) { bad = e.message; }
        return { mvr: g.userData.mvr, sizes, bad, hall };
    }""", base64.b64encode(path.read_bytes()).decode())
    assert info["mvr"]["instances"] == 3 and info["mvr"]["fixtures"] == 1 and info["mvr"]["missing"] == []
    props = sorted(info["sizes"])[:2]
    assert all(sorted(p_)[1:] == pytest.approx([4.0, 6.0], abs=0.01) for p_ in props), info  # a 4 x 6 m glTF quad stays 4 x 6 m
    assert "not an MVR file" in info["bad"]
    # Hall corners: x' = x cos30 - y sin30 + 2 + 1 (group), y' = x sin30 + y cos30 + 3. A transposed rotation or
    # the parent applied inside the child's frame would both move these.
    assert info["hall"] == pytest.approx([-7.0, 3.0, 0.0, 3 + 12 * 0.8660254, 3 + 6 + 20 * 0.8660254, 5.0], abs=0.02), info["hall"]
    # and through the import dialog: the room is measured from the 3DS hall
    page.get_by_role("tab", name="Room").click()
    with page.expect_file_chooser() as fc:
        page.get_by_role("button", name=re.compile("Import a 3D scan…|Replace…")).click()
    fc.value.set_files(str(path))
    page.wait_for_selector(".dialog .scan-preview", timeout=20000)
    page.wait_for_function("document.querySelector('.dialog [aria-label=\"Stage to back\"]').value !== ''", timeout=20000)
    dims = sorted(float(page.locator(f".dialog [aria-label='{k}']").input_value()) for k in ("Stage to back", "Width"))
    assert dims == pytest.approx([12.0, 20.0], abs=0.4)
    assert float(page.locator(".dialog [aria-label='Height']").input_value()) == pytest.approx(5.0, abs=0.2)
    shot(page, "27-mvr-dialog")
    page.locator(".dialog .btn.primary").click()
    page.wait_for_selector(".dialog", state="detached", timeout=20000)
    page.wait_for_selector(".view3d canvas", timeout=15000)
    page.get_by_role("tab", name="Room").click()
    assert page.locator(".scan-card .name").inner_text() == "venue"
    page.locator("body").click(position={"x": 700, "y": 20})
    page.keyboard.press("Control+z")
    page.wait_for_function("!document.querySelector('.scan-card')", timeout=5000)
    page.keyboard.press("Control+y")
    page.wait_for_selector(".scan-card", timeout=5000)
    time.sleep(1.5)  # autosave
    page.reload()
    page.wait_for_selector(".view3d canvas", timeout=20000)
    time.sleep(2.5)  # the stored .mvr is fetched and parsed again
    shot(page, "28-mvr-3d-after-reload")
    page.get_by_role("tab", name="Room").click()
    assert page.locator(".scan-card .name").inner_text() == "venue"
    page.get_by_role("button", name="Remove scan").click()
    page.locator("#stage-toolbar .seg button", has_text="Plan").click()
    # an .mvr that Windows unpacked into a folder
    import zipfile
    folder = tmp_path / "unpacked venue.mvr"
    zipfile.ZipFile(path).extractall(folder)
    with page.expect_file_chooser() as fc:
        page.get_by_role("button", name="Import a folder (unpacked MVR or glTF)…").click()
    fc.value.set_files(str(folder))
    page.wait_for_selector(".dialog .scan-preview", timeout=20000)
    page.wait_for_function("document.querySelector('.dialog [aria-label=\"Stage to back\"]').value !== ''", timeout=20000)
    dims = sorted(float(page.locator(f".dialog [aria-label='{k}']").input_value()) for k in ("Stage to back", "Width"))
    assert dims == pytest.approx([12.0, 20.0], abs=0.4)
    page.locator(".dialog .btn", has_text="Cancel").click()
    page.wait_for_selector(".dialog", state="detached", timeout=10000)


def test_stl_and_collada_readers(ui):
    import base64
    import struct

    tri = struct.pack("<12f", 0, 0, 1, 0, 0, 0, 3, 0, 0, 0, 4, 0) + b"\x00\x00"
    stl = b"\x00" * 80 + struct.pack("<I", 1) + tri
    dae = """<?xml version="1.0"?><COLLADA xmlns="http://www.collada.org/2005/11/COLLADASchema" version="1.4.1">
<asset><unit meter="0.01" name="centimeter"/><up_axis>Z_UP</up_axis></asset>
<library_geometries><geometry id="g"><mesh>
<source id="p"><float_array id="pa" count="9">0 0 0 300 0 0 0 400 0</float_array>
<technique_common><accessor source="#pa" count="3" stride="3"><param name="X" type="float"/><param name="Y" type="float"/><param name="Z" type="float"/></accessor></technique_common></source>
<vertices id="v"><input semantic="POSITION" source="#p"/></vertices>
<triangles count="1"><input semantic="VERTEX" source="#v" offset="0"/><p>0 1 2</p></triangles>
</mesh></geometry></library_geometries>
<library_visual_scenes><visual_scene id="s"><node id="n"><instance_geometry url="#g"/></node></visual_scene></library_visual_scenes>
<scene><instance_visual_scene url="#s"/></scene></COLLADA>"""
    fbx = """; FBX 7.3.0 project file
FBXHeaderExtension:  {
	FBXHeaderVersion: 1003
	FBXVersion: 7300
}
GlobalSettings:  {
	Version: 1000
	Properties70:  {
		P: "UnitScaleFactor", "double", "Number", "",100
	}
}
Objects:  {
	Geometry: 1000, "Geometry::tri", "Mesh" {
		Vertices: *9 {
			a: 0,0,0,3,0,0,0,4,0
		}
		PolygonVertexIndex: *3 {
			a: 0,1,-3
		}
		GeometryVersion: 124
	}
	Model: 2000, "Model::tri", "Mesh" {
		Version: 232
		Properties70:  {
		}
	}
}
Connections:  {
	C: "OO",1000,2000
	C: "OO",2000,0
}
"""
    sizes = ui["page"].evaluate("""async ([stl, dae, fbx, tds]) => {
        const THREE = await import('three');
        const { parseScan } = await import('/js/scan.js');
        const buf = (b64) => Uint8Array.from(atob(b64), (c) => c.charCodeAt(0)).buffer;
        const size = (o) => { o.updateMatrixWorld(true); const s = new THREE.Box3().setFromObject(o).getSize(new THREE.Vector3()); return [s.x, s.y, s.z].map((v) => +v.toFixed(3)); };
        const f = await parseScan(buf(fbx), 'fbx');
        return [size(await parseScan(buf(stl), 'stl')), size(await parseScan(buf(dae), 'dae')), size(f), f.userData.unitScaleFactor,
                size(await parseScan(buf(tds), '3ds'))];
    }""", [base64.b64encode(stl).decode(), base64.b64encode(dae.encode()).decode(), base64.b64encode(fbx.encode()).decode(),
           base64.b64encode(room_3ds()).decode()])
    assert sizes[0] == [3, 4, 0]  # STL as written (units chosen in the dialog)
    assert sizes[1] == pytest.approx([3, 0, 4], abs=0.01)  # COLLADA: centimetres to metres, Z up turned to Y up
    assert sizes[2] == pytest.approx([3, 4, 0], abs=0.01) and sizes[3] == 100  # FBX: geometry, and its unit (cm per unit)
    assert sorted(sizes[4]) == pytest.approx([5000, 12000, 20000], abs=1)  # a standalone 3DS, in its own units (mm here)


def test_import_scan_sets_the_room(ui, tmp_path):
    page = ui["page"]
    page.get_by_role("tab", name="Room").click()
    with page.expect_file_chooser() as fc:
        page.get_by_role("button", name="Import a 3D scan…").click()
    fc.value.set_files(str(room_obj(tmp_path / "hall.obj")))
    page.wait_for_selector(".dialog .scan-preview", timeout=20000)
    page.wait_for_function("document.querySelector('.dialog [aria-label=\"Stage to back\"]').value !== ''", timeout=20000)
    time.sleep(0.5)
    shot(page, "09-scan-dialog")
    dims = sorted(float(page.locator(f".dialog [aria-label='{k}']").input_value()) for k in ("Stage to back", "Width"))
    assert dims == pytest.approx([12.0, 20.0], abs=0.4)
    assert float(page.locator(".dialog [aria-label='Height']").input_value()) == pytest.approx(5.0, abs=0.2)
    page.locator(".dialog .btn.primary").click()
    page.wait_for_selector(".dialog", state="detached", timeout=20000)
    page.wait_for_selector(".view3d canvas", timeout=15000)
    time.sleep(2.5)
    shot(page, "10-scan-3d")
    page.locator(".view-buttons button", has_text="3D").click()
    time.sleep(1.2)
    shot(page, "11-scan-orbit")
    page.get_by_role("tab", name="Room").click()
    assert page.locator(".scan-card .name").inner_text() == "hall"
    length = float(page.locator("input[aria-label='Stage to back']").first.input_value())
    assert length == pytest.approx(20.0, abs=0.4)
    # undo removes the scan and restores the room; redo brings it back
    page.locator("body").click(position={"x": 700, "y": 20})
    page.keyboard.press("Control+z")
    time.sleep(0.8)
    assert page.locator(".scan-card").count() == 0
    page.keyboard.press("Control+y")
    time.sleep(0.8)
    assert page.locator(".scan-card").count() == 1
    page.locator("#stage-toolbar .seg button", has_text="Plan").click()


def broken_room_obj(path: Path, w=12.0, l=20.0, h=5.0, step=0.125) -> Path:
    """A finely meshed room with typical export damage: the whole back wall wound the wrong way, one missing triangle
    (a crack, narrower than the repair limit), one duplicated face and one face with a NaN corner."""
    verts, faces = [], []

    def grid(origin, du, dv, nu, nv, flip=False, skip=None):
        base = len(verts)
        for j in range(nv + 1):
            for i in range(nu + 1):
                verts.append(tuple(origin[k] + du[k] * i * step + dv[k] * j * step for k in range(3)))
        vid = lambda i, j: base + j * (nu + 1) + i + 1  # noqa: E731  (OBJ counts from 1)
        for j in range(nv):
            for i in range(nu):
                a, b, c, d = vid(i, j), vid(i + 1, j), vid(i + 1, j + 1), vid(i, j + 1)
                for t, tri in enumerate(((a, b, c), (a, c, d))):
                    if skip == (i, j, t):
                        continue
                    faces.append(tri[::-1] if flip else tri)

    nw, nl, nh = round(w / step), round(l / step), round(h / step)
    grid((0, 0, 0), (0, 1, 0), (1, 0, 0), nl, nw, skip=(40, 30, 0))  # floor (facing down)
    grid((0, 0, h), (1, 0, 0), (0, 1, 0), nw, nl)  # ceiling
    grid((0, 0, 0), (1, 0, 0), (0, 0, 1), nw, nh)  # front
    grid((0, l, 0), (0, 0, 1), (1, 0, 0), nh, nw)  # back (facing out like the rest), turned the wrong way round below
    back = (len(faces) - nw * nh * 2, len(faces))
    grid((0, 0, 0), (0, 0, 1), (0, 1, 0), nh, nl)  # left
    grid((w, 0, 0), (0, 1, 0), (0, 0, 1), nl, nh)  # right
    faces[back[0]:back[1]] = [f[::-1] for f in faces[back[0]:back[1]]]
    faces.append(faces[0])  # a duplicate
    verts.append((float("nan"),) * 3)
    faces.append((1, 2, len(verts)))  # a NaN corner
    lines = [f"v {x:.4f} {z:.4f} {-y:.4f}" for x, y, z in verts]  # y-up
    lines += [f"f {a} {b} {c}" for a, b, c in faces]
    path.write_text("\n".join(lines) + "\n")
    return path, nw * nh * 2


def test_import_repairs_a_broken_model(ui, tmp_path):
    page = ui["page"]
    page.get_by_role("tab", name="Room").click()
    obj, back_faces = broken_room_obj(tmp_path / "broken_hall.obj")
    with page.expect_file_chooser() as fc:
        page.get_by_role("button", name=re.compile("Import a 3D scan…|Replace…")).click()
    fc.value.set_files(str(obj))
    page.wait_for_selector(".dialog [data-testid='repair-report']", timeout=60000)
    page.wait_for_function("document.querySelector('.dialog [aria-label=\"Stage to back\"]').value !== ''", timeout=20000)
    shot(page, "12-repair-report")
    text = page.locator(".dialog [data-testid='repair-report']").inner_text()
    assert f"{back_faces:,} faces turned to match their neighbours" in text
    assert "1 small hole closed" in text
    assert "1 duplicate face removed" in text
    assert "1 broken face removed" in text
    dims = sorted(float(page.locator(f".dialog [aria-label='{k}']").input_value()) for k in ("Stage to back", "Width"))
    assert dims == pytest.approx([12.0, 20.0], abs=0.4)
    # switching back to the file as it is re-measures from the unrepaired model
    toggle = page.get_by_role("checkbox", name="Use the repaired model")
    toggle.uncheck()
    page.wait_for_function("document.querySelector('.dialog [aria-label=\"Stage to back\"]').value !== ''", timeout=20000)
    assert float(page.locator(".dialog [aria-label='Height']").input_value()) == pytest.approx(5.0, abs=0.2)
    toggle.check()
    page.locator(".dialog .btn", has_text="Cancel").click()
    page.wait_for_selector(".dialog", state="detached", timeout=10000)


def test_import_reports_a_clean_model(ui, tmp_path):
    page = ui["page"]
    page.get_by_role("tab", name="Room").click()
    with page.expect_file_chooser() as fc:
        page.get_by_role("button", name=re.compile("Import a 3D scan…|Replace…")).click()
    fc.value.set_files(str(room_obj(tmp_path / "clean.obj")))
    page.wait_for_selector(".dialog [data-testid='repair-report']", timeout=20000)
    assert "The model is clean" in page.locator(".dialog [data-testid='repair-report']").inner_text()
    assert page.get_by_role("checkbox", name="Use the repaired model").count() == 0
    page.locator(".dialog .btn", has_text="Cancel").click()
    page.wait_for_selector(".dialog", state="detached", timeout=10000)


def test_speaker_library_search_add_and_change_model(ui):
    page = ui["page"]
    page.get_by_role("tab", name="Speakers").click()
    before = page.locator(".row-list .row-item").count()
    page.get_by_role("button", name="Add a speaker…").click()
    search = page.get_by_role("combobox", name="Search loudspeakers")
    search.fill("ks28")
    items = page.locator(".picker-item")
    assert items.count() == 1 and "KS28" in items.first.inner_text()
    search.press("Enter")
    time.sleep(0.6)
    assert page.locator(".row-list .row-item").count() == before + 1
    assert "datasheet" in page.locator(".tab-body").inner_text().lower()
    assert page.locator("a.link", has_text="Manufacturer source").get_attribute("href").startswith("https://")
    # change the new speaker's model through the dialog, filtering by kind
    page.get_by_role("button", name="Change loudspeaker model").click()
    page.locator(".dialog .chip", has_text="Subwoofer").click()
    page.locator(".dialog input[type=search]").fill("srx")
    shot(page, "12-speaker-picker")
    page.locator(".dialog .picker-item", has_text="SRX828SP").click()
    page.wait_for_selector(".dialog", state="detached")
    assert "SRX828SP" in page.locator(".type-btn").inner_text()
    level = float(page.locator("input[aria-label='Max SPL at 1 m']").input_value())
    assert level == 141
    page.keyboard.press("Escape")
    page.locator("body").click(position={"x": 700, "y": 20})
    page.keyboard.press("Control+z")
    page.keyboard.press("Control+z")
    time.sleep(0.6)
    assert page.locator(".row-list .row-item").count() == before


def test_plan_zoom_and_pan(ui):
    page = ui["page"]
    canvas = page.locator("canvas.plan")
    box = canvas.bounding_box()
    cx, cy = box["x"] + box["width"] / 2, box["y"] + box["height"] / 2
    before = canvas.screenshot()
    page.mouse.move(cx, cy)
    page.mouse.wheel(0, -600)
    time.sleep(0.4)
    assert page.locator(".fit-btn").is_visible()
    page.mouse.down(button="right")
    page.mouse.move(cx + 80, cy + 30, steps=4)
    page.mouse.up(button="right")
    time.sleep(0.3)
    shot(page, "13-plan-zoomed")
    assert canvas.screenshot() != before
    page.locator(".fit-btn").click()
    time.sleep(0.3)
    assert not page.locator(".fit-btn").is_visible()


def test_venue_type_materials_and_air(ui):
    page = ui["page"]
    page.get_by_role("tab", name="Room").click()
    rt = lambda: page.locator(".stat-grid div", has_text="Reverb time").locator("b").inner_text()  # noqa: E731
    page.wait_for_function("document.querySelector('.stat-grid b').textContent.includes(' s')", timeout=15000)
    indoor = float(rt().split()[0])
    assert indoor > 0.3
    page.get_by_role("radio", name="Open air").click()
    page.wait_for_selector(".stat-grid b:has-text('None (open air)')", timeout=15000)
    assert page.locator(".stat-grid", has_text="Schroeder").count() == 0  # room modes do not apply outdoors
    assert page.locator(".open-surface", has_text="Open sky").count() == 1
    page.get_by_role("radio", name="Indoor room").click()
    page.get_by_role("button", name="Add a material").click()
    assert page.locator(".mat-row").count() == 1
    page.locator("input[aria-label='Altitude']").fill("1600")
    page.locator("input[aria-label='Altitude']").press("Enter")
    time.sleep(0.4)
    assert float(page.locator("input[aria-label='Pressure']").input_value()) == pytest.approx(83.5, abs=0.6)
    page.locator("input[aria-label='Audience occupancy']").fill("1")
    time.sleep(1.0)
    shot(page, "14-room-tab")
    for _ in range(5):
        page.locator("body").click(position={"x": 700, "y": 20})
        page.keyboard.press("Control+z")
    time.sleep(0.6)


def test_line_array_box_count_changes_level(ui):
    page = ui["page"]
    page.get_by_role("tab", name="Speakers").click()
    page.get_by_role("button", name="Add a speaker…").click()
    page.get_by_role("combobox", name="Search loudspeakers").fill("kara")
    page.get_by_role("combobox", name="Search loudspeakers").press("Enter")
    page.wait_for_selector("input[aria-label='Boxes']")
    level8 = float(page.locator("input[aria-label='Max SPL at 1 m']").input_value())
    page.locator("input[aria-label='Boxes']").fill("12")
    page.locator("input[aria-label='Boxes']").press("Enter")
    time.sleep(0.5)
    level12 = float(page.locator("input[aria-label='Max SPL at 1 m']").input_value())
    assert level12 == pytest.approx(level8 + 10 * __import__("math").log10(12 / 8), abs=0.6)
    shot(page, "15-line-array")
    page.locator("body").click(position={"x": 700, "y": 20})
    page.keyboard.press("Control+z")
    page.keyboard.press("Control+z")
    time.sleep(0.5)


def test_keyboard_nudge_scrub_and_shortcuts(ui):
    page = ui["page"]
    page.get_by_role("tab", name="Speakers").click()
    page.locator(".row-list .row-item").first.click()
    field = page.locator("input[aria-label='From stage']")
    y0 = float(field.input_value())
    page.locator("body").click(position={"x": 700, "y": 20})
    page.locator(".row-list .row-item").first.click()
    page.evaluate("document.activeElement.blur()")
    for _ in range(3):
        page.keyboard.press("Shift+ArrowRight")
    time.sleep(0.9)
    assert float(field.input_value()) == pytest.approx(y0 + 3, abs=0.05)
    page.keyboard.press("Control+z")  # one undo for the whole burst
    time.sleep(0.5)
    assert float(field.input_value()) == pytest.approx(y0, abs=0.05)
    # scrub the Height label
    label = page.locator("label.field", has=page.locator("input[aria-label='Height']")).locator(".scrub").first
    label.scroll_into_view_if_needed()
    h0 = float(page.locator("input[aria-label='Height']").first.input_value())
    box = label.bounding_box()
    page.mouse.move(box["x"] + 5, box["y"] + box["height"] / 2)
    page.mouse.down()
    page.mouse.move(box["x"] + 45, box["y"] + box["height"] / 2, steps=5)
    page.mouse.up()
    time.sleep(0.4)
    assert float(page.locator("input[aria-label='Height']").first.input_value()) > h0
    page.keyboard.press("Control+z")
    page.evaluate("document.activeElement.blur()")
    page.keyboard.press("?")
    page.wait_for_selector(".dialog dl.shortcuts")
    shot(page, "16-shortcuts")
    page.keyboard.press("Escape")
    page.wait_for_selector(".dialog", state="detached")


def test_drag_a_speaker_in_3d(ui):
    page = ui["page"]
    page.get_by_role("tab", name="Speakers").click()
    page.locator("#stage-toolbar .seg button", has_text="3D").click()
    page.wait_for_selector(".view3d canvas", timeout=15000)
    page.locator(".view-buttons button", has_text="Top").click()
    time.sleep(1.5)
    label = page.locator(".label3d", has_text="Sub C").first
    box = label.bounding_box()
    x, y = box["x"] + box["width"] / 2, box["y"] + box["height"] + 4
    before = page.evaluate("() => { const s = window.__apo?.state?.project.speakers.find(q => q.name === 'Sub C'); return s && [s.x, s.y]; }")
    page.mouse.move(x, y)
    page.mouse.down()
    page.mouse.move(x + 60, y + 30, steps=8)
    page.mouse.up()
    time.sleep(0.6)
    page.get_by_role("tab", name="Speakers").click()
    assert "Sub C" in page.locator(".row-list .row-item.selected").inner_text()
    moved = (float(page.locator("input[aria-label='From stage']").input_value()), float(page.locator("input[aria-label='Across']").input_value()))
    shot(page, "17-3d-drag")
    page.keyboard.press("Control+z")
    time.sleep(0.6)
    back = (float(page.locator("input[aria-label='From stage']").input_value()), float(page.locator("input[aria-label='Across']").input_value()))
    assert moved != back, (moved, back, before)
    page.locator("#stage-toolbar .seg button", has_text="Plan").click()


def test_section_view_and_tilt_by_handle(ui):
    page = ui["page"]
    page.locator("#stage-toolbar .seg button", has_text="Section").click()
    page.wait_for_selector(".section-view canvas")
    page.wait_for_function("document.querySelector('#stage-foot .scale') !== null", timeout=20000)
    time.sleep(0.5)
    shot(page, "18-section")
    page.locator(".section-buttons button", has_text="Across the room").click()
    page.wait_for_function("!document.querySelector('#stage-foot .badge')", timeout=20000)
    time.sleep(2.5)  # the theater slice takes about a second
    shot(page, "19-section-across")
    page.locator(".section-buttons button", has_text="Away from stage").click()
    time.sleep(1.0)
    # Select Hang L, then drag its aim handle upward: the tilt must change and undo in one step.
    page.get_by_role("tab", name="Speakers").click()
    page.locator(".row-list .row-item", has_text="Hang L").click()
    pitch0 = float(page.locator("input[aria-label='Tilt']").input_value())
    hx, hy = page.evaluate("""() => window.__apoSectionHandle?.('Hang L')""") or (None, None)
    assert hx is not None
    box = page.locator(".section-view canvas").bounding_box()
    page.mouse.move(box["x"] + hx, box["y"] + hy)
    page.mouse.down()
    page.mouse.move(box["x"] + hx, box["y"] + hy - 40, steps=6)
    page.mouse.up()
    time.sleep(0.6)
    pitch1 = float(page.locator("input[aria-label='Tilt']").input_value())
    assert pitch1 > pitch0 + 5, (pitch0, pitch1)
    page.locator("body").click(position={"x": 700, "y": 20})
    page.keyboard.press("Control+z")
    time.sleep(0.5)
    assert float(page.locator("input[aria-label='Tilt']").input_value()) == pytest.approx(pitch0)
    page.locator("#stage-toolbar .seg button", has_text="Plan").click()


def test_sub_array_builder(ui):
    page = ui["page"]
    page.get_by_role("tab", name="Speakers").click()
    n0 = page.locator(".row-list .row-item").count()
    page.get_by_role("button", name="Sub array…").click()
    page.wait_for_selector(".dialog .sub-sketch")
    page.get_by_role("radio", name="Cardioid").click()
    assert page.locator(".sub-sketch .box.inv").count() == 2  # two pairs, two inverted rear boxes
    shot(page, "20-sub-builder")
    page.get_by_role("button", name="Add the array").click()
    page.wait_for_selector(".dialog", state="detached")
    page.wait_for_function(f"document.querySelectorAll('.row-list .row-item').length !== {n0}", timeout=10000)
    names = page.locator(".row-list .row-item").all_inner_texts()
    assert sum("rear" in t for t in names) == 2
    assert page.locator(".group", has_text="Sub array").count() >= 1
    page.locator("#stage-toolbar .seg button", has_text="Section").click()
    page.wait_for_function("document.querySelector('#stage-foot .scale') !== null", timeout=20000)
    time.sleep(2.0)
    shot(page, "21-cardioid-section")
    page.locator("#stage-toolbar .seg button", has_text="Plan").click()
    page.locator("body").click(position={"x": 700, "y": 20})
    page.keyboard.press("Control+z")
    page.wait_for_function(f"document.querySelectorAll('.row-list .row-item').length === {n0}", timeout=10000)


def test_a_weighting(ui):
    page = ui["page"]
    page.locator("select[aria-label='Frequency weighting']").select_option("A")
    page.wait_for_function("document.querySelector('#stage-foot .scale').textContent.includes('dB(A)')", timeout=20000)
    page.wait_for_function("[...document.querySelectorAll('.metric')].some(m => m.textContent.includes('Average level') && m.textContent.includes('dB(A)'))", timeout=20000)
    page.locator("body").click(position={"x": 700, "y": 20})
    page.keyboard.press("Control+z")
    page.wait_for_function("!document.querySelector('#stage-foot .scale').textContent.includes('dB(A)')", timeout=20000)


def test_import_measurement_overlay(ui, tmp_path):
    import math

    page = ui["page"]
    rows = ["* Measurement data measured by REW V5.31", "* Source: test", "Freq(Hz) SPL(dB) Phase(degrees)"]
    f = 15.0
    while f < 22000:
        rows.append(f"{f:.3f} {100 - 6 * math.log2(max(f, 60) / 1000) / 3:.3f} {(-f * 0.36) % 360 - 180:.2f}")
        f *= 2 ** (1 / 96)
    path = tmp_path / "FOH centre.txt"
    path.write_text("\n".join(rows), encoding="utf-8")
    page.get_by_role("tab", name="Mics").click()
    with page.expect_file_chooser() as fc:
        page.get_by_role("button", name="Import a measurement…").click()
    fc.value.set_files(str(path))
    page.wait_for_selector(".meas-row")
    page.wait_for_selector(".chart-legend .chip:has-text('FOH centre (measured)')")
    assert page.locator(".chart-legend .chip .key.dashed").count() == 1
    shot(page, "22-measurement")
    # the parser also reads semicolon CSV with decimal commas (Smaart / European Excel)
    n = page.evaluate("""async () => {
        const m = await import('/js/measure.js');
        const text = 'Frequency;Magnitude;Phase\\n' + Array.from({length: 20}, (_, i) => `${(100 * 1.2 ** i).toFixed(1).replace('.', ',')};${(80.5).toString().replace('.', ',')};10`).join('\\n');
        const r = m.parseResponse(text);
        return [r.freqs.length, r.spl[0], r.phase[0]];
    }""")
    assert n == [20, 80.5, 10]
    page.get_by_role("button", name="Delete FOH centre").click()
    page.wait_for_selector(".meas-row", state="detached")


def test_sti_map_and_metric(ui):
    page = ui["page"]
    v = page.locator(".metric", has_text="Speech intelligibility").locator(".v").inner_text()
    assert any(w in v for w in ("excellent", "good", "fair", "poor", "bad")), v
    page.locator("select[aria-label='Map']").select_option("sti")
    page.wait_for_function("document.querySelector('#stage-foot .scale').textContent.includes('STI')", timeout=10000)
    shot(page, "23-sti-map")
    page.locator("select[aria-label='Map']").select_option("direct")


def test_raked_seating(ui):
    page = ui["page"]
    page.get_by_role("tab", name="Room").click()
    rake = page.get_by_role("spinbutton", name="Rake: back row higher by")
    rake.scroll_into_view_if_needed()
    rake.fill("2.5")
    rake.press("Enter")
    page.locator("#stage-toolbar .seg button", has_text="Section").click()
    page.wait_for_selector(".section-view canvas")
    time.sleep(2.5)
    shot(page, "28-raked-section")
    page.locator("#stage-toolbar .seg button", has_text="3D").click()
    page.wait_for_selector(".view3d canvas", timeout=15000)
    time.sleep(2)
    shot(page, "29-raked-3d")
    page.locator("#stage-toolbar .seg button", has_text="Plan").click()
    page.get_by_role("tab", name="Room").click()
    rake = page.get_by_role("spinbutton", name="Rake: back row higher by")
    rake.fill("0")
    rake.press("Enter")


def test_aim_at_the_audience(ui):
    page = ui["page"]
    # the rule itself: a 12-box hang 12 m up, 2 m from the stage; seats 8..60 m, ears 1.2 m rising 2 m
    r = page.evaluate("""async () => {
        const { aimPatch } = await import('/js/aim.js');
        const type = { spec: { kind: 'element', splay_max: 10, box_spl: 140, constant_curvature: false } };
        const s = { name: 'Main', role: 'main', x: 10, y: 2, z: 12, coverage_h: 70, array_boxes: 12, array_splay: 2 };
        const p = { audience: { x0: 0, x1: 20, y0: 8, y1: 60, ear_height: 1.2, rake: 2 } };
        const out = aimPatch(s, p, type);
        const el = (y, ear) => Math.atan2(ear - 12, y - 2) * 180 / Math.PI; // on the centre line: straight ahead
        return { ...out, back: el(60, 3.2), front: el(8, 1.2), sub: aimPatch({ ...s, role: 'sub' }, p, type).patch };
    }""")
    n, splay, pitch = 12, r["patch"]["array_splay"], r["patch"]["pitch"]
    assert pitch + (n - 1) / 2 * splay == pytest.approx(r["back"], abs=0.6)  # top box on the last row
    assert r["patch"]["yaw"] == 0
    # bottom box at the first row, within the 0.5 degree splay steps rigging allows
    assert pitch - (n - 1) / 2 * splay == pytest.approx(r["front"], abs=(n - 1) * 0.25 + 0.1)
    assert r["sub"] is None
    # in the app: the linked partner follows (turn mirrored, splay the same), one undo step
    page.get_by_role("tab", name="Speakers").click()
    page.locator(".row-list .row-item", has_text="Hang L").click()
    tilt0 = page.get_by_role("spinbutton", name="Tilt").input_value()
    page.get_by_role("button", name="Aim at the audience").click()
    page.wait_for_selector(".toast:has-text('Hang L')")
    note = page.locator(".toast:has-text('Hang L')").last.inner_text()
    assert note.startswith("Aimed Hang L: turn"), note  # a generic box here: the hang rule is checked above
    tilt = page.get_by_role("spinbutton", name="Tilt").input_value()
    turn = float(page.get_by_role("spinbutton", name="Turn").input_value())
    page.locator(".row-list .row-item", has_text="Hang R").click()
    assert page.get_by_role("spinbutton", name="Tilt").input_value() == tilt
    assert float(page.get_by_role("spinbutton", name="Turn").input_value()) == pytest.approx(-turn)
    page.locator("body").click(position={"x": 700, "y": 20})
    page.keyboard.press("Control+z")
    time.sleep(0.5)
    page.locator(".row-list .row-item", has_text="Hang L").click()
    assert page.get_by_role("spinbutton", name="Tilt").input_value() == tilt0


def test_imperial_units(ui):
    page = ui["page"]
    page.get_by_role("tab", name="Setup").click()
    page.locator("select[aria-label='Lengths and distances']").select_option("imperial")
    page.get_by_role("tab", name="Room").click()
    width = page.get_by_role("spinbutton", name="Width")
    page.wait_for_function("document.querySelector('[aria-label=Width]').parentElement.textContent.includes('ft')")
    metres_shown = float(width.input_value())
    assert metres_shown > 20  # feet, not metres, for any example room
    width.fill("60")
    width.press("Enter")
    room = page.locator("#app").inner_text()
    assert "ft³" in room and " m³" not in room  # the room volume
    # the plan's hover readout speaks feet too
    box = page.locator("#stage canvas").first.bounding_box()
    page.mouse.move(box["x"] + box["width"] * 0.5, box["y"] + box["height"] * 0.5)
    page.wait_for_selector(".plan-tip:not([hidden])")
    assert " ft " in page.locator(".plan-tip").inner_text()
    shot(page, "25-imperial")
    # the choice survives a reload (it is restored with the session, after the panel is first built)
    time.sleep(1.5)  # autosave
    page.reload()
    page.wait_for_selector(".row-list, .field")
    page.get_by_role("tab", name="Room").click()
    page.wait_for_function("document.querySelector('[aria-label=Width]').parentElement.textContent.includes('ft')", timeout=20000)
    assert abs(float(page.get_by_role("spinbutton", name="Width").input_value()) - 60) < 0.05
    # section: the hover readout
    page.locator("#stage-toolbar .seg button", has_text="Section").click()
    page.wait_for_selector(".section-view canvas")
    page.wait_for_function("document.querySelector('#stage-foot .scale') !== null", timeout=20000)
    time.sleep(1.0)
    box = page.locator(".section-view canvas").bounding_box()
    page.mouse.move(box["x"] + box["width"] * 0.6, box["y"] + box["height"] * 0.55)
    page.wait_for_function("/ ft /.test(document.querySelector('#stage-foot').textContent)", timeout=10000)
    shot(page, "26-imperial-section")
    # 3D: the hover tooltip
    page.locator("#stage-toolbar .seg button", has_text="3D").click()
    page.wait_for_selector(".view3d canvas", timeout=15000)
    page.locator(".view-buttons button", has_text="Top").click()
    time.sleep(1.5)
    box = page.locator(".view3d canvas").bounding_box()
    page.mouse.move(box["x"] + box["width"] * 0.5, box["y"] + box["height"] * 0.5)
    page.mouse.move(box["x"] + box["width"] * 0.52, box["y"] + box["height"] * 0.5)
    page.wait_for_selector(".view3d .tip:not([hidden])", timeout=10000)
    assert " ft " in page.locator(".view3d .tip").inner_text()
    txt = page.locator("#app").inner_text()
    for bad in ("null", "undefined", "NaN"):
        assert bad not in txt, bad
    page.locator("#stage-toolbar .seg button", has_text="Plan").click()
    page.get_by_role("tab", name="Setup").click()
    page.locator("select[aria-label='Lengths and distances']").select_option("metric")
    page.get_by_role("tab", name="Room").click()
    # 60 ft was stored as 18.29 m
    assert abs(float(page.get_by_role("spinbutton", name="Width").input_value()) - 18.29) < 0.01
    page.keyboard.press("Control+z")


def test_no_null_or_undefined_text_anywhere(ui):
    import re

    page = ui["page"]
    bad = re.compile(r"(null|undefined|NaN)")
    seen = []
    for tab in ("Room", "Speakers", "Mics", "Setup"):
        page.get_by_role("tab", name=tab).click()
        if tab == "Speakers":
            page.locator(".row-list .row-item").first.click()
        for m in ("direct", "total", "clarity", "interference", "sti", "spread"):
            page.locator("select[aria-label='Map']").select_option(m)
            time.sleep(0.15)
            txt = page.locator("#app").inner_text()
            seen += [f"{tab}/{m}: …{txt[max(0, x.start() - 40):x.end() + 10]}…" for x in bad.finditer(txt)]
    page.locator("select[aria-label='Map']").select_option("direct")
    for view in ("Section", "3D", "Plan"):
        page.locator("#stage-toolbar .seg button", has_text=view).click()
        time.sleep(1.2)
        txt = page.locator("#app").inner_text()
        seen += [f"{view}: …{txt[max(0, x.start() - 40):x.end() + 10]}…" for x in bad.finditer(txt)]
    assert not seen, seen


def test_delete_saved_and_discard_prompt(ui):
    page = ui["page"]
    ui["dialogs"].clear()
    page.get_by_role("button", name="Open").click()
    time.sleep(0.4)
    page.locator(".menu .icon-btn").first.click()
    assert confirm(page) or any("Delete" in d for d in ui["dialogs"])
    time.sleep(0.8)
    assert page.locator(".save-state").inner_text() == "Not saved"

    ui["dialogs"].clear()
    page.get_by_role("button", name="New").click()
    page.get_by_role("menuitem").first.click()
    assert confirm(page) or len(ui["dialogs"]) == 1


def test_no_console_errors(ui):
    assert not ui["errors"], ui["errors"]


def test_strict_csp_blocks_nothing(ui, tmp_path):
    ctx = ui["browser"].new_context(viewport={"width": 1560, "height": 960})
    page = ctx.new_page()
    errors = []
    page.on("console", lambda m: errors.append(m.text) if m.type == "error" else None)
    page.on("pageerror", lambda e: errors.append(str(e)))
    page.add_init_script("window.__csp = []; document.addEventListener('securitypolicyviolation', e => window.__csp.push(e.violatedDirective + ' ' + e.blockedURI));")
    page.goto(ui["url"])
    page.wait_for_selector(".score .num", timeout=30000)
    page.locator("#stage-toolbar .seg button", has_text="3D").click()
    page.wait_for_selector(".view3d canvas", timeout=15000)
    time.sleep(1.0)
    # the features that load code or data at run time, under the real policy
    page.locator("#stage-toolbar .seg button", has_text="Section").click()
    page.wait_for_selector("#stage-foot .scale", timeout=20000)
    page.get_by_role("tab", name="Speakers").click()
    page.get_by_role("button", name="Sub array…").click()
    page.wait_for_selector(".dialog .sub-sketch")
    page.keyboard.press("Escape")
    page.locator("#stage-toolbar .seg button", has_text="3D").click()
    page.get_by_role("tab", name="Room").click()
    with page.expect_file_chooser() as fc:
        # "Import a 3D scan…", or "Replace…" when an earlier test left a scan in the shared session
        page.locator("button", has_text=re.compile(r"^(Import a 3D scan…|Replace…)$")).first.click()
    fc.value.set_files(str(textured_glb(tmp_path / "textured.glb")))
    # The texture loads (no CSP violation); a flat test quad is then rightly refused as a room.
    page.wait_for_selector(".dialog :text('ft) tall')", timeout=20000)
    page.keyboard.press("Escape")
    with page.expect_file_chooser() as fc:
        page.locator("button", has_text=re.compile(r"^(Import a 3D scan…|Replace…)$")).first.click()
    fc.value.set_files(str(venue_mvr(tmp_path / "venue.mvr", tmp_path)))
    page.wait_for_selector(".dialog .scan-preview", timeout=20000)
    for _ in range(100):  # no wait_for_function here: it evaluates a string, which this policy rightly blocks
        if page.locator(".dialog [aria-label='Stage to back']").input_value():
            break
        time.sleep(0.2)
    assert float(page.locator(".dialog [aria-label='Height']").input_value()) == pytest.approx(5.0, abs=0.2)
    page.keyboard.press("Escape")
    page.wait_for_selector(".dialog", state="detached", timeout=10000)
    with page.expect_file_chooser() as fc:
        page.locator("button", has_text=re.compile(r"^(Import a 3D scan…|Replace…)$")).first.click()
    fc.value.set_files(str(pathlib.Path(__file__).parent / "fixtures" / "roomplan.usdz"))
    page.wait_for_selector(".dialog .scan-preview", timeout=20000)
    for _ in range(100):
        if page.locator(".dialog [aria-label='Stage to back']").input_value():
            break
        time.sleep(0.2)
    assert float(page.locator(".dialog [aria-label='Height']").input_value()) == pytest.approx(5.0, abs=0.2)
    violations = page.evaluate("window.__csp")
    ctx.close()
    errors = [e for e in errors if "status of 422" not in e]
    assert violations == [] and errors == [], (violations, errors)


FIXTURES = pathlib.Path(__file__).parent / "fixtures"


def gltf_room(folder: pathlib.Path, w=12.0, length=20.0, h=5.0, tex_dir="") -> pathlib.Path:
    """A box room as a .gltf with a separate .bin and a PNG texture (as Sketchfab and many exporters write)."""
    import json
    import struct
    import zlib

    folder.mkdir(parents=True, exist_ok=True)
    pos = [(x, y, z) for x in (0, w) for y in (0, h) for z in (0, length)]
    quads = [(0, 1, 3, 2), (4, 6, 7, 5), (0, 4, 5, 1), (2, 3, 7, 6), (0, 2, 6, 4), (1, 5, 7, 3)]
    idx = [i for a, b, c, d in quads for i in (a, b, c, a, c, d)]
    uv = [(x / w, z / length) for x, _, z in pos]
    blob = (b"".join(struct.pack("<3f", *v) for v in pos) + b"".join(struct.pack("<2f", *t) for t in uv)
            + b"".join(struct.pack("<H", i) for i in idx))
    (folder / "room data.bin").write_bytes(blob)

    def chunk(t, d):
        return struct.pack(">I", len(d)) + t + d + struct.pack(">I", zlib.crc32(t + d) & 0xFFFFFFFF)
    png = (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", 2, 2, 8, 2, 0, 0, 0))
           + chunk(b"IDAT", zlib.compress(b"\x00" + b"\xc0\x40\x40" * 2 + b"\x00" + b"\x40\xc0\x40" * 2)) + chunk(b"IEND", b""))
    if tex_dir:
        (folder / tex_dir).mkdir(exist_ok=True)
    (folder / tex_dir / "wall.png").write_bytes(png)
    gltf = {
        "asset": {"version": "2.0"}, "scene": 0, "scenes": [{"nodes": [0]}], "nodes": [{"mesh": 0}],
        "meshes": [{"primitives": [{"attributes": {"POSITION": 0, "TEXCOORD_0": 1}, "indices": 2, "material": 0}]}],
        "materials": [{"pbrMetallicRoughness": {"baseColorTexture": {"index": 0}}}],
        "textures": [{"source": 0}], "images": [{"uri": (f"{tex_dir}/" if tex_dir else "") + "wall.png"}],
        "buffers": [{"uri": "room%20data.bin", "byteLength": len(blob)}],
        "bufferViews": [{"buffer": 0, "byteOffset": 0, "byteLength": 96}, {"buffer": 0, "byteOffset": 96, "byteLength": 64},
                        {"buffer": 0, "byteOffset": 160, "byteLength": 72}],
        "accessors": [{"bufferView": 0, "componentType": 5126, "count": 8, "type": "VEC3", "min": [0, 0, 0], "max": [w, h, length]},
                      {"bufferView": 1, "componentType": 5126, "count": 8, "type": "VEC2"},
                      {"bufferView": 2, "componentType": 5123, "count": 36, "type": "SCALAR"}],
    }
    path = folder / "room.gltf"
    path.write_text(json.dumps(gltf), encoding="utf-8")
    return path


def _dialog_dims(page):
    page.wait_for_selector(".dialog .scan-preview", timeout=20000)
    page.wait_for_function("document.querySelector('.dialog [aria-label=\"Stage to back\"]').value !== ''", timeout=20000)
    return (sorted(float(page.locator(f".dialog [aria-label='{k}']").input_value()) for k in ("Stage to back", "Width")),
            float(page.locator(".dialog [aria-label='Height']").input_value()))


def test_usd_reader_matches_openusd(ui, tmp_path):
    """Binary (crate) USDZ written by OpenUSD itself; bounds from OpenUSD's own BBoxCache (see fixtures/make_usdz.py)."""
    import base64
    import io
    import zipfile

    expected = {
        "roomplan.usdz": [[1, 0, -4], [21.392, 5, 19.321]],  # Y up, metres: a turned and moved shell, a Cube riser
        "cm_zup.usdz": [[0, 0, -20], [12, 5, 0]],  # centimetres, Z up: 12 x 20 x 5 m, turned to Y up
        "mixed.usdz": [[0.5, 0, -3.75], [10.75, 3, 0.25]],  # matrix op, quaternion, fractional translate, pentagon
    }
    for name, want in expected.items():
        got = ui["page"].evaluate("""async (b64) => {
            const bytes = Uint8Array.from(atob(b64), (c) => c.charCodeAt(0));
            const THREE = await import('three');
            const g = await (await import('/js/scan.js')).parseScan(bytes.buffer, 'usdz');
            const bb = new THREE.Box3().setFromObject(g);
            return [bb.min.toArray(), bb.max.toArray()];
        }""", base64.b64encode((FIXTURES / name).read_bytes()).decode())
        assert got[0] == pytest.approx(want[0], abs=0.002) and got[1] == pytest.approx(want[1], abs=0.002), (name, got)
    # A package whose directory claims a huge unpacked size is refused before anything is unpacked.
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("scene.usda", "#usda 1.0\n" + " " * 5000)
    data = bytearray(buf.getvalue())
    cd = data.rfind(b"PK\x01\x02")
    data[cd + 24:cd + 28] = (0x7FFFFFFF).to_bytes(4, "little")
    err = ui["page"].evaluate("""async (b64) => {
        const bytes = Uint8Array.from(atob(b64), (c) => c.charCodeAt(0));
        try { await (await import('/js/scan.js')).parseScan(bytes.buffer, 'usdz'); return null; } catch (e) { return e.message; }
    }""", base64.b64encode(bytes(data)).decode())
    assert err and "more than the app will read" in err


def test_import_usdz_and_gltf(ui, tmp_path):
    page = ui["page"]
    page.get_by_role("tab", name="Room").click()
    if page.get_by_role("button", name="Remove scan").count():
        page.get_by_role("button", name="Remove scan").click()
    # USDZ from an iPhone-style export: measured, used, and read back after a reload
    with page.expect_file_chooser() as fc:
        page.get_by_role("button", name=re.compile("^(Import a 3D scan…|Replace…)$")).first.click()
    fc.value.set_files(str(FIXTURES / "roomplan.usdz"))
    dims, height = _dialog_dims(page)
    assert dims == pytest.approx([12.0, 20.0], abs=0.4) and height == pytest.approx(5.0, abs=0.2)
    page.locator(".dialog .btn.primary").click()
    page.wait_for_selector(".dialog", state="detached", timeout=20000)
    time.sleep(1.5)  # autosave
    page.reload()
    page.wait_for_selector(".view3d canvas", timeout=20000)
    time.sleep(2.0)
    page.get_by_role("tab", name="Room").click()
    assert page.locator(".scan-card .name").inner_text() == "roomplan"
    assert page.evaluate("import('/js/store.js').then((m) => m.state.project.scan.ext)") == "usdz"
    page.get_by_role("button", name="Remove scan").click()
    page.locator("#stage-toolbar .seg button", has_text="Plan").click()

    # glTF chosen together with its .bin and texture: packed into one GLB
    gltf = gltf_room(tmp_path / "files")
    with page.expect_file_chooser() as fc:
        page.get_by_role("button", name="Import a 3D scan…").click()
    fc.value.set_files([str(gltf), str(tmp_path / "files" / "room data.bin"), str(tmp_path / "files" / "wall.png")])
    dims, height = _dialog_dims(page)
    assert dims == pytest.approx([12.0, 20.0], abs=0.4) and height == pytest.approx(5.0, abs=0.2)
    assert not page.locator(".toast", has_text="left plain").count()  # the texture was found
    page.locator(".dialog .btn.primary").click()
    page.wait_for_selector(".dialog", state="detached", timeout=20000)
    page.get_by_role("tab", name="Room").click()
    assert page.evaluate("import('/js/store.js').then((m) => m.state.project.scan.ext)") == "glb"
    page.get_by_role("button", name="Remove scan").click()
    page.locator("#stage-toolbar .seg button", has_text="Plan").click()

    # the .gltf alone: its buffer is named in the message
    with page.expect_file_chooser() as fc:
        page.get_by_role("button", name="Import a 3D scan…").click()
    fc.value.set_files(str(gltf))
    page.locator(".toast", has_text="room data.bin").wait_for(timeout=10000)

    # the folder it came in, texture in a subfolder
    folder = tmp_path / "exported model"
    gltf_room(folder, tex_dir="textures")
    with page.expect_file_chooser() as fc:
        page.get_by_role("button", name="Import a folder (unpacked MVR or glTF)…").click()
    fc.value.set_files(str(folder))
    dims, height = _dialog_dims(page)
    assert dims == pytest.approx([12.0, 20.0], abs=0.4)
    assert not page.locator(".toast", has_text="left plain").count()
    page.locator(".dialog .btn", has_text="Cancel").click()
    page.wait_for_selector(".dialog", state="detached", timeout=10000)


def test_compressed_glb_draco_meshopt_ktx2(ui):
    """Each decoder runs in WebAssembly (Draco and KTX2 in a worker) under the app's own content policy."""
    import base64

    page = ui["page"]
    page.evaluate("window.__csp = []; document.addEventListener('securitypolicyviolation', e => window.__csp.push(e.violatedDirective + ' ' + e.blockedURI));")
    for name in ("room_draco.glb", "room_meshopt.glb", "room_ktx2.glb"):
        got = page.evaluate("""async (b64) => {
            const bytes = Uint8Array.from(atob(b64), (c) => c.charCodeAt(0));
            const THREE = await import('three');
            const g = await (await import('/js/scan.js')).parseScan(bytes.buffer, 'glb');
            const bb = new THREE.Box3().setFromObject(g);
            let tex = null;
            g.traverse((o) => { if (o.isMesh && o.material.map) tex = [o.material.map.image?.width, o.material.map.isCompressedTexture || false]; });
            return { min: bb.min.toArray(), max: bb.max.toArray(), tex };
        }""", base64.b64encode((FIXTURES / name).read_bytes()).decode())
        assert got["min"] == pytest.approx([0, 0, 0], abs=0.01) and got["max"] == pytest.approx([12, 5, 20], abs=0.01), (name, got)
        if name == "room_ktx2.glb":
            assert got["tex"] and got["tex"][0] == 128, got  # the Basis texture was transcoded
    assert page.evaluate("window.__csp") == []


def test_dxf_reader_matches_ezdxf(ui):
    """Every entity kind on its own layer; bounds from ezdxf (see fixtures/make_dxf_fixtures.py)."""
    import json

    expected = json.loads((FIXTURES / "dxf_expected.json").read_text())
    got = ui["page"].evaluate("""async (text) => {
        const THREE = await import('three');
        const g = (await import('/js/dxf.js')).parseDxf(text);
        const out = {};
        for (const c of g.children) { const b = new THREE.Box3().setFromObject(c); out[c.name] = [b.min.toArray(), b.max.toArray()]; }
        return { layers: out, info: g.userData.dxf };
    }""", (FIXTURES / "venue3d.dxf").read_text())
    for layer, (lo, hi) in expected.items():
        size = max(b - a for a, b in zip(lo, hi))
        tol = 1 + 0.004 * size  # arcs are drawn as 7.5-degree chords
        assert layer in got["layers"], layer
        glo, ghi = got["layers"][layer]
        assert glo == pytest.approx(lo, abs=tol) and ghi == pytest.approx(hi, abs=tol), (layer, got["layers"][layer], [lo, hi])
    info = got["info"]
    assert info["metersPerUnit"] == 0.001 and info["acis"] == 1 and info["inserts"] == (1 + 3) + (1 + 6 * 3) and not info["flat"]


def test_import_dxf(ui):
    page = ui["page"]
    page.get_by_role("tab", name="Room").click()
    if page.get_by_role("button", name="Remove scan").count():
        page.get_by_role("button", name="Remove scan").click()
    pick = re.compile("^(Import a 3D scan…|Replace…)$")
    # a 3D drawing in millimetres: converted, sized, stored as GLB
    with page.expect_file_chooser() as fc:
        page.get_by_role("button", name=pick).first.click()
    fc.value.set_files(str(FIXTURES / "venue3d.dxf"))
    dims, height = _dialog_dims(page)
    assert dims == pytest.approx([12.0, 20.0], abs=0.4) and height == pytest.approx(5.0, abs=0.2)
    assert page.locator(".dialog [aria-label='Scan units']").input_value() == "0.001"
    assert page.locator(".toast", has_text="1 ACIS solid left out").count()
    page.locator(".dialog .btn.primary").click()
    page.wait_for_selector(".dialog", state="detached", timeout=20000)
    page.get_by_role("tab", name="Room").click()
    assert page.evaluate("import('/js/store.js').then((m) => m.state.project.scan.ext)") == "glb"
    page.get_by_role("button", name="Remove scan").click()
    page.locator("#stage-toolbar .seg button", has_text="Plan").click()

    # a flat plan: raised into walls of the height asked for
    with page.expect_file_chooser() as fc:
        page.get_by_role("button", name=pick).first.click()
    fc.value.set_files(str(FIXTURES / "plan2d.dxf"))
    page.wait_for_selector(".dialog :text('Raise the plan into walls')", timeout=20000)
    assert page.locator(".dialog [aria-label='Drawing units']").input_value() == "1"
    page.locator(".dialog [aria-label='Wall height']").fill("26.25" if page.locator(".dialog .input-wrap em").first.inner_text() == "ft" else "8")
    page.locator(".dialog .btn.primary", has_text="Raise walls").click()
    dims, height = _dialog_dims(page)
    assert dims == pytest.approx([18.0, 30.0], abs=0.5) and height == pytest.approx(8.0, abs=0.3)
    page.locator(".dialog .btn", has_text="Cancel").click()
    page.wait_for_selector(".dialog", state="detached", timeout=10000)

    # nothing but ACIS: said so
    with page.expect_file_chooser() as fc:
        page.get_by_role("button", name=pick).first.click()
    fc.value.set_files(str(FIXTURES / "acis_only.dxf"))
    page.locator(".toast.error", has_text="ACIS").wait_for(timeout=10000)


def test_import_dwg_through_converter(ui, tmp_path, monkeypatch):
    import json
    import sys

    monkeypatch.setenv("APO_DWG_CONVERTER", json.dumps([sys.executable, str(FIXTURES / "fake_dwg2dxf.py")]))
    monkeypatch.setenv("APO_DWG_CONVERTER_KIND", "libredwg")
    dwg = tmp_path / "hall.dwg"
    dwg.write_bytes(b"AC1032" + b"\0" * 200)
    page = ui["page"]
    page.get_by_role("tab", name="Room").click()
    if page.get_by_role("button", name="Remove scan").count():
        page.get_by_role("button", name="Remove scan").click()
    with page.expect_file_chooser() as fc:
        page.get_by_role("button", name=re.compile("^(Import a 3D scan…|Replace…)$")).first.click()
    fc.value.set_files(str(dwg))
    dims, height = _dialog_dims(page)
    assert dims == pytest.approx([12.0, 20.0], abs=0.4) and height == pytest.approx(5.0, abs=0.2)
    page.locator(".dialog .btn", has_text="Cancel").click()
    page.wait_for_selector(".dialog", state="detached", timeout=10000)


def test_ifc_reader_and_import(ui):
    """Bounds from IfcOpenShell's geometry engine (fixtures/make_ifc_fixtures.py); spaces and openings left out."""
    import base64
    import json

    want = json.loads((FIXTURES / "ifc_expected.json").read_text())
    page = ui["page"]
    got = page.evaluate("""async (b64) => {
        const bytes = Uint8Array.from(atob(b64), (c) => c.charCodeAt(0));
        const THREE = await import('three');
        const g = await (await import('/js/formats.js')).parseIfc(bytes.buffer);
        const bb = new THREE.Box3().setFromObject(g);
        return { min: bb.min.toArray(), max: bb.max.toArray(), n: g.userData.ifc.elements };
    }""", base64.b64encode((FIXTURES / "hall.ifc").read_bytes()).decode())
    # web-ifc gives Y up: (x, y, z) Z-up = (x, -z, y) Y-up
    (x0, y0, z0), (x1, y1, z1) = want["min"], want["max"]
    assert got["min"] == pytest.approx([x0, z0, -y1], abs=0.01) and got["max"] == pytest.approx([x1, z1, -y0], abs=0.01), got
    assert got["n"] == 6  # four walls, two slabs: not the space, not the opening
    page.get_by_role("tab", name="Room").click()
    if page.get_by_role("button", name="Remove scan").count():
        page.get_by_role("button", name="Remove scan").click()
    with page.expect_file_chooser() as fc:
        page.get_by_role("button", name=re.compile("^(Import a 3D scan…|Replace…)$")).first.click()
    fc.value.set_files(str(FIXTURES / "hall.ifc"))
    dims, height = _dialog_dims(page)
    assert dims == pytest.approx([12.0, 20.0], abs=0.4) and height == pytest.approx(5.5, abs=0.3)
    assert page.locator(".dialog [aria-label='Scan units']").input_value() == "1"
    page.locator(".dialog .btn", has_text="Cancel").click()
    page.wait_for_selector(".dialog", state="detached", timeout=10000)


def test_step_iges_brep_readers_and_import(ui):
    """OpenCascade (occt-import-js) in a worker; fixtures and bounds from IfcOpenShell (fixtures/make_cad_fixtures.py)."""
    import base64
    import json

    want = json.loads((FIXTURES / "cad_expected.json").read_text())
    page = ui["page"]
    page.evaluate("window.__csp = []; document.addEventListener('securitypolicyviolation', e => window.__csp.push(e.violatedDirective + ' ' + e.blockedURI));")

    def read(name):
        return page.evaluate("""async ([b64, ext]) => {
            const bytes = Uint8Array.from(atob(b64), (c) => c.charCodeAt(0));
            const THREE = await import('three');
            const g = await (await import('/js/formats.js')).parseCad(bytes.buffer, ext);
            const bb = new THREE.Box3().setFromObject(g);
            const o = g.userData.cad.origin;
            return { min: bb.min.toArray().map((v, i) => v + o[i]), max: bb.max.toArray().map((v, i) => v + o[i]), parts: g.userData.cad.parts };
        }""", [base64.b64encode((FIXTURES / name).read_bytes()).decode(), name.rsplit(".", 1)[1]])

    step = read("hall.step")  # declares metres; the round tower sets +x
    assert step["min"] == pytest.approx(want["hall"]["min"], abs=0.02) and step["max"] == pytest.approx(want["hall"]["max"], abs=0.02), step
    assert step["parts"] == 5
    iges = read("hall.igs")  # declares millimetres around the same numbers: 1000 times smaller in metres
    assert iges["max"] == pytest.approx([v / 1000 for v in want["hall"]["max"]], abs=2e-5), iges
    brep = read("wall.brep")  # no units; a wall placed by a rotation
    assert brep["min"] == pytest.approx(want["wall"]["min"], abs=0.01) and brep["max"] == pytest.approx(want["wall"]["max"], abs=0.01), brep
    assert page.evaluate("window.__csp") == []

    page.get_by_role("tab", name="Room").click()
    if page.get_by_role("button", name="Remove scan").count():
        page.get_by_role("button", name="Remove scan").click()
    with page.expect_file_chooser() as fc:
        page.get_by_role("button", name=re.compile("^(Import a 3D scan…|Replace…)$")).first.click()
    fc.value.set_files(str(FIXTURES / "hall.step"))
    (width, length), height = _dialog_dims(page)
    assert length == pytest.approx(20.0, abs=0.5) and 11.5 <= width <= 15.7 and 4.8 <= height <= 7.1
    assert page.locator(".dialog [aria-label='Scan units']").input_value() == "1"
    assert page.locator(".dialog [aria-label='Up axis']").input_value() == "z"
    page.locator(".dialog .btn.primary").click()
    page.wait_for_selector(".dialog", state="detached", timeout=20000)
    page.get_by_role("tab", name="Room").click()
    assert page.evaluate("import('/js/store.js').then((m) => m.state.project.scan.ext)") == "glb"
    page.get_by_role("button", name="Remove scan").click()
    page.locator("#stage-toolbar .seg button", has_text="Plan").click()
