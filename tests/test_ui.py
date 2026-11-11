"""End-to-end checks of the web interface in a real browser.

Skipped when Playwright (or a Chromium/Edge it can drive) is not installed.
Set APO_UI_SHOTS to a folder to keep screenshots for a visual check.

The tests share one page and run in file order, like a person using the app.
"""

from __future__ import annotations

import os
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
    violations = page.evaluate("window.__csp")
    ctx.close()
    errors = [e for e in errors if "status of 422" not in e]
    assert violations == [] and errors == [], (violations, errors)
