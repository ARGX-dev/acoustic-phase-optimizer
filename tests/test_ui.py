"""End-to-end checks of the web interface in a real browser.

Skipped when Playwright (or a Chromium/Edge it can drive) is not installed.
Set APO_UI_SHOTS to a folder to keep screenshots for a visual check.

The tests share one page and run in file order, like a person using the app.
"""

from __future__ import annotations

import os
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
    page.get_by_role("radio", name="Open air").click()
    page.wait_for_function(f"!document.querySelector('.stat-grid b').textContent.startsWith('{rt().split()[0]}')", timeout=15000)
    assert float(rt().split()[0]) < indoor
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
    for _ in range(4):
        page.locator("body").click(position={"x": 700, "y": 20})
        page.keyboard.press("Control+z")
    time.sleep(0.6)


def test_line_array_box_count_changes_level(ui):
    page = ui["page"]
    page.get_by_role("tab", name="Speakers").click()
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


def test_strict_csp_blocks_nothing(ui):
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
    time.sleep(1.5)
    violations = page.evaluate("window.__csp")
    ctx.close()
    assert violations == [] and errors == [], (violations, errors)
