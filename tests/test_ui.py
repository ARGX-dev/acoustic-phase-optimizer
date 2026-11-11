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
    ctx = browser.new_context(viewport={"width": 1560, "height": 960}, accept_downloads=True)
    page = ctx.new_page()
    state = {"errors": [], "dialogs": [], "page": page, "ctx": ctx, "url": f"http://127.0.0.1:{port}/"}
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
