"""Command line entry point.

    python -m acoustic_phase_optimizer                 open the app
    python -m acoustic_phase_optimizer --browser       open it in your default browser instead
    python -m acoustic_phase_optimizer optimize venue.apo.json --deep --eq --report report.html
"""

from __future__ import annotations

import argparse
import os
import shutil
import socket
import subprocess
import sys
import threading
import time
import webbrowser
from pathlib import Path

from . import __version__

DEFAULT_PORT = 8765


def _free_port(preferred: int) -> int:
    for port in (preferred, 0):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(("127.0.0.1", port))
                return s.getsockname()[1]
            except OSError:
                continue
    raise RuntimeError("No free TCP port available")


def _app_browser() -> str | None:
    """A Chromium-based browser that can show the app in its own window."""
    candidates = []
    if sys.platform == "win32":
        for root in (os.environ.get("ProgramFiles(x86)"), os.environ.get("ProgramFiles"), os.environ.get("LOCALAPPDATA")):
            if root:
                candidates += [Path(root) / "Microsoft/Edge/Application/msedge.exe",
                               Path(root) / "Google/Chrome/Application/chrome.exe"]
    elif sys.platform == "darwin":
        candidates += [Path("/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"),
                       Path("/Applications/Microsoft Edge.app/Contents/MacOS/Microsoft Edge")]
    for c in candidates:
        if c.exists():
            return str(c)
    for name in ("google-chrome", "chromium", "chromium-browser", "microsoft-edge"):
        found = shutil.which(name)
        if found:
            return found
    return None


def serve(port: int, mode: str) -> int:
    import uvicorn

    from .server import create_app
    from .server.storage import data_dir

    port = _free_port(port)
    url = f"http://127.0.0.1:{port}/"
    app = create_app()
    server = uvicorn.Server(uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning"))
    print(f"Acoustic Phase Optimizer {__version__}  ->  {url}")

    def open_ui() -> None:
        deadline = time.time() + 15
        while not server.started and time.time() < deadline:
            time.sleep(0.05)
        browser = _app_browser() if mode == "window" else None
        if browser:
            profile = data_dir() / "window-profile"
            subprocess.Popen([browser, f"--app={url}", f"--user-data-dir={profile}", "--no-first-run",
                              "--no-default-browser-check", "--disable-background-mode", "--window-size=1560,980"],
                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            print("Close the app window to quit.")
        elif mode != "none":
            webbrowser.open(url)
            print("Close the browser tab, or press Ctrl+C here, to quit.")
        else:
            print("Press Ctrl+C to quit.")
            return
        # The browser process can outlive its window (background mode, a shared profile), so it is the
        # page itself that tells us it is gone: it pings while open and says goodbye when it closes.
        while not server.should_exit:
            time.sleep(1)
            if app.state.presence.gone():
                print("The app window was closed. Stopping.")
                server.should_exit = True

    threading.Thread(target=open_ui, daemon=True).start()
    try:
        server.run()
    except KeyboardInterrupt:
        pass
    return 0


def optimize(args: argparse.Namespace) -> int:
    from .engine.field import get_field
    from .engine.model import Project
    from .engine.optimize import auto_eq, deep_optimize, evaluate, quick_align
    from .export import to_text
    from .report import render_report

    src = Path(args.project)
    project = Project.model_validate_json(src.read_text(encoding="utf-8"))
    baseline = project.model_copy(deep=True)
    field = get_field(project)
    before = evaluate(project, field)
    print(f"{project.name}: {len(project.speakers)} speakers, score {before.get('score')}")
    if args.deep:
        speakers, info = deep_optimize(project, field, time_budget_s=args.seconds)
        print(f"  deep optimize: {info['generations']} generations, kept the {info['picked']} settings")
    else:
        speakers, notes = quick_align(project, field)
        for n in notes:
            print("  -", n)
    project.speakers = speakers
    if args.eq:
        project.speakers, report = auto_eq(project, field)
        for r in report:
            print(f"  EQ {r['name']}: {r['bands']} bands, {r['rms_before_db']} -> {r['rms_after_db']} dB")
    after = evaluate(project, field)
    print(f"  score {before.get('score')} -> {after.get('score')}")
    out = Path(args.output) if args.output else src.with_name(src.name.replace(".apo.json", "").replace(".json", "") + ".optimized.apo.json")
    out.write_text(project.model_dump_json(indent=2), encoding="utf-8")
    print(f"  saved {out}")
    if args.report:
        Path(args.report).write_text(render_report(project, baseline), encoding="utf-8")
        print(f"  report {args.report}")
    print()
    print(to_text(project))
    return 0


def main(argv: list[str] | None = None) -> int:
    # Started with pythonw (no console): give print() and logging somewhere harmless to write.
    if sys.stdout is None:
        sys.stdout = open(os.devnull, "w")  # noqa: SIM115
    if sys.stderr is None:
        sys.stderr = open(os.devnull, "w")  # noqa: SIM115
    parser = argparse.ArgumentParser(prog="acoustic_phase_optimizer", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--version", action="version", version=__version__)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help=f"port to serve on (default {DEFAULT_PORT})")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--browser", action="store_true", help="open in the default browser instead of an app window")
    group.add_argument("--no-window", action="store_true", help="start the server only")
    sub = parser.add_subparsers(dest="command")
    opt = sub.add_parser("optimize", help="align a saved project without the UI")
    opt.add_argument("project", help="path to a project .json file")
    opt.add_argument("-o", "--output", help="where to write the optimized project")
    opt.add_argument("--deep", action="store_true", help="run the global search instead of the quick alignment")
    opt.add_argument("--seconds", type=float, default=15.0, help="time budget for --deep")
    opt.add_argument("--eq", action="store_true", help="also fit the automatic EQ")
    opt.add_argument("--report", help="write an HTML report to this path")
    args = parser.parse_args(argv)
    if args.command == "optimize":
        return optimize(args)
    return serve(args.port, "browser" if args.browser else "none" if args.no_window else "window")


if __name__ == "__main__":
    sys.exit(main())
