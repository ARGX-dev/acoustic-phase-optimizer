"""Project files and the autosaved session, stored as plain JSON in the user's data folder."""

from __future__ import annotations

import json
import os
import re
import sys
from datetime import datetime
from pathlib import Path

from ..engine.model import Project

SUFFIX = ".apo.json"


def data_dir() -> Path:
    override = os.environ.get("APO_DATA_DIR")
    if override:
        base = Path(override)
    elif sys.platform == "win32":
        base = Path(os.environ.get("LOCALAPPDATA", Path.home() / "AppData" / "Local")) / "AcousticPhaseOptimizer"
    elif sys.platform == "darwin":
        base = Path.home() / "Library" / "Application Support" / "AcousticPhaseOptimizer"
    else:
        base = Path(os.environ.get("XDG_DATA_HOME", Path.home() / ".local" / "share")) / "acoustic-phase-optimizer"
    (base / "projects").mkdir(parents=True, exist_ok=True)
    return base


def slug(name: str) -> str:
    s = re.sub(r"[^\w\- ]+", "", name, flags=re.UNICODE).strip().replace(" ", "-")
    s = re.sub(r"-{2,}", "-", s)[:80].strip("-.")
    return s.lower() or "untitled"


def _path(key: str) -> Path:
    return data_dir() / "projects" / f"{slug(key)}{SUFFIX}"


def _write(path: Path, text: str) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def list_projects() -> list[dict]:
    out = []
    for p in (data_dir() / "projects").glob(f"*{SUFFIX}"):
        try:
            doc = json.loads(p.read_text(encoding="utf-8"))
            out.append({
                "key": p.name[: -len(SUFFIX)],
                "name": doc.get("name", p.stem),
                "modified": datetime.fromtimestamp(p.stat().st_mtime).isoformat(timespec="seconds"),
                "speakers": len(doc.get("speakers", [])),
                "size": [doc.get("room", {}).get(k) for k in ("width", "length", "height")],
            })
        except (OSError, ValueError):
            continue
    return sorted(out, key=lambda d: d["modified"], reverse=True)


def save_project(project: Project) -> dict:
    path = _path(project.name)
    _write(path, project.model_dump_json(indent=2))
    return {"key": path.name[: -len(SUFFIX)], "name": project.name, "path": str(path)}


def load_project(key: str) -> Project:
    path = _path(key)
    if not path.exists():
        raise FileNotFoundError(key)
    return Project.model_validate_json(path.read_text(encoding="utf-8"))


def delete_project(key: str) -> bool:
    path = _path(key)
    if path.exists():
        path.unlink()
        return True
    return False


SCAN_TYPES = {"glb": "model/gltf-binary", "obj": "text/plain", "ply": "application/octet-stream",
              "mvr": "application/zip", "3ds": "application/octet-stream", "dae": "model/vnd.collada+xml",
              "fbx": "application/octet-stream", "stl": "model/stl", "usdz": "model/vnd.usdz+zip",
              "usd": "application/octet-stream", "usda": "text/plain", "usdc": "application/octet-stream"}
MAX_SCAN_BYTES = 300 * 1024 * 1024


def scans_dir() -> Path:
    folder = data_dir() / "scans"
    folder.mkdir(exist_ok=True)
    return folder


def _scan_path(scan_id: str, ext: str) -> Path:
    if not re.fullmatch(r"[0-9a-f]{16,64}", scan_id) or ext not in SCAN_TYPES:
        raise ValueError("bad scan reference")
    return scans_dir() / f"{scan_id}.{ext}"


def save_scan(data: bytes, ext: str) -> dict:
    import hashlib

    scan_id = hashlib.sha256(data).hexdigest()[:32]
    path = _scan_path(scan_id, ext)
    if not path.exists():
        _write_bytes(path, data)
    return {"id": scan_id, "ext": ext, "bytes": len(data)}


def adopt_scan(tmp: Path, scan_id: str, ext: str, size: int) -> dict:
    """Keep an uploaded scan (already streamed to `tmp`) under its content hash."""
    path = _scan_path(scan_id, ext)
    if not path.exists():
        os.replace(tmp, path)
    return {"id": scan_id, "ext": ext, "bytes": size}


def scan_file(scan_id: str, ext: str) -> Path:
    path = _scan_path(scan_id, ext)
    if not path.exists():
        raise FileNotFoundError(scan_id)
    return path


# ------------------------------------------------------------ measured impulse responses (WAV, content-hashed)

MAX_IR_BYTES = 64 * 1024 * 1024


def irs_dir() -> Path:
    folder = data_dir() / "irs"
    folder.mkdir(exist_ok=True)
    return folder


def _ir_path(ir_id: str) -> Path:
    if not re.fullmatch(r"[0-9a-f]{16,64}", ir_id):
        raise ValueError("bad impulse response reference")
    return irs_dir() / f"{ir_id}.wav"


def save_ir(wav: bytes) -> str:
    import hashlib

    ir_id = hashlib.sha256(wav).hexdigest()[:32]
    path = _ir_path(ir_id)
    if not path.exists():
        _write_bytes(path, wav)
    return ir_id


def ir_file(ir_id: str) -> Path:
    path = _ir_path(ir_id)
    if not path.exists():
        raise FileNotFoundError(ir_id)
    return path


def _write_bytes(path: Path, data: bytes) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_bytes(data)
    os.replace(tmp, path)


# ------------------------------------------------------------ the user's own loudspeakers
def _user_speakers_path() -> Path:
    return data_dir() / "my_speakers.json"


def load_user_speakers() -> list[dict]:
    path = _user_speakers_path()
    if not path.exists():
        return []
    try:
        rows = json.loads(path.read_text(encoding="utf-8"))
        return rows if isinstance(rows, list) else []
    except (OSError, ValueError):
        return []


def save_user_speakers(rows: list[dict]) -> None:
    _write(_user_speakers_path(), json.dumps(rows, indent=1, ensure_ascii=False))


# ------------------------------------------------------------ the user's company, for report covers
def load_branding() -> dict:
    path = data_dir() / "branding.json"
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
        return doc if isinstance(doc, dict) else {}
    except (OSError, ValueError):
        return {}


def save_branding(doc: dict) -> None:
    _write(data_dir() / "branding.json", json.dumps(doc, ensure_ascii=False))


def save_session(payload: dict) -> None:
    _write(data_dir() / "session.json", json.dumps(payload))


def load_session() -> dict | None:
    path = data_dir() / "session.json"
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
        Project.model_validate(doc.get("project"))  # never hand a corrupt session to the UI
        return doc
    except (OSError, ValueError, TypeError):
        return None
