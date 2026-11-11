"""DWG to DXF through a converter installed on this computer.

DWG is a closed format and the open readers are GPL, which this MIT app cannot bundle, so DWG goes through one
of these if it is installed (checked in this order):

  1. APO_DWG_CONVERTER: a path to ODAFileConverter.exe or dwg2dxf, or a JSON list (a program and its first
     arguments; used by the tests). APO_DWG_CONVERTER_KIND = "oda" or "libredwg" if the name does not say.
  2. ODA File Converter (free from the Open Design Alliance) at its default place,
     C:\\Program Files\\ODA\\ODAFileConverter*\\ODAFileConverter.exe.
     Arguments, as ezdxf's odafc add-on passes them: IN_FOLDER OUT_FOLDER ACAD2018 DXF 0 0 FILTER
  3. LibreDWG's dwg2dxf on the PATH. Arguments, from its --help: -y -o OUT.dxf IN.dwg

The converter runs with a fixed argument list (no shell), in a fresh temporary folder, with a time limit.
"""

from __future__ import annotations

import glob
import json
import os
import shutil
import subprocess
import tempfile
from pathlib import Path

TIMEOUT_S = 180
MAX_DXF_BYTES = 1024 * 1024 * 1024


class NoConverter(Exception):
    pass


class ConvertFailed(Exception):
    pass


def find_converter() -> tuple[list[str], str] | None:
    """(command, kind) of the converter to use, or None."""
    env = os.environ.get("APO_DWG_CONVERTER", "").strip()
    if env:
        cmd = json.loads(env) if env.startswith("[") else [env.strip('"')]
        kind = os.environ.get("APO_DWG_CONVERTER_KIND") or ("oda" if "oda" in Path(cmd[-1]).name.lower() else "libredwg")
        return cmd, kind
    for base in (os.environ.get("ProgramFiles", r"C:\Program Files"), os.environ.get("ProgramFiles(x86)", "")):
        if base:
            hits = sorted(glob.glob(os.path.join(base, "ODA", "ODAFileConverter*", "ODAFileConverter.exe")))
            if hits:
                return [hits[-1]], "oda"
    exe = shutil.which("dwg2dxf")
    if exe:
        return [exe], "libredwg"
    return None


def convert(data: bytes) -> bytes:
    """DWG bytes -> ASCII DXF bytes."""
    if not data.startswith(b"AC10"):
        raise ConvertFailed("this is not a DWG file")
    found = find_converter()
    if not found:
        raise NoConverter
    cmd, kind = found
    with tempfile.TemporaryDirectory(prefix="apo-dwg-") as tmp:
        src_dir, out_dir = Path(tmp, "in"), Path(tmp, "out")
        src_dir.mkdir()
        out_dir.mkdir()
        src = src_dir / "drawing.dwg"
        src.write_bytes(data)
        if kind == "oda":
            args = [*cmd, str(src_dir), str(out_dir), "ACAD2018", "DXF", "0", "0", "drawing.dwg"]
        else:
            args = [*cmd, "-y", "-o", str(out_dir / "drawing.dxf"), str(src)]
        kwargs = {}
        if os.name == "nt":  # keep the converter's window from appearing
            si = subprocess.STARTUPINFO()
            si.dwFlags |= subprocess.STARTF_USESHOWWINDOW
            si.wShowWindow = subprocess.SW_HIDE
            kwargs["startupinfo"] = si
        try:
            proc = subprocess.run(args, cwd=tmp, capture_output=True, timeout=TIMEOUT_S, shell=False, check=False, **kwargs)
        except subprocess.TimeoutExpired:
            raise ConvertFailed(f"the converter took longer than {TIMEOUT_S} s") from None
        except OSError as exc:
            raise ConvertFailed(f"the converter could not be started: {exc}") from None
        out = out_dir / "drawing.dxf"
        if not out.is_file():
            err = (proc.stderr or proc.stdout or b"").decode(errors="replace").strip().splitlines()[-3:]
            raise ConvertFailed("the converter produced no DXF" + (f": {' '.join(err)}" if err else ""))
        if out.stat().st_size > MAX_DXF_BYTES:
            raise ConvertFailed("the converted drawing is larger than 1 GB")
        return out.read_bytes()
