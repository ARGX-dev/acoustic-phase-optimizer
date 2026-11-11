"""Stands in for LibreDWG's dwg2dxf in the tests: same arguments (-y -o OUT.dxf IN.dwg).

It checks it was called the way the app calls the real program, then writes venue3d.dxf as the "converted"
drawing. No real DWG is read; that needs ODA File Converter or LibreDWG installed.
"""
import pathlib
import shutil
import sys

args = sys.argv[1:]
assert args[0] == "-y" and args[1] == "-o" and len(args) == 4, args
out, src = pathlib.Path(args[2]), pathlib.Path(args[3])
if not src.read_bytes().startswith(b"AC10"):
    sys.exit("not a DWG")
if b"BROKEN" in src.read_bytes():
    sys.exit("dwg2dxf: ERROR 0x800 in decode")
shutil.copy(pathlib.Path(__file__).parent / "venue3d.dxf", out)
