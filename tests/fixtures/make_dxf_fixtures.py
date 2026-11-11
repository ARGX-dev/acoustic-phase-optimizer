"""Write the DXF test fixtures with ezdxf (pip install ezdxf==1.4.2) and the bounds each layer should have.

  venue3d.dxf   millimetres: one layer per entity kind (polyface shell, 3DFACE, MESH, polygon mesh, nested and
                arrayed block inserts, LWPOLYLINE with bulges in a mirrored OCS, extruded walls, a tilted circle,
                an extruded SOLID, 2D/3D POLYLINE, ELLIPSE) and one 3DSOLID (ACIS, unreadable on purpose)
  plan2d.dxf    metres: a flat 18 x 30 m plan (lines only), to be raised into walls
  acis_only.dxf a drawing with nothing but an ACIS solid
  dxf_expected.json  per-layer [min, max] of venue3d.dxf from ezdxf.bbox (or by hand where noted)

Usage: python make_dxf_fixtures.py OUTDIR
"""
import json
import math
import pathlib
import sys

import ezdxf
from ezdxf import bbox

out = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else pathlib.Path(__file__).parent)


def layer_box(msp, layer):
    ents = msp.query(f'*[layer=="{layer}"]')
    ext = bbox.extents(ents, fast=False)
    return [list(ext.extmin), list(ext.extmax)]


# ------------------------------------------------------------------ venue3d.dxf
SAT = ["400 0 1 0", "@8 ezdxf-test @7 unknown @24 Sat Jan 01 00:00:00 2000", "1 9.9999999999999995e-07 1e-10", "End-of-ACIS-data"]

doc = ezdxf.new("R2010", setup=False)  # R2010: MESH, and ACIS solids as SAT text
doc.header["$INSUNITS"] = 4  # millimetres
msp = doc.modelspace()
W, L, H = 12000, 20000, 5000
expected = {}

pf = msp.add_polyface(dxfattribs={"layer": "SHELL"})
c = [(x, y, z) for x in (0, W) for y in (0, L) for z in (0, H)]
for q in [(0, 2, 6, 4), (1, 5, 7, 3), (0, 4, 5, 1), (2, 3, 7, 6), (0, 1, 3, 2), (4, 6, 7, 5)]:
    pf.append_face([c[i] for i in q])

msp.add_3dface([(1000, 1000, 0), (4000, 1000, 0), (4000, 3000, 900), (1000, 3000, 900)], dxfattribs={"layer": "FACE"})

mesh = msp.add_mesh(dxfattribs={"layer": "MESH"})
with mesh.edit_data() as md:
    md.vertices = [(x, y, z) for x in (6000, 6600) for y in (1000, 1400) for z in (0, 700)]
    md.faces = [(0, 2, 6, 4), (1, 5, 7, 3), (0, 4, 5, 1), (2, 3, 7, 6), (0, 1, 3, 2), (4, 6, 7, 5)]

pm = msp.add_polymesh(size=(3, 4), dxfattribs={"layer": "PMESH"})
for i in range(3):
    for j in range(4):
        pm.set_mesh_vertex((i, j), (8000 + i * 500, 15000 + j * 600, 3000 + 200 * math.sin(i + j)))

seat = doc.blocks.new("SEAT", base_point=(250, 250, 0))
seat.add_3dface([(0, 0, 0), (500, 0, 0), (500, 500, 450), (0, 500, 450)], dxfattribs={"layer": "INSERTS"})
row = doc.blocks.new("ROW")
for k in range(3):
    row.add_blockref("SEAT", (k * 600, 0, 0), dxfattribs={"layer": "INSERTS", "rotation": 10 * k})
msp.add_blockref("ROW", (3000, 6000, 400), dxfattribs={"layer": "INSERTS", "rotation": 30, "xscale": 2, "yscale": 1, "zscale": 1.5})
grid = msp.add_blockref("ROW", (2000, 10000, 0), dxfattribs={"layer": "INSERTS", "rotation": -15})
grid.grid(size=(2, 3), spacing=(900, 2200))  # 2 rows x 3 columns

msp.add_lwpolyline([(0, 0, 0, 0, 0.5), (3000, 0, 0, 0, 0), (3000, 2000, 0, 0, -0.3), (0, 2000, 0, 0, 0)], format="xyseb", close=True,
                   dxfattribs={"layer": "LWPOLY", "elevation": 200, "extrusion": (0, 0, -1)})

wall_pts = [(500, 12000), (5500, 12000), (5500, 16000), (500, 16000)]
msp.add_lwpolyline(wall_pts, close=True, dxfattribs={"layer": "WALL", "thickness": 3000, "elevation": 100})
expected["WALL"] = [[500, 12000, 100], [5500, 16000, 3100]]  # by hand: ezdxf's bbox ignores thickness

msp.add_circle((2000, 3000, 1000), 800, dxfattribs={"layer": "TILT", "extrusion": (1, 0, 1)})

msp.add_solid([(9000, 2000, 0), (11000, 2000, 0), (9000, 3000, 0), (11000, 3500, 0)], dxfattribs={"layer": "SOLIDT", "thickness": 800})
expected["SOLIDT"] = [[9000, 2000, 0], [11000, 3500, 800]]  # by hand: the quad raised 800 along its normal

p2 = msp.add_polyline2d([(7000, 8000), (10000, 8000), (10000, 11000)], dxfattribs={"layer": "POLY2D", "elevation": (0, 0, 2500)})
for v, b in zip(p2.vertices, (0.4, 0, 0)):
    v.dxf.bulge = b
msp.add_polyline3d([(100, 100, 100), (11900, 200, 4900), (11800, 19900, 2500)], dxfattribs={"layer": "P3D"})
msp.add_ellipse((6000, 18000, 2000), major_axis=(1500, 500, 0), ratio=0.4, start_param=0.3, end_param=4.0, dxfattribs={"layer": "ELL"})
s3 = msp.add_3dsolid(dxfattribs={"layer": "ACIS"})
s3.sat = SAT

for layer in ("SHELL", "FACE", "MESH", "PMESH", "INSERTS", "LWPOLY", "TILT", "POLY2D", "P3D", "ELL"):
    expected[layer] = layer_box(msp, layer)
doc.saveas(out / "venue3d.dxf")
(out / "dxf_expected.json").write_text(json.dumps(expected, indent=1))

# ------------------------------------------------------------------ plan2d.dxf
doc = ezdxf.new("R2010", setup=False)
doc.header["$INSUNITS"] = 6  # metres
msp = doc.modelspace()
msp.add_lwpolyline([(0, 0), (18, 0), (18, 30), (0, 30)], close=True)
msp.add_line((0, 6), (18, 6))  # stage edge
msp.add_arc((9, 6), 4, 180, 360)  # a thrust
doc.saveas(out / "plan2d.dxf")

# ------------------------------------------------------------------ acis_only.dxf
doc = ezdxf.new("R2010", setup=False)
doc.modelspace().add_3dsolid().sat = SAT
doc.saveas(out / "acis_only.dxf")

for k, v in expected.items():
    print(k, [[round(x, 1) for x in p] for p in v])
