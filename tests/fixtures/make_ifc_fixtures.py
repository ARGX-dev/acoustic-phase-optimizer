"""Write the IFC test fixture with IfcOpenShell (pip install ifcopenshell==0.9.0.post1) and its expected bounds.

  hall.ifc           IFC4 in millimetres: four walls, a floor slab and a roof slab (about 12 x 20 x 5 m), under
                     a site moved to (100, 50, 0) m. Also an IfcSpace far larger than the hall and an
                     IfcOpeningElement 40 m away: neither is a physical surface, so neither may be imported.
  ifc_expected.json  world bounds (metres, Z up) of the physical elements, from IfcOpenShell's geometry engine

Usage: python make_ifc_fixtures.py OUTDIR
"""
import json
import pathlib
import sys

import ifcopenshell
import ifcopenshell.api
import ifcopenshell.geom
import numpy as np

out = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else pathlib.Path(__file__).parent)
run = ifcopenshell.api.run

m = ifcopenshell.file(schema="IFC4")
project = run("root.create_entity", m, ifc_class="IfcProject", name="Hall")
mm = run("unit.add_si_unit", m, unit_type="LENGTHUNIT", prefix="MILLI")
run("unit.assign_unit", m, units=[mm])
model = run("context.add_context", m, context_type="Model")
body = run("context.add_context", m, context_type="Model", context_identifier="Body", target_view="MODEL_VIEW", parent=model)
site = run("root.create_entity", m, ifc_class="IfcSite", name="Site")
building = run("root.create_entity", m, ifc_class="IfcBuilding", name="Building")
storey = run("root.create_entity", m, ifc_class="IfcBuildingStorey", name="Ground")
run("aggregate.assign_object", m, relating_object=project, products=[site])
run("aggregate.assign_object", m, relating_object=site, products=[building])
run("aggregate.assign_object", m, relating_object=building, products=[storey])


def place(product, x, y, z, rot_deg=0.0):
    a = np.radians(rot_deg)
    mat = np.eye(4)
    mat[:3, :3] = [[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]]
    mat[:3, 3] = [x, y, z]
    run("geometry.edit_object_placement", m, product=product, matrix=mat, is_si=True)


place(site, 100, 50, 0)
place(building, 0, 0, 0)
place(storey, 0, 0, 0)


def element(cls, name, rep, x, y, z, rot=0.0):
    e = run("root.create_entity", m, ifc_class=cls, name=name)
    if cls == "IfcSpace":  # spaces belong to a storey by aggregation
        run("aggregate.assign_object", m, relating_object=storey, products=[e])
    elif cls != "IfcOpeningElement":  # an opening hangs off nothing here
        run("spatial.assign_container", m, relating_structure=storey, products=[e])
    run("geometry.assign_representation", m, product=e, representation=rep)
    place(e, x, y, z, rot)
    return e


def wall(length, height=5.0, thickness=0.2):
    return run("geometry.add_wall_representation", m, context=body, length=length, height=height, thickness=thickness)


element("IfcWall", "South", wall(12.0), 0, 0, 0)
element("IfcWall", "North", wall(12.0), 0, 19.8, 0)
element("IfcWall", "West", wall(20.0), 0.2, 0, 0, 90)
element("IfcWall", "East", wall(20.0), 12.0, 0, 0, 90)
floor = run("geometry.add_slab_representation", m, context=body, depth=0.2,
            polyline=[(0.0, 0.0), (12.0, 0.0), (12.0, 20.0), (0.0, 20.0)])
element("IfcSlab", "Floor", floor, 0, 0, -0.2)
roof = run("geometry.add_slab_representation", m, context=body, depth=0.3,
           polyline=[(0.0, 0.0), (12.0, 0.0), (12.0, 20.0), (0.0, 20.0)])
element("IfcSlab", "Roof", roof, 0, 0, 5.0)
# not physical surfaces: must not be imported
element("IfcSpace", "Air", wall(60.0, height=30.0, thickness=60.0), -20, -20, -5)
element("IfcOpeningElement", "Stray opening", wall(2.0, height=2.0, thickness=2.0), 40, 0, 0)

path = out / "hall.ifc"
m.write(str(path))

settings = ifcopenshell.geom.settings()
settings.set("use-world-coords", True)
f = ifcopenshell.open(str(path))
lo, hi = np.full(3, np.inf), np.full(3, -np.inf)
for p in f.by_type("IfcProduct"):
    if p.is_a("IfcSpace") or p.is_a("IfcOpeningElement") or not p.Representation:
        continue
    shape = ifcopenshell.geom.create_shape(settings, p)
    v = np.array(shape.geometry.verts).reshape(-1, 3)
    lo, hi = np.minimum(lo, v.min(0)), np.maximum(hi, v.max(0))
(out / "ifc_expected.json").write_text(json.dumps({"min": lo.round(4).tolist(), "max": hi.round(4).tolist()}))
print("bounds (m, Z up):", lo.round(3), hi.round(3))
