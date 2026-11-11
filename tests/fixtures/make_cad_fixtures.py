"""Write the STEP, IGES and BREP test fixtures with IfcOpenShell's OpenCascade serializers
(pip install ifcopenshell==0.9.0.post1), and the bounds each should have.

  hall.step, hall.igs  four walls (12 x 20 x 5 m) and a round tower (radius 1.5 m, 7 m tall) standing outside
                       the east wall, so a curved surface sets the +x bound
  wall.brep            the west wall alone: 0.2 x 20 x 5 m, placed by a rotation (BREP has no units)
  cad_expected.json    bounds in metres, Z up, from IfcOpenShell's own triangulation

hall.step declares metres. hall.igs gets OpenCascade's default unit flag, millimetres, around the same
metre-valued shapes, so as a file it is a 15.5 x 20 x 7 mm model: read in metres it must come out 1000 times
smaller, which checks that the reader honours the unit a file declares.

Usage: python make_cad_fixtures.py OUTDIR
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
w = ifcopenshell.ifcopenshell_wrapper

m = ifcopenshell.file(schema="IFC4")
project = run("root.create_entity", m, ifc_class="IfcProject", name="Hall")
mm = run("unit.add_si_unit", m, unit_type="LENGTHUNIT", prefix="MILLI")
run("unit.assign_unit", m, units=[mm])
model = run("context.add_context", m, context_type="Model")
body = run("context.add_context", m, context_type="Model", context_identifier="Body", target_view="MODEL_VIEW", parent=model)
storey = run("root.create_entity", m, ifc_class="IfcBuildingStorey", name="Ground")
run("aggregate.assign_object", m, relating_object=project, products=[storey])


def place(product, x, y, z, rot_deg=0.0):
    a = np.radians(rot_deg)
    mat = np.eye(4)
    mat[:3, :3] = [[np.cos(a), -np.sin(a), 0], [np.sin(a), np.cos(a), 0], [0, 0, 1]]
    mat[:3, 3] = [x, y, z]
    run("geometry.edit_object_placement", m, product=product, matrix=mat, is_si=True)


def element(cls, name, rep, x, y, z, rot=0.0):
    e = run("root.create_entity", m, ifc_class=cls, name=name)
    run("spatial.assign_container", m, relating_structure=storey, products=[e])
    run("geometry.assign_representation", m, product=e, representation=rep)
    place(e, x, y, z, rot)
    return e


def wall(length, height=5.0, thickness=0.2):
    return run("geometry.add_wall_representation", m, context=body, length=length, height=height, thickness=thickness)


element("IfcWall", "South", wall(12.0), 0, 0, 0)
element("IfcWall", "North", wall(12.0), 0, 19.8, 0)
west = element("IfcWall", "West", wall(20.0), 0.2, 0, 0, 90)
element("IfcWall", "East", wall(20.0), 12.0, 0, 0, 90)
circle = m.create_entity("IfcCircleProfileDef", ProfileType="AREA", Radius=1500.0)
tower = run("geometry.add_profile_representation", m, context=body, profile=circle, depth=7.0)
element("IfcColumn", "Tower", tower, 14.0, 10.0, 0)
tmp = out / "_cad_source.ifc"
m.write(str(tmp))
f = ifcopenshell.open(str(tmp))


def bounds(products):
    tri = ifcopenshell.geom.settings()
    tri.set("use-world-coords", True)
    lo, hi = np.full(3, np.inf), np.full(3, -np.inf)
    for p in products:
        v = np.array(ifcopenshell.geom.create_shape(tri, p).geometry.verts).reshape(-1, 3)
        lo, hi = np.minimum(lo, v.min(0)), np.maximum(hi, v.max(0))
    return {"min": lo.round(4).tolist(), "max": hi.round(4).tolist()}


everything = [p for p in f.by_type("IfcProduct") if p.Representation]
expected = {"hall": bounds(everything), "wall": bounds([f.by_guid(west.GlobalId)])}

for name, make in (("hall.step", ifcopenshell.geom.serializers.step), ("hall.igs", ifcopenshell.geom.serializers.igs)):
    st = ifcopenshell.geom.settings()
    st.set("iterator-output", w.NATIVE)  # exact shapes, not triangles
    st.set("use-world-coords", True)
    ser = make(str(out / name), st)
    ser.setFile(f)
    ser.setUnitNameAndMagnitude("METER", 1.0)
    ser.writeHeader()
    it = ifcopenshell.geom.iterator(st, f, 1)
    if it.initialize():
        while True:
            ser.write(it.get())
            if not it.next():
                break
    ser.finalize()

st = ifcopenshell.geom.settings()
st.set("iterator-output", w.SERIALIZED)
st.set("use-world-coords", True)
(out / "wall.brep").write_text(ifcopenshell.geom.create_shape(st, f.by_guid(west.GlobalId)).geometry.brep_data)
tmp.unlink()
(out / "cad_expected.json").write_text(json.dumps(expected))
print(expected)
