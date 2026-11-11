"""Write the binary-crate USDZ test fixtures with OpenUSD (pip install usd-core) and print OpenUSD's own
world bounds for each, which test_ui.py::test_usd_reader_matches_openusd checks against.

Usage: python make_usdz.py OUTDIR
"""
import os
import sys

from pxr import Gf, Sdf, Usd, UsdGeom, UsdUtils

out = sys.argv[1]

def box(stage, path, sx, sy, sz):
    m = UsdGeom.Mesh.Define(stage, path)
    pts = [(x, y, z) for x in (0, sx) for y in (0, sy) for z in (0, sz)]
    faces = [(0, 1, 3, 2), (4, 6, 7, 5), (0, 4, 5, 1), (2, 3, 7, 6), (0, 2, 6, 4), (1, 5, 7, 3)]  # quads
    m.CreatePointsAttr([Gf.Vec3f(*p) for p in pts])
    m.CreateFaceVertexCountsAttr([4] * 6)
    m.CreateFaceVertexIndicesAttr([i for f in faces for i in f])
    return m

def make(name, up, mpu, build):
    layer = os.path.join(out, name + ".usdc")
    stage = Usd.Stage.CreateNew(layer)
    UsdGeom.SetStageUpAxis(stage, up)
    UsdGeom.SetStageMetersPerUnit(stage, mpu)
    root = UsdGeom.Xform.Define(stage, "/Room")
    stage.SetDefaultPrim(root.GetPrim())
    build(stage)
    stage.GetRootLayer().Save()
    assert open(layer, "rb").read(8) == b"PXR-USDC"
    pkg = os.path.join(out, name + ".usdz")
    assert UsdUtils.CreateNewUsdzPackage(Sdf.AssetPath(layer), pkg)
    cache = UsdGeom.BBoxCache(Usd.TimeCode.Default(), ["default"])
    r = cache.ComputeWorldBound(stage.GetPseudoRoot()).ComputeAlignedRange()
    print(name, up, mpu, "min", [round(v, 3) for v in r.GetMin()], "max", [round(v, 3) for v in r.GetMax()], os.path.getsize(pkg), "bytes")

def roomplan(stage):  # metres, Y up: a shell, turned and moved, plus a wall as a scaled Cube (RoomPlan style)
    g = UsdGeom.Xform.Define(stage, "/Room/Walls")
    g.AddTranslateOp().Set(Gf.Vec3d(1, 0, 2))
    g.AddRotateYOp().Set(30.0)
    box(stage, "/Room/Walls/Shell", 12, 5, 20)
    c = UsdGeom.Cube.Define(stage, "/Room/Walls/Riser")
    c.CreateSizeAttr(1.0)
    c.AddTranslateOp().Set(Gf.Vec3d(6, 0.25, 3))
    c.AddScaleOp().Set(Gf.Vec3f(8, 0.5, 2))

def cm_zup(stage):  # centimetres, Z up
    box(stage, "/Room/Shell", 1200, 2000, 500)

def mixed(stage):  # metres, Y up: a matrix transform, a quaternion, a fractional translate, a pentagon
    a = UsdGeom.Xform.Define(stage, "/Room/A")
    a.AddTransformOp().Set(Gf.Matrix4d().SetRotate(Gf.Rotation(Gf.Vec3d(0, 1, 0), 90)) * Gf.Matrix4d().SetTranslate(Gf.Vec3d(0.5, 0, 0.25)))
    box(stage, "/Room/A/Box", 4, 3, 2)
    b = UsdGeom.Xform.Define(stage, "/Room/B")
    b.AddTranslateOp().Set(Gf.Vec3d(10.75, 0.5, -3.125))
    b.AddOrientOp().Set(Gf.Quatf(0.7071068, 0, 0, 0.7071068))  # 90 degrees about z
    m = UsdGeom.Mesh.Define(stage, "/Room/B/Pent")
    m.CreatePointsAttr([Gf.Vec3f(0, 0, 0), Gf.Vec3f(2, 0, 0), Gf.Vec3f(2.5, 1.5, 0), Gf.Vec3f(1, 2.5, 0), Gf.Vec3f(-0.5, 1.5, 0)])
    m.CreateFaceVertexCountsAttr([5])
    m.CreateFaceVertexIndicesAttr([0, 1, 2, 3, 4])

make("roomplan", UsdGeom.Tokens.y, 1.0, roomplan)
make("mixed", UsdGeom.Tokens.y, 1.0, mixed)
make("cm_zup", UsdGeom.Tokens.z, 0.01, cm_zup)
