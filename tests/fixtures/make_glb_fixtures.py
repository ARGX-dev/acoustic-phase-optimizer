"""Write the compressed-GLB test fixtures: the same 12 x 5 x 20 m box room (x across, y up, z along) as

  room_draco.glb    KHR_draco_mesh_compression, encoded by Google's Draco (pip install DracoPy numpy)
  room_meshopt.glb  EXT_meshopt_compression, encoded here from the extension's bitstream specification
                    (Appendix A); test_ui.py decodes it with the official meshoptimizer decoder
  room_ktx2.glb     KHR_texture_basisu with a real ETC1S/BasisLZ texture (Dot_C.ktx2, see ATTRIBUTION.md)

Usage: python make_glb_fixtures.py OUTDIR
"""
import json
import pathlib
import struct
import sys

HERE = pathlib.Path(__file__).parent
W, H, L = 12.0, 5.0, 20.0
POS = [(x, y, z) for x in (0, W) for y in (0, H) for z in (0, L)]
QUADS = [(0, 1, 3, 2), (4, 6, 7, 5), (0, 4, 5, 1), (2, 3, 7, 6), (0, 2, 6, 4), (1, 5, 7, 3)]
IDX = [i for a, b, c, d in QUADS for i in (a, b, c, a, c, d)]
UV = [(x / W, z / L) for x, _, z in POS]


def glb(doc: dict, binary: bytes) -> bytes:
    js = json.dumps(doc).encode()
    js += b" " * (-len(js) % 4)
    binary += b"\0" * (-len(binary) % 4)
    total = 12 + 8 + len(js) + 8 + len(binary)
    return (struct.pack("<III", 0x46546C67, 2, total) + struct.pack("<II", len(js), 0x4E4F534A) + js
            + struct.pack("<II", len(binary), 0x004E4942) + binary)


def pad4(b: bytes) -> bytes:
    return b + b"\0" * (-len(b) % 4)


def base_doc(extensions):
    return {"asset": {"version": "2.0", "generator": "make_glb_fixtures.py"}, "scene": 0, "scenes": [{"nodes": [0]}],
            "nodes": [{"mesh": 0}], "extensionsUsed": extensions, "extensionsRequired": extensions}


POS_ACCESSOR = {"componentType": 5126, "count": 8, "type": "VEC3", "min": [0, 0, 0], "max": [W, H, L]}


# ------------------------------------------------------------------ Draco
def draco() -> bytes:
    import DracoPy
    import numpy as np

    pts = np.array(POS, dtype=np.float32)
    faces = np.array(IDX, dtype=np.uint32).reshape(-1, 3)
    data = DracoPy.encode(pts, faces, quantization_bits=16, preserve_order=True)
    dec = DracoPy.decode(data)
    n, m = len(dec.points), len(dec.faces)
    doc = base_doc(["KHR_draco_mesh_compression"])
    doc["meshes"] = [{"primitives": [{"attributes": {"POSITION": 0}, "indices": 1, "extensions": {
        "KHR_draco_mesh_compression": {"bufferView": 0, "attributes": {"POSITION": 0}}}}]}]
    doc["accessors"] = [{**POS_ACCESSOR, "count": n}, {"componentType": 5125, "count": m * 3, "type": "SCALAR"}]
    doc["bufferViews"] = [{"buffer": 0, "byteOffset": 0, "byteLength": len(data)}]
    doc["buffers"] = [{"byteLength": len(pad4(data))}]
    return glb(doc, data)


# ------------------------------------------------------------------ meshopt
def zigzag8(d: int) -> int:
    d &= 0xFF
    return (~(d << 1) if d & 0x80 else d << 1) & 0xFF


def encode_attributes(raw: bytes, stride: int) -> bytes:
    """Mode 0. Each group of 16 byte-deltas is stored as zero (header bits 0) or raw (header bits 3)."""
    count = len(raw) // stride
    out = bytearray([0xA0])
    max_block = min((8192 // stride) & ~15, 256)
    prev = raw[:stride]  # the baseline element, stored in the tail
    i = 0
    while i < count:
        n = min(count - i, max_block)
        groups = (n + 15) // 16
        for k in range(stride):
            deltas, last = [], prev[k]
            for e in range(groups * 16):
                v = raw[(i + e) * stride + k] if e < n else last  # padding elements repeat the last value
                deltas.append(zigzag8(v - last))
                last = v
            header = bytearray((groups + 3) // 4)
            body = bytearray()
            for g in range(groups):
                chunk = deltas[g * 16:(g + 1) * 16]
                mode = 0 if not any(chunk) else 3
                header[g // 4] |= mode << ((g % 4) * 2)
                if mode == 3:
                    body += bytes(chunk)
            out += header + body
        prev = raw[(i + n - 1) * stride:(i + n) * stride]
        i += n
    tail = max(stride, 32)
    out += b"\0" * (tail - stride) + raw[:stride]
    return bytes(out)


def varint(v: int) -> bytes:
    out = bytearray()
    while True:
        b = v & 0x7F
        v >>= 7
        if v:
            out.append(b | 0x80)
        else:
            out.append(b)
            return bytes(out)


def encode_indices(idx: list[int]) -> bytes:
    """Mode 2: each index as a zigzag delta from baseline 0."""
    out = bytearray([0xD1])
    last = 0
    for v in idx:
        d = v - last
        zz = (d << 1) if d >= 0 else ((~d) << 1) | 1
        out += varint((zz << 1) | 0)
        last = v
    return bytes(out + b"\0\0\0\0")


def meshopt() -> bytes:
    pos = b"".join(struct.pack("<3f", *p) for p in POS)
    uv = b"".join(struct.pack("<2f", *t) for t in UV)
    ind = b"".join(struct.pack("<I", i) for i in IDX)
    streams = [(encode_attributes(pos, 12), 12, 8, "ATTRIBUTES", len(pos)),
               (encode_attributes(uv, 8), 8, 8, "ATTRIBUTES", len(uv)),
               (encode_indices(IDX), 4, len(IDX), "INDICES", len(ind))]
    binary, views, fallback = b"", [], 0
    for data, stride, count, mode, raw_len in streams:
        off = len(binary)
        binary += pad4(data)
        view = {"buffer": 1, "byteOffset": fallback, "byteLength": raw_len, "extensions": {"EXT_meshopt_compression": {
            "buffer": 0, "byteOffset": off, "byteLength": len(data), "byteStride": stride, "count": count, "mode": mode}}}
        if mode == "ATTRIBUTES":
            view["byteStride"] = stride
        views.append(view)
        fallback += raw_len
    doc = base_doc(["EXT_meshopt_compression"])
    doc["meshes"] = [{"primitives": [{"attributes": {"POSITION": 0, "TEXCOORD_0": 1}, "indices": 2}]}]
    doc["accessors"] = [{**POS_ACCESSOR, "bufferView": 0}, {"bufferView": 1, "componentType": 5126, "count": 8, "type": "VEC2"},
                        {"bufferView": 2, "componentType": 5125, "count": len(IDX), "type": "SCALAR"}]
    doc["bufferViews"] = views
    doc["buffers"] = [{"byteLength": len(binary)},
                      {"byteLength": fallback, "extensions": {"EXT_meshopt_compression": {"fallback": True}}}]
    return glb(doc, binary)


# ------------------------------------------------------------------ KTX2 texture
def ktx2() -> bytes:
    tex = (HERE / "Dot_C.ktx2").read_bytes()
    pos = b"".join(struct.pack("<3f", *p) for p in POS)
    uv = b"".join(struct.pack("<2f", *t) for t in UV)
    ind = b"".join(struct.pack("<H", i) for i in IDX)
    binary = pos + uv + ind
    binary += b"\0" * (-len(binary) % 4)
    tex_off = len(binary)
    binary += tex
    doc = base_doc(["KHR_texture_basisu"])
    doc["meshes"] = [{"primitives": [{"attributes": {"POSITION": 0, "TEXCOORD_0": 1}, "indices": 2, "material": 0}]}]
    doc["materials"] = [{"pbrMetallicRoughness": {"baseColorTexture": {"index": 0}}}]
    doc["textures"] = [{"extensions": {"KHR_texture_basisu": {"source": 0}}}]
    doc["images"] = [{"bufferView": 3, "mimeType": "image/ktx2"}]
    doc["accessors"] = [{**POS_ACCESSOR, "bufferView": 0}, {"bufferView": 1, "componentType": 5126, "count": 8, "type": "VEC2"},
                        {"bufferView": 2, "componentType": 5123, "count": len(IDX), "type": "SCALAR"}]
    doc["bufferViews"] = [{"buffer": 0, "byteOffset": 0, "byteLength": 96}, {"buffer": 0, "byteOffset": 96, "byteLength": 64},
                          {"buffer": 0, "byteOffset": 160, "byteLength": len(ind)},
                          {"buffer": 0, "byteOffset": tex_off, "byteLength": len(tex)}]
    doc["buffers"] = [{"byteLength": len(pad4(binary))}]
    return glb(doc, binary)


if __name__ == "__main__":
    out = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else HERE)
    for name, make in (("room_meshopt.glb", meshopt), ("room_ktx2.glb", ktx2), ("room_draco.glb", draco)):
        data = make()
        (out / name).write_bytes(data)
        print(name, len(data), "bytes")
