"""The scanned room's real shape, for the sound prediction.

`build_shape` takes the repaired scan as triangles in room coordinates, which the browser sends once. It
reduces them to a few thousand triangles by vertex clustering. It then groups those triangles into flat
surfaces by region growing, with angle and distance tolerances, because scans are noisy and nothing is
exactly coplanar. Finally it names each surface and guesses its material. The result is small enough to
travel inside the project (`Room.shape`), so saving, sharing and undo need nothing extra.

`MeshRoom` is the runtime form of a shape:
  * the large flat surfaces become mirrors for image sources, up to second order (`mesh_images`);
  * `path_valid` keeps a receiver only when every reflection point lies inside its surface, in order, and
    no leg of the path is blocked;
  * `trace` follows seeded rays (specular plus scattered) and records which surface each one hits and
    when. `decay` turns those hit lists into the room's energy decay, and so its RT60, for any choice of
    materials, without tracing again.
Rays that leave through an opening (a doorway, an open roof, a hole in the scan) count as absorbed.

The volume comes from the rays' mean free path (V = mfp * S / 4), which works for open models too.
"""

from __future__ import annotations

import hashlib
from collections import OrderedDict
from dataclasses import dataclass, field

import numpy as np

MAX_TRIS = 4000  # triangles kept for the acoustic model
MAX_SURFACES = 160
MIN_SURFACE_M2 = 1.0  # a flat region smaller than this joins "Small parts and furniture"
MIN_REFLECTOR_M2 = 2.0  # surfaces this big (and flat) reflect as mirrors; the rest only absorb and scatter
MAX_FIRST_ORDER = 48  # mirrors for first-order reflections (largest first)
MAX_SECOND_ORDER = 20  # mirrors for second-order reflections
ANGLE_TOL_DEG = 14.0
ROLES = ("floor", "ceiling", "front", "back", "left", "right", "other")
ROLE_NAMES = {"floor": "Floor", "ceiling": "Ceiling", "front": "Stage wall", "back": "Back wall",
              "left": "Left wall", "right": "Right wall", "other": "Small parts and furniture"}
EPS = 1e-9

# Material names in the model file -> built-in materials (first match wins).
KEYWORDS = (
    (("acoustic tile", "acoustic_tile", "ceiling tile", "ceilingtile", "akustik"), "acoustic_tile"),
    (("absorber", "acoustic panel", "acoustic_panel", "foam"), "acoustic_panel"),
    (("window",), "window"),
    (("glass", "glazing", "mirror"), "glass"),
    (("carpet", "rug"), "carpet"),
    (("curtain", "drape", "velour", "fabric", "textile"), "drape"),
    (("seat", "chair", "sofa", "couch", "upholster", "bench"), "seats_empty"),
    (("bleacher",), "bleachers"),
    (("plywood",), "plywood_thin"),
    (("floor", "parquet", "timber floor", "wood floor"), "wood_floor"),
    (("wood", "timber", "oak", "pine", "veneer"), "wood_panel"),
    (("gypsum", "drywall", "plasterboard", "gyproc"), "gypsum"),
    (("plaster", "stucco"), "plaster"),
    (("brick",), "brick"),
    (("block", "cmu", "breeze"), "concrete_block"),
    (("concrete", "cement", "stone"), "concrete"),
    (("tile", "ceramic", "marble", "granite", "porcelain"), "tile"),
    (("vinyl", "linoleum", "lino"), "vinyl_floor"),
    (("steel", "metal", "deck"), "steel_deck"),
    (("grass", "lawn", "turf", "earth", "soil"), "grass"),
)


def material_for(name: str | None) -> str | None:
    n = (name or "").lower().replace("-", " ").replace("_", " ")
    if not n.strip():
        return None
    for words, mat in KEYWORDS:
        if any(w.replace("_", " ") in n for w in words):
            return mat
    return None


# ----------------------------------------------------------------------------- cleanup and reduction

def _tri_area_normal(V: np.ndarray, T: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    cr = np.cross(V[T[:, 1]] - V[T[:, 0]], V[T[:, 2]] - V[T[:, 0]])
    a2 = np.linalg.norm(cr, axis=1)
    return a2 / 2.0, cr / np.maximum(a2, EPS)[:, None]


def _clean(V: np.ndarray, T: np.ndarray, extra: np.ndarray | None = None, min_area: float = 1e-8):
    """Drop triangles with a repeated corner, zero area or a duplicate (same corners in any order)."""
    keep = (T[:, 0] != T[:, 1]) & (T[:, 1] != T[:, 2]) & (T[:, 0] != T[:, 2])
    T = T[keep]
    extra = None if extra is None else extra[keep]
    if len(T):
        area, _ = _tri_area_normal(V, T)
        keep = area > min_area
        T = T[keep]
        extra = None if extra is None else extra[keep]
    if len(T):
        _, first = np.unique(np.sort(T, axis=1), axis=0, return_index=True)
        first.sort()
        T = T[first]
        extra = None if extra is None else extra[first]
    return T, extra


def _compact(V: np.ndarray, T: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    used, inv = np.unique(T.ravel(), return_inverse=True)
    return V[used], inv.reshape(-1, 3)


def decimate(V: np.ndarray, T: np.ndarray, tag: np.ndarray | None = None, max_tris: int = MAX_TRIS):
    """Vertex clustering: snap vertices to a grid, growing the cell until at most `max_tris` remain.

    Each cluster's vertex sits where it best fits the planes of the faces around it (quadric clustering), so walls
    stay flat and corners sharp. Small objects collapse into a few faces. `tag` (one value per triangle, e.g. the material name index) follows the triangles that survive.
    """
    V = np.asarray(V, float)
    T = np.asarray(T, np.int64)
    T, tag = _clean(V, T, tag, 0.0)
    if len(T) <= max_tris:
        V2, T2 = _compact(V, T)
        return V2, T2, tag
    lo = V.min(axis=0)
    diag = float(np.linalg.norm(V.max(axis=0) - lo)) or 1.0
    cell = diag / 600.0
    # Each cluster's vertex goes where it best fits the planes of the faces around it (Lindstrom's quadric
    # clustering), so corners and edges stay sharp instead of rounding off by a cell.
    area, nrm = _tri_area_normal(V, T)
    off = np.einsum("ij,ij->i", nrm, V[T[:, 0]])
    qa = area[:, None, None] * nrm[:, :, None] * nrm[:, None, :]  # (m, 3, 3)
    qb = (area * off)[:, None] * nrm
    for _ in range(60):
        k = np.floor((V - lo) / cell).astype(np.int64)
        n = k.max(axis=0) + 1
        key = (k[:, 0] * n[1] + k[:, 1]) * n[2] + k[:, 2]
        uniq, inv = np.unique(key, return_inverse=True)
        nc = len(uniq)
        cnt = np.bincount(inv, minlength=nc).astype(float)
        mean = np.column_stack([np.bincount(inv, V[:, j], nc) for j in range(3)]) / cnt[:, None]
        corner_c = inv[T].ravel()
        A = np.zeros((nc, 3, 3))
        b = np.zeros((nc, 3))
        for i in range(3):
            b[:, i] = np.bincount(corner_c, np.repeat(qb[:, i], 3), nc)
            for j in range(3):
                A[:, i, j] = np.bincount(corner_c, np.repeat(qa[:, i, j], 3), nc)
        lam = 1e-2 * np.trace(A, axis1=1, axis2=2)[:, None, None] / 3 + 1e-12
        x = np.linalg.solve(A + lam * np.eye(3), (b + lam[:, :, 0] * mean)[..., None])[..., 0]
        far = np.linalg.norm(x - mean, axis=1) > 1.5 * cell
        Vc = np.where(far[:, None], mean, x)
        Tc, tg = _clean(Vc, inv[T], tag, (cell * 0.02) ** 2)
        if len(Tc) <= max_tris:
            V2, T2 = _compact(Vc, Tc)
            return V2, T2, tg
        cell *= 1.3
    raise ValueError("the model could not be simplified")


# ----------------------------------------------------------------------------- flat surfaces

def _edge_neighbours(T: np.ndarray) -> list[list[int]]:
    m = len(T)
    e = np.sort(np.concatenate([T[:, [0, 1]], T[:, [1, 2]], T[:, [2, 0]]]), axis=1)
    owner = np.tile(np.arange(m), 3)
    order = np.lexsort((e[:, 1], e[:, 0]))
    e, owner = e[order], owner[order]
    same = np.all(e[1:] == e[:-1], axis=1)
    nb: list[list[int]] = [[] for _ in range(m)]
    start = 0
    for i in range(1, len(e) + 1):
        if i == len(e) or not same[i - 1]:
            group = owner[start:i]
            if len(group) > 1:
                for a in group:
                    for b in group:
                        if a != b:
                            nb[a].append(int(b))
            start = i
    return nb


def flat_regions(V: np.ndarray, T: np.ndarray, angle_deg: float = ANGLE_TOL_DEG, dist_tol: float | None = None) -> np.ndarray:
    """Region label per triangle: edge-connected triangles that lie in one plane, within tolerances."""
    area, nrm = _tri_area_normal(V, T)
    cen = V[T].mean(axis=1)
    if dist_tol is None:
        diag = float(np.linalg.norm(V.max(axis=0) - V.min(axis=0)))
        dist_tol = float(np.clip(0.004 * diag, 0.04, 0.25))
    cos_tol = np.cos(np.radians(angle_deg))
    nb = _edge_neighbours(T)
    label = np.full(len(T), -1, np.int64)
    nxt = 0
    for seed in np.argsort(-area):
        if label[seed] >= 0:
            continue
        sn = nrm[seed]
        acc_n = sn * area[seed]
        acc_c = cen[seed] * area[seed]
        acc_a = area[seed]
        plane_n, plane_d = sn, float(sn @ cen[seed])
        label[seed] = nxt
        stack = [seed]
        while stack:
            t = stack.pop()
            for u in nb[t]:
                if label[u] >= 0:
                    continue
                dn = float(nrm[u] @ plane_n)
                if abs(dn) < cos_tol or abs(float(plane_n @ cen[u]) - plane_d) > dist_tol:
                    continue
                label[u] = nxt
                stack.append(u)
                acc_n = acc_n + (nrm[u] if dn > 0 else -nrm[u]) * area[u]
                acc_c = acc_c + cen[u] * area[u]
                acc_a += area[u]
                plane_n = acc_n / max(np.linalg.norm(acc_n), EPS)
                plane_d = float(plane_n @ (acc_c / acc_a))
        nxt += 1
    return _merge_regions(label, nb, area, nrm, cen, cos_tol, dist_tol)


def _merge_regions(label, nb, area, nrm, cen, cos_tol: float, dist_tol: float) -> np.ndarray:
    """Join neighbouring regions that lie in one plane (growing triangle by triangle can stop early on noise)."""
    n_reg = int(label.max()) + 1
    r_area = np.bincount(label, area, n_reg)
    planes = []
    for r in range(n_reg):
        tris = np.nonzero(label == r)[0]
        ref = nrm[tris[np.argmax(area[tris])]]
        n = (nrm[tris] * np.where(nrm[tris] @ ref >= 0, 1.0, -1.0)[:, None] * area[tris, None]).sum(axis=0)
        n /= max(np.linalg.norm(n), EPS)
        c = (cen[tris] * area[tris, None]).sum(axis=0) / max(area[tris].sum(), EPS)
        planes.append((n, c))
    adj: list[set[int]] = [set() for _ in range(n_reg)]
    for t, ns in enumerate(nb):
        for u in ns:
            if label[t] != label[u]:
                adj[label[t]].add(int(label[u]))
    new = np.full(n_reg, -1, np.int64)
    nxt = 0
    for seed in np.argsort(-r_area):
        if new[seed] >= 0:
            continue
        n0, c0 = planes[seed]
        d0 = float(n0 @ c0)
        new[seed] = nxt
        stack = [seed]
        while stack:
            r = stack.pop()
            for q in adj[r]:
                if new[q] >= 0:
                    continue
                n1, c1 = planes[q]
                if abs(float(n1 @ n0)) >= cos_tol and abs(float(n0 @ c1) - d0) <= dist_tol:
                    new[q] = nxt
                    stack.append(q)
        nxt += 1
    return new[label]


def _role(normal: np.ndarray, centroid: np.ndarray, dims: tuple[float, float, float]) -> str:
    W, L, H = dims
    if abs(normal[2]) > 0.7:
        return "floor" if centroid[2] < H / 2 else "ceiling"
    if abs(normal[0]) >= abs(normal[1]):
        return "left" if centroid[0] < W / 2 else "right"
    return "front" if centroid[1] < L / 2 else "back"


def build_shape(positions, indices, dims: tuple[float, float, float], tri_names=None, names: list[str] | None = None,
                max_tris: int = MAX_TRIS) -> dict:
    """The acoustic shape of a scan: reduced mesh, flat surfaces with names, roles and guessed materials.

    positions: (n, 3) in room coordinates (metres, z up); indices: (m, 3); tri_names: optional material-name
    index per triangle, into `names`. Returns the dict stored as `Room.shape`, plus `stats` for the user.
    """
    V = np.asarray(positions, float).reshape(-1, 3)
    T = np.asarray(indices, np.int64).reshape(-1, 3)
    if len(T) == 0 or len(V) < 3:
        raise ValueError("the model has no surfaces")
    if T.min() < 0 or T.max() >= len(V):
        raise ValueError("the model refers to corners it does not have")
    good = np.all(np.isfinite(V), axis=1)
    T = T[np.all(good[T], axis=1)]
    tag = None if tri_names is None else np.asarray(tri_names, np.int64).reshape(-1)
    if tag is not None and len(tag) != len(np.asarray(indices).reshape(-1, 3)):
        tag = None
    if tag is not None:
        tag = tag[np.all(good[np.asarray(indices, np.int64).reshape(-1, 3)], axis=1)]
    n_in = len(T)
    V, T, tag = decimate(V, T, tag, max_tris)
    if len(T) < 4:
        raise ValueError("the model has too few surfaces left after cleaning")
    area, nrm = _tri_area_normal(V, T)
    lab = flat_regions(V, T)
    n_reg = int(lab.max()) + 1
    reg_area = np.bincount(lab, area, n_reg)
    order = [r for r in np.argsort(-reg_area) if reg_area[r] >= MIN_SURFACE_M2][: MAX_SURFACES - 1]
    surfaces: list[dict] = []
    surf_of_region = np.full(n_reg, -1, np.int64)
    counts: dict[str, int] = {}
    for r in order:
        tris = np.nonzero(lab == r)[0]
        w = area[tris]
        ref = nrm[tris[np.argmax(w)]]
        n = (nrm[tris] * np.where(nrm[tris] @ ref >= 0, 1.0, -1.0)[:, None] * w[:, None]).sum(axis=0)
        n /= max(np.linalg.norm(n), EPS)
        c = (V[T[tris]].mean(axis=1) * w[:, None]).sum(axis=0) / w.sum()
        role = _role(n, c, dims)
        counts[role] = counts.get(role, 0) + 1
        mat = None
        if tag is not None and names:
            by = np.bincount(tag[tris], w, max(len(names), int(tag[tris].max()) + 1))
            idx = int(np.argmax(by))
            if idx < len(names):
                mat = material_for(names[idx])
        surf_of_region[r] = len(surfaces)
        surfaces.append({"name": ROLE_NAMES[role], "role": role, "material": mat, "area": round(float(reg_area[r]), 1)})
    # number repeated names, largest first: "Left wall", "Left wall 2", ...
    seen: dict[str, int] = {}
    for s in surfaces:
        if counts[s["role"]] > 1:
            seen[s["role"]] = seen.get(s["role"], 0) + 1
            if seen[s["role"]] > 1:
                s["name"] = f"{s['name']} {seen[s['role']]}"
    tri_surf = surf_of_region[lab]
    rest = tri_surf < 0
    flat_share = float(area[~rest].sum() / max(area.sum(), EPS))
    if rest.any():
        tri_surf[rest] = len(surfaces)
        surfaces.append({"name": ROLE_NAMES["other"], "role": "other", "material": None,
                         "area": round(float(area[rest].sum()), 1)})
    return {
        "vertices": [round(float(v), 3) for v in V.ravel()],
        "triangles": [int(i) for i in T.ravel()],
        "surface": [int(s) for s in tri_surf],
        "surfaces": surfaces,
        "stats": {"input_triangles": int(n_in), "triangles": int(len(T)), "surfaces": len(surfaces),
                  "flat_share": round(flat_share, 3), "area_m2": round(float(area.sum()), 1)},
    }


# ----------------------------------------------------------------------------- runtime

@dataclass
class Plate:
    """A flat surface as one plane plus a raster of where the surface really is.

    Ray casting against a plate is one plane intersection and one lookup, whatever its triangle count, which is
    what makes blocking tests and the ray trace fast. `mask` holds the cells whose centre lies on the surface (used
    to decide whether a reflection really happens); `solid` adds one cell all round, so rays cannot slip through
    the seam where two plates meet.
    """

    surface: int
    n: np.ndarray
    d: float
    u: np.ndarray  # in-plane axes
    w: np.ndarray
    origin: np.ndarray  # (2,) raster corner in (u, w)
    cell: float
    mask: np.ndarray  # (nu, nw) bool
    solid: np.ndarray  # mask grown by one cell
    verts: np.ndarray  # (k, 3) corners of its triangles, for pruning image sources
    area: float

    def inside(self, q: np.ndarray) -> np.ndarray:
        a = np.floor(((q @ self.u) - self.origin[0]) / self.cell).astype(np.int64)
        b = np.floor(((q @ self.w) - self.origin[1]) / self.cell).astype(np.int64)
        ok = (a >= 0) & (b >= 0) & (a < self.mask.shape[0]) & (b < self.mask.shape[1])
        out = np.zeros(len(q), bool)
        out[ok] = self.mask[a[ok], b[ok]]
        return out


def _raster(tri2: np.ndarray, origin: np.ndarray, cell: float, shape: tuple[int, int]) -> np.ndarray:
    """Cells whose centre lies inside any of the 2D triangles (k, 3, 2)."""
    mask = np.zeros(shape, bool)
    for t in tri2:
        lo = np.maximum(np.floor((t.min(axis=0) - origin) / cell).astype(int), 0)
        hi = np.minimum(np.ceil((t.max(axis=0) - origin) / cell).astype(int), shape)
        if np.any(hi <= lo):
            continue
        ia, ib = np.meshgrid(np.arange(lo[0], hi[0]), np.arange(lo[1], hi[1]), indexing="ij")
        p = origin + (np.stack([ia, ib], -1) + 0.5) * cell
        a, b, c = t
        v0, v1 = b - a, c - a
        v2 = p - a
        den = v0[0] * v1[1] - v0[1] * v1[0]
        if abs(den) < EPS:
            continue
        s = (v2[..., 0] * v1[1] - v2[..., 1] * v1[0]) / den
        r = (v0[0] * v2[..., 1] - v0[1] * v2[..., 0]) / den
        tol = 1e-6
        mask[ia, ib] |= (s >= -tol) & (r >= -tol) & (s + r <= 1 + tol)
    return mask


def _grow(mask: np.ndarray) -> np.ndarray:
    m = np.pad(mask, 1)
    out = m.copy()
    for da in (-1, 0, 1):
        for db in (-1, 0, 1):
            out |= np.roll(np.roll(m, da, 0), db, 1)
    return out[1:-1, 1:-1]


TRISET_CHUNK = 16  # triangles per bounding box in a TriSet


def _cross(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.stack([a[..., 1] * b[..., 2] - a[..., 2] * b[..., 1],
                     a[..., 2] * b[..., 0] - a[..., 0] * b[..., 2],
                     a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]], -1)


class TriSet:
    """Triangles tested exactly (Moller-Trumbore), in spatial chunks with bounding boxes.

    The triangles are ordered along a Morton curve and cut into chunks of TRISET_CHUNK; a ray is tested only
    against the triangles of chunks whose box it passes through before its current nearest hit.
    """

    def __init__(self, V: np.ndarray, T: np.ndarray, surf: np.ndarray):
        self.n = len(T)
        if not self.n:
            return
        P = V[T]  # (n, 3, 3)
        cen = P.mean(axis=1)
        lo, hi = cen.min(axis=0), cen.max(axis=0)
        q = np.clip(((cen - lo) / np.maximum(hi - lo, EPS) * 1023).astype(np.int64), 0, 1023)
        code = np.zeros(self.n, np.int64)
        for bit in range(10):
            for ax in range(3):
                code |= ((q[:, ax] >> bit) & 1) << (3 * bit + ax)
        order = np.argsort(code, kind="stable")
        P = P[order]
        self.index = order  # position in this set -> position in the T it was built from
        self.v0 = P[:, 0]
        self.e1 = P[:, 1] - P[:, 0]
        self.e2 = P[:, 2] - P[:, 0]
        cr = _cross(self.e1, self.e2)
        self.normal = cr / np.maximum(np.linalg.norm(cr, axis=1), EPS)[:, None]
        self.surf = surf[order]
        nc = -(-self.n // TRISET_CHUNK)
        pad = nc * TRISET_CHUNK - self.n
        Pp = np.concatenate([P, np.repeat(P[-1:], pad, axis=0)]).reshape(nc, TRISET_CHUNK * 3, 3)
        self.lo = Pp.min(axis=1) - 1e-6
        self.hi = Pp.max(axis=1) + 1e-6

    def surface_of(self, i: np.ndarray) -> np.ndarray:
        return self.surf[i] if self.n else np.full(len(i), -1)

    def normal_of(self, i: np.ndarray) -> np.ndarray:
        return self.normal[i] if self.n else np.zeros((len(i), 3))

    def hits(self, O: np.ndarray, D: np.ndarray, skip: float, t_limit: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Nearest hit (t, triangle) per ray nearer than t_limit; t = inf and triangle 0 where none."""
        R = len(O)
        best = np.full(R, np.inf)
        arg = np.zeros(R, np.int64)
        if not self.n or not R:
            return best, arg
        with np.errstate(divide="ignore", invalid="ignore"):
            inv = 1.0 / D
        C = len(self.lo)
        step = max(1, 2_000_000 // C)
        pairs_r, pairs_c = [], []
        for s0 in range(0, R, step):
            o, iv = O[s0:s0 + step, None, :], inv[s0:s0 + step, None, :]
            with np.errstate(invalid="ignore"):
                t1 = (self.lo[None] - o) * iv
                t2 = (self.hi[None] - o) * iv
            t1 = np.nan_to_num(t1, nan=-np.inf)  # a ray in the plane of a slab face counts as inside it
            t2 = np.nan_to_num(t2, nan=np.inf)
            tn = np.minimum(t1, t2).max(axis=2)
            tf = np.maximum(t1, t2).min(axis=2)
            lim = t_limit[s0:s0 + step, None]
            r, c = np.nonzero((tf >= np.maximum(tn, skip)) & (tn < lim))
            pairs_r.append(r + s0)
            pairs_c.append(c)
        r = np.concatenate(pairs_r)
        if not len(r):
            return best, arg
        c = np.concatenate(pairs_c)
        tri = (c[:, None] * TRISET_CHUNK + np.arange(TRISET_CHUNK)[None, :]).ravel()
        r = np.repeat(r, TRISET_CHUNK)
        keep = tri < self.n
        tri, r = tri[keep], r[keep]
        for s0 in range(0, len(r), 4_000_000):
            rr, tt = r[s0:s0 + 4_000_000], tri[s0:s0 + 4_000_000]
            d = D[rr]
            e1, e2 = self.e1[tt], self.e2[tt]
            p = _cross(d, e2)
            det = np.einsum("ij,ij->i", e1, p)
            ok = np.abs(det) > 1e-12
            inv_det = 1.0 / np.where(ok, det, 1.0)
            tv = O[rr] - self.v0[tt]
            u = np.einsum("ij,ij->i", tv, p) * inv_det
            qv = _cross(tv, e1)
            v = np.einsum("ij,ij->i", d, qv) * inv_det
            t = np.einsum("ij,ij->i", e2, qv) * inv_det
            ok &= (u >= 0) & (v >= 0) & (u + v <= 1) & (t > skip)
            rr, tt, t = rr[ok], tt[ok], t[ok]
            if not len(t):
                continue
            o2 = np.lexsort((t, rr))
            rr, tt, t = rr[o2], tt[o2], t[o2]
            first = np.ones(len(rr), bool)
            first[1:] = rr[1:] != rr[:-1]
            rr, tt, t = rr[first], tt[first], t[first]
            better = t < best[rr]
            best[rr[better]] = t[better]
            arg[rr[better]] = tt[better]
        return best, arg


@dataclass
class MeshRoom:
    V: np.ndarray
    T: np.ndarray
    surf: np.ndarray  # surface index per triangle
    roles: list[str]
    surface_area: np.ndarray  # per surface, m^2
    plates: list[Plate]
    mirrors: list[Plate]  # plates big enough to reflect as mirrors, largest first
    loose: np.ndarray  # triangles not on a plate (furniture, curved parts): tested exactly
    key: str
    _paths: dict = field(default_factory=dict)

    def __post_init__(self):
        P = self.plates
        self._pn = np.array([p.n for p in P]).reshape(-1, 3)
        self._pd = np.array([p.d for p in P])
        self._pu = np.array([p.u for p in P]).reshape(-1, 3)
        self._pw = np.array([p.w for p in P]).reshape(-1, 3)
        self._po = np.array([p.origin for p in P]).reshape(-1, 2)
        self._pc = np.array([p.cell for p in P])
        self._pa = np.array([p.solid.shape[0] for p in P], np.int64)
        self._pb = np.array([p.solid.shape[1] for p in P], np.int64)
        self._poff = np.concatenate([[0], np.cumsum(self._pa * self._pb)])[:-1].astype(np.int64)
        self._pflat = np.concatenate([p.solid.ravel() for p in P]) if P else np.zeros(0, bool)
        self._psurf = np.array([p.surface for p in P], np.int64)
        self._loose = TriSet(self.V, self.T[self.loose], self.surf[self.loose])
        self._every = TriSet(self.V, self.T, self.surf)

    @property
    def area(self) -> float:
        return float(self.surface_area.sum())

    def _plate_hits(self, O, D, skip):
        nd = D @ self._pn.T
        with np.errstate(divide="ignore", invalid="ignore"):
            t = (self._pd[None, :] - O @ self._pn.T) / nd
        ok = np.isfinite(t) & (np.abs(nd) > 1e-12) & (t > skip)
        t = np.where(ok, t, 0.0)
        q = O[:, None, :] + t[..., None] * D[:, None, :]
        a = np.floor((np.einsum("rsk,sk->rs", q, self._pu) - self._po[None, :, 0]) / self._pc[None, :]).astype(np.int64)
        b = np.floor((np.einsum("rsk,sk->rs", q, self._pw) - self._po[None, :, 1]) / self._pc[None, :]).astype(np.int64)
        ok &= (a >= 0) & (b >= 0) & (a < self._pa[None, :]) & (b < self._pb[None, :])
        flat = np.where(ok, self._poff[None, :] + a * self._pb[None, :] + b, 0)
        ok &= self._pflat[flat]
        return np.where(ok, t, np.inf)

    def first_hit(self, O: np.ndarray, D: np.ndarray, t_max: np.ndarray | None = None, skip: float = 1e-4,
                  exact_misses: bool = False):
        """Nearest hit along each ray: (t, surface, unit normal); t = inf where nothing is hit before t_max.

        With `exact_misses`, a ray that found nothing is tested again exactly against every triangle: it may have
        slipped past a plate's raster where the reduced mesh bends, and a traced ray should leave the room only
        through a real opening in the model.
        """
        R = len(O)
        best_t = np.full(R, np.inf)
        best_s = np.full(R, -1, np.int64)
        best_n = np.zeros((R, 3))
        n_p = len(self.plates)
        step = max(1, int(1_000_000 // max(n_p, 1)))
        for s0 in range(0, R, step):
            sl = slice(s0, s0 + step)
            o, dd = O[sl], D[sl]
            rows = np.arange(len(o))
            tpm = np.full(len(o), np.inf)
            ip = np.zeros(len(o), np.int64)
            if n_p:
                tp = self._plate_hits(o, dd, skip)
                ip = np.argmin(tp, axis=1)
                tpm = tp[rows, ip]
            tlm, il = self._loose.hits(o, dd, skip, tpm)
            use_l = tlm < tpm
            t = np.where(use_l, tlm, tpm)
            if t_max is not None:
                t = np.where(t < t_max[sl], t, np.inf)
            hit = np.isfinite(t)
            best_t[sl] = t
            sp = self._psurf[ip] if n_p else np.full(len(o), -1)
            best_s[sl] = np.where(hit, np.where(use_l, self._loose.surface_of(il), sp), -1)
            np_ = self._pn[ip] if n_p else np.zeros((len(o), 3))
            best_n[sl] = np.where(use_l[:, None], self._loose.normal_of(il), np_)
        if exact_misses:
            rows = np.nonzero(~np.isfinite(best_t))[0]
            if len(rows):
                t, it = self._every.hits(O[rows], D[rows], skip, np.full(len(rows), np.inf))
                if t_max is not None:
                    t = np.where(t < t_max[rows], t, np.inf)
                hit = np.isfinite(t)
                best_t[rows[hit]] = t[hit]
                best_s[rows[hit]] = self._every.surface_of(it[hit])
                best_n[rows[hit]] = self._every.normal_of(it[hit])
        return best_t, best_s, best_n

    def blocked(self, A: np.ndarray, B: np.ndarray, margin: float = 0.03) -> np.ndarray:
        """Whether the straight path from each A to each B passes through a surface (ends excluded)."""
        A = np.broadcast_to(A, B.shape) if A.ndim == 1 else A
        vec = B - A
        L = np.linalg.norm(vec, axis=1)
        out = np.zeros(len(B), bool)
        live = L > 2 * margin
        if not live.any():
            return out
        D = vec[live] / L[live, None]
        O = A[live] + D * margin
        t, _, _ = self.first_hit(O, D, L[live] - 2 * margin, skip=0.0)
        out[live] = np.isfinite(t)
        return out


_MESHES: OrderedDict[str, MeshRoom] = OrderedDict()
PLATE_MIN_FLAT = 0.5  # share of a surface's area that must lie on its plane for it to be cast as a plate


def plate_tolerance(cell: float) -> float:
    """How far a triangle's corners may stray from its surface's plane and still be cast as part of the plate.

    A ray meets the plane, not the triangle, so the point it finds is off by about this much times the tangent of
    the angle of incidence; the plate's solid raster is one cell bigger all round, which covers up to half a cell.
    Triangles further out (a scan's bumps, corners the triangle budget squeezed) are tested exactly, one by one.
    """
    return max(0.02, 0.5 * cell)


def shape_key(shape) -> str:
    h = hashlib.sha1()
    for part in (shape.vertices, shape.triangles, shape.surface):
        h.update(np.asarray(part, np.float64).tobytes())
    h.update(",".join(s.role for s in shape.surfaces).encode())
    return h.hexdigest()


def mesh_room(shape) -> MeshRoom:
    """Runtime form of a `RoomShape` (cached by content)."""
    key = shape_key(shape)
    hit = _MESHES.get(key)
    if hit is not None:
        _MESHES.move_to_end(key)
        return hit
    V = np.asarray(shape.vertices, float).reshape(-1, 3)
    T = np.asarray(shape.triangles, np.int64).reshape(-1, 3)
    surf = np.asarray(shape.surface, np.int64)
    roles = [s.role for s in shape.surfaces]
    area, nrm = _tri_area_normal(V, T)
    s_area = np.bincount(surf, area, len(roles))
    plates: list[Plate] = []
    on_plate = np.zeros(len(T), bool)
    for s in np.argsort(-s_area):
        if roles[s] == "other" or s_area[s] <= 0:
            continue
        tris = np.nonzero(surf == s)[0]
        w = area[tris]
        ref = nrm[tris[np.argmax(w)]]
        n = (nrm[tris] * np.where(nrm[tris] @ ref >= 0, 1.0, -1.0)[:, None] * w[:, None]).sum(axis=0)
        n /= max(np.linalg.norm(n), EPS)
        d = float(np.average(V[T[tris]].mean(axis=1) @ n, weights=w))
        cell = float(np.clip(np.sqrt(s_area[s]) / 150.0, 0.03, 0.25))
        for _ in range(2):  # keep the triangles on the plane, refit the plane to them
            flat = np.max(np.abs(V[T[tris]] @ n - d), axis=1) <= plate_tolerance(cell)
            if w[flat].sum() < PLATE_MIN_FLAT * w.sum():
                break
            d = float(np.average(V[T[tris[flat]]].mean(axis=1) @ n, weights=w[flat]))
        flat = np.max(np.abs(V[T[tris]] @ n - d), axis=1) <= plate_tolerance(cell)
        if w[flat].sum() < PLATE_MIN_FLAT * w.sum():
            continue
        tris, w = tris[flat], w[flat]
        pts = V[T[tris]].reshape(-1, 3)
        u = np.cross(n, [0.0, 0.0, 1.0] if abs(n[2]) < 0.9 else [1.0, 0.0, 0.0])
        u /= np.linalg.norm(u)
        wv = np.cross(n, u)
        p2 = np.stack([V[T[tris]] @ u, V[T[tris]] @ wv], -1)  # (k, 3, 2)
        lo = p2.reshape(-1, 2).min(axis=0) - cell
        hi = p2.reshape(-1, 2).max(axis=0) + cell
        shp = tuple(int(x) for x in np.maximum(np.ceil((hi - lo) / cell), 1))
        mask = _raster(p2, lo, cell, shp)
        plates.append(Plate(int(s), n, d, u, wv, lo, cell, mask, _grow(mask), pts, float(w.sum())))
        on_plate[tris] = True
    mirrors = [p for p in plates if p.area >= MIN_REFLECTOR_M2]
    m = MeshRoom(V, T, surf, roles, s_area, plates, mirrors, np.nonzero(~on_plate)[0], key)
    _MESHES[key] = m
    while len(_MESHES) > 4:
        _MESHES.popitem(last=False)
    return m


# ----------------------------------------------------------------------------- image sources

@dataclass
class MeshImage:
    pos: np.ndarray
    M: np.ndarray  # emission direction at the real source = M @ (receiver - image) direction
    chain: tuple[int, ...]  # mirror indices (into MeshRoom.mirrors), first reflection first
    images: tuple[np.ndarray, ...]  # the source, then each successive image (for back-tracing)
    order: int


def mesh_images(src: np.ndarray, mesh: MeshRoom, max_order: int) -> list[MeshImage]:
    I3 = np.eye(3)
    out = [MeshImage(src, I3, (), (src,), 0)]
    if max_order < 1:
        return out
    first: list[MeshImage] = []
    for k, mi in enumerate(mesh.mirrors[:MAX_FIRST_ORDER]):
        dist = float(mi.n @ src - mi.d)
        if abs(dist) < 1e-3:
            continue
        H = I3 - 2.0 * np.outer(mi.n, mi.n)
        pos = src - 2.0 * dist * mi.n
        first.append(MeshImage(pos, H, (k,), (src, pos), 1))
    out += first
    if max_order < 2:
        return out
    for im in first:
        k1 = im.chain[0]
        if k1 >= MAX_SECOND_ORDER:
            continue
        m1 = mesh.mirrors[k1]
        side = np.sign(m1.n @ src - m1.d)  # after the first bounce the sound travels on the source's side
        for k2, m2 in enumerate(mesh.mirrors[:MAX_SECOND_ORDER]):
            if k2 == k1:
                continue
            if not np.any(side * (m2.verts @ m1.n - m1.d) > 0.05):
                continue  # the second surface lies wholly behind the first: unreachable
            dist = float(m2.n @ im.pos - m2.d)
            if abs(dist) < 1e-3:
                continue
            H2 = I3 - 2.0 * np.outer(m2.n, m2.n)
            pos = im.pos - 2.0 * dist * m2.n
            out.append(MeshImage(pos, im.M @ H2, (k1, k2), (src, im.pos, pos), 2))
    return out


def path_valid(im: MeshImage, points: np.ndarray, mesh: MeshRoom, check_blocking: bool = True) -> np.ndarray:
    """Receivers this image really reaches: each reflection point inside its surface, in order, legs unblocked."""
    ok = np.ones(len(points), bool)
    cur = points
    legs = []
    for j in range(im.order, 0, -1):
        mi = mesh.mirrors[im.chain[j - 1]]
        target = im.images[j]
        dv = target - cur
        den = dv @ mi.n
        with np.errstate(divide="ignore", invalid="ignore"):
            t = (mi.d - cur @ mi.n) / den
        ok &= np.isfinite(t) & (t > 1e-6) & (t < 1 - 1e-6)
        q = cur + np.where(ok, t, 0.0)[:, None] * dv
        ok &= mi.inside(q)
        legs.append((cur, q))
        cur = q
        if not ok.any():
            return ok
    legs.append((cur, im.images[0]))
    if check_blocking:
        for a, b in legs:
            rows = np.nonzero(ok)[0]
            if not len(rows):
                break
            bb = np.broadcast_to(b, a.shape) if np.ndim(b) == 1 else b
            ok[rows] &= ~mesh.blocked(a[rows], np.ascontiguousarray(bb[rows]))
    return ok


# ----------------------------------------------------------------------------- ray tracing

@dataclass
class Paths:
    surf: np.ndarray  # (R, K) surface hit at each bounce; -1 after the ray escaped or the trace ended
    dist: np.ndarray  # (R, K) path length at each hit
    escaped_at: np.ndarray  # (R,) path length where the ray left the model (inf if it never did)
    traced: float  # path length every surviving ray reached
    mfp: float
    escaped: float  # fraction of rays that left through an opening


def _sphere_dirs(n: int) -> np.ndarray:
    i = np.arange(n) + 0.5
    phi = np.arccos(1 - 2 * i / n)
    th = np.pi * (1 + 5 ** 0.5) * i
    return np.column_stack([np.cos(th) * np.sin(phi), np.sin(th) * np.sin(phi), np.cos(phi)])


def trace(mesh: MeshRoom, start: np.ndarray, n_rays: int = 384, bounces: int = 120, scattering: float = 0.25,
          seed: int = 7) -> Paths:
    """Seeded ray trace from `start`; the hit lists do not depend on the materials (cached per shape)."""
    key = (tuple(np.round(start, 2)), n_rays, bounces, scattering, seed)
    if key in mesh._paths:
        return mesh._paths[key]
    rng = np.random.default_rng(seed)
    O = np.tile(np.asarray(start, float), (n_rays, 1))
    D = _sphere_dirs(n_rays)
    surf = np.full((n_rays, bounces), -1, np.int64)
    dist = np.zeros((n_rays, bounces))
    total = np.zeros(n_rays)
    escaped_at = np.full(n_rays, np.inf)
    alive = np.ones(n_rays, bool)
    seg = []
    for k in range(bounces):
        idx = np.nonzero(alive)[0]
        if not len(idx):
            break
        t, hs, n = mesh.first_hit(O[idx], D[idx], exact_misses=True)
        miss = ~np.isfinite(t)
        escaped_at[idx[miss]] = total[idx[miss]]
        alive[idx[miss]] = False
        idx, t, hs, n = idx[~miss], t[~miss], hs[~miss], n[~miss]
        seg.append(t)
        total[idx] += t
        surf[idx, k] = hs
        dist[idx, k] = total[idx]
        O[idx] = O[idx] + t[:, None] * D[idx]
        n = n * np.where(np.einsum("ij,ij->i", n, D[idx]) > 0, -1.0, 1.0)[:, None]  # facing the incoming ray
        spec = D[idx] - 2.0 * np.einsum("ij,ij->i", D[idx], n)[:, None] * n
        diffuse = rng.random(len(idx)) < scattering
        if diffuse.any():  # Lambert: cosine-weighted around the normal
            nd = n[diffuse]
            r1, r2 = rng.random(len(nd)), rng.random(len(nd))
            a = np.cross(nd, np.where(np.abs(nd[:, :1]) < 0.9, [[1.0, 0, 0]], [[0, 1.0, 0]]))
            a /= np.linalg.norm(a, axis=1, keepdims=True)
            b = np.cross(nd, a)
            phi, s = 2 * np.pi * r1, np.sqrt(r2)
            spec[diffuse] = (a * (np.cos(phi) * s)[:, None] + b * (np.sin(phi) * s)[:, None]
                             + nd * np.sqrt(1 - r2)[:, None])
        D[idx] = spec / np.linalg.norm(spec, axis=1, keepdims=True)
        O[idx] += n * 1e-4
    seg_all = np.concatenate(seg) if seg else np.zeros(0)
    mfp = float(seg_all.mean()) if len(seg_all) else 0.0
    traced = float(total[alive].min()) if alive.any() else float(np.max(escaped_at[np.isfinite(escaped_at)], initial=0.0))
    p = Paths(surf, dist, escaped_at, traced, mfp, float((~alive).mean() if n_rays else 0.0))
    if len(mesh._paths) > 8:
        mesh._paths.clear()
    mesh._paths[key] = p
    return p


def decay_rt60(paths: Paths, alpha: np.ndarray, air_m: np.ndarray, c: float) -> np.ndarray:
    """RT60 per band from the traced energy decay.

    alpha: (surfaces, B) absorption; air_m: (B,) intensity attenuation per metre. A T20 fit (-5 to -25 dB) of the
    energy left in the room; where the trace is too short for -25 dB, whatever it reached, down to at least -10 dB.
    Bands whose decay cannot be read come back as NaN.
    """
    R, K = paths.surf.shape
    B = alpha.shape[1]
    hit = paths.surf >= 0
    la = np.zeros((R, K, B))
    la[hit] = np.log(np.clip(1.0 - alpha[paths.surf[hit]], 1e-12, 1.0))
    cum = np.cumsum(la, axis=1)
    x_end = paths.traced
    if not np.isfinite(x_end) or x_end <= 0:
        return np.full(B, np.nan)
    xs = np.linspace(0.0, x_end, 400)
    # hits before each x, per ray (dist rows are increasing over the hits that happened)
    dist = np.where(hit, paths.dist, np.inf)
    n_before = (dist[:, :, None] <= xs[None, None, :]).sum(axis=1)  # (R, X)
    out = np.full(B, np.nan)
    gone = xs[None, :] >= paths.escaped_at[:, None]  # (R, X)
    for b in range(B):
        lc = np.concatenate([np.zeros((R, 1)), cum[:, :, b]], axis=1)
        e = np.exp(np.take_along_axis(lc, n_before, axis=1))
        e[gone] = 0.0
        E = e.sum(axis=0) * np.exp(-air_m[b] * xs)
        if E[0] <= 0:
            continue
        L = 10 * np.log10(np.maximum(E / E[0], 1e-30))
        lo = -5.0
        hi = max(-25.0, float(L.min()))
        if hi > -10.0:
            continue
        sel = (L <= lo) & (L >= hi)
        if sel.sum() < 5:
            continue
        slope = np.polyfit(xs[sel] / c, L[sel], 1)[0]  # dB per second
        if slope < 0:
            out[b] = -60.0 / slope
    return out
