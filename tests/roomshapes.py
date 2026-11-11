"""Room meshes for the shape tests: a box, built from grids of triangles, with optional noise and a partition."""

from __future__ import annotations

import numpy as np


def grid_quad(o, du, dv, nu, nv):
    o, du, dv = (np.asarray(x, float) for x in (o, du, dv))
    V = np.array([o + du * i / nu + dv * j / nv for j in range(nv + 1) for i in range(nu + 1)])
    T = []
    for j in range(nv):
        for i in range(nu):
            a = j * (nu + 1) + i
            T += [[a, a + 1, a + nu + 2], [a, a + nu + 2, a + nu + 1]]
    return V, np.array(T)


def box_mesh(W=12.0, L=20.0, H=5.0, step=1.0, noise=0.0, seed=1):
    parts = [
        ((0, 0, 0), (W, 0, 0), (0, L, 0)),  # floor
        ((0, 0, H), (W, 0, 0), (0, L, 0)),  # ceiling
        ((0, 0, 0), (W, 0, 0), (0, 0, H)),  # front (y = 0)
        ((0, L, 0), (W, 0, 0), (0, 0, H)),  # back
        ((0, 0, 0), (0, L, 0), (0, 0, H)),  # left (x = 0)
        ((W, 0, 0), (0, L, 0), (0, 0, H)),  # right
    ]
    Vs, Ts, off = [], [], 0
    for o, du, dv in parts:
        nu = max(1, int(round(np.linalg.norm(du) / step)))
        nv = max(1, int(round(np.linalg.norm(dv) / step)))
        V, T = grid_quad(o, du, dv, nu, nv)
        Vs.append(V)
        Ts.append(T + off)
        off += len(V)
    V, T = np.vstack(Vs), np.vstack(Ts)
    # weld the shared edges, as a scan's mesh would be
    key = np.round(V, 6)
    _, first, inv = np.unique(key, axis=0, return_index=True, return_inverse=True)
    V = V[first]
    T = inv.reshape(-1)[T]
    if noise:
        V = V + np.random.default_rng(seed).normal(0, noise, V.shape)
    return V, T
