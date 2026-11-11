"""Sound-field model.

The expensive physics (geometry, reflections, directivity, air and wall
absorption) depends only on where speakers are and what they are, never on
their DSP settings. For every speaker s, listener point p and frequency f we
precompute and cache:

  H[s, p, f]  complex  direct sound + reflections arriving within EARLY_MS of
                       that speaker's direct sound (summed coherently)
  L[s, p, f]  power    later modelled reflections (summed as energy)
  E[s, f]     power    statistical diffuse tail beyond the modelled order

Applying a DSP state (delay / gain / polarity / EQ) is then a weighted sum,
which is what makes live editing and population-based optimization fast.
Per-speaker results are cached, so moving one speaker only recomputes one.

Model per speaker:
  * shoebox image sources up to `reflection_order` (Allen & Berkley) with
    frequency-dependent wall reflection |R| = sqrt(1 - alpha)
  * frequency-dependent directivity: separable raised-cosine lobes whose -6 dB
    full angle equals the nominal coverage above `directivity_hz`, relaxing to
    omnidirectional at low frequencies
  * ISO 9613-1 air absorption on every path
"""

from __future__ import annotations

import hashlib
import itertools
import threading
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass

import numpy as np

from .air import air_absorption_db_per_m, speed_of_sound
from .eq import peq_response
from .materials import SURFACES, RoomStats, room_stats, surface_alphas
from .model import Project, Speaker

P_REF = 20e-6
PPO = 12  # frequency points per octave of the field grid
EARLY_MS = 50.0  # reflections later than this after the direct sound count as late energy
MAX_GRID_POINTS = 2600
PREVIEW_GRID_POINTS = 640
MIN_DISTANCE = 0.25
LOG_REAR_FLOOR = float(np.log(10 ** (-24 / 20)))


# --------------------------------------------------------------------------- grids

def freq_grid(f_min: float, f_max: float, points_per_octave: int = PPO) -> np.ndarray:
    n = int(np.ceil(np.log2(f_max / f_min) * points_per_octave)) + 1
    return f_min * 2.0 ** (np.arange(n) / points_per_octave)


@dataclass
class Grid:
    xs: np.ndarray
    ys: np.ndarray
    z: float

    @property
    def nx(self) -> int:
        return len(self.xs)

    @property
    def ny(self) -> int:
        return len(self.ys)

    def points(self) -> np.ndarray:
        X, Y = np.meshgrid(self.xs, self.ys)  # row-major: y rows, x columns
        return np.column_stack([X.ravel(), Y.ravel(), np.full(X.size, self.z)])


def audience_grid(project: Project, max_points: int = MAX_GRID_POINTS) -> Grid:
    a = project.audience
    w, l = a.x1 - a.x0, a.y1 - a.y0
    res = max(project.settings.grid_resolution, float(np.sqrt(w * l / max_points)))
    nx = max(2, int(round(w / res)) + 1)
    ny = max(2, int(round(l / res)) + 1)
    return Grid(np.linspace(a.x0, a.x1, nx), np.linspace(a.y0, a.y1, ny), a.ear_height)


# --------------------------------------------------------------------- directivity

def orientation(yaw_deg: float, pitch_deg: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    yw, pt = np.radians(yaw_deg), np.radians(pitch_deg)
    fwd = np.array([np.sin(yw) * np.cos(pt), np.cos(yw) * np.cos(pt), np.sin(pt)])
    right = np.array([np.cos(yw), -np.sin(yw), 0.0])
    up = np.cross(right, fwd)
    return fwd, right, up


def lobe_exponent(coverage_deg: float) -> float:
    """Exponent n so that ((1 + cos(theta)) / 2)^n = 0.5 (-6 dB) at theta = coverage / 2."""
    half = np.radians(min(coverage_deg, 340.0) / 2.0)
    return float(np.log(0.5) / np.log((1.0 + np.cos(half)) / 2.0))


def directivity_weight(freqs: np.ndarray, directivity_hz: float) -> np.ndarray:
    """0 at low frequency (omni) -> 1 above the pattern-control frequency."""
    r = (np.asarray(freqs, float) / directivity_hz) ** 2
    return r / (1.0 + r)


def log_directivity(spk: Speaker, dirs: np.ndarray, freqs: np.ndarray) -> np.ndarray:
    """Natural log of the pressure directivity, shape (N, F); 0 on axis."""
    fwd, right, up = orientation(spk.yaw, spk.pitch)
    a, b, c = dirs @ fwd, dirs @ right, dirs @ up
    lh = np.log(np.maximum((1.0 + np.cos(np.arctan2(b, a))) / 2.0, 1e-9))
    lv = np.log(np.maximum((1.0 + np.cos(np.arctan2(c, np.hypot(a, b)))) / 2.0, 1e-9))
    w = directivity_weight(freqs, spk.directivity_hz)
    out = np.outer(lh, lobe_exponent(spk.coverage_h) * w) + np.outer(lv, lobe_exponent(spk.coverage_v) * w)
    return np.maximum(out, LOG_REAR_FLOOR)


def directivity(spk: Speaker, dirs: np.ndarray, freqs: np.ndarray) -> np.ndarray:
    return np.exp(log_directivity(spk, dirs, freqs))


_SPHERE: np.ndarray | None = None


def _sphere(n: int = 1500) -> np.ndarray:
    global _SPHERE
    if _SPHERE is None:
        i = np.arange(n) + 0.5
        phi = np.arccos(1 - 2 * i / n)
        th = np.pi * (1 + 5 ** 0.5) * i
        _SPHERE = np.column_stack([np.cos(th) * np.sin(phi), np.sin(th) * np.sin(phi), np.cos(phi)])
    return _SPHERE


def directivity_factor(spk: Speaker, freqs: np.ndarray) -> np.ndarray:
    """Q(f) = 4 pi / integral(D^2 dOmega)."""
    d = directivity(spk, _sphere(), freqs)
    return 1.0 / np.mean(d * d, axis=0)


def box_response(spk: Speaker, freqs: np.ndarray) -> np.ndarray:
    """Loudspeaker band-limiting: 4th-order (LR4-shaped) high-pass / low-pass."""
    f = np.asarray(freqs, float)
    h = np.ones(len(f), dtype=complex)
    if spk.hp_hz > 0:
        s = 1j * f / spk.hp_hz
        bw2 = s * s / (s * s + np.sqrt(2) * s + 1)
        h *= bw2 * bw2
    if spk.lp_hz > 0:
        s = 1j * f / spk.lp_hz
        bw2 = 1 / (s * s + np.sqrt(2) * s + 1)
        h *= bw2 * bw2
    return h


def speaker_amplitude(spk: Speaker) -> float:
    """On-axis pressure amplitude at 1 m, Pa."""
    return P_REF * 10.0 ** (spk.level_db / 20.0)


# ---------------------------------------------------------------- image sources

@dataclass
class ImageSource:
    pos: np.ndarray
    flip: np.ndarray  # +-1 per axis: maps the emission direction back into the real source's frame
    hits: dict[str, int]
    order: int


_WALLS = (("left", "right"), ("front", "back"), ("floor", "ceiling"))


def image_sources(src: np.ndarray, dims: tuple[float, float, float], max_order: int) -> list[ImageSource]:
    out: list[ImageSource] = []
    rng = range(-max_order, max_order + 1)
    for n in itertools.product(rng, rng, rng):
        for u in itertools.product((0, 1), repeat=3):
            hits: dict[str, int] = {}
            order = 0
            pos = np.empty(3)
            flip = np.empty(3)
            for ax in range(3):
                lo, hi = _WALLS[ax]
                k_lo, k_hi = abs(n[ax] - u[ax]), abs(n[ax])
                order += k_lo + k_hi
                if k_lo:
                    hits[lo] = k_lo
                if k_hi:
                    hits[hi] = k_hi
                sgn = 1 - 2 * u[ax]
                pos[ax] = sgn * src[ax] + 2 * n[ax] * dims[ax]
                flip[ax] = sgn
            if order <= max_order:
                out.append(ImageSource(pos, flip, hits, order))
    out.sort(key=lambda im: im.order)
    return out


# -------------------------------------------------------------- per-speaker field

@dataclass
class Context:
    """Everything shared by all speakers for one room / environment / frequency grid."""

    freqs: np.ndarray
    c: float
    air_np: np.ndarray  # pressure attenuation, nepers per metre
    k: np.ndarray  # wavenumber
    log_beta: dict[str, np.ndarray]
    dims: tuple[float, float, float]
    order: int
    stats: RoomStats
    key: str


def make_context(project: Project, freqs: np.ndarray | None = None) -> Context:
    env, st, room = project.environment, project.settings, project.room
    if freqs is None:
        freqs = freq_grid(st.f_min, min(st.f_max, 20000.0))
    c = speed_of_sound(env.temperature_c, env.humidity_pct, env.pressure_kpa)
    air_db = air_absorption_db_per_m(freqs, env.temperature_c, env.humidity_pct, env.pressure_kpa)
    alphas = surface_alphas(project, freqs)
    log_beta = {s: 0.5 * np.log(1.0 - alphas[s]) for s in SURFACES}
    stats = room_stats(room.width, room.length, room.height, alphas, freqs, air_db)
    mat_key = ",".join(f"{v:.4f}" for s in SURFACES for v in alphas[s][:: max(1, len(freqs) // 24)])
    key = "|".join([room.model_dump_json(), env.model_dump_json(), str(st.reflection_order), mat_key,
                    f"{freqs[0]:.3f}", f"{freqs[-1]:.3f}", str(len(freqs))])
    return Context(freqs, c, air_db / 20.0 * np.log(10.0), 2 * np.pi * freqs / c, log_beta,
                   (room.width, room.length, room.height), st.reflection_order, stats, key)


def source_position(spk: Speaker, dims: tuple[float, float, float]) -> np.ndarray:
    return np.array([np.clip(spk.x, 0.01, dims[0] - 0.01),
                     np.clip(spk.y, 0.01, dims[1] - 0.01),
                     np.clip(spk.z, 0.01, dims[2] - 0.01)])


@dataclass
class SpeakerField:
    H: np.ndarray  # (P, F) complex64
    L: np.ndarray  # (P, F) float32
    E: np.ndarray  # (F,) float64
    dist: np.ndarray  # (P,) direct path length


def speaker_field(spk: Speaker, points: np.ndarray, ctx: Context, early_ms: float = EARLY_MS) -> SpeakerField:
    """Unit-DSP field of one speaker at `points`."""
    freqs = ctx.freqs
    src = source_position(spk, ctx.dims)
    P, F = len(points), len(freqs)
    H = np.zeros((P, F), dtype=np.complex64)
    L = np.zeros((P, F), dtype=np.float32)
    d0 = np.maximum(np.linalg.norm(points - src, axis=1), MIN_DISTANCE)
    early_limit = d0 + early_ms * 1e-3 * ctx.c
    for im in image_sources(src, ctx.dims, ctx.order):
        vec = points - im.pos
        d = np.maximum(np.linalg.norm(vec, axis=1), MIN_DISTANCE)
        early = d <= early_limit + 1e-9
        emit = vec * im.flip / d[:, None]
        la = log_directivity(spk, emit, freqs)
        for surf, cnt in im.hits.items():
            la += cnt * ctx.log_beta[surf][None, :]
        la -= np.outer(d, ctx.air_np)
        la -= np.log(d)[:, None]
        if early.all():
            H += np.exp(la - 1j * np.outer(d, ctx.k)).astype(np.complex64)
        elif not early.any():
            L += np.exp(2.0 * la).astype(np.float32)
        else:
            e, n = np.nonzero(early)[0], np.nonzero(~early)[0]
            H[e] += np.exp(la[e] - 1j * np.outer(d[e], ctx.k)).astype(np.complex64)
            L[n] += np.exp(2.0 * la[n]).astype(np.float32)
    box = box_response(spk, freqs)
    amp = speaker_amplitude(spk)
    H *= (box * amp).astype(np.complex64)[None, :]
    box2 = np.abs(box) ** 2
    L *= (box2 * amp * amp).astype(np.float32)[None, :]
    remaining = (1.0 - ctx.stats.mean_alpha) ** ctx.order
    E = amp * amp * 16 * np.pi / (directivity_factor(spk, freqs) * ctx.stats.room_constant) * remaining * box2
    return SpeakerField(H, L, E, np.linalg.norm(points - src, axis=1))


# ------------------------------------------------------------------ field model

@dataclass
class FieldModel:
    freqs: np.ndarray  # (F,)
    points: np.ndarray  # (P, 3); the first n_grid rows are the audience grid, the rest are mics
    grid: Grid
    n_grid: int
    speaker_ids: list[str]
    H: np.ndarray  # (S, P, F) complex64
    L: np.ndarray  # (S, P, F) float32
    E_rev: np.ndarray  # (S, F)
    dist: np.ndarray  # (S, P)
    c: float
    rt60: np.ndarray  # (F,) Eyring
    _H2: np.ndarray | None = None
    _LE: np.ndarray | None = None

    @property
    def grid_slice(self) -> slice:
        return slice(0, self.n_grid)

    @property
    def H2(self) -> np.ndarray:
        if self._H2 is None:
            self._H2 = (self.H.real ** 2 + self.H.imag ** 2).astype(np.float32)
        return self._H2

    @property
    def LE(self) -> np.ndarray:
        """All incoherent late energy per speaker: late reflections + diffuse tail."""
        if self._LE is None:
            self._LE = (self.L + self.E_rev[:, None, :].astype(np.float32)).astype(np.float32)
        return self._LE


class _LRU:
    def __init__(self, maxsize: int):
        self.maxsize = maxsize
        self._d: OrderedDict[str, object] = OrderedDict()

    def get(self, key: str):
        if key in self._d:
            self._d.move_to_end(key)
            return self._d[key]
        return None

    def put(self, key: str, value) -> None:
        self._d[key] = value
        self._d.move_to_end(key)
        while len(self._d) > self.maxsize:
            self._d.popitem(last=False)

    def clear(self) -> None:
        self._d.clear()


_SPEAKER_CACHE = _LRU(40)
_FIELD_CACHE = _LRU(6)
_LOCK = threading.RLock()

_GEOMETRY_FIELDS = {"x", "y", "z", "yaw", "pitch", "coverage_h", "coverage_v",
                    "level_db", "hp_hz", "lp_hz", "directivity_hz"}


def _sha(*parts: str) -> str:
    return hashlib.sha1("|".join(parts).encode()).hexdigest()


def speaker_key(spk: Speaker) -> str:
    return repr(sorted(spk.model_dump(include=_GEOMETRY_FIELDS).items()))


def clear_cache() -> None:
    with _LOCK:
        _SPEAKER_CACHE.clear()
        _FIELD_CACHE.clear()


def get_field(project: Project, preview: bool = False) -> FieldModel:
    """Field model for a project, reusing cached per-speaker results wherever possible."""
    ctx = make_context(project)
    grid = audience_grid(project, PREVIEW_GRID_POINTS if preview else MAX_GRID_POINTS)
    grid_key = repr((grid.nx, grid.ny, float(grid.xs[0]), float(grid.xs[-1]),
                     float(grid.ys[0]), float(grid.ys[-1]), grid.z))
    mic_key = repr([(m.x, m.y, m.z) for m in project.mics])
    spk_keys = [speaker_key(s) for s in project.speakers]
    full_key = _sha(ctx.key, grid_key, mic_key, *spk_keys)
    with _LOCK:
        cached = _FIELD_CACHE.get(full_key)
        if cached is not None:
            if cached.speaker_ids != [s.id for s in project.speakers]:
                cached.speaker_ids = [s.id for s in project.speakers]
            return cached
        gp = grid.points()
        mp = np.array([[m.x, m.y, m.z] for m in project.mics], dtype=float).reshape(-1, 3)
        keys = [_sha(ctx.key, grid_key, k) for k in spk_keys]
        missing = [i for i, k in enumerate(keys) if _SPEAKER_CACHE.get(k) is None]
        if missing:
            def work(i: int) -> SpeakerField:
                return speaker_field(project.speakers[i], gp, ctx)

            if len(missing) == 1:
                results = [work(missing[0])]
            else:
                with ThreadPoolExecutor(max_workers=min(len(missing), 8)) as ex:
                    results = list(ex.map(work, missing))
            for i, sf in zip(missing, results):
                _SPEAKER_CACHE.put(keys[i], sf)
        S, F = len(project.speakers), len(ctx.freqs)
        P = len(gp) + len(mp)
        H = np.zeros((S, P, F), dtype=np.complex64)
        L = np.zeros((S, P, F), dtype=np.float32)
        E = np.zeros((S, F))
        dist = np.zeros((S, P))
        for i, spk in enumerate(project.speakers):
            sf = _SPEAKER_CACHE.get(keys[i])
            if sf is None:  # evicted between put and get (more speakers than cache slots)
                sf = speaker_field(spk, gp, ctx)
            H[i, :len(gp)], L[i, :len(gp)], E[i], dist[i, :len(gp)] = sf.H, sf.L, sf.E, sf.dist
            if len(mp):
                mf = speaker_field(spk, mp, ctx)
                H[i, len(gp):], L[i, len(gp):], dist[i, len(gp):] = mf.H, mf.L, mf.dist
        fm = FieldModel(ctx.freqs, np.vstack([gp, mp]), grid, len(gp), [s.id for s in project.speakers],
                        H, L, E, dist, ctx.c, ctx.stats.rt60_eyring)
        _FIELD_CACHE.put(full_key, fm)
        return fm


# ----------------------------------------------------------------- DSP weights

def dsp_weights(speakers: list[Speaker], freqs: np.ndarray) -> np.ndarray:
    """Complex DSP transfer per speaker, shape (S, F)."""
    w = np.zeros((len(speakers), len(freqs)), dtype=complex)
    for i, s in enumerate(speakers):
        if s.mute:
            continue
        g = 10 ** (s.gain_db / 20) * (-1.0 if s.invert else 1.0)
        w[i] = g * np.exp(-2j * np.pi * freqs * s.delay_ms * 1e-3) * peq_response(s.peq, freqs)
    return w


@dataclass
class Combined:
    """One DSP state applied to a field. Powers in Pa^2, shape (P, F)."""

    early: np.ndarray  # |sum_s w_s H_s|^2  coherent direct + early reflections
    early_psum: np.ndarray  # sum_s |w_s H_s|^2  same energy with no interference between speakers
    late: np.ndarray  # incoherent late energy
    coh: np.ndarray  # complex coherent pressure

    @property
    def total(self) -> np.ndarray:
        return self.early + self.late

    def rows(self, sl) -> Combined:
        return Combined(self.early[sl], self.early_psum[sl], self.late[sl], self.coh[sl])


def combine(field: FieldModel, weights: np.ndarray) -> Combined:
    coh = np.einsum("sf,spf->pf", weights.astype(np.complex64), field.H)
    w2 = (np.abs(weights) ** 2).astype(np.float32)
    psum = np.einsum("sf,spf->pf", w2, field.H2)
    late = np.einsum("sf,spf->pf", w2, field.LE)
    return Combined(coh.real ** 2 + coh.imag ** 2, psum, late, coh)
