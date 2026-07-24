"""The vortex registry: seeded storm populations with latitude-conditioned
placement, power-law sizes, and Poisson-disc longitude spacing.

Each vortex contributes a Gaussian streamfunction
    psi_v = strength * exp(-(d/r_core)^2),  d = great-circle distance,
which yields a smooth coherent rotating core (tangential speed peaks at
d = r_core/sqrt(2)) and composes trivially in psi. THE PSI-AMPLITUDE TRAP:
because u = -dpsi/dphi and zeta = +laplacian(psi), a vortex's own core
vorticity is the OPPOSITE sign of its `strength` field: omega = -sign(strength).
Every seeded storm class in this module CO-ROTATES with the local ambient
shear vorticity (all of them: heroes, ovals, barges, pearls, accents,
companions, small storms) -- that is what lets a storm persist against
differential shear rather than getting torn apart by it; barges persist
inside belts by the same co-rotation rule, then get stretched into cigars by
the jets' shear gradient (not by an opposing rotation). Given the trap above,
co-rotation means `strength = -_ambient_sign(profiles, lat) * |magnitude|`
everywhere in this file -- the leading minus is not optional. Polar caps are
cyclonic per hemisphere (sign f, i.e. sign(pole_sign)), which under the same
trap means `s_central = -pole_sign * |magnitude|` too.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from gasgiant.params.model import (
    CAST_LEVER_SPECS,
    CastKind,
    PolesParams,
    StormsParams,
    WakeDir,
    hero_latitude_cap,
)
from gasgiant.params.seeds import subseed
from gasgiant.sim.bands import BandLayout
from gasgiant.sim.profiles import LatProfiles
from gasgiant.sim.resolution_scaling import scale_duration

# No discrete vortices poleward of this (polar patches own the caps; the
# nesting blend band must stay storm-free).
MAX_VORTEX_LAT = np.deg2rad(68.0)

# Hard cap on the total population: the psi/stamp kernels loop over every
# vortex per pixel per step, and the CPU repacks the SSBO per step.
MAX_VORTICES = 400

KIND_OVAL = 0.0
KIND_HERO = 1.0
KIND_BARGE = 2.0
KIND_PEARL = 3.0
KIND_KH = 4.0
KIND_POLAR = 5.0
# 6.0 is KIND_OUTBREAK (events.py).
KIND_DEBRIS = 7.0

# -- merger constants (frozen into checkpoint GENERATION_VERSION 3) --------------
# Capture when gap < COEF * merge_rate * (r1 + r2): at rate 1.0 an equal pair
# captures at separation 3r, matching the classic ~3.3a critical merger
# distance for equal Gaussian vortices.
MERGE_CAPTURE_COEF = 1.5
# Steps a merge product is ineligible for further merging (chain mergers stay
# a visible multi-step process instead of one-step cluster collapse).
MERGE_COOLDOWN = 25
# Product radius cap: excess area reads as filament shedding. Below the
# default hero radius (0.10); kind never becomes HERO regardless.
MERGE_MAX_R = 0.08
# Peak tangential speed cap for merge products: the solver's dt budgets
# _VORTEX_SPEED_MARGIN = 0.45 for vortices and never recomputes, so chain
# merges must not push v_peak past the CFL assumption.
MERGE_V_MAX = 0.40
# psi = S*exp(-(d/r)^2) peaks tangentially at d = r/sqrt(2):
# v_peak = sqrt(2)*exp(-1/2) * S / r.
_V_PEAK_COEF = float(np.sqrt(2.0) * np.exp(-0.5))
# Merger-debris collar: steps from merge to fully eroded, fade-in steps
# (avoids a relax-target pop), peak stamp brightness, and the exchange-band
# floor poleward of which no debris spawns (63-67 deg must stay storm-free).
MERGE_DEBRIS_LIFETIME = 250
MERGE_DEBRIS_RAMP = 15
MERGE_DEBRIS_BRIGHT = 0.9
_EXCHANGE_FLOOR = np.deg2rad(63.0)


@dataclass
class Vortex:
    lat: float
    lon: float
    r_core: float
    strength: float  # signed psi amplitude
    kind: float

    # T3 tint stamped at init: positive = warm/red, negative = blue-gray.
    tint: float = 0.0
    # T0 brightness stamped at init (ovals bright, barges dark).
    brightness: float = 0.0
    # Wake side, +1 east / -1 west. Heroes are always -1: the GRS turbulent
    # wake trails WNW (review F06) — the local jet is eastward across the
    # whole hero band, so a jet-sampled direction can never produce it.
    wake_dir: float = 0.0
    # Latitude offset of the wake wedge center (radians, signed toward the
    # equator for heroes; 0 = centered on the vortex).
    wake_lat_off: float = 0.0
    # Belt-bow gate (emergence pack): boundary strength within heroBandDeflect's
    # reach, in [0,1]. 0 = no band boundary to bow -> the deflection must not
    # paint a phantom displaced-band wrap (the sweep-invariant "red hook" /
    # symmetric funnel tells found by the per-latitude adversarial reviews).
    # Computed at generation from profiles.t0_stamp; always 0 when emergence off.
    bow_gain: float = 0.0
    # lon:lat elongation of the iso-contours (heroes only; 1.0 = round).
    aspect: float = 1.0
    # CPU-only (never packed into the SSBO): merger hysteresis countdown.
    cooldown: int = 0
    # CPU-only: remaining lifetime of transient entries (merger debris);
    # -1 = immortal (every ordinary vortex).
    ttl: int = -1
    # CPU-only provenance marker (never packed into the SSBO -- adding it to
    # pack_ssbo would change the byte stream): "seeded" for the RNG populations,
    # "cast" for art-directed cast-list storms. Cast storms are exempt from the
    # population cap and runtime mergers so a director's storm survives the run.
    origin: str = "seeded"
    # CPU-only: index into storms.cast for a cast HERO carrying per-storm
    # appearance/dynamics overrides (M2 CastLevers); -1 otherwise. Never packed
    # into the base SSBO -- consulted only by pack_cast_levers_ssbo, which
    # resolves each override-or-global lever value at pack time.
    cast_ref: int = -1


@dataclass
class VortexRegistry:
    vortices: list[Vortex] = field(default_factory=list)
    # Resolution-invariant step scale s = resolution / reference_resolution
    # (1.0 = feature off or authored-at-reference). Runtime merger/debris
    # lifetimes below are step counts; they are multiplied by this so a
    # transient occupies the same PHYSICAL fraction of the run at any
    # resolution. Set once by generate_vortices; recomputed on checkpoint
    # resume (which re-runs generation), so it is never serialized.
    step_scale: float = 1.0

    def heroes(self) -> list[Vortex]:
        return [v for v in self.vortices if v.kind == KIND_HERO]

    def pack_ssbo(self) -> np.ndarray:
        """(N, 12) float32, three vec4 per vortex:
        [x, y, z, r_core], [strength, kind, tint, brightness],
        [wake_dir, aspect, wake_lat_off, bow_gain].
        (bow_gain is 0 for every vortex unless hero_emergence is on — the
        default programs never read the .w slot, so the previously-constant
        zero stays byte-equivalent for them.) Vectorized: runs every sim step."""
        n = len(self.vortices)
        if n == 0:
            return np.zeros((1, 12), dtype=np.float32)
        fields = np.array(
            [
                (v.lat, v.lon, v.r_core, v.strength, v.kind,
                 v.tint, v.brightness, v.wake_dir, v.aspect, v.wake_lat_off,
                 v.bow_gain)
                for v in self.vortices
            ],
            dtype=np.float64,
        )
        lat, lon = fields[:, 0], fields[:, 1]
        cl = np.cos(lat)
        out = np.zeros((n, 12), dtype=np.float32)
        out[:, 0] = cl * np.cos(lon)
        out[:, 1] = np.sin(lat)
        out[:, 2] = cl * np.sin(lon)
        out[:, 3] = fields[:, 2]
        out[:, 4:8] = fields[:, 3:7]
        out[:, 8] = fields[:, 7]
        out[:, 9] = fields[:, 8]
        out[:, 10] = fields[:, 9]
        out[:, 11] = fields[:, 10]
        return out

    def pack_cast_levers_ssbo(self, storms: StormsParams) -> np.ndarray:
        """(N, 8) float32, two vec4 per vortex, packed in the SAME row order as
        ``pack_ssbo`` so row ``i`` indexes the same vortex in both buffers (the
        CAST_LEVERS variant reads it at binding 5):
        [rim_contrast, rim_tint, rim_warp, mottle],
        [tint_var, wake_detail, solid_core, 0(reserved)].

        Every row is a fully-RESOLVED value -- no inherit flag reaches the GPU. A
        cast HERO with a per-storm override packs the override; every other vortex
        (seeded hero, non-hero, companion) packs the live GLOBAL ``storms.*`` value.
        So once the variant is compiled in (because SOME cast hero overrides a
        lever), un-overridden heroes still render exactly as the global-uniform
        path would. Re-resolved every step because the vortex list order can shift
        (mergers/trim); the row<->vortex mapping survives only by iterating the
        same list in the same order as ``pack_ssbo``. Column ``j`` carries lever
        ``CAST_LEVER_SPECS[j]``; the flat 8-float row is the two vec4 the shader
        reads at ``2*i`` and ``2*i+1``.

        A cast_ref out of range resolves to the global -- safe only because
        storms.cast is RESTART tier: any add/remove/reorder rebuilds the whole
        Solver (and this registry), so a live cast_ref can never point past a
        shortened/reordered list within one run."""
        n = len(self.vortices)
        out = np.zeros((max(n, 1), 8), dtype=np.float32)
        # Every row defaults to the resolved GLOBAL values (cols 0..6; col 7 stays
        # 0); only cast-hero rows with an actual override differ, so patch just
        # those cells rather than re-assigning every cell every step.
        out[:n, :7] = [float(getattr(storms, g)) for _, g in CAST_LEVER_SPECS]
        for i, v in enumerate(self.vortices):
            if 0 <= v.cast_ref < len(storms.cast):
                entry = storms.cast[v.cast_ref]
                for j, (cast_attr, _) in enumerate(CAST_LEVER_SPECS):
                    override = getattr(entry, cast_attr)
                    if override is not None:
                        out[i, j] = float(override)
        return out

    def drift(self, profiles: LatProfiles, dt: float) -> None:
        """Advect vortex centers with the ambient zonal flow (polar vortices
        are pinned: the polar jets are weak and the clusters are long-lived).
        One vectorized interp for the whole population — this runs every step."""
        if not self.vortices:
            return
        lats = np.array([v.lat for v in self.vortices])
        dlon = zonal_rate(profiles, lats) * dt
        for v, d in zip(self.vortices, dlon, strict=True):
            if v.kind != KIND_POLAR:
                v.lon = float((v.lon + d + np.pi) % (2 * np.pi) - np.pi)


def drift_compensated_lon(
    profiles: LatProfiles,
    lat: float,
    target_deg: float,
    dt: float | None,
    n_steps: int,
) -> float:
    """Seed longitude that lands at ``target_deg`` (degrees) after ``n_steps``
    of zonal drift, wrapped to (-pi, pi]. Inverse-compensation of the closed-form
    drift VortexRegistry.drift applies every step: total longitude drift at a
    fixed latitude = zonal_rate(lat) * dt * n_steps, so the seed is the target
    minus that. With ``dt is None`` or ``n_steps <= 0`` there is no dev run to
    compensate for -- seed the target directly."""
    target = float(np.deg2rad(target_deg))
    if dt is not None and n_steps > 0:
        target -= float(zonal_rate(profiles, np.array([lat]))[0]) * dt * n_steps
    return float((target + np.pi) % (2.0 * np.pi) - np.pi)


def zonal_rate(profiles: LatProfiles, lats: np.ndarray) -> np.ndarray:
    """d(lon)/dt of the zonal drift at the given latitudes — THE drift formula,
    shared by drift(), the merger converging gate, and seeded-pair placement
    so they can never disagree. profiles.lat is DESCENDING and np.interp
    silently returns garbage on descending xp, hence the negated axis."""
    u = np.interp(-lats, -profiles.lat, profiles.u)
    return u / np.maximum(np.cos(lats), 0.2)


def _merge_pair(
    a: Vortex, b: Vortex, profiles: LatProfiles, cooldown: int = MERGE_COOLDOWN
) -> Vortex:
    """Coalesce two same-sign peers. Conserves PEAK TANGENTIAL VELOCITY
    (S*r), not the psi-impulse (S*r^2): impulse conservation makes the
    product spin ~29% slower than its parents and erode fastest — the
    showcase oval would be the planet's mushiest. With S*r conserved an
    equal pair keeps v_peak exactly; core vorticity falls x1/sqrt(2),
    which reads as filamentation loss. Peak psi <= sqrt(2)*max|S| is still
    below the pre-merge superposed peak, so the merge instant de-escalates."""
    w1 = abs(a.strength) * a.r_core * a.r_core
    w2 = abs(b.strength) * b.r_core * b.r_core
    wt = w1 + w2
    r_new = min(float(np.hypot(a.r_core, b.r_core)), MERGE_MAX_R)
    s_mag = (abs(a.strength) * a.r_core + abs(b.strength) * b.r_core) / r_new
    s_mag = min(s_mag, MERGE_V_MAX * r_new / _V_PEAK_COEF)
    sign = 1.0 if a.strength > 0.0 else -1.0
    lat = (w1 * a.lat + w2 * b.lat) / wt
    dlon = (b.lon - a.lon + np.pi) % (2.0 * np.pi) - np.pi  # shortest arc
    lon = (a.lon + (w2 / wt) * dlon + np.pi) % (2.0 * np.pi) - np.pi
    u_here = float(np.interp(-lat, -profiles.lat, profiles.u))
    return Vortex(
        float(lat), float(lon), r_new, sign * s_mag, KIND_OVAL,
        tint=(w1 * a.tint + w2 * b.tint) / wt,
        brightness=(w1 * a.brightness + w2 * b.brightness) / wt,
        # Jet-sampled and INERT: both wake consumers gate on KIND_HERO and
        # the product is KIND_OVAL. Kept for the packed lane only; if merge
        # products ever gain wakes, use the hero convention (wake_dir=-1,
        # equatorward wake_lat_off), not this jet sample (see F06).
        wake_dir=1.0 if u_here >= 0.0 else -1.0,
        cooldown=cooldown,
    )


def resolve_mergers(
    reg: VortexRegistry, profiles: LatProfiles, storms: StormsParams
) -> list[tuple[Vortex, Vortex, Vortex | None]]:
    """Coalesce converging same-sign ovals/pearls; heroes absorb ovals.
    RNG-free and a pure function of (registry, profiles, params), so a
    restored checkpoint steps identically to the live run. Returns the
    resolved (a, b, product) triples — product None for hero absorption,
    where a is the hero and b the shredded victim. (Seed stream name
    reserved for future asymmetric debris: "mergers".)

    Kind rules are a WHITELIST: OVAL+OVAL and PEARL+PEARL peer-merge
    (cross-kind pearl merges would let a drifting zone oval eat the
    deliberately even string); HERO absorbs OVAL, bit-unchanged (the GRS
    shreds small ovals into filaments rather than growing); every other
    kind — barges, KH, polar, outbreaks, debris — is inert.

    The CONVERGING GATE is the load-bearing protection: drift is purely
    zonal, so pairs at the same exact latitude (pearls; pre-sheared
    small-storm twins) have bit-identical drift rates, a closing rate of
    exactly 0.0, and must FAIL the strict > 0 test."""
    rate = storms.merge_rate
    if rate <= 0.0:
        return []
    _age_transients(reg, storms)
    # Resolution-invariant lifetimes: scale the per-step merger cooldown and the
    # debris collar's step budget so a transient spans the same physical fraction
    # of the run (s == 1 => the module constants unchanged).
    cooldown = scale_duration(MERGE_COOLDOWN, reg.step_scale)
    debris_life = scale_duration(MERGE_DEBRIS_LIFETIME, reg.step_scale)
    vs = reg.vortices
    for v in vs:  # hysteresis ages even on steps with no merges
        if v.cooldown > 0:
            v.cooldown -= 1
    n = len(vs)
    if n < 2:
        return []

    lat = np.array([v.lat for v in vs])
    lon = np.array([v.lon for v in vs])
    r = np.array([v.r_core for v in vs])
    s = np.array([v.strength for v in vs])
    kind = np.array([v.kind for v in vs])
    cool = np.array([v.cooldown for v in vs])

    live = np.abs(s) > 1e-6  # outbreaks/debris are zero-strength
    peer_kind = (kind == KIND_OVAL) | (kind == KIND_PEARL)
    if not (peer_kind & live).any():
        return []

    cl = np.cos(lat)
    p3 = np.stack([cl * np.cos(lon), np.sin(lat), cl * np.sin(lon)], axis=1)
    d = np.arccos(np.clip(p3 @ p3.T, -1.0, 1.0))

    peer = peer_kind[:, None] & peer_kind[None, :] & (kind[:, None] == kind[None, :])
    absorb = (kind[:, None] == KIND_HERO) & (kind[None, :] == KIND_OVAL)
    # Cast-list storms are art direction: exempt from runtime mergers on BOTH
    # axes so a director's storm neither merges away nor absorbs a neighbor. All
    # True (byte-identical) when there are no cast entries, so the seeded-only
    # merger behavior is unchanged.
    not_cast = np.array([v.origin != "cast" for v in vs])
    eligible = (
        (peer | absorb | absorb.T)
        & live[:, None] & live[None, :]
        & not_cast[:, None] & not_cast[None, :]
    )
    same_sign = (s[:, None] * s[None, :]) > 0.0
    no_cool = (cool[:, None] == 0) & (cool[None, :] == 0)
    capture = MERGE_CAPTURE_COEF * rate * (r[:, None] + r[None, :])

    rates = zonal_rate(profiles, lat)
    gap = (lon[None, :] - lon[:, None] + np.pi) % (2.0 * np.pi) - np.pi
    closing = -np.sign(gap) * (rates[None, :] - rates[:, None])  # symmetric

    mask = eligible & same_sign & no_cool & (d < capture) & (closing > 0.0)
    iu = np.triu_indices(n, k=1)
    hits = np.nonzero(mask[iu])[0]
    if hits.size == 0:
        return []
    ii, jj = iu[0][hits], iu[1][hits]
    order = np.lexsort((jj, ii, d[ii, jj]))  # greedy by (distance, i, j)

    consumed: set[int] = set()
    removed: set[int] = set()
    products: list[Vortex] = []
    resolved: list[tuple[Vortex, Vortex, Vortex | None]] = []
    for k in order:
        i, j = int(ii[k]), int(jj[k])
        if i in consumed or j in consumed:
            continue
        consumed.update((i, j))
        a, b = vs[i], vs[j]
        if a.kind == KIND_HERO or b.kind == KIND_HERO:
            hero, victim = (a, b) if a.kind == KIND_HERO else (b, a)
            removed.add(i if victim is vs[i] else j)
            resolved.append((hero, victim, None))
            debris = _spawn_debris(victim.lat, victim.lon, victim.r_core, storms,
                                   lifetime=debris_life)
        else:
            removed.update((i, j))
            product = _merge_pair(a, b, profiles, cooldown=cooldown)
            products.append(product)
            resolved.append((a, b, product))
            debris = _spawn_debris(product.lat, product.lon, product.r_core, storms,
                                   lifetime=debris_life)
        if debris is not None:
            products.append(debris)
    # One identity-based rebuild — never list.remove (dataclass == is field
    # equality and could drop the wrong entry). Products appended at the end
    # are same-step-ineligible by construction.
    reg.vortices = [v for idx, v in enumerate(vs) if idx not in removed] + products
    return resolved


def _spawn_debris(
    lat: float, lon: float, r_core: float, storms: StormsParams,
    lifetime: int = MERGE_DEBRIS_LIFETIME,
) -> Vortex | None:
    """The transient turbulent collar a fresh merger leaves behind: a
    zero-strength registry entry (no psi) whose bright ring stamp the ambient
    flow folds into filaments while it decays. The Outbreak mechanism, reused.
    Skipped near the 63-67 deg nesting exchange band, which must stay
    storm-free (the ring stamp reaches ~3*r_core)."""
    if storms.merge_debris <= 0.0:
        return None
    if abs(lat) + 3.0 * r_core > _EXCHANGE_FLOOR:
        return None
    return Vortex(lat, lon, r_core, 0.0, KIND_DEBRIS,
                  tint=0.0, brightness=0.0,  # ramps in via _age_transients
                  ttl=lifetime)


def _age_transients(reg: VortexRegistry, storms: StormsParams) -> None:
    """Decrement debris ttl; brightness = base * fade-in * fade-out, removed
    at expiry. Deterministic — pure function of ttl + params. Debris spawned
    with an s-scaled ttl, so age/fade use the same s-scaled lifetime/ramp
    (s == 1 => the module constants unchanged)."""
    base = MERGE_DEBRIS_BRIGHT * storms.merge_debris * storms.stamp_contrast
    lifetime = scale_duration(MERGE_DEBRIS_LIFETIME, reg.step_scale)
    ramp = scale_duration(MERGE_DEBRIS_RAMP, reg.step_scale)
    expired = False
    for v in reg.vortices:
        if v.ttl < 0:
            continue
        v.ttl -= 1
        if v.ttl <= 0:
            expired = True
            continue
        age = lifetime - v.ttl
        ramp_in = min(age / ramp, 1.0)
        v.brightness = base * ramp_in * (v.ttl / lifetime)
    if expired:
        reg.vortices = [v for v in reg.vortices if v.ttl != 0]


def _hero_bow_gain(profiles: LatProfiles, lat: float, r: float) -> float:
    """Boundary strength within the belt-bow's reach, in [0,1].

    heroBandDeflect paints its bow by displacing the band-target sampling —
    with NO band boundary inside its q<2.3 window that displacement
    manufactures a phantom wrap out of whatever latitude gradient exists
    (the -28-deg symmetric funnel tell). Gate on the largest t0_stamp step
    within +-1.6 r of the hero: a real belt/zone edge (step ~0.2) gives full
    gain, a flat zone gives ~0. Deterministic, profile-derived.
    """
    lo, hi = lat - 1.6 * r, lat + 1.6 * r
    win = (profiles.lat >= lo) & (profiles.lat <= hi)
    if not win.any():
        return 0.0
    t0 = profiles.t0_stamp[win]
    step = float(t0.max() - t0.min())
    # 0 below 0.04 (banding noise), full gain by 0.14 (a real edge).
    x = np.clip((step - 0.04) / 0.10, 0.0, 1.0)
    return float(x * x * (3.0 - 2.0 * x))


def _hero_wake_frame(profiles: LatProfiles, lat: float, r: float) -> tuple[float, float]:
    """(lane latitude offset, wake_dir) for the hero's DYNAMIC wake (emergence).

    The emergence wake is real fluid machinery (wedge eddy injection in
    omega_force + relaxation release in heroRelaxWeight), so the wedge must
    sit where the flow actually carries material: the lane goes to the
    strongest jet within [0.4 r, 2.5 r] equatorward of the hero, and the wake
    trails DOWNSTREAM of that jet (wake_dir = sign(u): +1 east, -1 west). The
    legacy authored frame (0.5 r equatorward, hardwired westward — review
    F06) is decorative: on gas_giant_warm the whole hero band flows EAST, so
    a west-authored wedge injects into flow that immediately drains the folds
    out the back. Falls back to the legacy frame when the search band has no
    real flow (|u| < 0.05). Deterministic (profile-derived, no RNG);
    latitude-ordering independent.
    """
    eq = 1.0 if lat < 0.0 else -1.0
    legacy = (0.5 * r * eq, -1.0)
    lo, hi = sorted((lat + eq * 0.4 * r, lat + eq * 2.5 * r))
    win = (profiles.lat >= lo) & (profiles.lat <= hi)
    if not win.any():
        return legacy
    u = profiles.u[win]
    i = int(np.argmax(np.abs(u)))
    if abs(u[i]) < 0.05:
        return legacy
    return float(profiles.lat[win][i] - lat), (1.0 if u[i] > 0.0 else -1.0)


def _ambient_sign(profiles: LatProfiles, lat: float) -> float:
    """Sign of the ambient relative vorticity (~ -du/dphi) at a latitude."""
    lats = profiles.lat
    du = np.gradient(profiles.u, lats)
    s = -float(np.interp(-lat, -lats, du))
    return 1.0 if s >= 0.0 else -1.0


def _band_centers(bands: BandLayout, want_belt: bool) -> list[tuple[float, float]]:
    """(center latitude, width) of zones or belts within the vortex latitude range.
    Identity from BandLayout.is_belt (frozen at layout build) — never from
    values, which a belt fade may later edit."""
    out = []
    for j in range(len(bands.values)):
        center = 0.5 * (bands.edges[j] + bands.edges[j + 1])
        width = float(bands.edges[j] - bands.edges[j + 1])
        if abs(center) > MAX_VORTEX_LAT:
            continue
        if bool(bands.is_belt[j]) == want_belt:
            out.append((float(center), width))
    return out


def _poisson_lons(rng: np.random.Generator, count: int, min_sep: float) -> list[float]:
    """Longitude placement with a minimum separation (rejection sampling)."""
    lons: list[float] = []
    for _ in range(count * 8):
        if len(lons) >= count:
            break
        cand = float(rng.uniform(-np.pi, np.pi))
        if all(abs((cand - x + np.pi) % (2 * np.pi) - np.pi) > min_sep for x in lons):
            lons.append(cand)
    return lons


def add_polar_vortices(
    reg: VortexRegistry,
    rng: np.random.Generator,
    pole_sign: float,
    style: str,
    cyclone_count: int,
    strength: float,
    field_density: float = 0.0,
) -> None:
    """Polar features that are vortex-shaped (cyclone clusters, plain vortex).
    The polygon jet is a streamfunction term in the patch kernel, not a vortex,
    but it still gets a tight central vortex."""
    if style == "calm" or strength <= 0.0:
        return
    pole_lat = pole_sign * (np.pi / 2.0)
    # Cyclonic sense, mirrored across hemispheres (sign f = sign(pole_sign)).
    # Under the psi-amplitude trap (omega = -sign(strength), see the module
    # docstring) that means the `strength` argument itself must carry the
    # OPPOSITE sign of pole_sign.
    s_central = -pole_sign * 0.032 * strength

    if style == "plain_vortex":
        reg.vortices.append(
            Vortex(pole_lat, 0.0, 0.09, s_central * 1.4, KIND_POLAR, tint=0.25, brightness=-0.22)
        )
    elif style == "polygon_jet":
        reg.vortices.append(
            Vortex(pole_lat, 0.0, 0.05, s_central, KIND_POLAR, tint=0.15, brightness=-0.14)
        )
    else:
        # cyclone_cluster: central cyclone + ring at polygon vertices (Juno's
        # 8-around-1 north / 5-around-1 south configuration generalized).
        reg.vortices.append(
            Vortex(pole_lat, 0.0, 0.055, s_central, KIND_POLAR, tint=0.3, brightness=-0.26)
        )
        ring_colat = 0.135
        base = float(rng.uniform(0.0, 2.0 * np.pi))
        for i in range(cyclone_count):
            theta = base + 2.0 * np.pi * i / cyclone_count
            lat = pole_sign * (np.pi / 2.0 - ring_colat)
            lon = (theta + np.pi) % (2.0 * np.pi) - np.pi
            reg.vortices.append(
                Vortex(lat, lon, 0.05, s_central * 0.85, KIND_POLAR, tint=0.25, brightness=-0.22)
            )

    # Background field: small cyclones of mixed size filling the cap (the
    # dense hierarchy PIA21641 shows). Strictly poleward of 70 deg — the
    # 63-67 deg nesting exchange band must stay storm-free.
    if field_density > 0.0:
        count = int(rng.poisson(field_density * 14.0))
        for _ in range(count):
            colat = float(rng.uniform(0.06, np.pi / 2.0 - np.deg2rad(70.0)))
            lat = pole_sign * (np.pi / 2.0 - colat)
            lon = float(rng.uniform(-np.pi, np.pi))
            u01 = float(rng.uniform(0.0, 1.0))
            r = 0.012 + (0.038 - 0.012) * u01 * u01
            reg.vortices.append(
                Vortex(lat, lon, r, s_central * (0.25 + 0.5 * u01), KIND_POLAR,
                       tint=0.18, brightness=-(0.10 + 3.5 * r))
            )


def generate_vortices(
    seed: int,
    bands: BandLayout,
    profiles: LatProfiles,
    storms: StormsParams,
    poles: PolesParams | None = None,
    dt: float | None = None,
    dev_steps: int = 0,
    step_scale: float = 1.0,
) -> VortexRegistry:
    rng = subseed(seed, "storms")
    reg = VortexRegistry(step_scale=step_scale)
    # Effective run length the solver will actually step (dev_steps * s). dt
    # already tracks 1/resolution, so drift compensation and merger scheduling
    # keyed to `eff` land storms at the same physical longitude/time at any
    # resolution. Intent guards below stay on raw dev_steps (the run length the
    # author asked for); s == 1 makes eff == dev_steps (byte-identical).
    eff = scale_duration(dev_steps, step_scale)

    zones = _band_centers(bands, want_belt=False)
    belts = _band_centers(bands, want_belt=True)

    # Hero anticyclones (GRS class): tropical-to-temperate zone latitudes.
    tropical = [z for z in zones if 0.15 < abs(z[0]) < 0.75] or zones
    for _ in range(storms.hero_count):
        if not tropical:
            break
        center, _w = tropical[rng.integers(0, len(tropical))]
        lat = float(np.clip(center + rng.normal(0.0, 0.02), -MAX_VORTEX_LAT, MAX_VORTEX_LAT))
        if storms.hero_latitude is not None:
            lat = float(np.deg2rad(storms.hero_latitude))
        r = storms.hero_radius * (1.0 + 0.2 * rng.uniform(-1.0, 1.0))
        s = -_ambient_sign(profiles, lat) * storms.hero_strength * 0.045
        # Legacy wake frame (review F06): authored WNW — westward, biased half
        # a core radius toward the equator. This static frame is the
        # emergence-OFF fallback; warm now bakes a westward SEBs-analog jet
        # across the hero band (jets.local_jet -0.9 @ -20°), so the real GRS's
        # WNW wake IS reproducible here — see build_warm_preset.py.
        # Under emergence the wake is DYNAMIC (wedge eddy injection +
        # relaxation release), so the frame must follow the actual flow —
        # _hero_wake_frame puts the lane in the strongest nearby jet and
        # trails downstream of it. Emergence-gated so legacy presets keep
        # byte-identical registries.
        # Consume the seeded longitude draw unconditionally (RNG-stream
        # position must not depend on the pin), then override for a pin.
        lon = float(rng.uniform(-np.pi, np.pi))
        if storms.hero_longitude is not None:
            lon = drift_compensated_lon(
                profiles, lat, storms.hero_longitude, dt, eff
            )
        woff = 0.5 * r * (1.0 if lat < 0.0 else -1.0)
        wdir = -1.0
        bow = 0.0
        if storms.hero_emergence > 0.0:
            woff, wdir = _hero_wake_frame(profiles, lat, r)
            bow = _hero_bow_gain(profiles, lat, r)
        # User override (storms.hero_wake_dir): auto = the frame above;
        # east/west force the trailing direction (a forced direction against
        # the local jet reads weaker — the flow drains the folds). The lane
        # offset keeps tracking the jet: that is where the MATERIAL is.
        if storms.hero_wake_dir == WakeDir.EAST:
            wdir = 1.0
        elif storms.hero_wake_dir == WakeDir.WEST:
            wdir = -1.0
        reg.vortices.append(
            Vortex(lat, lon, r, s, KIND_HERO,
                   tint=storms.hero_tint, brightness=storms.hero_brightness,
                   wake_dir=wdir,
                   wake_lat_off=woff,
                   bow_gain=bow,
                   aspect=storms.hero_aspect)
        )

    # White ovals: anticyclones in zones, power-law sizes, Poisson-disc lons.
    for center, width in zones:
        lam = storms.oval_density * 2.2
        count = int(rng.poisson(lam))
        if count == 0:
            continue
        lons = _poisson_lons(rng, count, min_sep=0.35)
        for lon in lons:
            u01 = float(rng.uniform(0.0, 1.0))
            r = 0.018 + (0.055 - 0.018) * u01 * u01  # power-law-ish: many small, few big
            lat = float(
                np.clip(center + rng.normal(0.0, 0.15 * width), -MAX_VORTEX_LAT, MAX_VORTEX_LAT)
            )
            s = -_ambient_sign(profiles, lat) * 0.012 * (r / 0.03)
            reg.vortices.append(
                Vortex(lat, lon, r, s, KIND_OVAL, tint=0.1, brightness=0.22)
            )

    # Brown barges: weak cyclones inside belts, co-rotating with the local
    # (cyclonic) belt shear the same as every other seeded class -- what
    # shears them into dark cigars is the jets' shear GRADIENT across the
    # barge's footprint, not an opposed rotation sense.
    for center, width in belts:
        count = int(rng.poisson(storms.barge_density * 1.2))
        if count == 0:
            continue
        lons = _poisson_lons(rng, count, min_sep=0.5)
        for lon in lons:
            lat = float(
                np.clip(center + rng.normal(0.0, 0.1 * width), -MAX_VORTEX_LAT, MAX_VORTEX_LAT)
            )
            r = float(rng.uniform(0.02, 0.045))
            s = -_ambient_sign(profiles, lat) * 0.006
            reg.vortices.append(
                Vortex(lat, lon, r, s, KIND_BARGE, tint=0.35, brightness=-0.28)
            )

    # String of pearls: evenly spaced same-sign ovals on one seeded zone latitude.
    if storms.pearls_count > 0 and zones:
        temperate = [z for z in zones if 0.4 < abs(z[0]) < 1.0] or zones
        center, _w = temperate[rng.integers(0, len(temperate))]
        lat = float(np.clip(center, -MAX_VORTEX_LAT, MAX_VORTEX_LAT))
        base = float(rng.uniform(-np.pi, np.pi))
        n = storms.pearls_count
        for i in range(n):
            lon = base + (2 * np.pi * i) / n + float(rng.normal(0.0, 0.04))
            lon = (lon + np.pi) % (2 * np.pi) - np.pi
            s = -_ambient_sign(profiles, lat) * 0.008
            reg.vortices.append(
                Vortex(lat, lon, 0.02, s, KIND_PEARL, tint=0.05, brightness=0.25)
            )

    # Small-storm field: sub-oval white spots and dark spots in loose latitude
    # rows (real maps are peppered with them). Own seed stream; defaults off.
    if storms.small_density > 0.0:
        _add_small_storms(reg, subseed(seed, "small-storms"), zones, belts,
                          profiles, storms.small_density)

    # Stamp contrast (1 = v1): scales how hard non-hero storms register in the
    # tracers; the velocity strength is untouched. stamp_tint_contrast (B5-7)
    # splits the tint amplitude from the brightness amplitude; None = follow
    # stamp_contrast, which reproduces the legacy coupled arithmetic exactly.
    tint_contrast = (storms.stamp_tint_contrast
                     if storms.stamp_tint_contrast is not None
                     else storms.stamp_contrast)
    if storms.stamp_contrast != 1.0 or tint_contrast != 1.0:
        for v in reg.vortices:
            if v.kind != KIND_HERO:
                v.brightness *= storms.stamp_contrast
                v.tint *= tint_contrast

    if poles is not None:
        polar_rng = subseed(seed, "poles")
        add_polar_vortices(
            reg, polar_rng, +1.0, poles.north.style.value,
            poles.north.cyclone_count, poles.north.strength,
            poles.north.field_density,
        )
        add_polar_vortices(
            reg, polar_rng, -1.0, poles.south.style.value,
            poles.south.cyclone_count, poles.south.strength,
            poles.south.field_density,
        )

    _enforce_cap(reg)

    # Convergent companion pairs: natural differential zonal drift closes only
    # ~0.03-0.07 rad over a dev run — far less than the Poisson-disc spacing —
    # so without seeding, mergers essentially never occur. Seeded AFTER the cap
    # so a host can never be cap-dropped out from under its companion.
    if storms.merge_rate > 0.0 and dt is not None and dev_steps > 0:
        _seed_convergent_pairs(
            reg, subseed(seed, "mergers"), zones, profiles,
            storms.merge_rate, dt, eff, step_scale,
        )

    # Accent ovals (A01) and hero companions (B5-5): explicitly colored stamps
    # on their own named substreams, seeded LAST (after the cap and the merger
    # pairs) so every pre-existing population is byte-identical when they are
    # off — and untouched even when they are on.
    if storms.accent_count > 0:
        _add_accent_ovals(reg, subseed(seed, "accent-ovals"), zones, profiles, storms,
                          dt, eff)
    if storms.hero_companions > 0:
        _add_hero_companions(reg, subseed(seed, "hero-companions"), profiles,
                             storms.hero_companions, storms.companion_aspect,
                             storms.companion_brightness)
    # Cast list (art-directed storms): stamped LAST, after the cap and every
    # seeded population, so they are byte-identical no-ops when the list is
    # empty and never perturb a seeded draw when present.
    if storms.cast:
        _add_cast(reg, profiles, storms, dt, eff)
    # Kind-aware atomic trim: cast-list storms (origin=="cast") are art
    # direction and are NEVER trimmed; drop the NEWEST non-cast entry first
    # (accents/companions). With zero cast entries every entry is non-cast, so
    # this pops from the end exactly like the old `reg.vortices.pop()` loop --
    # byte-identical. If the cast alone exceeds the cap there is nothing legal
    # to drop, so raise rather than silently mangling the population.
    if len(reg.vortices) > MAX_VORTICES:
        n_cast = sum(1 for v in reg.vortices if v.origin == "cast")
        if n_cast > MAX_VORTICES:
            raise ValueError(
                f"the cast list contributes {n_cast} storms but the vortex cap "
                f"is {MAX_VORTICES}; reduce storms.cast."
            )
        while len(reg.vortices) > MAX_VORTICES:
            for i in range(len(reg.vortices) - 1, -1, -1):
                if reg.vortices[i].origin != "cast":
                    reg.vortices.pop(i)
                    break
    return reg


def _add_accent_ovals(
    reg: VortexRegistry,
    rng: np.random.Generator,
    zones: list[tuple[float, float]],
    profiles: LatProfiles,
    storms: StormsParams,
    dt: float | None = None,
    dev_steps: int = 0,
) -> None:
    """Accent ovals (review A01 — the Oval BA unlock): KIND_OVAL storms whose
    tint/brightness are EXPLICIT params rather than the kind constants, so a
    second reddened oval can sit beside the white population. All accents share
    one latitude and one appearance; count=2 puts the pair at offset longitudes.
    Velocity treatment matches ordinary ovals (ambient-sign anticyclone, same
    strength-radius law), so oval_solid_core >= the 0.035 radius gate keeps them
    coherent in vorticity mode. stamp_contrast does NOT apply (color is verbatim
    by design; this runs after the contrast pass)."""
    # Same radius-coupled cap as the explicit-value validator: auto-placement
    # must never yield a latitude that accent_latitude=<same value> would reject.
    cap = min(MAX_VORTEX_LAT, float(np.deg2rad(hero_latitude_cap(storms.accent_radius))))
    if storms.accent_latitude is not None:
        lat = float(np.deg2rad(storms.accent_latitude))
    else:
        # Auto-place: seeded zone pick, tropical-to-temperate preference
        # (the hero's placement rule, one draw from the accent stream).
        cands = [z for z in zones if 0.15 < abs(z[0]) < min(1.0, cap)] or zones
        if not cands:
            return
        center, _w = cands[int(rng.integers(0, len(cands)))]
        lat = float(np.clip(center, -cap, cap))
    r = storms.accent_radius
    s = -_ambient_sign(profiles, lat) * 0.012 * (r / 0.03)
    # Consume the seeded Poisson-disc draw unconditionally (RNG-stream position
    # must not depend on the pin). When pinned, override every accent longitude
    # to the drift-compensated target, offset a fixed min_sep step per accent so
    # a count=2 pair stays separated at the snapshot.
    min_sep = 0.6
    lons = _poisson_lons(rng, storms.accent_count, min_sep=min_sep)
    pin_base = (
        drift_compensated_lon(profiles, lat, storms.accent_longitude, dt, dev_steps)
        if storms.accent_longitude is not None
        else None
    )
    # Hero-relative default (round B): a pinned-LATITUDE accent is an authored
    # neighbor of the hero (the Oval-BA-passing-south recipe), and an unpinned
    # longitude puts it anywhere on the circle — out of any hero-framed view
    # ~90% of the time. When the latitude is pinned, no explicit longitude is
    # given, and a hero exists, root the accent a seeded 0.3-0.55 rad
    # DOWNSTREAM of the hero instead of the Poisson draw. The draw is
    # unconditional and APPENDED after the existing ones (stream position and
    # the seeded-zone path — e.g. neptune's Scooter, accent_latitude None —
    # stay byte-identical).
    rel_off = float(rng.uniform(0.3, 0.55))
    heroes = reg.heroes()
    if storms.accent_latitude is not None and pin_base is None and heroes:
        pin_base = float(
            (heroes[0].lon + heroes[0].wake_dir * rel_off + np.pi) % (2.0 * np.pi)
            - np.pi
        )
    for k, lon in enumerate(lons):
        if pin_base is not None:
            lon = float((pin_base + k * min_sep + np.pi) % (2.0 * np.pi) - np.pi)
        reg.vortices.append(
            Vortex(lat, lon, r, s, KIND_OVAL,
                   tint=storms.accent_tint, brightness=storms.accent_brightness,
                   aspect=storms.accent_aspect)
        )


def _add_hero_companions(
    reg: VortexRegistry,
    rng: np.random.Generator,
    profiles: LatProfiles,
    count: int,
    aspect: float = 1.0,
    brightness: float = 0.32,
) -> None:
    """Bright companion clouds beside each hero (review B5-5: the Neptune GDS
    companion / Scooter class). KIND_PEARL stamps — bright spot with a slight
    collar, never cap-dropped — placed a few core radii from the hero on its
    wake-free flank (opposite wake_dir, so the wake turbulence does not shred
    them immediately), alternating equatorward/poleward. Positions are
    generation-time: differential zonal drift moves them relative to the hero
    over the dev run, which is physical (Neptune's companions do the same)."""
    for hero in reg.heroes():
        side = -hero.wake_dir if hero.wake_dir != 0.0 else 1.0
        eq = 1.0 if hero.lat < 0.0 else -1.0  # unit step toward the equator
        for i in range(count):
            dist = (1.7 + 0.8 * i) * hero.r_core
            dlat = (eq * (0.6 + 0.5 * float(rng.uniform())) * hero.r_core
                    * (1.0 if i % 2 == 0 else -0.8))
            lat = float(np.clip(hero.lat + dlat, -MAX_VORTEX_LAT, MAX_VORTEX_LAT))
            dlon = (side * dist / max(np.cos(lat), 0.2)
                    + float(rng.normal(0.0, 0.2 * hero.r_core)))
            lon = float((hero.lon + dlon + np.pi) % (2.0 * np.pi) - np.pi)
            r = float(np.clip(0.30 * hero.r_core, 0.015, 0.035))
            s = -_ambient_sign(profiles, lat) * 0.008
            reg.vortices.append(
                Vortex(lat, lon, r, s, KIND_PEARL, tint=0.0, brightness=brightness,
                       aspect=aspect)
            )


def _add_cast_companions(
    reg: VortexRegistry,
    profiles: LatProfiles,
    hero_lat: float,
    hero_lon: float,
    r_core: float,
    wake_dir: float,
    count: int,
    aspect: float,
    brightness: float,
) -> None:
    """Companion pearls beside one CAST hero (its per-storm ``companions``).
    Same geometry as the seeded ``_add_hero_companions`` but DETERMINISTIC -- no
    RNG draws -- because the cast path must never perturb a seeded stream (the
    jitter terms of the seeded version are dropped to fixed offsets). origin
    stays 'cast' so the pearls survive the population trim with their hero."""
    side = -wake_dir if wake_dir != 0.0 else 1.0
    eq = 1.0 if hero_lat < 0.0 else -1.0  # unit step toward the equator
    for i in range(count):
        dist = (1.7 + 0.8 * i) * r_core
        dlat = eq * 0.85 * r_core * (1.0 if i % 2 == 0 else -0.8)  # jitter -> fixed
        lat = float(np.clip(hero_lat + dlat, -MAX_VORTEX_LAT, MAX_VORTEX_LAT))
        dlon = side * dist / max(np.cos(lat), 0.2)
        lon = float((hero_lon + dlon + np.pi) % (2.0 * np.pi) - np.pi)
        r = float(np.clip(0.30 * r_core, 0.015, 0.035))
        s = -_ambient_sign(profiles, lat) * 0.008
        reg.vortices.append(
            Vortex(lat, lon, r, s, KIND_PEARL, tint=0.0, brightness=brightness,
                   aspect=aspect, origin="cast")
        )


def _add_cast(
    reg: VortexRegistry,
    profiles: LatProfiles,
    storms: StormsParams,
    dt: float | None,
    dev_steps: int,
) -> None:
    """Stamp the art-directed cast-list storms. DETERMINISTIC -- no RNG draws --
    so toggling, reordering, or editing the cast never perturbs any seeded
    population (an empty cast is a strict no-op). Each entry names its RENDERED
    longitude, drift-compensated over the dev run like the T1 pins. Cast storms
    carry origin="cast", which exempts them from the population cap (the
    kind-aware trim) and from runtime mergers (resolve_mergers). Appearance
    falls back to the per-kind seeded defaults (copied from the seeded blocks in
    generate_vortices) when tint/brightness are None."""
    for cast_idx, entry in enumerate(storms.cast):
        lat = float(np.deg2rad(entry.lat_deg))
        lon = drift_compensated_lon(profiles, lat, entry.lon_deg, dt, dev_steps)
        sign = -_ambient_sign(profiles, lat)
        kind = entry.kind
        # (KIND_* const, base strength law, default tint, default brightness).
        # Every base here is a positive magnitude; `sign` alone carries the
        # co-rotation flip, matching the seeded populations above (e.g. the
        # brown-barge convention s = -ambient_sign * 0.006 -- base=+0.006 here
        # since `sign` already supplies the leading minus).
        if kind == CastKind.HERO:
            k, base, d_tint, d_bright = (
                KIND_HERO, 0.045 * storms.hero_strength,
                storms.hero_tint, storms.hero_brightness,
            )
        elif kind == CastKind.OVAL:
            k, base, d_tint, d_bright = KIND_OVAL, 0.012 * (entry.radius / 0.03), 0.1, 0.22
        elif kind == CastKind.BARGE:
            k, base, d_tint, d_bright = KIND_BARGE, 0.006, 0.35, -0.28
        else:  # CastKind.PEARL
            k, base, d_tint, d_bright = KIND_PEARL, 0.008, 0.05, 0.25
        s = sign * base * entry.strength_scale
        tint = entry.tint if entry.tint is not None else d_tint
        brightness = entry.brightness if entry.brightness is not None else d_bright
        # Cast heroes get the GRS wake convention (WNW, biased equatorward); the
        # other kinds carry no wake (wake_dir 0). Every kind honors entry.aspect.
        wake_dir = -1.0 if kind == CastKind.HERO else 0.0
        wake_lat_off = (
            0.5 * entry.radius * (1.0 if lat < 0.0 else -1.0)
            if kind == CastKind.HERO else 0.0
        )
        bow = 0.0
        if kind == CastKind.HERO and storms.hero_emergence > 0.0:
            wake_lat_off, wake_dir = _hero_wake_frame(profiles, lat, entry.radius)
            bow = _hero_bow_gain(profiles, lat, entry.radius)
        if kind == CastKind.HERO:
            # User override applies UNCONDITIONALLY, exactly like the seeded
            # hero path above (PR-43 simplify-pass finding: the override was
            # emergence-gated only here, so a cast hero at emergence 0
            # silently ignored a forced direction a seeded hero honors).
            # Per-storm entry.wake_dir wins; None inherits the global
            # storms.hero_wake_dir (byte-identical to the pre-per-storm path).
            eff_wake = (entry.wake_dir if entry.wake_dir is not None
                        else storms.hero_wake_dir)
            if eff_wake == WakeDir.EAST:
                wake_dir = 1.0
            elif eff_wake == WakeDir.WEST:
                wake_dir = -1.0
        # Only a cast HERO carries per-storm CastLevers overrides; other kinds
        # (and heroes with no overrides) resolve to the global at pack time.
        cast_ref = cast_idx if kind == CastKind.HERO else -1
        reg.vortices.append(
            Vortex(lat, lon, entry.radius, s, k,
                   tint=tint, brightness=brightness,
                   wake_dir=wake_dir, wake_lat_off=wake_lat_off,
                   bow_gain=bow,
                   aspect=entry.aspect, origin="cast", cast_ref=cast_ref)
        )
        # Per-cast companion pearls (opt-in, default companions=0 -> no-op, so
        # existing presets are byte-identical). Deterministic placement (no RNG:
        # the cast path must never perturb a seeded draw), mirroring the seeded
        # _add_hero_companions geometry with the jitter terms dropped.
        if kind == CastKind.HERO and entry.companions > 0:
            c_aspect = (entry.companion_aspect if entry.companion_aspect is not None
                        else storms.companion_aspect)
            c_bright = (entry.companion_brightness if entry.companion_brightness
                        is not None else storms.companion_brightness)
            _add_cast_companions(reg, profiles, lat, lon, entry.radius, wake_dir,
                                 entry.companions, c_aspect, c_bright)


def _seed_convergent_pairs(
    reg: VortexRegistry,
    rng: np.random.Generator,
    zones: list[tuple[float, float]],
    profiles: LatProfiles,
    merge_rate: float,
    dt: float,
    dev_steps: int,
    step_scale: float = 1.0,
) -> None:
    """Spawn companion ovals placed KINEMATICALLY: measure the actual closure
    rate at the site via zonal_rate, draw a target merge step, and set the
    longitude gap so capture happens near that step. A fixed gap range cannot
    work — du/dphi at zone centers is ~1-2, an order short of closing a
    Poisson-disc gap in 500 steps. Targets alternate early/late so finished
    maps show both matured products and a still-live debris collar.

    No cast handling here: this runs BEFORE _add_cast in generate_vortices, so
    the registry holds no origin=="cast" entries yet -- convergent pairs never
    pick a cast storm as a host, and the cast exemption lives only in the
    runtime resolve_mergers path."""
    pair_index = 0
    for center, width in zones:
        if rng.uniform() >= 0.5 * merge_rate:
            continue
        hosts = [
            v for v in reg.vortices
            if v.kind == KIND_OVAL and abs(v.lat - center) < 0.5 * width
        ]
        if not hosts:
            continue
        host = hosts[int(rng.integers(0, len(hosts)))]
        u01 = float(rng.uniform(0.0, 1.0))
        r_c = 0.018 + (0.045 - 0.018) * u01 * u01
        # Dead-zone clamp: below merge_rate = 1/3 a half-(r1+r2) lat offset
        # exceeds the capture radius and the pair could never merge.
        capture = MERGE_CAPTURE_COEF * merge_rate * (host.r_core + r_c)
        dlat = min(0.5 * (host.r_core + r_c), 0.75 * capture)
        # Pick the lat side with the faster differential drift.
        cands = np.clip(
            np.array([host.lat + dlat, host.lat - dlat]),
            -MAX_VORTEX_LAT, MAX_VORTEX_LAT,
        )
        rates = zonal_rate(profiles, np.append(cands, host.lat))
        drates = rates[:2] - rates[2]
        pick = int(np.argmax(np.abs(drates)))
        comp_lat = float(cands[pick])
        drate = float(drates[pick])
        # Longitude gap at which capture occurs, given the fixed lat offset
        # (small-angle: d^2 ~ dlat^2 + (cos(lat)*dlon)^2).
        dlat_actual = comp_lat - host.lat
        dlon_capture = float(
            np.sqrt(max(capture**2 - dlat_actual**2, 0.0))
            / max(np.cos(host.lat), 0.2)
        )
        closure = abs(drate) * dt  # rad of gap closed per step
        # Alternate late targets (~120 steps before the end: the 250-step
        # debris collar is still bright in the final still) with early ones
        # (matured product drifts on). LATE FIRST: many runs seed only one
        # pair, and the live collar is the showcase.
        # Schedule offsets are step-count DURATIONS (a late merge fires ~220..80
        # steps before the end; an early one within the first ~280) -- scale them
        # by s so the capture lands at the same physical time. dev_steps here is
        # already the effective (scaled) run length. s == 1 => unchanged.
        off_hi = scale_duration(220, step_scale)
        off_lo = scale_duration(80, step_scale)
        early_hi = scale_duration(280, step_scale)
        if pair_index % 2 == 0:
            lo, hi = max(dev_steps - off_hi, off_lo), max(dev_steps - off_lo, off_lo + 1)
        else:
            lo, hi = off_lo, max(min(early_hi, dev_steps), off_lo + 1)
        target_step = int(rng.integers(lo, hi + 1))
        if closure * dev_steps < 0.02:
            # Flat shear: place just outside capture so any drift nudges it in.
            gap = 1.05 * dlon_capture
        else:
            # Spawn OUTSIDE capture by the distance closed in target_step
            # steps (a sub-1.0 coefficient here would mean already-captured
            # pairs merging at step 0).
            gap = 1.02 * dlon_capture + closure * target_step
        pair_index += 1
        # Converging side: the gap g = wrap(comp_lon - host_lon) evolves at
        # drate, so |g| shrinks iff sign(g) opposes it.
        signed_gap = -np.sign(drate) * gap if drate != 0.0 else gap
        comp_lon = float((host.lon + signed_gap + np.pi) % (2.0 * np.pi) - np.pi)
        sign = 1.0 if host.strength > 0.0 else -1.0
        # Match the host's stamp treatment (stamp_contrast already applied to
        # the base population by the time companions spawn).
        reg.vortices.append(
            Vortex(comp_lat, comp_lon, r_c, sign * 0.012 * (r_c / 0.03),
                   KIND_OVAL, tint=host.tint, brightness=host.brightness)
        )
    # Atomic trim: if companions pushed past the cap, drop companions (the
    # newest entries) — never a host — so no orphan halves distort anything.
    while len(reg.vortices) > MAX_VORTICES:
        reg.vortices.pop()


def _veil(lat: float) -> float:
    """Storm visibility falls poleward under haze (the anti-polka-dot guard:
    crisp full-contrast spots at 60 deg are a procedural tell)."""
    x = (abs(lat) - 0.6) / (1.15 - 0.6)
    x = min(max(x, 0.0), 1.0)
    return 1.0 - 0.55 * (x * x * (3.0 - 2.0 * x))


def _add_small_storms(
    reg: VortexRegistry,
    rng: np.random.Generator,
    zones: list[tuple[float, float]],
    belts: list[tuple[float, float]],
    profiles: LatProfiles,
    density: float,
) -> None:
    for center, width, is_belt in (
        [(c, w, False) for c, w in zones] + [(c, w, True) for c, w in belts]
    ):
        count = int(rng.poisson(density * 3.5))
        if count == 0:
            continue
        lons = _poisson_lons(rng, count, min_sep=0.12)
        for lon in lons:
            u01 = float(rng.uniform(0.0, 1.0))
            r = 0.007 + (0.020 - 0.007) * u01 * u01  # below the oval size range
            lat = float(np.clip(
                center + rng.normal(0.0, 0.30 * width), -MAX_VORTEX_LAT, MAX_VORTEX_LAT
            ))
            # Contrast tied to size and latitude; belts get dark spots, zones
            # bright ones (subtle either way — these are texture, not features).
            base = (0.08 + 5.0 * r) * _veil(lat)
            brightness = -0.8 * base if is_belt else base
            s = -_ambient_sign(profiles, lat) * (0.5 if is_belt else 1.0) * 0.004 * (r / 0.012)
            reg.vortices.append(
                Vortex(lat, lon, r, s, KIND_OVAL, tint=0.0, brightness=brightness)
            )
            # A fraction spawn pre-sheared: a weaker twin trailing in longitude
            # reads as a half-dissolved, elongated storm.
            if rng.uniform() < 0.3:
                trail = (lon + float(rng.normal(2.2, 0.6)) * r + np.pi) % (2 * np.pi) - np.pi
                reg.vortices.append(
                    Vortex(lat, trail, r * 1.3, s * 0.4, KIND_OVAL,
                           tint=0.0, brightness=brightness * 0.55)
                )


def _enforce_cap(reg: VortexRegistry) -> None:
    """Keep the population under MAX_VORTICES by dropping the smallest,
    faintest ovals first (heroes/barges/pearls/polar are never dropped)."""
    excess = len(reg.vortices) - MAX_VORTICES
    if excess <= 0:
        return
    ovals = sorted(
        (v for v in reg.vortices if v.kind == KIND_OVAL),
        key=lambda v: (v.r_core, abs(v.brightness)),
    )
    drop = set(map(id, ovals[:excess]))
    reg.vortices = [v for v in reg.vortices if id(v) not in drop]
