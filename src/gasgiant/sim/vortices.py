"""The vortex registry: seeded storm populations with latitude-conditioned
placement, power-law sizes, and Poisson-disc longitude spacing.

Each vortex contributes a Gaussian streamfunction
    psi_v = strength * exp(-(d/r_core)^2),  d = great-circle distance,
which yields a smooth coherent rotating core (tangential speed peaks at
d = r_core/sqrt(2)) and composes trivially in psi. Sign convention: vortex
rotation matches (heroes/ovals/pearls) or opposes (barges) the ambient shear
vorticity at its latitude, which is what lets matched storms persist and
mismatched barges get sheared into cigars.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from gasgiant.params.model import PolesParams, StormsParams
from gasgiant.params.seeds import subseed
from gasgiant.sim.bands import BandLayout
from gasgiant.sim.profiles import LatProfiles

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
    # Downstream direction of the ambient jet (heroes: wake side), +1 east.
    wake_dir: float = 0.0
    # CPU-only (never packed into the SSBO): merger hysteresis countdown.
    cooldown: int = 0


@dataclass
class VortexRegistry:
    vortices: list[Vortex] = field(default_factory=list)

    def heroes(self) -> list[Vortex]:
        return [v for v in self.vortices if v.kind == KIND_HERO]

    def pack_ssbo(self) -> np.ndarray:
        """(N, 12) float32, three vec4 per vortex:
        [x, y, z, r_core], [strength, kind, tint, brightness], [wake_dir, 0, 0, 0].
        Vectorized: this runs every sim step."""
        n = len(self.vortices)
        if n == 0:
            return np.zeros((1, 12), dtype=np.float32)
        fields = np.array(
            [
                (v.lat, v.lon, v.r_core, v.strength, v.kind, v.tint, v.brightness, v.wake_dir)
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


def zonal_rate(profiles: LatProfiles, lats: np.ndarray) -> np.ndarray:
    """d(lon)/dt of the zonal drift at the given latitudes — THE drift formula,
    shared by drift(), the merger converging gate, and seeded-pair placement
    so they can never disagree. profiles.lat is DESCENDING and np.interp
    silently returns garbage on descending xp, hence the negated axis."""
    u = np.interp(-lats, -profiles.lat, profiles.u)
    return u / np.maximum(np.cos(lats), 0.2)


def _merge_pair(a: Vortex, b: Vortex, profiles: LatProfiles) -> Vortex:
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
        wake_dir=1.0 if u_here >= 0.0 else -1.0,
        cooldown=MERGE_COOLDOWN,
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
    eligible = (peer | absorb | absorb.T) & live[:, None] & live[None, :]
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
        else:
            removed.update((i, j))
            product = _merge_pair(a, b, profiles)
            products.append(product)
            resolved.append((a, b, product))
    # One identity-based rebuild — never list.remove (dataclass == is field
    # equality and could drop the wrong entry). Products appended at the end
    # are same-step-ineligible by construction.
    reg.vortices = [v for idx, v in enumerate(vs) if idx not in removed] + products
    return resolved


def _ambient_sign(profiles: LatProfiles, lat: float) -> float:
    """Sign of the ambient relative vorticity (~ -du/dphi) at a latitude."""
    lats = profiles.lat
    du = np.gradient(profiles.u, lats)
    s = -float(np.interp(-lat, -lats, du))
    return 1.0 if s >= 0.0 else -1.0


def _band_centers(bands: BandLayout, want_belt: bool) -> list[tuple[float, float]]:
    """(center latitude, width) of zones or belts within the vortex latitude range."""
    values = bands.values
    median = float(np.median(values))
    out = []
    for j in range(len(values)):
        center = 0.5 * (bands.edges[j] + bands.edges[j + 1])
        width = float(bands.edges[j] - bands.edges[j + 1])
        if abs(center) > MAX_VORTEX_LAT:
            continue
        is_belt = values[j] < median
        if is_belt == want_belt:
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
    # Cyclonic sense, mirrored across hemispheres.
    s_central = pole_sign * 0.032 * strength

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
) -> VortexRegistry:
    rng = subseed(seed, "storms")
    reg = VortexRegistry()

    zones = _band_centers(bands, want_belt=False)
    belts = _band_centers(bands, want_belt=True)

    # Hero anticyclones (GRS class): tropical-to-temperate zone latitudes.
    tropical = [z for z in zones if 0.15 < abs(z[0]) < 0.75] or zones
    for _ in range(storms.hero_count):
        if not tropical:
            break
        center, _w = tropical[rng.integers(0, len(tropical))]
        lat = float(np.clip(center + rng.normal(0.0, 0.02), -MAX_VORTEX_LAT, MAX_VORTEX_LAT))
        r = storms.hero_radius * (1.0 + 0.2 * rng.uniform(-1.0, 1.0))
        s = _ambient_sign(profiles, lat) * storms.hero_strength * 0.045
        u_here = float(np.interp(-lat, -profiles.lat, profiles.u))
        reg.vortices.append(
            Vortex(lat, float(rng.uniform(-np.pi, np.pi)), r, s, KIND_HERO,
                   tint=0.9, brightness=0.05,
                   wake_dir=1.0 if u_here >= 0.0 else -1.0)
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
            s = _ambient_sign(profiles, lat) * 0.012 * (r / 0.03)
            reg.vortices.append(
                Vortex(lat, lon, r, s, KIND_OVAL, tint=0.1, brightness=0.22)
            )

    # Brown barges: weak cyclones inside belts (sign opposes ambient shear),
    # sheared into dark cigars by the jets.
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
            s = _ambient_sign(profiles, lat) * 0.008
            reg.vortices.append(
                Vortex(lat, lon, 0.02, s, KIND_PEARL, tint=0.05, brightness=0.25)
            )

    # Small-storm field: sub-oval white spots and dark spots in loose latitude
    # rows (real maps are peppered with them). Own seed stream; defaults off.
    if storms.small_density > 0.0:
        _add_small_storms(reg, subseed(seed, "small-storms"), zones, belts,
                          profiles, storms.small_density)

    # Stamp contrast (1 = v1): scales how hard non-hero storms register in the
    # tracers; the velocity strength is untouched.
    if storms.stamp_contrast != 1.0:
        for v in reg.vortices:
            if v.kind != KIND_HERO:
                v.brightness *= storms.stamp_contrast
                v.tint *= storms.stamp_contrast

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
            storms.merge_rate, dt, dev_steps,
        )
    return reg


def _seed_convergent_pairs(
    reg: VortexRegistry,
    rng: np.random.Generator,
    zones: list[tuple[float, float]],
    profiles: LatProfiles,
    merge_rate: float,
    dt: float,
    dev_steps: int,
) -> None:
    """Spawn companion ovals placed KINEMATICALLY: measure the actual closure
    rate at the site via zonal_rate, draw a target merge step, and set the
    longitude gap so capture happens near that step. A fixed gap range cannot
    work — du/dphi at zone centers is ~1-2, an order short of closing a
    Poisson-disc gap in 500 steps."""
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
        if closure * 500.0 < 0.02:
            # Flat shear: place just outside capture so any drift nudges it in.
            gap = 1.1 * dlon_capture
        else:
            target_step = int(rng.integers(80, min(421, max(dev_steps, 81))))
            gap = 0.8 * dlon_capture + closure * target_step
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
            s = _ambient_sign(profiles, lat) * (0.5 if is_belt else 1.0) * 0.004 * (r / 0.012)
            if is_belt:
                s = -s
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
