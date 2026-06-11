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
        u = np.interp(-lats, -profiles.lat, profiles.u)  # profile lats descending
        dlon = u / np.maximum(np.cos(lats), 0.2) * dt
        for v, d in zip(self.vortices, dlon, strict=True):
            if v.kind != KIND_POLAR:
                v.lon = float((v.lon + d + np.pi) % (2 * np.pi) - np.pi)


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
    return reg


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
