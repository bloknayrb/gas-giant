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

from gasgiant.params.model import StormsParams
from gasgiant.params.seeds import subseed
from gasgiant.sim.bands import BandLayout
from gasgiant.sim.profiles import LatProfiles

# No discrete vortices poleward of this (polar patches own the caps; the
# nesting blend band must stay storm-free).
MAX_VORTEX_LAT = np.deg2rad(68.0)

KIND_OVAL = 0.0
KIND_HERO = 1.0
KIND_BARGE = 2.0
KIND_PEARL = 3.0
KIND_KH = 4.0


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


@dataclass
class VortexRegistry:
    vortices: list[Vortex] = field(default_factory=list)

    def heroes(self) -> list[Vortex]:
        return [v for v in self.vortices if v.kind == KIND_HERO]

    def pack_ssbo(self) -> np.ndarray:
        """(N, 8) float32, two vec4 per vortex:
        [x, y, z, r_core], [strength, kind, tint, brightness]."""
        n = len(self.vortices)
        out = np.zeros((max(n, 1), 8), dtype=np.float32)
        for i, v in enumerate(self.vortices):
            cl = np.cos(v.lat)
            out[i, 0:4] = (cl * np.cos(v.lon), np.sin(v.lat), cl * np.sin(v.lon), v.r_core)
            out[i, 4:8] = (v.strength, v.kind, v.tint, v.brightness)
        return out

    def drift(self, profiles: LatProfiles, dt: float) -> None:
        """Advect vortex centers with the ambient zonal flow."""
        lats = profiles.lat
        for v in self.vortices:
            u = float(np.interp(-v.lat, -lats, profiles.u))  # lats descending
            v.lon = float((v.lon + u / max(np.cos(v.lat), 0.2) * dt + np.pi) % (2 * np.pi) - np.pi)


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


def generate_vortices(
    seed: int,
    bands: BandLayout,
    profiles: LatProfiles,
    storms: StormsParams,
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
        reg.vortices.append(
            Vortex(lat, float(rng.uniform(-np.pi, np.pi)), r, s, KIND_HERO,
                   tint=0.9, brightness=0.05)
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

    return reg
