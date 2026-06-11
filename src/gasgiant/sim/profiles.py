"""1D latitude profiles: zonal jet velocity, its streamfunction, shear, and
the band stamp patterns. Built exactly (numpy trapezoid integration) and
uploaded as LUT textures — no erf approximations in shaders.

Conventions: profiles are sampled at N uniform latitudes from +pi/2 (index 0)
down to -pi/2, matching texture v = (pi/2 - lat) / pi. Velocity u is physical
eastward speed on the unit sphere (rad/time at the equator scale); the
streamfunction pairing is u = -d(psi)/d(phi), v = (1/cos phi) d(psi)/d(lambda),
the spherically divergence-free pairing.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from gasgiant.params.model import BandsParams, JetsParams
from gasgiant.params.seeds import subseed
from gasgiant.sim.bands import BandLayout

PROFILE_SAMPLES = 2048

# Velocity fades to zero poleward of this band (polar patches own the caps).
POLAR_FADE_START = np.deg2rad(74.0)
POLAR_FADE_END = np.deg2rad(84.0)


@dataclass(frozen=True)
class LatProfiles:
    lat: np.ndarray          # (N,) latitudes, descending from +pi/2
    u: np.ndarray            # (N,) zonal jet speed
    psi: np.ndarray          # (N,) streamfunction of the jets
    shear_norm: np.ndarray   # (N,) |du/dphi| normalized to [0, 1]
    belt_mask: np.ndarray    # (N,) 1 inside dark (cyclonic) bands
    t0_stamp: np.ndarray     # (N,) banded color-index stamp
    t1_stamp: np.ndarray     # (N,) banded height stamp
    max_speed: float

    def dyn_lut(self) -> np.ndarray:
        """(N, 4) float32: u, psi, shear_norm, belt_mask."""
        return np.stack([self.u, self.psi, self.shear_norm, self.belt_mask], axis=1).astype(
            np.float32
        )

    def stamp_lut(self) -> np.ndarray:
        """(N, 4) float32: t0_stamp, t1_stamp, polar_fade, unused."""
        fade = polar_fade(self.lat)
        zero = np.zeros_like(self.lat)
        return np.stack([self.t0_stamp, self.t1_stamp, fade, zero], axis=1).astype(np.float32)


def polar_fade(lat: np.ndarray) -> np.ndarray:
    """1 -> 0 ramp over the polar fade band (smoothstep)."""
    x = (np.abs(lat) - POLAR_FADE_START) / (POLAR_FADE_END - POLAR_FADE_START)
    x = np.clip(x, 0.0, 1.0)
    return 1.0 - (x * x * (3.0 - 2.0 * x))


def build_profiles(
    seed: int, bands: BandLayout, bands_params: BandsParams, jets: JetsParams
) -> LatProfiles:
    n = PROFILE_SAMPLES
    lat = np.linspace(np.pi / 2.0, -np.pi / 2.0, n)
    rng = subseed(seed, "jets")

    edges = bands.edges.astype(np.float64)  # descending, [pi/2 ... -pi/2]
    interior = edges[1:-1]
    widths = -np.diff(edges)  # positive band widths

    u = np.zeros(n)
    # One jet per interior band boundary, alternating direction, amplitude
    # jittered and decaying poleward.
    for j, edge_lat in enumerate(interior):
        w_adj = min(widths[j], widths[j + 1])
        jet_width = max(0.25 * w_adj, 0.015)
        sign = 1.0 if j % 2 == 0 else -1.0
        amp = 0.55 * (1.0 + 0.5 * rng.uniform(-1.0, 1.0))
        decay = (np.cos(edge_lat) ** 2) * jets.polar_decay + (1.0 - jets.polar_decay)
        u += sign * amp * decay * np.exp(-(((lat - edge_lat) / jet_width) ** 2))

    # Equatorial superrotation jet (the dominant jet on both Jupiter and Saturn).
    u += jets.equatorial_speed * np.exp(-((lat / jets.equatorial_width) ** 2))
    u *= jets.strength
    u *= polar_fade(lat)

    # psi(phi) with u = -dpsi/dphi  =>  psi = -integral(u dphi).
    # lat is descending so cumulative trapezoid over the array runs from the
    # north pole; the sign works out via the negative d(lat) steps.
    dlat = np.diff(lat)  # negative steps
    psi = np.zeros(n)
    psi[1:] = -np.cumsum(0.5 * (u[1:] + u[:-1]) * dlat)

    du = np.gradient(u, lat)
    shear = np.abs(du)
    shear_norm = shear / max(shear.max(), 1e-9)

    t0, t1, belt = _stamp_profiles(lat, bands, bands_params)

    return LatProfiles(
        lat=lat,
        u=u,
        psi=psi,
        shear_norm=shear_norm,
        belt_mask=belt,
        t0_stamp=t0,
        t1_stamp=t1,
        max_speed=float(np.abs(u).max()),
    )


def _stamp_profiles(
    lat: np.ndarray, bands: BandLayout, params: BandsParams
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Banded T0/T1 stamps and the belt mask, with smoothstep edge transitions
    (the same shape the init/relaxation kernels expect)."""
    values = bands.values.astype(np.float64)
    heights = bands.heights.astype(np.float64)
    is_belt = (values < np.median(values)).astype(np.float64)

    t0 = np.full_like(lat, values[0])
    t1 = np.full_like(lat, heights[0])
    belt = np.full_like(lat, is_belt[0])
    soft = max(params.edge_softness, 1e-4)
    for j in range(1, len(values)):
        e = bands.edges[j]
        x = np.clip((e + soft - lat) / (2.0 * soft), 0.0, 1.0)
        t = x * x * (3.0 - 2.0 * x)
        t0 = t0 * (1.0 - t) + values[j] * t
        t1 = t1 * (1.0 - t) + heights[j] * t
        belt = belt * (1.0 - t) + is_belt[j] * t
    return t0, t1, belt
