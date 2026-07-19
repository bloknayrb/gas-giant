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
from gasgiant.sim import vorticity_ref
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
    # (lat_lo, lat_hi, lon, halfwidth) of the faded sector, radians.
    fade_sector: tuple[float, float, float, float] = (0.0, 0.0, 0.0, 0.0)
    # (N,) relative vorticity of the jets: ω_jet = −(1/cosφ) d(u cosφ)/dφ
    # vorticity_ref.jet_vorticity uses ascending lat; we store in descending
    # order (same as self.lat) so the LUT v-coordinate matches latProfileUV.
    omega_jet: np.ndarray = None  # type: ignore[assignment]

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

    def omega_lut(self) -> np.ndarray:
        """(N, 4) float32 LUT for upload via gpu.lut_texture.

        R channel = ω_jet (relative vorticity of the zonal jets).
        G/B/A = 0 (reserved for future vorticity-mode fields).
        """
        omega = self.omega_jet if self.omega_jet is not None else np.zeros_like(self.lat)
        zero = np.zeros_like(self.lat)
        return np.stack([omega, zero, zero, zero], axis=1).astype(np.float32)


def polar_fade(lat: np.ndarray) -> np.ndarray:
    """1 -> 0 ramp over the polar fade band (smoothstep)."""
    x = (np.abs(lat) - POLAR_FADE_START) / (POLAR_FADE_END - POLAR_FADE_START)
    x = np.clip(x, 0.0, 1.0)
    return 1.0 - (x * x * (3.0 - 2.0 * x))


def build_profiles(
    seed: int, bands: BandLayout, bands_params: BandsParams, jets: JetsParams,
    hero_lat_deg: float | None = None,
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

    # Optional additive local zonal jet (e.g. a westward SEBs-analog jet under
    # an anticyclonic hero storm). Structural guard, not a magnitude check --
    # `x + 0.0` would still flip -0.0 bits elsewhere in u, so the default
    # 0.0 must skip the term entirely to stay a true no-op.
    if jets.local_jet_speed != 0.0:
        lat0 = np.deg2rad(jets.local_jet_latitude)
        u += jets.local_jet_speed * np.exp(-(((lat - lat0) / jets.local_jet_width) ** 2))

    u *= jets.strength
    u *= polar_fade(lat)

    # Carve-and-impose hero jet override (jets.hero_bracket_*). Structural guard
    # (mirrors the local_jet != 0.0 skip): default north==south==0 -> the whole
    # block is skipped, byte-identical. Requires a pinned hero (hero_lat_deg).
    # Applied AFTER strength+polar_fade so the flat pedestal samples the same u
    # that is blended, and BEFORE psi/shear/omega so every derived field sees the
    # carved u. The bracket carries jets.strength (baked into the amplitudes) so
    # a later strength retune rescales it consistently; it is intentionally NOT
    # polar_faded (a documented LIMIT for a high-latitude hero).
    if hero_lat_deg is not None and (
        jets.hero_bracket_north != 0.0 or jets.hero_bracket_south != 0.0
    ):
        hero = np.deg2rad(hero_lat_deg)
        # C1 window: 1 within `window` deg of the hero, smoothstep to 0 by
        # window+feather deg. Zero derivative at both ends -> no du/dphi jump.
        full = np.deg2rad(jets.hero_bracket_window)
        outer = np.deg2rad(jets.hero_bracket_window + jets.hero_bracket_feather)
        x = np.clip((np.abs(lat - hero) - full) / max(outer - full, 1e-9), 0.0, 1.0)
        w = 1.0 - (x * x * (3.0 - 2.0 * x))            # 1 near hero, 0 outside
        # Flat pedestal = the base u at the hero (keeps the bracket zero-crossing
        # on the hero; a sloped ramp would reintroduce seed-dependent shear).
        pedestal = float(np.interp(hero, lat[::-1], u[::-1]))
        north_c = np.deg2rad(hero_lat_deg + jets.hero_bracket_north_offset)
        south_c = np.deg2rad(hero_lat_deg + jets.hero_bracket_south_offset)
        bracket = jets.strength * (
            jets.hero_bracket_north
            * np.exp(-(((lat - north_c) / jets.hero_bracket_north_width) ** 2))
            + jets.hero_bracket_south
            * np.exp(-(((lat - south_c) / jets.hero_bracket_south_width) ** 2))
        )
        u = u * (1.0 - w) + (pedestal + bracket) * w

    # psi(phi) with u = -dpsi/dphi  =>  psi = -integral(u dphi).
    # lat is descending so cumulative trapezoid over the array runs from the
    # north pole; the sign works out via the negative d(lat) steps.
    dlat = np.diff(lat)  # negative steps
    psi = np.zeros(n)
    psi[1:] = -np.cumsum(0.5 * (u[1:] + u[:-1]) * dlat)

    du = np.gradient(u, lat)
    shear = np.abs(du)
    shear_norm = shear / max(shear.max(), 1e-9)

    # Per-edge softness diversity on its own stream: some edges diffuse, some
    # sharp. edge_diversity == 0 gives uniform v1 softness.
    soft_rng = subseed(seed, "edge-softness")
    soft_mult = np.exp(
        bands_params.edge_diversity * soft_rng.uniform(-1.2, 1.2, max(len(bands.values) - 1, 1))
    )
    t0, t1, belt = _stamp_profiles(lat, bands, bands_params, soft_mult)

    # Compute ω_jet APPEND-ONLY after all seeded draws above.
    # vorticity_ref.jet_vorticity expects ascending lat; lat here is descending,
    # so we flip, compute, then flip back to match the descending LUT convention.
    lat_asc = lat[::-1]
    u_asc = u[::-1]
    omega_jet_asc = vorticity_ref.jet_vorticity(u_asc, lat_asc)
    omega_jet = omega_jet_asc[::-1].copy()  # back to descending order

    return LatProfiles(
        lat=lat,
        u=u,
        psi=psi,
        shear_norm=shear_norm,
        belt_mask=belt,
        t0_stamp=t0,
        t1_stamp=t1,
        max_speed=float(np.abs(u).max()),
        fade_sector=bands.fade_sector,
        omega_jet=omega_jet,
    )


# Green/amber/red thresholds for the natural-bearing seat meter. Coarse bands
# (the reading is a pre-development proxy; the developed velocity-zero sits
# ~1.8 deg poleward), calibrated so warm's iconic -22 hero reads amber/red
# (natural bearing poor -> enable the bracket) and its best natural seat ~-40
# reads green.
_SEAT_GREEN = 0.15
_SEAT_AMBER = 0.0


def seat_quality(
    profiles: LatProfiles, lat_deg: float, r_core_deg: float, spin_sign: float = 1.0
) -> float:
    """Natural two-sided bearing quality at `lat_deg` for a storm of half-extent
    `r_core_deg`, from the (bracket-off) profile. spin_sign +1 = anticyclone
    (wants westward equatorward rim + eastward poleward rim), -1 = cyclone.
    quality = min(-spin*u_equatorward, spin*u_poleward) - 0.5*|u_center|;
    two_sided is magnitude-based (not sign-only), so a correct-sign-but-weak
    bearing scores low. Reported as a coarse pre-development proxy.

    (Deliberate simplification vs spec 3.1: the explicit r_core-relative
    'moat-orientation weight' is DROPPED. two_sided is already magnitude-based,
    which addresses the spec's core 'not sign-only' concern; r_core_deg still
    sets WHERE the rims are sampled. The coarse green/amber/red band absorbs the
    calibration the weight would have carried. Restore the weight only if
    calibration shows a strong-sign/weak-moat false-green.)"""
    lat_asc = profiles.lat[::-1]
    u_asc = profiles.u[::-1]

    def u_at(ld: float) -> float:
        return float(np.interp(np.deg2rad(ld), lat_asc, u_asc))

    equatorward_rim = u_at(lat_deg + r_core_deg)   # less-negative side for a SH hero
    poleward_rim = u_at(lat_deg - r_core_deg)
    center = abs(u_at(lat_deg))
    two_sided = min(-spin_sign * equatorward_rim, spin_sign * poleward_rim)
    return two_sided - 0.5 * center


def seat_scan(
    profiles: LatProfiles, lats_deg, r_core_deg: float, spin_sign: float = 1.0
) -> list[tuple[float, float]]:
    """(lat_deg, quality) over a latitude sweep -- the GUI's 'find a good seat'
    readout. Diagnostic only: never moves the storm."""
    return [
        (float(ld), seat_quality(profiles, float(ld), r_core_deg, spin_sign))
        for ld in lats_deg
    ]


def seat_band(quality: float) -> str:
    """Coarse green/amber/red classification of a seat_quality value."""
    if quality >= _SEAT_GREEN:
        return "green"
    if quality >= _SEAT_AMBER:
        return "amber"
    return "red"


MAX_LANES = 16


def select_lanes(
    seed: int, bands: BandLayout, lane_density: float
) -> list[tuple[float, float]]:
    """(latitude, strength) of thin dark lane lines, drawn analytically at
    derive time. Lanes sit at jet cores (band interior edges) — a 1-3 px
    line at export resolution cannot survive the sim tracer grid, so this is
    a render-side feature. Own seed stream; density 0 selects nothing."""
    if lane_density <= 0.0:
        return []
    rng = subseed(seed, "lanes")
    lanes: list[tuple[float, float]] = []
    for edge in bands.edges[1:-1]:
        # Draws happen for every edge regardless of selection so a density
        # change never reshuffles which edges carry lanes.
        roll = float(rng.uniform(0.0, 1.0))
        strength = float(rng.uniform(0.12, 0.30))
        if roll < lane_density and abs(edge) < 1.1 and len(lanes) < MAX_LANES:
            lanes.append((float(edge), strength))
    return lanes


def select_wave_latitudes(bands: BandLayout, profiles: LatProfiles) -> tuple[float, float]:
    """(festoon latitude, ribbon latitude): the band edge nearest +7 deg (the
    NEB-S side — real festoons root on the belt edge NORTH of the equator, so
    the target is signed) for the festoon/hot-spot train, and the strongest
    mid-latitude jet for the ribbon wave (sign-blind by design: Saturn's
    ribbon is hemisphere-generic)."""
    interior = bands.edges[1:-1].astype(np.float64)
    if interior.size == 0:
        return 0.12, 0.82
    # Signed distance to the +0.12 rad target; a sign-blind |abs| pick chose
    # -7.31 deg over +5.91 deg on the Cassini template (review F12). Templates
    # with no edge inside +/-0.1 rad of the signed target keep the old
    # sign-blind nearest pick rather than jumping to a far-north edge.
    signed_dist = np.abs(interior - 0.12)
    if float(np.min(signed_dist)) <= 0.1:
        festoon = float(interior[np.argmin(signed_dist)])
    else:
        festoon = float(interior[np.argmin(np.abs(np.abs(interior) - 0.12))])

    mid = [e for e in interior if 0.6 < abs(e) < 1.0]
    if mid:
        speeds = [abs(float(np.interp(-e, -profiles.lat, profiles.u))) for e in mid]
        ribbon = float(mid[int(np.argmax(speeds))])
    else:
        ribbon = 0.82
    return festoon, ribbon


def select_hero_festoon_latitude(
    bands: BandLayout, hero_lat: float, primary_festoon_lat: float
) -> float | None:
    """Root latitude for the hero-adjacent festoon train (FESTOON2), or None.

    The interior band edge nearest the hero storm: on a hollow-straddling
    placement its plumes weave through the hero's wake lane with tails
    brushing the collar (the reference's streamers rooting on the SEB edge
    next to the GRS). None when the nearest edge is farther than 0.15 rad
    (a train rooted away from the storm is just a second equatorial comb)
    or when it coincides with the primary festoon edge (never double-train
    one edge). Deterministic — layout-derived, no RNG."""
    interior = bands.edges[1:-1].astype(np.float64)
    if interior.size == 0:
        return None
    d = np.abs(interior - hero_lat)
    i = int(np.argmin(d))
    if float(d[i]) > 0.15:
        return None
    if abs(float(interior[i]) - primary_festoon_lat) < 1e-6:
        return None
    return float(interior[i])


def _stamp_profiles(
    lat: np.ndarray,
    bands: BandLayout,
    params: BandsParams,
    soft_mult: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Banded T0/T1 stamps and the belt mask, with smoothstep edge transitions
    (the same shape the init/relaxation kernels expect). soft_mult: per-edge
    softness multipliers (edge diversity). T0 reads the STAMP view of the
    band values (bands.belt_fade applied); the belt mask reads the frozen
    pre-fade identity, so a fully faded belt still counts as a belt for
    every dynamics consumer."""
    values = bands.stamp_values.astype(np.float64)
    heights = bands.heights.astype(np.float64)
    is_belt = bands.is_belt.astype(np.float64)

    t0 = np.full_like(lat, values[0])
    t1 = np.full_like(lat, heights[0])
    belt = np.full_like(lat, is_belt[0])
    base_soft = max(params.edge_softness, 1e-4)
    for j in range(1, len(values)):
        e = bands.edges[j]
        soft = base_soft * (float(soft_mult[j - 1]) if soft_mult is not None else 1.0)
        x = np.clip((e + soft - lat) / (2.0 * soft), 0.0, 1.0)
        t = x * x * (3.0 - 2.0 * x)
        t0 = t0 * (1.0 - t) + values[j] * t
        t1 = t1 * (1.0 - t) + heights[j] * t
        belt = belt * (1.0 - t) + is_belt[j] * t
    return t0, t1, belt
