"""Vortex seeding chirality (F1 fix, 2026-07-17): every seeded/cast storm
co-rotates with the local ambient shear vorticity (not counter-rotates), and
polar caps are genuinely cyclonic per hemisphere. The trap: `strength` is a
psi AMPLITUDE (psi += strength*exp(-q^2), u = -dpsi/dphi, zeta = +laplacian
(psi)), so a vortex's own core vorticity is omega = -sign(strength) -- the
OPPOSITE sign of its `strength` field. Co-rotation with ambient sign S means
strength = -S * |magnitude|.
"""

from __future__ import annotations

import numpy as np

from gasgiant.params.model import CastKind, PlanetParams, PolesParams, StormOverride
from gasgiant.sim.bands import generate_bands
from gasgiant.sim.profiles import build_profiles
from gasgiant.sim.vortices import (
    KIND_BARGE,
    KIND_HERO,
    KIND_OVAL,
    KIND_PEARL,
    KIND_POLAR,
    Vortex,
    _ambient_sign,
    _merge_pair,
    generate_vortices,
)

AMBIENT_SIGNED_KINDS = (KIND_HERO, KIND_OVAL, KIND_BARGE, KIND_PEARL)


def _rich_registry(seed: int = 11):
    """A registry with every ambient-signed seeded population turned on
    (heroes, ovals, barges, pearls, accents, hero companions, small storms),
    poles off and mergers off so the ambient-sign check below is not
    contaminated by the relative-rule (merge/convergent-companion) classes."""
    p = PlanetParams(seed=seed)
    p.storms.hero_count = 2
    p.storms.oval_density = 3.0
    p.storms.barge_density = 2.0
    p.storms.pearls_count = 6
    p.storms.accent_count = 1
    p.storms.hero_companions = 2
    p.storms.small_density = 2.0
    p.storms.merge_rate = 0.0
    bands = generate_bands(seed, p.bands)
    profiles = build_profiles(seed, bands, p.bands, p.jets)
    reg = generate_vortices(seed, bands, profiles, p.storms, poles=None)
    return profiles, reg


def test_seeded_storms_corotate_with_ambient_shear():
    """sign(strength) == -_ambient_sign(profiles, lat) for every seeded,
    ambient-signed kind (heroes, white/accent ovals, small storms [KIND_OVAL],
    barges, pearls/companions)."""
    profiles, reg = _rich_registry()
    checked = {k: 0 for k in AMBIENT_SIGNED_KINDS}
    for v in reg.vortices:
        if v.kind not in AMBIENT_SIGNED_KINDS:
            continue
        expected = -_ambient_sign(profiles, v.lat)
        actual = 1.0 if v.strength >= 0.0 else -1.0
        assert actual == expected, (
            f"kind={v.kind} lat={np.rad2deg(v.lat):.2f} strength={v.strength} "
            f"expected sign {expected}"
        )
        checked[v.kind] += 1
    # Sanity: the rich config actually exercised every kind (a config typo
    # that silently produced zero of a kind would make the loop above vacuous).
    for k, n in checked.items():
        assert n > 0, f"kind {k} was never seeded — test config is not exercising it"
    # KIND_OVAL conflates four seeding sites (white ovals, accents, zone + belt
    # small storms), so `checked[KIND_OVAL] > 0` does NOT prove the belt path
    # ran. That path is the fragile one: its sign is BYTE-IDENTICAL to pre-fix
    # (the old `if is_belt: s = -s` folded into the leading minus of
    # `-_ambient_sign * (0.5 if is_belt else 1.0)`), so a regression that
    # re-introduces the explicit belt flip is caught ONLY if a belt small storm
    # is actually present. Belt small storms are the sole KIND_OVAL site with
    # negative brightness (dark spots, -0.8*base) at sub-oval radius; pin >=1
    # fired so the co-rotation loop above provably exercised it.
    belt_smalls = [v for v in reg.vortices
                   if v.kind == KIND_OVAL and v.brightness < 0.0 and v.r_core < 0.04]
    assert belt_smalls, "no belt small-storm seeded — the belt-path sign is untested"


def test_polar_caps_are_cyclonic_per_hemisphere():
    """zeta_center = +pole_sign (cyclonic: positive in the north, negative in
    the south), i.e. sign(strength) == -pole_sign under the psi-amplitude trap.
    Covers cyclone_cluster (central + ring), plain_vortex, polygon_jet, and the
    background field population, both hemispheres."""
    p = PlanetParams(seed=23)
    p.poles = PolesParams.model_validate({
        "north": {"style": "cyclone_cluster", "cyclone_count": 5, "strength": 1.0,
                   "field_density": 1.0},
        "south": {"style": "plain_vortex", "strength": 1.0, "field_density": 1.0},
    })
    bands = generate_bands(23, p.bands)
    profiles = build_profiles(23, bands, p.bands, p.jets)
    reg = generate_vortices(23, bands, profiles, p.storms, p.poles)
    polar = [v for v in reg.vortices if v.kind == KIND_POLAR]
    assert polar
    north_seen = south_seen = 0
    for v in polar:
        pole_sign = 1.0 if v.lat > 0.0 else -1.0
        expected = -pole_sign
        actual = 1.0 if v.strength >= 0.0 else -1.0
        assert actual == expected, f"pole_sign={pole_sign} lat={np.rad2deg(v.lat):.2f} strength={v.strength}"
        if pole_sign > 0:
            north_seen += 1
        else:
            south_seen += 1
    assert north_seen > 0 and south_seen > 0


def test_cast_barge_matches_seeded_barge_sign_convention():
    """The cast path's BARGE base (+0.006) times its `sign` (-_ambient_sign)
    must reproduce the seeded barge formula (s = -_ambient_sign * 0.006)
    BYTE-EXACTLY: -(a*b) == (-a)*b for a in {+1.0, -1.0} (plan F1)."""
    p = PlanetParams(seed=5)
    lat_deg = -18.0
    p.storms.cast = [
        StormOverride(kind=CastKind.BARGE, lat_deg=lat_deg, lon_deg=0.0, radius=0.03)
    ]
    bands = generate_bands(5, p.bands)
    profiles = build_profiles(5, bands, p.bands, p.jets)
    reg = generate_vortices(5, bands, profiles, p.storms, poles=None)
    cast_barges = [v for v in reg.vortices if v.kind == KIND_BARGE and v.origin == "cast"]
    assert len(cast_barges) == 1
    v = cast_barges[0]
    expected = -_ambient_sign(profiles, v.lat) * 0.006
    assert v.strength == expected  # byte-exact, not just same-sign


def test_merge_product_inherits_parent_sign():
    """_merge_pair is a RELATIVE rule (sign copied from parent a), unaffected
    by the F1 ambient-sign flip mechanics -- but must still hold given
    post-flip (now negative-at-this-lat) parent inputs."""
    p = PlanetParams(seed=1)
    bands = generate_bands(1, p.bands)
    profiles = build_profiles(1, bands, p.bands, p.jets)
    lat = 0.3
    ambient = -_ambient_sign(profiles, lat)  # co-rotating sign at this latitude
    a = Vortex(lat, 0.0, 0.03, ambient * 0.012, KIND_OVAL)
    b = Vortex(lat, 0.05, 0.025, ambient * 0.010, KIND_OVAL)
    product = _merge_pair(a, b, profiles)
    assert np.sign(product.strength) == np.sign(a.strength) == np.sign(b.strength)


def test_convergent_companion_inherits_host_sign():
    """_seed_convergent_pairs copies `sign = 1.0 if host.strength > 0.0 else
    -1.0` -- a relative rule. With merge_rate high enough to force a spawn,
    the companion's sign must match its host's (now co-rotating) sign."""
    p = PlanetParams(seed=9)
    p.storms.oval_density = 3.0
    p.storms.merge_rate = 1.0  # force spawns (rng.uniform() >= 0.5*rate never true at 1.0)
    bands = generate_bands(9, p.bands)
    profiles = build_profiles(9, bands, p.bands, p.jets)
    reg = generate_vortices(9, bands, profiles, p.storms, poles=None,
                             dt=1e-3, dev_steps=200)
    # Non-vacuity: the loop below re-checks every KIND_OVAL, so it would pass on
    # the regular ovals even if zero companions spawned. Pin that convergent
    # pairs actually fired by comparing against the same scene with merge_rate=0
    # (no spawns): at seed 9 the spawns net +2 vortices (measured), so the count
    # must strictly rise.
    p_ref = p.model_copy(deep=True)
    p_ref.storms.merge_rate = 0.0
    reg_ref = generate_vortices(9, bands, profiles, p_ref.storms, poles=None,
                                dt=1e-3, dev_steps=200)
    assert len(reg.vortices) > len(reg_ref.vortices), (
        "no convergent companions spawned — the relative-sign rule is untested")
    ovals = [v for v in reg.vortices if v.kind == KIND_OVAL]
    # Group by (lat, sign) is fragile; instead just confirm every KIND_OVAL
    # vortex still lands on the ambient co-rotating sign at its own latitude
    # (companions share the host's lat-region sign since they spawn close by
    # and the ambient field is smooth there).
    assert ovals
    for v in ovals:
        expected = -_ambient_sign(profiles, v.lat)
        actual = 1.0 if v.strength >= 0.0 else -1.0
        assert actual == expected


def test_omega_equals_negative_sign_strength_convention():
    """CPU-level pin of the documented psi-amplitude trap: for a Gaussian
    vortex psi = strength * exp(-(d/r_core)^2) on a small local Cartesian
    patch (u = -dpsi/dy, v = dpsi/dx, matching the sim's u=-dpsi/dphi
    convention), the core vorticity omega = laplacian(psi) at the vortex
    center has the OPPOSITE sign of `strength`, for both signs of strength."""
    n = 121
    extent = 3.0
    x = np.linspace(-extent, extent, n)
    y = np.linspace(-extent, extent, n)
    dx = x[1] - x[0]
    xx, yy = np.meshgrid(x, y, indexing="xy")
    r_core = 1.0
    for strength in (1.0, -1.0, 2.5, -0.3):
        psi = strength * np.exp(-((xx**2 + yy**2) / r_core**2))
        # 5-point discrete Laplacian.
        lap = (
            np.roll(psi, 1, axis=0) + np.roll(psi, -1, axis=0)
            + np.roll(psi, 1, axis=1) + np.roll(psi, -1, axis=1)
            - 4.0 * psi
        ) / (dx * dx)
        center = n // 2
        omega_center = lap[center, center]
        assert np.sign(omega_center) == -np.sign(strength), (
            f"strength={strength}: omega={omega_center} (expected opposite sign)"
        )
