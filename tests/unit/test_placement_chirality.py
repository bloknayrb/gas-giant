"""Placement chirality: festoon rooting edge (F12) and hero wake flank (F06).

2026-07-02 review criteria: real festoons root on the NEB-S edge (positive
latitude, ~+5..+8 deg) and the GRS wake trails WNW — west of the storm with
an equatorward bias — not due east. Also covers the explicit `is_belt`
identity on BandLayout (single source; consumers must not re-derive from
values, or a future value edit could flip a band's class).
"""

from __future__ import annotations

import numpy as np

from gasgiant.params.model import PlanetParams
from gasgiant.params.presets import load_factory_preset
from gasgiant.sim.bands import BandLayout, generate_bands
from gasgiant.sim.profiles import build_profiles, select_wave_latitudes
from gasgiant.sim.vortices import (
    generate_vortices,
)

FACTORY = ["gas_giant_warm", "jupiter_vorticity", "jupiter_like", "saturn_pale", "ice_giant"]


def _bands_profiles(name: str):
    p = load_factory_preset(name)
    bands = generate_bands(p.seed, p.bands)
    profiles = build_profiles(p.seed, bands, p.bands, p.jets)
    return p, bands, profiles


# ---------------------------------------------------------------- festoons

def test_festoon_roots_on_neb_s_edge_jupiter_template():
    """F12: on the shipped Cassini template the +5.91 deg edge (NEB-S) must
    win over -7.31 deg (SEB-N) for the +6.9 deg target — signed distance,
    not |abs| distance."""
    for name in ("jupiter_vorticity", "gas_giant_warm", "jupiter_like"):
        _, bands, profiles = _bands_profiles(name)
        festoon, _ = select_wave_latitudes(bands, profiles)
        assert festoon > 0.0, f"{name}: festoon rooted at {np.rad2deg(festoon):.2f} deg (southern edge)"
        assert 0.08 < festoon < 0.15, f"{name}: festoon {np.rad2deg(festoon):.2f} deg outside +5..+8 deg"


def test_festoon_degenerate_template_falls_back_sign_blind():
    """A template with no northern edge near the target must keep the old
    nearest-|abs| behavior rather than jumping to a far-north edge."""
    edges = np.deg2rad(np.array([90.0, -5.0, -20.0, -90.0], dtype=np.float64))
    layout = BandLayout(
        edges=edges.astype(np.float32),
        values=np.array([0.7, 0.3, 0.7], dtype=np.float32),
        heights=np.array([0.6, 0.4, 0.6], dtype=np.float32),
    )
    p = PlanetParams(seed=7)
    profiles = build_profiles(7, layout, p.bands, p.jets)
    festoon, _ = select_wave_latitudes(layout, profiles)
    assert np.isclose(festoon, np.deg2rad(-5.0), atol=1e-6)


def test_festoon_prefers_in_window_signed_edge_over_closer_southern():
    """Signed selection: -7 deg is closer in |abs| terms than +9 deg for a
    +6.9 deg target, but +9 deg is inside the +/-0.1 rad window and must win."""
    edges = np.deg2rad(np.array([90.0, 9.0, -7.0, -90.0], dtype=np.float64))
    layout = BandLayout(
        edges=edges.astype(np.float32),
        values=np.array([0.7, 0.3, 0.7], dtype=np.float32),
        heights=np.array([0.6, 0.4, 0.6], dtype=np.float32),
    )
    p = PlanetParams(seed=7)
    profiles = build_profiles(7, layout, p.bands, p.jets)
    festoon, _ = select_wave_latitudes(layout, profiles)
    assert festoon > 0.0
    assert np.isclose(festoon, np.deg2rad(9.0), atol=1e-6)


# ------------------------------------------------------------------- wake

def _hero_registry(hero_latitude: float):
    p = PlanetParams(seed=5)
    p.storms.hero_latitude = hero_latitude
    bands = generate_bands(5, p.bands)
    profiles = build_profiles(5, bands, p.bands, p.jets)
    return generate_vortices(5, bands, profiles, p.storms, p.poles)


def test_southern_hero_wake_is_westward_with_equatorward_bias():
    """F06: a -22.5 deg hero must carry a westward wake (down = -1) whose
    across-center is biased toward the equator (positive lat offset)."""
    heroes = _hero_registry(-22.5).heroes()
    assert heroes
    for v in heroes:
        assert v.wake_dir == -1.0
        assert v.wake_lat_off > 0.0
        assert np.isclose(v.wake_lat_off, 0.5 * v.r_core)


def test_northern_hero_wake_mirrors():
    heroes = _hero_registry(22.5).heroes()
    assert heroes
    for v in heroes:
        assert v.wake_dir == -1.0
        assert v.wake_lat_off < 0.0


def test_pack_ssbo_carries_wake_lat_off_in_lane_10():
    reg = _hero_registry(-22.5)
    packed = reg.pack_ssbo()
    for i, v in enumerate(reg.vortices):
        np.testing.assert_allclose(packed[i, 8], v.wake_dir, atol=1e-7)
        np.testing.assert_allclose(packed[i, 10], v.wake_lat_off, atol=1e-7)
