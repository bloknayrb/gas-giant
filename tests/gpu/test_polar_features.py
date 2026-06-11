"""GPU: v1.1 polar features (tint gating, background field, stipple)."""

from __future__ import annotations

import numpy as np
import pytest

from gasgiant.engine import Simulation
from gasgiant.params.model import PlanetParams
from gasgiant.sim.vortices import KIND_POLAR

pytestmark = pytest.mark.gpu


def _quick(seed: int = 27) -> PlanetParams:
    p = PlanetParams(seed=seed)
    p.sim.resolution = 512
    p.sim.dev_steps = 0
    return p


def test_field_density_populates_the_cap():
    from gasgiant.sim.bands import generate_bands
    from gasgiant.sim.profiles import build_profiles
    from gasgiant.sim.vortices import generate_vortices

    p = _quick()
    bands = generate_bands(p.seed, p.bands)
    profiles = build_profiles(p.seed, bands, p.bands, p.jets)
    base = generate_vortices(p.seed, bands, profiles, p.storms, p.poles)
    p.poles.north.field_density = 1.5
    dense = generate_vortices(p.seed, bands, profiles, p.storms, p.poles)
    new = len(dense.vortices) - len(base.vortices)
    assert new > 5
    added = [v for v in dense.vortices if v.kind == KIND_POLAR]
    assert all(abs(v.lat) >= np.deg2rad(70.0) - 1e-6 for v in added if v.lat > 0)


def test_polar_tint_changes_cap_not_equator(gpu):
    p = _quick()
    base = Simulation(p, gpu).render_maps(512)["color"][..., :3]
    p2 = _quick()
    p2.appearance.polar_tint_strength = 0.8
    tinted = Simulation(p2, gpu).render_maps(512)["color"][..., :3]
    h = base.shape[0]
    cap_delta = np.abs(tinted[: h // 8] - base[: h // 8]).mean()
    eq_delta = np.abs(
        tinted[3 * h // 8 : 5 * h // 8] - base[3 * h // 8 : 5 * h // 8]
    ).mean()
    assert cap_delta > 0.01
    assert eq_delta < 1e-5


def test_polar_stipple_brightens_cap_texture(gpu):
    p = _quick()
    base = Simulation(p, gpu).render_maps(512)["color"][..., :3]
    p2 = _quick()
    p2.detail.polar_stipple = 1.5
    speckled = Simulation(p2, gpu).render_maps(512)["color"][..., :3]
    h = base.shape[0]
    # Texture variance rises in the cap; the equator stays identical.
    cap_std_base = base[: h // 8].std()
    cap_std_speck = speckled[: h // 8].std()
    assert cap_std_speck > cap_std_base
    np.testing.assert_array_equal(
        speckled[3 * h // 8 : 5 * h // 8], base[3 * h // 8 : 5 * h // 8]
    )
