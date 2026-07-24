"""Resolution-invariant development scaling (sim/resolution_scaling.py).

Two things are asserted:

1. The pure transforms are structurally identity at ``s == 1`` (the byte-identity
   contract), decay-exact stays bounded in ``[0, 1)`` for ``s < 1`` (where the naive
   linear ``f / s`` would blow past 1 and invert a ``mix()``), and each is monotone.

2. TIMELINE COHERENCE: with the flag on, a *pinned* storm's drift-compensated SEED
   longitude is invariant across resolution. Because ``dt`` in proportion to
   ``1/resolution`` and ``effective_dev_steps = dev_steps * s``, the compensated
   drift ``zonal_rate * dt * eff`` is resolution-independent, so the seed lands at the
   same longitude at 512, 1024 and 2048 -- and develops to the same target. This is the
   guard against the "scale one site, desync the seeded timeline" failure mode.
"""

from __future__ import annotations

import numpy as np
import pytest

from gasgiant.params.model import PlanetParams
from gasgiant.params.presets import load_factory_preset
from gasgiant.sim.bands import generate_bands
from gasgiant.sim.profiles import build_profiles
from gasgiant.sim.resolution_scaling import (
    effective_dev_steps,
    scale_decay_fraction,
    scale_duration,
    scale_factor,
    scale_relax_tau,
    scale_stochastic_amp,
)
from gasgiant.sim.solver import compute_dt
from gasgiant.sim.vortices import generate_vortices, zonal_rate

# -- scale_factor / effective_dev_steps ----------------------------------------

def test_scale_factor_off_is_one():
    p = PlanetParams()
    assert p.sim.resolution_invariant is False
    assert scale_factor(p) == 1.0


def test_scale_factor_on_at_reference_is_one():
    p = PlanetParams()
    p.sim.resolution_invariant = True
    p.sim.reference_resolution = 1024
    p.sim.resolution = 1024
    assert scale_factor(p) == 1.0  # exact, not computed as res/res


def test_scale_factor_on_is_ratio():
    p = PlanetParams()
    p.sim.resolution_invariant = True
    p.sim.reference_resolution = 1024
    p.sim.resolution = 4096
    assert scale_factor(p) == 4.0
    p.sim.resolution = 512
    assert scale_factor(p) == 0.5


def test_effective_dev_steps():
    p = PlanetParams()
    p.sim.dev_steps = 500
    assert effective_dev_steps(p) == 500  # flag off
    p.sim.resolution_invariant = True
    p.sim.reference_resolution = 1024
    p.sim.resolution = 2048
    assert effective_dev_steps(p) == 1000
    p.sim.resolution = 512
    assert effective_dev_steps(p) == 250


# -- pure transforms: identity at s==1 -----------------------------------------

@pytest.mark.parametrize("s", [1.0])
def test_transforms_identity_at_s1(s):
    # Structural (is-the-same-object-value) no-op, not merely arithmetic identity.
    assert scale_duration(500, s) == 500
    assert scale_decay_fraction(0.35, s) == 0.35
    assert scale_relax_tau(350.0, s) == 350.0
    assert scale_stochastic_amp(1.8, s) == 1.8


# -- decay-exact bounded + monotone --------------------------------------------

@pytest.mark.parametrize("s", [0.125, 0.25, 0.5, 2.0, 4.0])
def test_decay_fraction_bounded(s):
    for f in (0.01, 0.1, 0.35, 0.7, 0.99):
        out = scale_decay_fraction(f, s)
        assert 0.0 <= out < 1.0, (f, s, out)


def test_decay_fraction_direction():
    # s < 1 (fewer, bigger steps) => each step must decay MORE to match the run.
    assert scale_decay_fraction(0.35, 0.25) > 0.35
    # s > 1 (more, smaller steps) => each step decays LESS.
    assert scale_decay_fraction(0.35, 2.0) < 0.35


def test_relax_tau_direction():
    # tau is 1/fraction, so it moves opposite the fraction.
    assert scale_relax_tau(350.0, 0.25) < 350.0
    assert scale_relax_tau(350.0, 2.0) > 350.0


def test_stochastic_amp_is_inv_sqrt_s():
    assert scale_stochastic_amp(1.8, 0.25) == pytest.approx(1.8 / 0.5)
    assert scale_stochastic_amp(1.8, 4.0) == pytest.approx(1.8 / 2.0)


def test_duration_rounds_and_scales():
    assert scale_duration(500, 0.25) == 125
    assert scale_duration(500, 4.0) == 2000
    assert scale_duration(3, 0.125) == 0  # rounds to zero for a tiny run


# -- timeline coherence: pinned seed longitude is resolution-invariant ---------

def _reg_at(resolution: int, reference: int, invariant: bool):
    """Build the vortex registry a dev run at ``resolution`` would use, with a
    pinned hero longitude so drift compensation is exercised."""
    p = load_factory_preset("gas_giant_warm")
    p.storms.hero_longitude = 40.0  # degrees: pin it so drift comp runs
    p.sim.dev_steps = 500
    p.sim.reference_resolution = reference
    p.sim.resolution_invariant = invariant
    p.sim.resolution = resolution
    bands = generate_bands(p.seed, p.bands)
    profiles = build_profiles(p.seed, bands, p.bands, p.jets)
    dt = compute_dt(p.sim.resolution, p.sim.dt_scale, profiles.max_speed)
    s = scale_factor(p)
    reg = generate_vortices(
        p.seed, bands, profiles, p.storms, p.poles, dt=dt,
        dev_steps=p.sim.dev_steps, step_scale=s,
    )
    return p, profiles, dt, reg


def _hero_final_lon(profiles, dt, reg, eff):
    """Seed longitude advanced by the closed-form zonal drift over eff steps."""
    hero = reg.heroes()[0]
    rate = float(zonal_rate(profiles, np.array([hero.lat]))[0])
    return float((hero.lon + rate * dt * eff + np.pi) % (2.0 * np.pi) - np.pi)


def test_pinned_hero_seed_longitude_invariant_with_flag():
    # With the flag ON, the compensated SEED longitude is the same at every
    # resolution (dt * eff is resolution-independent).
    _, _, _, reg_512 = _reg_at(512, 1024, invariant=True)
    _, _, _, reg_2048 = _reg_at(2048, 1024, invariant=True)
    lon_512 = reg_512.heroes()[0].lon
    lon_2048 = reg_2048.heroes()[0].lon
    assert lon_512 == pytest.approx(lon_2048, abs=1e-9)


def test_pinned_hero_develops_to_same_longitude_with_flag():
    # Seed + total drift = the pinned target (40 deg) at every resolution.
    target = float(np.deg2rad(40.0))
    for res in (512, 1024, 2048):
        p, profiles, dt, reg = _reg_at(res, 1024, invariant=True)
        eff = effective_dev_steps(p)
        final = _hero_final_lon(profiles, dt, reg, eff)
        assert final == pytest.approx(target, abs=1e-6), res


def test_pinned_hero_seed_longitude_drifts_without_flag():
    # Contrast: with the flag OFF, the seed longitude DOES vary with resolution
    # (dt changes, dev_steps fixed) -- the drift this feature exists to remove.
    _, _, _, reg_512 = _reg_at(512, 1024, invariant=False)
    _, _, _, reg_2048 = _reg_at(2048, 1024, invariant=False)
    assert reg_512.heroes()[0].lon != pytest.approx(reg_2048.heroes()[0].lon, abs=1e-6)
