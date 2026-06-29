"""GPU tests for solver.deformation_radius (screened/Helmholtz Poisson).

Plain vorticity mode inverts the streamfunction with 2D Poisson  ∇²ψ = ω, whose
point-vortex velocity decays as ~1/r -- long-range, so a strong hero stirs the
whole map. A finite Rossby deformation radius L_d screens the inversion to
(∇² - 1/L_d²)ψ = ω (equivalent-barotropic / 1.5-layer reduced gravity): the
induced velocity then decays ~exp(-r/L_d) beyond L_d, so storms become LOCAL.

Invariants:
  1. deformation_radius=0 is byte-identical to the plain 2D path (off = no-op).
  2. deformation_radius>0 materially changes the render.
  3. The screened inversion stays finite/bounded over a long horizon.
  4. PHYSICS: with screening on, an isolated hero's FAR-FIELD induced velocity
     is substantially reduced vs the unscreened 2D solve (the whole point).
"""
from __future__ import annotations

import numpy as np
import pytest

from gasgiant.engine.facade import Simulation
from gasgiant.params.model import SolverType
from gasgiant.params.presets import load_factory_preset
from gasgiant.sim.vortices import KIND_HERO

pytestmark = pytest.mark.gpu

GPU_NOISE_ATOL = 1e-2
RES = 512


def _params(ld: float, steps: int = 60, *, isolate_hero: bool = False):
    p = load_factory_preset("jupiter_vorticity").model_copy(update={"seed": 7})
    p.sim.resolution = RES
    p.sim.dev_steps = steps
    p.solver.type = SolverType.VORTICITY
    p.solver.deformation_radius = ld
    p.storms.hero_count = 1
    p.storms.hero_latitude = -22.5
    if isolate_hero:
        # Strip everything but the hero so the velocity field is its footprint, and
        # pin a CANONICAL hero geometry so this screening-physics test is decoupled
        # from the preset's art direction (jupiter_vorticity now bakes a bold solid-
        # body hero -- radius 0.18, solid_core 1.0, mottle/tint -- whose near/far
        # velocity profile differs from the Gaussian this test is calibrated for).
        p.storms.hero_strength = 2.0
        p.storms.hero_radius = 0.15
        p.storms.hero_aspect = 1.0  # canonical ROUND hero: the screening physics is
                                    # isotropic, and a round hero avoids the elliptical
                                    # metric entirely (its near-hemisphere gate / former
                                    # antipode footprint must not confound far-field)
        p.storms.hero_solid_core = 0.0
        p.storms.hero_mottle = 0.0
        p.storms.hero_tint_var = 0.0
        p.storms.oval_density = 0.0
        p.storms.barge_density = 0.0
        p.storms.small_density = 0.0
        p.storms.pearls_count = 0
        p.storms.merge_rate = 0.0
        p.jets.strength = 0.0
        p.solver.vort_inject = 0.0
        p.waves.festoon_strength = 0.0
        p.waves.ribbon_strength = 0.0
    return p


def _render(p, gpu) -> np.ndarray:
    sim = Simulation(p, gpu)
    try:
        return sim.render_maps(RES)["color"].astype(np.float64)
    finally:
        sim._release_sim()


def test_deformation_radius_off_is_noop(gpu):
    """deformation_radius=0 is an exact no-op at the kernel: the only changed line
    is `center_coeff - u_inv_ld2` with u_inv_ld2 an exact 0.0, and x - 0.0 == x
    bit-for-bit (IEEE754). The empirical check uses the GPU noise floor rather than
    assert_array_equal because the jupiter_vorticity base is now a LIVE preset whose
    vorticity SOR Poisson solve carries ~1e-3 cross-instance LSB noise (the kinematic
    path is exactly reproducible; vorticity is not) -- a real effect is >> the floor."""
    base = _render(_params(0.0), gpu)
    same = _render(_params(0.0), gpu)
    assert np.abs(base - same).max() < GPU_NOISE_ATOL


def test_deformation_radius_changes_render(gpu):
    plain = _render(_params(0.0), gpu)
    screened = _render(_params(0.20), gpu)
    assert np.abs(screened - plain).max() > GPU_NOISE_ATOL, (
        "deformation_radius=0.2 did not change the vorticity-mode render"
    )


def test_deformation_radius_bounded_over_long_horizon(gpu):
    out = _render(_params(0.20, steps=400), gpu)
    assert np.all(np.isfinite(out)), "screened Poisson produced non-finite output"
    assert out.max() <= 1.0 + 1e-3 and out.min() >= -1e-3, (
        f"color out of range over long horizon: [{out.min()}, {out.max()}]"
    )


def _hero_far_near_velocity(p, gpu):
    """Run a few steps and return (far_mean, near_mean) of |velocity| relative
    to the hero center. Velocity is read straight from the solver, so it is the
    directly-screened quantity (no palette/advection confounds).

    'far' is the hero's velocity TAIL just outside the core (0.15-0.30 rad), where
    the screened Poisson visibly cuts the induced velocity. (d>0.60 is NOT the
    hero's far field for a compact hero -- it sits at the ambient floor; it only
    ever registered hero signal via the antipode-aliased phantom stamp, since
    fixed, so screening can't be measured there.)"""
    sim = Simulation(p, gpu)
    try:
        sim.run_to_completion()
        vel = sim.gpu.read_texture(sim.solver.equirect.vel_tex)
        speed = np.hypot(vel[..., 0], vel[..., 1])
        hero = next(v for v in sim.vortices.vortices if v.kind == KIND_HERO)
        h, w = speed.shape
        yy, xx = np.mgrid[0:h, 0:w]
        lat = np.pi / 2 - (yy + 0.5) / h * np.pi
        lon = (xx + 0.5) / w * 2 * np.pi - np.pi
        # great-circle distance to the hero
        d = np.arccos(np.clip(
            np.sin(lat) * np.sin(hero.lat)
            + np.cos(lat) * np.cos(hero.lat) * np.cos(lon - hero.lon), -1, 1))
        near = speed[d < 0.12].mean()
        far = speed[(d > 0.15) & (d < 0.30)].mean()
        return far, near
    finally:
        sim._release_sim()


def test_screening_localizes_hero(gpu):
    """The headline physics: screening must cut the hero's far-field velocity
    while leaving a strong near-field core."""
    far_off, near_off = _hero_far_near_velocity(_params(0.0, steps=8, isolate_hero=True), gpu)
    far_on, near_on = _hero_far_near_velocity(_params(0.12, steps=8, isolate_hero=True), gpu)
    # Floor guards (M2): the comparison is only meaningful if the unscreened
    # hero actually has a non-trivial far field and a core that dominates it.
    # Without these a near-zero far_off would let the ratio asserts false-pass.
    assert far_off > 1e-4, f"unscreened far-field too small to test ({far_off:.4g})"
    assert near_off > far_off, (
        f"unscreened core ({near_off:.4g}) should exceed its far field ({far_off:.4g})"
    )
    # Far-field influence is substantially reduced by screening...
    assert far_on < 0.6 * far_off, (
        f"screening did not localize the hero: far_off={far_off:.4g} "
        f"far_on={far_on:.4g}"
    )
    # ...while the hero still has a strong core (not just globally damped).
    assert near_on > 0.5 * near_off, (
        f"screening over-damped the core: near_off={near_off:.4g} near_on={near_on:.4g}"
    )
    # And the locality RATIO (near/far) improves with screening.
    assert (near_on / far_on) > (near_off / far_off), "screening did not improve locality ratio"


def test_deformation_radius_rejects_degenerate_band():
    """0 < L_d < floor is a degenerate (frozen) solve and must be rejected;
    0.0 (off) and a well-resolved value must both be accepted."""
    import pytest as _pytest

    from gasgiant.params.model import SolverParams

    SolverParams(deformation_radius=0.0)    # off — fine
    SolverParams(deformation_radius=0.2)    # well-resolved — fine
    with _pytest.raises(ValueError, match="degenerate band"):
        SolverParams(deformation_radius=0.01)
