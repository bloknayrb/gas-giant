import numpy as np
import pytest
from gasgiant.sim import shallow_water_ref as ref


def test_ref_divergence_solid_body_zero():
    g = ref.Grid(W=64, H=32, a=1.0)
    h = np.ones((32, 64)); u = np.full((32, 64), 0.3); v = np.zeros((33, 64))
    assert np.max(np.abs(ref.divergence_hu(h, u, v, g))) < 1e-12


def test_ref_vorticity_rigid_rotation():
    g = ref.Grid(W=128, H=64, a=1.0)
    U = 0.5
    u = (U * g.cos_c)[:, None] * np.ones((1, 128)); v = np.zeros((65, 128))
    zeta = ref.vorticity(u, v, g)
    analytic = 2 * U * np.sin(g.phi_v)[:, None] * np.ones((1, 128))
    np.testing.assert_allclose(zeta[2:63], analytic[2:63], atol=2e-2)


def test_ref_grad_radius_scaling():
    # Doubling radius a halves the gradient magnitude (metric 1/a).
    M = np.linspace(0, 1, 32)[:, None] * np.ones((1, 16))
    g1 = ref.Grid(W=16, H=32, a=1.0); g2 = ref.Grid(W=16, H=32, a=2.0)
    _, gy1 = ref.grad_faces(M, g1); _, gy2 = ref.grad_faces(M, g2)
    np.testing.assert_allclose(gy2[1:32], 0.5 * gy1[1:32], rtol=1e-12)


def test_ref_divergence_radius_scaling():
    # Independent a-scaling for divergence (1/a prefactor): a=2 == a=1 * 1/2.
    rng = np.random.default_rng(4); W, H = 64, 32
    h = 1.0 + 0.2 * rng.standard_normal((H, W))
    u = 0.1 * rng.standard_normal((H, W))
    v = np.zeros((H + 1, W)); v[1:H] = 0.1 * rng.standard_normal((H - 1, W))
    g1 = ref.Grid(W, H, a=1.0); g2 = ref.Grid(W, H, a=2.0)
    d1 = ref.divergence_hu(h, u, v, g1); d2 = ref.divergence_hu(h, u, v, g2)
    np.testing.assert_allclose(d2, 0.5 * d1, rtol=1e-12)


def test_ref_continuity_radius_scaling():
    # Independent a-scaling for _apply_fluxes (continuity 1/(a cosφ) metric):
    # the per-step thickness tendency (h_new - h) must scale by 1/a.
    from gasgiant.sim import shallow_water_ref as ref
    rng = np.random.default_rng(6); W, H = 64, 32
    h = np.clip(1.0 + 0.1 * rng.standard_normal((H, W)), 0.2, None)
    u = 0.03 * rng.standard_normal((H, W))
    v = np.zeros((H + 1, W)); v[1:H] = 0.03 * rng.standard_normal((H - 1, W))
    g1 = ref.Grid(W, H, a=1.0); g2 = ref.Grid(W, H, a=2.0)
    # Use a dt small enough that the FCT floor never fires (so the metric, not the
    # clamp, governs) — same sub-CFL regime as the conservation tests.
    h1 = ref.continuity_step(h, u, v, g1, dt=0.005, h_floor=0.05)
    h2 = ref.continuity_step(h, u, v, g2, dt=0.005, h_floor=0.05)
    # tendency halves at a=2; compare where the a=1 tendency is non-trivial.
    t1 = h1 - h; t2 = h2 - h
    mask = np.abs(t1) > 1e-6
    np.testing.assert_allclose(t2[mask], 0.5 * t1[mask], rtol=1e-9)
