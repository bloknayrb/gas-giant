import numpy as np
import pytest
from gasgiant.sim import shallow_water_ref as ref


def test_sw_gpu_state_roundtrip(gpu):
    from gasgiant.sim import sw_gpu
    h = np.random.default_rng(0).random((32, 64)).astype(np.float32)
    st = sw_gpu.SwGpuState.create(gpu, W=64, H=32, a=1.0, gp=1.0, omega=2.0)
    st.upload_h(h)
    np.testing.assert_allclose(st.download_h(), h, atol=0)  # exact f4 round-trip


def test_ref_williamson2_stays_balanced():
    from gasgiant.sim import shallow_water_ref as ref
    st = ref.williamson2_state(W=128, H=64, a=1.0, omega=2.0, u0=0.2, gp=1.0, h0=5.0)
    m0 = ref.total_mass(st); e0 = ref.total_energy(st)
    for _ in range(80):
        st = ref.step(st)
    assert np.all(np.isfinite(st.h))
    assert ref.velocity_l2_drift(st) < 1e-2
    np.testing.assert_allclose(ref.total_mass(st), m0, rtol=1e-11)


def test_ref_williamson2_balanced_at_a2():
    # VM3 integrated a!=1: a missing `a` in continuity or the mass area-weight is
    # invisible at a=1.0 but breaks balance/conservation at a=2.0.
    from gasgiant.sim import shallow_water_ref as ref
    st = ref.williamson2_state(W=128, H=64, a=2.0, omega=2.0, u0=0.2, gp=1.0, h0=5.0)
    m0 = ref.total_mass(st)
    for _ in range(80):
        st = ref.step(st)
    assert np.all(np.isfinite(st.h))
    assert ref.velocity_l2_drift(st) < 1e-2
    np.testing.assert_allclose(ref.total_mass(st), m0, rtol=1e-11)


def test_ref_total_mass_radius_scaling():
    # a^2 area weight: total_mass at a=2 is 4x a=1 for the same h field.
    from gasgiant.sim import shallow_water_ref as ref
    st1 = ref.williamson2_state(W=64, H=32, a=1.0, omega=2.0, u0=0.2, gp=1.0, h0=5.0)
    st2 = ref.williamson2_state(W=64, H=32, a=2.0, omega=2.0, u0=0.2, gp=1.0, h0=5.0)
    # same h0/profile shape; compare mass of the SAME h on both grids:
    import numpy as np
    st2b = ref.williamson2_state(W=64, H=32, a=2.0, omega=2.0, u0=0.2, gp=1.0, h0=5.0)
    st2b.h = st1.h.copy()
    np.testing.assert_allclose(ref.total_mass(st2b), 4.0 * ref.total_mass(st1), rtol=1e-12)


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


# ---------------------------------------------------------------------------
# Task 4: GPU divergence tests
# ---------------------------------------------------------------------------

def _div_inputs(W, H, seed):
    rng = np.random.default_rng(seed)
    h = (1.0 + 0.2 * rng.standard_normal((H, W))).astype(np.float32)
    u = (0.1 * rng.standard_normal((H, W))).astype(np.float32)
    v = np.zeros((H + 1, W), np.float32); v[1:H] = 0.1 * rng.standard_normal((H - 1, W))
    return h, u, v


def test_gpu_divergence_matches_ref(gpu):
    from gasgiant.sim import sw_gpu, shallow_water_ref as ref
    for W, H in [(64, 32), (96, 48)]:   # incl. NON-power-of-2 W (wrapX bug surfaces here)
        h, u, v = _div_inputs(W, H, 1)
        g = ref.Grid(W, H, a=1.0)
        cpu = ref.divergence_hu(h.astype(np.float64), u.astype(np.float64), v.astype(np.float64), g)
        got = sw_gpu.run_divergence(gpu, h, u, v, a=1.0)
        cos_c = g.cos_c[:, None]
        # PRE-division: compare cos_c*div (removes f32 polar 1/cos amplification).
        np.testing.assert_allclose(cos_c * got, cos_c * cpu, atol=2e-5)


def test_gpu_divergence_radius_scaling(gpu):
    # INDEPENDENT a-scaling: GPU div at a=2 == a=1 output * 1/2 (single 1/a factor).
    from gasgiant.sim import sw_gpu
    h, u, v = _div_inputs(64, 32, 7)
    d1 = sw_gpu.run_divergence(gpu, h, u, v, a=1.0)
    d2 = sw_gpu.run_divergence(gpu, h, u, v, a=2.0)
    np.testing.assert_allclose(d2, 0.5 * d1, rtol=1e-5)
