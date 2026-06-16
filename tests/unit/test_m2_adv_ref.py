import numpy as np
from gasgiant.sim.shallow_water_ref import Grid, departure_points

def test_departure_solid_body_zonal():
    g = Grid(W=64, H=32, a=6.4e6)
    j0 = 8
    cosphi = g.cos_c[j0]
    u_val = 30.0
    dt = 600.0
    u = np.full((g.H, g.W), u_val)
    v = np.zeros((g.H + 1, g.W))
    i_dep, j_dep = departure_points(u, v, dt, g, n_iter=2)
    C = u_val * dt / (g.a * cosphi * g.dlam)
    assert np.allclose(i_dep[j0], np.arange(g.W) - C, atol=1e-9)
    assert np.allclose(j_dep[j0], j0 + 0.5, atol=1e-9)

def test_departure_a_scaling_invariant():
    for a in (1.0, 6.4e6):
        g = Grid(W=48, H=24, a=a)
        u = np.full((g.H, g.W), 20.0); v = np.zeros((g.H + 1, g.W))
        i_dep, _ = departure_points(u, v, 300.0, g, n_iter=2)
        C = 20.0 * 300.0 / (g.a * g.cos_c[10] * g.dlam)
        assert np.allclose(i_dep[10], np.arange(g.W) - C, atol=1e-9)

def test_departure_meridional_shift():
    g = Grid(W=32, H=32, a=6.4e6)
    u = np.zeros((g.H, g.W)); v = np.full((g.H + 1, g.W), 10.0); v[0] = 0.0; v[-1] = 0.0
    _, j_dep = departure_points(u, v, 300.0, g, n_iter=2)
    assert np.all(j_dep[5:25] > (np.arange(5, 25)[:, None] + 0.5) - 1e-9)


def test_ppm_remap_conserves_and_preserves_uniform():
    from gasgiant.sim.shallow_water_ref import ppm_remap_1d_periodic
    n = 64
    m = np.ones(n)
    edges = np.arange(n + 1, dtype=float) - 0.37
    out = ppm_remap_1d_periodic(m, edges)
    assert abs(out.sum() - m.sum()) < 1e-12
    assert np.allclose(out, 1.0, atol=1e-12)

def test_ppm_remap_no_new_extrema():
    from gasgiant.sim.shallow_water_ref import ppm_remap_1d_periodic
    m = 1.0 + 0.5 * np.sin(np.linspace(0, 4 * np.pi, 96))
    edges = np.arange(97, dtype=float) - 0.6
    out = ppm_remap_1d_periodic(m, edges)
    assert out.min() >= m.min() - 1e-9
    assert out.max() <= m.max() + 1e-9
    assert abs(out.sum() - m.sum()) < 1e-10

def test_ppm_remap_integer_shift_is_roll():
    from gasgiant.sim.shallow_water_ref import ppm_remap_1d_periodic
    m = np.arange(32, dtype=float) ** 1.0 + 1.0
    edges = np.arange(33, dtype=float) - 3.0
    out = ppm_remap_1d_periodic(m, edges)
    assert np.allclose(out, np.roll(m, 3), atol=1e-9)


def test_slice_advance_conserves_mass():
    from gasgiant.sim.shallow_water_ref import Grid, slice_remap_advance
    g = Grid(W=64, H=32, a=6.4e6)
    rng = np.random.default_rng(1)
    h = 1000.0 + 50.0 * rng.standard_normal((g.H, g.W))
    u = 40.0 * np.ones((g.H, g.W)); v = np.zeros((g.H + 1, g.W))
    dt = 1800.0
    h2 = slice_remap_advance(h, u, v, dt, g)
    m0 = float(np.sum(h * g.cos_c[:, None])); m1 = float(np.sum(h2 * g.cos_c[:, None]))
    assert abs(m1 - m0) / abs(m0) < 1e-12

def test_slice_advance_a_scaling():
    from gasgiant.sim.shallow_water_ref import Grid, slice_remap_advance
    for a in (1.0, 6.4e6):
        g = Grid(W=48, H=24, a=a)
        h = 800.0 + np.zeros((g.H, g.W)); h[10, 12] = 900.0
        u = np.full((g.H, g.W), 25.0 * (a / 6.4e6)); v = np.zeros((g.H + 1, g.W))
        h2 = slice_remap_advance(h, u, v, 900.0, g)
        m0 = np.sum(h * g.cos_c[:, None]); m1 = np.sum(h2 * g.cos_c[:, None])
        assert abs(m1 - m0) / m0 < 1e-12

def test_slice_advance_meridional_wall_conserves():
    from gasgiant.sim.shallow_water_ref import Grid, slice_remap_advance
    g = Grid(W=16, H=40, a=6.4e6)
    h = 500.0 + np.zeros((g.H, g.W)); h[20] = 600.0
    u = np.zeros((g.H, g.W)); v = np.full((g.H + 1, g.W), 8.0); v[0] = 0.0; v[-1] = 0.0
    h2 = slice_remap_advance(h, u, v, 1200.0, g)
    assert abs(np.sum(h2 * g.cos_c[:, None]) - np.sum(h * g.cos_c[:, None])) / np.sum(h * g.cos_c[:, None]) < 1e-12

def test_slice_advance_strong_shear_conserves_and_stays_positive():
    """REGRESSION (FATAL class): a sheared row whose cross-cell Courant gradient > 1
    makes departure points cross. Without the span-preserving edge monotonization
    this silently double-counts -> ~1.2% mass leak. Uniform-velocity tests cannot
    catch it; this MASS assertion does (it fires at ~1.2e-2 without the fix).

    NOTE: we do NOT assert h2.max() <= h.max(). The flow here is strongly
    *convergent* (du/dlam large), and conservative continuity Dh/Dt = -h*div(u)
    REQUIRES h to pile up where the flow converges -- a new, larger maximum is
    correct physics, not an overshoot. The maximum principle only holds for
    non-divergent flow. We assert positivity (no spurious negative mass) + exact
    mass instead."""
    from gasgiant.sim.shallow_water_ref import Grid, slice_remap_advance
    g = Grid(W=64, H=8, a=6.4e6)
    lam = np.arange(g.W) * g.dlam
    u = (150.0 * (1.0 + np.tanh(8.0 * np.sin(lam))))[None, :] * np.ones((g.H, 1))
    v = np.zeros((g.H + 1, g.W))
    h = 1000.0 + 100.0 * np.sin(4.0 * lam)[None, :] * np.ones((g.H, 1))
    h2 = slice_remap_advance(h, u, v, 6000.0, g)
    m0 = np.sum(h * g.cos_c[:, None]); m1 = np.sum(h2 * g.cos_c[:, None])
    assert abs(m1 - m0) / abs(m0) < 1e-10, "shear-crossing mass leak (edge monotonization missing)"
    assert h2.min() >= -1e-9, "PPM produced spurious negative h under shear"
    assert np.isfinite(h2).all()


def test_sl_momentum_advects_bump_zonally():
    from gasgiant.sim.shallow_water_ref import Grid, sl_advect_velocity
    g = Grid(W=128, H=4, a=6.4e6)
    j0 = 2; C = 3.0
    cosphi = g.cos_c[j0]
    u_adv = C * g.a * cosphi * g.dlam / 600.0
    u = np.full((g.H, g.W), u_adv)
    v = np.zeros((g.H + 1, g.W))
    q = np.zeros((g.H, g.W)); q[j0, 40:48] = 1.0
    q_adv = sl_advect_velocity(q, u, v, 600.0, g, kind="u")
    assert np.allclose(q_adv[j0], np.roll(q[j0], 3), atol=1e-9)


def test_sl_momentum_predictor_resting_layer_is_pressure_only():
    from gasgiant.sim.shallow_water_ref import Grid, sl_momentum_predictor
    g = Grid(W=16, H=8, a=6.4e6)
    h = np.full((g.H, g.W), 1000.0); u = np.zeros((g.H, g.W)); v = np.zeros((g.H + 1, g.W))
    us, vs = sl_momentum_predictor(h, u, v, 9.8, g, 600.0, 0.5)
    assert np.allclose(us, 0.0, atol=1e-12) and np.allclose(vs, 0.0, atol=1e-12)
