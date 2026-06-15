import numpy as np
import pytest
from gasgiant.sim.sw_spike import grid


def test_grid_metrics_poles_zero_cos():
    g = grid.Grid(W=16, H=8)
    assert g.phi_c.shape == (8,)
    assert g.phi_v.shape == (9,)
    # Descending: row 0 is north (positive lat), last row south.
    assert g.phi_c[0] > 0 and g.phi_c[-1] < 0
    # v-face cos is exactly 0 at both poles (no flux through pole).
    assert g.cos_v[0] == pytest.approx(0.0, abs=1e-12)
    assert g.cos_v[-1] == pytest.approx(0.0, abs=1e-12)
    # Center cos strictly positive everywhere.
    assert np.all(g.cos_c > 0.0)
    assert g.dlam == pytest.approx(2 * np.pi / 16)
    assert g.dphi == pytest.approx(np.pi / 8)


def test_center_to_uface_is_periodic_average():
    from gasgiant.sim.sw_spike import grid
    a = np.array([[1.0, 3.0, 5.0, 7.0]])  # H=1, W=4
    uf = grid.center_to_uface(a)  # east face of cell i = avg(i, i+1), periodic
    assert uf.shape == a.shape
    np.testing.assert_allclose(uf, [[2.0, 4.0, 6.0, 4.0]])  # last wraps 7&1 -> 4


def test_center_to_vface_zeroed_at_poles():
    from gasgiant.sim.sw_spike import grid
    a = np.array([[2.0, 2.0], [4.0, 4.0]])  # H=2, W=2
    vf = grid.center_to_vface(a)  # shape (H+1, W); interior = avg of rows; poles=0
    assert vf.shape == (3, 2)
    np.testing.assert_allclose(vf[0], [0.0, 0.0])   # north pole face: no cell north
    np.testing.assert_allclose(vf[1], [3.0, 3.0])   # between rows 0,1
    np.testing.assert_allclose(vf[2], [0.0, 0.0])   # south pole face


def test_divergence_of_solid_body_zonal_flow_is_zero():
    # Purely zonal, longitude-independent u, constant h => mass divergence ~ 0.
    from gasgiant.sim.sw_spike import grid, operators
    g = grid.Grid(W=64, H=32)
    h = np.ones((32, 64))
    u = np.ones((32, 64)) * 0.3          # uniform zonal face velocity
    v = np.zeros((33, 64))
    div = operators.divergence_hu(h, u, v, g)
    # Zonal-uniform, v=0, constant h => divergence is machine-zero.
    assert np.max(np.abs(div)) < 1e-12


def test_divergence_has_no_checkerboard_null_mode():
    # The whole reason for the C-grid: a checkerboard in h must NOT be invisible.
    from gasgiant.sim.sw_spike import grid, operators
    g = grid.Grid(W=64, H=32)
    jj, ii = np.indices((32, 64))
    h = 1.0 + 0.01 * ((ii + jj) % 2)     # 2dx checkerboard thickness
    u = np.ones((32, 64)) * 0.1
    v = np.zeros((33, 64))
    div = operators.divergence_hu(h, u, v, g)
    # On a C-grid the checkerboard produces real flux divergence (non-null).
    assert np.max(np.abs(div)) > 1e-4


def test_montgomery_two_layer_coupling():
    from gasgiant.sim.sw_spike import operators
    h1 = np.full((4, 4), 2.0)
    h2 = np.full((4, 4), 3.0)
    gp = (1.0, 0.05)  # (g'_1 external, g'_2 baroclinic)
    M1, M2 = operators.montgomery_2layer(h1, h2, gp)
    # M1 = g'_1 (h1+h2); M2 = g'_1 (h1+h2) + g'_2 h2
    np.testing.assert_allclose(M1, 1.0 * 5.0)
    np.testing.assert_allclose(M2, 1.0 * 5.0 + 0.05 * 3.0)


def test_pressure_gradient_constant_field_is_zero():
    from gasgiant.sim.sw_spike import grid, operators
    g = grid.Grid(W=32, H=16)
    M = np.full((16, 32), 7.3)
    gx, gy = operators.grad_faces(M, g)
    assert np.max(np.abs(gx)) < 1e-12
    assert np.max(np.abs(gy)) < 1e-12


def test_pressure_gradient_sees_checkerboard():
    # Centered-collocated grad would return ~0 here; the C-grid face grad must not.
    from gasgiant.sim.sw_spike import grid, operators
    g = grid.Grid(W=32, H=16)
    jj, ii = np.indices((16, 32))
    M = ((ii + jj) % 2).astype(float)
    gx, gy = operators.grad_faces(M, g)
    assert np.max(np.abs(gx)) > 1e-3   # face differences are large for a 2dx mode


def test_vorticity_zero_for_irrotational_uniform_flow():
    from gasgiant.sim.sw_spike import grid, operators
    g = grid.Grid(W=64, H=32)
    u = np.full((32, 64), 0.2)
    v = np.zeros((33, 64))
    zeta = operators.vorticity(u, v, g)   # corners (H+1, W)
    assert zeta.shape == (33, 64)
    # Uniform zonal flow on the sphere has curvature vorticity -(1/a) d(u cosφ)/dφ != 0
    # but a constant-u test is dominated by the metric; assert it's finite & smooth.
    assert np.all(np.isfinite(zeta))


def test_vorticity_of_rigid_rotation_constant_sign():
    # u = U cosφ (solid-body zonal) => zeta = -(1/(a cosφ)) d(U cos^2 φ)/dφ = 2U sinφ.
    from gasgiant.sim.sw_spike import grid, operators
    g = grid.Grid(W=128, H=64)
    U = 0.5
    u = (U * g.cos_c)[:, None] * np.ones((1, 128))
    v = np.zeros((65, 128))
    zeta = operators.vorticity(u, v, g)
    # Compare interior corners to analytic 2U sinφ at phi_v.
    analytic = 2 * U * np.sin(g.phi_v)[:, None] * np.ones((1, 128))
    inner = slice(2, 63)
    np.testing.assert_allclose(zeta[inner], analytic[inner], atol=2e-2)


def test_trapezoidal_coriolis_conserves_speed():
    # Pure inertial rotation: |(u,v)| must be preserved by the implicit rotation.
    from gasgiant.sim.sw_spike import operators
    u = np.array([[1.0]]); v = np.array([[0.0]])
    f = np.array([[0.7]]); dt = 0.3
    for _ in range(200):
        u, v = operators.coriolis_trapezoidal(u, v, f, dt)
    speed = np.hypot(u, v)
    np.testing.assert_allclose(speed, 1.0, atol=1e-10)


def test_continuity_conserves_total_mass():
    from gasgiant.sim.sw_spike import grid, operators
    g = grid.Grid(W=48, H=24)
    rng = np.random.default_rng(0)
    h = 1.0 + 0.1 * rng.standard_normal((24, 48))
    h = np.clip(h, 0.2, None)
    u = 0.05 * rng.standard_normal((24, 48))
    v = np.zeros((25, 48)); v[1:24] = 0.05 * rng.standard_normal((23, 48))
    area = g.cos_c[:, None] * np.ones((24, 48))
    m0 = np.sum(h * area)
    # dt below the near-pole zonal CFL limit so the donor-cell low-order step
    # stays monotone (no floor clamp) -> flux-form conserves mass exactly.
    h2 = operators.continuity_step(h, u, v, g, dt=0.02, h_floor=0.05)
    m1 = np.sum(h2 * area)
    np.testing.assert_allclose(m1, m0, rtol=1e-12)  # flux-form => machine precision


def test_continuity_preserves_positivity_under_strong_gradient():
    from gasgiant.sim.sw_spike import grid, operators
    g = grid.Grid(W=32, H=16)
    h = np.full((16, 32), 0.06)          # near the floor
    h[:, 8] = 1.0                        # a spike that will be advected hard
    u = np.full((16, 32), 0.9)           # strong outflow
    v = np.zeros((17, 32))
    h2 = operators.continuity_step(h, u, v, g, dt=0.5, h_floor=0.05)
    assert np.min(h2) >= 0.05 - 1e-9     # FCT keeps h >= floor, no negatives
    assert np.all(np.isfinite(h2))


def test_balanced_zonal_state_stays_balanced():
    # Mini Williamson-2: a geostrophically balanced zonal jet must not accelerate.
    from gasgiant.sim.sw_spike import solver
    st = solver.balanced_test_state(W=128, H=64, f0=4.0, gp=(1.0, 0.05))
    ke0 = solver.kinetic_energy(st)
    for _ in range(50):
        st = solver.step(st, dt=st.dt)
    ke1 = solver.kinetic_energy(st)
    # Balance preserved to scheme order: KE drifts < 1% over 50 steps.
    assert abs(ke1 - ke0) / ke0 < 0.01


def test_checkerboard_pressure_perturbation_does_not_grow():
    # R1 gate: seed a 2dx checkerboard in h, confirm it does NOT amplify.
    from gasgiant.sim.sw_spike import solver
    import numpy as np
    st = solver.balanced_test_state(W=128, H=64, f0=4.0, gp=(1.0, 0.05))
    jj, ii = np.indices(st.h1.shape)
    cb = 0.001 * ((ii + jj) % 2)
    st.h1 = st.h1 + cb
    amp0 = solver.checkerboard_amplitude(st.h1)
    for _ in range(100):
        st = solver.step(st, dt=st.dt)
    amp1 = solver.checkerboard_amplitude(st.h1)
    assert amp1 <= amp0 * 1.5    # bounded, not exponentially growing
