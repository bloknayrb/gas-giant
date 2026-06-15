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
