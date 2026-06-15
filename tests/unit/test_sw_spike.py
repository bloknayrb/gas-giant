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
