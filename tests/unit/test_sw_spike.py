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
