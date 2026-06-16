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
