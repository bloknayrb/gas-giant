import numpy as np


def test_montgomery_2layer_matches_spike():
    from gasgiant.sim.shallow_water_ref import montgomery_2layer
    from gasgiant.sim.sw_spike.operators import montgomery_2layer as spike_mont
    rng = np.random.default_rng(0)
    h1 = 5.0 + rng.random((16, 32)); h2 = 3.0 + rng.random((16, 32))
    M1, M2 = montgomery_2layer(h1, h2, 9.8, 0.3)
    sM1, sM2 = spike_mont(h1, h2, (9.8, 0.3))
    assert np.allclose(M1, sM1, atol=0) and np.allclose(M2, sM2, atol=0)


def test_montgomery_reduces_to_single_layer():
    """With h2=0, M1 = gp1*h1 (the single-layer reduced-gravity pressure)."""
    from gasgiant.sim.shallow_water_ref import montgomery_2layer
    h1 = np.full((8, 8), 4.0); h2 = np.zeros((8, 8))
    M1, _ = montgomery_2layer(h1, h2, 9.8, 0.3)
    assert np.allclose(M1, 9.8 * h1)


def test_momentum_step_M_reduces_to_m1():
    from gasgiant.sim.shallow_water_ref import Grid, momentum_step, momentum_step_M
    g = Grid(W=32, H=16, a=6.4e6)
    rng = np.random.default_rng(1)
    h = 1000 + rng.random((16, 32)); u = rng.random((16, 32)); v = rng.random((17, 32)); v[0]=v[16]=0
    gp = 9.8; omega = 7.292e-5; dt = 50.0
    u_a, v_a = momentum_step(h, u, v, gp, omega, g, dt)
    u_b, v_b = momentum_step_M(h, u, v, gp * h, omega, g, dt)
    assert np.array_equal(u_a, u_b) and np.array_equal(v_a, v_b)


def test_momentum_step_M_decoupled_matches_spike():
    """Validate the NON-reduction path (M != gp*h) against the validated M0 spike's
    _layer_momentum on the LOWER layer (spike a=1, f0 == 2*omega)."""
    from gasgiant.sim.shallow_water_ref import Grid, montgomery_2layer, momentum_step_M
    from gasgiant.sim.sw_spike.grid import Grid as SGrid
    from gasgiant.sim.sw_spike.solver import _layer_momentum
    W, H = 32, 16; omega = 0.5; dt = 30.0
    rng = np.random.default_rng(2)
    h1 = 5 + rng.random((H, W)); h2 = 3 + rng.random((H, W))
    u2 = rng.random((H, W)); v2 = rng.random((H+1, W)); v2[0]=v2[H]=0
    _, M2 = montgomery_2layer(h1, h2, 9.8, 0.3)
    gp = Grid(W=W, H=H, a=1.0); sg = SGrid(W, H)
    u_a, v_a = momentum_step_M(h2, u2, v2, M2, omega, gp, dt)
    u_b, v_b = _layer_momentum(h2, u2, v2, M2, 2*omega, sg, dt)
    assert np.allclose(u_a, u_b, atol=1e-12) and np.allclose(v_a, v_b, atol=1e-12)


def test_step_2layer_resting_stable():
    from gasgiant.sim.shallow_water_ref import Grid, Sw2State, step_2layer, layer_mass
    g = Grid(W=32, H=16, a=6.4e6)
    st = Sw2State(g=g, omega=7.292e-5, gp1=9.8, gp2=0.3,
                  h1=np.full((16,32),1000.0), u1=np.zeros((16,32)), v1=np.zeros((17,32)),
                  h2=np.full((16,32),500.0),  u2=np.zeros((16,32)), v2=np.zeros((17,32)),
                  dt=20.0, h_floor=1.0)
    m1_0, m2_0 = layer_mass(st)
    for _ in range(5):
        st = step_2layer(st)
    assert np.isfinite(st.h1).all() and st.h1.min() > 0 and st.h2.min() > 0
    m1_1, m2_1 = layer_mass(st)
    assert abs(m1_1 - m1_0)/m1_0 < 1e-10 and abs(m2_1 - m2_0)/m2_0 < 1e-10
