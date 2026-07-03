import numpy as np


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


def test_step_2layer_resting_stable():
    from gasgiant.sim.shallow_water_ref import Grid, Sw2State, layer_mass, step_2layer
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


def test_relaxation_pulls_h_toward_heq():
    from gasgiant.sim.shallow_water_ref import Grid, Sw2State, apply_forcing
    g = Grid(W=16, H=8, a=6.4e6)
    st = Sw2State(g=g, omega=0.0, gp1=9.8, gp2=0.3,
                  h1=np.full((8,16),1000.0), u1=np.zeros((8,16)), v1=np.zeros((9,16)),
                  h2=np.full((8,16),500.0),  u2=np.zeros((8,16)), v2=np.zeros((9,16)),
                  dt=20.0, h_floor=1.0, tau_rad=10.0,
                  h_eq1=np.full((8,16),1100.0), h_eq2=np.full((8,16),500.0))
    apply_forcing(st)
    assert 1000.0 < st.h1.mean() < 1100.0


def test_bottom_drag_only_lower_layer():
    from gasgiant.sim.shallow_water_ref import Grid, Sw2State, apply_forcing
    g = Grid(W=16, H=8, a=6.4e6)
    st = Sw2State(g=g, omega=0.0, gp1=9.8, gp2=0.3,
                  h1=np.full((8,16),1000.0), u1=np.full((8,16),5.0), v1=np.zeros((9,16)),
                  h2=np.full((8,16),500.0),  u2=np.full((8,16),5.0), v2=np.zeros((9,16)),
                  dt=20.0, h_floor=1.0, tau_drag=10.0)
    u1_before = st.u1.copy()
    apply_forcing(st)
    assert np.allclose(st.u1, u1_before)
    assert st.u2.mean() < 5.0


def test_2layer_williamson2_balance_stationary():
    """A 2-layer geostrophically-balanced state stays stationary to scheme tolerance
    -- pins the Montgomery sign/coefficient matrix (design §2.2)."""
    from gasgiant.sim.shallow_water_ref import balanced_2layer_state, step_2layer
    st = balanced_2layer_state(W=64, H=32, a=6.4e6, omega=7.292e-5, gp1=9.8, gp2=0.3, u0=20.0)
    u1_0 = st.u1.copy(); h1_0 = st.h1.copy()
    for _ in range(10):
        st = step_2layer(st)
    assert np.max(np.abs(st.u1 - u1_0)) < 1e-2
    assert np.max(np.abs(st.h1 - h1_0)) / h1_0.mean() < 1e-3


# ---------------------------------------------------------------------------
# M3 Task 7: conservation + budget closure + determinism gates
# ---------------------------------------------------------------------------

def test_mass_conserved_per_layer_unforced():
    """Unforced (tau off, sponge off), per-layer mass conserves to round-off.
    Achievable because step_2layer uses continuity_step_conservative; the polar
    sponge injects mass (relaxes h->h_eq) so it must be off here."""
    from gasgiant.sim.shallow_water_ref import baroclinic_test_state, layer_mass, step_2layer
    st = baroclinic_test_state(W=64, H=32, unstable=False, seed=1)
    st.tau_rad = 0.0; st.tau_drag = 0.0; st.nu4 = 0.0; st.sponge_rate = 0.0
    m0 = layer_mass(st)
    for _ in range(50): st = step_2layer(st)
    m1 = layer_mass(st)
    assert abs(m1[0]-m0[0])/m0[0] < 1e-9 and abs(m1[1]-m0[1])/m0[1] < 1e-9


def test_determinism_2layer():
    """Two identical runs produce byte-identical state (SHA1)."""
    import hashlib

    from gasgiant.sim.shallow_water_ref import baroclinic_test_state, step_2layer
    def run():
        st = baroclinic_test_state(W=48, H=24, unstable=True, seed=3)
        st.sponge_rate = 0.0
        for _ in range(20): st = step_2layer(st)
        m = hashlib.sha1()
        for f in (st.h1, st.u1, st.v1, st.h2, st.u2, st.v2):
            m.update(np.ascontiguousarray(f, dtype=np.float64).tobytes())
        return m.hexdigest()
    assert run() == run()


def test_forcing_params_change_output():
    """v1.6 no-op guard: each forcing param must actually change the evolved state
    (guard against a forcing term silently doing nothing)."""
    import numpy as np

    from gasgiant.sim.shallow_water_ref import baroclinic_test_state, step_2layer
    def evolve(field, **over):
        st = baroclinic_test_state(W=48, H=24, unstable=True, seed=4)
        st.sponge_rate = 0.0
        for k, v in over.items(): setattr(st, k, v)
        for _ in range(15): st = step_2layer(st)
        return getattr(st, field).copy()
    # Bottom drag acts on the LOWER layer (u2/v2), hyperviscosity on u1/u2; each
    # term's effect surfaces directly in the field it drives but only weakly
    # (indirectly, via Montgomery coupling) in the upper-layer h1. Check each
    # guard against the field the term actually drives.
    assert not np.allclose(evolve("u2"), evolve("u2", tau_drag=20.0)), "bottom drag had no effect"
    assert not np.allclose(evolve("u1"), evolve("u1", nu4=0.05)), "hyperviscosity had no effect"


def test_total_energy_2layer_finite_positive():
    """total_energy_2layer is finite and strictly positive on a balanced state."""
    from gasgiant.sim.shallow_water_ref import baroclinic_test_state, total_energy_2layer
    st = baroclinic_test_state(W=48, H=24, unstable=True, seed=5)
    e = total_energy_2layer(st)
    assert np.isfinite(e) and e > 0.0
