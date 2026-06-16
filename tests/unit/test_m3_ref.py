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
