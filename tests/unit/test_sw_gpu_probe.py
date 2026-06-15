"""Tests for the M0.5 GPU 2-layer shallow-water probe."""
import numpy as np


def test_swp_state_roundtrip(gpu):
    from gasgiant.sim.sw_gpu_probe import solver

    h = np.random.default_rng(0).random((32, 64)).astype(np.float32)
    st = solver.SwpState.create(gpu, W=64, H=32)
    st.upload("h1", h)
    np.testing.assert_allclose(st.download("h1"), h, atol=0)  # exact f4 round-trip
