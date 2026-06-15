"""Tests for the M0.5 GPU 2-layer shallow-water probe."""
import numpy as np


def test_swp_state_roundtrip(gpu):
    from gasgiant.sim.sw_gpu_probe import solver

    h = np.random.default_rng(0).random((32, 64)).astype(np.float32)
    st = solver.SwpState.create(gpu, W=64, H=32)
    st.upload("h1", h)
    np.testing.assert_allclose(st.download("h1"), h, atol=0)  # exact f4 round-trip


def test_swp_divergence_matches_ref(gpu):
    from gasgiant.sim.sw_gpu_probe import solver
    from gasgiant.sim.sw_spike import operators, grid

    rng = np.random.default_rng(1)
    W, H = 64, 32
    h = (1.0 + 0.2 * rng.standard_normal((H, W))).astype(np.float32)
    u = (0.1 * rng.standard_normal((H, W))).astype(np.float32)
    v = np.zeros((H + 1, W), np.float32)
    v[1:H] = 0.1 * rng.standard_normal((H - 1, W))
    g = grid.Grid(W, H)
    div_cpu = operators.divergence_hu(
        h.astype(np.float64), u.astype(np.float64), v.astype(np.float64), g
    )
    div_gpu = solver.run_divergence(gpu, h, u, v)
    # Compare the PRE-division flux (cos_c * div) so f32 polar 1/cos amplification cancels.
    cos_c = g.cos_c[:, None]
    np.testing.assert_allclose(cos_c * div_gpu, cos_c * div_cpu, atol=2e-5)
