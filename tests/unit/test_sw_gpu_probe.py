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


def test_swp_grad_montgomery_matches_ref(gpu):
    from gasgiant.sim.sw_gpu_probe import solver
    from gasgiant.sim.sw_spike import operators, grid
    import numpy as np
    rng = np.random.default_rng(5)
    W, H = 64, 32
    h1 = (5.0 + 0.3 * rng.standard_normal((H, W))).astype(np.float32)
    h2 = (3.0 + 0.3 * rng.standard_normal((H, W))).astype(np.float32)
    gp = (1.0, 0.05)
    g = grid.Grid(W, H)
    M1c, M2c = operators.montgomery_2layer(h1.astype(np.float64), h2.astype(np.float64), gp)
    gx1c, gy1c = operators.grad_faces(M1c, g)
    gx2c, gy2c = operators.grad_faces(M2c, g)
    out = solver.run_grad_montgomery(gpu, h1, h2, gp)  # dict: M1,M2 (H,W); gx1,gx2 (H,W); gy1,gy2 (H+1,W)
    cos_c = g.cos_c[:, None]
    # M values: no division -> flat 2e-5
    np.testing.assert_allclose(out["M1"], M1c, atol=2e-5)
    np.testing.assert_allclose(out["M2"], M2c, atol=2e-5)
    # face gradients: compare PRE-division (multiply zonal by cos_c*dlam, meridional is /dphi only so already O(1))
    np.testing.assert_allclose((cos_c * g.dlam) * out["gx1"], (cos_c * g.dlam) * gx1c, atol=2e-5)
    np.testing.assert_allclose((cos_c * g.dlam) * out["gx2"], (cos_c * g.dlam) * gx2c, atol=2e-5)
    np.testing.assert_allclose(out["gy1"][1:H], gy1c[1:H], atol=2e-5)  # gy has no 1/cos, only /dphi
    np.testing.assert_allclose(out["gy2"][1:H], gy2c[1:H], atol=2e-5)


def test_swp_vorticity_matches_ref(gpu):
    from gasgiant.sim.sw_gpu_probe import solver
    from gasgiant.sim.sw_spike import operators, grid
    import numpy as np
    rng = np.random.default_rng(7)
    W, H = 64, 32
    u = (0.2 * rng.standard_normal((H, W))).astype(np.float32)
    v = np.zeros((H + 1, W), np.float32)
    v[1:H] = 0.2 * rng.standard_normal((H - 1, W))
    g = grid.Grid(W, H)
    zc = operators.vorticity(u.astype(np.float64), v.astype(np.float64), g)  # (H+1,W)
    zg = solver.run_vorticity(gpu, u, v)                                      # (H+1,W)
    # zeta ~ 1/cos_v amplified at poles -> compare PRE-division cos_v*zeta (O(1)).
    cos_v = g.cos_v[:, None]
    np.testing.assert_allclose(cos_v * zg, cos_v * zc, atol=2e-5)


def test_swp_vorticity_rigid_rotation(gpu):
    from gasgiant.sim.sw_gpu_probe import solver
    from gasgiant.sim.sw_spike import grid
    import numpy as np
    W, H = 128, 64
    U = 0.5
    g = grid.Grid(W, H)
    u = (U * g.cos_c)[:, None] * np.ones((1, W), np.float32)
    v = np.zeros((H + 1, W), np.float32)
    zg = solver.run_vorticity(gpu, u.astype(np.float32), v)
    analytic = 2 * U * np.sin(g.phi_v)[:, None] * np.ones((1, W))
    np.testing.assert_allclose(zg[2:63], analytic[2:63], atol=2e-2)  # interior corners
