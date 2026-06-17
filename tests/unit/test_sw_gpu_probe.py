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
    from gasgiant.sim.sw_spike import grid, operators

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
    import numpy as np

    from gasgiant.sim.sw_gpu_probe import solver
    from gasgiant.sim.sw_spike import grid, operators
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
    import numpy as np

    from gasgiant.sim.sw_gpu_probe import solver
    from gasgiant.sim.sw_spike import grid, operators
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
    import numpy as np

    from gasgiant.sim.sw_gpu_probe import solver
    from gasgiant.sim.sw_spike import grid
    W, H = 128, 64
    U = 0.5
    g = grid.Grid(W, H)
    u = (U * g.cos_c)[:, None] * np.ones((1, W), np.float32)
    v = np.zeros((H + 1, W), np.float32)
    zg = solver.run_vorticity(gpu, u.astype(np.float32), v)
    analytic = 2 * U * np.sin(g.phi_v)[:, None] * np.ones((1, W))
    np.testing.assert_allclose(zg[2:63], analytic[2:63], atol=2e-2)  # interior corners


def test_swp_continuity_matches_ref(gpu):
    import numpy as np

    from gasgiant.sim.sw_gpu_probe import solver
    from gasgiant.sim.sw_spike import grid, operators
    rng=np.random.default_rng(2); W,H=64,32
    h=np.clip(1.0+0.1*rng.standard_normal((H,W)),0.2,None).astype(np.float32)
    u=(0.05*rng.standard_normal((H,W))).astype(np.float32)
    v=np.zeros((H+1,W),np.float32); v[1:H]=0.05*rng.standard_normal((H-1,W))
    g=grid.Grid(W,H)
    cpu=operators.continuity_step(h.astype(np.float64),u.astype(np.float64),v.astype(np.float64),g,dt=0.02,h_floor=0.05)
    gpu_out=solver.run_continuity(gpu,h,u,v,dt=0.02,h_floor=0.05)
    np.testing.assert_allclose(gpu_out, cpu, atol=2e-5)


def test_swp_continuity_conserves_mass(gpu):
    import numpy as np

    from gasgiant.sim.sw_gpu_probe import solver
    from gasgiant.sim.sw_spike import grid
    rng=np.random.default_rng(3); W,H=64,32
    h=np.clip(1.0+0.1*rng.standard_normal((H,W)),0.2,None).astype(np.float32)
    u=(0.03*rng.standard_normal((H,W))).astype(np.float32); v=np.zeros((H+1,W),np.float32)
    g=grid.Grid(W,H)
    out=solver.run_continuity(gpu,h,u,v,dt=0.01,h_floor=0.05)
    # cast to f64 BEFORE the cos-area sum (measure physical conservation, not f32 summation order)
    area=g.cos_c[:,None].astype(np.float64)
    m0=np.sum(h.astype(np.float64)*area); m1=np.sum(out.astype(np.float64)*area)
    np.testing.assert_allclose(m1, m0, rtol=2e-6)


def test_swp_momentum_matches_ref(gpu):
    import numpy as np

    from gasgiant.sim.sw_gpu_probe import solver
    from gasgiant.sim.sw_spike import grid, operators
    from gasgiant.sim.sw_spike import solver as cpu
    rng=np.random.default_rng(9); W,H=64,32
    h1=(5.0+0.3*rng.standard_normal((H,W))).astype(np.float32)
    h2=(3.0+0.3*rng.standard_normal((H,W))).astype(np.float32)
    u=(0.1*rng.standard_normal((H,W))).astype(np.float32)
    v=np.zeros((H+1,W),np.float32); v[1:H]=0.1*rng.standard_normal((H-1,W))
    gp=(1.0,0.05); f0=4.0; dt=0.01; g=grid.Grid(W,H)
    M1,_=operators.montgomery_2layer(h1.astype(np.float64),h2.astype(np.float64),gp)
    un_c,vn_c = cpu._layer_momentum(h1.astype(np.float64),u.astype(np.float64),v.astype(np.float64),M1,f0,g,dt)
    M1_f=(gp[0]*(h1+h2)).astype(np.float32)
    un_g,vn_g = solver.run_momentum(gpu, M1_f, u, v, f0=f0, dt=dt)
    np.testing.assert_allclose(un_g, un_c, atol=2e-5)
    np.testing.assert_allclose(vn_g[1:H], vn_c[1:H], atol=2e-5)


def test_swp_forcing_matches_ref(gpu):
    import numpy as np

    from gasgiant.sim.sw_gpu_probe import solver
    from gasgiant.sim.sw_spike import grid
    from gasgiant.sim.sw_spike import solver as cpu
    rng=np.random.default_rng(13); W,H=64,32; g=grid.Grid(W,H)
    fields={k:(0.5*rng.standard_normal((H,W))).astype(np.float32) for k in ["u1","u2","h1","h2","h_eq1","h_eq2"]}
    v1=np.zeros((H+1,W),np.float32); v1[1:H]=0.3*rng.standard_normal((H-1,W))
    v2=np.zeros((H+1,W),np.float32); v2[1:H]=0.3*rng.standard_normal((H-1,W))
    fields["h1"]+=5; fields["h2"]+=3
    params=dict(tau_rad=300.0, tau_drag=1000.0, nu4=0.05, h_floor=0.05)
    # Build a CPU SwState replica and call _apply_forcing on f64 copies
    st=cpu.SwState(g=g, f0=4.0, gp=(1.0,0.05),
        h1=fields["h1"].astype(np.float64).copy(), u1=fields["u1"].astype(np.float64).copy(), v1=v1.astype(np.float64).copy(),
        h2=fields["h2"].astype(np.float64).copy(), u2=fields["u2"].astype(np.float64).copy(), v2=v2.astype(np.float64).copy(),
        dt=0.001, h_floor=params["h_floor"], nu4=params["nu4"], tau_rad=params["tau_rad"], tau_drag=params["tau_drag"],
        h_eq1=fields["h_eq1"].astype(np.float64).copy(), h_eq2=fields["h_eq2"].astype(np.float64).copy())
    cpu._apply_forcing(st)
    out=solver.run_forcing(gpu, fields, v1, v2, params, f0=4.0)  # returns dict of post-forcing fields
    for k,exp in [("h1",st.h1),("h2",st.h2),("u1",st.u1),("u2",st.u2)]:
        np.testing.assert_allclose(out[k], exp, atol=2e-5)
    np.testing.assert_allclose(out["v1"][1:H], st.v1[1:H], atol=2e-5)
    np.testing.assert_allclose(out["v2"][1:H], st.v2[1:H], atol=2e-5)


def test_swp_step_matches_ref_n_steps(gpu):
    import numpy as np

    from gasgiant.sim.sw_gpu_probe import solver as gsolver
    from gasgiant.sim.sw_spike import init
    from gasgiant.sim.sw_spike import solver as cpu
    W, H = 96, 48
    st_cpu = init.emergent_init(W=W, H=H, f0=4.0, gp=(1.0, 0.05), n_bands=10, band_contrast=0.4)
    # Build the GPU solver from the SAME initial state (copy CPU init fields + params + dt)
    sg = gsolver.SwpSolver.from_cpu_state(gpu, st_cpu)
    for _ in range(30):
        st_cpu = cpu.step(st_cpu, dt=st_cpu.dt)
        sg.step()
    h1g = sg.download("h1")
    assert np.all(np.isfinite(h1g))
    assert np.max(np.abs(h1g - st_cpu.h1)) < 5e-4   # f32 GPU vs f64 CPU drift over 30 steps


def test_swp_continuity_conserves_mass_strong_gradient(gpu):
    """Under strong gradients the h_floor clamp is non-conservative (floor lifts cells),
    so GPU and CPU will both deviate from m0 at roughly the same rtol (~6e-4 for this seed).
    The key assertion is that GPU matches CPU cell-by-cell: same limiter, same conservation error."""
    import numpy as np

    from gasgiant.sim.sw_gpu_probe import solver
    from gasgiant.sim.sw_spike import grid, operators
    rng=np.random.default_rng(11); W,H=64,32
    h=np.clip(0.3+2.0*rng.random((H,W)),0.1,None).astype(np.float32)   # large thickness contrast
    u=(0.4*rng.standard_normal((H,W))).astype(np.float32)
    v=np.zeros((H+1,W),np.float32); v[1:H]=0.4*rng.standard_normal((H-1,W))
    g=grid.Grid(W,H); area=g.cos_c[:,None].astype(np.float64)
    out=solver.run_continuity(gpu,h,u,v,dt=0.01,h_floor=0.05)
    cpu=operators.continuity_step(h.astype(np.float64),u.astype(np.float64),v.astype(np.float64),g,dt=0.01,h_floor=0.05)
    # GPU must match CPU cell-by-cell (corrected limiter matches reference)
    np.testing.assert_allclose(out, cpu, atol=2e-5)
    # Both have the same (floor-driven) mass error; verify GPU mass error is no worse than CPU's
    m_gpu=np.sum(out.astype(np.float64)*area)
    m_cpu=np.sum(cpu.astype(np.float64)*area)
    # GPU and CPU mass integrals should agree to f32 summation precision
    np.testing.assert_allclose(m_gpu, m_cpu, rtol=1e-5)


# ── M0.5 Task 9: physical-time forcing rescale tests ─────────────────────────

def test_swp_forcing_dt_scale_unity_matches_existing(gpu):
    """With forcing_dt_scale=1.0 the N-step GPU run still matches the CPU reference.

    The physical-time rescale must be a no-op at dt=dt_ref (forcing_dt_scale=1.0),
    so existing 30-step accuracy is preserved.
    """
    import numpy as np

    from gasgiant.sim.sw_gpu_probe import solver as gsolver
    from gasgiant.sim.sw_spike import init
    from gasgiant.sim.sw_spike import solver as cpu
    W, H = 96, 48
    st_cpu = init.emergent_init(W=W, H=H, f0=4.0, gp=(1.0, 0.05), n_bands=10, band_contrast=0.4)
    # Build GPU solver with forcing_dt_scale=1.0 (explicit; must be identical to default)
    sg = gsolver.SwpSolver.from_cpu_state(gpu, st_cpu, forcing_dt_scale=1.0)
    for _ in range(30):
        st_cpu = cpu.step(st_cpu, dt=st_cpu.dt)
        sg.step()
    h1g = sg.download("h1")
    assert np.all(np.isfinite(h1g))
    assert np.max(np.abs(h1g - st_cpu.h1)) < 5e-4


def test_swp_reproduces_m0_eddy_curve(gpu):
    """GPU eddy_vorticity_std at 192×96 with forcing_dt_scale=1.0 tracks CPU within 25%.

    Runs ~3000 steps to see eddy growth, sampling at 3 checkpoints.
    Marked slow — bounded step count keeps CI runtime reasonable.
    """
    import pytest
    pytest.importorskip("moderngl")  # skip if no GPU
    import numpy as np

    from gasgiant.sim.sw_gpu_probe import solver as gsolver
    from gasgiant.sim.sw_spike import init
    from gasgiant.sim.sw_spike import solver as cpu

    W, H = 192, 96
    CHECK_AT = [1000, 2000, 3000]

    st_cpu = init.emergent_init(W=W, H=H, f0=4.0, gp=(1.0, 0.05), n_bands=10, band_contrast=0.4)
    sg = gsolver.SwpSolver.from_cpu_state(gpu, st_cpu, forcing_dt_scale=1.0)

    step_idx = 0
    for ck in CHECK_AT:
        n = ck - step_idx
        for _ in range(n):
            st_cpu = cpu.step(st_cpu, dt=st_cpu.dt)
            sg.step()
        step_idx = ck
        evs_cpu = cpu.eddy_vorticity_std(st_cpu)
        evs_gpu = sg.eddy_vorticity_std()
        assert np.isfinite(evs_gpu), f"GPU eddy std not finite at step {ck}"
        # Within 25% of each other (both > 0 after initial transient)
        if evs_cpu > 1e-6:
            ratio = abs(evs_gpu - evs_cpu) / evs_cpu
            assert ratio < 0.25, (
                f"step={ck}: GPU eddy_std={evs_gpu:.4f} CPU={evs_cpu:.4f} "
                f"ratio={ratio:.3f} > 0.25"
            )


def test_swp_highres_smoke(gpu):
    """512×256 smoke test with physical-time rescale: fields stay finite and eddies rise.

    Runs ~3000 steps with forcing_dt_scale=dt_512/dt_ref so the physical forcing
    timescale matches the 192×96 reference.  Only asserts finite + monotone eddy growth
    (not the full eddy regime — see swp_spinup.py for the full run).
    """
    import pytest
    pytest.importorskip("moderngl")
    import numpy as np

    from gasgiant.sim.sw_gpu_probe import solver as gsolver
    from gasgiant.sim.sw_spike import init
    from gasgiant.sim.sw_spike.grid import Grid

    W_ref, H_ref = 192, 96
    W_hi,  H_hi  = 512, 256

    # Compute dt_ref from the 192×96 emergent_init
    st_ref = init.emergent_init(W=W_ref, H=H_ref, f0=4.0, gp=(1.0, 0.05),
                                 n_bands=10, band_contrast=0.4)
    dt_ref = st_ref.dt

    # Compute dt for 512×256 using the same formula
    g_hi   = Grid(W_hi, H_hi)
    h_mean = 5.0
    c_gw   = np.sqrt(1.0 * h_mean)
    dx_min = min(g_hi.cos_c.min() * g_hi.dlam, g_hi.dphi)
    dt_hi  = 0.3 * dx_min / c_gw
    dt_scale = dt_hi / dt_ref

    # Init at high res and build GPU solver
    st_hi = init.emergent_init(W=W_hi, H=H_hi, f0=4.0, gp=(1.0, 0.05),
                                n_bands=10, band_contrast=0.4)
    sg = gsolver.SwpSolver.from_cpu_state(gpu, st_hi, forcing_dt_scale=dt_scale)

    STEPS = 3000
    CHECK_EVERY = 1000
    evs_prev = sg.eddy_vorticity_std()
    evs_history = [evs_prev]
    for ck in range(CHECK_EVERY, STEPS + 1, CHECK_EVERY):
        for _ in range(CHECK_EVERY):
            sg.step()
        h1 = sg.download("h1")
        assert np.all(np.isfinite(h1)), f"h1 non-finite at step {ck}"
        evs = sg.eddy_vorticity_std()
        assert np.isfinite(evs), f"eddy_vorticity_std non-finite at step {ck}"
        evs_history.append(evs)

    # Eddies should be growing (final value > initial)
    assert evs_history[-1] > evs_history[0], (
        f"eddies not growing: history={evs_history}"
    )
