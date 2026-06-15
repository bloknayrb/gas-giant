"""M2-T7a: GPU Helmholtz operator + red/black SOR kernels, validated per-field
against the CPU ground truth in shallow_water_ref.py.

The CPU functions (helmholtz_apply, helmholtz_sor, helmholtz_residual,
helmholtz_solve_exact) are the GROUND TRUTH.  GPU kernels diff against them
per-field.  A failing diff means a GLSL bug (indexing/sign/metric/pole-guard/
H_ref interp/wrapX) — NEVER loosen atol to hide it.
"""
from __future__ import annotations

import numpy as np
import pytest

from gasgiant.sim import shallow_water_ref as ref


def _rng(seed=0):
    return np.random.default_rng(seed)


def _random_inputs(W, H, seed=0):
    """Random dh (H,W) and strictly-positive H_ref_lat (H,)."""
    rng = _rng(seed)
    dh = rng.standard_normal((H, W)).astype(np.float32)
    H_ref_lat = (0.5 + rng.random(H)).astype(np.float32)  # in [0.5, 1.5), positive
    return dh, H_ref_lat


# --------------------------------------------------------------------------
# helmholtz_apply
# --------------------------------------------------------------------------

def test_gpu_helmholtz_apply_matches_ref(gpu):
    from gasgiant.sim import sw_gpu

    for (W, H) in [(64, 32), (96, 48)]:  # incl. non-pow-2 W=96
        g = ref.Grid(W=W, H=H, a=1.3)
        gp, theta, dt = 0.8, 0.5, 0.7
        dh, H_ref_lat = _random_inputs(W, H, seed=W)

        cpu = ref.helmholtz_apply(dh, H_ref_lat, gp, theta, dt, g)
        out = sw_gpu.run_helmholtz_apply(gpu, dh, H_ref_lat, gp, theta, dt, g.a)

        assert out.shape == (H, W)
        # High-frequency random dh makes L_sym(dh) genuinely O(1e4-7e4): the
        # composed grad->div_H Laplacian carries a 1/cos^2 near-pole metric, so
        # an O(1) checkerboard dh produces huge near-pole values.  The GPU is
        # f32; the CPU ground truth is f64.  The error is therefore RELATIVE to
        # the operator magnitude (max relative diff ~1.5e-4, consistent with f32
        # mantissa ~1.2e-7 amplified by ~6 cancellation-prone ops and the 1/cos^2
        # pole metric), NOT a fixed 2e-5 absolute floor.  rtol=3e-4 with a small
        # atol=2e-5 floor is the principled f32 bound; the stencil itself is
        # exact (see test_gpu_helmholtz_apply_a_scaling for the analytic check).
        np.testing.assert_allclose(out, cpu, rtol=3e-4, atol=2e-5)


def test_gpu_helmholtz_apply_a_scaling(gpu):
    # The (L - I) part scales as 1/a^2.  Compare run_helmholtz_apply at a=2 vs a=1:
    # the non-identity part (out - dh) must scale by 1/4.
    from gasgiant.sim import sw_gpu

    W, H = 64, 32
    gp, theta, dt = 0.8, 0.5, 0.7
    dh, H_ref_lat = _random_inputs(W, H, seed=7)

    out1 = sw_gpu.run_helmholtz_apply(gpu, dh, H_ref_lat, gp, theta, dt, a=1.0)
    out2 = sw_gpu.run_helmholtz_apply(gpu, dh, H_ref_lat, gp, theta, dt, a=2.0)

    nonid1 = out1 - dh
    nonid2 = out2 - dh
    # The non-identity part is O(3e4) (high-freq dh, 1/cos^2 pole metric).  The
    # 1/a^2 analytic scaling holds to f32 relative precision (max rel ~1.6e-5),
    # which is the clean independent confirmation the stencil's metric is exact.
    np.testing.assert_allclose(nonid2, 0.25 * nonid1, rtol=1e-4, atol=2e-5)


# --------------------------------------------------------------------------
# helmholtz_sor
# --------------------------------------------------------------------------

def test_gpu_helmholtz_sor_matches_ref(gpu):
    from gasgiant.sim import sw_gpu

    W, H = 64, 32
    g = ref.Grid(W=W, H=H, a=1.3)
    gp, theta, dt = 0.8, 0.5, 0.7
    rng = _rng(11)
    rhs = rng.standard_normal((H, W)).astype(np.float32)
    H_ref_lat = (0.5 + rng.random(H)).astype(np.float32)

    n_iters, sor_omega = 50, 1.7
    cpu = ref.helmholtz_sor(rhs, H_ref_lat, gp, theta, dt, g, n_iters, sor_omega)
    out = sw_gpu.run_helmholtz_sor(
        gpu, rhs, H_ref_lat, gp, theta, dt, g.a, n_iters, sor_omega
    )

    assert out.shape == (H, W)
    # 50 f32 SOR sweeps accumulate; 5e-5 is the principled bound (2.5x the flat
    # 2e-5 per-op tol, documenting f32 accumulation across 100 colored sweeps).
    np.testing.assert_allclose(out, cpu, atol=5e-5)


def test_gpu_helmholtz_sor_converges_to_exact(gpu):
    from gasgiant.sim import sw_gpu

    W, H = 64, 32
    g = ref.Grid(W=W, H=H, a=1.3)
    gp, theta, dt = 0.8, 0.5, 0.7
    rng = _rng(13)
    rhs = rng.standard_normal((H, W)).astype(np.float32)
    H_ref_lat = (0.5 + rng.random(H)).astype(np.float32)

    exact = ref.helmholtz_solve_exact(rhs, H_ref_lat, gp, theta, dt, g)
    out = sw_gpu.run_helmholtz_sor(
        gpu, rhs, H_ref_lat, gp, theta, dt, g.a, n_iters=400, sor_omega=1.7
    )
    # The GPU SOR fixed point matches the direct sparse solve to ~5e-4 absolute
    # (solution magnitude O(0.3)): the residual is limited by the f32 SOR
    # iteration floor, not the stencil.  This certifies the GPU iteration
    # converges to the true solution of L_sym dh = rhs.
    np.testing.assert_allclose(out, exact, atol=8e-4)


# --------------------------------------------------------------------------
# helmholtz_residual
# --------------------------------------------------------------------------

def test_gpu_helmholtz_residual_matches_ref(gpu):
    from gasgiant.sim import sw_gpu

    W, H = 64, 32
    g = ref.Grid(W=W, H=H, a=1.3)
    gp, theta, dt = 0.8, 0.5, 0.7
    rng = _rng(17)
    dh = rng.standard_normal((H, W)).astype(np.float32)
    rhs = rng.standard_normal((H, W)).astype(np.float32)
    H_ref_lat = (0.5 + rng.random(H)).astype(np.float32)

    cpu = ref.helmholtz_apply(dh, H_ref_lat, gp, theta, dt, g) - rhs
    out = sw_gpu.run_helmholtz_residual(gpu, dh, rhs, H_ref_lat, gp, theta, dt, g.a)

    assert out.shape == (H, W)
    # Same magnitude/precision regime as helmholtz_apply (residual = L_sym(dh) -
    # rhs, dominated by the O(1e4) L_sym term).  rtol=3e-4 f32 bound + 2e-5 atol.
    np.testing.assert_allclose(out, cpu, rtol=3e-4, atol=2e-5)


# --------------------------------------------------------------------------
# Helpers for the M2 SI-step per-field tests
# --------------------------------------------------------------------------

def _random_sw_state(W, H, a, seed=0, scale_uv=0.1):
    """A smooth, O(1) random shallow-water state (h>0, v poles zeroed)."""
    rng = _rng(seed)
    g = ref.Grid(W=W, H=H, a=a)
    # Smooth h around 5.0 with a small latitude+longitude modulation.
    lam = np.linspace(0.0, 2 * np.pi, W, endpoint=False)[None, :]
    phi = g.phi_c[:, None]
    h = (5.0 + 0.3 * np.cos(phi) * np.cos(lam)
         + 0.2 * np.sin(2 * phi)).astype(np.float32)
    u = (scale_uv * (np.cos(phi) + 0.2 * np.sin(lam))
         * np.ones((H, W))).astype(np.float32)
    v = np.zeros((H + 1, W), dtype=np.float32)
    v[1:H] = (scale_uv * 0.3 * np.sin(phi[1:H]) * np.cos(lam)).astype(np.float32)
    return g, h, u, v


# --------------------------------------------------------------------------
# SI predictor
# --------------------------------------------------------------------------

def test_gpu_si_predictor_matches_ref(gpu):
    from gasgiant.sim import sw_gpu

    for (W, H) in [(64, 32), (96, 48)]:
        g, h, u, v = _random_sw_state(W, H, a=1.3, seed=W)
        gp, dt, theta = 0.8, 0.05, 0.5

        cu, cv = ref._semi_implicit_predictor(h, u, v, gp, g, dt, theta)
        gu, gv = sw_gpu.run_si_predictor(gpu, h, u, v, g.a, gp, dt, theta)

        assert gu.shape == (H, W)
        assert gv.shape == (H + 1, W)
        # O(1) velocity fields; predictor is a handful of f32 ops on metric-baked
        # gradients (zeta/grad carry the 1/cos pole metric).  2e-5 absolute.
        np.testing.assert_allclose(gu, cu, atol=2e-5)
        np.testing.assert_allclose(gv, cv, atol=2e-5)


# --------------------------------------------------------------------------
# helmholtz_rhs
# --------------------------------------------------------------------------

def test_gpu_helmholtz_rhs_matches_ref(gpu):
    from gasgiant.sim import sw_gpu

    for (W, H) in [(64, 32), (96, 48)]:
        g, h, u, v = _random_sw_state(W, H, a=1.3, seed=W + 1)
        gp, omega, theta, dt = 0.8, 1.5, 0.5, 0.05
        rng = _rng(W + 2)
        # u_star/v_star: a perturbed copy of u/v as a stand-in predictor output.
        u_star = (u + 0.01 * rng.standard_normal((H, W))).astype(np.float32)
        v_star = v.copy(); v_star[1:H] += 0.01 * rng.standard_normal((H - 1, W)).astype(np.float32)
        dh_prev = (0.02 * rng.standard_normal((H, W))).astype(np.float32)
        H_ref_lat = ref.reference_depth(h)

        cpu = ref.helmholtz_rhs(h, u, v, u_star, v_star, dh_prev,
                                H_ref_lat, gp, omega, theta, dt, g)
        out = sw_gpu.run_helmholtz_rhs(gpu, h, u, v, u_star, v_star, dh_prev,
                                       H_ref_lat, gp, omega, theta, dt, g.a)

        assert out.shape == (H, W)
        # RHS carries a div_H (1/cos^2 near-pole metric) of O(1) sandwiched
        # velocities; values reach O(10) near the poles.  f32 vs f64: a tight
        # absolute floor plus a small principled rtol for the amplified pole rows.
        np.testing.assert_allclose(out, cpu, rtol=3e-4, atol=2e-5)


# --------------------------------------------------------------------------
# velocity_backsub
# --------------------------------------------------------------------------

def test_gpu_velocity_backsub_matches_ref(gpu):
    from gasgiant.sim import sw_gpu

    for (W, H) in [(64, 32), (96, 48)]:
        g, h, u, v = _random_sw_state(W, H, a=1.3, seed=W + 3)
        gp, omega, theta, dt = 0.8, 1.5, 0.5, 0.05
        rng = _rng(W + 4)
        u_star = (u + 0.01 * rng.standard_normal((H, W))).astype(np.float32)
        v_star = v.copy(); v_star[1:H] += 0.01 * rng.standard_normal((H - 1, W)).astype(np.float32)
        dh = (0.02 * rng.standard_normal((H, W))).astype(np.float32)
        h_impl = (h + dh).astype(np.float32)

        cu, cv = ref.velocity_backsub(u_star, v_star, h_impl, gp, theta, dt, omega, g)
        gu, gv = sw_gpu.run_velocity_backsub(gpu, u_star, v_star, h_impl,
                                             gp, theta, dt, omega, g.a)

        assert gu.shape == (H, W)
        assert gv.shape == (H + 1, W)
        # O(1) velocity fields, a grad + Cayley sandwich; 2e-5 absolute.
        np.testing.assert_allclose(gu, cu, atol=2e-5)
        np.testing.assert_allclose(gv, cv, atol=2e-5)


# --------------------------------------------------------------------------
# continuity_step_conservative
# --------------------------------------------------------------------------

def test_gpu_continuity_conservative_matches_ref(gpu):
    from gasgiant.sim import sw_gpu

    for (W, H) in [(64, 32), (96, 48)]:
        g, h, u, v = _random_sw_state(W, H, a=1.3, seed=W + 5, scale_uv=0.3)
        dt, h_floor = 0.02, 0.05

        cpu = ref.continuity_step_conservative(h, u, v, g, dt, h_floor)
        out = sw_gpu.run_continuity_conservative(gpu, h, u, v, g.a, dt, h_floor)

        assert out.shape == (H, W)
        assert np.all(out >= h_floor - 1e-6)
        # h is O(5); the FCT chain is many f32 ops (donor scales, Zalesak caps,
        # final outflux limit), so 2e-5 absolute on the height update.
        np.testing.assert_allclose(out, cpu, atol=2e-5)


# --------------------------------------------------------------------------
# Full semi-implicit step + capstone gravity-wave + determinism
# --------------------------------------------------------------------------

def _w2_params():
    return dict(W=64, H=32, a=1.0, omega=2.0, u0=0.2, gp=1.0, h0=5.0)


def test_gpu_si_step_matches_ref(gpu):
    from gasgiant.sim import sw_gpu

    p = _w2_params()
    theta, picard_iters, poisson_iters, sor_omega = 0.5, 3, 200, 1.7

    # CPU ground-truth state.
    st = ref.williamson2_state(**p)

    # GPU solver in the SI path with the same params.
    sg = sw_gpu.SwGpuSolver.from_williamson2(
        gpu, W=p["W"], H=p["H"], a=p["a"], omega=p["omega"], u0=p["u0"],
        gp=p["gp"], h0=p["h0"], semi_implicit=True, theta=theta,
        sor_omega=sor_omega, helmholtz_iters=poisson_iters,
        picard_iters=picard_iters, dt_multiplier=1.0,
    )

    # 1 step.
    st1 = ref.step_semi_implicit(st, theta=theta, picard_iters=picard_iters,
                                 poisson_iters=poisson_iters, sor_omega=sor_omega)
    sg.step()
    gh, gu, gv = sg.download_state()

    # f32 GPU vs f64 CPU across the full SI pipeline (predictor + 3 Picard
    # solves of 200 f32 SOR sweeps each + backsub + 5-pass FCT).  The dominant
    # error source is the SOR iteration floor (f32) feeding the velocity/anomaly;
    # 5e-4 absolute on O(0.2) velocities / O(5) heights is the principled bound.
    np.testing.assert_allclose(gh, st1.h, atol=5e-4)
    np.testing.assert_allclose(gu, st1.u, atol=5e-4)
    np.testing.assert_allclose(gv, st1.v, atol=5e-4)

    # N-step agreement bound (errors accumulate but stay bounded over 10 steps).
    st_n = st1
    for _ in range(9):
        st_n = ref.step_semi_implicit(st_n, theta=theta, picard_iters=picard_iters,
                                      poisson_iters=poisson_iters, sor_omega=sor_omega)
        sg.step()
    gh, gu, gv = sg.download_state()
    np.testing.assert_allclose(gh, st_n.h, atol=5e-3)
    np.testing.assert_allclose(gu, st_n.u, atol=5e-3)
    np.testing.assert_allclose(gv, st_n.v, atol=5e-3)


def test_gpu_si_gravity_wave_stable(gpu):
    # CAPSTONE: resting layer + small Gaussian bump, omega=0, large dt (N*dt_gw).
    # Removing the gravity-wave CFL on the GPU means this stays BOUNDED and energy
    # NEUTRAL even though dt >> the explicit gravity-wave step.
    from gasgiant.sim import sw_gpu

    W, H, a, gp, h0 = 64, 32, 1.0, 1.0, 5.0
    omega = 0.0
    g = ref.Grid(W=W, H=H, a=a)

    # Resting layer + small Gaussian bump in h.
    lam = np.linspace(0.0, 2 * np.pi, W, endpoint=False)[None, :]
    phi = g.phi_c[:, None]
    bump = 0.1 * np.exp(-((phi) ** 2 + (lam - np.pi) ** 2) / 0.2)
    h = (h0 + bump).astype(np.float32)
    u = np.zeros((H, W), dtype=np.float32)
    v = np.zeros((H + 1, W), dtype=np.float32)

    # Explicit gravity-wave dt and the large SI dt.
    c_gw = np.sqrt(gp * h.max())
    cos_min = max(g.cos_c.min(), 1e-6)
    dx_min = min(cos_min * a * g.dlam, a * g.dphi)
    dt_gw = 0.3 * dx_min / c_gw
    N = 20
    dt_si = N * dt_gw

    sg = sw_gpu.SwGpuSolver(
        gpu, W=W, H=H, a=a, gp=gp, omega=omega, dt=dt_si, h_floor=0.05,
        semi_implicit=True, theta=0.5, sor_omega=1.7,
        helmholtz_iters=200, picard_iters=3, dt_multiplier=1.0,
    )
    sg._tex_h.write(h.tobytes())
    sg._tex_u.write(u.tobytes())
    sg._tex_v.write(v.tobytes())
    sg.u_init = u.copy(); sg.v_init = v.copy()
    sg.H_ref = ref.reference_depth(h)

    e0 = sg.total_energy()
    for _ in range(40):
        sg.step()
    hh, uu, vv = sg.download_state()
    assert np.all(np.isfinite(hh)) and np.all(np.isfinite(uu)) and np.all(np.isfinite(vv))
    e1 = sg.total_energy()
    ratio = e1 / e0
    # BOUNDED + energy NEUTRAL: theta=0.5 Crank-Nicolson is non-dissipative, so
    # at dt = 20*dt_gw the wave neither blows up nor decays away.
    assert 0.9 <= ratio <= 1.1, f"energy ratio {ratio} out of [0.9,1.1]"


def test_gpu_si_deterministic(gpu):
    import hashlib
    from gasgiant.sim import sw_gpu

    p = _w2_params()

    def _run():
        sg = sw_gpu.SwGpuSolver.from_williamson2(
            gpu, W=p["W"], H=p["H"], a=p["a"], omega=p["omega"], u0=p["u0"],
            gp=p["gp"], h0=p["h0"], semi_implicit=True, theta=0.5,
            sor_omega=1.7, helmholtz_iters=200, picard_iters=3,
        )
        for _ in range(5):
            sg.step()
        h, u, v = sg.download_state()
        m = hashlib.sha1()
        for arr in (h, u, v):
            m.update(np.ascontiguousarray(arr, dtype=np.float32).tobytes())
        return m.hexdigest()

    assert _run() == _run()


# --------------------------------------------------------------------------
# T7a deferred: smooth-dh tight apply test (complements the high-freq rtol case)
# --------------------------------------------------------------------------

def test_gpu_helmholtz_apply_smooth_tight(gpu):
    from gasgiant.sim import sw_gpu

    W, H = 64, 32
    g = ref.Grid(W=W, H=H, a=1.3)
    gp, theta, dt = 0.8, 0.5, 0.7
    # SMOOTH (low-wavenumber) dh: no checkerboard, so the composed Laplacian does
    # NOT amplify by the 1/cos^2 pole metric to O(1e4); CPU output is O(10).
    lam = np.linspace(0.0, 2 * np.pi, W, endpoint=False)[None, :]
    phi = g.phi_c[:, None]
    dh = (np.cos(phi) * np.cos(lam) + 0.5 * np.sin(2 * phi)).astype(np.float32)
    H_ref_lat = (0.5 + 0.3 * np.cos(g.phi_c)).astype(np.float32)

    cpu = ref.helmholtz_apply(dh, H_ref_lat, gp, theta, dt, g)
    out = sw_gpu.run_helmholtz_apply(gpu, dh, H_ref_lat, gp, theta, dt, g.a)

    # Output is O(10) and smooth: f32 holds a TIGHT absolute bound here (no
    # checkerboard amplification), complementing T7a's high-freq rtol case.
    assert np.max(np.abs(cpu)) > 1.0  # genuinely O(1-10), not a trivial null field
    np.testing.assert_allclose(out, cpu, atol=2e-5)
