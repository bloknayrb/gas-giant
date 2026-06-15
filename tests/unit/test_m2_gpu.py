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
