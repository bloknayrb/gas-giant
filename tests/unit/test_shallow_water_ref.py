"""Unit tests for M2-T1 adjoint Helmholtz operator pair and M2-T2 Coriolis sandwich.

Tests:
- test_helmholtz_adjoint: exact discrete adjoint identity <grad h, U>_faces == -<h, div(U,g)>_centers
  for both a=1.0 and a=6.4e6, to rtol/atol 1e-12.
- test_helmholtz_conserves: mass conservation of (L - I), SPD sanity, symmetry of helmholtz_apply.
- test_coriolis_sandwich_matches_momentum: byte-identical to momentum_step's inline Coriolis sequence.
- test_velocity_backsub_zero_dh: with dh=0, backsub equals coriolis_sandwich.
"""
from __future__ import annotations

import numpy as np
import pytest

import gasgiant.sim.shallow_water_ref as ref
from gasgiant.sim.shallow_water_ref import (
    Grid,
    coriolis_sandwich,
    coriolis_trapezoidal,
    divergence_helmholtz,
    grad_faces,
    helmholtz_apply,
    helmholtz_residual_per_lat,
    helmholtz_rhs,
    helmholtz_solve_exact,
    helmholtz_sor,
    picard_contraction_factor,
    reference_depth,
    step,
    step_semi_implicit,
    velocity_backsub,
    velocity_l2_drift,
    williamson2_state,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_grid(W: int = 16, H: int = 8, a: float = 1.0) -> Grid:
    return Grid(W=W, H=H, a=a)


def _inner_centers(p: np.ndarray, q: np.ndarray, g: Grid) -> float:
    """cos-weighted L2 inner product on cell centers (H, W)."""
    return float(np.sum(p * q * g.cos_c[:, None]))


def _inner_faces(gx1: np.ndarray, gy1: np.ndarray,
                 gx2: np.ndarray, gy2: np.ndarray,
                 g: Grid) -> float:
    """cos-weighted L2 inner product on face pair (u-faces + v-faces)."""
    ip_u = float(np.sum(gx1 * gx2 * g.cos_c[:, None]))
    ip_v = float(np.sum(gy1 * gy2 * g.cos_v[:, None]))
    return ip_u + ip_v


# ---------------------------------------------------------------------------
# test_helmholtz_adjoint
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("a", [1.0, 6.4e6])
def test_helmholtz_adjoint(a: float) -> None:
    """<grad_faces(h), U>_faces == -<h, divergence_helmholtz(Ux, Uy, ones, g)>_centers

    Holds to rtol/atol 1e-12, for both a=1.0 and a=6.4e6.
    The identity must be exact (up to floating point) because it is derived
    analytically from summation-by-parts.
    """
    rng = np.random.default_rng(42)
    W, H = 16, 8
    g = _make_grid(W=W, H=H, a=a)

    # Random center field h
    h = rng.standard_normal((H, W))

    # Random face fields Ux (H,W) and Uy (H+1,W), with pole rows zeroed
    Ux = rng.standard_normal((H, W))
    Uy = rng.standard_normal((H + 1, W))
    Uy[0] = 0.0
    Uy[H] = 0.0

    # H_ref = 1 everywhere (scalar ones at centers)
    H_ref_lat = np.ones(H)

    # LHS: <grad_faces(h), U>_faces
    gx, gy = grad_faces(h, g)
    lhs = _inner_faces(gx, gy, Ux, Uy, g)

    # RHS: -<h, divergence_helmholtz(Ux, Uy, ones, g)>_centers
    div_U = divergence_helmholtz(Ux, Uy, H_ref_lat, g)
    rhs = -_inner_centers(h, div_U, g)

    # The identity must hold to machine precision (summed over H*W terms at most 16*8=128)
    np.testing.assert_allclose(
        lhs, rhs,
        rtol=1e-12, atol=1e-12,
        err_msg=f"Adjoint identity failed for a={a}: lhs={lhs}, rhs={rhs}",
    )


# ---------------------------------------------------------------------------
# test_helmholtz_conserves
# ---------------------------------------------------------------------------

def test_helmholtz_conserves() -> None:
    """Three properties of helmholtz_apply(dh, H_ref, gp, theta, dt, g):

    1. Mass conservation: the non-identity part (L - I) integrates to zero.
       Σ cos_c * (helmholtz_apply(x) - x) ≈ 0 to 1e-12.
    2. Symmetry: <x, L y>_c = <y, L x>_c to 1e-10.
    3. SPD: <x, L x>_c > 0 for positive H_ref_lat.
    """
    rng = np.random.default_rng(7)
    W, H = 16, 8
    g = _make_grid(W=W, H=H, a=1.0)

    # Positive H_ref_lat (latitude-dependent, strictly positive)
    H_ref_lat = 1.0 + 0.5 * np.cos(g.phi_c)   # shape (H,), >0 everywhere

    gp = 9.8
    theta = 0.5
    dt = 0.1

    # --- 1. Mass conservation of (L - I) ---
    x = rng.standard_normal((H, W))
    Lx = helmholtz_apply(x, H_ref_lat, gp, theta, dt, g)
    residual = float(np.sum((Lx - x) * g.cos_c[:, None]))
    assert abs(residual) < 1e-12, (
        f"helmholtz_apply non-identity part is not mass-conserving: residual={residual:.3e}"
    )

    # --- 2. Symmetry: <x, Ly> = <y, Lx> ---
    y = rng.standard_normal((H, W))
    Ly = helmholtz_apply(y, H_ref_lat, gp, theta, dt, g)
    xy = _inner_centers(x, Ly, g)
    yx = _inner_centers(y, Lx, g)
    np.testing.assert_allclose(
        xy, yx,
        rtol=1e-12, atol=1e-12,
        err_msg=f"helmholtz_apply not symmetric: <x,Ly>={xy:.15g}, <y,Lx>={yx:.15g}",
    )

    # --- 3. SPD: <x, Lx> > 0 ---
    xLx = _inner_centers(x, Lx, g)
    assert xLx > 0.0, f"helmholtz_apply is not positive definite: <x,Lx>={xLx}"


# ---------------------------------------------------------------------------
# M2-T2: coriolis_sandwich and velocity_backsub
# ---------------------------------------------------------------------------

def test_coriolis_sandwich_matches_momentum() -> None:
    """coriolis_sandwich is byte-identical to the Coriolis sandwich in momentum_step.

    Replicates momentum_step's exact inline Coriolis sequence and asserts
    np.array_equal (not just allclose) against coriolis_sandwich output.
    """
    rng = np.random.default_rng(1234)
    W, H = 16, 8
    g = _make_grid(W=W, H=H, a=6.4e6)
    omega = 7.292e-5
    dt = 300.0

    u_star = rng.standard_normal((H, W))
    v_star = rng.standard_normal((H + 1, W))
    # Zero pole rows as momentum_step produces (v_star[0] and v_star[H] are
    # never updated by the explicit step — they stay zero from v.copy()).
    v_star[0] = 0.0
    v_star[H] = 0.0

    # --- Reference: replicate momentum_step's inline Coriolis sandwich ---
    f_uf = 2.0 * omega * np.sin(g.phi_c)[:, None] * np.ones((1, W))    # (H, W)
    v_star_c = 0.5 * (v_star[0:H] + v_star[1:H + 1])                   # (H, W)
    u_ref, v_c_ref = coriolis_trapezoidal(u_star, v_star_c, f_uf, dt)
    v_ref = np.zeros_like(v_star)
    v_ref[1:H] = 0.5 * (v_c_ref[0:H - 1] + v_c_ref[1:H])

    # --- Function under test ---
    u_got, v_got = coriolis_sandwich(u_star, v_star, omega, g, dt)

    assert np.array_equal(u_got, u_ref), (
        "coriolis_sandwich u output differs byte-for-byte from momentum_step inline sequence"
    )
    assert np.array_equal(v_got, v_ref), (
        "coriolis_sandwich v output differs byte-for-byte from momentum_step inline sequence"
    )


def test_velocity_backsub_uniform_height() -> None:
    """velocity_backsub with a UNIFORM h_impl (zero gradient) equals coriolis_sandwich.

    h_impl is the full implicit height h^{n+1}; a uniform field has zero pressure
    gradient so no correction is applied. (Production passes h_n + dh.)"""
    rng = np.random.default_rng(5678)
    W, H = 16, 8
    g = _make_grid(W=W, H=H, a=6.4e6)
    omega = 7.292e-5
    dt = 300.0
    gp = 9.8
    theta = 0.5

    u_star = rng.standard_normal((H, W))
    v_star = rng.standard_normal((H + 1, W))
    v_star[0] = 0.0
    v_star[H] = 0.0

    h_impl = np.full((H, W), 5.0)   # uniform -> zero gradient

    u_back, v_back = velocity_backsub(u_star, v_star, h_impl, gp, theta, dt, omega, g)
    u_sand, v_sand = coriolis_sandwich(u_star, v_star, omega, g, dt)

    assert np.array_equal(u_back, u_sand), (
        "velocity_backsub with uniform h_impl differs from coriolis_sandwich in u"
    )
    assert np.array_equal(v_back, v_sand), (
        "velocity_backsub with uniform h_impl differs from coriolis_sandwich in v"
    )


def test_velocity_backsub_nonzero_dh() -> None:
    """A non-uniform h_impl must change the back-substituted velocity (catches a
    dropped/sign-flipped pressure-gradient correction). Mirrors the production
    convention: h_impl = h_n + dh."""
    rng = np.random.default_rng(9999)
    W, H = 16, 8
    g = _make_grid(W=W, H=H, a=6.4e6)
    omega, dt, gp, theta = 7.292e-5, 300.0, 9.8, 0.5

    u_star = rng.standard_normal((H, W))
    v_star = np.zeros((H + 1, W))          # isolate the pressure effect
    h_impl = 5.0 + rng.standard_normal((H, W))   # full height h_n + dh

    u_back, v_back = velocity_backsub(u_star, v_star, h_impl, gp, theta, dt, omega, g)
    u_sand, v_sand = coriolis_sandwich(u_star, v_star, omega, g, dt)

    assert not np.array_equal(u_back, u_sand), (
        "velocity_backsub with dh!=0 unexpectedly equals coriolis_sandwich (u)"
    )


# ---------------------------------------------------------------------------
# M2-T3: helmholtz_rhs and picard_contraction_factor
# ---------------------------------------------------------------------------

def test_rhs_zero_velocity() -> None:
    """helmholtz_rhs with zero velocities and dh_prev=0 reduces to the background term.

    New contract (increment theta-scheme): with u_n=v_n=0, u_star=v_star=0,
    dh_prev=0 the reference-divergence term (refdiv) and the deferred-Coriolis
    term (defer) both vanish (coriolis_sandwich(0,0)=(0,0), grad_faces(0)=(0,0)).
    Only the implicit BACKGROUND pressure term survives:

        rhs = theta*dt * div_H( H_ref * C( theta*dt*gp*grad(h^n) ) ).
    """
    rng = np.random.default_rng(314)
    W, H = 16, 8
    g = _make_grid(W=W, H=H, a=1.0)

    h_n = rng.standard_normal((H, W))
    u_n = np.zeros((H, W))
    v_n = np.zeros((H + 1, W))
    u_star = np.zeros((H, W))
    v_star = np.zeros((H + 1, W))
    dh_prev = np.zeros((H, W))

    H_ref_lat = 1.0 + 0.3 * np.cos(g.phi_c)   # positive, latitude-varying
    gp = 9.8
    omega = 7.292e-5
    theta = 0.5
    dt = 300.0

    result = helmholtz_rhs(h_n, u_n, v_n, u_star, v_star, dh_prev,
                           H_ref_lat, gp, omega, theta, dt, g)

    # Expected: only the background implicit-pressure term.
    tdt = theta * dt
    gx_n, gy_n = grad_faces(h_n, g)
    u_bg = tdt * gp * gx_n
    v_bg = tdt * gp * gy_n
    u_bgc, v_bgc = coriolis_sandwich(u_bg, v_bg, omega, g, dt)
    expected = tdt * divergence_helmholtz(u_bgc, v_bgc, H_ref_lat, g)

    np.testing.assert_allclose(
        result, expected, rtol=1e-13, atol=1e-15,
        err_msg="helmholtz_rhs with zero velocities/dh_prev must equal the background term",
    )

    # Also confirm the increment vanishes when there is no background gradient
    # either (flat h^n): then the whole RHS is exactly zero.
    h_flat = np.full((H, W), 3.0)
    result_flat = helmholtz_rhs(h_flat, u_n, v_n, u_star, v_star, dh_prev,
                                H_ref_lat, gp, omega, theta, dt, g)
    assert np.max(np.abs(result_flat)) < 1e-13, (
        "helmholtz_rhs with zero velocities AND flat h^n must be zero, got "
        f"max {np.max(np.abs(result_flat)):.3e}"
    )


def test_picard_rho_below_half() -> None:
    """picard_contraction_factor returns rho = 2*alpha/(1+alpha^2) for the worst latitude.

    - Benign config (small omega*dt): rho < 0.5.
    - Stiff config (omega=1.0, dt=2.0, alpha~1): rho >= 0.5.
    - Returned value matches closed-form for both configs.
    """
    W, H = 16, 8

    # --- Benign: small alpha ---
    g_benign = _make_grid(W=W, H=H, a=1.0)
    omega_b = 7.292e-5
    dt_b = 300.0
    rho_b = picard_contraction_factor(omega_b, theta=0.5, dt=dt_b, g=g_benign)

    # Verify closed form
    f_max_b = 2.0 * omega_b * np.max(np.abs(np.sin(g_benign.phi_c)))
    alpha_b = 0.5 * f_max_b * dt_b
    expected_b = 2.0 * alpha_b / (1.0 + alpha_b ** 2)
    np.testing.assert_allclose(rho_b, expected_b, rtol=1e-12,
                                err_msg="picard_contraction_factor benign: formula mismatch")
    assert rho_b < 0.5, (
        f"Benign config should have rho < 0.5, got {rho_b:.6f}"
    )

    # --- Stiff: large alpha ---
    g_stiff = _make_grid(W=W, H=H, a=1.0)
    omega_s = 1.0
    dt_s = 2.0
    rho_s = picard_contraction_factor(omega_s, theta=0.5, dt=dt_s, g=g_stiff)

    f_max_s = 2.0 * omega_s * np.max(np.abs(np.sin(g_stiff.phi_c)))
    alpha_s = 0.5 * f_max_s * dt_s
    expected_s = 2.0 * alpha_s / (1.0 + alpha_s ** 2)
    np.testing.assert_allclose(rho_s, expected_s, rtol=1e-12,
                                err_msg="picard_contraction_factor stiff: formula mismatch")
    # Anchor the parametric intent: rho = 2a/(1+a^2) >= 0.5 iff alpha >= tan(pi/12) ≈ 0.268.
    # Asserting the threshold here keeps the stiff case from silently going vacuous if
    # someone later lowers omega_s/dt_s below the rho>=0.5 boundary.
    assert alpha_s >= 0.268, (
        f"Stiff config must have alpha >= 0.268 to guarantee rho >= 0.5, got alpha={alpha_s:.3f}"
    )
    assert rho_s >= 0.5, (
        f"Stiff config (omega={omega_s}, dt={dt_s}, alpha={alpha_s:.3f}) should have "
        f"rho >= 0.5, got {rho_s:.6f}"
    )


def test_picard_factor_matches_formula() -> None:
    """Independent recompute of rho from alpha at the worst-case latitude.

    Uses a planetary-scale grid and verifies the binding latitude is
    the one closest to the pole (largest |sin phi|).
    """
    W, H = 32, 16
    g = _make_grid(W=W, H=H, a=6.4e6)
    omega = 7.292e-5
    theta = 0.6
    dt = 600.0

    rho = picard_contraction_factor(omega, theta, dt, g)

    # Independent formula
    sin_phi_max = np.max(np.abs(np.sin(g.phi_c)))
    f_max = 2.0 * omega * sin_phi_max
    alpha = 0.5 * f_max * dt
    expected = 2.0 * alpha / (1.0 + alpha ** 2)

    np.testing.assert_allclose(
        rho, expected,
        rtol=1e-13, atol=1e-15,
        err_msg=(
            f"picard_contraction_factor mismatch: got {rho:.15g}, "
            f"expected {expected:.15g} (alpha={alpha:.6f})"
        ),
    )


# ---------------------------------------------------------------------------
# M2-T4: semi-implicit step, SOR Helmholtz, end-to-end arbiters
# ---------------------------------------------------------------------------

# Chosen solver parameters (documented in step_semi_implicit):
#   poisson_iters = 200 SOR sweeps per Helmholtz solve
#   sor_omega     = 1.7 over-relaxation
# At the realistic W2 dt (polar CFL) the Helmholtz operator is well-conditioned
# (rho ~ 2.6e-3); 200 sweeps drive the per-latitude residual (incl. poles) to
# machine zero.  The polar-residual test below uses a Coriolis-active config
# (rho ~ 0.20, non-trivial gravity-wave alpha) to exercise the lag.
_POISSON_ITERS = 200
_SOR_OMEGA = 1.7


def test_sor_converges_to_exact() -> None:
    """helmholtz_sor with many sweeps matches helmholtz_solve_exact to rtol 1e-6.

    Proves the SOR fixed point is the true Helmholtz inverse (not just any
    contraction).  Random rhs, small grid W=24, H=12, positive latitude-varying
    H_ref.
    """
    rng = np.random.default_rng(2024)
    W, H = 24, 12
    g = Grid(W=W, H=H, a=1.0)
    H_ref_lat = 1.0 + 0.5 * np.cos(g.phi_c)
    gp, theta, dt = 9.8, 0.5, 0.3
    rhs = rng.standard_normal((H, W))

    dh_exact = helmholtz_solve_exact(rhs, H_ref_lat, gp, theta, dt, g)
    dh_sor = helmholtz_sor(rhs, H_ref_lat, gp, theta, dt, g,
                           n_iters=400, sor_omega=1.8)

    np.testing.assert_allclose(
        dh_sor, dh_exact, rtol=1e-6, atol=1e-10,
        err_msg="SOR fixed point does not match the exact Helmholtz inverse",
    )


def test_w2_geostrophic_stationary() -> None:
    """One semi-implicit step leaves the Williamson-2 state stationary.

    VELOCITY stationarity is the tight arbiter: the corrected theta-scheme keeps
    (u,v) stationary to the explicit step()'s own tolerance (matches to ~5
    significant figures; required <= 2x as a margin).

    HEIGHT stationarity is held to the matched theta-scheme's intrinsic
    O((theta*dt)^2) steady-state imbalance.  Making the FULL pressure gradient
    implicit (the change that removes the gravity-wave CFL — see the gravity-wave
    spike) introduces a small dt^2 pressure/Coriolis-coupling residual at a
    non-flat balanced state.  It is NOT present in the explicit step (which keeps
    pressure fully explicit) and it vanishes as dt -> 0
    (test_semi_implicit_reduces_to_m1_at_small_dt), confirming it is a temporal
    truncation term, not an imbalance bug.  We bound it well below the wave
    amplitude (a few 1e-6 on an O(1) height) and far below any dynamically
    significant drift.
    """
    st = williamson2_state(W=64, H=32, a=1.0, omega=2.0, u0=0.2, gp=1.0, h0=5.0)
    h0 = st.h.copy()

    # Reference: one explicit step.
    st_expl = step(st)
    vel_expl = velocity_l2_drift(st_expl)
    h_expl = float(np.max(np.abs(st_expl.h - h0)))

    # Semi-implicit step.
    st_si = step_semi_implicit(st, poisson_iters=_POISSON_ITERS, sor_omega=_SOR_OMEGA)
    vel_si = velocity_l2_drift(st_si)
    h_si = float(np.max(np.abs(st_si.h - h0)))

    # Velocity: must match the explicit step to its own tolerance.
    assert vel_si <= 2.0 * vel_expl + 1e-9, (
        f"semi-implicit velocity drift {vel_si:.3e} exceeds explicit {vel_expl:.3e}"
    )
    # Height: the matched theta-scheme's intrinsic O((theta*dt)^2) imbalance
    # (verified to scale as dt^2). At the W2 polar-CFL dt this is ~1.5e-6 on an
    # O(1) height; 5e-6 is a ~3x margin. See step_semi_implicit's
    # "NOTE (W2 height drift)" for the derivation. This is the price of making
    # the full pressure implicit (what removes the gravity-wave CFL).
    assert h_si < 5e-6, (
        f"semi-implicit height drift {h_si:.3e} exceeds the theta-scheme dt^2 bound "
        f"(explicit step drift {h_expl:.3e})"
    )


def test_backsub_continuity_consistency() -> None:
    """h_new - h equals the solved increment dh plus the explicit anomaly to atol 2e-5.

    The final height of step_semi_implicit is h + dh + anomaly, where dh is the
    matched theta-centered Helmholtz increment and anomaly is the explicit
    nonlinear/limited transport beyond the linear reference divergence.  This
    test reconstructs dh and anomaly the same way step_semi_implicit does and
    confirms h_new - h reproduces them — i.e. the back-substituted velocity and
    the continuity transport are consistent with the solved increment.

    NOTE (M2-T5): the FCT used in the anomaly is now continuity_step_conservative
    (the mass-conserving, positivity-preserving variant) instead of the M1
    continuity_step, so the anomaly is mass-neutral even at the floor.  This IC
    is well above the floor, so the two FCT variants agree here; the test is
    reconstructed against the variant the step actually uses.
    """
    st = williamson2_state(W=64, H=32, a=1.0, omega=2.0, u0=0.2, gp=1.0, h0=5.0)
    g, gp, omega, dt, theta = st.g, st.gp, st.omega, st.dt, 0.5

    u_star, v_star = ref._semi_implicit_predictor(st.h, st.u, st.v, gp, g, dt, theta)
    H_ref_lat = reference_depth(st.h)

    dh = np.zeros_like(st.h)
    for _ in range(3):
        rhs = helmholtz_rhs(st.h, st.u, st.v, u_star, v_star, dh,
                            H_ref_lat, gp, omega, theta, dt, g)
        dh = helmholtz_sor(rhs, H_ref_lat, gp, theta, dt, g,
                           _POISSON_ITERS, _SOR_OMEGA, dh0=dh)

    u_new, v_new = velocity_backsub(u_star, v_star, st.h + dh, gp, theta, dt, omega, g)
    h_fct = ref.continuity_step_conservative(st.h, u_new, v_new, g, dt, st.h_floor)
    h_linref = st.h - dt * divergence_helmholtz(u_new, v_new, H_ref_lat, g)
    expected = dh + (h_fct - h_linref)

    st_si = step_semi_implicit(st, poisson_iters=_POISSON_ITERS, sor_omega=_SOR_OMEGA)
    actual = st_si.h - st.h

    np.testing.assert_allclose(
        actual, expected, atol=2e-5,
        err_msg="h_new - h does not match the solved increment + anomaly",
    )


def test_semi_implicit_reduces_to_m1_at_small_dt() -> None:
    """At dt well below the explicit CFL, step_semi_implicit ~ step().

    As dt -> 0 the implicit increment dh -> 0 (O(dt^2)) and the semi-implicit
    scheme coincides with the explicit step.
    """
    st = williamson2_state(W=48, H=24, a=1.0, omega=2.0, u0=0.2, gp=1.0, h0=5.0)
    # Shrink dt far below CFL.
    import dataclasses
    st = dataclasses.replace(st, dt=st.dt * 1e-3)

    st_expl = step(st)
    st_si = step_semi_implicit(st, poisson_iters=_POISSON_ITERS, sor_omega=_SOR_OMEGA)

    np.testing.assert_allclose(st_si.u, st_expl.u, atol=1e-9,
                               err_msg="u disagrees with explicit step at small dt")
    np.testing.assert_allclose(st_si.v, st_expl.v, atol=1e-9,
                               err_msg="v disagrees with explicit step at small dt")
    np.testing.assert_allclose(st_si.h, st_expl.h, atol=1e-9,
                               err_msg="h disagrees with explicit step at small dt")


def test_per_lat_residual_polar() -> None:
    """Per-latitude residual (incl. pole rows) below gate at the chosen iters.

    Uses a Coriolis-active config (rho ~ 0.20, non-trivial gravity-wave alpha)
    so the deferred-Coriolis lag and the meridional polar stencil are exercised.
    Gate: max per-latitude L2 residual < 1e-10 at poisson_iters=200.
    """
    W, H = 48, 24
    g = Grid(W=W, H=H, a=1.0)
    gp, omega, theta, dt = 1.0, 2.0, 0.5, 0.05
    H_ref_lat = 5.0 - np.cos(g.phi_c)   # positive, latitude-varying

    # Confirm Coriolis is genuinely active (alpha > 0) for this config.
    assert picard_contraction_factor(omega, theta, dt, g) > 0.05

    rng = np.random.default_rng(11)
    h_n = 5.0 + rng.standard_normal((H, W)) * 0.02
    u_n = rng.standard_normal((H, W)) * 0.2
    v_n = np.zeros((H + 1, W))
    v_n[1:H] = rng.standard_normal((H - 1, W)) * 0.2
    u_star = rng.standard_normal((H, W)) * 0.2
    v_star = np.zeros((H + 1, W))
    v_star[1:H] = rng.standard_normal((H - 1, W)) * 0.2

    dh = np.zeros((H, W))
    rhs = helmholtz_rhs(h_n, u_n, v_n, u_star, v_star, dh,
                        H_ref_lat, gp, omega, theta, dt, g)
    dh = helmholtz_sor(rhs, H_ref_lat, gp, theta, dt, g,
                       _POISSON_ITERS, _SOR_OMEGA)

    resid = helmholtz_residual_per_lat(dh, rhs, H_ref_lat, gp, theta, dt, g)
    assert resid.shape == (H,)
    assert resid.max() < 1e-10, (
        f"polar per-lat residual {resid.max():.3e} exceeds gate at "
        f"{_POISSON_ITERS} iters (pole rows: {resid[0]:.3e}, {resid[-1]:.3e})"
    )
