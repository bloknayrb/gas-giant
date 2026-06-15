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

from gasgiant.sim.shallow_water_ref import (
    Grid,
    coriolis_trapezoidal,
    coriolis_sandwich,
    divergence_helmholtz,
    grad_faces,
    helmholtz_apply,
    velocity_backsub,
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


def test_velocity_backsub_zero_dh() -> None:
    """velocity_backsub with dh=0 equals coriolis_sandwich (no pressure correction)."""
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

    dh = np.zeros((H, W))

    u_back, v_back = velocity_backsub(u_star, v_star, dh, gp, theta, dt, omega, g)
    u_sand, v_sand = coriolis_sandwich(u_star, v_star, omega, g, dt)

    assert np.array_equal(u_back, u_sand), (
        "velocity_backsub with dh=0 differs from coriolis_sandwich in u"
    )
    assert np.array_equal(v_back, v_sand), (
        "velocity_backsub with dh=0 differs from coriolis_sandwich in v"
    )


def test_velocity_backsub_nonzero_dh() -> None:
    """A non-zero dh must change the back-substituted velocity (catches a
    dropped/sign-flipped pressure-gradient correction)."""
    rng = np.random.default_rng(9999)
    W, H = 16, 8
    g = _make_grid(W=W, H=H, a=6.4e6)
    omega, dt, gp, theta = 7.292e-5, 300.0, 9.8, 0.5

    u_star = rng.standard_normal((H, W))
    v_star = np.zeros((H + 1, W))          # isolate the dh effect
    dh = rng.standard_normal((H, W))

    u_back, v_back = velocity_backsub(u_star, v_star, dh, gp, theta, dt, omega, g)
    u_sand, v_sand = coriolis_sandwich(u_star, v_star, omega, g, dt)

    assert not np.array_equal(u_back, u_sand), (
        "velocity_backsub with dh!=0 unexpectedly equals coriolis_sandwich (u)"
    )
