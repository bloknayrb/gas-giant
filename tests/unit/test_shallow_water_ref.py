"""Unit tests for M2-T1 adjoint Helmholtz operator pair.

Tests:
- test_helmholtz_adjoint: exact discrete adjoint identity <grad h, U>_faces == -<h, div(U,g)>_centers
  for both a=1.0 and a=6.4e6, to rtol/atol 1e-12.
- test_helmholtz_conserves: mass conservation of (L - I), SPD sanity, symmetry of helmholtz_apply.
"""
from __future__ import annotations

import numpy as np
import pytest

from gasgiant.sim.shallow_water_ref import (
    Grid,
    divergence_helmholtz,
    grad_faces,
    helmholtz_apply,
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
        rtol=1e-10, atol=1e-10,
        err_msg=f"helmholtz_apply not symmetric: <x,Ly>={xy:.15g}, <y,Lx>={yx:.15g}",
    )

    # --- 3. SPD: <x, Lx> > 0 ---
    xLx = _inner_centers(x, Lx, g)
    assert xLx > 0.0, f"helmholtz_apply is not positive definite: <x,Lx>={xLx}"
