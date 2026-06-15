"""Canonical numpy reference for vorticity operators (P2, v1.6).

Single source of truth for the spherical vorticity math.  P3's GLSL kernels
mirror these formulas exactly.  Pure numpy — no GPU dependency.

Sign convention (CRITICAL):
    Relative vorticity  ω ≡ ζ  =  +∇²ψ
    (Verified from velocity.comp:  u = −∂ψ/∂φ,  v = (1/cosφ)∂ψ/∂λ
     ⟹  ζ = +∇²ψ.  Poisson equation: ∇²ψ = +ω.)
    Absolute vorticity:  q = ω + f,   f(φ) = f₀·sin φ.

Latitude convention for this module:
    All functions that accept a 1-D ``lat`` array expect ASCENDING order:
    lat[0] ≈ −π/2 (south pole),  lat[-1] ≈ +π/2 (north pole),
    matching the natural numpy linspace convention.  The grid row index ``i``
    therefore increases with latitude, so +∂/∂φ = +∂/∂row.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

# Minimum |cosφ| before clamping — mirrors the `max(cos(ll.y), 0.017)` in
# velocity.comp so the discrete Laplacian here matches the GPU operator exactly.
_COS_FLOOR: float = 0.017


# ---------------------------------------------------------------------------
# 1. Discrete spherical Laplacian
# ---------------------------------------------------------------------------


def laplacian_sphere(
    psi: NDArray[np.floating],
    lat: NDArray[np.floating],
) -> NDArray[np.floating]:
    """Discrete spherical Laplacian of *psi* on a 2-D lat/lon grid.

    Parameters
    ----------
    psi:
        2-D array, shape ``(H, W)``.  Rows index latitude (ASCENDING,
        row 0 = south), columns index longitude (periodic).
    lat:
        1-D array of latitude values in **radians**, length ``H``,
        ASCENDING (−π/2 … +π/2).

    Returns
    -------
    NDArray of the same shape as *psi*.

    Implements the NAIVE (non-flux-conservative) pointwise central-difference
    form that matches ``velocity.comp``::

        ∇²ψ = (1/cos²φ)·(ψ[i,j+1] − 2ψ[i,j] + ψ[i,j−1])/Δλ²
             + (ψ[i+1,j] − 2ψ[i,j] + ψ[i−1,j])/Δφ²
             − tanφ·(ψ[i+1,j] − ψ[i−1,j])/(2Δφ)

    Longitude is periodic; latitude rows are clamped (boundary rows use the
    same one-sided stencil the GPU clamp-Y produces).
    """
    psi = np.asarray(psi, dtype=float)
    lat = np.asarray(lat, dtype=float)
    H, W = psi.shape

    dlam = 2.0 * np.pi / W          # longitude step (uniform)
    dphi = (lat[-1] - lat[0]) / (H - 1)  # latitude step (uniform, positive)

    cos_phi = np.maximum(np.cos(lat), _COS_FLOOR)   # (H,)
    tan_phi = np.tan(lat)                            # (H,)

    # Longitude stencil — periodic wrap via np.roll
    psi_jp1 = np.roll(psi, -1, axis=1)   # ψ[i, j+1]
    psi_jm1 = np.roll(psi,  1, axis=1)   # ψ[i, j-1]

    # Latitude stencil — clamp boundary rows (mirrors GPU clampY)
    psi_ip1 = np.roll(psi, -1, axis=0)   # would be ψ[i+1, :]
    psi_im1 = np.roll(psi,  1, axis=0)   # would be ψ[i-1, :]
    # clamp: top row (i=H-1) repeats itself; bottom row (i=0) repeats itself
    psi_ip1[-1, :] = psi[-1, :]
    psi_im1[ 0, :] = psi[ 0, :]

    # Lambda (longitude) part: (1/cos²φ) · d²ψ/dλ²
    inv_cos2 = (1.0 / cos_phi**2)[:, np.newaxis]   # (H,1)
    d2_lam = (psi_jp1 - 2.0 * psi + psi_jm1) / dlam**2

    # Phi (latitude) part: d²ψ/dφ²
    d2_phi = (psi_ip1 - 2.0 * psi + psi_im1) / dphi**2

    # Mixed tanφ correction: −tanφ · dψ/dφ  (central difference)
    d1_phi = (psi_ip1 - psi_im1) / (2.0 * dphi)
    tan_col = tan_phi[:, np.newaxis]   # (H,1)

    return inv_cos2 * d2_lam + d2_phi - tan_col * d1_phi


def laplacian_patch(
    psi: NDArray[np.floating],
    rho_max: float,
) -> NDArray[np.floating]:
    """Discrete spherical Laplacian on an azimuthal-equidistant polar patch.

    The patch grid is Cartesian (s, t) in radians, s,t ∈ [−rho_max, rho_max],
    where rho = |(s,t)| is the colatitude (geodesic distance from the pole) and
    theta = atan2(t, s) is the longitude. The AE projection preserves radial
    distance, so the sphere metric is ds² = drho² + sin²rho dtheta² — the pole
    (rho→0) is REGULAR (no 1/cosφ singularity; that is why the patches exist).

    The Laplace-Beltrami ∇²ψ = ψ_ρρ + cot(rho)·ψ_ρ + ψ_θθ/sin²rho, expressed in
    (s,t) via e_r=(s,t)/rho, ψ_ρ=(s·ψ_s+t·ψ_t)/rho, ψ_θθ=(t²ψ_ss−2st·ψ_st+
    s²ψ_tt)−rho·ψ_ρ, gives:

        ∇²ψ = c_ss·ψ_ss + c_tt·ψ_tt + c_st·ψ_st + c_g·(s·ψ_s + t·ψ_t)/rho
        c_ss = s²/rho² + t²/sin²rho
        c_tt = t²/rho² + s²/sin²rho
        c_st = 2·s·t·(1/rho² − 1/sin²rho)
        c_g  = cot(rho) − rho/sin²rho

    All coefficients are finite as rho→0 (→ flat ψ_ss+ψ_tt). Boundary pixels are
    clamped. POLE_SIGN does not enter (Laplacian is independent of N/S).
    """
    psi = np.asarray(psi, dtype=float)
    n = psi.shape[0]
    dstep = 2.0 * rho_max / n        # st-radians per pixel

    # st at pixel centers.
    idx = (np.arange(n) + 0.5) / n * 2.0 - 1.0   # [-1+.., 1-..]
    s = (idx * rho_max)[np.newaxis, :] * np.ones((n, 1))   # varies along x (cols)
    t = (idx * rho_max)[:, np.newaxis] * np.ones((1, n))   # varies along y (rows)
    rho = np.hypot(s, t)
    rho = np.maximum(rho, 1e-6)
    sinr = np.maximum(np.sin(rho), 1e-6)

    # Central differences (clamp boundaries by edge replication).
    pe = np.pad(psi, 1, mode="edge")
    ps = (pe[1:-1, 2:] - pe[1:-1, :-2]) / (2.0 * dstep)        # ∂/∂s (cols = x)
    pt = (pe[2:, 1:-1] - pe[:-2, 1:-1]) / (2.0 * dstep)        # ∂/∂t (rows = y)
    pss = (pe[1:-1, 2:] - 2.0 * psi + pe[1:-1, :-2]) / dstep**2
    ptt = (pe[2:, 1:-1] - 2.0 * psi + pe[:-2, 1:-1]) / dstep**2
    pst = (pe[2:, 2:] - pe[2:, :-2] - pe[:-2, 2:] + pe[:-2, :-2]) / (4.0 * dstep**2)

    inv_r2 = 1.0 / rho**2
    inv_s2 = 1.0 / sinr**2
    c_ss = s**2 * inv_r2 + t**2 * inv_s2
    c_tt = t**2 * inv_r2 + s**2 * inv_s2
    c_st = 2.0 * s * t * (inv_r2 - inv_s2)
    c_g = (np.cos(rho) / sinr) - rho * inv_s2
    psi_rho = (s * ps + t * pt) / rho
    return c_ss * pss + c_tt * ptt + c_st * pst + c_g * psi_rho


# ---------------------------------------------------------------------------
# 2. Zonal-wind vorticity
# ---------------------------------------------------------------------------


def jet_vorticity(
    u: NDArray[np.floating],
    lat: NDArray[np.floating],
) -> NDArray[np.floating]:
    """Relative vorticity of a purely-zonal wind profile.

    Parameters
    ----------
    u:
        1-D eastward wind speed as a function of latitude (same length as
        *lat*).  Any latitude ordering works (np.gradient uses spacing).
    lat:
        1-D latitude in **radians** (ASCENDING or descending — np.gradient
        handles actual coordinate spacing correctly).

    Returns
    -------
    ζ_jet: 1-D array, same shape as *u*.

    Formula::

        ζ_jet = −(1/cosφ) · d(u·cosφ)/dφ

    This equals +∇²ψ_jet for the zonal streamfunction ψ_jet(φ) defined by
    u = −dψ/dφ  (i.e.  u·cosφ = d(ψ·cosφ)/dφ is NOT used — direct deriv).
    """
    u = np.asarray(u, dtype=float)
    lat = np.asarray(lat, dtype=float)
    cos_phi = np.cos(lat)
    d_ucosphi = np.gradient(u * cos_phi, lat)   # d(u cosφ)/dφ
    cos_floor = np.maximum(cos_phi, _COS_FLOOR)
    return -d_ucosphi / cos_floor


# ---------------------------------------------------------------------------
# 3. Gaussian-vortex closed-form vorticity
# ---------------------------------------------------------------------------


def vortex_omega_ref(
    S: float,
    r_core: float,
    d: NDArray[np.floating] | float,
) -> NDArray[np.floating]:
    """Closed-form relative vorticity ω of a Gaussian streamfunction.

    The streamfunction is::

        ψ(d) = S · exp(−q²),    q = d / r_core

    where *d* is the great-circle distance from the vortex centre (radians).

    Under the convention ω = +∇²ψ, the intrinsic 2-sphere Laplacian in
    geodesic-polar coordinates is  ∇²f(d) = f''(d) + cot(d)·f'(d),
    giving::

        f'(d)  = −(2S·q/r_core)·exp(−q²)
        f''(d) = (S/r_core²)·(4q²−2)·exp(−q²)

        ω = (S/r_core²)·(4q²−2)·exp(−q²)  −  (2S·q/r_core)·cot(d)·exp(−q²)

    Equivalently, substituting cot(d)·q = (d/r_core)·(cos d/sin d) = (1/r_core)·(d/tan d)::

        ω = (S/r_core²) · [(4q²−2) − 2·(d/tan d)] · exp(−q²)

    This ``d/tan(d)`` form avoids the cot(d) singularity at d=0 without
    needing a case split for regular evaluation; the small-d guard is only
    needed for finite-precision safety when d is within float rounding of 0::

        lim_{d→0} d/tan(d) = 1  (to leading order; series: 1 − d²/3 + …)

    The cot(d) term is essential for large-core vortices (GRS scale ~0.15 rad):
    skipping it introduces a relative error proportional to (r_core/a)² that
    is ~0.3% at r_core=0.15 but accumulates in the GRS round-trip test.

    Parameters
    ----------
    S:
        Streamfunction amplitude (same units as ψ).
    r_core:
        Core radius in **radians**.
    d:
        Great-circle distance(s) from the vortex centre, in **radians**.
        Scalar or array.

    Returns
    -------
    ω at each point *d* (same shape as *d*).
    """
    d = np.asarray(d, dtype=float)
    q = d / r_core

    # d/tan(d) with small-d series fallback (1 − d²/3) to avoid 0/tan(0).
    # Threshold d < 1e-4 keeps series error well below 1e-8 relative.
    _SMALL_D = 1e-4
    d_over_tand = np.where(
        d < _SMALL_D,
        1.0 - d**2 / 3.0,                          # series: d/tan(d) ≈ 1 − d²/3
        d / np.tan(np.where(d < _SMALL_D, 1.0, d)),
    )

    exp_q2 = np.exp(-q**2)
    # term1: radial curvature part  f''(d)
    term1 = (S / r_core**2) * (4.0 * q**2 - 2.0) * exp_q2
    # term2: geodesic-metric correction  cot(d)·f'(d)
    term2 = -(2.0 * S / r_core**2) * d_over_tand * exp_q2

    return term1 + term2


# ---------------------------------------------------------------------------
# 4. Coriolis parameter
# ---------------------------------------------------------------------------


def coriolis(
    lat: NDArray[np.floating] | float,
    f0: float,
) -> NDArray[np.floating]:
    """Coriolis parameter  f(φ) = f₀·sin φ.

    Parameters
    ----------
    lat:
        Latitude in **radians** (scalar or array).
    f0:
        Planetary rotation rate factor (e.g. 2Ω for Earth, or a
        non-dimensional analogue for gas giants).

    Returns
    -------
    f = f0 * sin(lat), same shape as *lat*.
    """
    return f0 * np.sin(np.asarray(lat, dtype=float))
