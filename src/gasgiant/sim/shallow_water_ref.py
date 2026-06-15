"""Single-layer C-grid shallow-water CPU reference implementation.

Graduated from M0 validated operators (sw_spike) with planetary radius `a`
threaded through every metric site. All GPU kernels diff against this module
1:1 on function signatures.

Metric sites carrying `a`:
  - grad_faces:      zonal  1/(a cosφ dλ),  meridional 1/(a dφ)
  - divergence_hu:   prefactor 1/(a cosφ)
  - vorticity:       prefactor 1/(a cosφ)
  - _apply_fluxes:   prefactor 1/(a cosφ), fluxes divided by a dλ / a dφ

At a=1.0 all factors collapse to the M0 sw_spike values (1/1 == 1), so
a-scaling tests are the only way to catch a missing `a`.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np


# ---------------------------------------------------------------------------
# Grid
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class Grid:
    W: int
    H: int
    a: float = 1.0  # M1 addition: planetary radius (metres or dimensionless)

    @property
    def dlam(self) -> float:
        return 2.0 * np.pi / self.W

    @property
    def dphi(self) -> float:
        return np.pi / self.H

    @property
    def phi_c(self) -> np.ndarray:
        j = np.arange(self.H)
        return 0.5 * np.pi - (j + 0.5) * self.dphi  # descending

    @property
    def phi_v(self) -> np.ndarray:
        j = np.arange(self.H + 1)
        return 0.5 * np.pi - j * self.dphi  # phi_v[0]=+pi/2, phi_v[H]=-pi/2

    @property
    def cos_c(self) -> np.ndarray:
        return np.cos(self.phi_c)

    @property
    def cos_v(self) -> np.ndarray:
        c = np.cos(self.phi_v)
        c[0] = 0.0
        c[-1] = 0.0
        return c

    @property
    def f_c(self) -> np.ndarray:
        # placeholder f0=1; callers scale. Kept as sin(lat) shape.
        return np.sin(self.phi_c)


# ---------------------------------------------------------------------------
# Interpolation helpers (unchanged from M0)
# ---------------------------------------------------------------------------

def center_to_uface(a: np.ndarray) -> np.ndarray:
    """East-face value = average of cell i and i+1 (periodic in lon)."""
    return 0.5 * (a + np.roll(a, -1, axis=1))


def center_to_vface(a: np.ndarray) -> np.ndarray:
    """Meridional face value; shape (H+1, W). Pole faces forced to 0."""
    H, W = a.shape
    vf = np.zeros((H + 1, W), dtype=a.dtype)
    vf[1:H] = 0.5 * (a[0:H - 1] + a[1:H])  # north row (j-1) and south row (j)
    return vf


# ---------------------------------------------------------------------------
# Operators — metric sites annotated with (a)
# ---------------------------------------------------------------------------

def divergence_hu(h: np.ndarray, u: np.ndarray, v: np.ndarray, g: Grid) -> np.ndarray:
    """Flux-form mass divergence ∇·(h u) at cell centers, shape (H, W).

    Spherical metric: (1/(a cosφ))[ ∂(h u)/∂λ + ∂(h v cosφ)/∂φ ].

    Zonal flux uses donor-cell pairing: Fx[j,i] = h[j,i] * u[j,i], where
    u[j,i] is the east-face velocity of cell i and h[j,i] is the collocated
    center value.  Differencing adjacent Fx values gives the correct C-grid
    divergence that sees the 2Δx checkerboard — centered h interpolation to
    faces would average the checkerboard away, creating a null mode.

    Meridional flux interpolates h to v-faces via center_to_vface (average of
    adjacent rows) with cosφ weighting; pole faces are zeroed by construction.
    """
    H, W = h.shape

    # Zonal term: donor-cell flux at east face of cell i.
    # Fx[j,i] = h[j,i] * u[j,i]; west face of cell i is Fx[j,i-1].
    Fx = h * u
    dFx = (Fx - np.roll(Fx, 1, axis=1)) / g.dlam  # angular dλ (no a here)

    # Meridional term: h interpolated to v-faces, weighted by cosφ.
    h_vf = center_to_vface(h)            # (H+1, W)
    Fy = h_vf * v                        # meridional mass flux at v-faces
    cos_v = g.cos_v[:, None]             # (H+1, 1)
    Fy_c = Fy * cos_v                    # h v cosφ
    # North face of row j is index j; south face is index j+1 (φ decreases with j).
    dFy = (Fy_c[0:H] - Fy_c[1:H + 1]) / g.dphi  # angular dφ (no a here)

    inv_metric = 1.0 / (g.a * g.cos_c[:, None])  # (a) 1/(a cosφ) — sole a site
    return inv_metric * (dFx + dFy)


def grad_faces(M: np.ndarray, g: Grid):
    """∇M evaluated on faces (single difference, no 2dx null space).

    Returns (gx at u-faces (H,W), gy at v-faces (H+1,W)).
    """
    H, W = M.shape
    # Zonal gradient at east face i = (M[i+1] - M[i]) / (a cosφ dλ).  (a)
    gx = (np.roll(M, -1, axis=1) - M) / (g.a * g.cos_c[:, None] * g.dlam)
    # Meridional gradient at v-face j = (M[north row j-1] - M[south row j]) / (a dφ).  (a)
    gy = np.zeros((H + 1, W))
    gy[1:H] = (M[0:H - 1] - M[1:H]) / (g.a * g.dphi)
    return gx, gy


def vorticity(u: np.ndarray, v: np.ndarray, g: Grid) -> np.ndarray:
    """Relative vorticity ζ = (1/(a cosφ))[∂v/∂λ − ∂(u cosφ)/∂φ] at corners (H+1, W)."""
    H, W = u.shape
    # ∂v/∂λ at corner (j, i): v lives at v-faces (H+1,W); corner i uses v[i]-v[i-1].
    # Angular dλ — no a here; a enters only in the outer 1/(a cosφ) prefactor below.
    dv_dlam = (v - np.roll(v, 1, axis=1)) / (g.cos_v[:, None] * g.dlam + 1e-30)
    # u cosφ at centers, differenced across the v-face (north row minus south row).
    # Angular dφ — no a here.
    ucos = u * g.cos_c[:, None]
    ducos = np.zeros((H + 1, W))
    ducos[1:H] = (ucos[0:H - 1] - ucos[1:H]) / g.dphi
    # (a) sole site: outer 1/(a cosφ) prefactor in ζ = (1/(a cosφ))[∂v/∂λ − ∂(ucosφ)/∂φ]
    zeta = (dv_dlam - ducos / (g.cos_v[:, None] + 1e-30)) / g.a
    zeta[0] = 0.0
    zeta[H] = 0.0
    return zeta


def corner_to_uface(zc: np.ndarray) -> np.ndarray:
    """Average corner field (H+1,W) to u-faces (H,W): mean of the 2 corners in φ."""
    return 0.5 * (zc[0:-1] + zc[1:])


def coriolis_trapezoidal(u: np.ndarray, v: np.ndarray, f: np.ndarray, dt: float):
    """Energy-neutral (norm-preserving) implicit Coriolis: trapezoidal rotation.

    Solves (u^{n+1}-u^n)/dt = f v*, (v^{n+1}-v^n)/dt = -f u*, with * = ½(n+n+1).
    Closed form is the Cayley rotation by angle θ=f dt.
    """
    alpha = 0.5 * f * dt
    denom = 1.0 + alpha * alpha
    u_new = ((1.0 - alpha * alpha) * u + 2.0 * alpha * v) / denom
    v_new = ((1.0 - alpha * alpha) * v - 2.0 * alpha * u) / denom
    return u_new, v_new


def _mass_fluxes(h, u, v, g):
    """Low-order (donor-cell / upwind) and high-order (centered) mass fluxes."""
    H, W = h.shape
    # Zonal east-face flux. Upwind donor by sign of u.
    hE_up = np.where(u >= 0, h, np.roll(h, -1, axis=1))
    Fx_low = hE_up * u
    Fx_high = center_to_uface(h) * u
    # Meridional v-face flux. Upwind donor: v>0 means flow toward south (row j),
    # so donor is the north row (j-1).
    h_north = np.zeros((H + 1, W)); h_north[1:H] = h[0:H - 1]
    h_south = np.zeros((H + 1, W)); h_south[1:H] = h[1:H]
    hV_up = np.where(v >= 0, h_north, h_south)
    Fy_low = hV_up * v
    Fy_high = center_to_vface(h) * v
    return Fx_low, Fx_high, Fy_low, Fy_high


def _apply_fluxes(h, Fx, Fy, g, dt):
    """Apply flux divergence to update h; metric identical to divergence_hu.

    (a) sole site: outer 1/(a cosφ) prefactor, matching divergence_hu.
    Angular dλ, dφ appear without a (angle-space differences).
    """
    H, W = h.shape
    dFx = (Fx - np.roll(Fx, 1, axis=1)) / g.dlam        # angular dλ
    Fy_c = Fy * g.cos_v[:, None]
    dFy = (Fy_c[0:H] - Fy_c[1:H + 1]) / g.dphi         # angular dφ
    return h - dt * (dFx + dFy) / (g.a * g.cos_c[:, None])  # (a) 1/(a cosφ)


def continuity_step(h, u, v, g, dt, h_floor):
    """Flux-corrected transport: mass-conserving AND positivity-preserving.

    Zalesak-style limiter: blend high-order toward low-order so the update
    introduces no new extremum below the floor.
    """
    Fx_low, Fx_high, Fy_low, Fy_high = _mass_fluxes(h, u, v, g)
    h_low = _apply_fluxes(h, Fx_low, Fy_low, g, dt)           # monotone, positive
    h_low = np.maximum(h_low, h_floor)
    # Anti-diffusive flux = high - low.
    Ax = Fx_high - Fx_low
    Ay = Fy_high - Fy_low
    # Limit each anti-diffusive flux so it cannot pull any cell below the floor.
    # Outgoing capacity of each cell above the floor:
    cap = np.maximum(h_low - h_floor, 0.0) * g.cos_c[:, None] / dt
    # Scale anti-diffusive fluxes by the most-restrictive adjacent capacity.
    sx = np.minimum(1.0, cap / (np.abs(Ax) + 1e-30))
    Ax_lim = Ax * np.minimum(sx, np.roll(sx, -1, axis=1))
    # Meridional limiter (simple, conservative): clamp by the donor-row capacity.
    cap_v = np.zeros((g.H + 1, g.W)); cap_v[1:g.H] = np.minimum(cap[0:g.H - 1], cap[1:g.H])
    sy = np.minimum(1.0, cap_v / (np.abs(Ay) + 1e-30))
    Ay_lim = Ay * sy
    h_new = _apply_fluxes(h, Fx_low + Ax_lim, Fy_low + Ay_lim, g, dt)
    return np.maximum(h_new, h_floor)


# ---------------------------------------------------------------------------
# M2 semi-implicit: adjoint Helmholtz operator pair
# ---------------------------------------------------------------------------

def divergence_helmholtz(
    Fx: np.ndarray,
    Fy: np.ndarray,
    H_ref_lat: np.ndarray,
    g: Grid,
) -> np.ndarray:
    """Negative adjoint of grad_faces under cos-weighted inner products.

    Computes the flux-form, cos-weighted divergence

        D[j,i] = [ d(H_x * Fx)/dlam + d(H_y * Fy * cosφ)/dphi ] / (a cosφ_c)

    where H_x = H_ref_lat[:, None] (center values broadcast to u-faces) and
    H_y[k] = 0.5*(H_ref_lat[k-1] + H_ref_lat[k]) (meridional average broadcast
    to v-faces; pole rows k=0 and k=H are zeroed because cos_v=0 there).

    Adjoint identity (with H_ref_lat ≡ 1):
        <grad_faces(h), (Fx, Fy)>_faces = -<h, divergence_helmholtz(Fx, Fy, 1, g)>_centers

    where:
        <p, q>_centers  = Σ p*q*cos_c[:,None]
        <(ax,ay),(bx,by)>_faces = Σ ax*bx*cos_c[:,None] + Σ ay*by*cos_v[:,None]

    The derivation is summation-by-parts: shift the λ index on the zonal term
    (periodic) and the φ index on the meridional term to move finite differences
    from M onto (Fx, Fy).  The result is this exact formula; the common factor
    a² dlam dphi cancels and is omitted consistently from both sides.

    Parameters
    ----------
    Fx : ndarray, shape (H, W)
        Flux at u-faces (east faces).
    Fy : ndarray, shape (H+1, W)
        Flux at v-faces (meridional faces).  Pole rows (0, H) should be zero;
        cos_v=0 there forces their contribution to vanish regardless.
    H_ref_lat : ndarray, shape (H,)
        Reference layer thickness profile (latitude-only).  Applied SYMMETRICALLY:
        same weighting on the GRAD side and the DIV side so the composed operator
        L = -div(H_ref * grad) is self-adjoint.
    g : Grid

    Returns
    -------
    ndarray, shape (H, W) — divergence at cell centers.
    """
    H, W = g.H, g.W

    # --- Apply H_ref symmetrically to face fluxes ---
    # u-faces: use center value at each latitude row.
    Hx = H_ref_lat[:, None]          # (H, 1), broadcast to (H, W)
    Fx_w = Hx * Fx                   # weighted zonal flux, (H, W)

    # v-faces: meridional average of adjacent centers.
    # H_y[k] = 0.5*(H_ref_lat[k-1] + H_ref_lat[k]) for k=1..H-1; poles = 0.
    H_ref_v = np.zeros(H + 1)
    H_ref_v[1:H] = 0.5 * (H_ref_lat[0:H - 1] + H_ref_lat[1:H])
    Fy_w = H_ref_v[:, None] * Fy     # weighted meridional flux, (H+1, W)

    # --- Zonal divergence term: (Fx_w[j,i] - Fx_w[j,i-1]) / (a dlam) ---
    # This is -d/dlam applied to Fx_w (roll by +1 brings i-1 into position i).
    dFx = (Fx_w - np.roll(Fx_w, 1, axis=1)) / (g.a * g.dlam)    # (H, W)

    # --- Meridional divergence term ---
    # The adjoint derivation (summation-by-parts on the φ index) gives:
    #   adjoint contribution: Σ_j h[j] * (Fy_cos[j+1] - Fy_cos[j]) / (a dphi)
    # where Fy_cos[k] = Fy[k] * cos_v[k].
    # Since divergence_helmholtz is the NEGATIVE adjoint we subtract this term.
    # The outer metric 1/cos_c converts from the centre inner product to a raw sum.
    Fy_cos = Fy_w * g.cos_v[:, None]       # (H+1, W): cosφ_v weighted flux
    # (Fy_cos[j+1] - Fy_cos[j]) is the adjoint meridional contribution per cell j
    dFy = (Fy_cos[1:H + 1] - Fy_cos[0:H]) / (g.a * g.dphi)      # (H, W)

    # Return the NEGATIVE adjoint of grad_faces:
    #   divergence_helmholtz = (dFx - dFy) / cos_c
    # Proof: <grad h, U>_faces
    #   zonal: Σ h[j,i] * (Ux[j,i-1]-Ux[j,i])/(a dlam)        = -<h, dFx/cos_c>_c
    #   merid: Σ h[j,i] * (Uy_cos[j+1]-Uy_cos[j])/(a dphi)    = -<h, (-dFy/cos_c)>_c
    # So <grad h, U>_faces = -<h, (dFx-dFy)/cos_c>_c = -<h, div_H>_c  ✓
    return (dFx - dFy) / g.cos_c[:, None]


def helmholtz_apply(
    dh: np.ndarray,
    H_ref_lat: np.ndarray,
    gp: float,
    theta: float,
    dt: float,
    g: Grid,
) -> np.ndarray:
    """Apply the semi-implicit Helmholtz operator to a height perturbation.

        L(dh) = dh - (theta * dt)^2 * gp * divergence_helmholtz(*grad_faces(dh), H_ref_lat, g)

    This operator arises from the θ-implicit wave equation.  It is self-adjoint
    and positive definite (for H_ref_lat > 0, gp > 0) because the inner operator
    is -div(H_ref * grad), which is SPD on the C-grid.

    Parameters
    ----------
    dh : ndarray, shape (H, W)
        Height perturbation to apply the operator to.
    H_ref_lat : ndarray, shape (H,)
        Reference layer thickness profile, latitude-only.
    gp : float
        Reduced gravity g'.
    theta : float
        Implicitness parameter (0=explicit, 1=fully implicit).
    dt : float
        Time step.
    g : Grid

    Returns
    -------
    ndarray, shape (H, W)
    """
    alpha = (theta * dt) ** 2 * gp
    gx, gy = grad_faces(dh, g)
    return dh - alpha * divergence_helmholtz(gx, gy, H_ref_lat, g)


# ---------------------------------------------------------------------------
# Single-layer Montgomery potential and pressure gradient
# ---------------------------------------------------------------------------

def pressure_grad(h: np.ndarray, gp: float, g: Grid):
    """Single-layer Montgomery potential gradient.

    M = g' h  (reduced gravity times layer thickness).
    Returns (gx, gy) = grad_faces(M, g).
    """
    return grad_faces(gp * h, g)


# ---------------------------------------------------------------------------
# SwRefState dataclass for single-layer Williamson-2 integration
# ---------------------------------------------------------------------------

@dataclass
class SwRefState:
    """Single-layer shallow-water state for CPU reference integration.

    Fields
    ------
    g      : Grid (W, H, a)
    gp     : reduced gravity g' (float)
    h      : layer thickness at centers (H, W)
    u      : zonal velocity at u-faces (H, W)
    v      : meridional velocity at v-faces (H+1, W)
    dt     : time step (set by CFL at construction)
    omega  : planetary rotation rate (½ f / sinφ)
    u_init : copy of initial u for drift diagnostics
    v_init : copy of initial v for drift diagnostics
    h_floor: positivity floor for h (default 0.05)
    """
    g: Grid
    gp: float
    h: np.ndarray
    u: np.ndarray
    v: np.ndarray
    dt: float
    omega: float
    u_init: np.ndarray
    v_init: np.ndarray
    h_floor: float = 0.05


# ---------------------------------------------------------------------------
# Williamson test 2 analytic initial condition
# ---------------------------------------------------------------------------

def williamson2_state(
    W: int, H: int, a: float,
    omega: float, u0: float, gp: float, h0: float,
    h_floor: float = 0.05,
) -> SwRefState:
    """Steady geostrophic solid-body flow (Williamson test 2).

    u = u0 · cosφ  (zonal, uniform in λ)
    v = 0
    h = h0 − (a·Ω·u0 + u0²/2) · sin²φ / g'

    This is an exact steady solution → dt chosen by polar CFL.
    """
    g = Grid(W, H, a)

    cos_c = g.cos_c[:, None] * np.ones((1, W))    # (H, W)
    sin_c = np.sin(g.phi_c)[:, None] * np.ones((1, W))  # (H, W)

    # Zonal velocity at u-faces (H, W): u = u0 cosφ, uniform in λ.
    u = u0 * cos_c

    # Meridional velocity at v-faces (H+1, W): zero.
    v = np.zeros((H + 1, W))

    # Analytic height at cell centers (H, W).
    h = h0 - (a * omega * u0 + 0.5 * u0 * u0) * sin_c * sin_c / gp
    h = np.maximum(h, h_floor)

    # CFL dt: gravity wave speed, minimum grid spacing (polar CFL governs).
    c_gw = np.sqrt(gp * h.max())
    cos_min = np.maximum(g.cos_c.min(), 1e-6)   # avoid division by zero at poles
    dx_min = min(cos_min * a * g.dlam, a * g.dphi)
    dt = 0.3 * dx_min / c_gw

    return SwRefState(
        g=g, gp=gp, h=h.copy(), u=u.copy(), v=v.copy(),
        dt=dt, omega=omega,
        u_init=u.copy(), v_init=v.copy(),
        h_floor=h_floor,
    )


# ---------------------------------------------------------------------------
# Single-layer momentum step (vector-invariant, relative vorticity flux only)
# ---------------------------------------------------------------------------

def momentum_step(
    h: np.ndarray, u: np.ndarray, v: np.ndarray,
    gp: float, omega: float, g: Grid, dt: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Vector-invariant momentum update for a single layer.

    Advection+pressure are explicit; Coriolis is implicit (trapezoidal).
    Only relative vorticity ζ enters the flux cross term; the full
    Coriolis f = 2Ω sinφ is applied separately via coriolis_trapezoidal.

    Matches sw_spike/solver.py::_layer_momentum with `a`-aware operators.
    """
    H, W = h.shape

    # Relative vorticity at corners (H+1, W).
    zeta = vorticity(u, v, g)

    # Interpolate ζ to u-faces and v-faces.
    zeta_uf = corner_to_uface(zeta)                                      # (H, W)
    zeta_vf = 0.5 * (zeta + np.roll(zeta, 1, axis=1))                   # (H+1, W) corner→v-face

    # v at cell centers, then interpolated to u-faces.
    v_c = 0.5 * (v[0:H] + v[1:H + 1])                                   # (H, W)
    v_at_uf = center_to_uface(v_c)                                       # (H, W)

    # u at cell centers, then interpolated to v-faces.
    u_c = 0.5 * (u + np.roll(u, 1, axis=1))                              # (H, W) centers
    u_at_vf = center_to_vface(u_c)                                       # (H+1, W)

    # Bernoulli potential B = g'h + ½(u² + v_c²) at centers.
    ke = 0.5 * (u * u + v_c * v_c)
    B = gp * h + ke
    gx, gy = grad_faces(B, g)                                             # face gradients

    # Explicit step: advection by relative vorticity + pressure gradient.
    u_star = u + dt * (zeta_uf * v_at_uf - gx)
    v_star = v.copy()
    v_star[1:H] = v[1:H] + dt * (-zeta_vf[1:H] * u_at_vf[1:H] - gy[1:H])

    # Coriolis f at u-faces (H, W): f = 2Ω sinφ_c (same as sw_spike _f_uface).
    f_uf = 2.0 * omega * np.sin(g.phi_c)[:, None] * np.ones((1, W))    # (H, W)

    # Trapezoidal Coriolis rotation on (u_star, v_star collapsed to centers).
    v_star_c = 0.5 * (v_star[0:H] + v_star[1:H + 1])                   # (H, W)
    u_new, v_c_new = coriolis_trapezoidal(u_star, v_star_c, f_uf, dt)

    # Scatter v_c_new back to v-faces: interior faces = avg of adjacent center rows.
    v_new = np.zeros_like(v)
    v_new[1:H] = 0.5 * (v_c_new[0:H - 1] + v_c_new[1:H])

    return u_new, v_new


# ---------------------------------------------------------------------------
# Top-level step function
# ---------------------------------------------------------------------------

def step(st: SwRefState) -> SwRefState:
    """Advance the state by one time step.

    Order: momentum from (h_old, u_old, v_old), then continuity from
    (h_old, u_new, v_new).  Matches sw_spike/solver.step.
    """
    u_new, v_new = momentum_step(st.h, st.u, st.v, st.gp, st.omega, st.g, st.dt)
    h_new = continuity_step(st.h, u_new, v_new, st.g, st.dt, st.h_floor)
    return SwRefState(
        g=st.g, gp=st.gp,
        h=h_new, u=u_new, v=v_new,
        dt=st.dt, omega=st.omega,
        u_init=st.u_init, v_init=st.v_init,
        h_floor=st.h_floor,
    )


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------

def total_mass(st: SwRefState) -> float:
    """Global mass integral: Σ h · cosφ · a² dλ dφ."""
    g = st.g
    return float(np.sum(st.h * g.cos_c[:, None]) * g.a * g.a * g.dlam * g.dphi)


def total_energy(st: SwRefState) -> float:
    """Global total energy: Σ [½h(u² + v_c²) + ½g'h²] cosφ · a² dλ dφ."""
    g = st.g
    H = g.H
    v_c = 0.5 * (st.v[0:H] + st.v[1:H + 1])
    ke = 0.5 * st.h * (st.u * st.u + v_c * v_c)
    pe = 0.5 * st.gp * st.h * st.h
    return float(np.sum((ke + pe) * g.cos_c[:, None]) * g.a * g.a * g.dlam * g.dphi)


def velocity_l2_drift(st: SwRefState) -> float:
    """RMS drift from initial velocity: sqrt(mean_u((u-u0)²) + mean_v((v-v0)²)).

    u and v live on different staggered grids (H,W) and (H+1,W), so each is
    averaged separately and the combined RMS is returned.
    """
    du = st.u - st.u_init
    dv = st.v - st.v_init
    return float(np.sqrt(np.mean(du * du) + np.mean(dv * dv)))


def total_potential_enstrophy(st: SwRefState) -> float:
    """Global potential enstrophy diagnostic: Σ ½(ζ+f)²/h_corner · cosφ_corner · a² dλ dφ.

    h_corner is averaged from adjacent cells; floored to avoid division by zero.
    """
    g = st.g
    H, W = g.H, g.W
    zeta = vorticity(st.u, st.v, g)           # (H+1, W)
    f_v = 2.0 * st.omega * np.sin(g.phi_v)[:, None] * np.ones((1, W))  # (H+1, W)
    abs_vort = zeta + f_v

    # h at corners: average of four adjacent cells (clipped to interior).
    h_corner = np.full((H + 1, W), st.h_floor)
    h_corner[1:H] = 0.5 * (st.h[0:H - 1] + st.h[1:H])  # meridional avg (simplified)
    h_corner = np.maximum(h_corner, st.h_floor)

    cos_v = g.cos_v[:, None] * np.ones((1, W))
    ens = 0.5 * abs_vort * abs_vort / h_corner * cos_v
    return float(np.sum(ens) * g.a * g.a * g.dlam * g.dphi)
