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


def coriolis_sandwich(
    u: np.ndarray,
    v: np.ndarray,
    omega: float,
    g: Grid,
    dt: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Coriolis sandwich: collapse v to centers, rotate, scatter back to v-faces.

    Reproduces the exact sequence in momentum_step: forward interpolation
    (S_fwd), trapezoidal rotation, and backward scatter (S_back).

    Parameters
    ----------
    u : ndarray, shape (H, W)
        Zonal velocity on u-faces.
    v : ndarray, shape (H+1, W)
        Meridional velocity on v-faces.  Pole rows (0 and H) must be zero.
    omega : float
        Planetary rotation rate Ω (rad/s).
    g : Grid
    dt : float
        Time step.

    Returns
    -------
    u_new : ndarray, shape (H, W)
    v_new : ndarray, shape (H+1, W)
        Pole rows remain zero.
    """
    H, W = u.shape

    # Coriolis parameter at u-faces / cell-center latitude (H, W).
    f_uf = 2.0 * omega * np.sin(g.phi_c)[:, None] * np.ones((1, W))

    # S_fwd: collapse v from v-faces to cell centers.
    v_c = 0.5 * (v[0:H] + v[1:H + 1])                   # (H, W)

    # Trapezoidal rotation on the center-collocated pair.
    u_new, v_c_new = coriolis_trapezoidal(u, v_c, f_uf, dt)

    # S_back: scatter v back to v-faces; pole rows stay zero.
    v_new = np.zeros_like(v)
    v_new[1:H] = 0.5 * (v_c_new[0:H - 1] + v_c_new[1:H])

    return u_new, v_new


def velocity_backsub(
    u_star: np.ndarray,
    v_star: np.ndarray,
    h_impl: np.ndarray,
    gp: float,
    theta: float,
    dt: float,
    omega: float,
    g: Grid,
) -> tuple[np.ndarray, np.ndarray]:
    """M2 semi-implicit back-substitution: subtract pressure gradient, then rotate.

    Computes the theta-implicit pressure-gradient correction and applies the
    Coriolis sandwich.  The predictor u_star/v_star already carries the
    (1-theta) explicit pressure half, so the field passed here is the FULL
    implicit height h^{n+1} (not an increment); its gradient supplies the
    theta-weighted implicit pressure.  When h_impl has zero gradient (uniform
    field) the output equals coriolis_sandwich(u_star, v_star) exactly.

    Parameters
    ----------
    u_star : ndarray, shape (H, W)
        Provisional zonal velocity after the explicit step (incl. the (1-theta)
        explicit pressure half).
    v_star : ndarray, shape (H+1, W)
        Provisional meridional velocity after the explicit step.
    h_impl : ndarray, shape (H, W)
        Full implicit layer height h^{n+1} = h^n + dh at which the theta
        pressure gradient is evaluated.
    gp : float
        Reduced gravity g'.
    theta : float
        Implicitness parameter (0=explicit, 1=fully implicit).
    dt : float
        Time step.
    omega : float
        Planetary rotation rate Ω (rad/s).
    g : Grid

    Returns
    -------
    u_new : ndarray, shape (H, W)
    v_new : ndarray, shape (H+1, W)
    """
    gx, gy = grad_faces(h_impl, g)                        # (H,W), (H+1,W)
    u_corr = u_star - theta * dt * gp * gx
    v_corr = v_star - theta * dt * gp * gy
    return coriolis_sandwich(u_corr, v_corr, omega, g, dt)


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


def _positive_lowflux_scales(h, Fx_low, Fy_low, g, dt, h_floor):
    """Per-cell outflux scale factors making the donor-cell pass positivity-preserving.

    The donor-cell (upwind) low-order pass is monotone for pure advection at
    Courant < 1, but it is NOT positivity-preserving under a DIVERGENT velocity:
    where div(u) > 0 it drains a cell, and a near-floor cell can be driven below
    h_floor.  The M1 continuity_step then clamps such cells UP to the floor with
    np.maximum, which INJECTS mass (non-conservative — this is the M2-T5 leak).

    The conservative cure is to make the low-order base itself positive by
    limiting each cell's total OUTFLUX to the mass it has above the floor, then
    applying the SAME scaled face flux to both adjacent cells (flux form ->
    exact conservation).  Each face has a single upwind donor, so scaling a face
    by its donor cell's factor s in [0,1] removes only outgoing mass.

    Returns (sx, sy): face-flux multipliers for Fx_low (H,W) and Fy_low (H+1,W).
    """
    H, W = h.shape
    pref = dt / (g.a * g.cos_c[:, None])          # _apply_fluxes outer factor

    # Per-cell available mass above the floor, in the same units the flux
    # divergence consumes it: dh_avail such that pref*(out_div) <= h - h_floor.
    avail = np.maximum(h - h_floor, 0.0)

    # Outgoing flux-divergence contribution of each cell (positive part only).
    # Zonal: east face Fx[i] leaves cell i if >0; west face Fx[i-1] leaves cell i
    #        if <0 (i.e. -Fx[i-1] when Fx[i-1] < 0).
    out_xe = np.maximum(Fx_low, 0.0) / g.dlam                       # leaves via east face
    out_xw = np.maximum(-np.roll(Fx_low, 1, axis=1), 0.0) / g.dlam  # leaves via west face
    # Meridional: Fy_c[j] (north face of cell j) leaves if >0; Fy_c[j+1] (south
    # face) leaves if <0.  Fy_c = Fy * cos_v.
    Fy_c = Fy_low * g.cos_v[:, None]
    out_yn = np.maximum(Fy_c[0:H], 0.0) / g.dphi                    # north face (row j)
    out_ys = np.maximum(-Fy_c[1:H + 1], 0.0) / g.dphi              # south face (row j)

    out_total = pref * (out_xe + out_xw + out_yn + out_ys)          # mass leaving cell
    # Scale so cell never loses more than its above-floor mass.
    s_cell = np.minimum(1.0, avail / (out_total + 1e-300))          # (H, W) in [0,1]

    # Map cell scales onto faces by the donor (upwind) cell.
    # Zonal east face i: donor is cell i if Fx>=0 else cell i+1.
    sx = np.where(Fx_low >= 0, s_cell, np.roll(s_cell, -1, axis=1))
    # Meridional v-face j (1..H-1): donor is north cell j-1 if Fy>=0 else south cell j.
    sy = np.ones((H + 1, W))
    sy[1:H] = np.where(Fy_low[1:H] >= 0, s_cell[0:H - 1], s_cell[1:H])
    return sx, sy


def continuity_step_conservative(h, u, v, g, dt, h_floor):
    """Mass-conserving FCT with donor-cell positivity limiting (EXACTLY conservative).

    Identical Zalesak anti-diffusive limiting to continuity_step, but the
    LOW-ORDER base is first made positivity-preserving by per-face outflux limiting
    (_positive_lowflux_scales).  Mass is conserved to round-off UNCONDITIONALLY
    (every scaled face flux is shared identically by its two adjacent cells, so the
    flux-form divergence telescopes exactly).

    Positivity is preserved only in the DONOR-CELL sub-CFL regime: the limiter
    scales each face by its upwind cell's available mass, so a cell drained through
    a face for which it is NOT the donor (e.g. a near-floor cell straddled by a
    divergent meridional velocity at Courant >~ 1) can still dip below h_floor.
    The caller (step_semi_implicit) guards this with a loud positivity check rather
    than a silent floor clamp, so an out-of-regime config fails instead of leaking
    mass.  Within M2's validated regime the result is >= h_floor and no clamp fires.

    Used by step_semi_implicit.  continuity_step (the M1 path) is left BYTE-FOR-
    BYTE unchanged for GPU parity.
    """
    Fx_low, Fx_high, Fy_low, Fy_high = _mass_fluxes(h, u, v, g)
    # 1. Make the low-order donor pass positivity-preserving via outflux limiting.
    #    Fx_low_pos / Fy_low_pos are the positivity-limited low-order fluxes; they
    #    define the monotone base h_low.  The anti-diffusive correction below is
    #    relative to THIS limited base (high - limited_low), which is consistent
    #    with the base h_low is built from.
    sx_pos, sy_pos = _positive_lowflux_scales(h, Fx_low, Fy_low, g, dt, h_floor)
    Fx_low_pos = Fx_low * sx_pos
    Fy_low_pos = Fy_low * sy_pos
    h_low = _apply_fluxes(h, Fx_low_pos, Fy_low_pos, g, dt)         # now >= h_floor
    # 2. Zalesak anti-diffusive correction toward the high-order (centered) flux.
    Ax = Fx_high - Fx_low_pos
    Ay = Fy_high - Fy_low_pos
    cap = np.maximum(h_low - h_floor, 0.0) * g.cos_c[:, None] / dt
    sx = np.minimum(1.0, cap / (np.abs(Ax) + 1e-30))
    Ax_lim = Ax * np.minimum(sx, np.roll(sx, -1, axis=1))
    cap_v = np.zeros((g.H + 1, g.W)); cap_v[1:g.H] = np.minimum(cap[0:g.H - 1], cap[1:g.H])
    sy = np.minimum(1.0, cap_v / (np.abs(Ay) + 1e-30))
    Ay_lim = Ay * sy
    # 3. Final donor-cell outflux limiting on the COMBINED flux.  The split Zalesak
    #    caps in (2) limit the zonal and meridional anti-diffusive fluxes
    #    independently; re-applying the conservative limiter to the total flux pulls
    #    most near-floor cells back to >= h_floor while staying EXACTLY mass-
    #    conserving (flux form).  It is NOT a full positivity guarantee: a floor cell
    #    drained through faces where it is the non-donor side (divergent meridional
    #    velocity, Courant >~ 1) can still dip below the floor — the caller guards
    #    that loudly rather than clamping (which would inject mass).
    Fx_tot = Fx_low_pos + Ax_lim
    Fy_tot = Fy_low_pos + Ay_lim
    gx, gy = _positive_lowflux_scales(h, Fx_tot, Fy_tot, g, dt, h_floor)
    return _apply_fluxes(h, Fx_tot * gx, Fy_tot * gy, g, dt)


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

    if H_ref_lat.shape != (H,):
        raise ValueError(
            f"H_ref_lat must be shape ({H},) (latitude-only), got {H_ref_lat.shape}"
        )

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
    # Backward difference — SBP adjoint of the forward difference in grad_faces.
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
# M2-T3: Helmholtz RHS assembly and Picard contraction certificate
# ---------------------------------------------------------------------------

def helmholtz_rhs(
    h_n: np.ndarray,
    u_n: np.ndarray,
    v_n: np.ndarray,
    u_star: np.ndarray,
    v_star: np.ndarray,
    dh_prev: np.ndarray,
    H_ref_lat: np.ndarray,
    gp: float,
    omega: float,
    theta: float,
    dt: float,
    g: Grid,
) -> np.ndarray:
    """Assemble the RHS of the semi-implicit Helmholtz system for the increment.

    Solves for the height INCREMENT dh = h^{n+1} - h^n via

        L_sym(dh) = b    (L_sym = helmholtz_apply)

    This is the textbook theta-scheme (Crank-Nicolson at theta=0.5) in which the
    linear gravity-wave terms are time-centered.  The momentum predictor carries
    only the (1-theta) explicit half of the pressure gradient; the theta-implicit
    half acts on the FULL height h^{n+1} = h^n + dh through velocity_backsub, so
    the implicit pressure is the dominant restoring force and the explicit
    gravity-wave CFL is removed (unconditional stability, neutral at theta=0.5).

    Eliminating u^{n+1} from the theta-centered continuity equation

        dh = -dt * div_H( H_ref * [theta * u^{n+1} + (1-theta) * C(u_n)] )

    with u^{n+1} = C(u_star) - theta*dt*gp * C( grad(h^n + dh) ) gives the three
    RHS terms (the identity part of the implicit pressure goes to L_sym):

    1. refdiv = -dt * div_H( H_ref * [theta*C(u_star) + (1-theta)*C(u_n)] )
       The theta-centered reference-layer divergence of the predictor velocities.
       C(.) = coriolis_sandwich(., omega, g, dt).

    2. bg = +theta*dt * div_H( H_ref * C( theta*dt*gp * grad(h^n) ) )
       The implicit pressure response to the (known, fixed) BACKGROUND height
       gradient grad(h^n).  This is the (theta*dt)^2 cross term carrying the
       full implicit pressure into the height update.  It is evaluated at h^n
       (not dh) and is therefore exact — no Picard lag.

    3. defer = +theta*dt * div_H( (C - I)( theta*dt*gp * grad(dh_prev) ) )
       DEFERRED Coriolis correction for the increment: only the (Coriolis - I)
       part of the implicit pressure response to dh, lagged at dh_prev.  The
       identity (I) part is on the LHS in L_sym and must NOT appear here.

    Parameters
    ----------
    h_n : ndarray, shape (H, W)
        Current height h^n (the background; its gradient is treated implicitly).
    u_n, v_n : ndarrays
        Current velocities (u^n at u-faces (H,W), v^n at v-faces (H+1,W)).
    u_star : ndarray, shape (H, W)
        Provisional zonal velocity after the explicit advection + (1-theta)
        pressure predictor (NO Coriolis).
    v_star : ndarray, shape (H+1, W)
        Provisional meridional velocity from the predictor.
    dh_prev : ndarray, shape (H, W)
        Height increment from the previous Picard iteration (dh^(m)).
    H_ref_lat : ndarray, shape (H,)
        Reference layer thickness profile (latitude-only).
    gp : float
        Reduced gravity g'.
    omega : float
        Planetary rotation rate Omega (rad/s).
    theta : float
        Off-centering parameter (0=explicit, 1=fully implicit; 0.5 is standard).
    dt : float
        Time step.
    g : Grid

    Returns
    -------
    ndarray, shape (H, W)
    """
    tdt = theta * dt

    # --- Term 1: theta-centered reference-layer divergence of predictor velocities ---
    u_ncs, v_ncs = coriolis_sandwich(u_n, v_n, omega, g, dt)
    u_pcs, v_pcs = coriolis_sandwich(u_star, v_star, omega, g, dt)
    u_tc = theta * u_pcs + (1.0 - theta) * u_ncs
    v_tc = theta * v_pcs + (1.0 - theta) * v_ncs
    refdiv = -dt * divergence_helmholtz(u_tc, v_tc, H_ref_lat, g)

    # --- Term 2: implicit pressure response to the background gradient grad(h^n) ---
    gx_n, gy_n = grad_faces(h_n, g)
    u_bg = tdt * gp * gx_n
    v_bg = tdt * gp * gy_n
    u_bgc, v_bgc = coriolis_sandwich(u_bg, v_bg, omega, g, dt)
    bg = tdt * divergence_helmholtz(u_bgc, v_bgc, H_ref_lat, g)

    # --- Term 3: deferred Coriolis correction for the increment (only (C - I) part) ---
    gx_d, gy_d = grad_faces(dh_prev, g)
    u_d = tdt * gp * gx_d
    v_d = tdt * gp * gy_d
    u_dc, v_dc = coriolis_sandwich(u_d, v_d, omega, g, dt)
    defer = tdt * divergence_helmholtz(u_dc - u_d, v_dc - v_d, H_ref_lat, g)

    return refdiv + bg + defer


def picard_contraction_factor(
    omega: float,
    theta: float,
    dt: float,
    g: Grid,
) -> float:
    """Picard contraction factor for the deferred-Coriolis iteration.

    The Picard iteration for the semi-implicit Helmholtz system contracts
    because the deferred Coriolis term is a bounded perturbation.  The
    contraction factor is bounded by the Cayley off-diagonal magnitude:

        rho = 2*alpha / (1 + alpha**2)

    evaluated at the worst-case (largest |f|) latitude, where:

        alpha = 0.5 * |f|_max * dt
        |f|_max = 2 * omega * max(|sin(phi_c)|)

    This is the Cayley rotation off-diagonal entry: the (Coriolis - I)
    operator has operator norm <= rho (the averaging sandwich has norm <= 1,
    so the Cayley off-diagonal is the binding bound).

    For rho < 1 the iteration is a contraction; for rho < 0.5 it converges
    rapidly.  A pre-run check in T4 rejects configs with rho too close to 1.

    Parameters
    ----------
    omega : float
        Planetary rotation rate Omega (rad/s).
    theta : float
        Off-centering parameter (unused in the bound, included for API
        consistency with the T4 caller which may extend the formula).
    dt : float
        Time step.
    g : Grid

    Returns
    -------
    float
        Contraction factor rho in [0, 1).
    """
    f_max = 2.0 * omega * float(np.max(np.abs(np.sin(g.phi_c))))
    alpha = 0.5 * f_max * dt
    return 2.0 * alpha / (1.0 + alpha ** 2)


# ---------------------------------------------------------------------------
# M2-T4: Helmholtz solvers (red-black SOR, exact sparse, residual diagnostic)
# ---------------------------------------------------------------------------

def _helmholtz_diagonal(
    H_ref_lat: np.ndarray,
    gp: float,
    theta: float,
    dt: float,
    g: Grid,
) -> np.ndarray:
    """Analytic diagonal D[j,i] = coefficient of dh[j,i] in helmholtz_apply(dh)[j,i].

    L_sym = I - alpha * div_H(grad), alpha = (theta*dt)**2 * gp.  The composed
    stencil is the standard cos-weighted Laplacian; its diagonal is derived
    analytically (verified to 1e-13 against a unit-basis numerical extraction).

    The identity contributes +1.  The zonal Laplacian contributes
        -alpha * Hx[j] * (-2) / (a*cos_c[j]*dlam)^2
    and the meridional Laplacian contributes
        -alpha * (Hv[j+1]*cos_v[j+1] + Hv[j]*cos_v[j]) / (a^2 * dphi^2 * cos_c[j]).

    Returns array (H, W) (uniform in i, broadcast for convenience).
    """
    H, W = g.H, g.W
    a = g.a
    dlam, dphi = g.dlam, g.dphi
    cos_c = g.cos_c
    cos_v = g.cos_v
    alpha = (theta * dt) ** 2 * gp

    Hx = H_ref_lat
    Hv = np.zeros(H + 1)
    Hv[1:H] = 0.5 * (H_ref_lat[0:H - 1] + H_ref_lat[1:H])

    # Zonal: gx[i] = (dh[i+1]-dh[i])/(a cos dlam); dh[j,i] appears in gx[i] (-) and
    # gx[i-1] (+). div zonal = Hx*(gx[i]-gx[i-1])/(a dlam)/cos_c. Coefficient of dh[j,i]:
    cz = Hx / (a * dlam * cos_c) * (-2.0 / (a * cos_c * dlam))   # (H,)

    # Meridional: gy[k] = (dh[k-1]-dh[k])/(a dphi). dh[j] appears in gy[j] (-) and gy[j+1] (+).
    # Fy_cos[k] = Hv[k]*cos_v[k]*gy[k]; dFy=(Fy_cos[j+1]-Fy_cos[j])/(a dphi).
    # div merid = -dFy/cos_c.  Coefficient of dh[j]:
    coeff_gyjp1 = Hv[1:H + 1] * cos_v[1:H + 1] * (1.0 / (a * dphi))   # from gy[j+1] (+)
    coeff_gyj = Hv[0:H] * cos_v[0:H] * (-1.0 / (a * dphi))           # from gy[j]   (-)
    dFy_coeff = (coeff_gyjp1 - coeff_gyj) / (a * dphi)               # (H,)
    cm = -dFy_coeff / cos_c                                          # (H,)

    D = 1.0 - alpha * (cz + cm)                                      # (H,)
    return D[:, None] * np.ones((1, W))


def helmholtz_sor(
    rhs: np.ndarray,
    H_ref_lat: np.ndarray,
    gp: float,
    theta: float,
    dt: float,
    g: Grid,
    n_iters: int,
    sor_omega: float,
    dh0: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Fixed-count red/black SOR for L_sym dh = rhs.

    L_sym = helmholtz_apply (symmetric SPD).  Each sweep updates red cells
    ((i+j)%2==0) then black cells, recomputing the residual between colors:

        dh += sor_omega * (rhs - helmholtz_apply(dh)) / D   (restricted to color)

    where D is the analytic diagonal.  No early-out — exactly n_iters sweeps for
    bit-reproducible determinism.  Starts from zeros unless dh0 is supplied
    (warm start).

    Parameters
    ----------
    rhs : ndarray (H, W)
    H_ref_lat : ndarray (H,)
    gp, theta, dt : floats
    g : Grid
    n_iters : int — number of red/black sweeps.
    sor_omega : float — over-relaxation factor in (0, 2).
    dh0 : ndarray (H, W), optional — warm start (default zeros).

    Returns
    -------
    ndarray (H, W)
    """
    H, W = g.H, g.W
    dh = np.zeros((H, W)) if dh0 is None else dh0.copy()
    D = _helmholtz_diagonal(H_ref_lat, gp, theta, dt, g)

    jj, ii = np.meshgrid(np.arange(H), np.arange(W), indexing="ij")
    red = ((ii + jj) % 2 == 0)
    black = ~red

    for _ in range(n_iters):
        for color in (red, black):
            resid = rhs - helmholtz_apply(dh, H_ref_lat, gp, theta, dt, g)
            dh = dh + sor_omega * np.where(color, resid / D, 0.0)
    return dh


def helmholtz_residual_per_lat(
    dh: np.ndarray,
    rhs: np.ndarray,
    H_ref_lat: np.ndarray,
    gp: float,
    theta: float,
    dt: float,
    g: Grid,
) -> np.ndarray:
    """Per-latitude L2 norm of the Helmholtz residual (L_sym dh - rhs).

    Returns shape (H,), including pole rows.  Used to gate polar convergence.
    """
    resid = helmholtz_apply(dh, H_ref_lat, gp, theta, dt, g) - rhs
    return np.sqrt(np.sum(resid * resid, axis=1))


def helmholtz_solve_exact(
    rhs: np.ndarray,
    H_ref_lat: np.ndarray,
    gp: float,
    theta: float,
    dt: float,
    g: Grid,
) -> np.ndarray:
    """Direct sparse solve of L_sym dh = rhs (independent ground truth).

    Assembles L_sym as a sparse matrix by applying helmholtz_apply to unit
    basis vectors, then spsolve.  Used to certify the SOR fixed point.
    """
    import scipy.sparse as sp
    import scipy.sparse.linalg as spla

    H, W = g.H, g.W
    N = H * W
    rows, cols, vals = [], [], []
    for k in range(N):
        e = np.zeros(N)
        e[k] = 1.0
        col = helmholtz_apply(e.reshape(H, W), H_ref_lat, gp, theta, dt, g).ravel()
        nz = np.nonzero(col)[0]
        for r in nz:
            rows.append(r)
            cols.append(k)
            vals.append(col[r])
    A = sp.csr_matrix((vals, (rows, cols)), shape=(N, N))
    dh = spla.spsolve(A, rhs.ravel())
    return dh.reshape(H, W)


def reference_depth(h: np.ndarray) -> np.ndarray:
    """Reference layer depth H_ref as a frozen latitude-only profile.

    Williamson-2 height is latitude-dependent, so the per-latitude zonal mean
    is the natural choice that makes the linearized gravity-wave operator
    consistent with the steady state (a single global mean would mis-balance
    the meridional pressure structure).

        H_ref_lat[j] = mean_i h[j, i]

    Returns shape (H,).
    """
    return h.mean(axis=1)


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

    # Coriolis sandwich (f only; ζ already applied above).
    # NOTE: this block is byte-identical to coriolis_sandwich(). It is kept
    # inline (not a call) to guarantee the explicit path's byte-identity. If you
    # change the Coriolis sequence, update coriolis_sandwich AND this copy
    # together, then re-run test_coriolis_sandwich_matches_momentum.
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
# M2-T4: semi-implicit step (Picard-Coriolis, SOR Helmholtz)
# ---------------------------------------------------------------------------

def _semi_implicit_predictor(
    h: np.ndarray, u: np.ndarray, v: np.ndarray,
    gp: float, g: Grid, dt: float, theta: float = 0.5,
) -> tuple[np.ndarray, np.ndarray]:
    """Explicit predictor (u_star, v_star): advection + KE + (1-theta) pressure.

    Mirrors momentum_step's vector-invariant advection and the kinetic-energy
    part of the Bernoulli potential, and adds ONLY the (1-theta) EXPLICIT half of
    the pressure gradient -(1-theta)*dt*g'*grad(h^n).  Coriolis is omitted (it is
    applied later in velocity_backsub).

    The remaining theta half of the pressure gradient is carried IMPLICITLY by
    velocity_backsub acting on the FULL solved height h^{n+1} = h^n + dh:

        u^{n+1} = C( u_star - theta*dt*g'*grad(h^n + dh) )

    so the total pressure kick on the velocity is the time-centered
    -(1-theta)*dt*g'*grad(h^n) - theta*dt*g'*grad(h^{n+1}).  This is the textbook
    theta-scheme split:

      - At a balanced steady state (Williamson-2), dh -> 0 and the kick collapses
        to -dt*g'*grad(h^n), the full explicit pressure that balances Coriolis, so
        the velocity is stationary (matches momentum_step exactly).
      - For a gravity wave about a flat resting layer, grad(h^n) = 0 and the kick
        is purely the implicit -theta*dt*g'*grad(h^{n+1}); the explicit
        gravity-wave velocity CFL is therefore REMOVED.

    The earlier variant baked the FULL explicit pressure g'h into the predictor
    Bernoulli; that retains the explicit gravity-wave kick (-dt*g'*grad(h^n)) and
    the wave blows up at ~2x the explicit CFL.  Carrying only the (1-theta) half
    here is what removes the CFL while keeping W2 stationary.
    """
    H, W = h.shape
    # NOTE: the vorticity-flux + KE-interpolation block below mirrors
    # momentum_step's vector-invariant advection (it omits only the g'h pressure
    # and the Coriolis sandwich). If you change momentum_step's advection stencil,
    # mirror it here AND re-run test_semi_implicit_reduces_to_m1_at_small_dt.
    zeta = vorticity(u, v, g)
    zeta_uf = corner_to_uface(zeta)
    zeta_vf = 0.5 * (zeta + np.roll(zeta, 1, axis=1))
    v_c = 0.5 * (v[0:H] + v[1:H + 1])
    v_at_uf = center_to_uface(v_c)
    u_c = 0.5 * (u + np.roll(u, 1, axis=1))
    u_at_vf = center_to_vface(u_c)
    ke = 0.5 * (u * u + v_c * v_c)
    gxk, gyk = grad_faces(ke, g)              # advective KE gradient
    gxn, gyn = grad_faces(h, g)               # background height gradient grad(h^n)
    # Advection + KE.
    u_adv = u + dt * (zeta_uf * v_at_uf - gxk)
    v_adv = np.zeros_like(v)
    v_adv[1:H] = v[1:H] + dt * (-zeta_vf[1:H] * u_at_vf[1:H] - gyk[1:H])
    # Subtract the (1-theta) EXPLICIT pressure half.
    u_star = u_adv - (1.0 - theta) * dt * gp * gxn
    v_star = v_adv.copy()
    v_star[1:H] = v_adv[1:H] - (1.0 - theta) * dt * gp * gyn[1:H]
    return u_star, v_star


def assert_positivity(h_raw: np.ndarray, h_floor: float) -> None:
    """Loud positivity guard shared by the CPU and GPU semi-implicit steps.

    continuity_step_conservative keeps h >= h_floor only in the donor-cell
    sub-CFL regime; a floor cell drained through BOTH faces by a divergent
    meridional velocity (meridional Courant >~ 1) can still dip below the floor.
    A subsequent np.maximum floor clamp would then SILENTLY inject mass
    (defeating mass closure), so we reject loudly instead — same hard-reject
    philosophy as the Picard contraction certificate.  The 1e-9 slack absorbs
    f64 round-off in the flux-form sums (h is O(1-10), round-off O(1e-15)).
    """
    h_min = float(h_raw.min())
    if h_min < h_floor - 1e-9:
        raise ValueError(
            f"semi-implicit positivity violation: min(h)={h_min:.3e} < "
            f"h_floor={h_floor:.3e}. The velocity field drove a floor cell below "
            f"the floor (meridional Courant too large); the conservative limiter "
            f"cannot keep mass closed here. Reduce dt or relax forcing."
        )


def step_semi_implicit(
    st: SwRefState,
    theta: float = 0.5,
    picard_iters: int = 3,
    poisson_iters: int = 200,
    sor_omega: float = 1.7,
    dh_warm: Optional[np.ndarray] = None,
) -> SwRefState:
    """One semi-implicit shallow-water step (textbook theta-scheme).

    The linear gravity-wave terms are theta-time-centered (theta=0.5 =
    Crank-Nicolson: unconditionally stable AND neutral), eliminating the explicit
    gravity-wave CFL.  We solve for the height INCREMENT dh = h^{n+1} - h^n; the
    implicit pressure acts on the FULL height h^{n+1} via velocity_backsub.

    Structure (see helmholtz_rhs for the term split):
      1. Predictor: advection + KE + (1-theta) explicit pressure half, no Coriolis.
      2. H_ref: frozen per-latitude zonal-mean depth (reference_depth); c^2=g'H_ref.
      3. Picard loop (deferred Coriolis): rebuild rhs at the current dh and
         re-solve the symmetric Helmholtz L_sym dh = rhs (L_sym = helmholtz_apply).
      4. Back-substitution: velocity_backsub applies the theta-implicit pressure
         gradient of the FULL height h^{n+1} = h^n + dh and the Coriolis sandwich.
      5. Final height: the matched theta-centered increment h^n + dh PLUS the
         explicit nonlinear/anomaly transport (FCT on the total h minus its linear
         reference-divergence part).  The theta-centered reference divergence is
         implicit (in dh) and supplies the unconditional stability; only the slow
         nonlinear anomaly transport is explicit.  The FCT is the CONSERVATIVE
         variant (continuity_step_conservative) so the anomaly is mass-neutral
         even when the velocity drives a near-floor cell to the floor (M2-T5).

    NOTE (W2 height drift): the matched theta-scheme has an intrinsic O((theta*dt)^2)
    steady-state height imbalance from the implicit pressure/Coriolis coupling on
    the background gradient (it vanishes as dt -> 0; see
    test_w2_geostrophic_stationary).  The velocity stays stationary to the
    explicit step's tolerance.  This is the price of making the FULL pressure
    implicit, which is what removes the gravity-wave CFL.

    Parameters
    ----------
    st : SwRefState
    theta : float — off-centering (0.5 default).
    picard_iters : int — fixed deferred-Coriolis iterations.
    poisson_iters : int — SOR sweeps per Helmholtz solve.
    sor_omega : float — SOR over-relaxation factor.
    dh_warm : ndarray (H, W), optional — warm-start increment (0 on first call).

    Returns
    -------
    SwRefState
    """
    g, gp, omega, dt = st.g, st.gp, st.omega, st.dt
    h, u, v = st.h, st.u, st.v
    H, W = h.shape

    # 1. Predictor (advection + KE + (1-theta) explicit pressure half, no Coriolis).
    u_star, v_star = _semi_implicit_predictor(h, u, v, gp, g, dt, theta)

    # 2. Frozen reference depth (latitude-only zonal mean).
    H_ref_lat = reference_depth(h)

    # 3. Picard loop with deferred Coriolis; SOR Helmholtz inner solve for dh.
    dh = np.zeros((H, W)) if dh_warm is None else dh_warm.copy()
    for _ in range(picard_iters):
        rhs = helmholtz_rhs(h, u, v, u_star, v_star, dh,
                            H_ref_lat, gp, omega, theta, dt, g)
        dh = helmholtz_sor(rhs, H_ref_lat, gp, theta, dt, g,
                           poisson_iters, sor_omega, dh0=dh)

    # 4. Back-substitution: implicit pressure of the FULL height h^{n+1} = h + dh.
    u_new, v_new = velocity_backsub(u_star, v_star, h + dh, gp, theta, dt, omega, g)

    # 5. Final height: matched theta-centered increment + explicit nonlinear anomaly.
    #    h_new = h + dh + anomaly, anomaly = FCT(total h, u_new) - linear ref-div(u_new).
    #    The theta-centered reference divergence is IMPLICIT (in dh) and supplies
    #    the unconditional gravity-wave stability; only the slow nonlinear anomaly
    #    transport is explicit.  This composition is what removes the gravity-wave
    #    CFL — replacing it with pure explicit FCT reinstates the CFL (the FCT
    #    transport is explicit and blows up at large dt).
    #
    #    CONSERVATION (M2-T5): this sum is exactly mass-conserving iff each piece
    #    integrates conservatively.  div(dh)-free?  Yes: the SOR Helmholtz operator
    #    is flux-form so Σ dh·cosφ = Σ rhs-mass = 0 to round-off.  The anomaly:
    #    Σ h_linref·cosφ = Σ h·cosφ (divergence is flux-form, sums to zero), so
    #    Σ anomaly·cosφ = Σ h_fct·cosφ - Σ h·cosφ.  This is zero ONLY if h_fct
    #    conserves mass.  The M1 continuity_step does NOT when the velocity drives a
    #    near-floor cell below h_floor: its np.maximum floor clamp INJECTS mass
    #    (~5e-7/step once min(h) touches the floor — the original T5 leak).  We use
    #    continuity_step_conservative instead, which makes the floor clamp a no-op
    #    via conservative outflux limiting, so Σ h_fct·cosφ = Σ h·cosφ exactly and
    #    the anomaly is mass-neutral.  (M1 continuity_step is left byte-identical
    #    for GPU parity.)
    h_fct = continuity_step_conservative(h, u_new, v_new, g, dt, st.h_floor)
    h_linref = h - dt * divergence_helmholtz(u_new, v_new, H_ref_lat, g)
    anomaly = h_fct - h_linref
    h_raw = h + dh + anomaly
    assert_positivity(h_raw, st.h_floor)
    h_new = np.maximum(h_raw, st.h_floor)

    return SwRefState(
        g=g, gp=gp,
        h=h_new, u=u_new, v=v_new,
        dt=dt, omega=omega,
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
