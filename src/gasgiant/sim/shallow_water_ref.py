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

from dataclasses import dataclass

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
    H = g.H

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
    dh0: np.ndarray | None = None,
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


class PositivityViolation(ValueError):
    """A layer thickness dropped below the floor (semi-implicit positivity
    failure / 2-layer lower-layer outcrop). Subclasses ValueError so every
    existing `except ValueError` catcher keeps working; raised as a distinct
    type so callers that want to handle a *physical* breakdown (e.g. the
    baroclinic driver holding its last good state on outcrop) can do so WITHOUT
    also swallowing an unrelated ValueError from a real bug."""


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
        raise PositivityViolation(
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
    dh_warm: np.ndarray | None = None,
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


# ---------------------------------------------------------------------------
# M2: semi-Lagrangian advection — departure-point trajectory solver
# ---------------------------------------------------------------------------

def departure_points(u, v, dt, g, n_iter=2):
    """Back-trajectory departure points for arrival CELL CENTERS, in fractional
    grid-index space (i_dep zonal, j_dep meridional).  Row 0 = north, phi descending.

    Angular velocities: lam_dot = u/(a cosphi), phi_dot = v/a.  Index velocities:
      di/dt = lam_dot / dlam ;  dj/dt = -phi_dot / dphi   (j increases southward).
    Two-iteration implicit midpoint: evaluate the velocity at the current
    midpoint estimate, refine.  Velocities are sampled at cell centers by
    averaging the C-grid faces (u east+west, v north+south).
    """
    H, W = u.shape
    u_c = 0.5 * (u + np.roll(u, 1, axis=1))                 # (H,W) west+east face
    v_c = 0.5 * (v[0:H] + v[1:H + 1])                       # (H,W) north+south face
    cosphi = g.cos_c[:, None]                               # (H,1)
    di = (u_c / (g.a * cosphi)) * dt / g.dlam               # eastward => +i
    dj = -(v_c / g.a) * dt / g.dphi                         # northward (v>0) => -j (toward row 0)
    i_arr = np.arange(W)[None, :] + np.zeros((H, 1))        # (H,W) arrival i
    j_arr = np.arange(H)[:, None] + 0.5                     # (H,W) arrival j (center)
    a_i, a_j = di.copy(), dj.copy()
    for _ in range(n_iter):
        im = i_arr - 0.5 * a_i
        jm = j_arr - 0.5 * a_j
        a_i = _bilinear_periodic(di, im, jm, g)
        a_j = _bilinear_periodic(dj, im, jm, g)
    return i_arr - a_i, j_arr - a_j


def _bilinear_periodic(field, i_idx, j_idx, g):
    """Bilinear sample of a center field at fractional (i_idx zonal, j_idx-0.5
    row).  Zonal periodic wrap; meridional clamp to [0, H-1]."""
    H, W = field.shape
    jj = np.clip(j_idx - 0.5, 0.0, H - 1.0)
    j0 = np.floor(jj).astype(int); j1 = np.minimum(j0 + 1, H - 1)
    fy = jj - j0
    i0f = np.floor(i_idx); fx = i_idx - i0f
    i0 = i0f.astype(int) % W; i1 = (i0 + 1) % W
    f00 = field[j0, i0]; f10 = field[j0, i1]
    f01 = field[j1, i0]; f11 = field[j1, i1]
    return ((1 - fx) * (1 - fy) * f00 + fx * (1 - fy) * f10
            + (1 - fx) * fy * f01 + fx * fy * f11)


# ---------------------------------------------------------------------------
# M2-adv Task 2: conservative 1-D PPM remap (SLICE kernel)
# ---------------------------------------------------------------------------

def ppm_remap_1d_periodic(m, edges):
    """Conservative PPM remap of per-cell masses `m` (length n, periodic) onto a
    new set of cell edges `edges` (length n+1, fractional source-index
    coordinates; edges[k]..edges[k+1] is the k-th destination cell in the SOURCE
    grid).  Returns remapped per-cell masses (length n).  Conservative: Σ∫ = Σm."""
    n = len(m)
    mL = np.roll(m, 1); mR = np.roll(m, -1)
    aL_raw = (7.0 * (m + mL) - (mR + np.roll(m, 2))) / 12.0
    aL = _ppm_monotone_edge(aL_raw, mL, m)
    aR = np.roll(aL, -1)
    aL, aR = _ppm_limit_parabola(m, aL, aR)
    def integral(s, x0, x1):
        d = aR[s] - aL[s]
        c6 = 6.0 * (m[s] - 0.5 * (aL[s] + aR[s]))
        def F(x):
            return (aL[s] * x + 0.5 * d * x * x
                    + c6 * (0.5 * x * x - x * x * x / 3.0))
        return F(x1) - F(x0)
    out = np.empty(n)
    for k in range(n):
        out[k] = _accumulate_interval(edges[k], edges[k + 1], n, integral)
    return out

def _ppm_monotone_edge(aL_raw, mL, m):
    """Clamp the raw edge value into [min,max] of the two bounding cells."""
    lo = np.minimum(mL, m); hi = np.maximum(mL, m)
    return np.clip(aL_raw, lo, hi)

def _ppm_limit_parabola(m, aL, aR):
    """Colella-Woodward parabola limiter: kill overshoots / enforce monotonicity."""
    aL = aL.copy(); aR = aR.copy()
    d = aR - aL
    excess = d * (m - 0.5 * (aL + aR))
    d2 = d * d / 6.0
    flat = (aR - m) * (m - aL) <= 0.0
    over_l = excess > d2
    over_r = excess < -d2
    aL = np.where(flat, m, aL); aR = np.where(flat, m, aR)
    aL = np.where(~flat & over_l, 3.0 * m - 2.0 * aR, aL)
    aR = np.where(~flat & over_r, 3.0 * m - 2.0 * aL, aR)
    return aL, aR

def _accumulate_interval(x0, x1, n, integral):
    """Integrate the reconstructed density over [x0, x1] in periodic source
    coordinates, summing whole and partial source-cell contributions."""
    total = 0.0
    lo = x0
    s = int(np.floor(x0))
    while lo < x1 - 1e-15:
        s_lo = float(s)
        s_hi = s_lo + 1.0
        seg_hi = min(x1, s_hi)
        xi0 = lo - s_lo; xi1 = seg_hi - s_lo
        total += integral(s % n, xi0, xi1)
        lo = seg_hi
        s += 1
    return total

def slice_remap_advance(h, u, v, dt, g):
    """Conservative semi-Lagrangian advance of total h over dt by (u,v).
    Drop-in, advective-CFL-free replacement for continuity_step_conservative.
    Cascade: zonal 1-D conservative remap, then meridional. m = h*cosφ."""
    H, W = h.shape
    i_dep, j_dep = departure_points(u, v, dt, g, n_iter=2)
    cosc = g.cos_c[:, None]
    m = h * cosc

    m_zon = np.empty_like(m)
    for j in range(H):
        centers = i_dep[j]
        edges = np.empty(W + 1)
        edges[1:W] = 0.5 * (centers[0:W - 1] + centers[1:W])
        edges[0] = centers[0] - 0.5 * (centers[1] - centers[0])
        edges[W] = edges[0] + W
        # FIX (FATAL): monotonize crossed edges. Clamp interior edges into the
        # fixed periodic frame [edges[0], edges[W]] BEFORE accumulating so the
        # W-wide span (required for periodic mass conservation) is preserved.
        edges[1:W] = np.clip(edges[1:W], edges[0], edges[W])
        edges[1:W] = np.maximum.accumulate(edges[1:W])
        m_zon[j] = ppm_remap_1d_periodic(m[j], edges)

    m_out = np.empty_like(m_zon)
    for i in range(W):
        centers = j_dep[:, i]
        m_out[:, i] = _remap_1d_meridional(m_zon[:, i], centers, H)

    return m_out / cosc

def _remap_1d_meridional(m_col, centers, H):
    """Conservative 1-D remap on a NON-periodic column (poles are walls; no mass
    flux across φ=±π/2). Clamps edges to [0, H]."""
    c = centers - 0.5
    edges = np.empty(H + 1)
    edges[1:H] = 0.5 * (c[0:H - 1] + c[1:H])
    edges[0] = 0.0; edges[H] = float(H)
    edges = np.clip(edges, 0.0, float(H))
    edges = np.maximum.accumulate(edges)   # FIX (FATAL): same crossing guard
    return _ppm_remap_1d_clamped(m_col, edges)

def _ppm_remap_1d_clamped(m, edges):
    n = len(m)
    mL = np.concatenate([m[:1], m[:-1]])
    mR = np.concatenate([m[1:], m[-1:]])
    mLL = np.concatenate([m[:1], m[:1], m[:-2]])
    aL_raw = (7.0 * (m + mL) - (mR + mLL)) / 12.0
    aL = _ppm_monotone_edge(aL_raw, mL, m)
    aR = np.concatenate([aL[1:], aL[-1:]])
    aL, aR = _ppm_limit_parabola(m, aL, aR)
    def integral(s, x0, x1):
        d = aR[s] - aL[s]; c6 = 6.0 * (m[s] - 0.5 * (aL[s] + aR[s]))
        def F(x):
            return aL[s] * x + 0.5 * d * x * x + c6 * (0.5 * x * x - x ** 3 / 3.0)
        return F(x1) - F(x0)
    out = np.empty(n)
    for k in range(n):
        x0, x1 = edges[k], edges[k + 1]
        total = 0.0; lo = x0; s = min(int(np.floor(x0)), n - 1)
        while lo < x1 - 1e-15 and s < n:
            s_hi = s + 1.0; seg_hi = min(x1, s_hi)
            total += integral(s, lo - s, seg_hi - s)
            lo = seg_hi; s += 1
        out[k] = total
    return out


def sl_advect_velocity(q, u, v, dt, g, kind):
    """Semi-Lagrangian transport of a face field q by (u,v) over dt.
    kind="u": q at u-faces (H,W); kind="v": q at v-faces (H+1,W) with pole rows 0.
    Bicubic (Catmull-Rom) interpolation at the departure points."""
    H, W = u.shape
    i_dep, j_dep = departure_points(u, v, dt, g, n_iter=2)
    if kind == "u":
        return _bicubic_periodic(q, i_dep, j_dep, g)
    i_dep_vf = np.zeros((H + 1, W)); j_dep_vf = np.zeros((H + 1, W))
    i_dep_vf[1:H] = 0.5 * (i_dep[0:H - 1] + i_dep[1:H])
    j_dep_vf[1:H] = 0.5 * (j_dep[0:H - 1] + j_dep[1:H]) - 0.5
    out = _bicubic_periodic_vface(q, i_dep_vf, j_dep_vf, g)
    out[0] = 0.0; out[H] = 0.0
    return out


def _catmull_rom_w(t):
    """4-point Catmull-Rom weights for fractional offset t in [0,1)."""
    t2 = t * t; t3 = t2 * t
    return np.stack([
        -0.5 * t3 + t2 - 0.5 * t,
        1.5 * t3 - 2.5 * t2 + 1.0,
        -1.5 * t3 + 2.0 * t2 + 0.5 * t,
        0.5 * t3 - 0.5 * t2], axis=0)


def _bicubic_periodic(field, i_idx, j_idx, g):
    """Bicubic sample of a center-row field (H,W) at (i_idx zonal, j_idx center
    coord). Zonal periodic; meridional clamped at the poles."""
    H, W = field.shape
    jj = np.clip(j_idx - 0.5, 0.0, H - 1.0)
    j0 = np.floor(jj).astype(int); fy = jj - j0
    i0 = np.floor(i_idx).astype(int); fx = i_idx - i0
    wx = _catmull_rom_w(fx); wy = _catmull_rom_w(fy)
    acc = np.zeros_like(i_idx)
    for dj in range(-1, 3):
        jr = np.clip(j0 + dj, 0, H - 1)
        row = np.zeros_like(i_idx)
        for di in range(-1, 3):
            ic = (i0 + di) % W
            row = row + wx[di + 1] * field[jr, ic]
        acc = acc + wy[dj + 1] * row
    return acc


def _bicubic_periodic_vface(field, i_idx, j_idx, g):
    """Bicubic sample of a v-face field (H+1,W) at (i_idx zonal, j_idx v-face row
    coord). Zonal periodic; meridional clamped to [0, H]."""
    Hp1, W = field.shape
    H = Hp1 - 1
    jj = np.clip(j_idx, 0.0, float(H))
    j0 = np.floor(jj).astype(int); fy = jj - j0
    i0 = np.floor(i_idx).astype(int); fx = i_idx - i0
    wx = _catmull_rom_w(fx); wy = _catmull_rom_w(fy)
    acc = np.zeros_like(i_idx)
    for dj in range(-1, 3):
        jr = np.clip(j0 + dj, 0, H)
        row = np.zeros_like(i_idx)
        for di in range(-1, 3):
            ic = (i0 + di) % W
            row = row + wx[di + 1] * field[jr, ic]
        acc = acc + wy[dj + 1] * row
    return acc


def sl_momentum_predictor(h, u, v, gp, g, dt, theta):
    """SL replacement for _semi_implicit_predictor: SL parcel transport of (u,v)
    PLUS the unchanged KE-gradient and (1-theta) explicit pressure half, no Coriolis."""
    H, W = h.shape
    u_sl = sl_advect_velocity(u, u, v, dt, g, kind="u")
    v_sl = sl_advect_velocity(v, u, v, dt, g, kind="v")
    v_c = 0.5 * (v[0:H] + v[1:H + 1])
    ke = 0.5 * (u * u + v_c * v_c)
    gxk, gyk = grad_faces(ke, g)
    gxn, gyn = grad_faces(h, g)
    c = 1.0 - theta
    u_star = u_sl - dt * (gxk + c * gp * gxn)
    v_star = v.copy() * 0.0
    v_star[1:H] = v_sl[1:H] - dt * (gyk[1:H] + c * gp * gyn[1:H])
    return u_star, v_star


# ---------------------------------------------------------------------------
# M2-adv crux: semi-Lagrangian semi-implicit (SLSI) step + fast-jet config
# ---------------------------------------------------------------------------

def step_slsi(st, theta=0.5, picard_iters=3, poisson_iters=200,
              sor_omega=1.7, dh_warm=None):
    """Semi-Lagrangian semi-implicit step: M2-core's SI core with the two explicit
    Eulerian transport operators replaced by SL equivalents (advective-CFL-free).
      site #1  _semi_implicit_predictor -> sl_momentum_predictor (SL momentum)
      site #2  continuity_step_conservative -> slice_remap_advance (SLICE remap)
    The Helmholtz solve (dh), Picard-Coriolis, and velocity_backsub are reused
    verbatim, preserving M2-core's gravity-wave CFL removal."""
    g, gp, omega, dt = st.g, st.gp, st.omega, st.dt
    h, u, v = st.h, st.u, st.v
    H, W = h.shape

    # 1. SL predictor (SL momentum transport, no Coriolis).
    u_star, v_star = sl_momentum_predictor(h, u, v, gp, g, dt, theta)

    # 2-4. Reuse M2-core verbatim: H_ref, Picard Helmholtz dh, back-substitution.
    H_ref_lat = reference_depth(h)
    dh = np.zeros((H, W)) if dh_warm is None else dh_warm.copy()
    for _ in range(picard_iters):
        rhs = helmholtz_rhs(h, u, v, u_star, v_star, dh,
                            H_ref_lat, gp, omega, theta, dt, g)
        dh = helmholtz_sor(rhs, H_ref_lat, gp, theta, dt, g,
                           poisson_iters, sor_omega, dh0=dh)
    u_new, v_new = velocity_backsub(u_star, v_star, h + dh, gp, theta, dt, omega, g)

    # 5. SL nonlinear anomaly via SLICE conservative remap. The reference part MUST
    #    be removed in the SAME conservative-remap form as h_sl (NOT the Eulerian
    #    divergence_helmholtz), or it fails to cancel at Courant>>1 and double-counts
    #    against dh. Remap the broadcast H_ref by the same trajectory and subtract.
    H_ref_field = np.broadcast_to(H_ref_lat[:, None], h.shape)
    h_sl = slice_remap_advance(h, u_new, v_new, dt, g)
    href_sl = slice_remap_advance(H_ref_field, u_new, v_new, dt, g)
    anomaly = (h_sl - h) - (href_sl - H_ref_field)
    h_raw = h + dh + anomaly
    assert_positivity(h_raw, st.h_floor)
    h_new = np.maximum(h_raw, st.h_floor)

    return SwRefState(g=g, gp=gp, h=h_new, u=u_new, v=v_new,
                      dt=dt, omega=omega,
                      u_init=st.u_init, v_init=st.v_init, h_floor=st.h_floor)


def fast_jet_state(W=128, H=64, a=6.4e6, u0=120.0, dt_mult=1, C_base=0.5, m_wave=4, amp=0.02):
    """Barotropically STABLE fast zonal flow for the M2-adv advective-CFL gate:
    solid-body rotation u0*cosφ (an exact, Rayleigh-stable SWE steady state) plus a
    small zonal wavenumber-m height perturbation that advects with the flow.  Unlike
    a narrow Gaussian jet, solid-body rotation has no inflection point, so the
    reference run survives and the gate measures ADVECTION ACCURACY (not a physical
    barotropic instability).  dt is set by the advective Courant: for solid body the
    Courant u/(a cosφ dλ)*dt = u0/(a dλ)*dt is latitude-independent = C_base*dt_mult."""
    st = williamson2_state(W=W, H=H, a=a, omega=7.292e-5, u0=u0, gp=9.8, h0=8000.0)
    g = st.g
    lam = np.arange(W) * g.dlam
    phi = g.phi_c
    env = np.exp(-((phi - np.deg2rad(45.0)) / np.deg2rad(20.0)) ** 2)[:, None]  # broad, smooth
    h = st.h + amp * 8000.0 * env * np.cos(m_wave * lam)[None, :]
    dt = dt_mult * C_base * (a * g.dlam) / u0
    return SwRefState(g=g, gp=st.gp, h=h, u=st.u, v=st.v, dt=dt, omega=st.omega,
                      u_init=st.u_init, v_init=st.v_init, h_floor=st.h_floor)


def montgomery_2layer(h1, h2, gp1, gp2):
    """Reduced-gravity Montgomery potentials for the 2-layer stack (design §2.2).
    M1 = gp1*(h1+h2); M2 = gp1*(h1+h2) + gp2*h2. A potential (no metric, a-agnostic);
    the planetary radius a enters only via grad_faces when -grad(M) is taken."""
    eta1 = h1 + h2
    M1 = gp1 * eta1
    M2 = gp1 * eta1 + gp2 * h2
    return M1, M2


# ---------------------------------------------------------------------------
# M3-T2: Montgomery-driven momentum step
# ---------------------------------------------------------------------------

def momentum_step_M(
    h: np.ndarray, u: np.ndarray, v: np.ndarray,
    M: np.ndarray, omega: float, g: Grid, dt: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Vector-invariant momentum update driven by a precomputed Montgomery
    potential M (Bernoulli B = M + ke), generalizing M1's momentum_step.

    This is a verbatim copy of momentum_step's body with the single pressure
    line changed from `B = gp * h + ke` to `B = M + ke`. With M = gp*h it is
    byte-identical to momentum_step (see test_momentum_step_M_reduces_to_m1).

    NOTE: `h` is now VESTIGIAL -- it is used only for its shape (H, W); the
    pressure forcing comes entirely from M. NOTE: the vorticity-flux block and
    the Coriolis sandwich below are SHARED logic with momentum_step (and with
    sw_spike/solver.py::_layer_momentum) and MUST be kept in sync. If you change
    the explicit advection or the Coriolis sequence here, update momentum_step,
    coriolis_sandwich, and this copy together, then re-run the reduction and
    decoupled tests.
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

    # Bernoulli potential B = M + ½(u² + v_c²) at centers (M replaces g'h).
    ke = 0.5 * (u * u + v_c * v_c)
    B = M + ke
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
# M3-T3: 2-layer prognostic state + explicit step
# ---------------------------------------------------------------------------

@dataclass
class Sw2State:
    g: Grid
    omega: float
    gp1: float
    gp2: float
    h1: np.ndarray; u1: np.ndarray; v1: np.ndarray
    h2: np.ndarray; u2: np.ndarray; v2: np.ndarray
    dt: float
    h_floor: float = 1.0
    # forcing fields (off by default; filled by Task 4)
    tau_rad: float = 0.0
    tau_drag: float = 0.0
    nu4: float = 0.0
    sponge_rate: float = 0.0
    h_eq1: np.ndarray | None = None
    h_eq2: np.ndarray | None = None


def layer_mass(st):
    """Per-layer global mass: Sum h_i * cos_c * a^2 * dlam * dphi."""
    g = st.g; w = g.cos_c[:, None] * g.a * g.a * g.dlam * g.dphi
    return float(np.sum(st.h1 * w)), float(np.sum(st.h2 * w))


def total_energy_2layer(st) -> float:
    """Global 2-layer total energy diagnostic (cos-weighted area integral).

    Per layer i: kinetic ½ h_i (u_i² + v_c_i²) plus Montgomery potential energy
    ½ M_i h_i, where M_i is the reduced-gravity Montgomery potential from
    montgomery_2layer (M1 = gp1·η1, M2 = gp1·η1 + gp2·h2).  Using ½ M_i h_i (rather
    than a single-layer ½ g' h²) is the natural stacked-layer PE: it reduces to the
    single-layer ½ gp1 h1² when h2→0 (then M1→gp1·h1).  Summed over both layers and
    integrated with cosφ · a² dλ dφ.  Diagnostic only (not a conserved invariant of
    the discrete scheme); used to confirm the energy budget is finite and positive.
    """
    g = st.g
    H = g.H
    M1, M2 = montgomery_2layer(st.h1, st.h2, st.gp1, st.gp2)
    v1_c = 0.5 * (st.v1[0:H] + st.v1[1:H + 1])
    v2_c = 0.5 * (st.v2[0:H] + st.v2[1:H + 1])
    ke1 = 0.5 * st.h1 * (st.u1 * st.u1 + v1_c * v1_c)
    ke2 = 0.5 * st.h2 * (st.u2 * st.u2 + v2_c * v2_c)
    pe1 = 0.5 * M1 * st.h1
    pe2 = 0.5 * M2 * st.h2
    e_density = (ke1 + ke2 + pe1 + pe2) * g.cos_c[:, None]
    return float(np.sum(e_density) * g.a * g.a * g.dlam * g.dphi)


def _biharmonic(field: np.ndarray) -> np.ndarray:
    """Grid-normalized ∇⁴ proxy: iterated 5-point Laplacian on the lon-lat grid.

    Index-space (no metric), a-agnostic — ports as-is from the M0 spike.
    """
    def lap(a):
        return (np.roll(a, 1, 1) + np.roll(a, -1, 1)
                + np.roll(a, 1, 0) + np.roll(a, -1, 0) - 4 * a)
    return lap(lap(field))


def _smoothstep(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, 0.0, 1.0)
    return x * x * (3.0 - 2.0 * x)


def _polar_sponge(phi: np.ndarray, lat0=np.radians(65.0), lat1=np.radians(85.0)) -> np.ndarray:  # noqa: B008
    """Ramp 0->1 poleward of lat0; used to relax velocity->0 and h->h_eq near poles."""
    return _smoothstep((np.abs(phi) - lat0) / (lat1 - lat0))


def apply_forcing(st):
    """2-layer forcing (M3 Task 4), ported from the M0 spike `_apply_forcing`.

    All forcing is STEP-based (tau in steps), dt-independent — the explicit
    gravity-wave dt is tiny so time-based timescales would be infeasible. Each
    term is guarded on its field being active; with all fields off it is a no-op.

    Order: (1) thermal relaxation both layers; (2) Rayleigh bottom drag on the
    LOWER layer only; (3) biharmonic hyperviscosity (v1.6 /64 grid-norm) on
    u1,u2; (4) polar sponge with st.sponge_rate; (5) positivity floor.
    """
    g = st.g
    # (1) Thermal (mass) relaxation toward h_eq (per-step fraction 1/tau_rad).
    if st.tau_rad > 0.0 and st.h_eq1 is not None:
        st.h1 = st.h1 + (st.h_eq1 - st.h1) / st.tau_rad
        st.h2 = st.h2 + (st.h_eq2 - st.h2) / st.tau_rad
    # (2) Rayleigh bottom drag on the lower layer only (per-step).
    if st.tau_drag > 0.0:
        st.u2 = st.u2 * (1.0 - 1.0 / st.tau_drag)
        st.v2 = st.v2 * (1.0 - 1.0 / st.tau_drag)
    # (3) Grid-normalized biharmonic hyperviscosity on velocity (v1.6 lesson: /64).
    if st.nu4 > 0.0:
        st.u1 = st.u1 - (st.nu4 / 64.0) * _biharmonic(st.u1)
        st.u2 = st.u2 - (st.nu4 / 64.0) * _biharmonic(st.u2)
    # (4) Polar sponge: relax velocity->0 and h->h_eq poleward. rate is the
    #     Sw2State field (M3 difference from the spike's hardcoded 0.5); when
    #     sponge_rate == 0.0 the sponge is a no-op.
    if st.sponge_rate > 0.0:
        rate = st.sponge_rate
        sc = _polar_sponge(g.phi_c)[:, None]    # (H,1) at centers
        sv = _polar_sponge(g.phi_v)[:, None]    # (H+1,1) at v-faces
        st.u1 = st.u1 * (1.0 - rate * sc)
        st.u2 = st.u2 * (1.0 - rate * sc)
        st.v1 = st.v1 * (1.0 - rate * sv)
        st.v2 = st.v2 * (1.0 - rate * sv)
        if st.h_eq1 is not None:
            st.h1 = st.h1 + rate * sc * (st.h_eq1 - st.h1)
            st.h2 = st.h2 + rate * sc * (st.h_eq2 - st.h2)
    # (5) Positivity floor.
    st.h1 = np.maximum(st.h1, st.h_floor)
    st.h2 = np.maximum(st.h2, st.h_floor)
    return None


# ---------------------------------------------------------------------------
# M3-T5: balanced 2-layer init + h_eq profiles + Montgomery balance gate
# ---------------------------------------------------------------------------

def heq_profiles(g: Grid, h0_1: float = 5000.0, h0_2: float = 5000.0,
                 tilt1: float = 800.0, tilt2: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
    """Radiative-equilibrium thickness targets h_eq_i(phi), a-aware (latitude-only).

    Repurposed from sw_spike/init.py::h_eq_profile but stripped to a simple
    pole-to-equator tilt suitable for the M3 forcing (Task 6 swaps in the
    unstable banded tilt).  The tilt is a sin^2(phi) bump so the top layer is
    THICKER at the equator (warm) and thinner at the poles, like Williamson-2:

        h_eq1(phi) = h0_1 + tilt1 * (1 - sin^2 phi)  =  h0_1 + tilt1 * cos^2 phi
        h_eq2(phi) = h0_2 + tilt2 * cos^2 phi

    Returns (h_eq1, h_eq2), each shape (H, W) (broadcast uniform in lambda).
    """
    cos2 = (g.cos_c ** 2)[:, None] * np.ones((1, g.W))   # (H, W)
    h_eq1 = h0_1 + tilt1 * cos2
    h_eq2 = h0_2 + tilt2 * cos2
    return h_eq1, h_eq2


def balanced_2layer_state(
    W: int, H: int, a: float,
    omega: float, gp1: float, gp2: float, u0: float,
    h0_1: float = 25000.0, h0_2: float = 25000.0,
    dt_safety: float = 0.3, h_floor: float = 1.0,
) -> Sw2State:
    """Geostrophically-balanced 2-layer steady state (Williamson-2 generalization).

    The top layer carries a solid-body zonal jet u1 = u0 cosφ (v1 = 0); the lower
    layer is quiescent (u2 = v2 = 0).  Both layers are in balance, which PINS the
    Montgomery coupling (M1 = gp1*eta1, M2 = gp1*eta1 + gp2*h2):

    Balance derivation (design §2.2)
    --------------------------------
    The momentum step forms the Bernoulli potential B_i = M_i + ke_i and balances
    -grad(B_i) against Coriolis.  For zonal flow the binding (meridional) balance
    is  (1/a) dB_i/dφ = -f u_i,  f = 2Ω sinφ.

    TOP LAYER (u1 = u0 cosφ, ke1 = ½u1²):  B1 = gp1*eta1 + ½u0²cos²φ.
        (1/a) d/dφ[gp1 eta1 + ½u0²cos²φ] = -f u0 cosφ
    Integrating exactly as Williamson-2 (the u0²/2 KE term folds in identically):
        eta1(φ) = eta1_0 - (a Ω u0 + ½u0²) sin²φ / gp1.
    (Check: d/dφ of the RHS = -(aΩu0+½u0²)(2 sinφ cosφ)/gp1; multiply by gp1/a and
     add the KE-gradient (1/a)d(½u0²cos²φ)/dφ = -(u0²/a)sinφcosφ; total
     = -(2Ωu0+u0²/a)sinφcosφ·... ⇒ -f u1 exactly, same algebra as williamson2_state.)

    LOWER LAYER (u2 = 0, ke2 = 0):  B2 = M2 = gp1 eta1 + gp2 h2.  Quiescent balance
    needs (1/a) dM2/dφ = 0 ⇒ M2 = const ⇒
        h2(φ) = (M2_const - gp1 eta1(φ)) / gp2.
    Choosing M2_const so that mean(h2) = h0_2 fixes the lower-layer thickness:
        h2 = h0_2 + gp1/gp2 * (mean(eta1) - eta1(φ)).
    Then h1 = eta1 - h2.  BOTH layers are balanced; if the Montgomery sign or the
    gp1/gp2 coupling is wrong the state drifts/blows up (the balance gate).

    The default reference thicknesses h0_1 = h0_2 = 25000 are deliberately LARGE:
    the lower-layer interface tilt is amplified by gp1/gp2 (≈33 here), so h2 swings
    by ~gp1/gp2 times the eta1 tilt and h1 = eta1 - h2 swings oppositely.  The mean
    thicknesses must dominate that swing to keep both layers strictly positive
    (no floor clipping — clipping would break the balance and the gate would fail).

    dt uses the PRODUCTION a-aware polar CFL with the barotropic external-mode
    speed c_gw = sqrt(gp1*(h1+h2).max()) (mirrors williamson2_state's dx_min).
    """
    g = Grid(W, H, a)

    cos_c = g.cos_c[:, None] * np.ones((1, W))            # (H, W)
    sin_c = np.sin(g.phi_c)[:, None] * np.ones((1, W))    # (H, W)

    # Top-layer solid-body zonal jet (Williamson-2), v1 = 0.
    u1 = u0 * cos_c
    v1 = np.zeros((H + 1, W))

    # Balanced eta1 = h1 + h2 (full gradient-wind incl. KE, as williamson2_state).
    # eta1_0 is a free offset; set so mean(eta1) lands near h0_1 + h0_2 (the
    # total column).  We pin the offset AFTER computing the shape.
    eta1_shape = -(a * omega * u0 + 0.5 * u0 * u0) * sin_c * sin_c / gp1   # (H, W)
    # cos-weighted mean of the shape (each latitude column is uniform in lambda):
    eta1_shape_mean = float((g.cos_c * eta1_shape[:, 0]).sum() / g.cos_c.sum())
    eta1_0 = (h0_1 + h0_2) - eta1_shape_mean
    eta1 = eta1_0 + eta1_shape                                            # (H, W)

    # Lower-layer balance: M2 = const ⇒ h2 = h0_2 + (gp1/gp2)*(mean(eta1) - eta1).
    eta1_mean = float((g.cos_c * eta1[:, 0]).sum() / g.cos_c.sum())
    h2 = h0_2 + (gp1 / gp2) * (eta1_mean - eta1)                          # (H, W)
    h1 = eta1 - h2                                                        # (H, W)

    h1 = np.maximum(h1, h_floor)
    h2 = np.maximum(h2, h_floor)
    u2 = np.zeros((H, W))
    v2 = np.zeros((H + 1, W))

    # Production a-aware polar CFL with the barotropic external-mode speed.
    c_gw = np.sqrt(gp1 * (h1 + h2).max())
    cos_min = max(g.cos_c.min(), 1e-6)
    dx_min = min(cos_min * g.a * g.dlam, g.a * g.dphi)
    dt = dt_safety * dx_min / c_gw

    return Sw2State(
        g=g, omega=omega, gp1=gp1, gp2=gp2,
        h1=h1.copy(), u1=u1.copy(), v1=v1.copy(),
        h2=h2.copy(), u2=u2.copy(), v2=v2.copy(),
        dt=dt, h_floor=h_floor,
    )


# ---------------------------------------------------------------------------
# M3-T6: baroclinic instability CRUX gate helpers
#
# These build a BALANCED 2-layer base with a controlled mid-latitude vertical
# shear, add a BALANCED interface perturbation at the f-plane Phillips K_max,
# and provide the eddy-variance diagnostic + the closed-form growth target.
#
# Design point (feasibility-tuned; see derivation in baroclinic_test_state):
#   a=6.4e6, gp1=0.5 (LOWERED from 9.8 to enlarge the explicit dt; the
#   baroclinic mode speed is set by gp2, not gp1, so this does not change the
#   instability — only the barotropic external-wave CFL, which sets dt), gp2=0.3,
#   H=25000, phi_test=45deg, zonal wavenumber m=5 (K = m/(a cos phi) ~ K_max).
#   This gives an e-fold of ~9000 steps (< the 20000 budget) with L_D resolved
#   (~1.4 cells) and K_max ~ 0.1*Nyquist (hyperviscosity-safe).
# ---------------------------------------------------------------------------

# Mid-latitude test band: shear + perturbation are confined here, away from the
# polar sponge (|phi|>65deg) and the equator. f-plane Phillips theory assumes f0
# roughly constant, so the band is centred on phi_test with a cos-bell envelope.
_PHI_TEST_DEG = 45.0
_BAND_HALFWIDTH_DEG = 25.0   # envelope -> 0 at phi_test +/- 25deg (i.e. 20..70deg)


def _band_envelope(phi: np.ndarray, phi0: float, halfwidth: float) -> np.ndarray:
    """Smooth cos^2 bell, 1 at phi0, 0 at phi0 +/- halfwidth (radians); 0 outside."""
    x = (phi - phi0) / halfwidth
    env = np.where(np.abs(x) < 1.0, np.cos(0.5 * np.pi * x) ** 2, 0.0)
    return env


def _balanced_sheared_base(
    W, H, a, omega, gp1, gp2, H1_mean, H2_mean,
    shear, phi_test_rad, band_hw_rad,
):
    """Construct a geostrophically-balanced 2-layer base with a LOCALIZED eastward
    vertical shear (U1-U2 = shear at the band centre), with bounded thicknesses.

    Design choice (bounded base): keep the top free surface FLAT, eta1=h1+h2=const,
    so the top layer is quiescent (u1=0, see below) and the vertical shear lives
    entirely in the LOWER layer (u2<0 => U1-U2 = -u2 > 0, eastward shear).  The
    interface tilt is LOCALIZED in the band (a Gaussian-bump derivative), so h2
    returns to its baseline on both sides and the thickness swing stays bounded --
    avoiding the unbounded hemispheric swing of a globally-balanced single-signed
    jet (which floors h1 and breaks balance).

    Balance (zonal flow, meridional momentum), B_i = M_i + ke_i:
        (1/a) dB_i/dphi = -f u_i,  f = 2*omega*sin(phi).
      Layer1, eta1 flat: gp1 d(eta1)/dphi = -a f u1  =>  u1 = 0.
      Layer2: d/dphi[gp1 eta1 + gp2 h2] = -a f u2  (eta1 flat, ke2 small)
              =>  u2 = -(gp2/(a f)) dh2/dphi.
    We PRESCRIBE dh2/dphi = A * bump(phi) (a localized Gaussian centred on the
    band), so u2(phi) = -(gp2/(a f)) A bump, and choose A so that the band-centre
    shear U1-U2 = -u2(phi_test) equals `shear`.  h2 = H2 + A*(cumint(bump) - mean).

    Charney-Stern (LOWER layer): for this eastward shear the lower-layer QG PV
    gradient beta2 = beta - (f0^2/(gp2*H2))*shear; supercritical shear drives it
    negative (the instability criterion).

    Returns (g, h1, h2, u1, v1, u2, v2) ready for Sw2State.
    """
    g = Grid(W, H, a)
    phi = g.phi_c                                   # (H,) descending
    f = 2.0 * omega * np.sin(phi)
    f_test = 2.0 * omega * np.sin(phi_test_rad)

    # Localized interface-slope bump (Gaussian); band_hw_rad ~ 2 sigma.
    sigma_phi = 0.5 * band_hw_rad
    bump = np.exp(-((phi - phi_test_rad) / sigma_phi) ** 2)   # (H,)

    # Size A so band-centre shear U1-U2 = -u2(phi_test) = (gp2/(a f_test)) A = shear.
    A = shear * a * f_test / gp2
    u2_prof = -(gp2 / (a * f)) * A * bump            # (H,)  (u1 = 0)

    # h2 from the localized slope: h2 = H2_mean + A*(cumint(bump) - cos-weighted mean).
    dphi_arr = np.diff(phi)
    cumint = np.concatenate([[0.0], np.cumsum(0.5 * (bump[:-1] + bump[1:]) * dphi_arr)])
    cumint = cumint - float((g.cos_c * cumint).sum() / g.cos_c.sum())
    h2_prof = H2_mean + A * cumint
    # Flat top free surface: eta1 = H1_mean + H2_mean (const) => h1 = eta1 - h2.
    h1_prof = (H1_mean + H2_mean) - h2_prof

    h1 = h1_prof[:, None] * np.ones((1, W))
    h2 = h2_prof[:, None] * np.ones((1, W))
    u1 = np.zeros((H, W))
    u2 = u2_prof[:, None] * np.ones((1, W))
    v1 = np.zeros((H + 1, W))
    v2 = np.zeros((H + 1, W))
    return g, h1, h2, u1, v1, u2, v2


def baroclinic_test_state(
    W, H, unstable, seed, a=6.4e6,
    omega=7.292e-5, gp1=0.5, gp2=0.3,
    H1_mean=12500.0, H2_mean=12500.0,
    m_zonal=5, pert_amp_frac=1e-3, dt_safety=0.3, h_floor=1.0,
    nu4=0.0, xi_unstable=2.0, xi_stable=0.5,
):
    """Balanced 2-layer base with a mid-latitude eastward shear + balanced
    interface perturbation at the f-plane Phillips K_max (M3 Task 6 Step 2).

    Charney-Stern (LOWER layer): for eastward shear (U1-U2)>0 the lower-layer QG
    PV gradient beta2 = beta - (f0^2/(gp2*H2))*(U1-U2) goes NEGATIVE when the
    shear is supercritical. The `unstable` flag toggles the supercriticality
    xi = (U1-U2)/U_crit, U_crit = beta*gp2*H2/f0^2 at phi_test:
       unstable=True  -> xi = xi_unstable (>1, beta2<0)
       unstable=False -> xi = xi_stable   (<1, both gradients positive)

    The shear is realized as a balanced banded jet (top faster than bottom) via
    _balanced_sheared_base (geostrophic/gradient-wind balance inverted from the
    prescribed u(phi)). A balanced interface perturbation at zonal wavenumber
    m_zonal (~K_max) is added: h2' = A*env*cos(m*lambda), with the perturbation
    velocity in geostrophic balance (u' = -(gp2/f0)(1/a) dh2'/dphi,
    v' = (gp2/f0)(1/(a cos)) dh2'/dlambda) so it does NOT radiate gravity waves
    (the M2-adv lesson).

    H = H1_mean + H2_mean is the total layer-mean depth used in L_D.
    """
    phi_test = np.radians(_PHI_TEST_DEG)
    band_hw = np.radians(_BAND_HALFWIDTH_DEG)
    f0 = 2.0 * omega * np.sin(phi_test)
    beta = 2.0 * omega * np.cos(phi_test) / a
    Htot = H1_mean + H2_mean

    # Critical shear & the requested supercriticality.
    U_crit = beta * gp2 * H2_mean / (f0 * f0)
    xi = xi_unstable if unstable else xi_stable
    shear = xi * U_crit          # = U1 - U2 (eastward) at the band centre

    g, h1, h2, u1, v1, u2, v2 = _balanced_sheared_base(
        W, H, a, omega, gp1, gp2, H1_mean, H2_mean,
        shear, phi_test, band_hw,
    )

    # --- Balanced interface perturbation seeding the f-plane Phillips K_max ---
    # Seed the single zonal mode m_zonal (~K_max) PLUS a small broadband interface
    # noise, both confined to the band and balanced (geostrophic perturbation
    # velocity).  The single mode gives the instability a clean target; the noise
    # lets the discrete growing normal mode self-select (a pure single-K interface
    # bump alone projects mostly onto neutral/decaying modes).  Both are tiny
    # (pert_amp_frac of mean h2) so the run stays linear for many e-foldings.
    rng = np.random.default_rng(seed)
    phase = rng.uniform(0.0, 2.0 * np.pi)            # seed-dependent phase
    lam = (np.arange(W) + 0.5) * g.dlam              # cell-center longitudes
    env = _band_envelope(g.phi_c, phi_test, band_hw)[:, None] * np.ones((1, W))
    h2_mean_local = float(h2.mean())
    A = pert_amp_frac * h2_mean_local
    cos_lam = np.cos(m_zonal * lam + phase)[None, :] * np.ones((H, 1))
    noise = rng.standard_normal((H, W))
    noise = noise - noise.mean(axis=1, keepdims=True)   # zero zonal mean (eddy only)
    h2_pert = env * (A * cos_lam + 0.5 * A * noise)     # (H, W)

    # Geostrophic perturbation velocity from the interface perturbation. Use the
    # reduced gravity gp2 and the LOCAL f (Coriolis). u' = -(gp2/(a f)) dh2'/dphi,
    # v' = (gp2/(a f cos)) dh2'/dlambda.  Built on the same C-grid faces as v.
    f_c = 2.0 * omega * np.sin(g.phi_c)
    f_safe = np.where(np.abs(f_c) < 1e-12, 1e-12, f_c)[:, None]
    # dh2'/dphi at centers (descending phi): use central difference.
    dh2_dphi = np.gradient(h2_pert, g.phi_c, axis=0)
    u2_pert = -(gp2 / (a * f_safe)) * dh2_dphi        # (H, W) at u-faces (approx)
    # dh2'/dlambda
    dh2_dlam = np.gradient(h2_pert, g.dlam, axis=1)
    v2c_pert = (gp2 / (a * f_safe * (g.cos_c[:, None] + 1e-30))) * dh2_dlam
    # scatter v perturbation to v-faces (interior only)
    v2_pert = np.zeros((H + 1, W))
    v2_pert[1:H] = 0.5 * (v2c_pert[0:H - 1] + v2c_pert[1:H])

    h2 = h2 + h2_pert
    u2 = u2 + u2_pert
    v2 = v2 + v2_pert
    # keep eta1 fixed so the perturbation lives in the interface: h1 -= h2_pert
    h1 = h1 - h2_pert

    h1 = np.maximum(h1, h_floor)
    h2 = np.maximum(h2, h_floor)

    # dt: production a-aware polar CFL with the barotropic external-mode speed.
    c_gw = np.sqrt(gp1 * (h1 + h2).max())
    cos_min = max(g.cos_c.min(), 1e-6)
    dx_min = min(cos_min * g.a * g.dlam, g.a * g.dphi)
    dt = dt_safety * dx_min / c_gw

    st = Sw2State(
        g=g, omega=omega, gp1=gp1, gp2=gp2,
        h1=h1.copy(), u1=u1.copy(), v1=v1.copy(),
        h2=h2.copy(), u2=u2.copy(), v2=v2.copy(),
        dt=dt, h_floor=h_floor, nu4=nu4,
    )
    # Stash diagnostics used by predicted_growth_rate_fplane / efold_steps_estimate.
    st._phi_test = phi_test
    st._shear = shear          # realized band-centre vertical shear U1-U2
    st._H_mean = Htot
    st._m_zonal = m_zonal
    st._xi = xi
    return st


def eddy_interface_var(st):
    """Variance of the eddy (non-zonal) interface height: var(h2 - zonal_mean(h2)).

    The zonal mean is removed so only the wave/eddy signal remains; this grows
    with the baroclinic mode and is NOT excited by gravity waves to leading order
    (unlike kinetic energy)."""
    zonal_mean = st.h2.mean(axis=1, keepdims=True)
    eddy = st.h2 - zonal_mean
    return float(np.var(eddy))


def predicted_growth_rate_fplane(st):
    """f-plane Phillips closed-form max growth rate (M3 Task 6 physics note 2).

    For a 2-layer reduced-gravity stack with deformation wavenumber
        k_d^2 = f0^2/gp2 * (1/H1 + 1/H2) = 4*f0^2/(gp2*H)  for H1=H2=H/2,
        L_D = 1/k_d,
    the f-plane growth rate is sigma(K) = U_s*K*sqrt((k_d^2-K^2)/(k_d^2+K^2)),
    maximized at K_max^2 = k_d^2*(sqrt2-1), giving
        sigma_max = U_s*k_d*sqrt(3-2*sqrt2) = 0.31*U_s*k_d = 0.31*U_s/L_D.
    Here U_s = (U1-U2)/2 (the peak in-band shear half), f0 = 2*omega*sin(phi_test),
    H = layer-mean total depth.  (sqrt(3-2*sqrt2) = sqrt2-1 = 0.4142..., and
    sqrt((sqrt2-1)) factor folds in; 0.31 is the standard Phillips coefficient.)

    CORRECTION (M3 finalize): the prior code cited the plan's
    `k_d^2 = 2*f0^2/(gp2*H)`, which is a factor sqrt(2) too low. The correct
    equal-layer (H1=H2=H/2) 2-layer QG deformation wavenumber is
    `k_d^2 = f0^2/gp2 * (1/H1 + 1/H2) = 4*f0^2/(gp2*H)`. Fixed here; sigma_max
    rises by sqrt(2), reducing the measured-vs-theory ratio accordingly.
    """
    f0 = 2.0 * st.omega * np.sin(st._phi_test)
    U_s = 0.5 * st._shear          # U_s = (U1-U2)/2, the band-centre shear half
    kd = np.sqrt(4.0 * f0 * f0 / (st.gp2 * st._H_mean))
    return float(0.31 * U_s * kd)


def efold_steps_estimate(st):
    """Estimated e-folding time in solver STEPS: 1/(sigma_max*dt). The explicit
    gravity-wave dt is tiny, so the run must span thousands of steps to see a few
    e-foldings; this sizes the run length in the gate."""
    sigma = predicted_growth_rate_fplane(st)
    if sigma <= 0.0:
        return 10 ** 9
    return int(np.ceil(1.0 / (sigma * st.dt)))


# ---------------------------------------------------------------------------
# M3-T6 Step 4: finite-amplitude vortex coherence helpers (gate d)
# ---------------------------------------------------------------------------

def vortex_test_state(
    W, H, seed, a=6.4e6, omega=7.292e-5, gp1=0.5, gp2=0.3,
    H1_mean=12500.0, H2_mean=12500.0, ro_target=0.3,
    dt_safety=0.3, h_floor=1.0,
):
    """A GRS-scale balanced anticyclonic vortex in the LOWER layer (Ro>0.1) for
    the finite-amplitude coherence check. The vortex is a localized interface
    depression in geostrophic/gradient-wind balance at phi_test; sized so the
    local Rossby number exceeds 0.1."""
    phi_test = np.radians(_PHI_TEST_DEG)
    f0 = 2.0 * omega * np.sin(phi_test)
    g = Grid(W, H, a)

    # Resting balanced base (no shear), then add a balanced Gaussian interface bump.
    g2, h1, h2, u1, v1, u2, v2 = _balanced_sheared_base(
        W, H, a, omega, gp1, gp2, H1_mean, H2_mean,
        0.0, phi_test, np.radians(_BAND_HALFWIDTH_DEG),
    )
    # Gaussian vortex centred at (phi_test, lambda_c) with radius L ~ deformation.
    L_D = np.sqrt(gp2 * (H1_mean + H2_mean)) / (f0 * np.sqrt(2.0))
    Lr = 2.0 * L_D                      # vortex radius
    lam = (np.arange(W) + 0.5) * g.dlam
    lam_c = np.pi
    # great-circle-ish local distance on the band (small-angle): dx=a cos*dlam, dy=a*dphi
    X = a * np.cos(g.phi_c)[:, None] * ((lam[None, :] - lam_c + np.pi) % (2 * np.pi) - np.pi)
    Y = a * (g.phi_c[:, None] - phi_test)
    r2 = (X * X + Y * Y) / (Lr * Lr)
    # Amplitude sized to reach the target Rossby: Ro ~ gp2*amp/(f0^2 Lr^2) (geostrophic).
    amp = ro_target * (f0 * f0) * (Lr * Lr) / gp2
    bump = -amp * np.exp(-r2)           # anticyclone: interface depression
    h2 = h2 + bump
    h1 = h1 - bump

    # Geostrophic vortex velocity from the bump (lower layer).
    f_c = 2.0 * omega * np.sin(g.phi_c)
    f_safe = np.where(np.abs(f_c) < 1e-12, 1e-12, f_c)[:, None]
    dbump_dphi = np.gradient(bump, g.phi_c, axis=0)
    dbump_dlam = np.gradient(bump, g.dlam, axis=1)
    u2 = u2 - (gp2 / (a * f_safe)) * dbump_dphi
    v2c = (gp2 / (a * f_safe * (g.cos_c[:, None] + 1e-30))) * dbump_dlam
    v2 = np.zeros((H + 1, W))
    v2[1:H] = 0.5 * (v2c[0:H - 1] + v2c[1:H])

    h1 = np.maximum(h1, h_floor)
    h2 = np.maximum(h2, h_floor)

    c_gw = np.sqrt(gp1 * (h1 + h2).max())
    cos_min = max(g.cos_c.min(), 1e-6)
    dx_min = min(cos_min * g.a * g.dlam, g.a * g.dphi)
    dt = dt_safety * dx_min / c_gw

    st = Sw2State(
        g=g, omega=omega, gp1=gp1, gp2=gp2,
        h1=h1.copy(), u1=u1.copy(), v1=v1.copy(),
        h2=h2.copy(), u2=u2.copy(), v2=v2.copy(),
        dt=dt, h_floor=h_floor,
    )
    st._phi_test = phi_test
    return st


def local_rossby_number(st):
    """Peak local Rossby number Ro = |zeta_2| / f0 of the lower-layer flow at the
    test latitude (a finite-amplitude vortex must have Ro>0.1 to be meaningful)."""
    f0 = 2.0 * st.omega * np.sin(getattr(st, "_phi_test", np.radians(_PHI_TEST_DEG)))
    zeta2 = vorticity(st.u2, st.v2, st.g)
    return float(np.abs(zeta2).max() / abs(f0))


def step_2layer(st):
    M1, M2 = montgomery_2layer(st.h1, st.h2, st.gp1, st.gp2)
    u1, v1 = momentum_step_M(st.h1, st.u1, st.v1, M1, st.omega, st.g, st.dt)
    u2, v2 = momentum_step_M(st.h2, st.u2, st.v2, M2, st.omega, st.g, st.dt)
    h1 = continuity_step_conservative(st.h1, u1, v1, st.g, st.dt, st.h_floor)
    h2 = continuity_step_conservative(st.h2, u2, v2, st.g, st.dt, st.h_floor)
    assert_positivity(h1, st.h_floor); assert_positivity(h2, st.h_floor)
    st.h1, st.u1, st.v1 = h1, u1, v1
    st.h2, st.u2, st.v2 = h2, u2, v2
    apply_forcing(st)
    return st
