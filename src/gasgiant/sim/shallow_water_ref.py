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
# Single-layer Montgomery potential and pressure gradient
# ---------------------------------------------------------------------------

def pressure_grad(h: np.ndarray, gp: float, g: Grid):
    """Single-layer Montgomery potential gradient.

    M = g' h  (reduced gravity times layer thickness).
    Returns (gx, gy) = grad_faces(M, g).
    """
    return grad_faces(gp * h, g)
