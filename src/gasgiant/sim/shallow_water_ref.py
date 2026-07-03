"""2-layer C-grid shallow-water CPU reference implementation.

The explicit-operator 2-layer baroclinic path that powers the shipped opt-in
baroclinic coupling (``sim/baroclinic_driver.py`` -> ``engine/facade.py``):
C-grid operators, positivity-limited conservative FCT continuity,
Montgomery-driven vector-invariant momentum, 2-layer forcing, and the
balanced baroclinic-instability test states + growth diagnostics.

History: graduated from the M0 sw_spike CPU spike with planetary radius `a`
threaded through every metric site.  The single-layer Williamson-2 states,
the Helmholtz semi-implicit (M2) machinery, and the falsified SLSI/PPM
advection family were removed in the 2026-07 dead-solver prune together with
the sw_gpu/sw_gpu_probe GPU scaffold they served as test oracle for (see
docs/reviews/2026-07-02-comprehensive-review.md section 2.2; the milestone
verdicts in docs/superpowers/specs/ are the historical record).

Metric sites carrying `a`:
  - grad_faces:      zonal  1/(a cosφ dλ),  meridional 1/(a dφ)
  - vorticity:       prefactor 1/(a cosφ)
  - _apply_fluxes:   prefactor 1/(a cosφ), fluxes divided by a dλ / a dφ

At a=1.0 all factors collapse to the M0 spike values (1/1 == 1), so
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
    """Apply flux divergence to update h: (1/(a cosφ))[∂Fx/∂λ + ∂(Fy cosφ)/∂φ].

    (a) sole site: outer 1/(a cosφ) prefactor.
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
    h_floor.  A naive fix would clamp such cells UP to the floor with np.maximum,
    which INJECTS mass (non-conservative — the historical M2-T5 leak).

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

    Zalesak anti-diffusive limiting where the LOW-ORDER base is first made
    positivity-preserving by per-face outflux limiting (_positive_lowflux_scales).
    Mass is conserved to round-off UNCONDITIONALLY (every scaled face flux is
    shared identically by its two adjacent cells, so the flux-form divergence
    telescopes exactly).

    Positivity is preserved only in the DONOR-CELL sub-CFL regime: the limiter
    scales each face by its upwind cell's available mass, so a cell drained through
    a face for which it is NOT the donor (e.g. a near-floor cell straddled by a
    divergent meridional velocity at Courant >~ 1) can still dip below h_floor.
    The caller (step_2layer via assert_positivity) guards this with a loud
    positivity check rather than a silent floor clamp, so an out-of-regime config
    fails instead of leaking mass.  Within the validated regime the result is
    >= h_floor and no clamp fires.
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


def momentum_step(
    h: np.ndarray, u: np.ndarray, v: np.ndarray,
    gp: float, omega: float, g: Grid, dt: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Vector-invariant momentum update for a single layer (B = g'h + ke).

    Advection+pressure are explicit; Coriolis is implicit (trapezoidal).
    Only relative vorticity ζ enters the flux cross term; the full
    Coriolis f = 2Ω sinφ is applied separately via coriolis_trapezoidal.

    Kept as the reduction baseline for momentum_step_M (with M = gp*h the two
    are byte-identical; see test_momentum_step_M_reduces_to_m1).
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
    # NOTE: this block is byte-identical to the one in momentum_step_M. If you
    # change the Coriolis sequence, update both copies together, then re-run
    # test_momentum_step_M_reduces_to_m1.
    f_uf = 2.0 * omega * np.sin(g.phi_c)[:, None] * np.ones((1, W))    # (H, W)

    # Trapezoidal Coriolis rotation on (u_star, v_star collapsed to centers).
    v_star_c = 0.5 * (v_star[0:H] + v_star[1:H + 1])                   # (H, W)
    u_new, v_c_new = coriolis_trapezoidal(u_star, v_star_c, f_uf, dt)

    # Scatter v_c_new back to v-faces: interior faces = avg of adjacent center rows.
    v_new = np.zeros_like(v)
    v_new[1:H] = 0.5 * (v_c_new[0:H - 1] + v_c_new[1:H])

    return u_new, v_new


class PositivityViolation(ValueError):
    """A layer thickness dropped below the floor (semi-implicit positivity
    failure / 2-layer lower-layer outcrop). Subclasses ValueError so every
    existing `except ValueError` catcher keeps working; raised as a distinct
    type so callers that want to handle a *physical* breakdown (e.g. the
    baroclinic driver holding its last good state on outcrop) can do so WITHOUT
    also swallowing an unrelated ValueError from a real bug."""


def assert_positivity(h_raw: np.ndarray, h_floor: float) -> None:
    """Loud positivity guard for the conservative continuity step (step_2layer).

    continuity_step_conservative keeps h >= h_floor only in the donor-cell
    sub-CFL regime; a floor cell drained through BOTH faces by a divergent
    meridional velocity (meridional Courant >~ 1) can still dip below the floor.
    A subsequent np.maximum floor clamp would then SILENTLY inject mass
    (defeating mass closure), so we reject loudly instead.  The 1e-9 slack
    absorbs f64 round-off in the flux-form sums (h is O(1-10), round-off
    O(1e-15)).
    """
    h_min = float(h_raw.min())
    if h_min < h_floor - 1e-9:
        raise PositivityViolation(
            f"semi-implicit positivity violation: min(h)={h_min:.3e} < "
            f"h_floor={h_floor:.3e}. The velocity field drove a floor cell below "
            f"the floor (meridional Courant too large); the conservative limiter "
            f"cannot keep mass closed here. Reduce dt or relax forcing."
        )


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
    the Coriolis sandwich below are SHARED logic with momentum_step and MUST be
    kept in sync. If you change the explicit advection or the Coriolis sequence
    here, update momentum_step and this copy together, then re-run the
    reduction test.
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
    # NOTE: this block is byte-identical to the one in momentum_step. It is kept
    # inline (not a call) to guarantee the explicit path's byte-identity. If you
    # change the Coriolis sequence, update both copies together, then re-run
    # test_momentum_step_M_reduces_to_m1.
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
     = -(2Ωu0+u0²/a)sinφcosφ·... ⇒ -f u1 exactly, the standard Williamson-2 algebra.)

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
    speed c_gw = sqrt(gp1*(h1+h2).max()) (Williamson-2-style dx_min).
    """
    g = Grid(W, H, a)

    cos_c = g.cos_c[:, None] * np.ones((1, W))            # (H, W)
    sin_c = np.sin(g.phi_c)[:, None] * np.ones((1, W))    # (H, W)

    # Top-layer solid-body zonal jet (Williamson-2), v1 = 0.
    u1 = u0 * cos_c
    v1 = np.zeros((H + 1, W))

    # Balanced eta1 = h1 + h2 (full gradient-wind incl. KE, Williamson-2 style).
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
