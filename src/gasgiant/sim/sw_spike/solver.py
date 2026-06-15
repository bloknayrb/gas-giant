from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from .grid import Grid, center_to_uface, center_to_vface
from . import operators as ops


@dataclass
class SwState:
    g: Grid
    f0: float
    gp: tuple[float, float]
    h1: np.ndarray; u1: np.ndarray; v1: np.ndarray
    h2: np.ndarray; u2: np.ndarray; v2: np.ndarray
    dt: float
    h_floor: float = 0.05
    nu4: float = 0.0
    tau_rad: float = 0.0
    tau_drag: float = 0.0
    h_eq1: np.ndarray | None = None
    h_eq2: np.ndarray | None = None


def _f_uface(g: Grid, f0: float) -> np.ndarray:
    return f0 * np.sin(g.phi_c)[:, None] * np.ones((1, g.W))


def _f_vface(g: Grid, f0: float) -> np.ndarray:
    return f0 * np.sin(g.phi_v)[:, None] * np.ones((1, g.W))


def _layer_momentum(h, u, v, M, f0, g, dt):
    """Vector-invariant momentum update for one layer (explicit advection+pressure,
    implicit trapezoidal Coriolis)."""
    zeta = ops.vorticity(u, v, g)                 # corners (H+1,W)
    zeta_uf = ops.corner_to_uface(zeta)           # (H,W) at u-faces
    v_c = 0.5 * (v[0:g.H] + v[1:g.H + 1])         # v at centers
    ke = 0.5 * (u * u + v_c * v_c)
    B = M + ke
    gx, gy = ops.grad_faces(B, g)                 # face gradients of Bernoulli
    f_uf = _f_uface(g, f0)
    q_uf = zeta_uf
    v_at_uf = center_to_uface(v_c)                # v interpolated to u-face
    u_star = u + dt * (q_uf * v_at_uf - gx)       # explicit advect+pressure (zonal)
    zeta_vf = 0.5 * (zeta + np.roll(zeta, 1, axis=1))  # corner->v-face (avg in lon)
    q_vf = zeta_vf
    u_c = 0.5 * (u + np.roll(u, 1, axis=1))       # u at centers
    u_at_vf = center_to_vface(u_c)
    v_star = v.copy()
    v_star[1:g.H] = (v[1:g.H]
                     + dt * (-(q_vf[1:g.H]) * u_at_vf[1:g.H] - gy[1:g.H]))
    u_new, v_c_new = ops.coriolis_trapezoidal(u_star, 0.5 * (v_star[0:g.H] + v_star[1:g.H + 1]),
                                              f_uf, dt)
    v_new = np.zeros_like(v); v_new[1:g.H] = 0.5 * (v_c_new[0:g.H - 1] + v_c_new[1:g.H])
    return u_new, v_new


def step(st: SwState, dt: float) -> SwState:
    g = st.g
    M1, M2 = ops.montgomery_2layer(st.h1, st.h2, st.gp)
    u1, v1 = _layer_momentum(st.h1, st.u1, st.v1, M1, st.f0, g, dt)
    u2, v2 = _layer_momentum(st.h2, st.u2, st.v2, M2, st.f0, g, dt)
    h1 = ops.continuity_step(st.h1, u1, v1, g, dt, st.h_floor)
    h2 = ops.continuity_step(st.h2, u2, v2, g, dt, st.h_floor)
    st.h1, st.u1, st.v1 = h1, u1, v1
    st.h2, st.u2, st.v2 = h2, u2, v2
    return st


def kinetic_energy(st: SwState) -> float:
    area = st.g.cos_c[:, None]
    vc1 = 0.5 * (st.v1[0:st.g.H] + st.v1[1:st.g.H + 1])
    vc2 = 0.5 * (st.v2[0:st.g.H] + st.v2[1:st.g.H + 1])
    ke = st.h1 * (st.u1 ** 2 + vc1 ** 2) + st.h2 * (st.u2 ** 2 + vc2 ** 2)
    return float(np.sum(0.5 * ke * area))


def checkerboard_amplitude(field: np.ndarray) -> float:
    jj, ii = np.indices(field.shape)
    sign = (-1.0) ** ((ii + jj) % 2)
    return float(np.abs(np.mean(field * sign)))


def balanced_test_state(W, H, f0, gp) -> SwState:
    """Geostrophically balanced zonal jet: choose h from u via gradient balance."""
    g = Grid(W, H)
    U = 0.3
    u_prof = U * g.cos_c
    geff = gp[0]
    dphi = g.dphi
    h_prof = np.zeros(H)
    f_prof = f0 * np.sin(g.phi_c)
    integrand = -(f_prof * u_prof) / geff
    h_prof[0] = 5.0
    for j in range(1, H):
        h_prof[j] = h_prof[j - 1] - integrand[j] * dphi
    h_prof -= h_prof.min() - 1.0
    h1 = np.repeat(h_prof[:, None], W, axis=1)
    u1 = np.repeat(u_prof[:, None], W, axis=1)
    v1 = np.zeros((H + 1, W))
    # dt MUST respect the MINIMUM grid spacing. Near the poles the zonal spacing
    # cos(phi)*dlam shrinks far below dphi, so the gravity-wave CFL is set by the
    # polar zonal spacing, NOT dphi. Using dphi alone gives a polar instability.
    c_gw = np.sqrt(geff * h1.max())
    dx_min = min(g.cos_c.min() * g.dlam, g.dphi)
    dt = 0.3 * dx_min / c_gw
    return SwState(g=g, f0=f0, gp=gp,
                   h1=h1, u1=u1, v1=v1,
                   h2=np.full((H, W), 3.0), u2=np.zeros((H, W)), v2=np.zeros((H + 1, W)),
                   dt=dt)
