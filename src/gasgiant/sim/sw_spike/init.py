from __future__ import annotations
import numpy as np
from .grid import Grid
from .solver import SwState


def h_eq_profile(H, n_bands, band_contrast, h_mean):
    g = Grid(W=1, H=H)
    phi = g.phi_c
    heq = h_mean + band_contrast * np.cos(n_bands * phi) * np.cos(phi)
    return heq


def _gradient_wind_u(heq, g, f0, geff, phi_eq=np.radians(8.0)):
    """Invert balance for u. Geostrophic poleward; finite (capped) near equator."""
    phi = g.phi_c
    dheq_dphi = np.gradient(heq, phi)
    f = f0 * np.sin(phi)
    u = np.zeros_like(heq)
    far = np.abs(phi) >= phi_eq
    u[far] = -(geff * dheq_dphi[far]) / f[far]
    near = ~far
    u[near] = -(geff * dheq_dphi[near]) / (f0 * np.sin(np.sign(phi[near]) * phi_eq))
    u = np.clip(u, -3.0, 3.0)
    return u


def emergent_init(W, H, f0, gp, n_bands, band_contrast, h_mean=5.0) -> SwState:
    g = Grid(W, H)
    geff = gp[0]
    heq1 = h_eq_profile(H, n_bands, band_contrast, h_mean)
    heq2 = h_eq_profile(H, n_bands, 0.5 * band_contrast, h_mean * 0.6)
    u1p = _gradient_wind_u(heq1, g, f0, geff)
    rng = np.random.default_rng(1234)
    seed = 0.002 * rng.standard_normal((H, W))
    h1 = np.repeat(heq1[:, None], W, axis=1) + seed
    h2 = np.repeat(heq2[:, None], W, axis=1)
    u1 = np.repeat(u1p[:, None], W, axis=1)
    # dt MUST respect the MINIMUM grid spacing (polar zonal spacing collapses).
    c_gw = np.sqrt(geff * h1.max())
    dx_min = min(g.cos_c.min() * g.dlam, g.dphi)
    dt = 0.3 * dx_min / c_gw
    st = SwState(g=g, f0=f0, gp=gp,
                 h1=np.maximum(h1, 0.1), u1=u1, v1=np.zeros((H + 1, W)),
                 h2=np.maximum(h2, 0.1), u2=np.zeros((H, W)), v2=np.zeros((H + 1, W)),
                 dt=dt,
                 tau_rad=300.0, tau_drag=1000.0, nu4=0.2,
                 h_eq1=np.repeat(heq1[:, None], W, axis=1),
                 h_eq2=np.repeat(heq2[:, None], W, axis=1))
    return st
