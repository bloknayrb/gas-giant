from __future__ import annotations
import numpy as np
from .grid import Grid, center_to_vface


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
    dFx = (Fx - np.roll(Fx, 1, axis=1)) / g.dlam

    # Meridional term: h interpolated to v-faces, weighted by cosφ.
    h_vf = center_to_vface(h)            # (H+1, W)
    Fy = h_vf * v                        # meridional mass flux at v-faces
    cos_v = g.cos_v[:, None]             # (H+1, 1)
    Fy_c = Fy * cos_v                    # h v cosφ
    # North face of row j is index j; south face is index j+1 (φ decreases with j).
    dFy = (Fy_c[0:H] - Fy_c[1:H + 1]) / g.dphi

    inv_metric = 1.0 / g.cos_c[:, None]  # (H, 1)
    return inv_metric * (dFx + dFy)
