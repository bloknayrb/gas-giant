"""Net-circulation renorm for storms.hero_flow_aspect (u_hero_flow_renorm).

The emergence ring/skirt in vortex_omega.glsl is authored on a K-widened
ellipse when hero_flow_aspect (K) > 1. On the tangent plane the EW stretch
scales the ring+skirt integral by exactly K, so 1/K would restore the net
circulation — but the SPHERICAL area element (orthographic coords x,y:
dA = dx dy / sqrt(1 - x^2 - y^2)) inflates the wide, far-reaching skirt more
than the compact ring, and since the net is the ~24% residual of a ~76%
cancellation the differential inflation amplifies: plain 1/K leaves a ~16%
net-circulation deficit at K=2 (warm scale) — the same magnitude the taper's
wedge deficit produced, which measurably shifted bands 25+ degrees away
through the global Poisson solve. So the renorm is computed here by direct
quadrature of the analytic ring+skirt profile on the sphere and wired as a
uniform; the shader applies it via the existing ``ring *= tcomp``.

The profile constants below MIRROR vortex_omega.glsl's ring/skirt smoothstep
windows and amplitudes; the blocks-agree unit test cross-pins both sides
(tests/unit/test_hero_shape_constants.py) so neither can drift alone.

Called with the ACTUAL hero r_core (seeded jitter is +/-20% around
storms.hero_radius and the curvature correction scales ~r_core^2, so the
authored mean would be up to ~7% off at K=2). Exact for the shipped
single-hero case; for multiple heroes the caller passes the mean r_core
(mean-field — heroes share aspect, and no shipped preset has hero_count > 1).
"""
from __future__ import annotations

import numpy as np

# vortex_omega.glsl mirror: (inner0, inner1, outer0, outer1, amplitude).
RING_WINDOW = (0.29, 0.55, 0.78, 1.04, -6.0)
SKIRT_WINDOW = (1.05, 1.35, 1.8, 2.4, 1.0)
_SUPPORT_Q = 2.4  # profile is exactly 0 beyond the skirt's outer edge


def _smoothstep(e0: float, e1: float, x: np.ndarray) -> np.ndarray:
    t = np.clip((x - e0) / (e1 - e0), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def _profile(qh: np.ndarray) -> np.ndarray:
    out = np.zeros_like(qh)
    for i0, i1, o0, o1, amp in (RING_WINDOW, SKIRT_WINDOW):
        out += amp * (_smoothstep(i0, i1, qh) - _smoothstep(o0, o1, qh))
    return out


def _net_circulation(r_core: float, aspect: float, k: float, n: int) -> float:
    """Integral of the ring+skirt profile over the sphere (orthographic
    tangent-plane quadrature; support is well inside the near hemisphere for
    every legal r_core/aspect/K, so no hemisphere clipping is needed)."""
    aspf = aspect * k
    hx = min(_SUPPORT_Q * aspf * r_core * 1.02, 0.99)
    hy = min(_SUPPORT_Q * r_core * 1.02, 0.99)
    x = np.linspace(-hx, hx, n)
    y = np.linspace(-hy, hy, n // 2 * 2 + 1)
    xx, yy = np.meshgrid(x, y)
    z2 = np.clip(1.0 - xx * xx - yy * yy, 1e-12, 1.0)
    qh = np.sqrt((xx / aspf) ** 2 + yy ** 2) / r_core
    integ = (_profile(qh) / np.sqrt(z2)).sum() * (x[1] - x[0]) * (y[1] - y[0])
    return float(integ)


def hero_flow_renorm(
    r_core: float, aspect: float, flow_aspect: float, n: int = 1601
) -> float:
    """Uniform amplitude factor that keeps the ring+skirt NET circulation
    invariant in K (the planet-scale moment must not move — taper lesson);
    the local cancellation FRACTION shifts only by the curvature differential
    (0.774 -> 0.807 at K=2, warm scale), which is the safe direction.
    Returns exactly 1.0 at K == 1 (the lever's off state never reaches the
    shader anyway — the uniform is consumed only inside the K != 1 branch)."""
    if flow_aspect == 1.0:
        return 1.0
    net1 = _net_circulation(r_core, aspect, 1.0, n)
    netk = _net_circulation(r_core, aspect, flow_aspect, n)
    return net1 / netk
