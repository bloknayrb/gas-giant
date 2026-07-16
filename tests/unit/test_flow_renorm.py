"""GL-free guards for sim/flow_renorm.py (u_hero_flow_renorm).

The renorm keeps the K-widened emergence ring/skirt's NET circulation
invariant (the planet-scale moment must not move — the taper's wedge deficit
of the same magnitude produced a measurable coherent band shift 25+ degrees
away). The GLSL<->numpy profile-mirror pin lives in
test_hero_shape_constants.py::test_flow_renorm_mirrors_ring_skirt_windows;
this file guards the quadrature itself: identity, convergence at the shipped
grid, agreement with an independently-constructed (polar) integral, and the
warm-scale regression values.
"""
from __future__ import annotations

import numpy as np

from gasgiant.sim.flow_renorm import _net_circulation, _profile, hero_flow_renorm

RC, ASP = 0.062, 2.2  # warm authored scale (the actual hero jitters +/-20%)


def test_identity_at_k1():
    assert hero_flow_renorm(RC, ASP, 1.0) == 1.0


def test_quadrature_converged_at_shipped_grid():
    a = hero_flow_renorm(RC, ASP, 2.0, n=801)
    b = hero_flow_renorm(RC, ASP, 2.0, n=3201)
    assert abs(a / b - 1.0) < 1e-3, (
        f"shipped-grid quadrature not converged: n=801 -> {a:.6f}, "
        f"n=3201 -> {b:.6f}"
    )


def _net_polar(rc: float, asp: float, k: float, nq: int = 4000,
               nt: int = 720) -> float:
    """Independent construction of the same integral: polar coordinates in
    the SQUASHED plane (x = aspf*rc*q*cos t, y = rc*q*sin t, Jacobian
    aspf*rc^2*q), against the Cartesian tangent-plane grid in flow_renorm."""
    aspf = asp * k
    q = np.linspace(0.0, 2.4, nq)[None, :]
    t = np.linspace(0.0, 2.0 * np.pi, nt, endpoint=False)[:, None]
    x = aspf * rc * q * np.cos(t)
    y = rc * q * np.sin(t)
    z = np.sqrt(np.clip(1.0 - x * x - y * y, 1e-12, 1.0))
    qg = np.broadcast_to(q, z.shape)
    dq = float(q[0, 1] - q[0, 0])
    dt = 2.0 * np.pi / nt
    return float((_profile(qg) / z * qg).sum() * dq * dt * aspf * rc * rc)


def test_matches_independent_polar_quadrature():
    for k in (1.0, 1.5, 2.0, 2.5):
        cart = _net_circulation(RC, ASP, k, n=1601)
        pol = _net_polar(RC, ASP, k)
        assert abs(cart / pol - 1.0) < 2e-3, (
            f"K={k}: cartesian {cart:.6e} vs polar {pol:.6e}"
        )


def test_warm_scale_regression_values():
    """Pinned loosely as a regression record. Plain 1/K would be 0.5 at K=2 —
    the 16% gap IS the curvature correction this module exists for. The
    renorm decreases monotonically in K, and renorm*K > 1 always (the sphere
    inflates the widened footprint, never shrinks it)."""
    assert abs(hero_flow_renorm(RC, ASP, 2.0) - 0.5805) < 0.003
    prev = 1.0
    for k in (1.25, 1.5, 2.0, 2.5):
        r = hero_flow_renorm(RC, ASP, k)
        assert r < prev, f"renorm not monotone at K={k}"
        assert r * k > 1.0, f"curvature correction inverted at K={k}"
        prev = r
