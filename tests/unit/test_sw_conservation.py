"""Conservation gate for the conservative FCT continuity step.

Historical note: this file once held three additional gates for the M2
semi-implicit step (mass conservation, H_ref shape guard, positivity), removed
with the dead sw_gpu/semi-implicit solver family in the 2026-07 prune (see
docs/reviews/2026-07-02-comprehensive-review.md section 2.2).  The surviving
gate locks in the exact-conservation property of continuity_step_conservative,
which the live 2-layer baroclinic step (step_2layer) relies on.
"""
from __future__ import annotations

import numpy as np

from gasgiant.sim.shallow_water_ref import (
    Grid,
    continuity_step_conservative,
)


def test_continuity_conservative_exact_even_when_positivity_fails() -> None:
    """continuity_step_conservative is EXACTLY mass-conserving (flux-form) even in
    the out-of-regime case where its donor-cell positivity limiting cannot keep a
    floor cell non-negative.

    A floor cell straddled by a divergent meridional velocity (drained through
    both faces, donor of neither) can dip below the floor — but mass MUST stay
    closed to round-off, which is what step_2layer's loud positivity guard
    (assert_positivity) relies on (it rejects the sub-floor case rather than
    clamping, so a leak can never be silent). This locks in the
    exact-conservation property under stress.
    """
    g = Grid(W=4, H=3, a=1.0)
    h_floor = 0.05
    rng = np.random.default_rng(7)
    # A near-floor middle band with strong, sign-varying meridional velocity to
    # create divergence at floor cells; plus some zonal flow.
    h = np.full((g.H, g.W), 1.0)
    h[1, :] = h_floor * 1.02            # near-floor interior row
    u = 0.3 * rng.standard_normal((g.H, g.W))
    v = np.zeros((g.H + 1, g.W))
    # DIVERGENT meridional flow draining the near-floor row 1 through BOTH faces
    # (row 1 is the non-donor side at each), so the donor-cell limiter cannot
    # keep it >= floor — the documented out-of-regime case.
    v[1, :] = -4.0                      # north face of row 1: drains row 1 northward
    v[2, :] = +4.0                      # south face of row 1: drains row 1 southward
    dt = 0.5

    m0 = float(np.sum(h * g.cos_c[:, None]) * g.a * g.a * g.dlam * g.dphi)
    h_new = continuity_step_conservative(h, u, v, g, dt, h_floor)
    m1 = float(np.sum(h_new * g.cos_c[:, None]) * g.a * g.a * g.dlam * g.dphi)

    # Mass is conserved EXACTLY (flux-form telescoping) under strong divergent
    # drain at a near-floor cell — whether or not the donor-cell limiter manages
    # to keep this particular config >= floor. Exact conservation is the property
    # step_2layer's loud positivity guard relies on (it can reject a sub-floor
    # result knowing mass was never silently injected).
    assert abs(m1 - m0) <= 1e-12 * abs(m0), (
        f"continuity_step_conservative leaked mass under stress: "
        f"rel drift {(m1 - m0) / m0:.3e}"
    )
