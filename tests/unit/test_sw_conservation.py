"""M2-T5: Conservation and positivity gates for the semi-implicit shallow-water step.

Three hard gates:

1. test_mass_conserved_si
   Fast-jet near-floor IC (|u| >> W2 u0, min(h) within ~10-20% of h_floor).
   Run 30 steps of step_semi_implicit and assert UN-RENORMALIZED total_mass
   drift <= 1e-12 * M0 (round-off, not a physical tolerance).

2. test_href_must_be_lat_only (de-no-op guard)
   Prove gate 1 is not vacuous by showing that a 2-D H_ref_lat injected into
   divergence_helmholtz (which enforces shape (H,)) raises ValueError — i.e.
   the shape check is load-bearing and a future regression that accidentally
   passes a 2-D array WILL trip.
   Separately, verify reference_depth returns shape (H,) (latitude-only) on
   non-uniform h.

3. test_no_new_negative_h
   Same IC, run 30 steps, assert min(h) >= h_floor at every step (no new
   negatives, FCT floor is respected).
"""
from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from gasgiant.sim.shallow_water_ref import (
    Grid,
    SwRefState,
    continuity_step_conservative,
    divergence_helmholtz,
    reference_depth,
    step_semi_implicit,
    total_mass,
    williamson2_state,
)

# ---------------------------------------------------------------------------
# Helpers / shared IC construction
# ---------------------------------------------------------------------------

def _fast_jet_near_floor_state(seed: int = 42) -> SwRefState:
    """Build a fast-jet near-floor initial state.

    Design choices:
    - Grid: 48 x 24 (coarse for test speed, fine enough for FCT stresses).
    - Planetary radius a=1.0 (dimensionless), omega=2.0.
    - Base W2 state: u0=1.5 (already large), gp=1.0, h0=5.0.
    - On top of W2: add small-amplitude white-noise zonal-jet perturbation
      so u is zonally non-uniform (stresses the vorticity flux and FCT).
    - Depress h locally near the south pole so min(h) is within ~15% of
      h_floor (= 0.05 default): target min(h) ~ 0.05 * 1.15 = 0.0575.
    - dt kept at the W2 CFL (fast jets -> small dt -> test is still fast for
      30 steps).
    """
    rng = np.random.default_rng(seed)
    W, H = 48, 24
    # u0=1.5 gives jets 7.5x the usual W2 u0=0.2 (strong zonal jets).
    st = williamson2_state(W=W, H=H, a=1.0, omega=2.0, u0=1.5, gp=1.0, h0=5.0)

    # Add ~2% white-noise to u so the flow is not a trivial zonally symmetric steady state.
    u_perturbed = st.u + 0.03 * rng.standard_normal(st.u.shape)

    # Depress h in the bottom (southernmost) ~3 rows so min(h) ~ 0.057.
    # h_floor = 0.05 by default; we target ~15% above floor.
    h_perturbed = st.h.copy()
    h_perturbed[-3:, :] = np.maximum(st.h_floor * 1.15, st.h_floor * 1.15
                                     + 0.01 * rng.standard_normal((3, W)))
    # Ensure the depression is truly near-floor (not accidentally lifted).
    h_perturbed[-1, :] = st.h_floor * 1.14   # pin the absolute minimum row

    return dataclasses.replace(st, h=h_perturbed, u=u_perturbed,
                               u_init=u_perturbed.copy(),
                               v_init=st.v.copy())


# Shared parameters (fewer SOR iters to keep test suite fast; 100 is sufficient
# for the coarse grid used here with modest gravity-wave stiffness).
_N_STEPS     = 30
_POISSON_ITS = 100
_SOR_OMEGA   = 1.7


# ---------------------------------------------------------------------------
# Gate 1: mass conservation
# ---------------------------------------------------------------------------

def test_mass_conserved_si() -> None:
    """UN-RENORMALIZED total_mass drift over 30 SI steps must be <= rtol 1e-12.

    The fast-jet near-floor IC stresses mass conservation:
    - Large velocities excite strong FCT limiting (anti-diffusive flux clamping).
    - Near-floor h values engage the FCT positivity limiter (h_low = max(h_low, h_floor)).
    - The floor clamp adds mass; the test verifies the SI step keeps the total
      drift at machine-epsilon level, not at a limiter-induced drift level.

    If this gate fails the mass drift number will be >> 1e-12 (likely O(1e-8)
    or larger if there is a real mass leak) — do NOT widen the tolerance.
    """
    st = _fast_jet_near_floor_state(seed=42)
    M0 = total_mass(st)

    # Confirm the IC has the intended near-floor h (gate the gate itself).
    assert st.h.min() < st.h_floor * 1.25, (
        f"IC min(h)={st.h.min():.4f} is not within 25% of h_floor={st.h_floor}; "
        "IC construction may be wrong"
    )

    max_drift = 0.0
    for _step_idx in range(_N_STEPS):
        st = step_semi_implicit(st, poisson_iters=_POISSON_ITS, sor_omega=_SOR_OMEGA)
        M = total_mass(st)
        drift = abs(M - M0) / abs(M0)
        if drift > max_drift:
            max_drift = drift

    assert max_drift <= 1e-12, (
        f"Total mass drift rtol = {max_drift:.3e} exceeds round-off gate 1e-12 "
        f"after {_N_STEPS} semi-implicit steps (M0={M0:.10g})"
    )


# ---------------------------------------------------------------------------
# Gate 2: de-no-op guard — H_ref latitude-only is load-bearing
# ---------------------------------------------------------------------------

def test_href_must_be_lat_only() -> None:
    """Prove that divergence_helmholtz enforces H_ref_lat shape (H,) — not (H, W).

    Sub-test A: reference_depth returns shape (H,) on a non-uniform h field.
    Sub-test B: passing a 2-D array of shape (H, W) into divergence_helmholtz
                raises ValueError — the shape check is load-bearing so any
                future regression that accidentally passes a 2-D H_ref will trip.

    Why this is sufficient: divergence_helmholtz is the only place H_ref_lat
    is consumed (both directly and via helmholtz_apply -> helmholtz_sor ->
    step_semi_implicit).  The ValueError guard at its entry means a 2-D
    H_ref cannot silently pass through the SI step without error; a future
    regression must also suppress the check to go undetected.
    """
    rng = np.random.default_rng(7)
    W, H = 16, 8
    g = Grid(W=W, H=H, a=1.0)

    # Non-uniform h so reference_depth is genuinely computing a per-latitude mean.
    h = 3.0 + 0.5 * np.cos(g.phi_c[:, None]) + 0.1 * rng.standard_normal((H, W))

    # Sub-test A: reference_depth must return a (H,) array.
    H_ref_lat = reference_depth(h)
    assert H_ref_lat.shape == (H,), (
        f"reference_depth returned shape {H_ref_lat.shape}; expected ({H},) "
        "(latitude-only profile)"
    )
    # Must be strictly positive (h is well above zero).
    assert np.all(H_ref_lat > 0), "reference_depth must be strictly positive for h > 0"
    # Must equal the row-wise mean.
    np.testing.assert_allclose(
        H_ref_lat, h.mean(axis=1),
        rtol=1e-14,
        err_msg="reference_depth must equal the zonal mean h.mean(axis=1)",
    )

    # Sub-test B: injecting a 2-D H_ref into divergence_helmholtz must raise ValueError.
    H_ref_2d = np.tile(H_ref_lat[:, None], (1, W))   # spurious zonal structure
    assert H_ref_2d.shape == (H, W), "sanity: H_ref_2d is 2-D"

    Fx = rng.standard_normal((H, W))
    Fy = np.zeros((H + 1, W))

    with pytest.raises(ValueError, match=r"H_ref_lat must be shape"):
        divergence_helmholtz(Fx, Fy, H_ref_2d, g)


# ---------------------------------------------------------------------------
# Gate 3: no new negative h (positivity)
# ---------------------------------------------------------------------------

def test_no_new_negative_h() -> None:
    """min(h) >= h_floor at every step over a 30-step fast-jet near-floor run.

    The FCT continuity_step floors at h_floor (0.05 default); step_semi_implicit
    also applies np.maximum(h + dh + anomaly, h_floor) as a final guard.  This
    test asserts that guard is effective: no cell ever goes below h_floor,
    even when starting from an IC with h dipping to ~1.14 * h_floor.

    If this gate fails, report the step index and the min(h) value — do NOT
    weaken the threshold.
    """
    st = _fast_jet_near_floor_state(seed=42)
    h_floor = st.h_floor

    # Confirm IC itself satisfies the gate.
    assert st.h.min() >= h_floor, (
        f"IC already violates positivity: min(h)={st.h.min():.6f} < h_floor={h_floor}"
    )

    min_h_over_run = float(st.h.min())
    first_violation_step = None

    for step_idx in range(1, _N_STEPS + 1):
        st = step_semi_implicit(st, poisson_iters=_POISSON_ITS, sor_omega=_SOR_OMEGA)
        step_min = float(st.h.min())
        if step_min < min_h_over_run:
            min_h_over_run = step_min
        if step_min < h_floor and first_violation_step is None:
            first_violation_step = step_idx

    assert first_violation_step is None, (
        f"Positivity violated at step {first_violation_step}: "
        f"min(h)={min_h_over_run:.6e} < h_floor={h_floor} "
        f"(over {_N_STEPS} steps)"
    )
    # Informational: record the run minimum (used in commit report).
    # min_h_over_run >= h_floor by the assertion above.


def test_continuity_conservative_exact_even_when_positivity_fails() -> None:
    """continuity_step_conservative is EXACTLY mass-conserving (flux-form) even in
    the out-of-regime case where its donor-cell positivity limiting cannot keep a
    floor cell non-negative.

    A floor cell straddled by a divergent meridional velocity (drained through
    both faces, donor of neither) can dip below the floor — but mass MUST stay
    closed to round-off, which is what step_semi_implicit's loud positivity guard
    relies on (it rejects the sub-floor case rather than clamping, so a leak can
    never be silent). This locks in the exact-conservation property under stress.
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
    # step_semi_implicit's loud positivity guard relies on (it can reject a
    # sub-floor result knowing mass was never silently injected).
    assert abs(m1 - m0) <= 1e-12 * abs(m0), (
        f"continuity_step_conservative leaked mass under stress: "
        f"rel drift {(m1 - m0) / m0:.3e}"
    )
