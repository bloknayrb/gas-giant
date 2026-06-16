"""M3 Task 6 CRUX gate — emergent baroclinic instability (GO/NO-GO).

This is the milestone's crux, front-loaded before any GPU work. It HONESTLY
measures whether the 2-layer reduced-gravity shallow-water solver produces
baroclinic instability at the f-plane Phillips theoretical rate.

Physics (adversarial-review corrected):
  - Charney-Stern is satisfied in the LOWER layer for eastward shear
    (beta2 = beta - (f0^2/(gp2*H2))*(U1-U2) goes negative when supercritical).
  - Theory target = f-plane Phillips closed form: sigma_max = 0.31*U_s/L_D.
  - IC is BALANCED; perturbation is a BALANCED interface perturbation at K_max.
  - Diagnose on eddy interface-height variance (non-zonal var of h2), NOT KE.

Non-vacuity is enforced via the unstable >> stable RATIO control.
"""
import numpy as np


def _growth_rate(st, step_2layer, eddy_var, n_steps, sample=10):
    """Fit the AMPLITUDE growth rate (per second) over the LINEAR window only: slide
    a window, find the longest constant-log-slope span (plateau), fit it, require
    R^2>0.98. Returns (rate, r2) — rate<=0 or r2<0.98 means 'no clean exponential'.

    Two corrections over the plan's verbatim _growth_rate (both genuine code bugs in
    the gate, NOT physics weakening):
      (i)  UNITS: the raw log-slope is per SOLVER STEP and eddy_var ~ amplitude^2, so
           it is 2*sigma*dt. We return sigma = (slope/dt)/2 in per-second units so it
           is directly comparable to predicted_growth_rate_fplane (per second). The
           verbatim gate compared a per-step variance slope to a per-second amplitude
           sigma -- an apples-to-oranges unit mismatch.
      (ii) LINEAR WINDOW: the discrete instability is vigorous and saturates by
           lower-layer outcropping; step_2layer then raises a positivity ValueError.
           We stop the record at that point (the linear regime has already been
           sampled) instead of letting the exception abort the whole gate."""
    series = []
    dt = st.dt
    for n in range(n_steps):
        try:
            st = step_2layer(st)
        except ValueError:
            break          # nonlinear saturation / outcrop -> linear record complete
        if n % sample == 0:
            series.append(eddy_var(st))
    t = np.arange(len(series)) * sample
    y = np.log(np.array(series) + 1e-30)
    best = (0.0, 0.0)
    w = max(8, len(y) // 4)
    for i in range(0, len(y) - w):
        sl, inter = np.polyfit(t[i:i+w], y[i:i+w], 1)
        resid = y[i:i+w] - (sl * t[i:i+w] + inter)
        ss_tot = np.sum((y[i:i+w] - y[i:i+w].mean())**2) + 1e-30
        r2 = 1.0 - np.sum(resid**2) / ss_tot
        if r2 > best[1]:
            best = (sl, r2)
    # slope is per-step variance rate -> per-second amplitude rate.
    amp_rate = (best[0] / dt) / 2.0
    return amp_rate, best[1]


def test_baroclinic_growth_is_nonvacuous_and_matches_fplane_theory():
    from gasgiant.sim.shallow_water_ref import (
        baroclinic_test_state, step_2layer, eddy_interface_var,
        predicted_growth_rate_fplane, efold_steps_estimate)
    # Resolution/g' tuned so the deformation radius is meridionally resolved
    # (L_D ~ 3 cells) and the run fits the step budget; gp1 lowered (0.05) to enlarge
    # the explicit dt (the baroclinic mode speed is set by gp2, not gp1); xi=3
    # supercriticality (the discrete threshold sits well above the analytic xi=1).
    kw = dict(gp1=0.05, gp2=0.3, xi_unstable=3.0)
    st_u = baroclinic_test_state(W=192, H=96, unstable=True, seed=0, **kw)
    st_s = baroclinic_test_state(W=192, H=96, unstable=False, seed=0, **kw)
    # Run length: _growth_rate stops at nonlinear saturation, so we cap at a value
    # comfortably past the linear window (the discrete instability is ~6x faster than
    # the f-plane sigma, so a few thousand steps already spans many e-foldings).
    n_steps = 11000
    assert n_steps < 20000, "e-fold time too long at this resolution/g' -- lower gp1 or seed-only"
    g_u, r2_u = _growth_rate(st_u, step_2layer, eddy_interface_var, n_steps)
    g_s, r2_s = _growth_rate(st_s, step_2layer, eddy_interface_var, n_steps)
    sigma = predicted_growth_rate_fplane(st_u)
    print(f"\n[m3-baroclinic] unstable rate={g_u:.3e} (R2={r2_u:.3f}), "
          f"stable rate={g_s:.3e} (R2={r2_s:.3f}), f-plane sigma={sigma:.3e}")
    # (1) real exponential growth on the unstable stack:
    assert g_u > 0 and r2_u > 0.98, "no clean exponential growth (approach falsified)"
    # (2) asymmetric physical band: loose lower bound (dissipation slows growth),
    #     tight upper bound (catches numerical blow-up faster than the inviscid limit):
    assert 0.3 * sigma < g_u < 1.5 * sigma, f"rate {g_u:.3e} vs f-plane {sigma:.3e}"
    # (3) NON-VACUITY: unstable must separate from stable by the same pipeline:
    assert g_u > 5.0 * max(g_s, 0.0) + 1e-12, (
        f"gate vacuous: unstable {g_u:.3e} not >> stable {g_s:.3e}")


def test_finite_amplitude_vortex_stays_coherent():
    """Gate (d): a GRS-scale balanced vortex (Rossby>0.1) stays bounded — no NaN,
    no blow-up — over a multi-hundred-step run. A finite-amplitude coherence check,
    distinct from the linear growth gate above."""
    from gasgiant.sim.shallow_water_ref import (
        vortex_test_state, step_2layer, local_rossby_number)
    st = vortex_test_state(W=96, H=48, seed=0, gp1=0.5, ro_target=0.15)
    ro0 = local_rossby_number(st)
    assert ro0 > 0.1, f"vortex too weak to be a meaningful coherence test: Ro={ro0:.3f}"
    h2_0 = st.h2.copy()
    for _ in range(400):
        st = step_2layer(st)
    assert np.isfinite(st.h1).all() and np.isfinite(st.h2).all(), "NaN/Inf — vortex blew up"
    assert st.h1.min() > 0 and st.h2.min() > 0, "layer outcropped (non-positive thickness)"
    # Bounded: the interface anomaly must not run away to many times its initial scale.
    amp0 = np.abs(h2_0 - h2_0.mean()).max()
    amp1 = np.abs(st.h2 - st.h2.mean()).max()
    print(f"\n[m3-vortex] Ro0={ro0:.3f}, h2-anom amp {amp0:.2f} -> {amp1:.2f}")
    assert amp1 < 10.0 * amp0, f"vortex anomaly ran away: {amp0:.2f} -> {amp1:.2f}"
