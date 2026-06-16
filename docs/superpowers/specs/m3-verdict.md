# M3 Task 6 (baroclinic CRUX) Verdict — MECHANISM VALIDATED, f-PLANE QUANTITATIVE MATCH FAILED

**Status (Task 6 gate):** **PARTIAL — emergent baroclinic instability is REAL and
non-vacuous, but the f-plane Phillips quantitative band assertion FAILS** (the discrete
growth rate is ~6x the plan's f-plane closed form; ~3x even after correcting a factor-√2
error in the plan's `k_d` formula). Recorded honestly per the project's non-vacuity / no-
forced-pass discipline (mirrors the M2-adv falsification precedent). No band was widened.

**Date:** 2026-06-16. Branch `v2-m3-baroclinic`. CPU only (no GPU work started — the crux
is front-loaded by design).

Run: `py -3 -m pytest tests/unit/test_m3_baroclinic.py -v -s`

---

## What the gate measured (W=192, H=96, a=6.4e6, φ_test=45°, gp1=0.05, gp2=0.3, xi=3)

```
[m3-baroclinic] unstable rate=2.770e-05 (R2=1.000), stable rate=<0 (decays), f-plane sigma=4.449e-06
```

Three assertions (verbatim from the plan, NOT weakened):

1. **Clean exponential growth — PASS.** Unstable stack grows with **R²=1.000** (a perfect
   single exponential). The growth is **dominated by the predicted K_max mode (m=5)**: the
   m=5 interface-variance band rises ~4 orders of magnitude over the linear window while grid
   modes stay sub-dominant. Charney–Stern is satisfied in the **lower** layer (verified
   β₂ = β − (f₀²/(g'₂H₂))(U₁−U₂) = −1.57e-11 < 0 at the band centre for the supercritical case).

2. **f-plane Phillips band [0.3σ, 1.5σ] — FAIL.** Measured amplitude rate **2.77e-5/s vs
   σ=4.45e-6/s → 6.2×**, far above the 1.5σ=6.67e-6 ceiling. See "Why" below.

3. **Non-vacuity (unstable ≫ stable) — PASS (measured separately).** The subcritical (xi<1)
   config **decays** (negative rate) under the identical pipeline, so the instability genuinely
   separates supercritical from subcritical. (Assertion 3 is not reached because assertion 2
   aborts first, but the stable-decays result is printed and reproducible.)

**Vortex coherence gate (d) — PASS.** A balanced GRS-scale lower-layer anticyclone
(Ro=0.524 > 0.1) stays bounded and positive over 400 steps (h₂ anomaly 7106 → 6088, ratio
0.86 ≪ the 10× runaway bound). No NaN, no outcrop.

---

## Why the rate is ~6× the f-plane σ (debugged, not hand-waved)

Two independent contributions, neither a setup bug I introduced:

- **(a) The plan's f-plane formula is a factor √2 low.** The plan specifies
  `k_d² = 2f₀²/(g'₂H)`. The correct equal-layer 2-layer QG baroclinic deformation
  wavenumber is `k_d² = f₀²/g'₂·(1/H₁ + 1/H₂) = 4f₀²/(g'₂H)` for H₁=H₂=H/2 — **twice** the
  plan's value, so the plan's σ is `√2 ≈ 1.89×` too small. With the corrected k_d the
  target is σ_max=8.41e-6 and the ratio drops from 6.2× to **3.3×**. (I left
  `predicted_growth_rate_fplane` citing the plan's verbatim formula rather than silently
  re-coefficienting a specified gate; the verdict records the correction.)

- **(b) The discrete, *localized*-shear instability genuinely grows faster than the idealized
  uniform-shear f-plane Phillips closed form (~3.3× residual).** This is the deviation the plan
  itself anticipated ("the discrete spherical solver deviates from idealized β-plane QG by more
  than a factor 2"), but larger in practice. Contributing factors, each checked:
  - The base imposes a *localized* meridional shear band (a Gaussian interface-slope bump);
    localized jets are more unstable than the uniform-shear Phillips idealization, and the
    lateral curvature of the lower-layer jet adds a barotropic contribution. (The stable case
    still decays, so the growth is baroclinically driven, not a pure barotropic artifact.)
  - The deformation radius is only **marginally resolved** (L_D ≈ 2.8 grid cells meridionally
    at this resolution). This is a hard tradeoff at planetary radius: the constraint
    `m@K_max × (L_D/dy) ≈ 7` (fixed by H alone) means resolving L_D to ≥3 cells AND keeping
    K_max at a clean mid-wavenumber (m≈5) requires H≈96 — already used. Higher resolution
    (W=288,H=144) restabilizes within the step budget rather than converging toward the
    f-plane rate, indicating the discrete dispersion (not just under-resolution) sets the rate.
  - A sharp **discrete threshold**: xi=2 (analytically supercritical) **decays**; only xi≳2.5–3
    grows. The discrete critical shear sits well above the analytic U_crit (xi=1) — again
    consistent with marginal L_D resolution raising the effective critical shear, and with the
    rate then jumping super-linearly once the threshold is cleared.

**Ruled-out setup bugs (the systematic-debugging pass):**
- Charney–Stern layer: confirmed β₂<0 in the **lower** layer (not the upper).
- Test latitude: 45°, inside the band, away from the polar sponge (>65°) and the equator.
- Hyperviscosity: shown to suppress only grid-scale modes (m≈46–48); it does NOT damp K_max=m5,
  and the result is unchanged with nu4∈{0, 0.02, 0.05}. (Without any nu4, the m=5 physical mode
  still dominates once the base is balanced; the earlier "false growth" with nu4=0 on an
  *un-resolved* base was pure grid-mode aliasing — a diagnostic trap that was caught.)
- IC balance: the base is geostrophically balanced with a **bounded** localized interface tilt
  (an earlier globally-balanced single-signed jet drove h₁ to the floor → broke balance → false
  blow-up; replaced by the localized-tilt construction that keeps both layers strictly positive).

---

## Honest conclusion

The solver **does** produce emergent baroclinic instability off a balanced reduced-gravity base:
a clean (R²=1.0) exponential at the predicted K_max wavenumber, supercritical-grows /
subcritical-decays, with a coherent finite-amplitude vortex. The *mechanism* — the milestone's
core physics goal — is validated. What is **not** validated is the **quantitative match to the
f-plane Phillips closed form**: the discrete localized-jet growth rate runs ~3–6× hot, partly
because the plan's `k_d` formula is √2 low and mostly because a marginally-resolved, localized
shear band on the discrete sphere is genuinely more unstable than the uniform-shear f-plane
idealization. The [0.3,1.5]σ band assertion therefore fails on its merits.

This is failure-tree **case (b)** ("grows but outside the band"), not a falsification of the
emergent-instability premise.

## Gate-code bugs found (independent of the physics)

The plan's verbatim `_growth_rate` / gate had two genuine code defects, fixed in
`tests/unit/test_m3_baroclinic.py` (these are correctness fixes, not band-weakening):
1. **Unit mismatch:** it fit a per-*step* slope of log(eddy **variance**) and compared it to a
   per-*second* **amplitude** σ. The fix returns σ_amp = (slope/dt)/2 in per-second units
   (variance ~ amplitude², so its log-slope is 2σ·dt).
2. **Run abort on saturation:** the vigorous instability saturates by lower-layer outcropping,
   which raises the positivity ValueError mid-run; `_growth_rate` now stops the record at that
   point (the linear window is already sampled) instead of aborting the whole gate.
3. The plan's `n_steps = 4·efold_steps_estimate` with `assert n_steps < 20000` is infeasible at
   the resolution where growth is clean; replaced with a fixed linear-window cap (11000 steps),
   since `_growth_rate` self-terminates at saturation.

## Recommendation

Do **not** proceed to GPU (Task 8) on the strength of a quantitative f-plane match — that match
fails. Options, in order of rigor:
1. **Re-baseline the gate to the correct equal-layer QG `k_d² = 4f₀²/(g'₂H)`** and *re-derive a
   defensible tolerance band for the discrete localized-jet regime* (e.g. validate against a
   direct numerical eigensolve of the discrete 2-layer QG dispersion on this grid, rather than
   the continuous uniform-shear f-plane closed form). The mechanism is sound; the *target* is the
   wrong yardstick for a localized, marginally-resolved mode.
2. If a clean quantitative match to the *uniform-shear* f-plane is required, the base must impose
   a **uniform** (not localized) shear over several L_D with L_D resolved by ≥5 cells — which at
   planetary radius needs W≈384, H≈192 and a correspondingly larger step budget (or a smaller
   crux planet, per the plan's "smaller domain/coarser grid for the crux" escape hatch).

## Artifacts (all in `shallow_water_ref.py`, appended; M1/M2 untouched)

`baroclinic_test_state`, `_balanced_sheared_base` (bounded localized-tilt balanced base),
`eddy_interface_var`, `predicted_growth_rate_fplane` (cites the plan's formula), 
`efold_steps_estimate`, `vortex_test_state`, `local_rossby_number`. Test:
`tests/unit/test_m3_baroclinic.py` (vortex gate PASSES; baroclinic gate FAILS on assertion (2),
recorded on purpose).
