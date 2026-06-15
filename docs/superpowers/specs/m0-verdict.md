# Milestone 0 — Shallow-Water Image Spike: Verdict

**Date:** 2026-06-15
**Branch:** `v2-shallow-water`
**Decision:** ✅ **PROCEED TO M1 (eyes open)** — user call, 2026-06-15.
The SW solver is validated and eddies emerge; the M0 visual loss to v1.6 is
judged resolution-bound (192×96 CPU vs 4096 GPU), not a refutation. M1 (GPU
C-grid solver at 4096, Williamson-validated) + M2 (semi-implicit, usable dt)
become the *fair* visual test. Risk accepted: even at 4096 the SW morphology may
not beat v1.6's tuned appearance — M1's render-fidelity check (spec §5 M3) is the
next real go/no-go.

## What M0 built

A throwaway CPU NumPy 2-layer **C-grid** reduced-gravity shallow-water solver
(equirect only), fed through the existing GPU render. 22 unit tests, all green.
Operators built bottom-up and individually validated:

- C-grid divergence (no checkerboard null mode), Montgomery 2-layer coupling +
  face pressure gradient, relative vorticity (analytic `2U sinφ` validated),
  trapezoidal Coriolis (norm-preserving), **FCT continuity** (mass-conserving to
  1e-12 + positivity-preserving), explicit 2-layer step, h_eq + balanced init
  with equatorial closure, thermal relaxation + bottom drag + grid-normalized
  hyperviscosity + polar sponge, eddy-vorticity metric, render encoder.

## Gate results

| Gate | Result |
|------|--------|
| R1 — checkerboard 2Δx mode | **PASS** — decays (C-grid suppresses it; amp→0.000) |
| R3 — thickness positivity / no NaN | **PASS** — min h1 = 2.92 > floor, no NaN |
| Mass conservation (flux-form) | **PASS** — 1e-12 in the sub-CFL regime |
| Emergent baroclinic eddies | **YES** — eddy vorticity grew machine-ε → ~1.09 (filamentary regime) |
| Stability / perf | 4.7 ms/step at 192×96; 12000 steps ≈ 140 s |
| coher (v1.6 0.355 vs SW 0.977) | **UNRELIABLE** — coarse 192×96 upscaled to 4K scores trivially-horizontal fields ~1.0; not a usable signal |

## Honest visual assessment (the binding gate)

The SW spike **did develop emergent baroclinic eddies** — a regular eddy/Rossby-
wave train forms along the jet shear lines (visible as a serrated pattern at the
belt edges in `sw_render_full.png`). This is real emergent dynamics the v1.6
kinematic/nudged solver cannot produce by mechanism.

**But the rendered image does NOT beat v1.6.** v1.6 shows rich, mottled
*folded-filament* belt morphology; the SW spike shows smooth bands with a regular
eddy wave-train and little fine turbulent texture. Two compounding reasons:

1. **Resolution handicap.** The spike sim is 192×96 (CPU, explicit). v1.6's sim is
   4096. Upscaling the coarse SW field to 4K blurs out the fine structure that
   would carry folded-filament richness. This is the dominant factor and is *not*
   a property of the SW approach — it is the M0 spike's deliberate cheapness.
2. **Regime.** At feasible CPU resolution + explicit-CFL-limited physical time,
   the eddies are in an early, laminar-wave-train regime, not the fully developed
   2D turbulence that produces folded filaments. Reaching that needs longer
   integration (semi-implicit for a usable dt) and finer resolution — i.e. M1–M2.

## Interpretation

- **The SW approach is validated as sound:** operators correct, C-grid kills the
  checkerboard, mass conserves, positivity holds, baroclinic instability emerges
  on its own, the thing is stable and deterministic.
- **The literal kill-gate ("beat v1.6 now") is NOT met** at M0's render-feasible
  CPU resolution. The visual loss is resolution-bound, not a refutation of SW.
- The kill-gate exists to prevent sinking M1–M5 effort on faith. The honest signal
  is: *a fair visual test requires M1's GPU solver at 4096 with the semi-implicit
  step (M2) for a usable timestep* — which is exactly the part M0 deliberately
  skipped. M0 cannot, by construction, render at parity with a 4096 GPU sim.

## The decision (user's)

Per the spec, proceed to M1 only if the blind panel prefers the SW render. On the
literal rule, **NO-GO** (the coarse render does not beat v1.6). The judgment call
is whether the loss is disqualifying or merely resolution-bound:

- **STOP / keep v1.6:** the kill-gate did its job — don't invest M1–M5 until SW
  shows a visual win.
- **PROCEED with eyes open:** treat M0 as proof the *solver* works and eddies
  emerge; accept that a fair render needs M1 (GPU, 4096) + M2 (semi-implicit), and
  fund M1 as the real visual test — but acknowledge the elevated risk that even at
  4096 the SW morphology may not beat v1.6's tuned appearance.
- **CHEAPER MIDDLE:** before committing M1, push the spike further (finer grid via
  a faster GPU port of these exact operators, or longer integration) to get a less
  resolution-handicapped render — a smaller bet than full M1.

## M1 caveats carried from M0 (for whoever builds M1)

1. Add planetary radius `a` to the grid metric (spike is nondimensional, a=1).
2. `divergence_hu` uses a same-index C-grid flux (not centered) to avoid a
   checkerboard null mode — keep mass-flux/momentum forms consistent in the
   energy-conserving reference.
3. The FCT positivity floor is a small non-conservative mass source when it fires;
   M1 wanting hard conservation+positivity must redistribute the deficit (Zalesak
   mass-fix).
4. The lon-lat polar zonal CFL forces a tiny explicit dt; M0 mitigates with a
   polar sponge + step-based forcing. M1's semi-implicit step (M2) is the real fix
   — without it, render-resolution sims are CFL-prohibitive on CPU.
5. nu4 stability window is narrow and resolution-dependent (0.05 rich but blows up
   at coarse/long runs; 0.07–0.08 stable). M1's semi-implicit + proper hypervisc
   should widen this.
6. `sw_spike/operators.py` (C-grid operators) graduates into M1's
   `shallow_water_ref.py` CPU ground truth.
