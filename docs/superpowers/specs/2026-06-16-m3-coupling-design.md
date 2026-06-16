# M3-coupling — Baroclinic Source → v1.6 Vorticity Turbulence (design)

**Status:** Design (spike-validated GO). Next: implementation plan.
**Branch:** `v2-m3-baroclinic`. **Date:** 2026-06-16.

## Purpose

Produce a natural, Jupiter-like render whose eddy/vortex structure is *physically
grounded* in the validated 2-layer baroclinic instability — without asking the
baroclinic solver to render itself (which was falsified, see below).

**Principle (textbook geostrophic turbulence):** the baroclinic instability
decides *where and at what scale* eddies form (a physically-grounded vorticity
source at the supercritical mid-latitude bands, scale ~L_D); the v1.6 barotropic
2D-turbulence solver + render make them *look natural* (filaments via the
velocity-backtrace noise advection in `detail.comp`). Neither does the other's
job.

## Why not render the baroclinic field directly (falsified)

Two adversarial review passes + diagnostics (committed: `scripts/sw_m3_*`) showed
the direct-render path is a dead end:

- **Resolution is not the lever.** The grid striping is a C-grid *vorticity*
  2Δx checkerboard (curl of single-differenced staggered u,v), intrinsic to the
  stencil; velocity-hyperviscosity cannot damp it at any resolution (`nu4=0.05`
  did nothing; biharmonic eigenvalue on a velocity checkerboard is 64, so the
  mode is in the curl, not the velocity).
- **The coherent signal is sparse and smooth.** Only ~5 thickness blobs in one
  band; Jupiter's filaments/festoons come from the procedural FBM/warp/lane +
  velocity-backtrace layers in `render/kernels/derive.comp`, not the baroclinic
  field. No operating point makes baroclinic structure both visible and natural.
- **Multi-jet tailoring is geometrically self-defeating** (U_crit varies ~20×
  across latitude → transonic unbalanced low-lat jets vs under-resolved violent
  high-lat bands), and **outcrop-prevention dissipation = growth-killing
  dissipation** (thermal-relax e-fold ≈ growth e-fold, both ~3.6e4 s).

The 2-layer baroclinic solver remains **validated physics** (σ=2.77e-5/s, R²=1.0)
— it is simply the wrong *direct* render driver. Coupling uses it for what it is
good at: placing eddies.

## Architecture

```
 baroclinic CPU solver            coherent source module          v1.6 solver (GPU)         render
 (shallow_water_ref,      ──►    h2 eddy → smooth → geostrophic ──►  q += gain*src   ──►  detail.comp
  validated config)               vorticity proxy → polar mask        (omega_force,        velocity-
  evolving each step               → resample to equirect grid         SUBPASS 0)           backtrace
                                                                                            filaments
```

### Components & interfaces

1. **Injection hook (DONE — spike-validated, committed).**
   `Simulation.set_external_vorticity_source(field, gain)` →
   `Solver.external_omega_tex/external_gain` → `omega_force.comp` SUBPASS 0,
   equirect domain only, injected after the nudge and before the polar confine.
   **Strict no-op when `gain==0`** (regression-verified byte-identical to the
   default `jupiter_vorticity` path). Solver q scale is non-dimensional
   `coriolis_f0=3.0` — gains are fractions of ~3.0, not physical 1e-4.
   *Build work:* harden from spike-grade to production (tests below); decide
   whether to keep equirect-only or extend to the AE polar patches (default:
   keep equirect-only — the bands are mid-latitude).

2. **Coherent source module (recipe validated; needs production form).**
   From the baroclinic state: `h2e = h2 − zonal_mean(h2)` (the coherent m~5–8
   interface eddy), Gaussian-smooth (σ~2–3 cells), geostrophic-vorticity proxy
   `ζ_src = (gp2/f)·∇²(h2e)`, polar band mask ([10°,80°] cosine taper to kill
   `1/cos²` + `np.gradient` pole artifacts), resample (H,W)→equirect (W, W//2).
   Hard gate: the source's dominant zonal wavenumber must be low (m≲15), never
   the m~40+ checkerboard. (Spike got m=8.)
   *Open design decision baked into the plan:* **per-step EVOLVING injection.**
   The spike injected a *static* stamp → weak (v1.6 re-equilibrates; gain
   sensitivity 0.053→0.054). The build must inject the *evolving* baroclinic
   source as the instability grows/saturates, for a strong dynamic imprint.

3. **Source cadence + residency (measurement-driven).**
   The baroclinic CPU step is cheap at 192×96; the source is a small (H,W) field
   uploaded to one texture. Options to measure: (a) advance the baroclinic CPU
   solver alongside v1.6 and re-upload the source every N v1.6 steps; (b) make
   the baroclinic solver GPU-resident (the `SwpSolver` scaffold exists but is a
   ~70% rewrite onto a-aware kernels — only pursue if (a) is too slow). **Decide
   by benchmarking (a) first.** Do NOT assume residency is needed.

4. **Render gate + honest verdict (replaces the mis-targeted T9 gate).**
   Old gate measured top-layer eddy Rossby — wrong layer, wrong field. New
   metric: (i) vortex concentration at the baroclinically-active latitudes vs
   v1.6's latitude-flat FBM (a latitude-banded eddy-energy contrast); (ii)
   natural-texture preservation (the coupled render must not regress v1.6's
   `coher`/filament character); (iii) the human blind panel. Plus a source-side
   guard: dominant-wavenumber check (no checkerboard).

## Build order

1. **Harden + test the injection hooks** (currently spike-grade). Tests:
   no-op byte-identity when gain==0 (P0.5 discipline); a nonzero-gain changes
   output; source binding/rebinding is leak-free.
2. **Coherent source production module** + its dominant-wavenumber gate
   (CPU/numpy, unit-tested against the recipe; reject checkerboard sources).
3. **Per-step evolving injection pipeline** + the cadence/residency decision
   via benchmark (option (a) first). This is the main new engineering.
4. **New render gate + verdict doc** (`docs/.../m3-coupling-verdict.md`), blind
   panel PNG + key, and the latitude-concentration metric.

Each task via `superpowers:subagent-driven-development` with the project's
GPU↔CPU 2e-5 + P0.5 byte-identity discipline.

## Already de-risked vs still open

- **De-risked:** the coupling mechanism (opt-in, byte-identical off); the
  coherent-source recipe (m=8, no checkerboard); that the coupled output looks
  natural and concentrates eddies at the active bands.
- **Open (the plan's real work):** the static→evolving injection upgrade; the
  source cadence/residency decision (measure, don't assume); the new honest gate.

## Risks

- **R1 — Evolving injection still modulates weakly.** v1.6 re-equilibrates fast;
  even a per-step source may only modulate. Mitigation: tune gain + consider
  injecting at the nudge-target level (bias `q_target`) rather than additively,
  and/or reduce v1.6's own FBM inject so the baroclinic source dominates the
  eddy budget. Fallback: accept "physically-biased FBM" as the honest outcome.
- **R2 — Grid/units mismatch** (a-aware baroclinic vs non-dim v1.6 q scale 3.0).
  Mitigation: calibrate the source amplitude empirically (done in spike); pin a
  unit test on the normalization.
- **R3 — Cadence cost** if the baroclinic CPU step + upload dominates. Mitigation:
  benchmark; sub-sample the source in time; only then consider residency.
- **R4 — The verdict stays subjective.** The latitude-concentration metric +
  blind panel must be defined so the gate can actually fail.

## Verification

- Unit: hook no-op byte-identity; source dominant-wavenumber gate; amplitude
  normalization.
- Integration: coupled render at 4096; latitude-concentration metric vs v1.6
  baseline; `coher` not regressed; blind panel PNG.
- Verdict doc records gain calibration, cadence benchmark, and the honest read.
