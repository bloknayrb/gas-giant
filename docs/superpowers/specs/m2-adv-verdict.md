# M2-adv Verdict — PREMISE FALSIFIED at the CPU crux gate

**Status:** **NO-GO.** The semi-Lagrangian semi-implicit (SLSI) approach does not deliver the
fast-jet headline on the lat-lon grid. Falsified on the CPU at Task 5, **before** any GPU
kernels were written — the front-loaded crux gate worked as designed.

**Date:** 2026-06-15. Branch `v2-m2-adv`.

---

## What was claimed

M2-adv set out to remove the *residual advective CFL* (`C = |u|·dt/(a·dλ·cosφ)`) after M2-core
removed the gravity-wave CFL, giving a larger stable+accurate timestep on fast jets via
conservative semi-Lagrangian transport (SLICE remap for height, SL interpolation for momentum)
wrapped around M2-core's Helmholtz solve.

## What the experiments found (Earth-like params: g'=9.8, H0=8000, c_gw≈280 m/s)

Three isolation experiments (CPU, `shallow_water_ref.py`):

1. **The advective CFL is not the binding constraint.** At the gate's large dt, the *exact
   geostrophically-balanced Williamson-2 steady state* blows up under BOTH `step_slsi` AND
   M2-core's `step_semi_implicit`, at advective Courant ≈ **0.5** — far below the advective
   limit of 1. The binding failure is the θ-scheme's O((θdt)²) balanced-flow imbalance +
   polar positivity collapse at high **gravity-wave** Courant (≈48 at the pole at this dt,
   because c_gw ≫ |u| and the polar cell is tiny).

2. **Damping θ does not help.** θ ∈ {0.55, 0.60, 0.65} are indistinguishable from 0.50 at
   dt_mult=8; all die in ≤2 steps. The imbalance is independent of the advection scheme.

3. **The SL operators REGRESS stability.** At every sub-unit advective Courant tested on the
   balanced solid-body flow (θ=0.6), `step_semi_implicit` (Eulerian) survives 160 steps while
   `step_slsi` (SL) does not:

   | advective Courant (polar gravity-wave Courant) | Eulerian | SLSI |
   |---|---|---|
   | 0.03 (~3) | survives 160 | fails @111 |
   | 0.13 (~12) | survives 160 | fails @48 |
   | 0.25 (~24) | survives 160 (max\|v\| 127) | fails @38 |
   | 0.50 (~48) | diverges @151 | fails @2 |

   Max stable advective Courant: **Eulerian ≈ 0.25–0.375; SLSI ≈ 0**. SL loses, it does not win.

**No stable C>1 regime exists for `step_slsi`**, so the advection-accuracy headline is undefined.

## Honest conclusion

On this lat-lon grid with Earth-like gravity, the post-M2-core binding step limit is the
gravity-wave/θ imbalance + polar positivity, **not** advection — so removing the advective CFL
buys nothing, and the SLICE/bicubic SL operators add their own large-dt instability on the
balanced base state. The "larger stable dt from SL advection" headline is **not achievable** as
designed.

## What was salvaged (correct, tested, reusable)

The standalone CPU SL operators are validated and conserve mass exactly — reusable for the
M2-AE chart coupling or future transport work:
- `departure_points` (iterated-midpoint trajectory) — Task 1, 3 tests green.
- `ppm_remap_1d_periodic` + Colella-Woodward limiter — Task 2, 3 tests green.
- `slice_remap_advance` (2-D cascade conservative remap, span-preserving edge fix) — Task 3,
  4 tests green incl. the strong-shear conservation regression.
- `sl_advect_velocity` / `sl_momentum_predictor` (bicubic SL momentum) — Task 4, 2 tests green.

These remain in `shallow_water_ref.py`. `step_slsi` is retained but flagged NO-GO (unstable at
large dt; do not use for production). No GPU kernels were built. M2-core (`step_semi_implicit`)
and M1 remain byte-identical and untouched.

## Untested regime (the one caveat)

All experiments used Earth-like **g'=9.8** → fast gravity waves (c_gw≈280) that dominate the
flow (u≈120), so advection never binds. A realistic gas-giant **reduced-gravity** layer has
small g' → slower gravity waves comparable to the jets, the one regime where the advective CFL
could genuinely bind and SL could help. This was not tested; it is the only path under which
M2-adv might be salvageable, and it requires re-deriving the gate in the project's true
baroclinic parameter regime.

## Recommendation

The next real render gate is **M3 (2-layer baroclinic)**, which also establishes the realistic
reduced-gravity regime. Either (a) proceed to M3 and revisit large-dt advection only if it
actually binds there, or (b) pursue **M2-AE** (azimuthal-equidistant polar chart), which
directly attacks the *real* binding constraint found here (polar gravity-wave imbalance).
M2-adv as a standalone "advective-CFL removal" milestone is not justified on the current grid.

`shallow_water_ref.py` remains the gold-standard CPU ground truth for M2-AE and M3.
