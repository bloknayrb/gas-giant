# M1 Verdict — Shallow-Water GPU Solver

**Status: M1 PASS**
**Date:** 2026-06-15
**Branch:** v2-shallow-water

---

## Gate Table

| Gate | Value | Threshold | Result |
|------|-------|-----------|--------|
| Per-field GPU-vs-CPU diff (atol=2e-5) | 7/7 tests pass | all pass | PASS |
| Williamson-2 velocity_l2_drift | 5.31e-06 | < 2e-02 | PASS |
| Mass conservation (rtol) | 0.00e+00 | < 1e-05 | PASS |
| Energy drift (rel, monitored) | 4.58e-08 | < 1e-02 | PASS (monitored) |
| Potential-enstrophy drift (rel, monitored) | 8.36e-07 | < 1e-02 | PASS (monitored) |
| Determinism (byte-identical SHA1) | 2c2cdd59a54c1d25… | equal | PASS |

Grid: 128×64, 80 steps, a=1.0, omega=2.0, u0=0.2, gp=1.0, h0=5.0.

---

## What Passed / How Validated

**Ground truth:** `gasgiant/sim/shallow_water_ref.py` — a pure-NumPy CPU reference implementation of
the single-layer shallow-water equations on a C-grid (Arakawa staggering). Every GPU kernel is
verified against this reference.

**Per-field diff (7 tests, atol=2e-5):** Each core kernel — divergence, gradient, vorticity,
continuity, momentum, single-step, and N-step integration — is run on both GPU and CPU reference,
and the maximum absolute difference must be below 2e-5. All 7 pass. Tests are in
`tests/unit/test_sw_gpu.py`, keyword `matches_ref`.

**Williamson-2 balance:** The classic Williamson Test Case 2 (solid-body rotation steady state) is
used as an integration test. After 80 steps the velocity L2 drift is 5.31e-06, well below the
2e-02 gate, confirming the GPU timestepper preserves the geostrophic balance.

**Mass conservation:** The discrete mass integral is conserved to machine zero (0.00e+00 relative
drift). This is an exact algebraic property of the C-grid divergence discretisation.

**Energy and potential-enstrophy:** These are drift-monitored quantities (not hard-closed gates).
Energy relative drift 4.58e-08 and potential-enstrophy relative drift 8.36e-07 are both many
orders of magnitude below the 1e-02 monitoring bound, indicating excellent numerical conservation
for an explicit scheme.

**GPU-vs-GPU determinism:** Two independent solver instances built from the same Williamson-2
initial condition, each stepped 40 times, produce byte-identical output (same SHA1 hash
`2c2cdd59a54c1d25…`). This confirms there is no non-deterministic GPU ordering or race condition.

---

## M2 Caveats

The explicit RK time-stepping in M1 is polar-CFL-limited: near the poles the minimum longitudinal
grid spacing shrinks to zero, forcing a very small global dt to satisfy the CFL condition. This
is not an M1 defect — it is the precise motivation for M2 (semi-implicit time integration), which
will remove the polar CFL bottleneck. M1 is a standalone, fully validated single-layer module.

M1 is **not** wired into the renderer or solver registry. Integration into the simulation pipeline
is deferred to M5.

---

## Forward Pointer

`gasgiant/sim/shallow_water_ref.py` is now the established ground-truth reference for the
single-layer shallow-water equations. It will serve as the CPU baseline for M3's two-layer
baroclinic extension, providing the same per-field atol validation discipline.
