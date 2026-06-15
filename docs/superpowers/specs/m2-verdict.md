# M2 Verdict — Semi-Implicit Shallow-Water GPU Solver

**Status: M2-core PASS**
**Date:** 2026-06-15
**Branch:** v2-m2-semi-implicit

---

## Gate Table

| Gate | Value | Threshold | Result |
|------|-------|-----------|--------|
| Per-field M2 GPU-vs-CPU diff (helmholtz apply, sor, residual, SI predictor, rhs, backsub, FCT) | 11/11 tests pass | all pass | PASS |
| Gravity-wave stability (N=20×dt\_gw, 40 SI steps, resting+bump, energy ratio) | 1.0000 | [0.9, 1.1] | PASS |
| Williamson-2 velocity\_l2\_drift (SI, 20 steps, 64×32) | 2.07e-05 | < 2e-02 | PASS |
| Williamson-2 mass drift rtol (SI) | 4.45e-08 | < 1e-05 | PASS |
| SI determinism (byte-identical SHA1, two fresh solvers, 5 steps) | ae7d1eca5be406f7… | equal | PASS |
| SOR-50 GPU vs CPU ref (max\_diff) | 6.38e-08 | ≤ 5e-05 | PASS |
| Checkpoint v2 round-trip (SI params + H\_ref + bit-exact continuation) | PASS | byte-identical | PASS |
| Version-1 checkpoint backward-compat (explicit solver load) | PASS | loads cleanly | PASS |

Grid for W2 gates: 64×32, a=1.0, omega=2.0, u0=0.2, gp=1.0, h0=5.0.
SI params: theta=0.5 (Crank-Nicolson), sor\_omega=1.7, helmholtz\_iters=200, picard\_iters=3.

---

## Corrected CFL Claim

**M2-core removes the GRAVITY-WAVE term from the CFL** (stable and energy-neutral to N≥20× the explicit
gravity-wave step on a resting layer; theta=0.5 Crank-Nicolson is non-dissipative, so the wave
neither blows up nor decays away at dt = 20×dt\_gw, demonstrated above at energy\_ratio=1.0000).

**The ADVECTIVE CFL still binds.** The cell-Courant on the donor-cell + FCT transport step is still
limited by the advective velocity. The fast-jet headline dt factor is the separate **M2-adv** milestone
(T0 spike falsified the cheap polar-filter route). Realized factor = min(c\_gw/|u|, Picard-alpha cap).

---

## What Passed / How Validated

**Ground truth:** `gasgiant/sim/shallow_water_ref.py` — a pure-NumPy CPU reference implementation.
All GPU kernels are verified against this reference per field, at the principled f32 tolerance.

**Per-field diff (11 tests):** Each M2 kernel — `helmholtz_apply` (random + smooth + a-scaling),
`helmholtz_sor` (50-iter match + convergence-to-exact), `helmholtz_residual`, `si_predictor`,
`helmholtz_rhs`, `velocity_backsub`, `continuity_conservative` — is run on both GPU and CPU
reference. All 11 pass. Tests in `tests/unit/test_m2_gpu.py`.

**Gravity-wave stability (capstone):** A resting layer + small Gaussian bump with omega=0 is
evolved at dt = 20×dt\_gw for 40 SI steps. The energy ratio is 1.0000, confirming the implicit
gravity-wave treatment is both BOUNDED and ENERGY-NEUTRAL at large dt. This is the definitive
demonstration that M2-core has lifted the gravity-wave CFL constraint.

**Williamson-2 SI balance:** The classic Williamson Test Case 2 (solid-body rotation steady state)
is run under the semi-implicit path for 20 steps. Velocity L2 drift = 2.07e-05 (gate < 2e-02)
and mass drift = 4.45e-08 (gate < 1e-05), confirming geostrophic balance is preserved.

**SI determinism:** Two independent SI solver instances produce byte-identical output (SHA1
ae7d1eca5be406f7…) confirming no non-deterministic GPU ordering or race condition.

**SOR-50 match:** 50-iter red/black SOR on the GPU agrees with the CPU reference to 6.38e-08
(gate ≤ 5e-05), well within the f32 accumulation bound.

**Checkpoint v2 round-trip:** `save_checkpoint` / `load_checkpoint` now version-2 format storing
all SI parameters (semi\_implicit, theta, sor\_omega, helmholtz\_iters, picard\_iters, dt\_multiplier)
and H\_ref (the latitude reference-depth profile). Continuation after load is bit-exact.
Version-1 files (M1 explicit solvers) still load cleanly with SI defaults = False.

---

## Implementation Note

`_step_semi_implicit` is GPU-kernel for all field math (predictor, SOR sweeps, backsub, FCT)
but **CPU-orchestrated** per step: the small anomaly assembly and max-floor check are NumPy
ops on f32 readbacks (matching the CPU reference algebra; the per-field tolerance tests document
the f32 vs f64 gap). A fully-resident GPU step with no per-step CPU readback is a future
performance item, not an M2-core correctness item.

---

## What's Deferred

- **M2-adv** — Semi-Lagrangian / implicit advection for fast jets. The explicit-CFL advective
  limit still binds; the T0 spike falsified the cheap polar-filter route. Needs its own
  brainstorm → spec → plan cycle.

- **M2-AE** — Prognostic available-energy (AE) shallow-water chart. Needs its own
  brainstorm → spec → plan cycle.

- **M3** — Two-layer baroclinic extension. `gasgiant/sim/shallow_water_ref.py` is the
  established ground truth for M2-adv / M2-AE / M3.

---

## Forward Pointer

`gasgiant/sim/shallow_water_ref.py` is the ground truth for all future milestones
(M2-adv, M2-AE, M3). The per-field atol=2e-5 / rtol=3e-4 discipline established in
M1 and confirmed in M2 carries forward.
