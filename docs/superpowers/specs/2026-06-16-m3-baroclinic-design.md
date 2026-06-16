# M3 — Multi-layer Baroclinic Shallow-Water Solver: Design Spec

**Status:** Approved (design) — 2026-06-16. Successor to M1 (validated single-layer C-grid solver,
merged) and the M0 spike (validated 2-layer physics, `src/gasgiant/sim/sw_spike/`, 22 tests).
Builds under the master design `2026-06-15-shallow-water-gas-giant-solver-design.md` §5 "M3".

**Goal:** A production 2-layer **explicit** reduced-gravity shallow-water solver whose jets,
belts, and vortices **emerge** from baroclinic instability off a thermally-relaxed equilibrium,
validated against linear theory, rendering a top-of-atmosphere a blind panel prefers to v1.6.

---

## 1. Why this milestone, and what it builds on

M1 delivered a validated single-layer C-grid reduced-gravity solver (`shallow_water_ref.py` CPU
ground truth + `sw_gpu.py` GPU, `a`-aware metric, Williamson-2 balance, flux-form/FCT continuity,
trapezoidal Coriolis). The M0 spike (`sw_spike/`) already proved the **2-layer physics** —
Montgomery coupling, thermal relaxation + bottom drag, emergent eddy growth — at low resolution
(22 green CPU tests). M3 **promotes M0's validated 2-layer physics into the production M1 solver**:
the same operators, now `a`-aware, Williamson-rigorous, with the baroclinic-instability validation
gates and the render-fidelity gate the M0 spike could not run at production resolution.

**Confirmed scope decisions (user, 2026-06-16):**
- **Explicit time-stepping**, building on M1's explicit operators — NOT the M2-core semi-implicit
  path. In the realistic gas-giant reduced-gravity regime both g'₁ and g'₂ are small, so gravity
  waves are slow and the explicit CFL is workable (M0/M0.5 ran explicit 2-layer). M2-adv showed the
  semi-implicit path is fragile at large dt on balanced rotating flow; M3's goal is the baroclinic
  *physics*, not large dt. Semi-implicit is deferred unless a stiffness gate later proves it needed.
- **2 active layers** (the design's production target; 3rd deferred until it earns its cost).
- **Emergent default** (jets self-organize off `h_eq`); seeded init shipped as a first-class option.
- **Render-fidelity gate is IN M3** (early ≥4K blind panel vs v1.6) — not deferred to M5.

## 2. Governing model (per the master design §2)

Two stacked Boussinesq reduced-gravity layers, vector-invariant momentum + flux-form continuity:

```
∂u_i/∂t = −(ζ_i + f) k̂ × u_i − ∇B_i + F_i^u + D_i^u ,   B_i = M_i + ½|u_i|²
∂h_i/∂t = −∇·(h_i u_i)                    + F_i^h + D_i^h
```

**Montgomery coupling (the only new prognostic operator):**
```
M_1 = g'_1 (h_1 + h_2)
M_2 = g'_1 (h_1 + h_2) + g'_2 · h_2
```
`g'_1` = top reduced gravity (barotropic mode), `g'_2 ≪ g'_1` = baroclinic mode. The mid-latitude
baroclinic deformation radius `L_D = √(g'_2 H)/f` sets the eddy/vortex scale; near the equator the
equatorial radius `L_eq = (√(g'_2 H)/β)^{1/2}` governs trapped modes (carried implicitly by the
global sphere). **Sign/coefficient discipline:** the `g'` coupling is the most error-prone part;
it is pinned by a 2-layer Williamson-2 steady-balance test, not hand-trusted (§5).

## 3. Components (CPU reference first, each independently testable)

All extend `shallow_water_ref.py` (M1 operators reused verbatim; M0 `sw_spike` is the physics
reference but production code is `a`-aware, not the a=1 spike).

1. **`montgomery_2layer(h1, h2, gp1, gp2)`** → `(M1, M2)`. Direct port of `sw_spike` op, `a`-agnostic
   (Montgomery is a potential, no metric). Gate: matches `sw_spike.montgomery_2layer` exactly.
2. **`momentum_step_M(h, u, v, M, omega, g, dt)`** — generalize M1's `momentum_step` to take a
   precomputed Montgomery `M` (Bernoulli `B = M + ke`) instead of the hard-wired `g'h`. M1's
   single-layer `momentum_step` must stay **byte-identical** (it becomes `momentum_step_M(...,
   M=gp*h, ...)` internally OR is left untouched and the 2-layer path calls the new function).
3. **2-layer state + `step_2layer`** — per layer: `momentum_step_M` with `M_i`, then `continuity_step`
   (M1's FCT, reused verbatim), then `apply_forcing`. Mirrors `sw_spike.solver.step`.
4. **`apply_forcing`** — thermal relaxation `(h_eq_i − h_i)/τ_rad` (step-based), Rayleigh bottom drag
   on layer 2, biharmonic hypervisc (grid-normalized /64, the v1.6 fix), equirect polar sponge,
   positivity floor. Port of `sw_spike.solver._apply_forcing`, `a`-aware where metric enters.
5. **`h_eq` profiles + balanced 2-layer init** — `h_eq_i(φ)` from `profiles.py`; balanced init in
   gradient-wind balance (equator-safe, `|u_init|` capped), emergent default + seeded option.
6. **Encoder + render** — top-layer h-anomaly + ζ₁ (hero channel) → existing `derive.comp` via the
   M0 `encode.py` logic promoted to a production encoder.
7. **GPU** — extend `sw_gpu.py` to 2-layer textures + a `sw_montgomery.comp` kernel; per-field
   GPU↔CPU diffs at `atol=2e-5` on pre-division quantities; a-scaling; determinism (SHA1).

## 4. Dual-path / non-regression

`solver.type = "shallow_water"` with `n_layers` ∈ {1,2}. `n_layers=1` reproduces the M1
single-layer path **byte-identically** (P0.5 hash). Kinematic (v1.5) / vorticity (v1.6) solvers
untouched → byte-identical. Checkpoint schema gains layer-indexed keys + a version bump.

## 5. Validation gates (the crux — front-loaded)

CPU `shallow_water_ref.py` is ground truth; GPU diffed per-field. The **baroclinic-instability
gates are the milestone's crux and are built/run on CPU before the GPU port** (M2-core/M2-adv
discipline: falsify early).

- **(a) Montgomery balance** — 2-layer Williamson-2 steady geostrophic balance: the coupled
  `(h_i,u_i,v_i)` stay stationary to scheme tolerance. Pins the `g'` sign matrix.
- **(b) Baroclinic instability growth rate** — small perturbation on a baroclinically-unstable
  `h_eq` tilt; measured exponential growth rate matches **β-plane two-layer linear theory** incl.
  the Charney-Stern PV-gradient-sign-change criterion (selects unstable latitudes). Plus an
  **f-plane Phillips QG-limit** small-amplitude unit test. NON-VACUOUS: assert the growth is real
  (a stable tilt must NOT grow; an unstable one must, at the predicted rate ± tolerance).
- **(c) Emergent jet spacing** — free run to equilibration; jet spacing matches
  `L_Rhines = π√(2 U_rms/β)` within tolerance (quantitative).
- **(d) Finite-amplitude vortex** — GRS-scale vortex (Rossby > 0.1) stays coherent (no spurious
  blowup / outcropping NaN).
- **(e) Conservation / budget closure** — mass to round-off (flux-form); energy + potential
  enstrophy drift fully accounted by diagnosed dissipation (forcing/drag/hypervisc), residual ≈ 0.
- **(f) Determinism** — byte-identical SHA1 over a fixed multi-step run; n_layers=1 byte-identical
  to M1; the v1.6 no-op-test discipline (each param changes output; fields evolve).
- **(g) Render-fidelity gate (the project's actual point)** — high-res GPU spin-up → encode top
  layer → ≥4096 render → v1.6 blind 3-judge forced-choice panel. **Must prefer the M3 render
  (ties fail)**, or `coher` moves measurably toward the 0.62 reference.

## 6. Spin-up budget risk (carried from the master design R5)

Emergent equilibration time is the dominant cost and is NOT shortened by balanced init. A
**spin-up kill-gate** measures steps-to-acceptable-jets on a coarse run; if emergent equilibration
exceeds budget, fall back to seeded mode or partial nudging. The M0 spike measured this at low res;
M3 re-measures at production res before committing to the full emergent gate (g).

## 7. Risks (for the adversarial plan review)

1. **Montgomery sign/coefficient matrix** — the single most error-prone part; gate (a) pins it, but
   the plan must diff `montgomery_2layer` against `sw_spike` AND validate the balance, not just one.
2. **Explicit barotropic CFL** — `c_gw = √(g'_1 H)` sets dt; with the binding *polar zonal* Δx this
   can be tiny. The plan must name the binding speed AND Δx (the M0 lesson), and confirm the chosen
   reduced-gravity regime keeps dt feasible at production resolution (else the spin-up budget blows).
3. **Equatorial init singularity** — geostrophic `u=(g'/f)∂h/∂y` is singular at f→0; use gradient-wind
   / Gill closure, cap `|u_init|`, brief bounded adjustment. Validate with an equatorial wave check.
4. **Baroclinic gate must be non-vacuous** — like the M2-core gravity-wave gate and the M2-adv crux,
   a stable tilt must demonstrably NOT grow (prove the gate can fail) before trusting that an
   unstable one grows at the right rate. (The M2-adv lesson: verify what actually binds/grows.)
5. **Render gate may refute the whole approach** — if the emergent un-nudged ζ₁ is less
   filament-rich than v1.6's nudged field, the blind panel may reject it (master design R8). This is
   the real go/no-go; run it early (don't wait for polish), accept the verdict honestly.
6. **Spin-up budget** (R5/§6) — emergent may be too slow at production res; seeded fallback exists.

## 8. Out of scope (later milestones)

- **M4** — three-domain AE polar patches + flux-matched halo exchange.
- **M5** — full render integration + art-direction + preset re-authoring + final proof gate.
- **Semi-implicit** multi-layer (modal Helmholtz) — only if a stiffness gate proves the explicit
  barotropic CFL is infeasible in the chosen regime.

`shallow_water_ref.py` remains the gold-standard CPU ground truth.
