# Milestone 0.5 — GPU 2-Layer Render Probe (the cheap hypothesis test)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development or superpowers:executing-plans. Steps use checkbox (`- [ ]`) syntax.

**Goal:** GPU-port the **existing, validated M0 2-layer shallow-water solver** just enough to run its emergent config at high resolution (1024×512+, which the CPU spike could not), render the top layer at ≥4096 through the existing pipeline, and re-run the v1.6 blind panel. This is the **cheapest test of the resolution hypothesis** that M0's verdict and the M1 adversarial review both identified as the real go/no-go.

**Why this exists:** The M0 verdict's loss to v1.6 was judged *resolution-bound* (192×96 CPU vs v1.6's 4096). The M1 plan's adversarial review (finding F1, fatal) confirmed that M1 — a single-layer numerics-validation milestone — produces **no** evidence on that hypothesis, while a GPU port of the *2-layer* M0 solver does. **Gate:** if the high-res GPU 2-layer render still loses the blind panel to v1.6, the resolution hypothesis is refuted → **stop, keep v1.6, do not build M1–M5.** If it wins or ties strongly, proceed to M1 (the clean Williamson-validated rebuild) with the hypothesis confirmed.

**Architecture:** Port the M0 CPU operators (`src/gasgiant/sim/sw_spike/{grid,operators,solver,init,encode}.py`) to moderngl compute kernels, **2 layers**, equirect only, explicit, with the M0 forcing (relaxation + drag + grid-normalized hyperviscosity + polar sponge). Every kernel is **diffed per-field against the trusted M0 CPU operators** (`atol=2e-5`) — these operators are already validated (22 green tests), so M0.5 needs no Williamson rigor; the CPU ops ARE the spec. GPU runs the emergent spin-up at high res, encodes the top layer (reuse `sw_spike/encode.py`), and renders via the existing `maps.derive_from_tracer` (built in M0). Compare to v1.6 with `measure_morphology` + the blind panel.

**Tech Stack:** Python 3.13, moderngl GPU compute (R32F textures), NumPy (the M0 CPU reference for diffing), existing `gpu` session fixture + `derive_from_tracer` render, pytest.

**Scope:** This is a THROWAWAY probe (like M0), namespaced `src/gasgiant/sim/sw_gpu_probe/`. NOT production: no Williamson suite, no semi-implicit, no AE patches, no `solver.type` wiring, no radius `a` (stays nondimensional a=1 to reproduce M0's exact config). If the gate passes, M1 rebuilds this cleanly and rigorously.

**Conventions (inherited from M0, fixed once):**
- C-grid, rows descending, a=1; `h` center `(W,H)`, `u` east-face `(W,H)`, `v` vface `(W,H+1)`, `ζ` corner `(W,H+1)`; 2 layers (subscript 1=top, 2=bottom).
- GPU: each staggered field per layer is its own R32F texture. `texture2d((W,H),1,"f4")`, `repeat_x=True`.
- **`texelFetch` does NOT honor `repeat_x`** (M1-review F1): every zonal neighbor uses explicit `wrapX(i±1, W)`. Add `wrapX` to `sw_common.glsl`.
- **Reuse the existing session-scoped `gpu` fixture** in `tests/conftest.py` (M1-review F2) — do NOT create a new one; the autouse `_gpu_context_current` handles `make_current()`. One context per process.
- **Per-field GPU-vs-CPU diff tolerance (review-corrected).** A *flat* `atol=2e-5` is unachievable for any op containing a `1/cosφ` or `1/(cosφ·dlam)` division: f32 ULPs amplified by `1/cosφ` (~5e4 near the poles) exceed it on every row at ≥512². So: **diff the PRE-division quantity** at `atol=2e-5` — i.e. compare the raw flux/difference before the metric divide (`dFx+dFy` before `/cosφ`, `M[i+1]−M[i]` before `/(cosφ·dlam)`), or equivalently `cosφ·gx`. This is precision-floor-clean at all resolutions (verified max 1.5e-6) and still validates indexing/sign/metric logic — which is what the gate exists to catch. Keep flat `atol=2e-5` only for ops WITHOUT polar metric division (vorticity numerator terms, raw fluxes). The CPU op in `sw_spike/operators.py` is ground truth. Do not loosen the pre-division tolerance.

---

## File structure

- Create `src/gasgiant/sim/kernels/swp_common.glsl` — C-grid metric helpers + `wrapX`.
- Create `src/gasgiant/sim/kernels/swp_divergence.comp`, `swp_grad_montgomery.comp` (2-layer Montgomery → face grad), `swp_vorticity.comp`, `swp_continuity.comp` (FCT two-pass), `swp_momentum.comp`, `swp_forcing.comp` (relaxation+drag+hypervisc+sponge).
- Create `src/gasgiant/sim/sw_gpu_probe/__init__.py` and `solver.py` — `SwpState` (2-layer textures) + `SwpSolver` (kernel wiring, `step`, spin-up, readback).
- Create `tests/unit/test_sw_gpu_probe.py` — per-field GPU-vs-CPU diffs + a short GPU spin-up eddy-growth check.
- Create `scripts/swp_killgate.py` — high-res GPU spin-up → encode → 4096 render → blind-panel strip vs v1.6 + report.
- Create `docs/superpowers/specs/m0p5-verdict.md` — the gate result.

---

## Task 1: swp_common.glsl + GPU 2-layer state plumbing

**Files:** Create `src/gasgiant/sim/kernels/swp_common.glsl`, `src/gasgiant/sim/sw_gpu_probe/__init__.py`, `src/gasgiant/sim/sw_gpu_probe/solver.py`; Test `tests/unit/test_sw_gpu_probe.py`.

**Context:** `SwpState` holds, per layer i∈{1,2}: `h_i (W,H)`, `u_i (W,H)`, `v_i (W,H+1)` R32F textures, plus `h_eq_i` textures (forcing target) and scratch. `swp_common.glsl` provides `cosCenter(row,H)`, `cosVface(row,H)` (0 at poles), `wrapX(int x,int w)` = `((x%w)+w)%w`, `PI`.

- [ ] **Step 1: Failing test** — texture round-trip using the existing `gpu` fixture:
```python
import numpy as np
def test_swp_state_roundtrip(gpu):
    from gasgiant.sim.sw_gpu_probe import solver
    h = np.random.default_rng(0).random((32,64)).astype(np.float32)
    st = solver.SwpState.create(gpu, W=64, H=32)
    st.upload("h1", h)
    np.testing.assert_allclose(st.download("h1"), h, atol=0)
```
- [ ] **Step 2: FAIL.**
- [ ] **Step 3:** Implement `swp_common.glsl`; `SwpState.create` (allocate per-layer textures, sizes `(W,H)` and `(W,H+1)`), `upload(name,arr)`/`download(name)` (mirror `sw_gpu`/`solver.py` texture handling). Use the existing `gpu` conftest fixture (M1-review F2).
- [ ] **Step 4: PASS.** - [ ] **Step 5: Commit** `git commit -m "M0.5: GPU 2-layer state + swp_common.glsl (wrapX)"`

---

## Task 2: swp_divergence.comp + diff

**Context:** Port `sw_spike/operators.divergence_hu` (same-index zonal flux, `1/cosφ` metric, north-minus-south). Used by continuity but also a standalone diffable op. Single-field (per layer).

- [ ] **Step 1: Failing test** `test_swp_divergence_matches_ref`: random `h,u,v`; compare the **pre-division** flux sum (`dFx+dFy` before the `/cosφ` divide — have `run_divergence` optionally return the pre-division field, or diff `cosφ·div`) to the CPU equivalent at `atol=2e-5` (per Conventions — the divided field would fail at the poles in f32).
- [ ] **Step 2: FAIL.**
- [ ] **Step 3:** Port the M1 plan's `sw_divergence.comp` GLSL (Task 4 there) but with `a` removed (a=1) and `wrapX` for the west neighbor. Mirror M0 `divergence_hu` exactly (read it).
- [ ] **Step 4: PASS.** - [ ] **Step 5: Commit** `git commit -m "M0.5: swp_divergence.comp"`

---

## Task 3: swp_grad_montgomery.comp + diff

**Context:** 2-layer: compute Montgomery `M1=g1·(h1+h2)`, `M2=g1·(h1+h2)+g2·h2` (port `montgomery_2layer`), then face gradients of each (port `grad_faces`). One dispatch can output `gx1,gy1,gx2,gy2`, or two dispatches (one per layer's M). Keep `wrapX`.

- [ ] **Step 1: Failing test** `test_swp_grad_montgomery_matches_ref`: random `h1,h2`, `gp=(1.0,0.05)`; compute `M1,M2` via `montgomery_2layer`, then the **pre-division** face differences (`M[i+1]−M[i]`, north−south) — diff those at `atol=2e-5` (per Conventions); equivalently compare `cosφ·gx`. The Montgomery values `M1,M2` themselves diff at flat `atol=2e-5` (no division).
- [ ] **Step 2-4:** Implement (face-difference, `1/(cosφ·dlam)` zonal with `wrapX`, north-minus-south meridional, poles 0). - [ ] **Step 5: Commit** `git commit -m "M0.5: swp_grad_montgomery.comp"`

---

## Task 4: swp_vorticity.comp + diff

**Context:** Port `vorticity` (per layer), corner field `(W,H+1)`, pole guards, `wrapX` for the `∂v/∂λ` term. Re-assert rigid-rotation analytic as a strong check.

- [ ] **Step 1: Failing test** `test_swp_vorticity_matches_ref` (random `u,v`, atol 2e-5) + `test_swp_vorticity_rigid_rotation` (`u=U·cosφ` → `ζ≈2U·sinφ`, atol 2e-2).
- [ ] **Step 2-4:** Implement (port M0 metric exactly). - [ ] **Step 5: Commit** `git commit -m "M0.5: swp_vorticity.comp"`

---

## Task 5: swp_continuity.comp (FCT two-pass) + diff — HIGHEST RISK

**Context:** Port `continuity_step` (FCT). Two passes (A: `h_low`+`cap`; B: limited anti-diffusive flux), separated by `ctx.memory_barrier()` (repo convention, e.g. `omega_force.comp` SUBPASS 0/1). `cap` is a pass-A output TEXTURE (not thread-local). **The zonal and meridional limiters are NOT the same rule — port `operators.py` VERBATIM and separately** (review F3): zonal `Ax_lim = Ax·min(sx[i], sx[i+1])` where `sx=min(1, cap/|Ax|)` per face (operators.py:136-137); meridional `cap_v = min(cap_north, cap_south)` THEN a single `sy = min(1, cap_v/|Ay|)`, `Ay_lim = Ay·sy` (operators.py:139-141). Per-face conservation is exact because `_apply_fluxes` differences one flux array (each face added to one cell, subtracted from its neighbor from the same value).

- [ ] **Step 1: Failing tests** `test_swp_continuity_matches_ref` (atol 2e-5 vs `continuity_step`) + `test_swp_continuity_conserves_mass` (sub-CFL). **The mass test must cast the downloaded f32 `h` to float64 BEFORE the cos-area-weighted sum** (review F4) — measure physical conservation, not f32 summation order — then assert `rtol=2e-6`.
- [ ] **Step 2-4:** Implement two-pass GLSL mirroring M0 `_mass_fluxes`/`_apply_fluxes`/`continuity_step` exactly (zonal/meridional limiters separate). If the diff fails, replicate pass A in Python and diff the intermediates `cap, sx, cap_v, sy, Ax_lim, Ay_lim` individually at `atol=2e-5` to localize. Do NOT loosen atol.
- [ ] **Step 5: Commit** `git commit -m "M0.5: swp_continuity.comp FCT (two-pass)"`

---

## Task 6: swp_momentum.comp + diff

**Context:** Port single-step vector-invariant momentum per layer (from `sw_spike/solver._layer_momentum`, with the Coriolis double-count fix: **relative vorticity only in the flux**, `f` via trapezoidal). Reads a precomputed `ζ` corner texture (dispatch vorticity first). Bernoulli `B=M+0.5(u²+v_c²)` where `M` is that layer's Montgomery. Trapezoidal Coriolis (Cayley) on the center-collapsed `(u_star, v_c)` then scatter `v` back to faces — **match the M0 structure exactly** (M1-review minor GLSL-1: M0 rotates the center-collapsed pair, not face-u against face-v).

- [ ] **Step 1: Failing test** `test_swp_momentum_matches_ref`: random `h1,h2,u1,v1`; compare GPU `run_momentum` for layer 1 (Montgomery M1) to `sw_spike.solver._layer_momentum(h1,u1,v1,M1,f0,g,dt)` at `atol=2e-5`.
- [ ] **Step 2-4:** Implement, mirroring `_layer_momentum` (including the `v_c`→rotate→`v_face` scatter). - [ ] **Step 5: Commit** `git commit -m "M0.5: swp_momentum.comp"`

---

## Task 7: swp_forcing.comp (relaxation + drag + hypervisc + polar sponge) + diff

**Context:** Port `sw_spike/solver._apply_forcing` (step-based tau, grid-normalized `/64` biharmonic, polar sponge poleward of 65°, positivity floor). Single dispatch over both layers (or per layer). `wrapX` in the biharmonic Laplacian.

- [ ] **Step 1: Failing test** `test_swp_forcing_matches_ref`: build a state, apply one GPU forcing pass vs a Python `_apply_forcing` replica on the same arrays, compare all fields at `atol=2e-5`.
- [ ] **Step 2-4:** Implement. - [ ] **Step 5: Commit** `git commit -m "M0.5: swp_forcing.comp"`

---

## Task 8: SwpSolver.step + N-step GPU≈CPU agreement

**Context:** Assemble: vorticity → momentum (both layers) → continuity (both layers, new u,v) → forcing → ping-pong. Match `sw_spike/solver.step` ordering.

- [ ] **Step 1: Failing test** `test_swp_step_matches_ref_n_steps`: build identical small (96×48) emergent states on CPU (`sw_spike.init.emergent_init`) and GPU; advance 30 steps each; assert `max|h1_gpu−h1_cpu| < 5e-4` (drift bound; R32F vs f64). If it diverges more, a kernel mismatches — localize with the per-field tests.
- [ ] **Step 2-4:** Implement `SwpSolver` (`from_emergent_config`, `step`, `eddy_vorticity_std`, `download`). - [ ] **Step 5: Commit** `git commit -m "M0.5: SwpSolver.step (N-step GPU=CPU)"`

---

## Task 9: High-res GPU spin-up develops eddies (with the physical-time fix)

**Context:** The payoff: run the emergent config at HIGH resolution where the CPU spike could not. **Primary resolution is 512×256** (the minimal viable test per the feasibility review); 1024×512 is a stretch goal. Use the M0 config (f0=4, gp=(1.0,0.05), n_bands=14, band_contrast=0.5).

**Two corrections from the feasibility review, both mandatory:**
1. **Step budget is large and must be sized up front.** `dt ∝ 1/(W·H)` (the binding constraint is the polar zonal spacing `cosφ_min·dlam`), so reaching M0's eddy regime (~12000 steps at 192×96) needs **~32k steps at 512×256** and **~64k at 1024×512** — NOT a ~2000-step cap. On GPU these are cheap (ms/step), but they MUST be the run size.
2. **M0's forcing is STEP-based (tau in steps), so eddies grow on a step clock, not a physical clock.** At a smaller dt the per-step relaxation/drag/sponge over-damp the (now longer) integration. **Fix:** rescale the forcing per-step fractions by `dt_old/dt_new` (where `dt_old` is the M0 192×96 dt) so the *physical* regime is reproduced — i.e. multiply `1/tau_rad`, `1/tau_drag`, sponge `rate`, and confirm `nu4/64`. **Validate** by first reproducing M0's `eddy_vorticity_std(t)` growth CURVE at 192×96 on the GPU (vs the CPU reference), THEN confirming 512×256 follows the same physical-time curve with the rescaled constants. The M0 team already hit this exact wall at 256×128 (`sw_spike_killgate.py:45-47`) — do not repeat it.

- [ ] **Step 1: Test** `test_swp_forcing_physical_time_rescale`: assert the rescaled-forcing GPU run at 192×96 reproduces the CPU `eddy_vorticity_std` at a matched physical time within ~20% (proves forcing fidelity). Plus a bounded `test_swp_highres_smoke` (512×256, ~2000 steps, mark slow): assert finite + eddy_vort_std rising (NOT the full regime — that's the killgate).
- [ ] **Step 2-4:** Implement the forcing rescale + spin-up. **nu4:** start 0.05; run a short (~1-2k step) stability probe at the FINAL resolution and bisect to a stable-and-rich value (record it); note `_biharmonic` ignores the cosφ metric but the polar sponge (>65°) masks the polar over-damping. - [ ] **Step 5: Commit** `git commit -m "M0.5: high-res GPU spin-up + physical-time forcing rescale"`

---

## Task 10: Kill-gate render vs v1.6 + verdict

**Files:** Create `scripts/swp_killgate.py`, `docs/superpowers/specs/m0p5-verdict.md`.

**Context:** The binding test — redesigned for the two fairness fatals.

**FAIRNESS (review F1/F4): the comparison MUST be apples-to-apples.** v1.6's `jupiter_vorticity` renders WITH a detail-synthesis filament pass + warp + lanes (`detail.intensity=0.95`, `warp_amount=0.04`); `derive_from_tracer` renders the SW field WITHOUT any of them. Judging those against each other scores the appearance pipeline, not the sim morphology — and this same confound affected the M0 result. **Primary mode = MORPHOLOGY-ONLY:** render v1.6 with `detail.intensity=0`, `warp_amount=0`, `lanes=None`, and the SAME jupiter-like palette as SW, so the panel judges raw band+vorticity STRUCTURE only — the actual hypothesis. (Optional follow-up: give SW the detail pass by synthesizing a detail_tex from its own `u1/v1` velocity, for a "both with filaments" test.) Render both at the highest stable resolution (512×256 primary; 1024×512 stretch), encode top layer (`sw_spike/encode.to_tracer`), 4096 render, matched belt crops.

**FALSIFIABILITY GUARD (review F4/F5): a LOSE does not always refute.** The decision rule:
- **WIN** (SW preferred / indistinguishable on structure) → **proceed to M1**, even under asymmetric pipelines (SW won despite the handicap).
- **LOSE** → refutes the resolution hypothesis **ONLY IF** (a) the pipelines were MATCHED (morphology-only), AND (b) the forcing was physical-time-rescaled (Task 9), AND (c) the SW eddy field reached the M0-equivalent filamentary regime (`eddy_vorticity_std ≥ ~1.0`, M0 reached 1.09 — NOT the old 0.5). Otherwise the result is **INCONCLUSIVE**, not a NO-GO.

**BLIND PROTOCOL (review F5):** the comparison image MUST strip all text labels and the coher number; randomize which crop is top/bottom; render both through the same palette; instruct the judge to score filament-folding / belt STRUCTURE only, not color.

- [ ] **Step 1:** Implement `scripts/swp_killgate.py`: GPU spin-up at the sized step budget (Task 9; print `eddy_vort_std` progression; **assert ≥1.0 before rendering** — else emit INCONCLUSIVE, do not refute), encode, 4096 render of SW AND of v1.6 with detail/warp/lanes DISABLED + matched palette, R1/R3 gates, write `out/audit/m0p5/swp_vs_v16_blind.png` (UNLABELED, randomized order) + `sw_render_full.png` + `report.txt` (coher both, R1/R3, eddy std, the regime/match flags for the falsifiability guard, ms/step, wall time, and which crop is which — for AFTER the judgment).
- [ ] **Step 2:** Run `uv run python scripts/swp_killgate.py` (minutes). Inspect.
- [ ] **Step 3 (human gate):** Show the UNLABELED `swp_vs_v16_blind.png` to the user with the structure-only instruction. Apply the falsifiability-guarded rule above.
- [ ] **Step 4:** Write `docs/superpowers/specs/m0p5-verdict.md`: resolution, eddy_vort_std (regime reached?), pipeline-match mode, coher, R1/R3, blind-panel outcome, and GO (build M1) / NO-GO (refute, keep v1.6) / INCONCLUSIVE (re-run matched/longer) decision. Commit.

---

## Self-review notes (for the implementer)

> **Rev 2 (adversarial-review fold-in):** 2 fatal + 4 major findings folded in. (1) Tolerance: diff PRE-division quantities, not the metric-divided field (f32 polar amplification). (2) Fairness: the kill-gate renders v1.6 morphology-only (detail/warp/lanes OFF, matched palette) — the M0 result was confounded by this same asymmetry. (3) Step budget ~32k @512² / ~64k @1024² (dt∝1/(W·H)), not ~2k. (4) Forcing rescaled to physical time (the M0 256×128 wall). (5) Falsifiability guard: a LOSE refutes only under matched pipeline + physical-time forcing + eddy regime ≥1.0. (6) Blind protocol: unlabeled, randomized, structure-only.

- **The M0 CPU operators are the spec.** Every GPU kernel is judged by per-field diff against `src/gasgiant/sim/sw_spike/operators.py` / `solver.py` (PRE-division at `atol=2e-5` for metric-divided ops; flat `atol=2e-5` otherwise). If a diff fails, the GLSL port has an indexing/sign/metric bug — localize against the CPU field; NEVER loosen.
- **`texelFetch` does not wrap** — use `wrapX(i±1,W)` for EVERY zonal neighbor (M1-review F1).
- **Reuse the existing `gpu` conftest fixture** — do not spawn a second GL context (M1-review F2 / v1.6 multi-context bug).
- **Coriolis: relative vorticity only in the flux**; `f` via trapezoidal on the center-collapsed pair (M0 bug-fix + M1-review GLSL-1).
- **FCT two-pass** to avoid the limiter race; per-face symmetric limiter for conservation.
- **nu4 is resolution-sensitive** (M0 finding) — retune at high res; record the stable value.
- **The polar CFL still bites** (no semi-implicit here) — dt is small; GPU speed compensates with many cheap steps. If 1024×512 is too slow to reach filamentary eddies, fall back to 512×256 and note it (still ~3× finer than M0's CPU 192×96 in each dim).
- This is a THROWAWAY probe — if the gate passes, M1 rebuilds cleanly with Williamson rigor and radius `a`. Do not over-engineer.
