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
- **Per-field GPU-vs-CPU diff `atol=2e-5`** (R32F). The CPU op in `sw_spike/operators.py` is ground truth. Never loosen.

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

- [ ] **Step 1: Failing test** `test_swp_divergence_matches_ref`: random `h,u,v`; compare `solver.run_divergence(gpu,h,u,v)` to `sw_spike.operators.divergence_hu(h,u,v, sw_spike.grid.Grid(W,H))` at `atol=2e-5`.
- [ ] **Step 2: FAIL.**
- [ ] **Step 3:** Port the M1 plan's `sw_divergence.comp` GLSL (Task 4 there) but with `a` removed (a=1) and `wrapX` for the west neighbor. Mirror M0 `divergence_hu` exactly (read it).
- [ ] **Step 4: PASS.** - [ ] **Step 5: Commit** `git commit -m "M0.5: swp_divergence.comp"`

---

## Task 3: swp_grad_montgomery.comp + diff

**Context:** 2-layer: compute Montgomery `M1=g1·(h1+h2)`, `M2=g1·(h1+h2)+g2·h2` (port `montgomery_2layer`), then face gradients of each (port `grad_faces`). One dispatch can output `gx1,gy1,gx2,gy2`, or two dispatches (one per layer's M). Keep `wrapX`.

- [ ] **Step 1: Failing test** `test_swp_grad_montgomery_matches_ref`: random `h1,h2`, `gp=(1.0,0.05)`; compute `M1,M2` via `montgomery_2layer`, then `grad_faces(M1)`, `grad_faces(M2)`; compare GPU `run_grad_montgomery(gpu,h1,h2,gp)` → `(gx1,gy1,gx2,gy2)` at `atol=2e-5`.
- [ ] **Step 2-4:** Implement (face-difference, `1/(cosφ·dlam)` zonal with `wrapX`, north-minus-south meridional, poles 0). - [ ] **Step 5: Commit** `git commit -m "M0.5: swp_grad_montgomery.comp"`

---

## Task 4: swp_vorticity.comp + diff

**Context:** Port `vorticity` (per layer), corner field `(W,H+1)`, pole guards, `wrapX` for the `∂v/∂λ` term. Re-assert rigid-rotation analytic as a strong check.

- [ ] **Step 1: Failing test** `test_swp_vorticity_matches_ref` (random `u,v`, atol 2e-5) + `test_swp_vorticity_rigid_rotation` (`u=U·cosφ` → `ζ≈2U·sinφ`, atol 2e-2).
- [ ] **Step 2-4:** Implement (port M0 metric exactly). - [ ] **Step 5: Commit** `git commit -m "M0.5: swp_vorticity.comp"`

---

## Task 5: swp_continuity.comp (FCT two-pass) + diff — HIGHEST RISK

**Context:** Port `continuity_step` (FCT). Two passes to avoid the limiter read-write race (M1 plan Task 7 design): pass A → `h_low` + `cap`; pass B → limited anti-diffusive flux. Per-face symmetric limiter so mass conserves. **M1-review caveat:** the per-face limiter factor must be applied symmetrically on both sides of each face (zonal: `min(cap_here, cap_east)`; meridional: `min(cap_north, cap_south)`).

- [ ] **Step 1: Failing tests** `test_swp_continuity_matches_ref` (atol 2e-5 vs `continuity_step`) + `test_swp_continuity_conserves_mass` (sub-CFL, rtol 2e-6 with cos-area weighting).
- [ ] **Step 2-4:** Implement two-pass GLSL mirroring M0 `_mass_fluxes`/`_apply_fluxes`/`continuity_step`. If the diff fails, replicate pass A in Python and diff intermediates first. Do NOT loosen atol.
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

## Task 9: High-res GPU spin-up develops eddies

**Context:** The payoff: run the emergent config at HIGH resolution (start 512×256; push to 1024×512 if stable+fast) for enough steps that eddies develop richly — the thing the CPU spike could not do. Use the M0-validated config (f0=4, gp=(1.0,0.05), n_bands=14, band_contrast=0.5, nu4 retuned for the higher res — start 0.05, raise if it blows up, per the M0 nu4 stability finding).

- [ ] **Step 1: Failing/passing test** `test_swp_highres_develops_eddies` (mark slow): 512×256, run until `eddy_vorticity_std > 0.5` or a step cap; assert finite + eddies grew. Keep the test bounded (~2000 steps) so CI is feasible; the FULL spin-up is in the killgate script.
- [ ] **Step 2-4:** Implement; tune nu4 for stability at 512×256 (record the value). - [ ] **Step 5: Commit** `git commit -m "M0.5: high-res GPU spin-up develops eddies"`

---

## Task 10: Kill-gate render vs v1.6 + verdict

**Files:** Create `scripts/swp_killgate.py`, `docs/superpowers/specs/m0p5-verdict.md`.

**Context:** The binding test. Spin up the GPU 2-layer solver at the highest stable/feasible resolution (target 1024×512), encode the top layer (`sw_spike/encode.to_tracer`), render at 4096 via `maps.derive_from_tracer`, and produce the blind-panel comparison strip vs v1.6 `jupiter_vorticity` (reuse the M0 `sw_spike_killgate.py` structure + `measure_morphology`). R1/R3 gates as before. coher noted (more reliable now that the sim is finer, but still treat the visual panel as binding).

- [ ] **Step 1:** Implement `scripts/swp_killgate.py`: GPU spin-up (1024×512, enough steps for filamentary eddies — print eddy_vort_std progression + assert >0.5 before rendering), encode, 4096 render, v1.6 4096 render, matched belt crops, `out/audit/m0p5/swp_vs_v16.png` + `sw_render_full.png` + `report.txt` (coher both, R1/R3, eddy std, ms/step, total wall time).
- [ ] **Step 2:** Run `uv run python scripts/swp_killgate.py` (minutes). Inspect the output images.
- [ ] **Step 3 (human gate):** Show `swp_vs_v16.png` to the user for the **blind forced-choice panel**. Per the gate: proceed to M1 only if the high-res GPU SW render WINS or ties strongly; if it still loses, the resolution hypothesis is refuted → stop.
- [ ] **Step 4:** Write `docs/superpowers/specs/m0p5-verdict.md`: resolution reached, eddy_vort_std, coher, R1/R3, the blind-panel outcome, and the GO (build M1) / NO-GO (refute hypothesis, keep v1.6) decision. Commit.

---

## Self-review notes (for the implementer)

- **The M0 CPU operators are the spec.** Every GPU kernel is judged by `atol=2e-5` per-field diff against `src/gasgiant/sim/sw_spike/operators.py` / `solver.py`. If a diff fails, the GLSL port has an indexing/sign/metric bug — localize against the CPU field; NEVER loosen atol.
- **`texelFetch` does not wrap** — use `wrapX(i±1,W)` for EVERY zonal neighbor (M1-review F1).
- **Reuse the existing `gpu` conftest fixture** — do not spawn a second GL context (M1-review F2 / v1.6 multi-context bug).
- **Coriolis: relative vorticity only in the flux**; `f` via trapezoidal on the center-collapsed pair (M0 bug-fix + M1-review GLSL-1).
- **FCT two-pass** to avoid the limiter race; per-face symmetric limiter for conservation.
- **nu4 is resolution-sensitive** (M0 finding) — retune at high res; record the stable value.
- **The polar CFL still bites** (no semi-implicit here) — dt is small; GPU speed compensates with many cheap steps. If 1024×512 is too slow to reach filamentary eddies, fall back to 512×256 and note it (still ~3× finer than M0's CPU 192×96 in each dim).
- This is a THROWAWAY probe — if the gate passes, M1 rebuilds cleanly with Williamson rigor and radius `a`. Do not over-engineer.
