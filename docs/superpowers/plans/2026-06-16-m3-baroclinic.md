# M3 — Multi-layer Baroclinic Solver Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development. Steps use checkbox (`- [ ]`). CPU reference (`shallow_water_ref.py`) is the gold-standard ground truth, extended FIRST in every task; GPU kernels diffed per-field at `atol=2e-5` on pre-division quantities. The M0 spike (`src/gasgiant/sim/sw_spike/`) is VALIDATED 2-layer physics (22 tests) and is the porting reference — but production code is `a`-aware (M1 graduated the spike with planetary radius threaded through every metric site).

**Goal:** Production 2-layer **explicit** reduced-gravity shallow-water solver whose jets/belts/vortices emerge from baroclinic instability off a thermally-relaxed equilibrium, validated vs linear theory, rendering a top-of-atmosphere a blind panel prefers to v1.6.

**Architecture:** Promote M0's validated 2-layer physics (Montgomery coupling + thermal relaxation + bottom drag) into the production M1 single-layer solver. Per layer: M1's `a`-aware vector-invariant `momentum_step` (generalized to take a Montgomery potential `M`) + M1's flux-form/FCT `continuity_step`, then forcing. Explicit time-stepping (slow reduced-gravity waves; no semi-implicit). `n_layers=1` stays byte-identical to M1.

**Tech Stack:** Python 3.13 + NumPy CPU reference; ModernGL R32F compute kernels reusing M1's `sw_common.glsl`; pytest with the session-scoped `gpu` fixture; the existing `derive.comp` render path + M0 `encode.py`.

---

## Background the engineer needs (read before Task 1)

**The two solvers you bridge:**
- **Production single-layer** `src/gasgiant/sim/shallow_water_ref.py` (M1/M2-core): `Grid(W,H,a)` with `a`-aware metric; `momentum_step(h,u,v,gp,omega,g,dt)` (vector-invariant, `B=gp*h+ke`, trapezoidal Coriolis); `continuity_step(h,u,v,g,dt,h_floor)` (flux-form/FCT); `vorticity`, `grad_faces`, `coriolis_trapezoidal`. Row 0 = NORTH, φ descending. h/u (H,W), v (H+1,W) poles 0.
- **M0 spike** `src/gasgiant/sim/sw_spike/` (a=1, validated, the physics reference): `operators.montgomery_2layer(h1,h2,(g1,g2))` → `(M1=g1(h1+h2), M2=g1(h1+h2)+g2*h2)`; `solver.step` (per-layer momentum+continuity+`_apply_forcing`); `solver._apply_forcing` (relaxation/drag/hypervisc/sponge/floor); `solver.balanced_test_state`; `encode.py` (top-layer render encoder).

**The M3 surgery:** add 2-layer state, `montgomery_2layer`, a Montgomery-driven `momentum_step_M`, `step_2layer`, and `apply_forcing` to `shallow_water_ref.py`; keep M1's single-layer `momentum_step`/`step` byte-identical. The pressure term changes from `−∇(gp·h)` (single layer) to `−∇M_i` (Montgomery), which is the ONLY governing-equation change; everything else is M1 operators reused per layer.

**Conventions (every task):** CPU ref first; per-field GPU↔CPU diff at `atol=2e-5` on pre-division quantities; independent a-scaling (`a=1` vs `a=6.4e6`) at every metric site; `wrapX` branch form in kernels; reuse the session `gpu` fixture; determinism via fixed iteration counts; first-NaN/negative-h trap. **Non-vacuous gates** (the project's hard-won lesson): every validation gate must be shown able to FAIL (a stable config must not grow; a no-op must be caught) before its pass is trusted.

---

## File structure

| File | Responsibility | Action |
|---|---|---|
| `src/gasgiant/sim/shallow_water_ref.py` | CPU ground truth: montgomery_2layer, momentum_step_M, step_2layer, apply_forcing, 2-layer init | Modify (append; M1 funcs byte-identical) |
| `src/gasgiant/sim/sw_gpu.py` | GPU 2-layer state + Montgomery kernel + step_2layer dispatch | Modify |
| `src/gasgiant/sim/kernels/sw_montgomery.comp` | 2-layer Montgomery → per-layer M | Create |
| `src/gasgiant/sim/kernels/sw_forcing.comp` | relaxation + drag + hypervisc + sponge + floor | Create |
| `src/gasgiant/sim/sw_encode.py` | top-layer h-anomaly + ζ₁ → RGBA tracer (promote sw_spike/encode.py) | Create |
| `tests/unit/test_m3_ref.py` | CPU unit tests (montgomery, momentum_M, step, forcing, init, conservation) | Create |
| `tests/unit/test_m3_baroclinic.py` | CRUX: β-plane growth rate + Charney-Stern + Rhines (non-vacuous) | Create |
| `tests/unit/test_m3_gpu.py` | per-field GPU↔CPU diffs + dual-path (n_layers=1 ≡ M1) | Create |
| `scripts/sw_m3_rendergate.py` | high-res GPU spin-up → encode → 4K render → blind panel vs v1.6 | Create |
| `docs/superpowers/specs/m3-verdict.md` | gate numbers + render verdict | Create |

---

## Task 1: Montgomery 2-layer coupling (CPU ref)

**Files:** Modify `shallow_water_ref.py`; Test `tests/unit/test_m3_ref.py` (create).

- [ ] **Step 1: Failing test** — matches the validated sw_spike op exactly + Montgomery is `a`-independent (it is a potential, no metric):

```python
import numpy as np
def test_montgomery_2layer_matches_spike():
    from gasgiant.sim.shallow_water_ref import montgomery_2layer
    from gasgiant.sim.sw_spike.operators import montgomery_2layer as spike_mont
    rng = np.random.default_rng(0)
    h1 = 5.0 + rng.random((16, 32)); h2 = 3.0 + rng.random((16, 32))
    M1, M2 = montgomery_2layer(h1, h2, 9.8, 0.3)
    sM1, sM2 = spike_mont(h1, h2, (9.8, 0.3))
    assert np.allclose(M1, sM1, atol=0) and np.allclose(M2, sM2, atol=0)

def test_montgomery_reduces_to_single_layer():
    """With h2=0 and g2 arbitrary, M1 = g1*h1 (the single-layer reduced-gravity pressure)."""
    from gasgiant.sim.shallow_water_ref import montgomery_2layer
    h1 = np.full((8, 8), 4.0); h2 = np.zeros((8, 8))
    M1, _ = montgomery_2layer(h1, h2, 9.8, 0.3)
    assert np.allclose(M1, 9.8 * h1)
```

- [ ] **Step 2: Run → FAIL** (`montgomery_2layer` undefined).
- [ ] **Step 3: Implement** (append):
```python
def montgomery_2layer(h1, h2, gp1, gp2):
    """Reduced-gravity Montgomery potentials for the 2-layer stack (design §2.2).
    M1 = gp1*(h1+h2); M2 = gp1*(h1+h2) + gp2*h2. A potential (no metric, a-agnostic)."""
    eta1 = h1 + h2
    M1 = gp1 * eta1
    M2 = gp1 * eta1 + gp2 * h2
    return M1, M2
```
- [ ] **Step 4: Run → PASS.**
- [ ] **Step 5: Commit** `M3: 2-layer Montgomery coupling (CPU ref)`.

---

## Task 2: Montgomery-driven momentum step (CPU ref) — M1 byte-identical

**Files:** Modify `shallow_water_ref.py`; Test `tests/unit/test_m3_ref.py`.

Generalize M1's `momentum_step` to take a precomputed Montgomery `M` (Bernoulli `B = M + ke`) instead of the hard-wired `gp*h`. M1's `momentum_step` must remain BYTE-IDENTICAL.

- [ ] **Step 1: Failing test** — `momentum_step_M` with `M = gp*h` reproduces `momentum_step` exactly:
```python
def test_momentum_step_M_reduces_to_m1():
    from gasgiant.sim.shallow_water_ref import Grid, momentum_step, momentum_step_M
    g = Grid(W=32, H=16, a=6.4e6)
    rng = np.random.default_rng(1)
    h = 1000 + rng.random((16, 32)); u = rng.random((16, 32)); v = rng.random((17, 32)); v[0]=v[16]=0
    gp = 9.8; omega = 7.292e-5; dt = 50.0
    u_a, v_a = momentum_step(h, u, v, gp, omega, g, dt)
    u_b, v_b = momentum_step_M(h, u, v, gp * h, omega, g, dt)
    assert np.array_equal(u_a, u_b) and np.array_equal(v_a, v_b)
```
- [ ] **Step 2: Run → FAIL.**
- [ ] **Step 3: Implement** `momentum_step_M` by copying M1's `momentum_step` body and replacing the single line `B = gp * h + ke` with `B = M + ke` (and dropping the `gp`/`h` pressure-only usage — `M` carries the full pressure). Leave M1's `momentum_step` untouched. Add a NOTE comment that the two share the vorticity-flux + Coriolis block and must be kept in sync (mirror the M2-core predictor note discipline). Verify the bodies match line-for-line except the `B=` line.
- [ ] **Step 4: Run → PASS** (byte-identical via `np.array_equal`).
- [ ] **Step 5: Commit** `M3: Montgomery-driven momentum step (M1 byte-identical)`.

---

## Task 3: 2-layer state + step_2layer (CPU ref)

**Files:** Modify `shallow_water_ref.py`; Test `tests/unit/test_m3_ref.py`.

- [ ] **Step 1: Failing test** — one 2-layer step on a resting balanced stack stays bounded + mass-conserving per layer; structurally mirrors `sw_spike.solver.step`:
```python
def test_step_2layer_resting_stable():
    from gasgiant.sim.shallow_water_ref import Grid, Sw2State, step_2layer, layer_mass
    g = Grid(W=32, H=16, a=6.4e6)
    st = Sw2State(g=g, omega=7.292e-5, gp1=9.8, gp2=0.3,
                  h1=np.full((16,32),1000.0), u1=np.zeros((16,32)), v1=np.zeros((17,32)),
                  h2=np.full((16,32),500.0),  u2=np.zeros((16,32)), v2=np.zeros((17,32)),
                  dt=20.0, h_floor=1.0)
    m1_0, m2_0 = layer_mass(st)
    for _ in range(5):
        st = step_2layer(st)
    assert np.isfinite(st.h1).all() and st.h1.min() > 0 and st.h2.min() > 0
    m1_1, m2_1 = layer_mass(st)
    assert abs(m1_1 - m1_0)/m1_0 < 1e-10 and abs(m2_1 - m2_0)/m2_0 < 1e-10  # flux-form mass
```
- [ ] **Step 2: Run → FAIL.**
- [ ] **Step 3: Implement** `Sw2State` dataclass (g, omega, gp1, gp2, h1/u1/v1, h2/u2/v2, dt, h_floor, and forcing fields tau_rad/tau_drag/nu4/h_eq1/h_eq2 defaulting off), `layer_mass(st)` (Σ h_i·cos_c·a²·dλ·dφ per layer), and `step_2layer(st)`:
```python
def step_2layer(st):
    M1, M2 = montgomery_2layer(st.h1, st.h2, st.gp1, st.gp2)
    u1, v1 = momentum_step_M(st.h1, st.u1, st.v1, M1, st.omega, st.g, st.dt)
    u2, v2 = momentum_step_M(st.h2, st.u2, st.v2, M2, st.omega, st.g, st.dt)
    h1 = continuity_step(st.h1, u1, v1, st.g, st.dt, st.h_floor)
    h2 = continuity_step(st.h2, u2, v2, st.g, st.dt, st.h_floor)
    st.h1, st.u1, st.v1 = h1, u1, v1
    st.h2, st.u2, st.v2 = h2, u2, v2
    apply_forcing(st)            # Task 4 (no-op until forcing fields are set)
    return st
```
(For Task 3, `apply_forcing` can be a stub `def apply_forcing(st): return None` filled in Task 4; with all forcing fields off the resting test must pass via the floor only.)
- [ ] **Step 4: Run → PASS.**
- [ ] **Step 5: Commit** `M3: 2-layer state + step_2layer`.

---

## Task 4: Forcing — relaxation + drag + hypervisc + sponge (CPU ref)

**Files:** Modify `shallow_water_ref.py`; Test `tests/unit/test_m3_ref.py`.

Port `sw_spike.solver._apply_forcing` (and `_biharmonic`, `_polar_sponge`, `_smoothstep`) — the validated forcing — into `apply_forcing(st)`, `a`-aware where metric enters (the biharmonic is a grid-normalized proxy; keep the /64 v1.6 fix). All forcing is STEP-based (τ in steps).

- [ ] **Step 1: Failing tests:**
```python
def test_relaxation_pulls_h_toward_heq():
    from gasgiant.sim.shallow_water_ref import Grid, Sw2State, apply_forcing
    g = Grid(W=16, H=8, a=6.4e6)
    st = Sw2State(g=g, omega=0.0, gp1=9.8, gp2=0.3,
                  h1=np.full((8,16),1000.0), u1=np.zeros((8,16)), v1=np.zeros((9,16)),
                  h2=np.full((8,16),500.0),  u2=np.zeros((8,16)), v2=np.zeros((9,16)),
                  dt=20.0, h_floor=1.0, tau_rad=10.0,
                  h_eq1=np.full((8,16),1100.0), h_eq2=np.full((8,16),500.0))
    apply_forcing(st)
    assert 1000.0 < st.h1.mean() < 1100.0          # moved 1/tau toward h_eq

def test_bottom_drag_only_lower_layer():
    from gasgiant.sim.shallow_water_ref import Grid, Sw2State, apply_forcing
    g = Grid(W=16, H=8, a=6.4e6)
    st = Sw2State(g=g, omega=0.0, gp1=9.8, gp2=0.3,
                  h1=np.full((8,16),1000.0), u1=np.full((8,16),5.0), v1=np.zeros((9,16)),
                  h2=np.full((8,16),500.0),  u2=np.full((8,16),5.0), v2=np.zeros((9,16)),
                  dt=20.0, h_floor=1.0, tau_drag=10.0)
    u1_before = st.u1.copy()
    apply_forcing(st)
    assert np.allclose(st.u1, u1_before)            # top layer undrag­ged
    assert st.u2.mean() < 5.0                        # lower layer drag­ged
```
- [ ] **Step 2: Run → FAIL.**
- [ ] **Step 3: Implement** `apply_forcing` + helpers, ported from `sw_spike.solver` (lines 58-105). Order: thermal relaxation (both layers), Rayleigh drag (layer 2 only), biharmonic hypervisc (/64), polar sponge (velocity→0 + h→h_eq poleward), positivity floor.
- [ ] **Step 4: Run → PASS.**
- [ ] **Step 5: Commit** `M3: 2-layer forcing (relaxation/drag/hypervisc/sponge)`.

---

## Task 5: Balanced 2-layer init + h_eq + Montgomery balance gate (CPU)

**Files:** Modify `shallow_water_ref.py`; Test `tests/unit/test_m3_ref.py`.

- [ ] **Step 1: Failing test — the Montgomery balance gate (pins the g' sign matrix):**
```python
def test_2layer_williamson2_balance_stationary():
    """A 2-layer geostrophically-balanced state stays stationary to scheme tolerance
    over several steps -- pins the Montgomery sign/coefficient matrix (design §2.2)."""
    from gasgiant.sim.shallow_water_ref import balanced_2layer_state, step_2layer
    st = balanced_2layer_state(W=64, H=32, a=6.4e6, omega=7.292e-5, gp1=9.8, gp2=0.3, u0=20.0)
    u1_0 = st.u1.copy(); h1_0 = st.h1.copy()
    for _ in range(10):
        st = step_2layer(st)
    assert np.max(np.abs(st.u1 - u1_0)) < 1e-2      # velocity stationary
    assert np.max(np.abs(st.h1 - h1_0)) / h1_0.mean() < 1e-3
```
- [ ] **Step 2: Run → FAIL.**
- [ ] **Step 3: Implement** `balanced_2layer_state` (gradient-wind balanced top layer like M1's `williamson2_state` but with the Montgomery pressure; quiescent lower layer in balance; dt from `c_gw=√(gp1·(h1+h2).max())` and the binding polar Δx — name both per the M0 lesson) and a `heq_profiles(g, ...)` helper building `h_eq_i(φ)` (repurpose `profiles.py`; a pole-to-equator tilt that is baroclinically unstable for Task 6). Equator-safe (`|u_init|` capped, gradient-wind not raw geostrophic).
- [ ] **Step 4: Run → PASS** (tune the balance tolerance to the scheme's O(dt) imbalance, documented; it pins the sign matrix — if M1/M2 are swapped or a g' sign flips, the state drifts/blows and the test fails).
- [ ] **Step 5: Commit** `M3: balanced 2-layer init + h_eq + Montgomery balance gate`.

---

## Task 6: Baroclinic instability CRUX gate (CPU, GO/NO-GO)

**Files:** Test `tests/unit/test_m3_baroclinic.py` (create). The milestone's crux — front-loaded before any GPU work. If emergent baroclinic instability does not arise at the predicted rate, the whole approach is in question; falsify here, on CPU.

- [ ] **Step 1: Non-vacuity guard FIRST** — a baroclinically STABLE stack must NOT grow:
```python
import numpy as np
def test_stable_stack_does_not_grow():
    """Control: a stack whose h_eq tilt does NOT satisfy Charney-Stern (no PV-gradient
    sign change) must NOT spontaneously grow eddies -- proves the gate can fail."""
    from gasgiant.sim.shallow_water_ref import baroclinic_test_state, step_2layer, eddy_energy
    st = baroclinic_test_state(W=96, H=48, unstable=False, seed=0)
    e0 = eddy_energy(st)
    for _ in range(400):
        st = step_2layer(st)
    assert eddy_energy(st) < 5.0 * e0      # bounded, no exponential growth
```
- [ ] **Step 2: The growth-rate gate** — an UNSTABLE tilt grows at the β-plane two-layer rate:
```python
def test_baroclinic_growth_matches_theory():
    """An unstable h_eq tilt (Charney-Stern satisfied) grows eddies exponentially at
    a rate consistent with two-layer beta-plane linear theory (within a factor ~2 --
    the discrete/spherical solver vs idealized QG theory)."""
    from gasgiant.sim.shallow_water_ref import (
        baroclinic_test_state, step_2layer, eddy_energy, predicted_growth_rate)
    st = baroclinic_test_state(W=96, H=48, unstable=True, seed=0)
    energies = []
    for n in range(600):
        st = step_2layer(st)
        if n % 20 == 0: energies.append(eddy_energy(st))
    energies = np.array(energies)
    growth = np.polyfit(np.arange(len(energies)) * 20, np.log(energies + 1e-30), 1)[0]
    sigma_theory = predicted_growth_rate(st)
    assert growth > 0, "no baroclinic growth on an unstable stack (approach falsified)"
    assert 0.5 * sigma_theory < growth < 2.0 * sigma_theory, (
        f"growth {growth:.3e} vs theory {sigma_theory:.3e} -- outside factor-2 band")
```
- [ ] **Step 3: Implement** `baroclinic_test_state` (a 2-layer stack with a vertically-sheared zonal flow / tilted interface; `unstable=True` makes the upper-layer PV gradient reverse sign — Charney-Stern — and seeds a small wavenumber perturbation; `unstable=False` a stable tilt), `eddy_energy(st)` (non-zonal KE, like `sw_spike.eddy_vorticity_std`), and `predicted_growth_rate(st)` (two-layer β-plane Phillips/Charney-Stern fastest-growing-mode `σ` from `gp2`, the shear, `β=2Ω cosφ/a`, and `L_D`). Cite the formula in a comment.
- [ ] **Step 4: Run the crux** — `pytest tests/unit/test_m3_baroclinic.py -v -s`.
  - **PASS** (stable doesn't grow; unstable grows in the factor-2 band) → emergent baroclinic instability validated; proceed.
  - **FAIL** → STOP. If the unstable stack does not grow, the explicit 2-layer setup is not capturing baroclinic instability — debug (systematic-debugging) the Montgomery coupling / shear / Charney-Stern setup before any GPU work. Record in `m3-verdict.md`. Do NOT proceed to GPU on a falsified crux.
- [ ] **Step 5: Commit** `M3: baroclinic instability crux gate (growth rate vs theory, non-vacuous)`.

---

## Task 7: Conservation / budget closure + determinism (CPU)

**Files:** Test `tests/unit/test_m3_ref.py`.

- [ ] **Step 1: Failing tests:**
```python
def test_mass_conserved_per_layer_unforced():
    """Unforced (tau off), per-layer mass conserves to round-off (flux-form)."""
    from gasgiant.sim.shallow_water_ref import baroclinic_test_state, step_2layer, layer_mass
    st = baroclinic_test_state(W=64, H=32, unstable=True, seed=1)
    st.tau_rad = 0.0; st.tau_drag = 0.0; st.nu4 = 0.0   # pure dynamics, no sponge mass src
    m0 = layer_mass(st)
    for _ in range(50): st = step_2layer(st)
    m1 = layer_mass(st)
    assert abs(m1[0]-m0[0])/m0[0] < 1e-9 and abs(m1[1]-m0[1])/m0[1] < 1e-9

def test_energy_budget_closes():
    """Forced-dissipative run: d(energy)/dt = forcing_in - drag_out - hypervisc_out to
    within a small unaccounted residual (the correctness signal is residual ~ 0)."""
    # measure total_energy each step; accumulate diagnosed forcing/drag/hypervisc work;
    # assert |dE - (F_in - D_out)| / |dE| < 0.05 over the run.
    ...  # implement with energy + per-term work diagnostics
```
(Note: `test_energy_budget_closes` needs per-term work diagnostics; if the polar sponge injects mass, exclude sponge latitudes from the mass test or run the mass test with the sponge off, as shown.)
- [ ] **Step 2: Run → FAIL.**
- [ ] **Step 3: Implement** `total_energy_2layer`, per-term work diagnostics, and a determinism test (byte-identical SHA1 over a fixed run; n_layers semantics). Add the v1.6 no-op guards (each forcing param changes output; fields evolve).
- [ ] **Step 4: Run → PASS.**
- [ ] **Step 5: Commit** `M3: conservation + budget-closure + determinism gates`.

---

## Task 8: GPU 2-layer port + Montgomery kernel + dual-path (GPU)

**Files:** Create `kernels/sw_montgomery.comp`, `kernels/sw_forcing.comp`; Modify `sw_gpu.py`; Test `tests/unit/test_m3_gpu.py` (create).

- [ ] **Step 1: Failing tests** — per-field GPU↔CPU diff at `atol=2e-5` (pre-division) for `sw_montgomery` (M1,M2), the per-layer momentum/continuity (reuse M1 kernels with M_i), and forcing; a-scaling; full 2-layer step (1-step + N-step); **dual-path**: `n_layers=1` byte-identical to the M1 single-layer GPU path (`np.array_equal` + SHA1, mirror `test_dual_path.py`); determinism SHA1.
- [ ] **Step 2: Implement** `sw_montgomery.comp` (M1=g1(h1+h2), M2=g1(h1+h2)+g2 h2 — trivial, per-cell), wire the existing M1 momentum/continuity kernels to run per layer fed by M_i (the pressure kernel takes M instead of gp*h), `sw_forcing.comp` (relaxation/drag/hypervisc/sponge/floor). Extend `SwGpuSolver` with `n_layers`, layer-2 textures, `gp1/gp2/tau_rad/tau_drag/h_eq` params, and a `step_2layer` dispatch guarded so `n_layers=1` is the unchanged M1 path. Checkpoint version bump + layer-indexed keys.
- [ ] **Step 3: Run → PASS** at `atol=2e-5`; dual-path byte-identical; determinism SHA1 stable.
- [ ] **Step 4: Commit** `M3: GPU 2-layer port + Montgomery/forcing kernels + dual-path`.

---

## Task 9: Render-fidelity gate + verdict (GPU) — the project's go/no-go

**Files:** Create `src/gasgiant/sim/sw_encode.py`, `scripts/sw_m3_rendergate.py`, `docs/superpowers/specs/m3-verdict.md`.

- [ ] **Step 1: Encoder** — promote `sw_spike/encode.py` to `sw_encode.py`: top-layer h-anomaly → cloud altitude/color; **ζ₁ → the primary detail/contrast channel** (the v1.6 hero signal); advected dye → existing tracer. Feed the unchanged `derive.comp`. Unit-test the encoder maps fields to a sane RGBA range.
- [ ] **Step 2: Render gate script** — `sw_m3_rendergate.py`: high-res GPU emergent spin-up (≥1024×512) to equilibration (or the seeded-mode finishing pass if the spin-up kill-gate exceeds budget), encode the top layer, render at ≥4096 via `maps.derive_from_tracer`, and run the v1.6 blind 3-judge forced-choice panel (mirror `scripts/swp_killgate.py` / the M0.5 methodology) + `measure_morphology`/`coher`.
- [ ] **Step 3: Run the gate.** Record steps-to-equilibration vs budget and the panel result.
  - **PREFER M3 (ties fail)** OR `coher` measurably toward 0.62 → M3 render validated.
  - **LOSE** → the emergent un-nudged render does not beat painted-jet v1.6 (master design R8). Record honestly in the verdict; this is the real go/no-go — try the seeded finishing-pass mode before concluding, but do not fake a win.
- [ ] **Step 4: Write `m3-verdict.md`** — all gate numbers (Montgomery balance, baroclinic growth vs theory, Rhines spacing, conservation residuals, determinism), the spin-up budget, and the render verdict with the honest claim.
- [ ] **Step 5: Commit** `M3: render-fidelity gate + verdict`.

---

## After all tasks: final whole-implementation review (Opus), then `superpowers:finishing-a-development-branch`.

---

## Validation summary (gates)

- **(a)** Montgomery balance: 2-layer Williamson-2 stationary (Task 5) — pins the g' sign matrix.
- **(b)** Baroclinic growth vs β-plane theory + Charney-Stern, NON-VACUOUS (stable doesn't grow) (Task 6 CRUX).
- **(c)** Emergent jet spacing vs `L_Rhines` (Task 9 spin-up / a dedicated check in Task 6/7).
- **(d)** Finite-amplitude GRS-scale vortex coherence (add to Task 6 or 7).
- **(e)** Mass to round-off per layer; energy budget closes (Task 7).
- **(f)** Determinism SHA1; n_layers=1 ≡ M1 byte-identical (Tasks 7, 8).
- **(g)** ≥4K blind-panel render prefers M3 to v1.6 (Task 9) — the project's actual point.

## Risks (for the adversarial plan review)

1. **Montgomery sign matrix** — gate (a) pins it; the plan diffs `montgomery_2layer` vs sw_spike AND validates balance. The most error-prone part.
2. **Explicit barotropic CFL** — `c_gw=√(gp1·H)` with the binding polar Δx can make dt tiny → spin-up budget risk. Task 5 names both; if dt is infeasible at production res, the spin-up kill-gate (Task 9) routes to seeded mode (or, last resort, the deferred semi-implicit).
3. **Baroclinic gate calibration** — the factor-2 theory band (Task 6) must be justified (discrete spherical solver vs idealized QG); the predicted_growth_rate formula must be the right two-layer mode. Non-vacuity guard (stable doesn't grow) is mandatory.
4. **Render gate may refute the approach** (R8) — emergent ζ₁ may be less filament-rich than v1.6's nudged field; seeded finishing-pass is the fallback; the verdict is honest either way.
5. **Equatorial init** — gradient-wind/Gill closure, capped `|u_init|`; validate it doesn't seed a spurious equatorial instability.
6. **Spin-up budget** — emergent may be too slow at production res (R5); seeded fallback first-class.
