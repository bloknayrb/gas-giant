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

def test_momentum_step_M_decoupled_matches_spike():
    """The reduction test above only exercises M=gp*h. Validate the NON-reduction
    path (M != gp*h, the real 2-layer case) against the validated M0 spike's
    _layer_momentum on the LOWER layer (a=1 so the spike's f0 == 2*omega)."""
    from gasgiant.sim.shallow_water_ref import Grid, montgomery_2layer, momentum_step_M
    from gasgiant.sim.sw_spike.grid import Grid as SGrid
    from gasgiant.sim.sw_spike.solver import _layer_momentum
    W, H = 32, 16; omega = 0.5; dt = 30.0
    rng = np.random.default_rng(2)
    h1 = 5 + rng.random((H, W)); h2 = 3 + rng.random((H, W))
    u2 = rng.random((H, W)); v2 = rng.random((H+1, W)); v2[0]=v2[H]=0
    _, M2 = montgomery_2layer(h1, h2, 9.8, 0.3)          # M2 != gp*h2
    gp = Grid(W=W, H=H, a=1.0); sg = SGrid(W, H)
    u_a, v_a = momentum_step_M(h2, u2, v2, M2, omega, gp, dt)
    u_b, v_b = _layer_momentum(h2, u2, v2, M2, 2*omega, sg, dt)   # spike f0 = 2*omega
    assert np.allclose(u_a, u_b, atol=1e-12) and np.allclose(v_a, v_b, atol=1e-12)
```
- [ ] **Step 2: Run → FAIL.**
- [ ] **Step 3: Implement** `momentum_step_M` by copying M1's `momentum_step` body and replacing the single line `B = gp * h + ke` with `B = M + ke`. Per review: M1's `momentum_step` uses `gp`/`h` in the pressure path in EXACTLY one place (`B = gp*h + ke`, confirmed at `shallow_water_ref.py:1006`); `gp`/`h` appear nowhere else (vorticity flux, Coriolis, KE are all independent), so the single-line swap is correct and complete. In `momentum_step_M`, `h` is now vestigial (used only for `H, W = h.shape`) — note this in a comment so no reader assumes `h` and `M` must be consistent. Leave M1's `momentum_step` untouched; add a sync NOTE (mirror the M2-core predictor note discipline).
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
    # ADVERSARIAL-REVIEW FIX (MAJOR): use continuity_step_CONSERVATIVE, not
    # continuity_step. M1's continuity_step floor-clamps with np.maximum, which
    # INJECTS mass at the floor (the M2-T5 leak) -- it would silently violate the
    # per-layer 1e-9 mass gate (Task 7) on an eddying unstable stack. The
    # conservative variant makes the clamp a no-op; pair it with the loud
    # positivity guard. (M1's continuity_step is kept only for the n_layers=1
    # byte-identical path -- Task 8.)
    h1 = continuity_step_conservative(st.h1, u1, v1, st.g, st.dt, st.h_floor)
    h2 = continuity_step_conservative(st.h2, u2, v2, st.g, st.dt, st.h_floor)
    assert_positivity(h1, st.h_floor); assert_positivity(h2, st.h_floor)
    st.h1, st.u1, st.v1 = h1, u1, v1
    st.h2, st.u2, st.v2 = h2, u2, v2
    apply_forcing(st)            # Task 4 (no-op until forcing fields are set)
    return st
```
(For Task 3, `apply_forcing` can be a stub `def apply_forcing(st): return None` filled in Task 4; with all forcing fields off the resting test must pass via the floor only. `continuity_step_conservative` and `assert_positivity` already exist in `shallow_water_ref.py` from M2-core.)
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
- [ ] **Step 3: Implement** `balanced_2layer_state` (gradient-wind balanced top layer like M1's `williamson2_state` but with the Montgomery pressure; quiescent lower layer in balance) and a `heq_profiles(g, ...)` helper building `h_eq_i(φ)`. Two review-mandated corrections:
  - **dt MUST use the production `a`-AWARE `dx_min`** (copy from M1 `williamson2_state`): `cos_min = max(g.cos_c.min(), 1e-6); dx_min = min(cos_min * g.a * g.dlam, g.a * g.dphi); dt = dt_safety * dx_min / c_gw` with `c_gw = √(gp1·(h1+h2).max())` (the barotropic external-mode speed — the binding wave). Do **NOT** copy the M0 spike's `balanced_test_state` line (it is a=1: `min(cos_c.min()*dlam, dphi)` with no `a`) — that makes dt too large by ~`a≈6.4e6` and the run explodes instantly.
  - **`h_eq` lives in `sw_spike/init.py`, NOT `profiles.py`.** Repurpose `sw_spike/init.py` (`h_eq_profile`, `_gradient_wind_u`, `emergent_init`) — `profiles.py` builds v1.6 kinematic-render `u(φ)` LUTs and has no `h_eq` / gradient-wind inversion. Make the tilt a pole-to-equator profile that is baroclinically unstable for Task 6. Equator-safe (`|u_init|` capped, gradient-wind not raw geostrophic).
- [ ] **Step 4: Run → PASS** (tune the balance tolerance to the scheme's O(dt) imbalance, documented; it pins the sign matrix — if M1/M2 are swapped or a g' sign flips, the state drifts/blows and the test fails).
- [ ] **Step 5: Commit** `M3: balanced 2-layer init + h_eq + Montgomery balance gate`.

---

## Task 6: Baroclinic instability CRUX gate (CPU, GO/NO-GO)

**Files:** Test `tests/unit/test_m3_baroclinic.py` (create). The milestone's crux — front-loaded before any GPU work. **This task was substantially corrected after adversarial review (2 FATAL physics findings); read the physics notes carefully — a literal naive implementation builds a possibly-STABLE state and produces a false NO-GO that would wrongly kill the milestone.**

**Physics (get this right — the review caught both):**
- **Charney-Stern is satisfied in the LOWER layer, not the upper.** For a standard *eastward* vertical shear `U_s ≡ (U₁−U₂)/2 > 0` (top layer faster, realized as a poleward interface tilt in Montgomery balance), the QG PV gradients are `β₁ = β + (f₀²/g'₂H₁)(U₁−U₂)` (increased, stays positive) and `β₂ = β − (f₀²/g'₂H₂)(U₁−U₂)` (decreased, goes **negative** when the shear is supercritical). The sign change is in **β₂ (lower layer)**. The `unstable` switch toggles the **supercriticality** `ξ = (U₁−U₂)/U_crit`, `U_crit = β·g'₂·H₂/f₀²`; `unstable=True` ⇒ `ξ > 1` (β₂<0), `unstable=False` ⇒ `ξ < 1` (both gradients positive).
- **Theory target = the f-plane Phillips closed form** (β-plane has no clean closed-form max and the discrete spherical solver deviates from idealized β-plane QG by more than a factor 2). With `k_d² = 2f₀²/(g'₂H)` (deformation radius `L_D = 1/k_d = √(g'₂H)/(f₀√2)`): the f-plane growth `σ(K) = U_s·K·√[(k_d²−K²)/(k_d²+K²)]`, maximized at `K²_max = k_d²(√2−1)` giving **`σ_max = U_s·k_d·√(3−2√2) ≈ 0.31·U_s·k_d = 0.31·U_s/L_D`**.
- **The IC must be BALANCED and the perturbation a BALANCED PV/interface perturbation** (the M2-adv lesson: an unbalanced perturbation radiates gravity waves that swamp the eddy signal). Build the base from `balanced_2layer_state` (Task 5, validated stationary); impose the shear via the balanced interface tilt; perturb the interface `h₂` at `K_max` with the perturbation velocity in geostrophic balance.
- **Diagnose on eddy interface-height variance** (non-zonal var of `h₂`), NOT kinetic energy — interface variance grows with the baroclinic mode but is not excited by gravity waves to leading order.

- [ ] **Step 1: Non-vacuity via the RATIO control** (an absolute "<5×" bound is vacuous — it passes whether or not the mechanism works). Run the SAME pipeline on stable and unstable configs; a real instability must SEPARATE them:
```python
import numpy as np
def _growth_rate(st, step_2layer, eddy_var, n_steps, sample=10):
    """Fit the exponential growth rate over the LINEAR window only: slide a window,
    find the longest constant-log-slope span (plateau), fit it, require R^2>0.98.
    Returns (rate, r2) — rate<=0 or r2<0.98 means 'no clean exponential' (a stable or
    saturated record cannot yield a confident positive rate)."""
    series = []
    for n in range(n_steps):
        st = step_2layer(st)
        if n % sample == 0:
            series.append(eddy_var(st))
    t = np.arange(len(series)) * sample
    y = np.log(np.array(series) + 1e-30)
    best = (0.0, 0.0)
    w = max(8, len(y) // 4)
    for i in range(0, len(y) - w):
        sl, inter = np.polyfit(t[i:i+w], y[i:i+w], 1)
        resid = y[i:i+w] - (sl * t[i:i+w] + inter)
        ss_tot = np.sum((y[i:i+w] - y[i:i+w].mean())**2) + 1e-30
        r2 = 1.0 - np.sum(resid**2) / ss_tot
        if r2 > best[1]:
            best = (sl, r2)
    return best

def test_baroclinic_growth_is_nonvacuous_and_matches_fplane_theory():
    from gasgiant.sim.shallow_water_ref import (
        baroclinic_test_state, step_2layer, eddy_interface_var,
        predicted_growth_rate_fplane, efold_steps_estimate)
    st_u = baroclinic_test_state(W=96, H=48, unstable=True, seed=0)
    st_s = baroclinic_test_state(W=96, H=48, unstable=False, seed=0)
    # Size the run to span >=3 e-foldings after the transient (fail loudly if budget can't).
    n_steps = max(800, 4 * efold_steps_estimate(st_u))
    assert n_steps < 20000, "e-fold time too long at this resolution/g' -- lower gp1 or seed-only"
    g_u, r2_u = _growth_rate(st_u, step_2layer, eddy_interface_var, n_steps)
    g_s, r2_s = _growth_rate(st_s, step_2layer, eddy_interface_var, n_steps)
    sigma = predicted_growth_rate_fplane(st_u)
    print(f"\n[m3-baroclinic] unstable rate={g_u:.3e} (R2={r2_u:.3f}), "
          f"stable rate={g_s:.3e} (R2={r2_s:.3f}), f-plane sigma={sigma:.3e}")
    # (1) real exponential growth on the unstable stack:
    assert g_u > 0 and r2_u > 0.98, "no clean exponential growth (approach falsified)"
    # (2) asymmetric physical band: loose lower bound (dissipation slows growth),
    #     tight upper bound (catches numerical blow-up faster than the inviscid limit):
    assert 0.3 * sigma < g_u < 1.5 * sigma, f"rate {g_u:.3e} vs f-plane {sigma:.3e}"
    # (3) NON-VACUITY: unstable must separate from stable by the same pipeline:
    assert g_u > 5.0 * max(g_s, 0.0) + 1e-12, (
        f"gate vacuous: unstable {g_u:.3e} not >> stable {g_s:.3e}")
```
- [ ] **Step 2: Implement** the helpers:
  - `baroclinic_test_state(W, H, unstable, seed, a=6.4e6, ...)` — balanced base (reuse `balanced_2layer_state`) with an eastward shear via a poleward `h₂(φ)` tilt sized so `ξ>1` (unstable) or `ξ<1` (stable) relative to `U_crit = β·gp2·H₂/f₀²` at the test latitude; add a balanced interface perturbation at `K_max = k_d√(√2−1)` (amplitude ~1e-3 of mean), perturbation velocity in geostrophic balance. State the chosen test latitude (mid-latitude band away from the polar sponge and the equator).
  - `eddy_interface_var(st)` — variance of `h₂ − zonal_mean(h₂)`.
  - `predicted_growth_rate_fplane(st)` — `σ_max = 0.31·U_s·√(2·f₀²/(gp2·H))`, `U_s=(U₁−U₂)/2`, `f₀=2Ω sinφ_test`, `H` the layer-mean depth. Cite the derivation in a comment.
  - `efold_steps_estimate(st)` — `1/(σ_max·dt)` (so the run length spans enough e-foldings given the tiny explicit dt).
- [ ] **Step 3: Run the crux** — `pytest tests/unit/test_m3_baroclinic.py -v -s`.
  - **PASS** (clean exponential on unstable, R²>0.98, in the [0.3,1.5]σ band, AND unstable ≫ stable) → emergent baroclinic instability validated; proceed.
  - **FAIL** → STOP and apply systematic-debugging. Distinguish the cases: (a) unstable doesn't grow ⇒ the Montgomery/shear/Charney-Stern *setup* is wrong (check β₂<0 in the LOWER layer, check `U_crit`) — most likely a setup bug, not a falsified approach; (b) grows but outside the band ⇒ check the f-plane formula / test latitude / hyperviscosity damping K_max; (c) e-fold time exceeds budget ⇒ lower `gp1` (larger dt) or seed the single K_max mode. Record findings in `m3-verdict.md`. Do NOT proceed to GPU on a falsified crux, and do NOT widen the band to force a pass.
- [ ] **Step 4: Add the finite-amplitude vortex coherence check** (gate d) — a GRS-scale balanced vortex (Rossby>0.1) stays coherent (bounded, no outcropping NaN) over a multi-hundred-step run. Small, lives here.
- [ ] **Step 5: Commit** `M3: baroclinic crux gate (f-plane Phillips, ratio-control non-vacuity, balanced PV perturbation)`.

---

## Task 7: Conservation / budget closure + determinism (CPU)

**Files:** Test `tests/unit/test_m3_ref.py`.

- [ ] **Step 1: Failing tests:**
```python
def test_mass_conserved_per_layer_unforced():
    """Unforced (tau off, sponge off), per-layer mass conserves to round-off.
    Achievable ONLY because step_2layer uses continuity_step_conservative (Task 3
    fix); with M1's clamping continuity_step this would leak at the floor (M2-T5).
    The polar sponge also injects mass (it relaxes h->h_eq), so it must be off here."""
    from gasgiant.sim.shallow_water_ref import baroclinic_test_state, step_2layer, layer_mass
    st = baroclinic_test_state(W=64, H=32, unstable=True, seed=1)
    st.tau_rad = 0.0; st.tau_drag = 0.0; st.nu4 = 0.0; st.sponge_rate = 0.0  # pure dynamics
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

**Files:** Create `kernels/sw_montgomery.comp`, `kernels/sw_forcing.comp`; Modify `sw_gpu.py`, `kernels/sw_bernoulli.comp`; Test `tests/unit/test_m3_gpu.py` (create).

**PRIOR ART (review-flagged — do NOT rebuild from scratch):** a complete **throwaway** GPU 2-layer solver already exists at `src/gasgiant/sim/sw_gpu_probe/solver.py` (`SwpSolver`) with validated `swp_*` kernels (`swp_grad_montgomery.comp`, `swp_momentum.comp`, `swp_continuity.comp`, `swp_forcing.comp`, `swp_divergence.comp`, `swp_vorticity.comp`, `swp_common.glsl`). It is **a=1** (probe). M3's genuine new work is **promoting that validated GLSL into the a-aware production `sw_gpu.py`**, NOT reimplementing it. Diff the promoted kernels against the M3 CPU ref; reuse the probe's kernel bodies, adding the planetary-radius `a` to every metric site (as M1 did when graduating the CPU spike).

**THE PRESSURE SEAM (review-corrected):** the production M1 momentum kernel takes pre-computed Bernoulli-gradient textures; the `gp*h` term lives in **`sw_bernoulli.comp`** (`B = u_gp*h + ke`), NOT the momentum kernel. So the Montgomery injection point is the **Bernoulli stage**: feed `B = M_i + ke`. The momentum + continuity kernels are M-agnostic and reuse unchanged.

- [ ] **Step 1: Failing tests** — per-field GPU↔CPU diff at `atol=2e-5` (pre-division) for `sw_montgomery` (M1,M2), the Bernoulli stage with `B=M+ke`, per-layer momentum/continuity, and forcing; a-scaling; full 2-layer step (1-step + N-step); **dual-path**: `n_layers=1` byte-identical to the M1 single-layer GPU path (`np.array_equal` + SHA1, mirror `test_dual_path.py`); determinism SHA1.
- [ ] **Step 2: Implement** `sw_montgomery.comp` (M1=g1(h1+h2), M2=g1(h1+h2)+g2 h2 — per-cell), make `sw_bernoulli.comp` accept a Montgomery `M` texture (so `B = M + ke`; for the single-layer path `M = gp*h` reproduces the current kernel byte-for-byte), promote `swp_continuity`/`swp_momentum`/`swp_forcing` to a-aware production kernels (diff vs the M3 CPU ref). Extend `SwGpuSolver` with `n_layers`, layer-2 textures, `gp1/gp2/tau_rad/tau_drag/sponge_rate/h_eq` params. **Dual-path guard (review-mandated):** `n_layers==1` HARD-BRANCHES to the unchanged M1 `step` / M1 kernel chain — NOT a 2-layer path with `h2=0`; the layer-2 textures and `sw_montgomery`/`sw_forcing` kernels are unreachable when `n_layers==1`, and that path uses M1's clamping `continuity_step` (byte-identical), even though the 2-layer path uses the conservative variant. The dual-path test asserts the **kernel dispatch list** for `n_layers=1` equals M1's (not just output equality). Checkpoint version bump + layer-indexed keys.
- [ ] **Step 3: Run → PASS** at `atol=2e-5`; dual-path byte-identical (output AND dispatch list); determinism SHA1 stable.
- [ ] **Step 4: Commit** `M3: GPU 2-layer port (promote swp_* to a-aware) + Bernoulli-M seam + hard-branch dual-path`.

---

## Task 9: Render-fidelity gate + verdict (GPU) — the project's go/no-go

**Files:** Create `src/gasgiant/sim/sw_encode.py`, `scripts/sw_m3_rendergate.py`, `docs/superpowers/specs/m3-verdict.md`.

- [ ] **Step 1: Encoder** — promote `sw_spike/encode.py` to `sw_encode.py`: top-layer h-anomaly → cloud altitude/color; **ζ₁ → the primary detail/contrast channel** (the v1.6 hero signal); advected dye → existing tracer. Feed the unchanged `derive.comp`. Unit-test the encoder maps fields to a sane RGBA range.
- [ ] **Step 2: Render gate script** — `sw_m3_rendergate.py` (mirror `scripts/swp_killgate.py`): high-res GPU emergent spin-up (≥1024×512) to equilibration (or the seeded-mode finishing pass if the spin-up kill-gate exceeds budget), encode the top layer, render at ≥4096 via `maps.derive_from_tracer`, and compute `measure_morphology`/`coher` for both M3 and v1.6 + write the blind-comparison PNG.
- [ ] **Step 3: Run the gate.** Record steps-to-equilibration vs budget. **Two-tier verdict (review-corrected — the blind 3-judge panel is human-only; an autonomous agent cannot execute it):**
  - **AUTONOMOUS gate (agent-runnable, requires a real GL4.3 GPU — skips otherwise):** the non-vacuity guard `eddy_vorticity_std ≥ 1.0` (the emergent solver actually produced eddies, mirroring `swp_killgate.py`) **AND** `coher` measurably moved toward the 0.62 reference vs v1.6's 0.384. This is the gate the implementer/agent reports.
  - **MANUAL confirmation (human, or an explicitly-specified vision-model judge):** the v1.6 blind 3-judge forced-choice panel on the written PNG — must PREFER M3 (ties fail). Flag this clearly as a step OUTSIDE the agent's reach; the script writes the PNG + blind key and stops at "PENDING BLIND PANEL".
  - **LOSE** (coher does not improve, or the manual panel rejects) → the emergent un-nudged render does not beat painted-jet v1.6 (master design R8). Record honestly in the verdict; try the seeded finishing-pass mode before concluding, but do not fake a win.
- [ ] **Step 4: Write `m3-verdict.md`** — all gate numbers (Montgomery balance, baroclinic growth vs theory, Rhines spacing, conservation residuals, determinism), the spin-up budget, and the render verdict with the honest claim.
- [ ] **Step 5: Commit** `M3: render-fidelity gate + verdict`.

---

## After all tasks: final whole-implementation review (Opus), then `superpowers:finishing-a-development-branch`.

---

## Validation summary (gates)

- **(a)** Montgomery balance: 2-layer Williamson-2 stationary (Task 5) — pins the g' sign matrix.
- **(b)** Baroclinic growth vs **f-plane Phillips** `σ_max=0.31 U_s/L_D` (asymmetric [0.3,1.5]σ band) + Charney-Stern in the **lower** layer, NON-VACUOUS via the **unstable≫stable ratio** control (Task 6 CRUX).
- **(c)** Emergent jet spacing vs `L_Rhines` (Task 9 spin-up / a dedicated check in Task 6/7).
- **(d)** Finite-amplitude GRS-scale vortex coherence (Task 6 Step 4).
- **(e)** Mass to round-off per layer (via `continuity_step_conservative`); energy budget closes (Task 7).
- **(f)** Determinism SHA1; n_layers=1 ≡ M1 byte-identical (output + dispatch list) (Tasks 7, 8).
- **(g)** Autonomous: `eddy_vorticity_std≥1.0` + `coher` toward 0.62; Manual: ≥4K blind-panel prefers M3 (Task 9).

## Risks (resolved by the adversarial plan review — fixes folded in)

1. **Crux physics: Charney-Stern is LOWER-layer (was FATAL — FIXED).** The original Task 6 put the PV reversal in the upper layer; for eastward shear it is β₂<0 (lower). Fixed: `unstable` toggles supercriticality `ξ=(U₁−U₂)/U_crit`, `U_crit=β gp2 H₂/f₀²`.
2. **Crux theory: f-plane closed form (was FATAL — FIXED).** β-plane has no closed-form max and the discrete spherical solver deviates >2×. Fixed: target the **f-plane Phillips** `σ_max=0.31 U_s/L_D`, asymmetric band, balanced PV perturbation, interface-variance diagnostic, plateau-window fit (R²>0.98), ratio-based non-vacuity.
3. **Per-layer conservation (was MAJOR — FIXED).** `step_2layer` uses `continuity_step_conservative` + positivity guard (M1's clamping `continuity_step` injects mass — the M2-T5 leak).
4. **Dual-path (was MAJOR — FIXED).** `n_layers=1` hard-branches to the unchanged M1 `step`/kernel chain (not a `h2=0` 2-layer path); test asserts the dispatch list.
5. **GPU prior art + pressure seam (was MAJOR — FIXED).** Promote the existing `sw_gpu_probe`/`swp_*` 2-layer kernels to a-aware production (don't rebuild); the Montgomery seam is `sw_bernoulli.comp` (`B=M+ke`), not the momentum kernel.
6. **a-aware dt (was MAJOR — FIXED).** Task 5 uses the production `dx_min=min(cos_min·a·dlam, a·dphi)`; copying the spike's a=1 form makes dt too large by ~a (explodes).
7. **Render gate automatability (was HIGH — FIXED).** Autonomous gate = `eddy_vorticity_std≥1.0` + coher-toward-0.62 (needs a real GPU); the blind 3-judge panel is flagged human-only.
8. **`h_eq` source (FIXED).** Lives in `sw_spike/init.py` (gradient-wind), not `profiles.py`.
9. **Open — explicit barotropic CFL / spin-up budget** — `c_gw=√(gp1·H)` with the polar Δx can make dt tiny; if infeasible at production res, the spin-up kill-gate routes to seeded mode (or, last resort, the deferred semi-implicit). Emergent may be too slow (R5); seeded fallback is first-class.
10. **Open — equatorial init** — gradient-wind/Gill closure, capped `|u_init|`; validate it doesn't seed a spurious equatorial instability.
