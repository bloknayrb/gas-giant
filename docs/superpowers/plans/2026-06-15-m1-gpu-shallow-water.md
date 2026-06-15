# Milestone 1 — GPU C-grid Shallow-Water Core Solver Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A single-layer, equirect-only, GPU **Arakawa C-grid** reduced-gravity shallow-water solver, validated per-field against a NumPy CPU reference and against the Williamson et al. (1992) test 2, with the split solution-accuracy + self-budget-closure gate, determinism, and byte-exact checkpoint.

**Architecture:** The validated M0 CPU operators (`src/gasgiant/sim/sw_spike/operators.py`) graduate into a clean single-layer **`shallow_water_ref.py`** (adds planetary radius `a`, drops the 2nd layer, adds the Williamson-2 analytic state). Each C-grid operator is then ported to a moderngl compute kernel and **diffed per-field against the reference** (the v1.6 GPU-vs-CPU-ground-truth discipline). A Python `SwGpuSolver` wires the kernels into the explicit step. Validation: per-field diffs → 1-step → N-step GPU≈CPU → Williamson-2 balance + l2 → conservation budget closure → determinism/hash → checkpoint.

**Tech Stack:** Python 3.13, NumPy (CPU reference + tests), moderngl GPU compute (R32F textures, `gpu.compute`/`bind_to_image`/`run`/`memory_barrier`/`read_texture`), pytest. Existing GPU API in `src/gasgiant/gl/context.py`.

**Scope (spec §5 M1):** SINGLE layer, equirect ONLY, explicit gravity-wave step, trapezoidal Coriolis, flux-form FCT continuity, vector-invariant momentum. NOT in M1: 2nd layer (M3), semi-implicit (M2), AE polar patches (M4), render/art-direction (M5). Behind `solver.type` is NOT wired in M1 — M1 is a standalone validated solver module exercised by tests/scripts; production wiring is M5.

**Conventions (inherited from M0, fixed once):**
- Arakawa C-grid on a lon-lat sphere, **planetary radius `a` now explicit** (M0 was a=1; M1 adds it to the metric — carried M0 caveat #1). Nondimensional default `a=1.0` keeps tests comparable; the metric divides by `a`.
- Rows descending (row 0 = north). `h` at centers `(H,W)`; `u` at east faces `(H,W)`; `v` at meridional faces `(H+1,W)`, pole faces 0; `ζ` at corners `(H+1,W)`.
- Single-layer reduced gravity: pressure term `−g'∇h` (Montgomery collapses to `M = g'·h`).
- GPU: each staggered field is its own R32F texture sized `(W, H)` or `(W, H+1)` (moderngl `texture2d` takes `(width, height)`). `repeat_x=True` (periodic lon), `repeat_y=False`. Read via `texelFetch`; write via `imageStore`. v-face/corner textures have height `H+1`.
- **Per-field diff tolerance:** GPU vs CPU reference `atol=2e-5` (R32F single precision; the v1.6 cross-validation convention). Never loosen without reporting.

---

## File structure

- Create `src/gasgiant/sim/shallow_water_ref.py` — CPU single-layer C-grid reference (graduated M0 operators + radius `a` + Williamson-2 analytic state + one-step driver). The authoritative ground truth.
- Create `src/gasgiant/sim/kernels/sw_common.glsl` — C-grid metric helpers (latitudes, cosφ at centers/faces, radius, indexing helpers).
- Create `src/gasgiant/sim/kernels/sw_divergence.comp` — `∇·(h u)` at centers.
- Create `src/gasgiant/sim/kernels/sw_grad.comp` — `g'∇h` at faces (single-layer pressure gradient).
- Create `src/gasgiant/sim/kernels/sw_vorticity.comp` — `ζ` at corners.
- Create `src/gasgiant/sim/kernels/sw_continuity.comp` — FCT flux-form thickness update.
- Create `src/gasgiant/sim/kernels/sw_momentum.comp` — vector-invariant momentum + trapezoidal Coriolis.
- Create `src/gasgiant/sim/sw_gpu.py` — `SwGpuState` (textures) + `SwGpuSolver` (kernel wiring, `step`, readback, checkpoint).
- Create `tests/unit/test_sw_gpu.py` — per-field GPU-vs-CPU diffs, Williamson-2, N-step agreement, conservation, determinism, checkpoint.
- Create `scripts/sw_m1_williamson.py` — Williamson-2 report + conservation budget print.

Kernels are pure (one responsibility each), mirroring the CPU reference function-for-function so per-field diffing is exact.

---

## Task 1: CPU reference — single-layer C-grid operators (graduate M0 + radius a)

**Files:**
- Create: `src/gasgiant/sim/shallow_water_ref.py`
- Test: `tests/unit/test_sw_gpu.py`

**Context:** Port the validated M0 operators from `src/gasgiant/sim/sw_spike/{grid,operators}.py` into a single-layer reference. Single layer ⇒ Montgomery is just `M = g'·h`, so the pressure gradient is `grad_faces(g'·h)`. Add planetary radius `a` to every metric: zonal grad denominator becomes `a·cosφ·dλ`, meridional `a·dφ`, divergence prefactor `1/(a·cosφ)`, vorticity `1/(a·cosφ)`. Keep the M0 sign conventions (validated) verbatim; just thread `a` through.

- [ ] **Step 1: Write the failing tests** (these are the M0 analytic checks, re-asserted with `a`)

```python
# tests/unit/test_sw_gpu.py
import numpy as np
import pytest
from gasgiant.sim import shallow_water_ref as ref


def test_ref_divergence_solid_body_zero():
    g = ref.Grid(W=64, H=32, a=1.0)
    h = np.ones((32, 64)); u = np.full((32, 64), 0.3); v = np.zeros((33, 64))
    assert np.max(np.abs(ref.divergence_hu(h, u, v, g))) < 1e-12


def test_ref_vorticity_rigid_rotation():
    g = ref.Grid(W=128, H=64, a=1.0)
    U = 0.5
    u = (U * g.cos_c)[:, None] * np.ones((1, 128)); v = np.zeros((65, 128))
    zeta = ref.vorticity(u, v, g)
    analytic = 2 * U * np.sin(g.phi_v)[:, None] * np.ones((1, 128))
    np.testing.assert_allclose(zeta[2:63], analytic[2:63], atol=2e-2)


def test_ref_grad_radius_scaling():
    # Doubling radius a halves the gradient magnitude (metric 1/a).
    M = np.linspace(0, 1, 32)[:, None] * np.ones((1, 16))
    g1 = ref.Grid(W=16, H=32, a=1.0); g2 = ref.Grid(W=16, H=32, a=2.0)
    _, gy1 = ref.grad_faces(M, g1); _, gy2 = ref.grad_faces(M, g2)
    np.testing.assert_allclose(gy2[1:32], 0.5 * gy1[1:32], rtol=1e-12)
```

- [ ] **Step 2: Run, verify FAIL**
Run: `uv run pytest tests/unit/test_sw_gpu.py -k "ref_" -v`  Expected: FAIL (module missing).

- [ ] **Step 3: Implement `shallow_water_ref.py`**
Port `Grid` (add `a: float = 1.0` field; multiply metric denominators by `a` in `grad_faces`, `divergence_hu`, `vorticity`), `center_to_uface`, `center_to_vface`, `divergence_hu`, `grad_faces`, `vorticity`, `corner_to_uface`, `coriolis_trapezoidal`, `_mass_fluxes`, `_apply_fluxes`, `continuity_step` from M0 `sw_spike/operators.py` and `grid.py` — VERBATIM except threading `a` through the three metric operators. Single layer: add `def pressure_grad(h, gp, g): return grad_faces(gp * h, g)` where `gp` is the scalar `g'`. Each function keeps its M0 signature so GPU kernels diff against it 1:1.

(The exact M0 bodies are in `src/gasgiant/sim/sw_spike/operators.py` and `grid.py` — read them and port; do not re-derive. The only edits are `a` in the metric.)

- [ ] **Step 4: Run, verify PASS**  Run: `uv run pytest tests/unit/test_sw_gpu.py -k "ref_" -v`  Expected: PASS.

- [ ] **Step 5: Commit**
```
git add src/gasgiant/sim/shallow_water_ref.py tests/unit/test_sw_gpu.py
git commit -m "M1: CPU single-layer C-grid reference (graduate M0 + radius a)"
```

---

## Task 2: Williamson-2 analytic state + balance check (CPU)

**Files:**
- Modify: `src/gasgiant/sim/shallow_water_ref.py`
- Test: `tests/unit/test_sw_gpu.py`

**Context:** Williamson test 2 is steady solid-body geostrophic flow `u = u0·cosφ`, with analytic height `h = h0 − (a·Ω·u0 + u0²/2)·sin²φ / g'`. It is an exact steady solution: integrated forward, the balanced state must not drift. This pins the sign/coefficient discipline (carried M0 caveat #2: keep the same-index mass-flux form consistent).

- [ ] **Step 1: Write the failing test**

```python
def test_ref_williamson2_stays_balanced():
    from gasgiant.sim import shallow_water_ref as ref
    st = ref.williamson2_state(W=128, H=64, a=1.0, omega=2.0, u0=0.2, gp=1.0, h0=5.0)
    m0 = ref.total_mass(st); e0 = ref.total_energy(st)
    for _ in range(80):
        st = ref.step(st)
    assert np.all(np.isfinite(st.h))
    # Steady solution: l2 velocity drift small; mass exactly conserved.
    assert ref.velocity_l2_drift(st) < 1e-2
    np.testing.assert_allclose(ref.total_mass(st), m0, rtol=1e-11)
```

- [ ] **Step 2: Run, verify FAIL.**  Run: `uv run pytest tests/unit/test_sw_gpu.py -k williamson2_stays -v`

- [ ] **Step 3: Implement** `SwRefState` (dataclass: g, gp, h, u, v, dt, plus `u_init/v_init` copies for drift), `williamson2_state` (builds `u=u0·cosφ`, analytic `h`, `f=2Ω·sinφ`, dt from min-spacing polar CFL `dt=0.3·min(cos_c.min()·a·dlam, a·dphi)/sqrt(gp·h.max())`), `step` (single-layer: `pressure_grad` → vector-invariant momentum with **relative-vorticity flux only** + trapezoidal Coriolis [carried M0 caveat: do NOT double-count f] → `continuity_step`), `total_mass` (`sum(h·cos_c)·a²·dlam·dphi`), `total_energy` (`sum(0.5·(h·(u²+v_c²) + gp·h²)·cos_c)`), `velocity_l2_drift` (l2 norm of `(u−u_init, v−v_init)` over `sqrt` of cell count), `total_potential_enstrophy` (diagnostic, `q=(ζ+f)/max(h,floor)` at corners). Reuse the M0 single-layer momentum assembly structure from `sw_spike/solver.py:_layer_momentum` with the Coriolis-double-count fix already applied there.

- [ ] **Step 4: Run, verify PASS.**

- [ ] **Step 5: Commit**
```
git add src/gasgiant/sim/shallow_water_ref.py tests/unit/test_sw_gpu.py
git commit -m "M1: Williamson-2 analytic state + CPU balance check"
```

---

## Task 3: GPU plumbing — SW textures + sw_common.glsl

**Files:**
- Create: `src/gasgiant/sim/kernels/sw_common.glsl`
- Create: `src/gasgiant/sim/sw_gpu.py`
- Test: `tests/unit/test_sw_gpu.py`

**Context:** Establish the texture layout and a round-trip so later kernels have a home. `SwGpuState` holds R32F textures: `h (W,H)`, `u (W,H)`, `v (W,H+1)`, plus scratch outputs. `sw_common.glsl` provides metric helpers usable from every kernel: `phiCenter(int row, int H)`, `phiVface(int row, int H)`, `cosClamp`, and the constants via uniforms (`u_size`, `u_a`, `u_dlam`, `u_dphi`, `u_gp`, `u_f0` or `u_omega`).

- [ ] **Step 1: Write the failing test** (texture round-trip + helper module loads)

```python
def test_sw_gpu_state_roundtrip(gpu):  # gpu fixture = shared headless GpuContext
    from gasgiant.sim import sw_gpu
    import numpy as np
    h = np.random.default_rng(0).random((32, 64)).astype(np.float32)
    st = sw_gpu.SwGpuState.create(gpu, W=64, H=32, a=1.0, gp=1.0, omega=2.0)
    st.upload_h(h)
    np.testing.assert_allclose(st.download_h(), h, atol=0)  # exact f4 round-trip
```

Add a `gpu` pytest fixture (session-scoped) creating ONE `GpuContext.headless()` + `make_current()` (the v1.6 one-context-per-process rule), in `tests/unit/test_sw_gpu.py` or a local conftest.

- [ ] **Step 2: Run, verify FAIL.**

- [ ] **Step 3: Implement** `sw_common.glsl` (metric helpers + `#define PI`), and `sw_gpu.py` `SwGpuState` with `create` (allocate textures via `gpu.texture2d((W,H),1,"f4")` etc.), `upload_h/u/v` (write via `tex.write(arr.tobytes())`), `download_h/u/v` (`gpu.read_texture(tex)[...,0]`). Mirror the existing solver's texture handling in `src/gasgiant/sim/solver.py`.

- [ ] **Step 4: Run, verify PASS.**  - [ ] **Step 5: Commit** `git commit -m "M1: GPU SW texture state + sw_common.glsl"`

---

## Task 4: sw_divergence.comp + GPU-vs-CPU diff

**Files:**
- Create: `src/gasgiant/sim/kernels/sw_divergence.comp`
- Modify: `src/gasgiant/sim/sw_gpu.py` (dispatch helper)
- Test: `tests/unit/test_sw_gpu.py`

**Context:** Port `ref.divergence_hu` to GLSL. `h` (center, sampler), `u` (east face, sampler), `v` (vface, sampler, height H+1) → `div` (center, image). Same-index zonal flux `Fx[i]=h[i]·u[i]` (carried M0 caveat #2 — NOT centered), meridional `Fy=h_vface·v`, metric `1/(a·cosφ)`, north-minus-south. The diff test is the spec.

- [ ] **Step 1: Write the failing test**

```python
def test_gpu_divergence_matches_ref(gpu):
    from gasgiant.sim import sw_gpu, shallow_water_ref as ref
    import numpy as np
    rng = np.random.default_rng(1)
    W, H = 64, 32
    h = (1.0 + 0.2 * rng.standard_normal((H, W))).astype(np.float32)
    u = (0.1 * rng.standard_normal((H, W))).astype(np.float32)
    v = np.zeros((H + 1, W), np.float32); v[1:H] = 0.1 * rng.standard_normal((H - 1, W))
    g = ref.Grid(W, H, a=1.0)
    cpu = ref.divergence_hu(h.astype(np.float64), u.astype(np.float64), v.astype(np.float64), g)
    got = sw_gpu.run_divergence(gpu, h, u, v, a=1.0)   # dispatch helper
    np.testing.assert_allclose(got, cpu, atol=2e-5)
```

- [ ] **Step 2: Run, verify FAIL.**

- [ ] **Step 3: Implement** `sw_divergence.comp` (full GLSL):

```glsl
#version 430
layout(local_size_x = 16, local_size_y = 16) in;
#include "sw_common.glsl"
uniform sampler2D u_h;      // center (W,H)
uniform sampler2D u_u;      // east face (W,H)
uniform sampler2D u_v;      // vface (W,H+1)
layout(r32f, binding = 0) uniform image2D out_div;   // center (W,H)
uniform ivec2 u_size;       // (W,H)
uniform float u_a, u_dlam, u_dphi;
void main() {
    ivec2 px = ivec2(gl_GlobalInvocationID.xy);
    int W = u_size.x, H = u_size.y;
    if (px.x >= W || px.y >= H) return;
    float cphi = cosCenter(px.y, H);                 // cos at this center row
    // same-index zonal flux Fx[i] = h[i]*u[i]; west neighbor at i-1 (periodic).
    float FxE = texelFetch(u_h, px, 0).r * texelFetch(u_u, px, 0).r;
    ivec2 pw = ivec2((px.x - 1 + W) % W, px.y);
    float FxW = texelFetch(u_h, pw, 0).r * texelFetch(u_u, pw, 0).r;
    float dFx = (FxE - FxW) / u_dlam;
    // meridional: Fy_c = h_vface * v * cos_vface, north face row = px.y, south = px.y+1.
    float cN = cosVface(px.y, H), cS = cosVface(px.y + 1, H);
    float hN = 0.5 * (texelFetch(u_h, ivec2(px.x, max(px.y-1,0)), 0).r + texelFetch(u_h, px, 0).r);
    float hS = 0.5 * (texelFetch(u_h, px, 0).r + texelFetch(u_h, ivec2(px.x, min(px.y+1,H-1)), 0).r);
    float FyN = (px.y == 0)   ? 0.0 : hN * texelFetch(u_v, ivec2(px.x, px.y),   0).r * cN;
    float FyS = (px.y == H-1) ? 0.0 : hS * texelFetch(u_v, ivec2(px.x, px.y+1), 0).r * cS;
    float dFy = (FyN - FyS) / u_dphi;
    float div = (dFx + dFy) / (u_a * cphi);
    imageStore(out_div, px, vec4(div, 0, 0, 0));
}
```
And `sw_common.glsl` `cosCenter(row,H)=cos(0.5*PI-(row+0.5)*PI/H)`, `cosVface(row,H)= (row==0||row==H)?0.0:cos(0.5*PI-row*PI/H)`. Add `run_divergence(gpu,h,u,v,a)` to `sw_gpu.py`: upload, `_set` uniforms, `bind_to_image`, `run(ceil(W/16),ceil(H/16),1)`, `memory_barrier`, download.

NOTE the GLSL h_vface uses the SAME `center_to_vface` averaging as the reference (`0.5*(h[north]+h[south])`); confirm the reference uses centered h on the vface (it does in M0 `_mass_fluxes` for `Fy_high`, but `divergence_hu` uses `center_to_vface(h)` — match THAT). If the diff fails, the bug is the h-at-vface interpolation or a metric/sign mismatch — debug against the reference field, do not loosen atol.

- [ ] **Step 4: Run, verify PASS.**  - [ ] **Step 5: Commit** `git commit -m "M1: sw_divergence.comp (matches CPU ref)"`

---

## Task 5: sw_grad.comp + diff

**Files:** Create `src/gasgiant/sim/kernels/sw_grad.comp`; modify `sw_gpu.py`; test.

**Context:** Port `ref.grad_faces` (applied to `gp*h`). Outputs `gx (W,H)` at east faces, `gy (W,H+1)` at vfaces. `gx = (M[i+1]-M[i])/(a·cosφ·dlam)`; `gy[row] = (M[north]-M[south])/(a·dphi)`, poles 0.

- [ ] **Step 1: Failing test** (`test_gpu_grad_matches_ref`): random `h`, compare `run_grad(gpu,h,gp,a)` returning `(gx,gy)` to `ref.grad_faces(gp*h, g)` at `atol=2e-5` (gx full; gy rows 1..H-1).
- [ ] **Step 2: FAIL.**
- [ ] **Step 3:** `sw_grad.comp` writes two images (`out_gx (W,H)`, `out_gy (W,H+1)`); dispatch over the larger `(W,H+1)` and guard each store by its valid range. Zonal store always (row<H); meridional store for `0<row<H` else 0. Uniforms `u_a,u_dlam,u_dphi,u_gp,u_size`. `M = u_gp*texelFetch(u_h,...)`.
- [ ] **Step 4: PASS.**  - [ ] **Step 5: Commit** `git commit -m "M1: sw_grad.comp (face pressure gradient)"`

---

## Task 6: sw_vorticity.comp + diff

**Files:** Create `src/gasgiant/sim/kernels/sw_vorticity.comp`; modify `sw_gpu.py`; test.

**Context:** Port `ref.vorticity` → `ζ (W,H+1)` at corners. `ζ = (1/(a·cosφ_v))[∂v/∂λ − ∂(u·cosφ_c)/∂φ]`, poles 0, with the M0 `+1e-30` pole guards. The rigid-rotation analytic check is the strong gate.

- [ ] **Step 1: Failing test** `test_gpu_vorticity_matches_ref`: random `u,v`, compare `run_vorticity(gpu,u,v,a)` to `ref.vorticity` at `atol=2e-5` over rows 1..H-1. ALSO re-assert rigid rotation: `u=U·cosφ` ⇒ GPU `ζ≈2U·sinφ` (atol 2e-2).
- [ ] **Step 2: FAIL.**
- [ ] **Step 3:** `sw_vorticity.comp` over `(W,H+1)`; corner row uses center rows `row-1` (north) and `row` (south), and v at `(i)−(i-1)`. Guard poles (store 0 at row 0 and H). Match the M0 metric exactly.
- [ ] **Step 4: PASS.**  - [ ] **Step 5: Commit** `git commit -m "M1: sw_vorticity.comp"`

---

## Task 7: sw_continuity.comp (FCT) + diff — HIGHEST RISK

**Files:** Create `src/gasgiant/sim/kernels/sw_continuity.comp`; modify `sw_gpu.py`; test.

**Context:** Port `ref.continuity_step` (FCT) to GLSL. This is the hardest kernel: the Zalesak per-face limiter needs each cell's capacity and the neighbor's. Strategy: TWO kernels to avoid race conditions — **pass A** computes per-cell `cap` and the low-order updated `h_low` into textures; **pass B** reads `cap`/`h_low`/fluxes and applies the limited anti-diffusive correction. This mirrors the reference's two-stage structure (`h_low` then limited `h_new`). The per-field diff against `ref.continuity_step` is the gate; mass conservation in the sub-CFL regime must hold to ~1e-6 on GPU (R32F, looser than CPU's 1e-12 — note this).

- [ ] **Step 1: Write the failing tests**

```python
def test_gpu_continuity_matches_ref(gpu):
    from gasgiant.sim import sw_gpu, shallow_water_ref as ref
    import numpy as np
    rng = np.random.default_rng(2); W,H = 64,32
    h = np.clip(1.0+0.1*rng.standard_normal((H,W)),0.2,None).astype(np.float32)
    u = (0.05*rng.standard_normal((H,W))).astype(np.float32)
    v = np.zeros((H+1,W),np.float32); v[1:H]=0.05*rng.standard_normal((H-1,W))
    g = ref.Grid(W,H,a=1.0)
    cpu = ref.continuity_step(h.astype(np.float64),u.astype(np.float64),v.astype(np.float64),g,dt=0.02,h_floor=0.05)
    got = sw_gpu.run_continuity(gpu,h,u,v,a=1.0,dt=0.02,h_floor=0.05)
    np.testing.assert_allclose(got,cpu,atol=2e-5)

def test_gpu_continuity_conserves_mass(gpu):
    # sub-CFL regime: GPU mass drift bounded (R32F => ~1e-6, not 1e-12).
    from gasgiant.sim import sw_gpu, shallow_water_ref as ref
    import numpy as np
    rng=np.random.default_rng(3); W,H=64,32
    h=np.clip(1.0+0.1*rng.standard_normal((H,W)),0.2,None).astype(np.float32)
    u=(0.03*rng.standard_normal((H,W))).astype(np.float32); v=np.zeros((H+1,W),np.float32)
    g=ref.Grid(W,H,a=1.0); area=g.cos_c[:,None]
    got=sw_gpu.run_continuity(gpu,h,u,v,a=1.0,dt=0.01,h_floor=0.05)
    np.testing.assert_allclose(np.sum(got*area),np.sum(h*area),rtol=2e-6)
```

- [ ] **Step 2: FAIL.**
- [ ] **Step 3:** Implement two compute passes. Pass A (`sw_continuity.comp` with `#define PASS 0`): compute low-order donor-cell fluxes + `_apply_fluxes` low-order, floor it, write `h_low` and `cap = max(h_low−floor,0)·cosφ/dt` to textures. Pass B (`#define PASS 1`): recompute high & low fluxes, anti-diffusive `A=Fhigh−Flow`, per-face limiter `s = min(1, min(cap_here,cap_nbr)/(|A|+1e-30))` (zonal uses east-neighbor `cap`; meridional uses min of the two rows' `cap`), apply `Flow + s·A`, floor, write `out_h`. Match the reference's `_mass_fluxes`/`_apply_fluxes` exactly (donor by sign of u/v; centered high-order; metric in `_apply_fluxes`). `sw_gpu.run_continuity` dispatches A, barrier, B, barrier, download.
- [ ] **Step 4: PASS** (both tests). If the diff fails, compare the GPU `h_low`/`cap` intermediates to a Python replica of pass A first to localize — do NOT loosen atol.
- [ ] **Step 5: Commit** `git commit -m "M1: sw_continuity.comp FCT (two-pass, matches CPU ref)"`

---

## Task 8: sw_momentum.comp + diff

**Files:** Create `src/gasgiant/sim/kernels/sw_momentum.comp`; modify `sw_gpu.py`; test.

**Context:** Port the single-layer vector-invariant momentum (from `sw_spike/solver.py:_layer_momentum`, single layer, Coriolis-double-count fix applied). Inputs `h,u,v`; outputs `u_new (W,H), v_new (W,H+1)`. Uses `ζ` (from Task 6 kernel or inline), relative-vorticity flux ONLY (`q=ζ`), Bernoulli `B=g'·h+0.5·(u²+v_c²)`, face grad of B, trapezoidal Coriolis with `f=2Ω·sinφ`. The plan's momentum needs `ζ` at u-faces and v-faces (corner→face averaging) — compute `ζ` in a prior dispatch into a texture, then momentum reads it (avoids recompute races).

- [ ] **Step 1: Failing test** `test_gpu_momentum_matches_ref`: random `h,u,v`; compare `run_momentum(gpu,h,u,v,a,gp,omega,dt)` `(u_new,v_new)` to a `ref.momentum_step(h,u,v,g,gp,omega,dt)` (extract the single-layer momentum from the reference `step`) at `atol=2e-5`.
- [ ] **Step 2: FAIL.**
- [ ] **Step 3:** Add `ref.momentum_step` to `shallow_water_ref.py` (the single-layer `_layer_momentum` body). Implement `sw_momentum.comp` reading a precomputed `u_zeta` corner texture; `run_momentum` dispatches vorticity kernel → momentum kernel. Trapezoidal Coriolis inline (Cayley: `a=0.5·f·dt; (u,v)→((1−a²)u+2av, (1−a²)v−2au)/(1+a²)`), `f` at centers for the co-located rotation as in M0.
- [ ] **Step 4: PASS.**  - [ ] **Step 5: Commit** `git commit -m "M1: sw_momentum.comp (vector-invariant + trapezoidal Coriolis)"`

---

## Task 9: SwGpuSolver.step — assemble + 1-step GPU≈CPU

**Files:** Modify `src/gasgiant/sim/sw_gpu.py`; test.

**Context:** Wire the kernels into one explicit step matching `ref.step`: vorticity → momentum (u,v) → continuity (h, using NEW u,v) → ping-pong. Use double-buffered textures.

- [ ] **Step 1: Failing test** `test_gpu_step_matches_ref_one_step`: build a Williamson-2 state on both CPU (`ref.williamson2_state`) and GPU (`SwGpuSolver.from_williamson2` with identical params/dt). Advance ONE step each. Compare `h,u,v` at `atol=2e-5`.
- [ ] **Step 2: FAIL.**
- [ ] **Step 3:** Implement `SwGpuSolver` (`from_williamson2`, `step()`, `download_state()`). `step` = dispatch sequence + ping-pong + `memory_barrier` between dependent kernels.
- [ ] **Step 4: PASS.**  - [ ] **Step 5: Commit** `git commit -m "M1: SwGpuSolver.step (1-step GPU=CPU)"`

---

## Task 10: N-step GPU≈CPU agreement

**Files:** Test only.

**Context:** Single-precision GPU diverges slowly from f64 CPU; bound the drift over many steps (not byte-identity vs CPU — that's impossible across precisions; byte-identity is GPU-vs-GPU, Task 13).

- [ ] **Step 1: Failing/【then passing】 test** `test_gpu_matches_ref_n_steps`: advance both 50 steps from Williamson-2; assert `max|h_gpu−h_cpu| < 5e-4` and velocity likewise (drift bound, not exact). If drift exceeds this, a kernel has a real discrepancy — localize with the per-field tests, do not loosen.
- [ ] **Step 2-4:** Run; if it passes immediately (kernels already correct) that is the expected outcome — this test guards against assembly/ordering bugs. - [ ] **Step 5: Commit** `git commit -m "M1: N-step GPU-vs-CPU agreement bound"`

---

## Task 11: Williamson-2 on GPU — balance + l2 gate

**Files:** Create `scripts/sw_m1_williamson.py`; test.

**Context:** The solution-accuracy half of the split gate (spec §6). Steady geostrophic flow must stay balanced; report the l2 error vs the analytic state.

- [ ] **Step 1: Failing test** `test_gpu_williamson2_balanced`: GPU Williamson-2, 80 steps; assert finite, `velocity_l2_drift < 2e-2`, mass drift `< 1e-5` (R32F). 
- [ ] **Step 2-3:** Add `velocity_l2_drift`/`total_mass` readback to `SwGpuSolver`. `scripts/sw_m1_williamson.py` runs it and prints l2 + mass/energy/enstrophy drift over the run.
- [ ] **Step 4: PASS.**  - [ ] **Step 5: Commit** `git commit -m "M1: Williamson-2 GPU balance + l2 gate + report script"`

---

## Task 12: Conservation budget closure (self-budget gate)

**Files:** Modify `scripts/sw_m1_williamson.py`; test.

**Context:** The conservation half of the split gate (spec §6): mass to ~1e-5 (flux-form, R32F); energy/enstrophy drift accounted by the diagnosed dissipation (in M1 there is NO forcing/hypervisc, so the only sink is the FCT floor + numerical truncation — energy should be NEARLY conserved, drift bounded and one-signed from FCT diffusion).

- [ ] **Step 1: Failing test** `test_gpu_conservation_budget`: Williamson-2, 80 steps, no forcing; assert mass drift `< 1e-5`, total-energy drift `< 1%` (bounded by FCT numerical diffusion), and potential-enstrophy finite/bounded. Document that exact energy conservation is not claimed (collocated-vs-Cgrid AL caveat from the spec; M1 monitors bounded drift).
- [ ] **Step 2-4:** Implement diagnostics + run. - [ ] **Step 5: Commit** `git commit -m "M1: conservation budget-closure diagnostics"`

---

## Task 13: Determinism + hash gate

**Files:** Test only.

**Context:** Same-machine GPU determinism (v1.6 discipline): two runs from identical init produce byte-identical state. A float32 SHA over the final `h,u,v`.

- [ ] **Step 1: Failing/passing test** `test_gpu_deterministic`: run Williamson-2 50 steps twice (fresh solver each); assert `download_state()` arrays are byte-identical (`np.array_equal` on the f4 bytes) and the SHA1 of concatenated bytes matches between runs. Also assert the field EVOLVED (final != initial) — guard the v1.6 no-op trap.
- [ ] **Step 2-4:** Run. - [ ] **Step 5: Commit** `git commit -m "M1: GPU determinism + hash gate"`

---

## Task 14: Checkpoint byte-exact round-trip

**Files:** Modify `src/gasgiant/sim/sw_gpu.py`; test.

**Context:** Serialize `h,u,v` + params to npz and restore byte-exact (spec §9.7). Reuse the v1.6 npz discipline.

- [ ] **Step 1: Failing test** `test_gpu_checkpoint_roundtrip`: run 30 steps, `save_checkpoint(path)`, advance 10 more (state A); reload into a fresh solver, advance 10 (state B); assert A==B byte-identical.
- [ ] **Step 2-4:** Implement `save_checkpoint`/`load_checkpoint` on `SwGpuSolver` (npz of f4 arrays + scalar params + a version int). - [ ] **Step 5: Commit** `git commit -m "M1: SW GPU checkpoint byte-exact round-trip"`

---

## Task 15: M1 validation report + verdict

**Files:** Modify `scripts/sw_m1_williamson.py`; create `docs/superpowers/specs/m1-verdict.md`.

**Context:** Tie the split gate together and record M1's exit status.

- [ ] **Step 1:** Make `scripts/sw_m1_williamson.py` print a single PASS/FAIL summary across: per-field diffs (run the pytest subset), Williamson-2 l2, mass/energy/enstrophy drift, determinism hash. 
- [ ] **Step 2:** Write `docs/superpowers/specs/m1-verdict.md`: the gate numbers, what passed, any M2 caveats (semi-implicit is next; the explicit dt is still polar-CFL-limited — this is WHY M2 exists), and confirm `shallow_water_ref.py` is now the ground truth for M3's 2-layer extension.
- [ ] **Step 3: Commit** `git commit -m "M1: validation report + verdict"`

---

## Self-review notes (for the implementer)

- **The CPU reference is the spec.** Every GPU kernel is judged by per-field `atol=2e-5` diff against `shallow_water_ref.py`. If a diff fails, the bug is in the GLSL port (indexing, sign, metric, a pole guard) — localize against the reference field; NEVER loosen `atol`.
- **C-grid GLSL indexing is the #1 risk.** `v`/`ζ` textures have height `H+1`; centers/`u` have height `H`. Pole rows (0 and H of the vface/corner textures) must store 0. Periodic longitude via `(i±1+W)%W` (and `repeat_x=True` makes `texelFetch` wrap too — but be explicit).
- **Coriolis: relative vorticity only in the flux** (carried M0 bug-fix #1) — `f` is handled solely by the trapezoidal step. Do not reintroduce `q=ζ+f`.
- **FCT two-pass** avoids the limiter read-write race; verify pass A intermediates before blaming pass B.
- **Precision:** GPU is R32F; GPU-vs-CPU is a DRIFT bound (~5e-4 over 50 steps), GPU-vs-GPU is byte-identical (determinism). Don't conflate them.
- **One GpuContext per process** (v1.6 rule) — the `gpu` fixture is session-scoped.
- **dt** uses the min-spacing polar CFL (carried M0 caveat #4); M1 stays explicit, so the dt is small — that is expected and is the motivation for M2 (semi-implicit), not an M1 defect.
- Do NOT wire `solver.type=shallow_water` or touch the render in M1 — that is M5. M1 is a standalone validated module.
```
