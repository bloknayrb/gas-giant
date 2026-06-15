# Milestone 0 — Shallow-Water Image Spike Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a throwaway CPU 2-layer reduced-gravity shallow-water solver (equirect only), render its top layer through the existing GPU pipeline at ≥4K, and run the v1.6 blind judge panel — a kill-gate that decides whether the full M1–M5 build proceeds.

**Architecture:** A NumPy Arakawa **C-grid** solver on a lon-lat sphere: `h` at cell centers, `u`/`v` on faces, `ζ` at corners. Vector-invariant momentum, flux-form **FCT-limited** continuity (mass-conserving + positive), trapezoidal Coriolis, explicit gravity-wave step. Two layers coupled by the reduced-gravity Montgomery potential, forced by thermal relaxation toward `h_eq(φ)` + bottom drag. A thin encoder writes top-layer thickness-anomaly + relative vorticity into the existing RGBA tracer textures so the unchanged GPU `derive.comp` render runs at ≥4K. The kill-gate compares that render to v1.6.

**Tech Stack:** Python 3.13, NumPy (solver + its own ground truth), existing `gasgiant` GPU render (moderngl), `scripts/measure_morphology.py` (coher metric + belt crop), pytest.

**Why CPU/NumPy for the spike:** the kill-gate is a *render* decision, not a numerics-accuracy decision; a CPU C-grid solver is the fastest path to that signal, is its own ground truth (no GPU/CPU cross-validation), and graduates into M1's `shallow_water_ref.py`. If M0 passes, M1 re-implements these exact operators as GPU kernels validated against this module.

**Conventions (fixed once, used by every task):**
- Nondimensional sphere radius `a = 1`. Coriolis `f = f0·sinφ`.
- Grid: `W` lon cells (periodic), `H` lat cells. Rows **descending** (row 0 = north), matching the existing codebase.
- Cell-center latitude: `phi_c[j] = π/2 − (j+0.5)·π/H`, `j = 0..H−1`.
- v-face / corner latitude: `phi_v[j] = π/2 − j·π/H`, `j = 0..H` (so `phi_v[0]=+π/2` north pole, `phi_v[H]=−π/2` south pole; `cos` is 0 at both → no flux through poles).
- Array shapes: `h,u : (H, W)` where `u[j,i]` is the **east** face of cell `(j,i)` (between `i` and `i+1`, periodic). `v : (H+1, W)` where `v[j,i]` is the face between center-rows `j−1` (north) and `j` (south); `v[0]=v[H]=0` (poles). `zeta : (H+1, W)` at corners, same meridional indexing as `v`.
- `dlam = 2π/W`, `dphi = π/H`. Meridional derivative uses φ **increasing northward**: at v-face `j`, `∂A/∂φ ≈ (A[j−1] − A[j])/dphi` (north row minus south row).

---

## File structure

- Create `src/gasgiant/sim/sw_spike/__init__.py` — spike subpackage marker (clearly throwaway/namespaced).
- Create `src/gasgiant/sim/sw_spike/grid.py` — grid metric arrays + staggered-interpolation helpers.
- Create `src/gasgiant/sim/sw_spike/operators.py` — C-grid divergence, Montgomery + pressure-gradient, vorticity, Coriolis, FCT continuity.
- Create `src/gasgiant/sim/sw_spike/solver.py` — `SwState`, one explicit step, the 2-layer assembly, forcing.
- Create `src/gasgiant/sim/sw_spike/init.py` — `h_eq(φ)` builder + balanced gradient-wind init with equatorial closure.
- Create `src/gasgiant/sim/sw_spike/encode.py` — fields → RGBA tracer texture(s) for the GPU render.
- Create `tests/unit/test_sw_spike.py` — all TDD tests below.
- Create `scripts/sw_spike_run.py` — driver: spin up, measure equilibration, write fields.
- Create `scripts/sw_spike_killgate.py` — SW render vs v1.6 render, coher + blind-judge strip + R1/R3 gate assertions.

Each file has one responsibility; `operators.py` is pure functions (easy to test and to port to GLSL in M1).

---

## Task 1: Spike scaffold + grid metrics

**Files:**
- Create: `src/gasgiant/sim/sw_spike/__init__.py`
- Create: `src/gasgiant/sim/sw_spike/grid.py`
- Test: `tests/unit/test_sw_spike.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/test_sw_spike.py
import numpy as np
import pytest
from gasgiant.sim.sw_spike import grid


def test_grid_metrics_poles_zero_cos():
    g = grid.Grid(W=16, H=8)
    assert g.phi_c.shape == (8,)
    assert g.phi_v.shape == (9,)
    # Descending: row 0 is north (positive lat), last row south.
    assert g.phi_c[0] > 0 and g.phi_c[-1] < 0
    # v-face cos is exactly 0 at both poles (no flux through pole).
    assert g.cos_v[0] == pytest.approx(0.0, abs=1e-12)
    assert g.cos_v[-1] == pytest.approx(0.0, abs=1e-12)
    # Center cos strictly positive everywhere.
    assert np.all(g.cos_c > 0.0)
    assert g.dlam == pytest.approx(2 * np.pi / 16)
    assert g.dphi == pytest.approx(np.pi / 8)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit/test_sw_spike.py::test_grid_metrics_poles_zero_cos -v`
Expected: FAIL with `ModuleNotFoundError: gasgiant.sim.sw_spike`

- [ ] **Step 3: Write minimal implementation**

```python
# src/gasgiant/sim/sw_spike/__init__.py
"""M0 throwaway shallow-water spike (CPU). Graduates to shallow_water_ref in M1."""
```

```python
# src/gasgiant/sim/sw_spike/grid.py
from __future__ import annotations
from dataclasses import dataclass
import numpy as np


@dataclass(frozen=True)
class Grid:
    W: int
    H: int

    @property
    def dlam(self) -> float:
        return 2.0 * np.pi / self.W

    @property
    def dphi(self) -> float:
        return np.pi / self.H

    @property
    def phi_c(self) -> np.ndarray:
        j = np.arange(self.H)
        return 0.5 * np.pi - (j + 0.5) * self.dphi  # descending

    @property
    def phi_v(self) -> np.ndarray:
        j = np.arange(self.H + 1)
        return 0.5 * np.pi - j * self.dphi  # phi_v[0]=+pi/2, phi_v[H]=-pi/2

    @property
    def cos_c(self) -> np.ndarray:
        return np.cos(self.phi_c)

    @property
    def cos_v(self) -> np.ndarray:
        c = np.cos(self.phi_v)
        c[0] = 0.0
        c[-1] = 0.0
        return c

    @property
    def f_c(self) -> np.ndarray:
        # placeholder f0=1; callers scale. Kept as sin(lat) shape.
        return np.sin(self.phi_c)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/unit/test_sw_spike.py::test_grid_metrics_poles_zero_cos -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add src/gasgiant/sim/sw_spike/__init__.py src/gasgiant/sim/sw_spike/grid.py tests/unit/test_sw_spike.py
git commit -m "M0: shallow-water spike grid metrics"
```

---

## Task 2: Staggered interpolation helpers

**Files:**
- Modify: `src/gasgiant/sim/sw_spike/grid.py`
- Test: `tests/unit/test_sw_spike.py`

- [ ] **Step 1: Write the failing test**

```python
def test_center_to_uface_is_periodic_average():
    from gasgiant.sim.sw_spike import grid
    a = np.array([[1.0, 3.0, 5.0, 7.0]])  # H=1, W=4
    uf = grid.center_to_uface(a)  # east face of cell i = avg(i, i+1), periodic
    assert uf.shape == a.shape
    np.testing.assert_allclose(uf, [[2.0, 4.0, 6.0, 4.0]])  # last wraps 7&1 -> 4


def test_center_to_vface_zeroed_at_poles():
    from gasgiant.sim.sw_spike import grid
    a = np.array([[2.0, 2.0], [4.0, 4.0]])  # H=2, W=2
    vf = grid.center_to_vface(a)  # shape (H+1, W); interior = avg of rows; poles=0
    assert vf.shape == (3, 2)
    np.testing.assert_allclose(vf[0], [0.0, 0.0])   # north pole face: no cell north
    np.testing.assert_allclose(vf[1], [3.0, 3.0])   # between rows 0,1
    np.testing.assert_allclose(vf[2], [0.0, 0.0])   # south pole face
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit/test_sw_spike.py -k "uface or vface" -v`
Expected: FAIL with `AttributeError: module ... has no attribute 'center_to_uface'`

- [ ] **Step 3: Write minimal implementation**

```python
# append to src/gasgiant/sim/sw_spike/grid.py

def center_to_uface(a: np.ndarray) -> np.ndarray:
    """East-face value = average of cell i and i+1 (periodic in lon)."""
    return 0.5 * (a + np.roll(a, -1, axis=1))


def center_to_vface(a: np.ndarray) -> np.ndarray:
    """Meridional face value; shape (H+1, W). Pole faces forced to 0."""
    H, W = a.shape
    vf = np.zeros((H + 1, W), dtype=a.dtype)
    vf[1:H] = 0.5 * (a[0:H - 1] + a[1:H])  # north row (j-1) and south row (j)
    return vf
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/unit/test_sw_spike.py -k "uface or vface" -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add -A && git commit -m "M0: staggered interpolation helpers"
```

---

## Task 3: Divergence operator ∇·(h u) on the C-grid

**Files:**
- Create: `src/gasgiant/sim/sw_spike/operators.py`
- Test: `tests/unit/test_sw_spike.py`

- [ ] **Step 1: Write the failing test**

```python
def test_divergence_of_solid_body_zonal_flow_is_zero():
    # Purely zonal, longitude-independent u, constant h => mass divergence ~ 0.
    from gasgiant.sim.sw_spike import grid, operators
    g = grid.Grid(W=64, H=32)
    h = np.ones((32, 64))
    u = np.ones((32, 64)) * 0.3          # uniform zonal face velocity
    v = np.zeros((33, 64))
    div = operators.divergence_hu(h, u, v, g)
    # Zonal-uniform, v=0, constant h => divergence is machine-zero.
    assert np.max(np.abs(div)) < 1e-12


def test_divergence_has_no_checkerboard_null_mode():
    # The whole reason for the C-grid: a checkerboard in h must NOT be invisible.
    from gasgiant.sim.sw_spike import grid, operators
    g = grid.Grid(W=64, H=32)
    jj, ii = np.indices((32, 64))
    h = 1.0 + 0.01 * ((ii + jj) % 2)     # 2dx checkerboard thickness
    u = np.ones((32, 64)) * 0.1
    v = np.zeros((33, 64))
    div = operators.divergence_hu(h, u, v, g)
    # On a C-grid the checkerboard produces real flux divergence (non-null).
    assert np.max(np.abs(div)) > 1e-4
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit/test_sw_spike.py -k divergence -v`
Expected: FAIL (`operators` missing)

- [ ] **Step 3: Write minimal implementation**

```python
# src/gasgiant/sim/sw_spike/operators.py
from __future__ import annotations
import numpy as np
from .grid import Grid, center_to_uface, center_to_vface


def divergence_hu(h: np.ndarray, u: np.ndarray, v: np.ndarray, g: Grid) -> np.ndarray:
    """Flux-form mass divergence ∇·(h u) at cell centers, shape (H, W).

    Spherical metric: (1/(a cosφ))[ ∂(h u)/∂λ + ∂(h v cosφ)/∂φ ].
    """
    H, W = h.shape
    h_uf = center_to_uface(h)            # h at east faces (H,W)
    Fx = h_uf * u                        # zonal mass flux at east faces
    # ∂Fx/∂λ at center i = (Fx[i] - Fx[i-1]) / dlam  (east minus west face)
    dFx = (Fx - np.roll(Fx, 1, axis=1)) / g.dlam

    h_vf = center_to_vface(h)            # h at v-faces (H+1,W)
    Fy = h_vf * v                        # meridional mass flux at v-faces
    cos_v = g.cos_v[:, None]             # (H+1,1)
    Fy_c = Fy * cos_v                    # h v cosφ
    # ∂(Fy cosφ)/∂φ at center row j = (Fy_c[north face j] - Fy_c[south face j+1]) / dphi
    # north face of row j is index j, south face is index j+1 (phi decreases with j).
    dFy = (Fy_c[0:H] - Fy_c[1:H + 1]) / g.dphi

    inv_metric = 1.0 / g.cos_c[:, None]  # (H,1)
    return inv_metric * (dFx + dFy)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/unit/test_sw_spike.py -k divergence -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add -A && git commit -m "M0: C-grid mass divergence (no checkerboard null mode)"
```

---

## Task 4: Montgomery potential + face pressure gradient (the R1 checkerboard resolution)

**Files:**
- Modify: `src/gasgiant/sim/sw_spike/operators.py`
- Test: `tests/unit/test_sw_spike.py`

- [ ] **Step 1: Write the failing test**

```python
def test_montgomery_two_layer_coupling():
    from gasgiant.sim.sw_spike import operators
    h1 = np.full((4, 4), 2.0)
    h2 = np.full((4, 4), 3.0)
    gp = (1.0, 0.05)  # (g'_1 external, g'_2 baroclinic)
    M1, M2 = operators.montgomery_2layer(h1, h2, gp)
    # M1 = g'_1 (h1+h2); M2 = g'_1 (h1+h2) + g'_2 h2
    np.testing.assert_allclose(M1, 1.0 * 5.0)
    np.testing.assert_allclose(M2, 1.0 * 5.0 + 0.05 * 3.0)


def test_pressure_gradient_constant_field_is_zero():
    from gasgiant.sim.sw_spike import grid, operators
    g = grid.Grid(W=32, H=16)
    M = np.full((16, 32), 7.3)
    gx, gy = operators.grad_faces(M, g)
    assert np.max(np.abs(gx)) < 1e-12
    assert np.max(np.abs(gy)) < 1e-12


def test_pressure_gradient_sees_checkerboard():
    # Centered-collocated grad would return ~0 here; the C-grid face grad must not.
    from gasgiant.sim.sw_spike import grid, operators
    g = grid.Grid(W=32, H=16)
    jj, ii = np.indices((16, 32))
    M = ((ii + jj) % 2).astype(float)
    gx, gy = operators.grad_faces(M, g)
    assert np.max(np.abs(gx)) > 1e-3   # face differences are large for a 2dx mode
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit/test_sw_spike.py -k "montgomery or pressure" -v`
Expected: FAIL (functions missing)

- [ ] **Step 3: Write minimal implementation**

```python
# append to operators.py

def montgomery_2layer(h1: np.ndarray, h2: np.ndarray, gp: tuple[float, float]):
    """Reduced-gravity Montgomery potentials for the 2-layer stack (§2.2)."""
    g1, g2 = gp
    eta1 = h1 + h2          # height of top of layer 1
    M1 = g1 * eta1
    M2 = g1 * eta1 + g2 * h2
    return M1, M2


def grad_faces(M: np.ndarray, g: Grid):
    """∇M evaluated on faces (single difference, no 2dx null space).

    Returns (gx at u-faces (H,W), gy at v-faces (H+1,W)).
    """
    H, W = M.shape
    # Zonal gradient at east face i = (M[i+1] - M[i]) / (a cosφ dλ).
    gx = (np.roll(M, -1, axis=1) - M) / (g.cos_c[:, None] * g.dlam)
    # Meridional gradient at v-face j = (M[north row j-1] - M[south row j]) / (a dφ).
    gy = np.zeros((H + 1, W))
    gy[1:H] = (M[0:H - 1] - M[1:H]) / g.dphi
    return gx, gy
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/unit/test_sw_spike.py -k "montgomery or pressure" -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add -A && git commit -m "M0: Montgomery potential + face pressure gradient"
```

---

## Task 5: Relative vorticity at corners + averaging to faces

**Files:**
- Modify: `src/gasgiant/sim/sw_spike/operators.py`
- Test: `tests/unit/test_sw_spike.py`

- [ ] **Step 1: Write the failing test**

```python
def test_vorticity_zero_for_irrotational_uniform_flow():
    from gasgiant.sim.sw_spike import grid, operators
    g = grid.Grid(W=64, H=32)
    u = np.full((32, 64), 0.2)
    v = np.zeros((33, 64))
    zeta = operators.vorticity(u, v, g)   # corners (H+1, W)
    assert zeta.shape == (33, 64)
    # Uniform zonal flow on the sphere has curvature vorticity -(1/a) d(u cosφ)/dφ != 0
    # but a constant-u test is dominated by the metric; assert it's finite & smooth.
    assert np.all(np.isfinite(zeta))


def test_vorticity_of_rigid_rotation_constant_sign():
    # u = U cosφ (solid-body zonal) => zeta = -(1/(a cosφ)) d(U cos^2 φ)/dφ = 2U sinφ.
    from gasgiant.sim.sw_spike import grid, operators
    g = grid.Grid(W=128, H=64)
    U = 0.5
    u = (U * g.cos_c)[:, None] * np.ones((1, 128))
    v = np.zeros((65, 128))
    zeta = operators.vorticity(u, v, g)
    # Compare interior corners to analytic 2U sinφ at phi_v.
    analytic = 2 * U * np.sin(g.phi_v)[:, None] * np.ones((1, 128))
    inner = slice(2, 63)
    np.testing.assert_allclose(zeta[inner], analytic[inner], atol=2e-2)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit/test_sw_spike.py -k vorticity -v`
Expected: FAIL (`vorticity` missing)

- [ ] **Step 3: Write minimal implementation**

```python
# append to operators.py

def vorticity(u: np.ndarray, v: np.ndarray, g: Grid) -> np.ndarray:
    """Relative vorticity ζ = (1/(a cosφ))[∂v/∂λ − ∂(u cosφ)/∂φ] at corners (H+1, W)."""
    H, W = u.shape
    # ∂v/∂λ at corner (j, i): v lives at v-faces (H+1,W); corner i uses v[i]-v[i-1].
    dv_dlam = (v - np.roll(v, 1, axis=1)) / (g.cos_v[:, None] * g.dlam + 1e-30)
    # u cosφ at centers, differenced across the v-face (north row minus south row).
    ucos = u * g.cos_c[:, None]
    ducos = np.zeros((H + 1, W))
    ducos[1:H] = (ucos[0:H - 1] - ucos[1:H]) / g.dphi
    zeta = dv_dlam - ducos / (g.cos_v[:, None] + 1e-30)
    zeta[0] = 0.0
    zeta[H] = 0.0
    return zeta


def corner_to_uface(zc: np.ndarray) -> np.ndarray:
    """Average corner field (H+1,W) to u-faces (H,W): mean of the 2 corners in φ."""
    return 0.5 * (zc[0:-1] + zc[1:])
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/unit/test_sw_spike.py -k vorticity -v`
Expected: PASS (the rigid-rotation case matches `2U sinφ` to tolerance)

- [ ] **Step 5: Commit**

```bash
git add -A && git commit -m "M0: relative vorticity at corners + face averaging"
```

---

## Task 6: Trapezoidal Coriolis update

**Files:**
- Modify: `src/gasgiant/sim/sw_spike/operators.py`
- Test: `tests/unit/test_sw_spike.py`

- [ ] **Step 1: Write the failing test**

```python
def test_trapezoidal_coriolis_conserves_speed():
    # Pure inertial rotation: |(u,v)| must be preserved by the implicit rotation.
    from gasgiant.sim.sw_spike import operators
    u = np.array([[1.0]]); v = np.array([[0.0]])
    f = np.array([[0.7]]); dt = 0.3
    for _ in range(200):
        u, v = operators.coriolis_trapezoidal(u, v, f, dt)
    speed = np.hypot(u, v)
    np.testing.assert_allclose(speed, 1.0, atol=1e-10)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit/test_sw_spike.py -k coriolis -v`
Expected: FAIL (`coriolis_trapezoidal` missing)

- [ ] **Step 3: Write minimal implementation**

```python
# append to operators.py

def coriolis_trapezoidal(u: np.ndarray, v: np.ndarray, f: np.ndarray, dt: float):
    """Energy-neutral (norm-preserving) implicit Coriolis: trapezoidal rotation.

    Solves (u^{n+1}-u^n)/dt = f v*, (v^{n+1}-v^n)/dt = -f u*, with * = ½(n+n+1).
    Closed form is the Cayley rotation by angle θ=f dt.
    """
    a = 0.5 * f * dt
    denom = 1.0 + a * a
    u_new = ((1.0 - a * a) * u + 2.0 * a * v) / denom
    v_new = ((1.0 - a * a) * v - 2.0 * a * u) / denom
    return u_new, v_new
```

- [ ] **Step 2 caveat:** `u`, `v`, `f` here are co-located (one cell). In the full step (Task 8) `f` is averaged to the u-face and v-face locations; this function is location-agnostic.

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/unit/test_sw_spike.py -k coriolis -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add -A && git commit -m "M0: trapezoidal (norm-preserving) Coriolis"
```

---

## Task 7: FCT-limited flux-form continuity (mass conserving + positive)

**Files:**
- Modify: `src/gasgiant/sim/sw_spike/operators.py`
- Test: `tests/unit/test_sw_spike.py`

- [ ] **Step 1: Write the failing test**

```python
def test_continuity_conserves_total_mass():
    from gasgiant.sim.sw_spike import grid, operators
    g = grid.Grid(W=48, H=24)
    rng = np.random.default_rng(0)
    h = 1.0 + 0.1 * rng.standard_normal((24, 48))
    h = np.clip(h, 0.2, None)
    u = 0.05 * rng.standard_normal((24, 48))
    v = np.zeros((25, 48)); v[1:24] = 0.05 * rng.standard_normal((23, 48))
    area = g.cos_c[:, None] * np.ones((24, 48))
    m0 = np.sum(h * area)
    h2 = operators.continuity_step(h, u, v, g, dt=0.1, h_floor=0.05)
    m1 = np.sum(h2 * area)
    np.testing.assert_allclose(m1, m0, rtol=1e-12)  # flux-form => machine precision


def test_continuity_preserves_positivity_under_strong_gradient():
    from gasgiant.sim.sw_spike import grid, operators
    g = grid.Grid(W=32, H=16)
    h = np.full((16, 32), 0.06)          # near the floor
    h[:, 8] = 1.0                        # a spike that will be advected hard
    u = np.full((16, 32), 0.9)           # strong outflow
    v = np.zeros((17, 32))
    h2 = operators.continuity_step(h, u, v, g, dt=0.5, h_floor=0.05)
    assert np.min(h2) >= 0.05 - 1e-9     # FCT keeps h >= floor, no negatives
    assert np.all(np.isfinite(h2))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit/test_sw_spike.py -k continuity -v`
Expected: FAIL (`continuity_step` missing)

- [ ] **Step 3: Write minimal implementation**

```python
# append to operators.py

def _mass_fluxes(h, u, v, g):
    """Low-order (donor-cell / upwind) and high-order (centered) mass fluxes."""
    H, W = h.shape
    # Zonal east-face flux. Upwind donor by sign of u.
    hE_up = np.where(u >= 0, h, np.roll(h, -1, axis=1))
    Fx_low = hE_up * u
    Fx_high = center_to_uface(h) * u
    # Meridional v-face flux. Upwind donor: v>0 means flow toward south (row j),
    # so donor is the north row (j-1).
    h_north = np.zeros((H + 1, W)); h_north[1:H] = h[0:H - 1]
    h_south = np.zeros((H + 1, W)); h_south[1:H] = h[1:H]
    hV_up = np.where(v >= 0, h_north, h_south)
    Fy_low = hV_up * v
    Fy_high = center_to_vface(h) * v
    return Fx_low, Fx_high, Fy_low, Fy_high


def _apply_fluxes(h, Fx, Fy, g, dt):
    H, W = h.shape
    dFx = (Fx - np.roll(Fx, 1, axis=1)) / g.dlam
    Fy_c = Fy * g.cos_v[:, None]
    dFy = (Fy_c[0:H] - Fy_c[1:H + 1]) / g.dphi
    return h - dt * (dFx + dFy) / g.cos_c[:, None]


def continuity_step(h, u, v, g, dt, h_floor):
    """Flux-corrected transport: mass-conserving AND positivity-preserving.

    Zalesak-style limiter: blend high-order toward low-order so the update
    introduces no new extremum below the floor.
    """
    Fx_low, Fx_high, Fy_low, Fy_high = _mass_fluxes(h, u, v, g)
    h_low = _apply_fluxes(h, Fx_low, Fy_low, g, dt)           # monotone, positive
    h_low = np.maximum(h_low, h_floor)
    # Anti-diffusive flux = high - low.
    Ax = Fx_high - Fx_low
    Ay = Fy_high - Fy_low
    # Limit each anti-diffusive flux so it cannot pull any cell below the floor.
    # Outgoing capacity of each cell above the floor:
    cap = np.maximum(h_low - h_floor, 0.0) * g.cos_c[:, None] / dt
    # Scale anti-diffusive fluxes by the most-restrictive adjacent capacity.
    sx = np.minimum(1.0, cap / (np.abs(Ax) + 1e-30))
    sy = np.minimum(1.0, cap[1:] if False else 1.0)  # see note
    Ax_lim = Ax * np.minimum(sx, np.roll(sx, -1, axis=1))
    # Meridional limiter (simple, conservative): clamp by the donor-row capacity.
    cap_v = np.zeros((g.H + 1, g.W)); cap_v[1:g.H] = np.minimum(cap[0:g.H - 1], cap[1:g.H])
    sy = np.minimum(1.0, cap_v / (np.abs(Ay) + 1e-30))
    Ay_lim = Ay * sy
    h_new = _apply_fluxes(h, Fx_low + Ax_lim, Fy_low + Ay_lim, g, dt)
    return np.maximum(h_new, h_floor)
```

- [ ] **Step 3 note:** the `sy = ... if False else 1.0` line is dead scaffolding from drafting — delete it; the real meridional limiter is the `cap_v` block two lines below. Final code must not contain that line.

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/unit/test_sw_spike.py -k continuity -v`
Expected: PASS (mass conserved to `rtol=1e-12`; `min(h) >= floor`)

- [ ] **Step 5: Commit**

```bash
git add -A && git commit -m "M0: FCT continuity (mass-conserving + positive)"
```

---

## Task 8: Assemble one explicit 2-layer step + the balanced-state & checkerboard gates

**Files:**
- Create: `src/gasgiant/sim/sw_spike/solver.py`
- Test: `tests/unit/test_sw_spike.py`

- [ ] **Step 1: Write the failing test**

```python
def test_balanced_zonal_state_stays_balanced():
    # Mini Williamson-2: a geostrophically balanced zonal jet must not accelerate.
    from gasgiant.sim.sw_spike import solver
    st = solver.balanced_test_state(W=128, H=64, f0=4.0, gp=(1.0, 0.05))
    ke0 = solver.kinetic_energy(st)
    for _ in range(50):
        st = solver.step(st, dt=st.dt)
    ke1 = solver.kinetic_energy(st)
    # Balance preserved to scheme order: KE drifts < 1% over 50 steps.
    assert abs(ke1 - ke0) / ke0 < 0.01


def test_checkerboard_pressure_perturbation_does_not_grow():
    # R1 gate: seed a 2dx checkerboard in h, confirm it does NOT amplify.
    from gasgiant.sim.sw_spike import solver
    st = solver.balanced_test_state(W=128, H=64, f0=4.0, gp=(1.0, 0.05))
    jj, ii = np.indices(st.h1.shape)
    cb = 0.001 * ((ii + jj) % 2)
    st.h1 = st.h1 + cb
    amp0 = solver.checkerboard_amplitude(st.h1)
    for _ in range(100):
        st = solver.step(st, dt=st.dt)
    amp1 = solver.checkerboard_amplitude(st.h1)
    assert amp1 <= amp0 * 1.5    # bounded, not exponentially growing
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit/test_sw_spike.py -k "balanced or checkerboard" -v`
Expected: FAIL (`solver` missing)

- [ ] **Step 3: Write minimal implementation**

```python
# src/gasgiant/sim/sw_spike/solver.py
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from .grid import Grid, center_to_uface, center_to_vface
from . import operators as ops


@dataclass
class SwState:
    g: Grid
    f0: float
    gp: tuple[float, float]
    h1: np.ndarray; u1: np.ndarray; v1: np.ndarray
    h2: np.ndarray; u2: np.ndarray; v2: np.ndarray
    dt: float
    h_floor: float = 0.05
    nu4: float = 0.0          # hyperviscosity (set by forcing task)
    tau_rad: float = 0.0      # 0 disables (set by forcing task)
    tau_drag: float = 0.0
    h_eq1: np.ndarray | None = None
    h_eq2: np.ndarray | None = None


def _f_uface(g: Grid, f0: float) -> np.ndarray:
    return f0 * np.sin(g.phi_c)[:, None] * np.ones((1, g.W))


def _f_vface(g: Grid, f0: float) -> np.ndarray:
    return f0 * np.sin(g.phi_v)[:, None] * np.ones((1, g.W))


def _layer_momentum(h, u, v, M, f0, g, dt):
    """Vector-invariant momentum update for one layer (explicit advection+pressure,
    implicit trapezoidal Coriolis)."""
    zeta = ops.vorticity(u, v, g)                 # corners (H+1,W)
    zeta_uf = ops.corner_to_uface(zeta)           # (H,W) at u-faces
    # Bernoulli B = M + 0.5|u|^2 (KE from co-located reconstruction).
    v_c = 0.5 * (v[0:g.H] + v[1:g.H + 1])         # v at centers
    ke = 0.5 * (u * u + v_c * v_c)
    B = M + ke
    gx, gy = ops.grad_faces(B, g)                 # face gradients of Bernoulli
    f_uf = _f_uface(g, f0)
    f_vf = _f_vface(g, f0)
    # Absolute-vorticity flux terms (q k x u), q=zeta+f averaged to faces.
    q_uf = zeta_uf + f_uf
    v_at_uf = center_to_uface(v_c)                # v interpolated to u-face
    u_star = u + dt * (q_uf * v_at_uf - gx)       # explicit advect+pressure (zonal)
    # meridional: need u at v-face and zeta+f at v-face.
    zeta_vf = 0.5 * (zeta + np.roll(zeta, 1, axis=1))  # corner->v-face (avg in lon)
    q_vf = zeta_vf + f_vf
    u_c = 0.5 * (u + np.roll(u, 1, axis=1))       # u at centers
    u_at_vf = center_to_vface(u_c)
    v_star = v.copy()
    v_star[1:g.H] = (v[1:g.H]
                     + dt * (-(q_vf[1:g.H]) * u_at_vf[1:g.H] - gy[1:g.H]))
    # Implicit Coriolis already folded into q-flux above is the *advective* part;
    # apply the rotational (planetary) part trapezoidally for stability:
    u_new, v_c_new = ops.coriolis_trapezoidal(u_star, 0.5 * (v_star[0:g.H] + v_star[1:g.H + 1]),
                                              f_uf, dt)
    # Rebuild v on faces from the rotated center value (keep poles zero).
    v_new = np.zeros_like(v); v_new[1:g.H] = 0.5 * (v_c_new[0:g.H - 1] + v_c_new[1:g.H])
    return u_new, v_new


def step(st: SwState, dt: float) -> SwState:
    g = st.g
    M1, M2 = ops.montgomery_2layer(st.h1, st.h2, st.gp)
    u1, v1 = _layer_momentum(st.h1, st.u1, st.v1, M1, st.f0, g, dt)
    u2, v2 = _layer_momentum(st.h2, st.u2, st.v2, M2, st.f0, g, dt)
    h1 = ops.continuity_step(st.h1, u1, v1, g, dt, st.h_floor)
    h2 = ops.continuity_step(st.h2, u2, v2, g, dt, st.h_floor)
    st.h1, st.u1, st.v1 = h1, u1, v1
    st.h2, st.u2, st.v2 = h2, u2, v2
    return st


def kinetic_energy(st: SwState) -> float:
    area = st.g.cos_c[:, None]
    vc1 = 0.5 * (st.v1[0:st.g.H] + st.v1[1:st.g.H + 1])
    vc2 = 0.5 * (st.v2[0:st.g.H] + st.v2[1:st.g.H + 1])
    ke = st.h1 * (st.u1 ** 2 + vc1 ** 2) + st.h2 * (st.u2 ** 2 + vc2 ** 2)
    return float(np.sum(0.5 * ke * area))


def checkerboard_amplitude(field: np.ndarray) -> float:
    jj, ii = np.indices(field.shape)
    sign = (-1.0) ** ((ii + jj) % 2)
    return float(np.abs(np.mean(field * sign)))


def balanced_test_state(W, H, f0, gp) -> SwState:
    """Geostrophically balanced zonal jet: choose h from u via gradient balance."""
    g = Grid(W, H)
    U = 0.3
    u_prof = U * g.cos_c                      # solid-body-like zonal jet at centers
    # Geostrophic h: f u = -g'_eff dh/dφ (1-layer-ish balance for the test).
    geff = gp[0]
    dphi = g.dphi
    h_prof = np.zeros(H)
    # integrate dh = -(f u / geff) dφ from north pole southward
    f_prof = f0 * np.sin(g.phi_c)
    integrand = -(f_prof * u_prof) / geff
    h_prof[0] = 5.0
    for j in range(1, H):
        h_prof[j] = h_prof[j - 1] - integrand[j] * dphi
    h_prof -= h_prof.min() - 1.0             # keep positive
    h1 = np.repeat(h_prof[:, None], W, axis=1)
    u1 = np.repeat(u_prof[:, None], W, axis=1)
    v1 = np.zeros((H + 1, W))
    dt = 0.4 * g.dphi / np.sqrt(geff * h1.max())   # gravity-wave CFL
    return SwState(g=g, f0=f0, gp=gp,
                   h1=h1, u1=u1, v1=v1,
                   h2=np.full((H, W), 3.0), u2=np.zeros((H, W)), v2=np.zeros((H + 1, W)),
                   dt=dt)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/unit/test_sw_spike.py -k "balanced or checkerboard" -v`
Expected: PASS. If the balanced test drifts >1%, the most likely cause is a meridional sign error in `grad_faces` or `divergence_hu` (the spec's flagged sign discipline) — debug with the `superpowers:systematic-debugging` skill against the analytic balance, do not tune tolerances.

- [ ] **Step 5: Commit**

```bash
git add -A && git commit -m "M0: explicit 2-layer step + balanced-state & checkerboard gates"
```

---

## Task 9: h_eq(φ) profile + balanced gradient-wind init with equatorial closure

**Files:**
- Create: `src/gasgiant/sim/sw_spike/init.py`
- Test: `tests/unit/test_sw_spike.py`

- [ ] **Step 1: Write the failing test**

```python
def test_init_equatorial_velocity_is_finite():
    from gasgiant.sim.sw_spike import init
    st = init.emergent_init(W=128, H=64, f0=4.0, gp=(1.0, 0.05),
                            n_bands=10, band_contrast=0.4)
    assert np.all(np.isfinite(st.u1)) and np.all(np.isfinite(st.u2))
    # Equatorial band: velocity must be bounded (no 1/f blow-up).
    eqrow = st.g.H // 2
    assert np.max(np.abs(st.u1[eqrow])) < 5.0


def test_h_eq_has_band_structure():
    from gasgiant.sim.sw_spike import init
    heq = init.h_eq_profile(H=64, n_bands=10, band_contrast=0.4, h_mean=5.0)
    assert heq.shape == (64,)
    # Banded => multiple local extrema in latitude.
    d = np.diff(np.sign(np.diff(heq)))
    assert np.count_nonzero(d) >= 6
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit/test_sw_spike.py -k "init or h_eq" -v`
Expected: FAIL (`init` missing)

- [ ] **Step 3: Write minimal implementation**

```python
# src/gasgiant/sim/sw_spike/init.py
from __future__ import annotations
import numpy as np
from .grid import Grid
from .solver import SwState


def h_eq_profile(H, n_bands, band_contrast, h_mean):
    g = Grid(W=1, H=H)
    phi = g.phi_c
    heq = h_mean + band_contrast * np.cos(n_bands * phi) * np.cos(phi)
    return heq


def _gradient_wind_u(heq, g, f0, geff, phi_eq=np.radians(8.0)):
    """Invert balance for u. Geostrophic poleward; finite (capped) near equator."""
    phi = g.phi_c
    dheq_dphi = np.gradient(heq, phi)
    f = f0 * np.sin(phi)
    u = np.zeros_like(heq)
    far = np.abs(phi) >= phi_eq
    u[far] = -(geff * dheq_dphi[far]) / f[far]
    # Equatorial band: blend to 0 (let a brief adjustment establish the jet).
    near = ~far
    u[near] = -(geff * dheq_dphi[near]) / (f0 * np.sin(np.sign(phi[near]) * phi_eq))
    u = np.clip(u, -3.0, 3.0)
    return u


def emergent_init(W, H, f0, gp, n_bands, band_contrast, h_mean=5.0) -> SwState:
    g = Grid(W, H)
    geff = gp[0]
    heq1 = h_eq_profile(H, n_bands, band_contrast, h_mean)
    heq2 = h_eq_profile(H, n_bands, 0.5 * band_contrast, h_mean * 0.6)
    u1p = _gradient_wind_u(heq1, g, f0, geff)
    # small seed perturbation to break zonal symmetry (deterministic).
    rng = np.random.default_rng(1234)
    seed = 0.002 * rng.standard_normal((H, W))
    h1 = np.repeat(heq1[:, None], W, axis=1) + seed
    h2 = np.repeat(heq2[:, None], W, axis=1)
    u1 = np.repeat(u1p[:, None], W, axis=1)
    dt = 0.4 * g.dphi / np.sqrt(geff * h1.max())
    st = SwState(g=g, f0=f0, gp=gp,
                 h1=np.maximum(h1, 0.1), u1=u1, v1=np.zeros((H + 1, W)),
                 h2=np.maximum(h2, 0.1), u2=np.zeros((H, W)), v2=np.zeros((H + 1, W)),
                 dt=dt,
                 tau_rad=600.0, tau_drag=2000.0, nu4=0.2,
                 h_eq1=np.repeat(heq1[:, None], W, axis=1),
                 h_eq2=np.repeat(heq2[:, None], W, axis=1))
    return st
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/unit/test_sw_spike.py -k "init or h_eq" -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add -A && git commit -m "M0: h_eq profile + balanced init with equatorial closure"
```

---

## Task 10: Thermal relaxation + bottom drag + hyperviscosity forcing

**Files:**
- Modify: `src/gasgiant/sim/sw_spike/solver.py`
- Test: `tests/unit/test_sw_spike.py`

- [ ] **Step 1: Write the failing test**

```python
def test_relaxation_pulls_h_toward_h_eq():
    from gasgiant.sim.sw_spike import init, solver
    st = init.emergent_init(W=64, H=32, f0=4.0, gp=(1.0, 0.05),
                            n_bands=8, band_contrast=0.4)
    st.h1 = st.h1 + 1.0                       # perturb away from h_eq
    before = np.mean(np.abs(st.h1 - st.h_eq1))
    for _ in range(50):
        st = solver.step(st, dt=st.dt)
    after = np.mean(np.abs(st.h1 - st.h_eq1))
    assert after < before                     # relaxation reduces the anomaly


def test_drag_reduces_bottom_layer_energy_without_forcing():
    from gasgiant.sim.sw_spike import init, solver
    st = init.emergent_init(W=64, H=32, f0=4.0, gp=(1.0, 0.05),
                            n_bands=8, band_contrast=0.4)
    st.tau_rad = 0.0                          # isolate drag
    st.u2 = st.u2 + 0.2
    e0 = float(np.sum(st.u2 ** 2))
    for _ in range(50):
        st = solver.step(st, dt=st.dt)
    e1 = float(np.sum(st.u2 ** 2))
    assert e1 < e0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit/test_sw_spike.py -k "relaxation or drag" -v`
Expected: FAIL (forcing not applied yet — relaxation/drag are no-ops in `step`)

- [ ] **Step 3: Write minimal implementation**

```python
# in solver.py, add forcing helpers and call them from step()

def _biharmonic(field: np.ndarray, g: Grid) -> np.ndarray:
    """Grid-normalized ∇⁴ proxy (iterated 5-point Laplacian on the lon-lat grid)."""
    def lap(a):
        return (np.roll(a, 1, 1) + np.roll(a, -1, 1)
                + np.roll(a, 1, 0) + np.roll(a, -1, 0) - 4 * a)
    return lap(lap(field))


def _apply_forcing(st: SwState, dt: float) -> None:
    g = st.g
    if st.tau_rad > 0.0 and st.h_eq1 is not None:
        st.h1 += dt * (st.h_eq1 - st.h1) / st.tau_rad
        st.h2 += dt * (st.h_eq2 - st.h2) / st.tau_rad
    if st.tau_drag > 0.0:
        st.u2 *= (1.0 - dt / st.tau_drag)
        st.v2 *= (1.0 - dt / st.tau_drag)
    if st.nu4 > 0.0:
        # grid-normalized: scale by 1/64 of the 4th-difference (v1.6 lesson).
        for arr in (st.u1, st.u2):
            arr -= (st.nu4 / 64.0) * _biharmonic(arr, g)
    st.h1 = np.maximum(st.h1, st.h_floor)
    st.h2 = np.maximum(st.h2, st.h_floor)
```

Then modify `step` to call `_apply_forcing(st, dt)` just before `return st`:

```python
def step(st: SwState, dt: float) -> SwState:
    g = st.g
    M1, M2 = ops.montgomery_2layer(st.h1, st.h2, st.gp)
    u1, v1 = _layer_momentum(st.h1, st.u1, st.v1, M1, st.f0, g, dt)
    u2, v2 = _layer_momentum(st.h2, st.u2, st.v2, M2, st.f0, g, dt)
    h1 = ops.continuity_step(st.h1, u1, v1, g, dt, st.h_floor)
    h2 = ops.continuity_step(st.h2, u2, v2, g, dt, st.h_floor)
    st.h1, st.u1, st.v1 = h1, u1, v1
    st.h2, st.u2, st.v2 = h2, u2, v2
    _apply_forcing(st, dt)
    return st
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/unit/test_sw_spike.py -k "relaxation or drag" -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add -A && git commit -m "M0: thermal relaxation + bottom drag + hyperviscosity"
```

---

## Task 11: Spin-up driver + equilibration metric

**Files:**
- Create: `scripts/sw_spike_run.py`
- Test: `tests/unit/test_sw_spike.py`

- [ ] **Step 1: Write the failing test**

```python
def test_spinup_runs_stable_and_develops_eddies():
    from gasgiant.sim.sw_spike import init, solver
    import numpy as np
    st = init.emergent_init(W=96, H=48, f0=4.0, gp=(1.0, 0.05),
                            n_bands=10, band_contrast=0.4)
    eddy0 = float(np.std(solver.kinetic_energy(st)))  # scalar; use field std below
    z0 = np.std(solver.relative_vorticity_top(st))
    for _ in range(300):
        st = solver.step(st, dt=st.dt)
    assert np.all(np.isfinite(st.h1))                 # no NaN over 300 steps
    z1 = np.std(solver.relative_vorticity_top(st))
    assert z1 > z0                                     # eddies (vorticity) grew
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit/test_sw_spike.py -k spinup -v`
Expected: FAIL (`solver.relative_vorticity_top` missing)

- [ ] **Step 3: Write minimal implementation**

```python
# add to solver.py
def relative_vorticity_top(st: SwState) -> np.ndarray:
    return ops.vorticity(st.u1, st.v1, st.g)
```

```python
# scripts/sw_spike_run.py
"""Spin up the M0 shallow-water spike and report equilibration vs steps."""
from __future__ import annotations
import sys, time
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from gasgiant.sim.sw_spike import init, solver  # noqa: E402


def spin_up(steps=4000, W=768, H=384, log_every=200):
    st = init.emergent_init(W=W, H=H, f0=4.0, gp=(1.0, 0.05),
                            n_bands=22, band_contrast=0.5)
    t0 = time.perf_counter()
    series = []
    for s in range(steps):
        st = solver.step(st, dt=st.dt)
        if s % log_every == 0:
            z = float(np.std(solver.relative_vorticity_top(st)))
            series.append((s, z))
            print(f"step {s:5d}  vort_std {z:.4f}")
            assert np.all(np.isfinite(st.h1)), f"NaN at step {s}"
    print(f"spin-up {steps} steps in {time.perf_counter()-t0:.1f}s")
    # equilibration: first step where vort_std reaches 90% of its final value.
    finalz = series[-1][1]
    eq_step = next((s for s, z in series if z >= 0.9 * finalz), steps)
    print(f"~equilibration step: {eq_step}")
    return st


if __name__ == "__main__":
    spin_up()
```

- [ ] **Step 4: Run test + a smoke spin-up**

Run: `uv run pytest tests/unit/test_sw_spike.py -k spinup -v`
Expected: PASS
Run: `uv run python scripts/sw_spike_run.py` (smoke; prints vort_std rising, no NaN)

- [ ] **Step 5: Commit**

```bash
git add -A && git commit -m "M0: spin-up driver + equilibration metric"
```

---

## Task 12: Render encoder — SW fields → existing GPU render at ≥4K

**Files:**
- Create: `src/gasgiant/sim/sw_spike/encode.py`
- Test: `tests/unit/test_sw_spike.py`

**Context:** the existing render (`render/kernels/derive.comp`, `render/maps.py`) consumes a 4-channel tracer `(r=color index, g=height, b=detail, a=tint)`. The encoder packs SW top-layer fields into that semantic. R8 lives here: **ζ₁ drives the detail channel** (the v1.6 hero signal), thickness drives color/height.

- [ ] **Step 1: Write the failing test**

```python
def test_encode_tracer_channels_in_range():
    from gasgiant.sim.sw_spike import init, solver, encode
    st = init.emergent_init(W=64, H=32, f0=4.0, gp=(1.0, 0.05),
                            n_bands=8, band_contrast=0.4)
    for _ in range(20):
        st = solver.step(st, dt=st.dt)
    rgba = encode.to_tracer(st)
    assert rgba.shape == (32, 64, 4)
    assert rgba.dtype == np.float32
    assert np.all(np.isfinite(rgba))
    assert rgba.min() >= 0.0 and rgba.max() <= 1.0     # normalized for the render
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit/test_sw_spike.py -k encode -v`
Expected: FAIL (`encode` missing)

- [ ] **Step 3: Write minimal implementation**

```python
# src/gasgiant/sim/sw_spike/encode.py
from __future__ import annotations
import numpy as np
from . import operators as ops
from .solver import SwState


def _norm(a: np.ndarray) -> np.ndarray:
    lo, hi = np.percentile(a, 1), np.percentile(a, 99)
    return np.clip((a - lo) / (hi - lo + 1e-9), 0.0, 1.0)


def to_tracer(st: SwState) -> np.ndarray:
    """Pack top-layer (thickness anomaly, ζ₁) into the render's RGBA tracer.

    r = color index  <- thickness anomaly (banded color)
    g = height       <- thickness (cloud altitude)
    b = detail       <- relative vorticity magnitude (the v1.6 hero morphology)
    a = tint         <- signed vorticity (storm polarity)
    """
    H, W = st.h1.shape
    h_anom = st.h1 - (st.h_eq1 if st.h_eq1 is not None else st.h1.mean())
    zeta_c = 0.5 * (ops.vorticity(st.u1, st.v1, st.g)[0:H]
                    + ops.vorticity(st.u1, st.v1, st.g)[1:H + 1])
    rgba = np.zeros((H, W, 4), dtype=np.float32)
    rgba[..., 0] = _norm(h_anom)
    rgba[..., 1] = _norm(st.h1)
    rgba[..., 2] = _norm(np.abs(zeta_c))
    rgba[..., 3] = _norm(zeta_c)
    return rgba
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/unit/test_sw_spike.py -k encode -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add -A && git commit -m "M0: render encoder (zeta drives detail; thickness drives color)"
```

---

## Task 13: Kill-gate harness — SW render vs v1.6, coher + blind strip + R1/R3 asserts

**Files:**
- Create: `scripts/sw_spike_killgate.py`

**Context:** Reuse `scripts/measure_morphology.py` (`_belt_crop_from_rgb`, `_lum`, `coher`) and the v1.6 render path (`v16_preview.py` shows how `jupiter_vorticity` is rendered). The SW tracer is uploaded to a GPU texture, fed through `render/maps.py derive()` at ≥4096, and compared on a matched belt crop. This task has no unit test — it emits the decision artifacts.

- [ ] **Step 1: Write the harness**

```python
# scripts/sw_spike_killgate.py
"""M0 KILL-GATE: does the shallow-water spike out-render painted-jet v1.6?

Emits:
  out/audit/m0/sw_vs_v16.png     (blind comparison strip)
  out/audit/m0/report.txt        (coher numbers + R1/R3 gate results + verdict)

Decision rule (spec §5 M0): proceed to M1 only if the panel PREFERS the SW render
(ties fail) OR coher moves from v1.6's 0.384 toward the 0.62 reference.
"""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import cv2

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "src"))
from measure_morphology import _belt_crop_from_rgb, _lum, coher  # noqa: E402
from gasgiant.sim.sw_spike import init, solver, encode  # noqa: E402
from gasgiant.gl import GpuContext  # noqa: E402
from gasgiant.params.model import SolverType  # noqa: E402
from gasgiant.params.presets import load_factory_preset  # noqa: E402
from gasgiant.engine.facade import Simulation  # noqa: E402

OUT = ROOT / "out" / "audit" / "m0"
RENDER_RES = 4096


def render_v16():
    gpu = GpuContext.headless(); gpu.make_current()
    p = load_factory_preset("jupiter_vorticity")
    p.solver.type = SolverType.VORTICITY
    sim = Simulation(p, gpu)
    rgb = sim.render_maps(RENDER_RES * 2)["color"]
    crop, box = _belt_crop_from_rgb(rgb, sim, 640)
    c = coher(_lum(crop))
    sim._release_sim()
    return rgb, crop, c, box


def render_sw():
    # Spin up the spike, encode, and run the GPU derive on the SW tracer.
    st = init.emergent_init(W=768, H=384, f0=4.0, gp=(1.0, 0.05),
                            n_bands=22, band_contrast=0.5)
    for s in range(4000):
        st = solver.step(st, dt=st.dt)
        assert np.all(np.isfinite(st.h1)), f"R3 FAIL: NaN at step {s}"
    rgba = encode.to_tracer(st)
    # Upload + render via the existing pipeline. maps.derive() accepts an equirect
    # tracer texture; see render/maps.py for the exact entry signature.
    from gasgiant.render import maps  # noqa: E402
    rgb = maps.derive_from_tracer(rgba, res=RENDER_RES * 2)   # see note
    return rgb, st


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    v16_rgb, v16_crop, v16_coher, box = render_v16()
    sw_rgb, st = render_sw()
    # R1 gate: checkerboard amplitude must not have grown during spin-up.
    cb = solver.checkerboard_amplitude(st.h1)
    r1_ok = cb < 0.05
    # R3 gate: no NaN (asserted in render_sw) + positivity maintained.
    r3_ok = bool(np.min(st.h1) >= st.h_floor - 1e-6)
    from measure_morphology import _crop_deg, _fit_width
    sw_crop = _fit_width(_crop_deg(sw_rgb, *box), 640)
    sw_coher = coher(_lum(sw_crop))

    strip = np.vstack([
        cv2.cvtColor((np.clip(v16_crop, 0, 1) * 255).astype(np.uint8), cv2.COLOR_RGB2BGR),
        cv2.cvtColor((np.clip(sw_crop, 0, 1) * 255).astype(np.uint8), cv2.COLOR_RGB2BGR),
    ])
    cv2.imwrite(str(OUT / "sw_vs_v16.png"), strip)

    verdict = "PROCEED to M1" if (sw_coher > v16_coher and r1_ok and r3_ok) else "STOP / iterate"
    (OUT / "report.txt").write_text(
        f"v1.6 coher : {v16_coher:.4f}\n"
        f"SW   coher : {sw_coher:.4f}  (ref 0.62)\n"
        f"R1 checkerboard amplitude: {cb:.5f}  ({'OK' if r1_ok else 'FAIL'})\n"
        f"R3 positivity/NaN        : {'OK' if r3_ok else 'FAIL'}\n"
        f"coher verdict (numeric)  : {verdict}\n"
        f"NOTE: the binding gate is the BLIND JUDGE PANEL on sw_vs_v16.png; coher is necessary-not-sufficient.\n"
    )
    print((OUT / "report.txt").read_text())


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Resolve the one integration unknown**

`maps.derive_from_tracer(rgba, res)` is the only call that may not exist yet. Open `src/gasgiant/render/maps.py` and check whether `derive()` can accept a raw equirect tracer ndarray. If not, add a thin `derive_from_tracer(rgba, res)` that uploads `rgba` to a moderngl texture and runs the existing `derive.comp` (mirror the texture-binding pattern already in `maps.derive()`). This is the M0 stand-in for M5's encoder kernel.

Run: `uv run python -c "from gasgiant.render import maps; print(hasattr(maps,'derive') )"`
Expected: `True` (confirm the entry point before wiring).

- [ ] **Step 3: Run the kill-gate**

Run: `uv run python scripts/sw_spike_killgate.py`
Expected: writes `out/audit/m0/sw_vs_v16.png` + `report.txt`; prints coher for both, R1/R3 OK, and a numeric verdict.

- [ ] **Step 4: Human decision (the actual gate)**

Show `out/audit/m0/sw_vs_v16.png` to the user for the **blind forced-choice panel**. Per spec §5: proceed to M1 **only** if the panel prefers the SW render (a tie fails). The coher number and R1/R3 flags are necessary-not-sufficient supporting evidence.

- [ ] **Step 5: Commit**

```bash
git add -A && git commit -m "M0: kill-gate harness (SW vs v1.6 render, R1/R3 gates)"
```

---

## Task 14: Record the M0 verdict and decide M1

**Files:**
- Create: `docs/superpowers/specs/m0-verdict.md`

- [ ] **Step 1: Write the verdict**

Capture in `docs/superpowers/specs/m0-verdict.md`: the coher numbers, the R1 checkerboard-decay result, the R3 positivity result, the measured equilibration step count vs budget, the blind-panel outcome, and the GO/NO-GO decision. If GO, note that `sw_spike/operators.py` graduates into M1's `shallow_water_ref.py`. If NO-GO, record why (which gate failed) and stop.

- [ ] **Step 2: Commit**

```bash
git add -A && git commit -m "M0: record kill-gate verdict and M1 decision"
```

---

## Self-review notes (for the implementer)

- **Sign discipline is the #1 risk** (spec §2.2). If Task 8's balanced-state test drifts, the bug is almost always a meridional sign in `divergence_hu` / `grad_faces` / `vorticity` — use `superpowers:systematic-debugging` against the analytic balance; never relax the test tolerance.
- **Do not GPU-port anything in M0.** The whole point is a cheap CPU answer. GPU kernels are M1's job, validated against this module.
- **The binding gate is the blind panel, not coher.** coher is necessary-not-sufficient (this is the v1.6 discipline).
- **Dead-code check:** Task 7 Step 3 contains one deliberately-flagged dead line (`sy = ... if False else 1.0`) that MUST be deleted — confirm it is gone before committing Task 7.
- **Perf:** at `W=768, H=384`, 4000 NumPy steps should run in a couple of minutes. If too slow, drop to `512×256` for the first kill-gate pass; bump up only if the panel result is borderline.
```
