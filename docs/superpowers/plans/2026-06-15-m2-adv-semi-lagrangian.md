# M2-adv — Semi-Lagrangian Semi-Implicit Advection Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking. The CPU reference (`shallow_water_ref.py`) is the gold-standard ground truth and is extended FIRST in every task; GPU kernels are then diffed per-field against it at `atol=2e-5` on pre-division quantities.

**Goal:** Remove the residual *advective* CFL from the M2-core semi-implicit shallow-water solver so it takes large stable **and accurate** steps on fast zonal jets, while keeping mass exactly conserved and the M2-core / M1 paths byte-identical.

**Architecture:** M2-core's `step_semi_implicit` already makes the gravity-wave terms implicit (height increment `dh` via a symmetric Helmholtz solve) and isolates the *explicit* transport into exactly two operators: (1) the predictor's Eulerian vector-invariant momentum advection (`_semi_implicit_predictor`), and (2) the step-5 nonlinear anomaly transport (`continuity_step_conservative`, an FCT). Both carry the advective CFL. M2-adv replaces **only** those two operators with semi-Lagrangian equivalents — SLICE-style conservative cascade remapping for height, tri-cubic departure-point interpolation for momentum — feeding the *unchanged* M2-core Helmholtz core. A `fast_advection` flag gives a nested byte-identical fallback to M2-core (and thence to M1).

**Tech Stack:** Python 3.13 + NumPy CPU reference; ModernGL R32F compute kernels reusing `sw_common.glsl` metric helpers and the M2-core Helmholtz kernels verbatim; pytest with the session-scoped `gpu` fixture in `tests/conftest.py`.

---

## Background the engineer needs (read before Task 1)

**The C-grid layout** (`shallow_water_ref.py:28-66`): row 0 = NORTH, φ descending. `h`, `u` are `(H, W)`; `v` is `(H+1, W)` with `v[0]=v[H]=0` (poles); vorticity `ζ` at corners `(H+1, W)`. `Grid(W, H, a)` carries planetary radius `a`; metric sites multiply by `1/(a cosφ)` zonally and `1/a` meridionally. `dlam = 2π/W`, `dphi = π/H`, `phi_c` (centers) descending, `cos_v[0]=cos_v[H]=0`.

**The M2-core step** (`step_semi_implicit`, `shallow_water_ref.py:1136-1240`) does:
1. `u_star, v_star = _semi_implicit_predictor(h, u, v, gp, g, dt, theta)` — Eulerian vorticity-flux advection + KE gradient + `(1-θ)` explicit pressure, **no Coriolis**. ← *advective-CFL site #1*
2. `H_ref_lat = reference_depth(h)` — frozen latitude-only mean depth.
3. Picard loop: `rhs = helmholtz_rhs(...)`, `dh = helmholtz_sor(...)` — implicit gravity-wave increment. *(CFL-free; reuse verbatim.)*
4. `u_new, v_new = velocity_backsub(u_star, v_star, h + dh, gp, theta, dt, omega, g)` — implicit pressure + Coriolis. *(Reuse verbatim.)*
5. Final height:
   ```python
   h_fct   = continuity_step_conservative(h, u_new, v_new, g, dt, st.h_floor)  # advective-CFL site #2
   h_linref = h - dt * divergence_helmholtz(u_new, v_new, H_ref_lat, g)        # linear ref-div (implicit part)
   anomaly  = h_fct - h_linref
   h_raw    = h + dh + anomaly
   ```

**The M2-adv surgery:** replace site #1 (`_semi_implicit_predictor`'s advection) and site #2 (`continuity_step_conservative`) with SL operators. Site #1's vorticity-flux advection becomes SL departure-point interpolation of `u, v`, keeping the same KE-gradient and `(1-θ)` pressure terms.

Site #2 is **not** a one-line swap — this is the coupling subtlety the adversarial review caught (FATAL). M2-core's step-5 used `anomaly = h_fct − h_linref` where both `h_fct` (FCT) **and** `h_linref = h − dt·divergence_helmholtz(u_new, v_new, H_ref, g)` are *Eulerian flux-divergence* transports, so the reference flux `div(H_ref·u_new)` cancels term-by-term, leaving the anomaly transport `−dt·div((h−H_ref)·u_new)`. If we keep the Eulerian `h_linref` but make `h_sl` a *finite-Courant Lagrangian* remap, the reference flux no longer cancels (the SL reference part is multi-order in Courant, the Eulerian one is first-order) — the reference divergence is partially double-counted against the implicit `dh`, with an O(1) error exactly at Courant ≫ 1. **Mass conservation does NOT detect this** (each term is individually flux-form). The fix: remove the reference part in the **same conservative-remap form** as `h_sl`, by remapping the (broadcast) `H_ref` field along the same trajectory:

```python
H_ref_field = np.broadcast_to(H_ref_lat[:, None], h.shape)
h_sl    = slice_remap_advance(h,           u_new, v_new, dt, g)   # remap of full h     ≈ -dt div(h u)
href_sl = slice_remap_advance(H_ref_field, u_new, v_new, dt, g)   # remap of H_ref      ≈ -dt div(H_ref u)
anomaly = (h_sl - h) - (href_sl - H_ref_field)                    # = -dt div((h-H_ref) u), Courant-consistent
h_raw   = h + dh + anomaly
```

Now both pieces of `anomaly` are conservative-remap form, so the reference flux cancels at **all** Courant numbers, and `dh` keeps the implicit θ-centered reference divergence exactly as M2-core does (no double-count). Because conservative remap of a zonally-uniform `H_ref` under a *divergent* `u` concentrates/dilutes it (the departure-footprint Jacobian = `div(u)`), `href_sl − H_ref` captures the full `−dt·div(H_ref·u)` (including the `H_ref·∇·u` gravity-wave part) — the finite-Courant-correct version of M2-core's `divergence_helmholtz` term. `helmholtz_rhs` is therefore genuinely reused verbatim. *(Verified by a second adversarial pass: all five coupling claims — cancellation at all Courant, single reference appearance, mass-neutrality, small-dt reduction, RHS-verbatim — confirmed.)*

**Why two remaps, not a single `slice_remap_advance(h − H_ref)`:** the monotone PPM limiter assumes a positive-definite field. `h` and `H_ref` are both positive, so each remap is well-limited; the signed anomaly `h − H_ref` is not, so remapping it directly would mis-limit. The two-remap form leaves only a benign, mass-neutral O(limiter) cross-term (vs the FATAL O(Courant²) double-count of the Eulerian `h_linref`), so it is the robust choice here.

**Why this is conservative:** SLICE conserves mass by construction (Σ remapped ≡ Σ source in the `cosφ·a²` measure), so `Σ h_sl·cosφ = Σ h·cosφ` and `Σ href_sl·cosφ = Σ H_ref·cosφ` exactly; hence `Σ anomaly·cosφ = 0` and (with `Σ dh·cosφ = 0` from the flux-form Helmholtz) `Σ h_raw·cosφ = Σ h·cosφ`. The mass-conservation gate (M2-core T5) carries over — but note it is **insensitive** to the coupling bug above, so the crux accuracy gate (Task 5) and a dedicated balanced-anomaly coupling test (Task 5 Step 5c) are the real guards.

**Determinism discipline (every task):** fixed iteration counts, no convergence early-out, no in-loop branching on data; `wrapX` branch form (`if(x<0)return x+W; if(x>=W)return x-W;`) never `((x%W)+W)%W`; reuse the session `gpu` fixture; independent a-scaling test (`a=1` vs `a=6.4e6`) at every metric site.

---

## File structure

| File | Responsibility | Action |
|---|---|---|
| `src/gasgiant/sim/shallow_water_ref.py` | CPU ground truth: trajectory, conservative remap, SL momentum, `step_slsi` | Modify (append; M2-core funcs untouched) |
| `src/gasgiant/sim/sw_gpu.py` | GPU solver: `fast_advection` flag, SL drivers, `_step_slsi` | Modify |
| `src/gasgiant/sim/kernels/sw_departure.comp` | Iterated-midpoint trajectory kernel | Create |
| `src/gasgiant/sim/kernels/sw_slice_remap.comp` | Conservative cascade PPM remap (2 passes) | Create |
| `src/gasgiant/sim/kernels/sw_sl_momentum.comp` | Tri-cubic departure interpolation of u,v | Create |
| `src/gasgiant/sim/kernels/sw_sl_common.glsl` | Shared SL helpers (PPM reconstruct, cubic weights, departure trace) | Create |
| `tests/unit/test_m2_adv_ref.py` | CPU unit tests (trajectory, remap, momentum, step) | Create |
| `tests/unit/test_m2_adv_gpu.py` | Per-kernel + full-step GPU↔CPU diffs | Create |
| `tests/unit/test_m2_adv_conservation.py` | Mass/positivity/determinism gates | Create |
| `tests/unit/test_dual_path_adv.py` | Nested byte-identity (fast_advection=False) | Create |
| `tests/spikes/test_slsi_fastjet_spike.py` | The crux accuracy gate (go/no-go) | Create |
| `scripts/sw_m2_adv_validation.py` | Consolidated PASS/FAIL report | Create |
| `docs/superpowers/specs/m2-adv-verdict.md` | Gate numbers + honest headline factor | Create |

---

## Task 1: Departure-point trajectory solver (CPU ref)

**Files:**
- Modify: `src/gasgiant/sim/shallow_water_ref.py` (append after `total_potential_enstrophy`)
- Test: `tests/unit/test_m2_adv_ref.py` (create)

The departure point of an arrival grid point is where the parcel now at the arrival point was one step ago: `x_dep = x_arr − ∫ u dt`. We use the standard 2-iteration midpoint (implicit) trajectory: estimate the midpoint velocity, refine. Velocities are angular: zonal `λ̇ = u/(a cosφ)`, meridional `φ̇ = v/a`. We return departure points in **fractional grid index space** `(i_dep, j_dep)` for the remap/interpolation tasks (arrival cell center `(j+0.5, i)` maps back).

- [ ] **Step 1: Write the failing test** (solid-body zonal flow ⇒ pure longitude shift)

```python
# tests/unit/test_m2_adv_ref.py
import numpy as np
from gasgiant.sim.shallow_water_ref import Grid, departure_points

def test_departure_solid_body_zonal():
    """Constant u (solid-body in index space along a mid-latitude row): the
    departure point is shifted west by exactly C = u*dt/(a cosφ dλ) cells."""
    g = Grid(W=64, H=32, a=6.4e6)
    j0 = 8                                  # a mid-latitude row (row 0 = north)
    cosphi = g.cos_c[j0]
    u_val = 30.0                            # m/s eastward
    dt = 600.0
    u = np.full((g.H, g.W), u_val)
    v = np.zeros((g.H + 1, g.W))
    i_dep, j_dep = departure_points(u, v, dt, g, n_iter=2)
    C = u_val * dt / (g.a * cosphi * g.dlam)
    i_arr = np.arange(g.W)[None, :] + 0.0   # arrival zonal index = i (u east-face of cell i)
    expected_i = i_arr - C                  # shifted west by C cells
    assert np.allclose(i_dep[j0], expected_i[0] if False else (np.arange(g.W) - C), atol=1e-9)
    assert np.allclose(j_dep[j0], j0 + 0.5, atol=1e-9)   # no meridional motion
```

- [ ] **Step 2: Run to verify it fails** — `pytest tests/unit/test_m2_adv_ref.py::test_departure_solid_body_zonal -v` → FAIL (`departure_points` undefined).

- [ ] **Step 3: Implement** (append to `shallow_water_ref.py`)

```python
def departure_points(u, v, dt, g, n_iter=2):
    """Back-trajectory departure points for arrival CELL CENTERS, in fractional
    grid-index space (i_dep zonal, j_dep meridional).  Row 0 = north, φ descending.

    Angular velocities: lam_dot = u/(a cosφ), phi_dot = v/a.  Index velocities:
      di/dt = lam_dot / dlam ;  dj/dt = -phi_dot / dphi   (j increases southward).
    Two-iteration implicit midpoint: evaluate the velocity at the current
    midpoint estimate, refine.  Velocities are sampled at cell centers by
    averaging the C-grid faces (u east+west, v north+south).
    """
    H, W = u.shape
    # Cell-center velocities (average the two bounding faces).
    u_c = 0.5 * (u + np.roll(u, 1, axis=1))                 # (H,W) west+east face
    v_c = 0.5 * (v[0:H] + v[1:H + 1])                       # (H,W) north+south face
    cosphi = g.cos_c[:, None]                               # (H,1)
    # Index-space velocities at the arrival centers (cells per step).
    di = (u_c / (g.a * cosphi)) * dt / g.dlam               # eastward => +i
    dj = -(v_c / g.a) * dt / g.dphi                         # northward (v>0) => -j (toward row 0)
    i_arr = np.arange(W)[None, :] + np.zeros((H, 1))        # (H,W) arrival i
    j_arr = np.arange(H)[:, None] + 0.5                     # (H,W) arrival j (center)
    # Midpoint iteration: sample velocity at the half-way departure estimate.
    a_i, a_j = di.copy(), dj.copy()
    for _ in range(n_iter):
        im = i_arr - 0.5 * a_i                              # midpoint index
        jm = j_arr - 0.5 * a_j
        a_i = _bilinear_periodic(di, im, jm, g)             # resample index-velocity
        a_j = _bilinear_periodic(dj, im, jm, g)
    return i_arr - a_i, j_arr - a_j

def _bilinear_periodic(field, i_idx, j_idx, g):
    """Bilinear sample of a center field at fractional (i_idx zonal, j_idx-0.5
    row).  Zonal periodic wrap (branch form); meridional clamp to [0, H-1]."""
    H, W = field.shape
    jj = np.clip(j_idx - 0.5, 0.0, H - 1.0)                 # center-row coordinate
    j0 = np.floor(jj).astype(int); j1 = np.minimum(j0 + 1, H - 1)
    fy = jj - j0
    i0f = np.floor(i_idx); fx = i_idx - i0f
    i0 = i0f.astype(int) % W; i1 = (i0 + 1) % W             # periodic
    f00 = field[j0, i0]; f10 = field[j0, i1]
    f01 = field[j1, i0]; f11 = field[j1, i1]
    return ((1 - fx) * (1 - fy) * f00 + fx * (1 - fy) * f10
            + (1 - fx) * fy * f01 + fx * fy * f11)
```

- [ ] **Step 4: Run to verify it passes** — same command → PASS.

- [ ] **Step 5: Add the a-scaling + meridional tests**

```python
def test_departure_a_scaling_invariant():
    """Index-space departure is a-invariant: u_phys/a sets the angular speed, so
    scaling a with u fixed scales the shift; the cells-per-step C must change with a."""
    for a in (1.0, 6.4e6):
        g = Grid(W=48, H=24, a=a)
        u = np.full((g.H, g.W), 20.0); v = np.zeros((g.H + 1, g.W))
        i_dep, _ = departure_points(u, v, 300.0, g, n_iter=2)
        C = 20.0 * 300.0 / (g.a * g.cos_c[10] * g.dlam)
        assert np.allclose(i_dep[10], np.arange(g.W) - C, atol=1e-9)

def test_departure_meridional_shift():
    """Constant v>0 (northward) moves the departure point south (+j)."""
    g = Grid(W=32, H=32, a=6.4e6)
    u = np.zeros((g.H, g.W)); v = np.full((g.H + 1, g.W), 10.0); v[0] = 0.0; v[-1] = 0.0
    _, j_dep = departure_points(u, v, 300.0, g, n_iter=2)
    Cj = 10.0 / g.a * 300.0 / g.dphi
    assert np.all(j_dep[5:25] > (np.arange(5, 25)[:, None] + 0.5) - 1e-9)  # moved toward larger j
```

- [ ] **Step 6: Run all three** → PASS. **Commit**

```bash
git add src/gasgiant/sim/shallow_water_ref.py tests/unit/test_m2_adv_ref.py
git commit -m "M2-adv: departure-point trajectory solver (CPU ref)"
```

---

## Task 2: Conservative 1-D PPM remap (the SLICE building block)

**Files:**
- Modify: `src/gasgiant/sim/shallow_water_ref.py`
- Test: `tests/unit/test_m2_adv_ref.py`

SLICE remaps mass conservatively by reconstructing a piecewise-parabolic (PPM) sub-grid profile of the density, then integrating it over each departure cell. The 1-D conservative remap is the kernel reused twice (zonal then meridional) in the cascade. It takes a cell-integrated quantity `m` (mass per cell) on a periodic 1-D line and the fractional cell edges of the departure cells, and returns the remapped masses. **Exact conservation** because every sub-interval integral is added to exactly one source and one destination.

- [ ] **Step 1: Write the failing test** (uniform field is preserved; total mass exact)

```python
def test_ppm_remap_conserves_and_preserves_uniform():
    from gasgiant.sim.shallow_water_ref import ppm_remap_1d_periodic
    n = 64
    m = np.ones(n)                                  # uniform mass per cell
    edges = np.arange(n + 1, dtype=float) - 0.37    # rigid shift by 0.37 cell (periodic)
    out = ppm_remap_1d_periodic(m, edges)
    assert abs(out.sum() - m.sum()) < 1e-12         # exact conservation
    assert np.allclose(out, 1.0, atol=1e-12)        # uniform preserved exactly
```

- [ ] **Step 2: Run to verify it fails** → FAIL (`ppm_remap_1d_periodic` undefined).

- [ ] **Step 3: Implement** (append)

```python
def ppm_remap_1d_periodic(m, edges):
    """Conservative PPM remap of per-cell masses `m` (length n, periodic) onto a
    new set of cell edges `edges` (length n+1, fractional source-index
    coordinates; edges[k]..edges[k+1] is the k-th destination cell expressed in
    the SOURCE grid).  Returns remapped per-cell masses (length n).

    Method: build a monotone parabolic reconstruction of the density (mass per
    unit index = m, since source cells have unit width) and integrate it over
    each destination interval.  Conservative because Σ ∫ = Σ m exactly.
    """
    n = len(m)
    # PPM edge values (4th-order) with monotonic limiting (Colella-Woodward).
    mL = np.roll(m, 1); mR = np.roll(m, -1)
    # Edge value at the left face of cell i (between i-1 and i).
    aL_raw = (7.0 * (m + mL) - (mR + np.roll(m, 2))) / 12.0
    aL = _ppm_monotone_edge(aL_raw, mL, m)
    aR = np.roll(aL, -1)                              # right face = next cell's left face
    aL, aR = _ppm_limit_parabola(m, aL, aR)
    # Integrate the parabola of source cell s over [0,1] local coordinate.
    # density(s, xi) = aL[s] + xi*(aR[s]-aL[s]) + xi*(1-xi)*6*(m[s]-0.5*(aL[s]+aR[s]))
    def integral(s, x0, x1):
        d = aR[s] - aL[s]
        c6 = 6.0 * (m[s] - 0.5 * (aL[s] + aR[s]))
        def F(x):                                    # antiderivative of density
            return (aL[s] * x + 0.5 * d * x * x
                    + c6 * (0.5 * x * x - x * x * x / 3.0))
        return F(x1) - F(x0)
    out = np.empty(n)
    for k in range(n):                               # destination cell k
        out[k] = _accumulate_interval(edges[k], edges[k + 1], n, integral)
    return out

def _ppm_monotone_edge(aL_raw, mL, m):
    """Clamp the raw edge value into [min,max] of the two bounding cells."""
    lo = np.minimum(mL, m); hi = np.maximum(mL, m)
    return np.clip(aL_raw, lo, hi)

def _ppm_limit_parabola(m, aL, aR):
    """Colella-Woodward parabola limiter: kill overshoots / enforce monotonicity."""
    aL = aL.copy(); aR = aR.copy()
    d = aR - aL
    excess = d * (m - 0.5 * (aL + aR))
    d2 = d * d / 6.0
    flat = (aR - m) * (m - aL) <= 0.0                # local extremum -> flat
    over_l = excess > d2
    over_r = excess < -d2
    aL = np.where(flat, m, aL); aR = np.where(flat, m, aR)
    aL = np.where(~flat & over_l, 3.0 * m - 2.0 * aR, aL)
    aR = np.where(~flat & over_r, 3.0 * m - 2.0 * aL, aR)
    return aL, aR

def _accumulate_interval(x0, x1, n, integral):
    """Integrate the reconstructed density over [x0, x1] in periodic source
    coordinates, summing whole and partial source-cell contributions."""
    total = 0.0
    lo = x0
    # Walk source cells from floor(x0) to ceil(x1), wrapping mod n.
    s = int(np.floor(x0))
    while lo < x1 - 1e-15:
        s_lo = float(s)                              # left edge of source cell s
        s_hi = s_lo + 1.0
        seg_hi = min(x1, s_hi)
        xi0 = lo - s_lo; xi1 = seg_hi - s_lo
        total += integral(s % n, xi0, xi1)
        lo = seg_hi
        s += 1
    return total
```

- [ ] **Step 4: Run to verify it passes** → PASS.

- [ ] **Step 5: Add monotonicity + linear-profile tests**

```python
def test_ppm_remap_no_new_extrema():
    from gasgiant.sim.shallow_water_ref import ppm_remap_1d_periodic
    rng = np.random.default_rng(0)
    m = 1.0 + 0.5 * np.sin(np.linspace(0, 4 * np.pi, 96))
    edges = np.arange(97, dtype=float) - 0.6
    out = ppm_remap_1d_periodic(m, edges)
    assert out.min() >= m.min() - 1e-9              # no undershoot
    assert out.max() <= m.max() + 1e-9              # no overshoot
    assert abs(out.sum() - m.sum()) < 1e-10         # exact conservation

def test_ppm_remap_integer_shift_is_roll():
    from gasgiant.sim.shallow_water_ref import ppm_remap_1d_periodic
    m = np.arange(32, dtype=float) ** 1.0 + 1.0
    edges = np.arange(33, dtype=float) - 3.0        # exact 3-cell shift
    out = ppm_remap_1d_periodic(m, edges)
    assert np.allclose(out, np.roll(m, 3), atol=1e-9)
```

- [ ] **Step 6: Run all** → PASS. **Commit** `M2-adv: conservative 1-D PPM remap (SLICE kernel)`.

---

## Task 3: 2-D cascade conservative remap (`slice_remap_advance`)

**Files:**
- Modify: `src/gasgiant/sim/shallow_water_ref.py`
- Test: `tests/unit/test_m2_adv_ref.py`

The 2-D conservative advance of total `h` over `dt` by `(u, v)`: convert `h` to per-cell mass `m = h · cosφ` (the `cosφ a² dλ dφ` measure factors out except `cosφ`, which varies by row), remap zonally then meridionally using the departure edges from the trajectory, convert back to `h`. The cascade (1-D then 1-D) is what SLICE uses; it is exactly conservative because each 1-D pass conserves. This is the **drop-in replacement for `continuity_step_conservative`** on the fast-advection path.

- [ ] **Step 1: Write the failing test** (exact mass conservation of the full advance)

```python
def test_slice_advance_conserves_mass():
    from gasgiant.sim.shallow_water_ref import Grid, slice_remap_advance
    g = Grid(W=64, H=32, a=6.4e6)
    rng = np.random.default_rng(1)
    h = 1000.0 + 50.0 * rng.standard_normal((g.H, g.W))
    u = 40.0 * np.ones((g.H, g.W)); v = np.zeros((g.H + 1, g.W))
    dt = 1800.0                                     # large: zonal C ~ several at mid-lat
    h2 = slice_remap_advance(h, u, v, dt, g)
    m0 = float(np.sum(h * g.cos_c[:, None])); m1 = float(np.sum(h2 * g.cos_c[:, None]))
    assert abs(m1 - m0) / abs(m0) < 1e-12           # exact mass conservation
```

- [ ] **Step 2: Run to verify it fails** → FAIL.

- [ ] **Step 3: Implement** (append). Reuses `departure_points` (Task 1) and `ppm_remap_1d_periodic` (Task 2).

```python
def slice_remap_advance(h, u, v, dt, g):
    """Conservative semi-Lagrangian advance of total h over dt by (u,v).
    Drop-in, advective-CFL-free replacement for continuity_step_conservative.

    Mass m = h*cosφ per cell.  Cascade remap: (1) zonal 1-D conservative remap
    per row using the zonal departure edges, (2) meridional 1-D conservative
    remap per column using the meridional departure edges.  Convert back h = m/cosφ.
    """
    H, W = h.shape
    i_dep, j_dep = departure_points(u, v, dt, g, n_iter=2)
    cosc = g.cos_c[:, None]
    m = h * cosc                                     # per-cell mass (cosφ-weighted)

    # --- Zonal pass: per row, build destination edges from i_dep and remap. ---
    m_zon = np.empty_like(m)
    for j in range(H):
        centers = i_dep[j]                           # fractional source-i of arrival centers
        edges = np.empty(W + 1)
        edges[1:W] = 0.5 * (centers[0:W - 1] + centers[1:W])
        edges[0] = centers[0] - 0.5 * (centers[1] - centers[0])
        edges[W] = edges[0] + W                      # periodic wrap of the ring
        # ADVERSARIAL-REVIEW FIX (FATAL): under strong cross-cell shear the
        # midpoint edges can become NON-MONOTONIC (departure points cross), which
        # makes _accumulate_interval integrate negative-width / overlapping cells
        # -> silent mass double-count (~5% leak) + 3x overshoots.  Monotonizing the
        # destination edges collapses crossed cells to zero width (they were
        # unphysically over-resolved), keeping the partition valid and mass exactly
        # closed.  np.maximum.accumulate preserves the ring length (end-start == W).
        edges = np.maximum.accumulate(edges)
        m_zon[j] = ppm_remap_1d_periodic(m[j], edges)

    # --- Meridional pass: per column, remap with NON-periodic clamped edges. ---
    m_out = np.empty_like(m_zon)
    for i in range(W):
        centers = j_dep[:, i]                         # fractional source-j (center coord +0.5 space)
        m_out[:, i] = _remap_1d_meridional(m_zon[:, i], centers, H)

    return m_out / cosc

def _remap_1d_meridional(m_col, centers, H):
    """Conservative 1-D remap on a NON-periodic column (poles are walls; no mass
    flux across φ=±π/2).  Mirrors ppm_remap_1d_periodic but clamps edges to
    [0, H] and reflects the reconstruction at the boundary cells."""
    # Edges from centers (centers are in 'row-center+0.5' index space).
    c = centers - 0.5                                 # to row-index coordinate
    edges = np.empty(H + 1)
    edges[1:H] = 0.5 * (c[0:H - 1] + c[1:H])
    edges[0] = 0.0; edges[H] = float(H)               # wall: outermost edges fixed
    edges = np.clip(edges, 0.0, float(H))
    edges = np.maximum.accumulate(edges)              # FIX (FATAL): same crossing guard as the zonal pass
    return _ppm_remap_1d_clamped(m_col, edges)
```

(Implement `_ppm_remap_1d_clamped` as `ppm_remap_1d_periodic` with `np.roll` neighbours replaced by edge-clamped neighbours — first/last cells use one-sided edge values — and `_accumulate_interval` walking `s` without `% n`. Full code in Step 3b.)

- [ ] **Step 3b: Implement the clamped variant** (append; ~25 lines mirroring Task 2 with `min/max` index clamps instead of `% n`, edges already restricted to `[0,H]`).

```python
def _ppm_remap_1d_clamped(m, edges):
    n = len(m)
    mL = np.concatenate([m[:1], m[:-1]])              # clamp: cell -1 == cell 0
    mR = np.concatenate([m[1:], m[-1:]])
    mLL = np.concatenate([m[:1], m[:1], m[:-2]])
    aL_raw = (7.0 * (m + mL) - (mR + mLL)) / 12.0
    aL = _ppm_monotone_edge(aL_raw, mL, m)
    aR = np.concatenate([aL[1:], aL[-1:]])
    aL, aR = _ppm_limit_parabola(m, aL, aR)
    def integral(s, x0, x1):
        d = aR[s] - aL[s]; c6 = 6.0 * (m[s] - 0.5 * (aL[s] + aR[s]))
        F = lambda x: aL[s] * x + 0.5 * d * x * x + c6 * (0.5 * x * x - x ** 3 / 3.0)
        return F(x1) - F(x0)
    out = np.empty(n)
    for k in range(n):
        x0, x1 = edges[k], edges[k + 1]
        total = 0.0; lo = x0; s = min(int(np.floor(x0)), n - 1)
        while lo < x1 - 1e-15 and s < n:
            s_hi = s + 1.0; seg_hi = min(x1, s_hi)
            total += integral(s, lo - s, seg_hi - s)
            lo = seg_hi; s += 1
        out[k] = total
    return out
```

- [ ] **Step 4: Run to verify it passes** → PASS.

- [ ] **Step 5: Add a-scaling + monotonicity + meridional-wall mass tests**

```python
def test_slice_advance_a_scaling():
    from gasgiant.sim.shallow_water_ref import Grid, slice_remap_advance
    for a in (1.0, 6.4e6):
        g = Grid(W=48, H=24, a=a)
        h = 800.0 + np.zeros((g.H, g.W)); h[10, 12] = 900.0
        u = np.full((g.H, g.W), 25.0 * (a / 6.4e6)); v = np.zeros((g.H + 1, g.W))
        h2 = slice_remap_advance(h, u, v, 900.0, g)
        m0 = np.sum(h * g.cos_c[:, None]); m1 = np.sum(h2 * g.cos_c[:, None])
        assert abs(m1 - m0) / m0 < 1e-12

def test_slice_advance_meridional_wall_conserves():
    """Pure meridional flow must not leak mass across the poles."""
    from gasgiant.sim.shallow_water_ref import Grid, slice_remap_advance
    g = Grid(W=16, H=40, a=6.4e6)
    h = 500.0 + np.zeros((g.H, g.W)); h[20] = 600.0
    u = np.zeros((g.H, g.W)); v = np.full((g.H + 1, g.W), 8.0); v[0] = 0.0; v[-1] = 0.0
    h2 = slice_remap_advance(h, u, v, 1200.0, g)
    assert abs(np.sum(h2 * g.cos_c[:, None]) - np.sum(h * g.cos_c[:, None])) / np.sum(h * g.cos_c[:, None]) < 1e-12

def test_slice_advance_strong_shear_conserves_and_no_overshoot():
    """ADVERSARIAL-REVIEW REGRESSION (FATAL class): a STRONGLY SHEARED row whose
    cross-cell Courant gradient > 1 makes departure points cross.  Without the
    np.maximum.accumulate edge-monotonization this leaks ~5% mass and overshoots
    3x.  The uniform-velocity tests above CANNOT catch this; this one must."""
    from gasgiant.sim.shallow_water_ref import Grid, slice_remap_advance
    g = Grid(W=64, H=8, a=6.4e6)
    # A sharp zonal jet flank: u varies by >1 Courant-unit per cell.
    lam = np.arange(g.W) * g.dlam
    u = (150.0 * (1.0 + np.tanh(8.0 * np.sin(lam))))[None, :] * np.ones((g.H, 1))
    v = np.zeros((g.H + 1, g.W))
    h = 1000.0 + 100.0 * np.sin(4.0 * lam)[None, :] * np.ones((g.H, 1))
    h2 = slice_remap_advance(h, u, v, 6000.0, g)            # dt large -> crossing edges
    m0 = np.sum(h * g.cos_c[:, None]); m1 = np.sum(h2 * g.cos_c[:, None])
    assert abs(m1 - m0) / abs(m0) < 1e-10, "shear-crossing mass leak (edge monotonization missing)"
    assert h2.max() <= h.max() + 1e-6, "shear-crossing overshoot (limiter/edge bug)"
    assert h2.min() >= h.min() - 1e-6, "shear-crossing undershoot"
```

- [ ] **Step 6: Run all** → PASS (the shear test FAILS without the `np.maximum.accumulate` lines — verify by temporarily removing them). **Commit** `M2-adv: 2-D cascade conservative remap (slice_remap_advance)`.

---

## Task 4: Semi-Lagrangian momentum interpolation

**Files:**
- Modify: `src/gasgiant/sim/shallow_water_ref.py`
- Test: `tests/unit/test_m2_adv_ref.py`

Replace the predictor's Eulerian vorticity-flux advection (advective-CFL site #1) with SL departure-point interpolation of the velocity components. We keep the **KE-gradient and `(1-θ)` pressure** terms of `_semi_implicit_predictor` exactly (those are not advection and carry no advective CFL); only the parcel transport of `u, v` becomes SL. Interpolation is bicubic (Catmull-Rom) on the face fields, evaluated at the face departure points.

- [ ] **Step 1: Write the failing test** (solid-body zonal flow: SL transport of a velocity bump shifts it west by C cells, undistorted to interpolation order)

```python
def test_sl_momentum_advects_bump_zonally():
    from gasgiant.sim.shallow_water_ref import Grid, sl_advect_velocity
    g = Grid(W=128, H=4, a=6.4e6)
    j0 = 2; C = 3.0                                   # integer shift => exact bicubic
    cosphi = g.cos_c[j0]
    u_adv = C * g.a * cosphi * g.dlam / 600.0         # u giving C-cell shift in dt=600
    u = np.full((g.H, g.W), u_adv)
    v = np.zeros((g.H + 1, g.W))
    q = np.zeros((g.H, g.W)); q[j0, 40:48] = 1.0      # a zonal velocity feature on row j0
    q_adv = sl_advect_velocity(q, u, v, 600.0, g, kind="u")
    assert np.allclose(q_adv[j0], np.roll(q[j0], 3), atol=1e-9)  # shifted west by 3 cells
```

- [ ] **Step 2: Run to verify it fails** → FAIL.

- [ ] **Step 3: Implement** (append). Reuses `departure_points`; adds a bicubic sampler.

```python
def sl_advect_velocity(q, u, v, dt, g, kind):
    """Semi-Lagrangian transport of a face field q by (u,v) over dt.
    kind="u": q at u-faces (H,W); kind="v": q at v-faces (H+1,W) with pole rows 0.
    Bicubic (Catmull-Rom) interpolation at the departure points.  Non-conservative
    (momentum need not conserve exactly; only mass does — gated separately)."""
    H, W = u.shape
    i_dep, j_dep = departure_points(u, v, dt, g, n_iter=2)
    if kind == "u":
        return _bicubic_periodic(q, i_dep, j_dep, g)         # (H,W)
    # v-faces: build face-located departure points by averaging adjacent center rows.
    i_dep_vf = np.zeros((H + 1, W)); j_dep_vf = np.zeros((H + 1, W))
    i_dep_vf[1:H] = 0.5 * (i_dep[0:H - 1] + i_dep[1:H])
    j_dep_vf[1:H] = 0.5 * (j_dep[0:H - 1] + j_dep[1:H]) - 0.5  # to v-face row coordinate
    out = _bicubic_periodic_vface(q, i_dep_vf, j_dep_vf, g)
    out[0] = 0.0; out[H] = 0.0                                # poles stay zero
    return out

def _catmull_rom_w(t):
    """4-point Catmull-Rom weights for fractional offset t in [0,1)."""
    t2 = t * t; t3 = t2 * t
    return (np.stack([
        -0.5 * t3 + t2 - 0.5 * t,
        1.5 * t3 - 2.5 * t2 + 1.0,
        -1.5 * t3 + 2.0 * t2 + 0.5 * t,
        0.5 * t3 - 0.5 * t2], axis=0))                       # shape (4, ...)

def _bicubic_periodic(field, i_idx, j_idx, g):
    """Bicubic sample of a center-row field (H,W) at (i_idx zonal, j_idx center
    coord).  Zonal periodic (branch wrap); meridional clamped at the poles."""
    H, W = field.shape
    jj = np.clip(j_idx - 0.5, 0.0, H - 1.0)
    j0 = np.floor(jj).astype(int); fy = jj - j0
    i0 = np.floor(i_idx).astype(int); fx = i_idx - i0
    wx = _catmull_rom_w(fx); wy = _catmull_rom_w(fy)
    acc = np.zeros_like(i_idx)
    for dj in range(-1, 3):
        jr = np.clip(j0 + dj, 0, H - 1)
        row = np.zeros_like(i_idx)
        for di in range(-1, 3):
            ic = (i0 + di) % W                               # periodic
            row = row + wx[di + 1] * field[jr, ic]
        acc = acc + wy[dj + 1] * row
    return acc
```

(`_bicubic_periodic_vface` is the same with `H+1` rows and the pole rows clamped — full code in Step 3b.)

- [ ] **Step 3b: Implement `_bicubic_periodic_vface`** (append; identical structure, `H+1` rows, `np.clip(j0+dj, 0, H)`).

- [ ] **Step 4: Run to verify it passes** → PASS.

- [ ] **Step 5: Add the momentum-predictor wrapper + its test**

```python
def sl_momentum_predictor(h, u, v, gp, g, dt, theta):
    """SL replacement for _semi_implicit_predictor: SL parcel transport of (u,v)
    PLUS the unchanged KE-gradient and (1-theta) explicit pressure half, no Coriolis.
    At small dt this matches _semi_implicit_predictor to advection-scheme order
    (the bump test below pins the SL transport; this wrapper pins the forcing)."""
    H, W = h.shape
    u_sl = sl_advect_velocity(u, u, v, dt, g, kind="u")
    v_sl = sl_advect_velocity(v, u, v, dt, g, kind="v")
    v_c = 0.5 * (v[0:H] + v[1:H + 1])
    ke = 0.5 * (u * u + v_c * v_c)
    gxk, gyk = grad_faces(ke, g)
    gxn, gyn = grad_faces(h, g)
    c = 1.0 - theta
    u_star = u_sl - dt * (gxk + c * gp * gxn)
    v_star = v.copy() * 0.0
    v_star[1:H] = v_sl[1:H] - dt * (gyk[1:H] + c * gp * gyn[1:H])
    return u_star, v_star

def test_sl_momentum_predictor_resting_layer_is_pressure_only():
    """Flat resting layer (u=v=0, h uniform): predictor returns zero (no advection,
    no pressure gradient)."""
    from gasgiant.sim.shallow_water_ref import Grid, sl_momentum_predictor
    g = Grid(W=16, H=8, a=6.4e6)
    h = np.full((g.H, g.W), 1000.0); u = np.zeros((g.H, g.W)); v = np.zeros((g.H + 1, g.W))
    us, vs = sl_momentum_predictor(h, u, v, 9.8, g, 600.0, 0.5)
    assert np.allclose(us, 0.0, atol=1e-12) and np.allclose(vs, 0.0, atol=1e-12)
```

- [ ] **Step 6: Run all** → PASS. **Commit** `M2-adv: SL momentum transport + predictor`.

---

## Task 5: `step_slsi` assembly + crux accuracy gate (GO/NO-GO)

**Files:**
- Modify: `src/gasgiant/sim/shallow_water_ref.py`
- Test: `tests/spikes/test_slsi_fastjet_spike.py` (create), `tests/unit/test_m2_adv_ref.py`

Assemble the full step by swapping the two SL operators into the M2-core structure; **everything else (Helmholtz, Picard, backsub) is reused verbatim**. Then the front-loaded crux gate: a fast polar jet advanced at advective `C ≫ 1` must match a fine-dt reference to a stated L2 tol (accuracy, not just finiteness). **If this fails, the combined approach is falsified before any GPU work** — stop and re-brainstorm.

- [ ] **Step 1: Write `step_slsi` and a small-dt consistency test** (at small dt, SLSI ≈ M2-core)

```python
def test_slsi_matches_m2core_at_small_dt():
    """At dt well below the advective CFL, step_slsi tracks step_semi_implicit:
    both reduce to the same physics; SL vs FCT transport differ only at O(dt^2)."""
    import numpy as np
    from gasgiant.sim.shallow_water_ref import williamson2_state, step_slsi, step_semi_implicit
    # NOTE (API): williamson2_state(W, H, a, omega, u0, gp, h0, h_floor=0.05) — omega,
    # gp, h0 are MANDATORY (shallow_water_ref.py:929).  Mirror test_sw_conservation.py.
    st = williamson2_state(W=64, H=32, a=6.4e6, omega=7.292e-5, u0=20.0, gp=9.8, h0=8000.0)
    a = step_semi_implicit(st, theta=0.5, picard_iters=3, poisson_iters=200)
    b = step_slsi(st, theta=0.5, picard_iters=3, poisson_iters=200)
    # Velocities (geostrophically balanced, transport-light) agree tightly.
    assert np.max(np.abs(a.u - b.u)) < 5e-4
    assert np.max(np.abs(a.v - b.v)) < 5e-4
    # Heights agree to the transport-scheme difference at this dt.
    assert np.max(np.abs(a.h - b.h)) < 5e-2
```

- [ ] **Step 2: Run to verify it fails** → FAIL (`step_slsi` undefined).

- [ ] **Step 3: Implement `step_slsi`** (append; mirrors `step_semi_implicit:1186-1240` with two substitutions, marked `# SL:`)

```python
def step_slsi(st, theta=0.5, picard_iters=3, poisson_iters=200,
              sor_omega=1.7, dh_warm=None):
    """Semi-Lagrangian semi-implicit step: M2-core's SI core with the two explicit
    Eulerian transport operators replaced by SL equivalents (advective-CFL-free).

    Substitutions vs step_semi_implicit:
      site #1  _semi_implicit_predictor -> sl_momentum_predictor  (SL momentum)
      site #2  continuity_step_conservative -> slice_remap_advance (SLICE remap)
    The Helmholtz solve (dh), Picard-Coriolis, and velocity_backsub are reused
    verbatim, so the gravity-wave CFL removal (M2-core) is preserved and the
    advective CFL is now also removed.
    """
    g, gp, omega, dt = st.g, st.gp, st.omega, st.dt
    h, u, v = st.h, st.u, st.v
    H, W = h.shape

    # 1. SL: predictor with SL momentum transport (no Coriolis).
    u_star, v_star = sl_momentum_predictor(h, u, v, gp, g, dt, theta)

    # 2-4. Reuse M2-core verbatim: H_ref, Picard Helmholtz dh, back-substitution.
    H_ref_lat = reference_depth(h)
    dh = np.zeros((H, W)) if dh_warm is None else dh_warm.copy()
    for _ in range(picard_iters):
        rhs = helmholtz_rhs(h, u, v, u_star, v_star, dh,
                            H_ref_lat, gp, omega, theta, dt, g)
        dh = helmholtz_sor(rhs, H_ref_lat, gp, theta, dt, g,
                           poisson_iters, sor_omega, dh0=dh)
    u_new, v_new = velocity_backsub(u_star, v_star, h + dh, gp, theta, dt, omega, g)

    # 5. SL: nonlinear anomaly via SLICE conservative remap (replaces FCT).
    #    ADVERSARIAL-REVIEW FIX (FATAL): the reference part must be removed in the
    #    SAME conservative-remap form as h_sl (NOT the Eulerian divergence_helmholtz),
    #    or it fails to cancel at Courant>>1 and double-counts against dh.  Remap the
    #    broadcast H_ref by the same trajectory and subtract; both remaps conserve,
    #    so anomaly is mass-neutral and Courant-consistent.  See the Background note.
    H_ref_field = np.broadcast_to(H_ref_lat[:, None], h.shape)
    h_sl = slice_remap_advance(h, u_new, v_new, dt, g)            # was continuity_step_conservative
    href_sl = slice_remap_advance(H_ref_field, u_new, v_new, dt, g)
    anomaly = (h_sl - h) - (href_sl - H_ref_field)               # -dt div((h-H_ref) u), finite-Courant
    h_raw = h + dh + anomaly
    assert_positivity(h_raw, st.h_floor)
    h_new = np.maximum(h_raw, st.h_floor)

    return SwRefState(g=g, gp=gp, h=h_new, u=u_new, v=v_new,
                      dt=dt, omega=omega,
                      u_init=st.u_init, v_init=st.v_init, h_floor=st.h_floor)
```

- [ ] **Step 4: Run the small-dt test** → PASS (tune the three tolerances to the observed transport-scheme delta if needed; they bound SL−FCT at this dt, not correctness).

- [ ] **Step 5a: Add `fast_jet_state`** to `shallow_water_ref.py`.

ADVERSARIAL-REVIEW FIX (FATAL — vacuous gate): the naive version (polar jet, dt from the gravity-wave/polar CFL, zonally symmetric) gave advective C≈0.077 (NOT ≫1) *and* had ∂/∂λ≡0 so SL zonal advection was identity — it tested nothing (the M2-core stationarity trap again). Two corrections: (1) **dt is set by the ADVECTIVE limit AT THE JET** (`dt = dt_mult · C_base · a·cosφ_jet·dλ / u0`), so `dt_mult=8` genuinely gives advective C≈8 at the jet; (2) **add a balanced zonal wavenumber-`m` perturbation** so zonal advection actually transports a propagating feature.

```python
def fast_jet_state(W=128, H=64, a=6.4e6, u0=80.0, dt_mult=1, C_base=0.5, m_wave=4, amp=0.04):
    """Fast mid-latitude zonal jet with a wavenumber-m height perturbation, for
    the M2-adv crux gate.  The jet (φ_jet=45°) advects the perturbation zonally at
    ~u0; the SL scheme must track that translation at large advective Courant.

    dt is set by the ADVECTIVE Courant limit AT THE JET (not the polar gravity-wave
    CFL, which M2-core already removed): dt = dt_mult * C_base * a*cosφ_jet*dλ / u0,
    so the advective Courant at the jet is C_base*dt_mult (=4 at dt_mult=8, C_base=0.5).
    """
    g = Grid(W=W, H=H, a=a)
    gp = 9.8; omega = 7.292e-5; H0 = 8000.0
    phi = g.phi_c; lam = np.arange(W) * g.dlam
    phi_jet = np.deg2rad(45.0); sigma = np.deg2rad(10.0)
    u_prof = u0 * np.exp(-((phi - phi_jet) / sigma) ** 2)             # (H,)
    u = np.repeat(u_prof[:, None], W, axis=1)
    # Geostrophic mean height: integrate dh/dφ = -(a/g') f u from south pole up.
    f = 2.0 * omega * np.sin(phi)
    h_prof = H0 + np.cumsum((-(a / gp) * f * u_prof * g.dphi)[::-1])[::-1]
    h = np.repeat(h_prof[:, None], W, axis=1)
    # Zonal wavenumber-m perturbation localized on the jet (gives ∂/∂λ != 0 so the
    # zonal SL remap is genuinely exercised).  Small amplitude -> stays positive.
    env = np.exp(-((phi - phi_jet) / sigma) ** 2)[:, None]
    h = h + amp * H0 * env * np.cos(m_wave * lam)[None, :]
    v = np.zeros((H + 1, W))
    cphi_jet = np.cos(phi_jet)
    dt = dt_mult * C_base * (a * cphi_jet * g.dlam) / u0
    return SwRefState(g=g, gp=gp, h=h, u=u, v=v, dt=dt, omega=omega,
                      u_init=u.copy(), v_init=v.copy(), h_floor=1.0)
```

- [ ] **Step 5b: Write the CRUX accuracy gate** — with a self-convergence-calibrated tol and a no-op falsification check.

```python
# tests/spikes/test_slsi_fastjet_spike.py
"""M2-adv crux gate: SLSI must advect a fast jet's zonal perturbation at advective
C~4 with ACCURACY (not just boundedness) matching a fine-dt reference, AND conserve
mass.  Falsifies the SLSI+SLICE approach early if it fails."""
import numpy as np
from gasgiant.sim.shallow_water_ref import (
    fast_jet_state, step_slsi, step_semi_implicit, total_mass)

def _run(step_fn, dt_mult, n_big):
    st = fast_jet_state(dt_mult=dt_mult)
    dh = None
    for _ in range(n_big):
        st = step_fn(st, theta=0.5, picard_iters=3, poisson_iters=300, dh_warm=dh)
    return st

def test_crux_setup_is_nonvacuous():
    """Guard against the M2-core stationarity trap: the gate must actually exercise
    the large-advective-Courant regime AND carry zonal structure."""
    st = fast_jet_state(dt_mult=8)
    # advective Courant at the jet must be >> 1 (target ~4).
    u_c = 0.5 * (st.u + np.roll(st.u, 1, axis=1))
    C = np.max(np.abs(u_c) * st.dt / (st.g.a * st.g.cos_c[:, None] * st.g.dlam))
    assert C > 3.0, f"crux gate vacuous: max advective C={C:.2f} (need >>1)"
    # zonal structure must be present (else SL zonal advection is identity).
    assert np.max(np.abs(st.h - st.h.mean(axis=1, keepdims=True))) > 1.0

def test_reference_is_self_converged():
    """Calibration: the fine-dt reference must be converged, else the accuracy gate
    could fail on reference drift.  L2 between dt_mult=1 and dt_mult=2 must be small
    relative to the 0.05 gate tol (if not, drop the reference dt_mult)."""
    ref1 = _run(step_slsi, dt_mult=1, n_big=160)
    ref2 = _run(step_slsi, dt_mult=2, n_big=80)        # same physical time, coarser
    drift = float(np.sqrt(np.mean((ref2.h - ref1.h) ** 2)) / np.sqrt(np.mean(ref1.h ** 2)))
    assert drift < 0.02, f"reference not converged (self-L2={drift:.4f}); gate tol unsafe"

def test_slsi_fastjet_accuracy_at_large_courant():
    ref = _run(step_slsi, dt_mult=1, n_big=160)        # converged fine-dt baseline
    big = _run(step_slsi, dt_mult=8, n_big=20)         # same physical time (8*20 == 1*160)
    assert np.isfinite(big.h).all() and big.h.min() > 0.0          # (a) bounded & positive
    def l2(a, b): return float(np.sqrt(np.mean((a - b) ** 2)) / np.sqrt(np.mean(b ** 2)))
    err = l2(big.h, ref.h)
    print(f"\n[slsi-spike] 8x-dt vs fine-dt relative L2(h) = {err:.4f}")
    assert err < 0.05, f"SLSI accuracy gate FAILED at 8x dt: L2={err:.4f} (approach falsified)"
    assert abs(total_mass(big) - total_mass(ref)) / abs(total_mass(ref)) < 1e-6  # (c) mass

def test_eulerian_path_fails_the_same_gate():
    """NO-OP-TEST discipline (this project's hard-won lesson): prove the gate CAN
    fail.  The unmodified M2-core Eulerian step (step_semi_implicit, explicit FCT
    advection) must blow up or badly miss at dt_mult=8 — otherwise the gate isn't
    isolating advective-CFL removal."""
    try:
        big_eul = _run(step_semi_implicit, dt_mult=8, n_big=20)
        ref = _run(step_slsi, dt_mult=1, n_big=160)
        err = float(np.sqrt(np.mean((big_eul.h - ref.h) ** 2)) / np.sqrt(np.mean(ref.h ** 2)))
        assert err > 0.05 or not np.isfinite(big_eul.h).all(), (
            f"gate not discriminating: Eulerian path also passes (L2={err:.4f})")
    except (ValueError, FloatingPointError):
        pass   # Eulerian FCT blew up / tripped the positivity guard at C>1 — expected.
```

- [ ] **Step 5c: Write the balanced-anomaly coupling test** (catches the FATAL coupling bug that mass conservation cannot — added at the first reviewer's request).

```python
# in tests/unit/test_m2_adv_ref.py
def test_slsi_steady_anomaly_no_spurious_tendency():
    """A state whose ANOMALY (h - H_ref) is ~steady but with u != 0 at large Courant:
    the correct anomaly tendency is ~0, so step_slsi must leave h ~ h + dh.  The
    buggy Eulerian-h_linref coupling shows an O(1) spurious height tendency here
    that mass conservation hides."""
    import numpy as np
    from gasgiant.sim.shallow_water_ref import fast_jet_state, step_slsi, reference_depth
    st = fast_jet_state(dt_mult=6, amp=0.0)            # balanced jet, NO perturbation
    H_ref = reference_depth(st.h)[:, None]
    anom0 = st.h - H_ref
    out = step_slsi(st, theta=0.5, picard_iters=3, poisson_iters=300)
    anom1 = out.h - reference_depth(out.h)[:, None]
    # Anomaly should barely change (balanced, zonally symmetric -> no zonal transport).
    drift = np.max(np.abs(anom1 - anom0)) / (np.max(np.abs(anom0)) + 1e-9)
    assert drift < 1e-2, f"spurious anomaly tendency {drift:.4f} (coupling double-count?)"
```

- [ ] **Step 6: Run the crux gate** — `pytest tests/spikes/test_slsi_fastjet_spike.py tests/unit/test_m2_adv_ref.py::test_slsi_steady_anomaly_no_spurious_tendency -v -s`. First confirm `test_crux_setup_is_nonvacuous` and `test_eulerian_path_fails_the_same_gate` PASS (the gate is real and can fail), then the accuracy gate:
  - **PASS** → the SLSI approach is validated; proceed to Task 6.
  - **FAIL** (L2 ≥ 0.05, or the coupling test trips) → **STOP**. If the coupling test fails, the `step_slsi` step-5 composition is still wrong — revisit before declaring the approach falsified. If only the accuracy gate fails with the coupling test passing, the combined SLSI+SLICE approach is genuinely falsified at large Courant: record the L2 in `docs/superpowers/specs/m2-adv-verdict.md`, do NOT build GPU kernels, and re-enter brainstorming with the spec §5 fallbacks (reduced polar grid keeping explicit FCT; or non-conservative SL + global mass fixer). Treat a failure as data.

- [ ] **Step 7: Commit** `M2-adv: step_slsi assembly + non-vacuous crux fast-jet accuracy gate + coupling test`.

---

## Task 6: Dual-path byte-identity (`fast_advection=False` ≡ M2-core ≡ M1)

**Files:**
- Modify: `src/gasgiant/sim/sw_gpu.py`
- Test: `tests/unit/test_dual_path_adv.py` (create)

`SwGpuSolver` gains a `fast_advection: bool = False` flag and trajectory params (`sl_iters: int = 2`). When `False`, the step dispatch is the M2-core `_step_semi_implicit` unchanged → byte-identical checkpoints; when that solver also has `semi_implicit=False`, it is byte-identical to M1. No SL-only construction may touch shared `__init__` state.

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/test_dual_path_adv.py
import numpy as np
import hashlib
import numpy as np
from gasgiant.sim import sw_gpu

# NOTE (API): use the from_williamson2 classmethod (the canonical GPU entry point,
# used by every existing GPU test) — SwGpuSolver.__init__ requires gp/omega/dt
# positionally.  Compare state via np.array_equal + SHA1, NOT raw .npz bytes (zip
# metadata/ordering is not guaranteed byte-stable).  Mirror test_dual_path.py.
_IC = dict(W=64, H=32, a=1.0, omega=2.0, u0=0.2, gp=1.0, h0=5.0)

def _sha1_state(h, u, v):
    m = hashlib.sha1()
    for f in (h, u, v):
        m.update(np.ascontiguousarray(f, dtype=np.float32).tobytes())
    return m.hexdigest()

def test_fast_advection_false_byte_identical_to_m2core(gpu):
    """fast_advection=False reproduces the M2-core SI state byte-for-byte."""
    def run(**kw):
        sg = sw_gpu.SwGpuSolver.from_williamson2(gpu, **_IC, semi_implicit=True, theta=0.5, **kw)
        for _ in range(5):
            sg.step()
        return sg.download_state()
    h_b, u_b, v_b = run()                          # M2-core path (no flag)
    h_a, u_a, v_a = run(fast_advection=False)      # M2-adv flag, off
    assert np.array_equal(h_a, h_b) and np.array_equal(u_a, u_b) and np.array_equal(v_a, v_b)
    assert _sha1_state(h_a, u_a, v_a) == _sha1_state(h_b, u_b, v_b)

def test_no_sl_state_in_shared_init(gpu):
    """SL-only buffers must not exist when fast_advection=False."""
    s = sw_gpu.SwGpuSolver.from_williamson2(gpu, **_IC, semi_implicit=True, fast_advection=False)
    assert not getattr(s, "_has_sl_buffers", False)
```

- [ ] **Step 2: Run to verify it fails** → FAIL (`fast_advection` kwarg unknown).

- [ ] **Step 3: Implement** the flag + guarded dispatch. Add `fast_advection: bool = False` and `sl_iters: int = 2` to BOTH `SwGpuSolver.__init__` AND the `from_williamson2` classmethod (it must forward the new kwargs — the M2-core test pattern always constructs via `from_williamson2`). In `step()`: `if self.semi_implicit and self.fast_advection: self._step_slsi() elif self.semi_implicit: self._step_semi_implicit() else: <M1 explicit>`. Allocate SL buffers (`_has_sl_buffers=True`) only inside the `fast_advection` branch's lazy init, never in shared `__init__`.

- [ ] **Step 4: Run to verify it passes** → PASS.

- [ ] **Step 5: Commit** `M2-adv: dual-path flag (fast_advection=False byte-identical to M2-core)`.

---

## Task 7: GPU kernels + per-field diffs

**Files:**
- Create: `kernels/sw_departure.comp`, `kernels/sw_slice_remap.comp`, `kernels/sw_sl_momentum.comp`, `kernels/sw_sl_common.glsl`
- Modify: `src/gasgiant/sim/sw_gpu.py`
- Test: `tests/unit/test_m2_adv_gpu.py` (create)

Each kernel diffs against its CPU-ref counterpart at `atol=2e-5` on pre-division quantities, with a-scaling. Reuse `cosCenter`, `cosVface`, `sinAcc`, `wrapX` (branch form) from `sw_common.glsl`; reuse the M2-core Helmholtz kernels verbatim. The SLICE remap is the hardest kernel — the 1-D PPM reconstruction + interval accumulation port directly from Task 2/3; the cascade is two passes with a `memory_barrier()` between (zonal then meridional), each pass writing to a scratch R32F texture. Departure trace ports `departure_points` (2 fixed midpoint iterations). The cubic momentum sampler ports `_bicubic_periodic`.

- [ ] **Step 1: Failing per-kernel tests** — `test_departure_gpu_matches_cpu`, `test_slice_remap_gpu_matches_cpu` (uniform + random IC; assert exact-mass on GPU too via a reduction), `test_sl_momentum_gpu_matches_cpu`, each at `atol=2e-5` + an `a=1` vs `a=6.4e6` pair.

- [ ] **Step 2: Implement `sw_sl_common.glsl`** — `vec2 trace_departure(ivec2 cell)` (2 midpoint iters, branch `wrapX`); `float ppm_reconstruct_integral(...)`; `vec4 catmull_rom(float t)`.

- [ ] **Step 3: Implement `sw_departure.comp`** — writes `(i_dep, j_dep)` into an RG32F texture; cell-center velocity averages of faces; `1/(a cosφ)` metric; deterministic fixed 2 iters.

- [ ] **Step 4: Implement `sw_slice_remap.comp`** — two-pass cascade. Pass 0 (zonal): per row, monotone-limited PPM edges, integrate over destination interval from the departure edges (periodic `wrapX`), write scratch. Pass 1 (meridional): same on columns with clamped pole edges. Mass weight `cosφ` applied on input, removed on output. `memory_barrier()` + dispatch boundary between passes.

- [ ] **Step 5: Implement `sw_sl_momentum.comp`** — bicubic Catmull-Rom sample of `u` (and `v` at v-faces, pole rows forced 0) at the departure points.

- [ ] **Step 6: Wire `_step_slsi` in `sw_gpu.py`** — dispatch order: departure → sl_momentum (predictor forcing reuses the existing KE/pressure kernels) → [reuse M2-core Helmholtz Picard/SOR/backsub kernels] → slice_remap → assemble `h_new`. Checkpoint `version=3` (adds `fast_advection`, `sl_iters`).

- [ ] **Step 7: Run per-kernel diffs** → PASS at `atol=2e-5`; GPU exact-mass reduction `< 1e-6` rel.

- [ ] **Step 8: Commit** `M2-adv: GPU SL kernels (departure, SLICE remap, cubic momentum) + per-field diffs`.

---

## Task 8: GPU full-step integration, headline verdict, validation script

**Files:**
- Test: `tests/unit/test_m2_adv_gpu.py`, `tests/unit/test_m2_adv_conservation.py` (create)
- Create: `scripts/sw_m2_adv_validation.py`, `docs/superpowers/specs/m2-adv-verdict.md`

- [ ] **Step 1: Failing tests** — full-step GPU↔CPU (`step_slsi`) at 1-step and N-step, `atol=2e-5`; GPU fast-jet headline accuracy gate (mirror Task 5 spike on GPU, L2 < 0.05 at 8× dt); determinism (byte-identical SHA1 of `(h,u,v,dh-warmstart)` over a fixed warm-started run); long-run mass `rtol < 1e-6`, potential enstrophy bounded `< 1e-2`.

- [ ] **Step 2: Implement** any kernel fixes the integration surfaces (none expected if Tasks 1–7 are correct; these gates lock it in).

- [ ] **Step 3: Write `scripts/sw_m2_adv_validation.py`** — one consolidated PASS/FAIL across: per-kernel diffs, full-step diff, headline accuracy factor, dual-path byte-identity, determinism, conservation. Exits non-zero on any failure (mirror `scripts/sw_m2_validation.py`).

- [ ] **Step 4: Write `docs/superpowers/specs/m2-adv-verdict.md`** — gate numbers + the **honest headline**: "advective CFL removed; realized fast-jet factor = largest dt within the L2 accuracy tol ÷ explicit dt = <measured>×; gravity-wave CFL already removed by M2-core; the only remaining step limit is accuracy, not stability." Confirm `shallow_water_ref.py` remains the M2-AE / M3 ground truth.

- [ ] **Step 5: Commit** `M2-adv: GPU integration + headline verdict + validation script`.

---

## After all tasks: final whole-implementation review (Opus), then `superpowers:finishing-a-development-branch`.

---

## Validation summary (gates a–g)

- **(a)** Per-field GPU↔CPU at `atol=2e-5` for `sw_departure`, `sw_slice_remap`, `sw_sl_momentum`, and full `step_slsi` (1-step + N-step), each pre-division, with a-scaling. *(Tasks 7, 8)*
- **(b)** Conservative remap exact mass to round-off + monotonicity on adversarial IC; meridional pole-wall no-leak. *(Tasks 2, 3)*
- **(c)** Trajectory convergence order vs analytic solid-body rotation. *(Task 1)*
- **(d)** Crux fast-jet **accuracy** gate at advective `C ≫ 1`: relative L2(h) `< 0.05` vs fine-dt reference (CPU Task 5, GPU Task 8) — the headline, an accuracy gate not a stability claim.
- **(e)** Nested dual-path byte-identity: `fast_advection=False` ≡ M2-core ≡ M1 (P0.5 hash). *(Task 6)*
- **(f)** Determinism: byte-identical SHA1 over a fixed warm-started multi-step run. *(Task 8)*
- **(g)** Long-run conservation: `total_mass` rtol `< 1e-6` (un-renormalized); potential enstrophy bounded `< 1e-2`. *(Task 8)*

## Risks (resolved by the adversarial plan review — fixes folded in)

1. **Coupling double-counts the reference divergence (was FATAL — FIXED).** The original `anomaly = h_sl − h_linref` subtracted an Eulerian first-order reference divergence from a finite-Courant Lagrangian SL height; they fail to cancel at C ≫ 1, double-counting against the implicit `dh`, and mass conservation cannot detect it. **Fixed** by removing the reference part in the same conservative-remap form: `anomaly = (h_sl − h) − (href_sl − H_ref_field)` (Task 5 Step 3 + Background). Guarded by the balanced-anomaly coupling test (Task 5 Step 5c), NOT the mass gate.
2. **SLICE leaks under shear (was FATAL — FIXED).** Non-monotonic departure points (strong cross-cell shear, the fast-jet regime) make destination edges cross → silent mass double-count + overshoots. The uniform-velocity conservation tests cannot catch it. **Fixed** by `np.maximum.accumulate(edges)` in both cascade passes (Tasks 3) + a sheared-departure regression test (`test_slice_advance_strong_shear_conserves_and_no_overshoot`).
3. **Crux gate was vacuous (was FATAL — FIXED).** The original `fast_jet_state` gave advective C ≈ 0.077 (dt throttled by the polar cell while the jet sat at 70°) AND was zonally symmetric (SL zonal advection = identity) — the M2-core stationarity trap. **Fixed** by basing dt on the advective limit at the jet and adding a zonal wavenumber-m perturbation, plus `test_crux_setup_is_nonvacuous` (asserts C > 3 and zonal structure) and `test_eulerian_path_fails_the_same_gate` (no-op-test: proves the gate can fail).
4. **API grounding (was CRITICAL — FIXED).** `williamson2_state` and `SwGpuSolver` mandatory-arg signatures corrected; GPU tests use the `from_williamson2` classmethod; dual-path compares via `np.array_equal`/SHA1, not `.npz` bytes.
5. **SLICE cascade splitting error (open, monitored).** The zonal-then-meridional cascade is dimensionally split; at very large meridional Courant the split may degrade accuracy. The crux gate (d) measures it; if it fails on the meridional component, a Strang (half-zonal / full-meridional / half-zonal) cascade is the first fallback.
6. **Momentum SL non-conservation (acceptable, gated).** Only mass must conserve exactly; PV/enstrophy drift is gated (g); if it grows, a PV-conserving SL is the fallback.
7. **Realized factor may be accuracy-bound below the naive `c_gw/|u|`** — reported honestly as an accuracy factor in the verdict, never a bare stability number.
