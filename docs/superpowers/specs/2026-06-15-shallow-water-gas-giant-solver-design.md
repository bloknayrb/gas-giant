# Shallow-Water Gas-Giant Solver — Architecture Design

**Date:** 2026-06-15
**Status:** Revised after adversarial review (rev 2)
**Author:** brainstormed with the user
**Supersedes nothing.** Adds `solver.type = "shallow_water"` alongside the existing
`kinematic` (v1.5) and `vorticity` (v1.6) solvers, both of which remain byte-identical.

> **Revision note (rev 2).** This draft was reviewed by five hostile reviewers
> (numerics, physics, code-integration, conservation, scope), each finding
> independently verified — 33 findings upheld (0 fatal, 13 major, 20 minor). The
> changes from rev 1 are summarized in §11. The headline changes: a de-risking
> **Milestone 0 image spike** now precedes the build; the grid is **C-grid /
> Rhie-Chow, decided before M1** (not "start collocated"); **flux-form FV +
> FCT** for `h` is mandatory; the Sadourny-conservation justification is
> corrected; §6 validation is split and broadened; §7 distinguishes reuse from
> replacement; painted/seeded jets stay a first-class init option.

---

## 1. Motivation & goal

The v1.6 vorticity-streamfunction solver is 2D, barotropic, single-layer, and
**prescription-driven**: it nudges absolute vorticity toward an analytic jet
target `q_target = ω_jet + ω_vortex + f`. The alternating-band morphology it
produces is real shear-instability physics, but the jets themselves are painted,
there is no vertical structure, and there is no intrinsic energy source.

**Goal:** a *prognostic, baroclinic* weather-layer solver in which jets, belts,
and vortices **emerge** from baroclinic instability off a thermally-relaxed
equilibrium, with vertical structure (≥2 layers), valid at the equator and for
large-amplitude vortices (the Great Red Spot).

**Chosen model (decided in brainstorming):**
- **Vertical:** 2–3 stacked layers (start with 2).
- **Formulation:** stacked **reduced-gravity shallow water**, primitive variables
  `(h_i, u_i, v_i)` per layer.
- **Energy source:** Newtonian thermal relaxation of layer thickness toward a
  latitudinally-varying radiative-equilibrium profile `h_eq_i(φ)` + Rayleigh
  bottom drag; baroclinic instability does the rest. The user steers via
  `h_eq(φ)` and/or a seeded jet init (§4.1, §4.4).
- **Render:** existing pipeline, fed from the **top layer** through a new
  encoder shim (§7): thickness anomaly → cloud altitude/color; **relative
  vorticity ζ₁ → the primary detail/contrast channel** (this is what won the
  v1.6 panel — it stays the hero signal); advected tracers → existing dye.

Lineage: forced–dissipative multi-layer shallow-water weather-layer models of
Williams / Cho-Polvani / Scott-Polvani / Showman / Dowling (EPIC). **Scope
caveat:** this is a *shallow weather-layer* model; it does not represent
deep-convection jets (Busse-column / Kaspi-Galanti deep-wind regime).

---

## 2. Governing equations

### 2.1 Continuous form (per layer i, i = 1 top … N bottom)

Vector-invariant momentum + flux-form continuity on the rotating sphere:

```
∂u_i/∂t = −(ζ_i + f) k̂ × u_i − ∇B_i + F_i^u + D_i^u
∂h_i/∂t = −∇·(h_i u_i)            + F_i^h + D_i^h
```

`ζ_i = k̂·(∇ × u_i)`, `B_i = M_i + ½|u_i|²` (Bernoulli), `M_i` = Montgomery
potential (§2.2), `f = f0·sinφ`, `F` = forcing (§4), `D` = dissipation (§4.3).

The vector-invariant form keeps `ζ_i` central so the existing vorticity stencils
are reused. **Conservation note (corrected per review):** Sadourny (1975) /
Arakawa-Lamb (1981) energy- and potential-enstrophy conservation is a property
of a *specific C-grid PV-flux + KE-averaging discretization*, **not** of the
continuous vector-invariant PDE. We therefore either (a) adopt the C-grid AL
discretization and *claim* enstrophy conservation, or (b) treat energy/enstrophy
as **bounded-drift diagnostics** controlled by hyperviscosity, not conserved
invariants. §3.1 resolves this. We do **not** claim Sadourny conservation while
using operators that do not implement it.

### 2.2 Montgomery potential (the layer coupling)

For N stacked Boussinesq isopycnal layers (ρ_1 < … < ρ_N) with reduced gravities
`g'_k = g (ρ_k − ρ_{k−1})/ρ_ref`:

```
M_i = Σ_{k=1}^{i} g'_k · η_k ,    η_k = Σ_{j=k}^{N} h_j
```

`η_k` = height of the top of layer k. Special cases:

- **N=1 reduced gravity:** `M_1 = g'·h_1` → `−g'∇h_1`. Active top layer, free
  upper surface, quiescent deep abyss at the bottom interface.
- **N=2 (production target):** active layers 1,2 over a deep at-rest abyss:
  ```
  M_1 = g'_1 (h_1 + h_2)
  M_2 = g'_1 (h_1 + h_2) + g'_2 · h_2
  ```
  Barotropic mode (`g'_1`, fast external wave) and baroclinic mode (`g'_2 ≪
  g'_1`, slow internal wave). The **mid-latitude** baroclinic deformation radius
  `L_D = √(g'_2 H)/f` sets the eddy/vortex scale. **Near the equator** the
  relevant scale is the *equatorial* deformation radius `L_eq = (c/β)^{1/2}`,
  `c = √(g'_2 H)`, with equatorially-trapped Kelvin / Rossby / mixed-Rossby-
  gravity modes (Matsuno 1966; Gill 1982). The global spherical formulation
  carries equatorial β implicitly, so these modes emerge natively — no separate
  β-plane channel is required, but they must be *validated* (§6).

> ⚠️ **Sign/coefficient discipline:** the `g'_{ij}` coupling is the single most
> error-prone part of the model. It is pinned by the Williamson steady
> geostrophic-balance test (§6); we validate the matrix, we do not hand-trust it.

### 2.3 Reduced-gravity formulation (one clean choice)

The model is a **true reduced-gravity** stacked system: active upper layer(s),
free upper surface, pressure gradient `−∇M_i` per §2.2, quiescent deep abyss at
the bottom interface. There is **no rigid lid** and **no artificial `g'_eff`**
softened-lid term (the rev-1 "fast but finite external mode" was ill-posed — it
gives the barotropic mode a tunable, physically meaningless speed and corrupts
geostrophic adjustment and equatorial wave reflection). If the external/
barotropic gravity-wave CFL is the concern, it is handled by solving the **true**
barotropic mode implicitly (§3.4, Milestone 2), not by corrupting `g'_1`.

---

## 3. Discretization

### 3.1 Grid, staggering, domains — DECIDED BEFORE M1

- Reuse the three domains: equirect (`W × W/2`) + two AE polar patches with the
  existing AE-coordinate machinery.
- **Staggering (review-inverted decision).** A *collocated* A-grid composes the
  centered gradient and divergence operators into a Laplacian that **decouples
  even/odd points**; the `2Δx` pressure checkerboard sits in its null space
  (zero centered gradient), is geostrophically inert (`f × u_checkerboard` is
  itself a checkerboard, no restoring force), and is continuously re-excited by
  forcing. It corrupts exactly the balance (Williamson test 2) and conservation
  diagnostics that are **M1's own exit gate** — so it cannot be deferred to
  M3/M4. **Decision (LOCKED): Arakawa C-grid** (h at centers, u/v on faces) —
  `∇M` becomes a single face difference with no `2Δx` null space, and it gives
  the clean PV/KE averaging that the C-grid Arakawa-Lamb enstrophy-conserving
  discretization needs. (Rhie-Chow momentum interpolation on the existing
  collocated layout was the considered fallback; C-grid is chosen for the
  cleaner conservation.) The Milestone 0 spike (§5) still probes a checkerboard
  initial condition to confirm the C-grid is implemented correctly (it must show
  no growing 2Δx mode).

### 3.2 Prognostic state (per layer, per domain)

| Field | Texture | Location (C-grid) |
|-------|---------|-------------------|
| `h_i` | R32F | cell center |
| `u_i` | R32F | east/west faces |
| `v_i` | R32F | north/south faces |

(Face-staggered channels may be packed; exact texture packing is a Milestone 0
implementation detail.) For N=2, 3 domains, ping-pong: ~24–30 textures. Memory
is not a constraint.

### 3.3 Momentum form

Vector-invariant **velocity** form (reuses ζ; §2.1). Velocity is **not**
semi-Lagrangian-advected — its nonlinear advection is carried entirely by the
`−(ζ+f)k̂×u` and `−∇B` terms. At sharp fronts/outcropping (GRS regime, §R3) the
Bernoulli form is grid-noisier than flux-form momentum; this is controlled by
hyperviscosity and validated by the Williamson 5/6 front cases plus an explicit
large-amplitude front stress test (§6). If those show velocity-form is
inadequate at fronts, the fallback is flux-form momentum `∂(hu)/∂t` with a TVD
limiter (accepting loss of exact enstrophy conservation) — a Milestone 0
decision.

### 3.4 Time integration

- **Advective + Coriolis + forcing:** explicit, with **time-centered
  (trapezoidal) Coriolis** evaluated consistently with `∇M` at the same time
  level and grid location, so discrete geostrophic balance holds to scheme order
  (a stand-alone exact-rotation operator would inject an O(fΔt) balance error
  that fails Williamson test 2 — and is easily misdiagnosed as a Montgomery sign
  error).
- **Gravity-wave terms** (`∇M` / `∇·(hu)`):
  - **Explicit (M1):** forward step, CFL `dt < Δx_min / c_gw`,
    `c_gw = √(g'_max H)`, where `Δx_min` is the **minimum** grid spacing over the
    active domain (the equirect zonal spacing `∝ cosφ` shrinks toward the poles;
    `dt_safety` must name the binding speed *and* binding Δx). Single active layer
    at M1 → one unambiguous `g'`.
  - **Semi-implicit (M2):** linearize **only the fast gravity-wave terms about a
    fixed reference** `H_ref` (resting/layer-mean thicknesses); keep the
    thickness-anomaly-dependent part of `∇M` in the **explicit RHS**. This keeps
    the Helmholtz operator **constant** (one diagonalization, reused) and the
    splitting consistent at large amplitude. Diagonalize the **symmetrized**
    (layer-mass-metric) vertical coupling matrix → guaranteed real modal wave
    speeds `c_m`. Each normal mode is a Helmholtz solve `(∇² − λ_m²)φ_m = RHS`
    via the red-black SOR **harness** (the operator, the per-mode loop, the `λ_m²`
    center-coefficient term, and the RHS assembly are **new**; only the sweep
    scaffolding + metric stencils are reused). Use a **time-centered
    (Crank-Nicolson, θ=½)** implicit step so it is energy-neutral, not
    dissipative; require the SOR residual to converge well below the §6
    energy-drift budget (e.g. < 0.01× per-step drift tolerance).

### 3.5 Transport schemes

- **`h`: flux-form finite-volume, mandatory (resolves R2).** Conserves mass to
  machine precision. Semi-Lagrangian is **not** used for `h` — it is
  non-conservative and incompatible with success criterion §9.3. Positivity is
  enforced by an **FCT (flux-corrected transport) limiter** on the FV flux so
  `h > h_floor` *and* mass conservation hold in the same operator (resolves R3);
  no post-hoc clip. Near outcropping, wave speed uses `√(g'·max(h, h_floor))`
  and a near-floor layer zeroes its contribution to `∇M` (goes passive rather
  than producing imaginary speeds). A per-step assert traps the first
  NaN/negative `h` (the v1.6 sticky-NaN discipline).
- **Passive tracers/dye:** reuse semi-Lagrangian MacCormack (non-conservation
  cosmetically acceptable, as v1.6 already accepts).

---

## 4. Forcing & dissipation

### 4.1 Thermal (mass) relaxation — the energy source

```
F_i^h = (h_eq_i(φ) − h_i) / τ_rad
```

`h_eq_i(φ)` is a per-layer, per-latitude radiative-equilibrium thickness built by
`profiles.py` (repurposed from `u(φ)` to `h_eq(φ)`). The pole-to-equator and
per-band contrast tilts the interfaces; the available potential energy drives
**baroclinic instability**, generating eddies and maintaining jets via eddy
momentum-flux convergence. **τ_rad is a local profile constraint, not a
total-mass conserver** — it cannot mask transport mass error (which is why §3.5
mandates conservative FV).

**Balanced init.** The initial state is set in **gradient-wind balance** with
`h_eq` (not at rest) so the run starts near-balanced. *Scope of what this buys
(corrected per review):* balanced init removes only the fast geostrophic-
adjustment / gravity-wave transient (which also avoids the R3 NaN shock). It does
**not** accelerate the baroclinic eddy-growth + nonlinear-saturation + inverse-
cascade jet-organization timescale — that is the dominant step cost and is
governed by §4.4 / R5.

**Equatorial init closure.** A naive geostrophic inversion `u = (g'/f)∂h/∂y` is
singular as `f → 0`. Within `|φ| < φ_eq`, use full **gradient-wind balance**
(centrifugal/curvature terms stay finite) or an equatorial-β-plane (Gill)
balanced solution, blended to geostrophic poleward; cap `|u_init|` and let a
brief bounded adjustment absorb the residual. (Williamson test 2's analytic
solid-body field is well-behaved across the equator and does *not* exercise this
production path — hence the dedicated equatorial wave test in §6.)

### 4.2 Bottom drag

Rayleigh drag on the lowest active layer: `F_N^u ⊇ −u_N / τ_drag` — the canonical
Held-Larichev / Thompson-Young representation of frictional coupling to the deep
interior (this is correct, not a layer-placement bug). **Equilibrated jet
amplitude/count is an emergent coupled balance** among `τ_drag`, `β`, `τ_rad`,
and `L_D` (the zonostrophic / Rhines result), *not* a free knob set by drag
alone. Target regime: strongly **zonostrophic** (Jupiter analog). Guardrails:
under-damped → runaway jets (frictionless arrest at the domain/Rhines scale);
over-damped → inverse cascade killed before jets form.

### 4.3 Dissipation

- **Biharmonic hyperviscosity** on velocity (and optionally `h`), reusing the
  grid-normalized `∇⁴` machinery from v1.6 (carry the grid-norm fix forward).
- **Thickness positivity** via the FCT limiter in §3.5 (not a separate floor).
- Energy/enstrophy are **monitored as bounded-drift diagnostics** (§2.1, §6),
  controlled by hyperviscosity; exact conservation is claimed only for mass
  (flux-form) and, if the C-grid AL discretization is adopted, potential
  enstrophy. PV `q = (ζ+f)/h` is defined with `max(h, h_floor)` to avoid the
  `1/h` singularity in the outcropping regime.

### 4.4 Spin-up control & art-direction (seeded vs emergent jets)

Two first-class init/forcing modes (resolves the art-direction regression).
**Emergent is the DEFAULT product mode** (the "true" solver the project exists
for); seeded ships as a first-class option for deterministic, art-directed
presets and as the spin-up fallback.

- **Seeded (preserves the v1.6 preset workflow exactly):** seed the initial
  condition with the v1.6 *painted* jet (+ optionally eddy) field as the IC; the
  prognostic solver runs as a **finishing pass** adding folded-filament
  richness over a few hundred steps. Deterministic, art-directable, short bake.
- **Emergent (the "true" mode):** jets self-organize off the `h_eq` tilt.
  Steered indirectly via `h_eq`; jet count/strength set by the physics
  (Rhines/deformation scale). The claim that "the preset workflow survives" is a
  **hypothesis gated on validation** (§6 mapping-stability gate), not an
  assertion. Preset re-authoring is an explicit budgeted M5 task.

A **spin-up kill-gate** (§5, M0/M3) measures steps-to-acceptable-jets on a
coarse run with the existing `spinup_sweep.py` methodology; if emergent
equilibration exceeds budget, fall back to seeded mode or partial nudging.

---

## 5. Milestone decomposition

Each milestone is a separate spec→plan→implement cycle, behind
`solver.type = "shallow_water"`; default stays `kinematic`, byte-identical.

**M0 — Image-first de-risking spike (NEW, gates everything else).**
A throwaway minimal 2-layer reduced-gravity solver, **equirect only**, no
Williamson suite / no semi-implicit / no polar patches. Resolve **R1 (grid) and
R3 (positivity) here** (checkerboard-decay probe + outcropping stress). Feed
top-layer `h`-anomaly + `ζ₁` through a draft render encoder and run the **v1.6
blind judge panel at ≥4K**. **Kill-gate:** the panel must *prefer* the spike
render to v1.6 (a tie is a fail), or `coher` must move from v1.6's 0.384 toward
the 0.62 reference. Also measure steps-to-equilibration vs budget. **If the
spike cannot beat painted-jet v1.6, stop — do not build M1–M5.**

**M1 — Core single-layer solver (validated).**
1 layer, equirect only, C-grid (or Rhie-Chow) vector-invariant + flux-form/FCT
continuity, explicit, trapezoidal Coriolis. CPU reference `shallow_water_ref.py`.
Exit gate: §6 solution-accuracy + self-budget-closure; determinism + P0.5 hash
gate; checkpoint byte-exact.

**M2 — Semi-implicit gravity-wave step.**
Symmetrized fixed-reference Helmholtz via SOR harness (§3.4). Exit gate:
solution agreement with M1 explicit *to truncation error* at equal dt (expect a
small bounded one-sided energy offset, growing with the dt multiplier — **not**
invariant agreement); **plus a large-thickness-anomaly stability test** before
claiming 5–10× dt headroom; per-mode SOR convergence-vs-iteration validated for
both stiff and slow-baroclinic modes.

**M3 — Multi-layer + baroclinic forcing.**
2 layers, Montgomery coupling, thermal relaxation + bottom drag. Exit gates:
emergent baroclinic instability (growth rate vs **β-plane** linear theory incl.
Charney-Stern PV-gradient-sign-change, plus the f-plane QG-limit Phillips check
as a small-amplitude unit test); emergent jet spacing matches
`L_Rhines = π√(2 U_rms/β)` within tolerance (quantitative, not "matches the
Rhines scale" qualitatively); **finite-amplitude** GRS-scale vortex stability
(Rossby > 0.1); energy budget closes (forcing in ≈ drag + hypervisc out); the
spin-up kill-gate; the `h_eq → jet count/latitude/amplitude` mapping is stable
across ≥2 resolution tiers; **early ≥4K render-fidelity check** vs v1.6 (don't
wait for M5).

**M4 — Three-domain (polar patches) + halo exchange.**
AE patches with the metric in the SW flux/gradient terms (the divergence/flux
and Bernoulli-gradient operators on the AE metric are **new**, derived +
validated in `shallow_water_ref.py`). Exchange `h`/momentum across halos via
**re-typed** exchange kernels (R32F/RG32F). Exit gates: seam-free, poles clean
(v1.6 polar-judge standard) **and** a new global-conservation gate — total mass
over all 3 domains (metric-weighted) within tolerance, seam energy/enstrophy
flux balanced. Prefer a **flux-matched (mortar)** coupling transferring `h·u_n`;
if infeasible, downgrade the mass claim to "bounded leak < tolerance" for the
3-domain config and state the tolerance.

**M5 — Render integration + art-direction + proof gate.**
A new `h/ζ/dye → RGBA-tracer` **encoder kernel** feeding the unchanged
`derive.comp` (this shim is where R8 lives). `h_eq` driven by band presets;
budgeted preset re-authoring. Blind 3-judge forced-choice proof gate — **strict
win** required (panel prefers the new render; ties fail), or quantified `coher`
improvement toward 0.62.

---

## 6. Validation strategy (CPU ground truth)

Mirrors v1.6 `vorticity_ref.py` discipline. **The gate is split** (rev-1's
"within published bounds" conflated two different things):

- **`shallow_water_ref.py`** — NumPy reference for gradient/divergence/curl on the
  equirect metric **and the AE patch metric** (both new operators), the Montgomery
  coupling matrix, the C-grid PV/Coriolis flux stencil (state explicitly where
  `q`, `ζ`, `h`, `f`, mass flux live), and one explicit SW step. GPU kernels
  diffed per-field.
- **Solution accuracy** — Williamson et al. (1992): Test 2 (steady geostrophic;
  pins the Montgomery sign matrix + Coriolis) and Test 5/6 (mountain /
  Rossby-Haurwitz; nonlinear advection + fronts) compared via **l2 error norms
  vs the analytic/reference solution**.
- **Conservation** — the solver's **own budget closure**, not an external
  spectral number: mass drift to machine precision (flux-form); energy &
  potential-enstrophy drift **fully accounted by the diagnosed dissipation
  terms** (forcing/drag/hypervisc) so the residual closes to round-off. The
  correctness signal is "unaccounted invariant residual ≈ 0."
- **Equatorial waves** — Matsuno dispersion (Kelvin + n=1 Rossby): Williamson
  2/5/6 do not exercise equatorial trapping, and the equatorial jet is Jupiter's
  most prominent (and contested) feature.
- **Baroclinic instability (M3)** — β-plane two-layer growth rate incl.
  Charney-Stern criterion (selects jet latitudes); f-plane QG-limit Phillips as a
  small-amplitude unit test; finite-amplitude GRS-scale vortex stability.
- **Gates carried from v1.6** — determinism, P0.5 hash, byte-exact checkpoint,
  and the v1.6 lesson: tests that each parameter *changes* output and that fields
  *evolve* (guard against a no-op solver passing a static-field determinism test).

---

## 7. Integration with existing code — reused vs replaced

**Genuinely reused (primitives & helpers):**

| Asset | Reuse |
|-------|-------|
| `GpuContext`, moderngl compute, R32F ping-pong/dispatch | directly |
| AE **coordinate maps + scalar Laplacian + metric coefficients** | directly |
| `advect.comp` **backtrace + Catmull-Rom + min/max-limiter helpers** | for tracer/dye transport only |
| red-black SOR **sweep harness + metric stencils** | M2 Helmholtz (operator/RHS/per-mode loop are new) |
| `∇⁴` grid-normalized hyperviscosity | directly (carry grid-norm fix) |
| vorticity / Laplacian stencils | `ζ_i` |
| `profiles.py` | build `h_eq(φ)` per band |
| checkpoint **npz + version-gate + absent-key-tolerant load** discipline | + new layer-indexed key schema, version bump |
| `derive.comp` / `maps.py` render kernels | unchanged, behind the new encoder |
| `params/model.py` `SolverParams`, tiers, presets | new `shallow_water` fields |

**Replaced or net-new (NOT drop-in reuse — the rev-1 §7 understated this):**

- **`solver.step()` orchestration, `_produce_psi`, and `velocity.comp`/`vel_tex`
  streamfunction→velocity derivation are REPLACED** by a new SW step loop where
  `(h_i, u_i, v_i)` are prognostic and co-evolve (explicit in M1, Helmholtz-
  coupled in M2). The freeze-ψ-then-advect-tracers data flow does not survive.
- **New stamp-free scalar-`h` (FV/FCT) and face-velocity kernels** — `advect.comp`
  PASS 2 bakes in band/vortex/wave relaxation stamps (appearance physics) and is
  hardwired RGBA; it is not a generic transport kernel.
- **New divergence/flux and Bernoulli-gradient operators on equirect + AE metric**
  (no divergence operator exists in the codebase today).
- **New Helmholtz operator** (λ² center-coeff term, per-mode loop, RHS assembly).
- **Re-typed halo-exchange kernels** (R32F/RG32F; the v1.6 `mix()` blend is a
  non-conservative resample — see M4 flux-matching).
- **New `h/ζ/dye → RGBA-tracer` encoder kernel** feeding `derive.comp` (M5; R8).
- **New layer-indexed checkpoint schema** + GENERATION_VERSION bump.

New `SolverType.SHALLOW_WATER = "shallow_water"`. New params (RESTART tier):
`n_layers`, `g_reduced[]`, `tau_rad`, `tau_drag`, `h_mean[]`, `sw_hypervisc`,
`dt_safety`, semi-implicit toggle, init-mode (seeded/emergent). Kinematic/
vorticity untouched → byte-identical, P0.5 gate stays green.

---

## 8. Risks & open decisions (post-review status)

- **R1 — Grid staggering.** RESOLVED: C-grid (Rhie-Chow fallback), decided in M0
  before M1. No longer deferred.
- **R2 — Mass conservation.** RESOLVED: flux-form FV mandatory for `h`; SL only
  for passive tracers.
- **R3 — Thickness positivity.** RESOLVED in scheme: FCT limiter in the FV
  transport (positivity + conservation in one operator) + passive-layer
  outcropping handling + first-NaN trap.
- **R4 — Equatorial validity.** Addressed: equatorial init closure (§4.1),
  equatorial deformation radius (§2.2), Matsuno wave validation (§6); global
  sphere carries β implicitly. Open: whether the super-rotating equatorial jet
  needs an explicit eddy/forcing source (flagged in §9).
- **R5 — Spin-up budget.** Addressed: timescale decoupling (§4.1), seeded-mode
  fallback (§4.4), quantified kill-gate (M0/M3). Open until M0 measures it.
- **R6 — Semi-implicit modes.** RESOLVED: symmetrized fixed-reference
  linearization + explicit anomaly + large-amplitude M2 test.
- **R7 — Cross-domain conservation.** Addressed: M4 global-conservation gate +
  flux-matched coupling (or stated bounded-leak tolerance).
- **R8 — Render fidelity from thickness.** Addressed: ζ₁ stays the detail channel
  (not thickness); early ≥4K fidelity check at M0/M3; encoder shim named. Open:
  whether MacCormack-advected (un-nudged) ζ₁ is as filament-rich as v1.6's
  nudged field — the M0 kill-gate answers this.
- **R9 — Scope realism.** Addressed by M0: value is proven before M1–M5 are
  built; the four hard unknowns (R1, R3, R5, R8) are pulled into the spike.

---

## 9. Success criteria

1. A free-running 2-layer SW solver whose jets/belts/vortices **emerge** from
   baroclinic instability off `h_eq` (or reproduce the seeded preset
   deterministically in seeded mode).
2. Passes the split §6 validation (Williamson l2 norms; equatorial Matsuno
   waves; β-plane baroclinic growth).
3. Conserves mass to machine precision **(flux-form interior)**; energy/enstrophy
   drift fully accounted by diagnosed dissipation (residual ≈ 0). For the
   3-domain config, mass conserved or bounded-leak < stated tolerance.
4. Three-domain, seam-free, poles-clean (v1.6 polar-judge standard) with the M4
   global-conservation gate met.
5. Renders a top-of-atmosphere that a blind forced-choice panel **prefers** to
   v1.6 (ties fail), or a quantified `coher` improvement toward 0.62.
6. Default `kinematic`/`vorticity` remain byte-identical (P0.5 hash gate green).
7. Deterministic, byte-exact checkpoint round-trip, full test suite green.

*Open caveat (R4):* if the emergent equatorial super-rotating jet cannot be
maintained by the resolved dynamics, it may require an explicit eddy/forcing
source the model does not yet contain — to be decided at M3.

---

## 10. Open decisions

**Resolved by the user (2026-06-15):**
1. **Grid → Arakawa C-grid** (locked, §3.1). Rhie-Chow fallback dropped.
2. **Default product mode → emergent** (§4.4). Seeded shipped as a first-class
   option, not the default.
3. **M0 kill-gate stands** — the project stops if a real solver cannot out-render
   painted-jet v1.6 at ≥4K.

**Still open (decide during implementation):**
4. **2 vs 3 layers** for production (2 is the M3 target; 3 if the vertical
   richness earns its cost).
5. **Velocity-form vs flux-form momentum** if the M0/M1 front stress test shows
   Bernoulli-form grid noise is unacceptable.

---

## 11. Changes from rev 1 (adversarial-review fold-in)

- **NEW Milestone 0** image-first spike gates the whole build (scope/R9).
- **Grid decision inverted** to C-grid/Rhie-Chow before M1 (was "start
  collocated") — the 2Δx checkerboard hits M1's own gate.
- **Flux-form FV + FCT for `h` mandatory** (was SL-MacCormack option); SL kept
  only for passive tracers.
- **§2.3 rewritten** to one clean reduced-gravity model; deleted the ill-posed
  `g'_eff` softened-lid.
- **Sadourny-conservation justification corrected** — it needs the C-grid AL
  discretization; otherwise energy/enstrophy are bounded-drift diagnostics.
- **§3.4 semi-implicit** resolved: symmetrized fixed-reference linearization +
  explicit anomaly + Crank-Nicolson + per-mode convergence gate + large-amplitude
  M2 test.
- **Coriolis** specified time-centered/trapezoidal, consistent with `∇M`.
- **§6 validation split** into solution-accuracy + self-budget-closure; added
  equatorial Matsuno waves, β-plane Charney-Stern, finite-amplitude checks.
- **§7 rewritten** to separate genuine reuse from the replaced step-loop / new
  kernels (step orchestration, divergence operators, Helmholtz, encoder shim,
  layer checkpoint schema).
- **Art-direction** preserved via first-class seeded mode; "preset survives"
  downgraded to a validation-gated hypothesis; preset re-authoring budgeted.
- **Spin-up** timescales decoupled; balanced-init no longer credited with
  equilibration speed-up; quantified kill-gate added.
- **Success bar raised** to a strict render win (ties fail).
- **Equatorial init/scale** treatment added (gradient-wind/Gill closure; L_eq).
- **M4 global-conservation gate** + flux-matched halo coupling added.
- **Drag** reworded as an emergent coupled balance; quantitative Rhines gate.
```
