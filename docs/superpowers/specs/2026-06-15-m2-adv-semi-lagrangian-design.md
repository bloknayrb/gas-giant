# M2-adv — Semi-Lagrangian Semi-Implicit Advection: Design Spec

**Status:** Approved (design) — 2026-06-15. Successor to M2-core (`ede3d89`, gravity-wave CFL removed).

**Goal:** Remove the residual *advective* CFL so the single-layer shallow-water solver
takes genuinely large stable **and accurate** steps on fast zonal jets — the headline
`dt` factor — while keeping mass exactly conserved and the explicit / M2-core paths
byte-identical.

---

## 1. Problem statement

M2-core made the gravity-wave terms (`g'∇h`, `∇·(H_ref u)`) implicit via a scalar
symmetric Helmholtz solve, removing the gravity-wave term from the CFL. What still
binds the step is the **advective** Courant number

```
C(φ) = |u| · dt / (a · dλ · cosφ)
```

which diverges at the pole (`cosφ → 0`) on fast jets. This is a *cell-Courant* limit on
the explicit donor-cell + FCT transport, not a per-mode spectral one. The T0 spike
(`tests/spikes/test_polar_advective_spike.py`) **falsified** the cheap spectral-filter
route: a Boyd filter at `C = 1.5` stays bounded but the retained low-wavenumber band has
~0.41 relative L2 error against a `C = 0.3` reference — the mean flow itself is corrupted
because the upwind branch still runs at `C > 1`. The transport *scheme* must change.

## 2. Chosen approach — combined SLSI with conservative remap

A combined **semi-Lagrangian semi-implicit (SLSI)** step. For `fast_advection=True`:

| Term | Treatment |
|---|---|
| Height / continuity | **SLICE-style conservative cascade remap** along departure trajectories — exact mass conservation, unconditional advective stability |
| Momentum | **Advective semi-Lagrangian** along the *same* trajectories (replaces M2-core's Eulerian vector-invariant advection); KE/Bernoulli applied at arrival |
| Gravity-wave pressure + divergence | **Implicit** via the *unchanged* M2-core symmetric Helmholtz solve; SL departure values feed its RHS (this is the "combined", not operator-split, coupling) |
| Coriolis | Deferred-correction (Picard), lagged along the trajectory |

**Why SLICE, not Lin-Rood FFSL.** Pure Lin-Rood FFSL is *Eulerian* flux-form; "combined
SL-SI" is naturally a *material-derivative* form that is non-conservative for the
transported field. The scheme that delivers **both** exact conservation **and** the SL-SI
coupling is SLICE (Semi-Lagrangian Inherently Conserving and Efficient; Zerroukat et al.) —
conservative cascade remapping along Lagrangian trajectories. "Conservative FFSL" (the
approved intent: exact mass + unconditional advective stability) resolves to conservative
semi-Lagrangian = SLICE.

## 3. Components (CPU reference first, each independently testable)

1. **`departure_points`** — fixed-count iterated-midpoint trajectory on the lon-lat sphere
   with the `1/(a cosφ)` metric. One departure point per arrival h-center and per u/v-face.
   Deterministic: fixed iteration count, no convergence early-out (same discipline as the
   SOR loop). Polar handling: trajectories crossing a pole wrap in longitude by π.

2. **`slice_remap`** — 2-D conservative remap factored into a zonal→meridional **cascade**
   of two 1-D conservative PPM (piecewise-parabolic) remaps. Exact mass conservation by
   construction (`Σ remapped ≡ Σ source` in the `cosφ·a²` measure). Monotone PPM limiter
   for positivity — this replaces FCT's positivity role on the SL path. cos-weighting on the
   meridional sweep; pole rows handled by the `cos_v = 0` structural zero.

3. **`sl_momentum`** — cubic-Lagrange (tri-cubic on the 2-D stencil) interpolation of `u, v`
   at departure points along the same trajectories; arrival-side KE/Bernoulli gradient and
   the `(1-θ)` explicit pressure half. The `θ` implicit pressure half is carried by the
   existing `velocity_backsub` on the full solved height (unchanged from M2-core).

4. **`step_slsi`** — assembly. SL departure values (`h̃`, interpolated `u*, v*`) feed the
   **unchanged** M2-core `helmholtz_rhs` → `helmholtz_sor` → `velocity_backsub`. `L_sym`,
   the SOR solver, and Picard-Coriolis are reused verbatim. Coriolis lagged along trajectory.

## 4. Layered dual-path (extends M2-core's hash discipline)

Three nested, hash-gated fallbacks:

```
fast_advection=False  ≡ M2-core SI step (byte-identical)
   semi_implicit=False ≡ M1 explicit step (byte-identical)
```

No SLSI-only construction (trajectory buffers, remap reconstruction state) may touch shared
`__init__` state — asserted, as in M2-core T6.

## 5. Crux spike — front-loaded de-risk (`T0-adv`, CPU-only, before production code)

A fast polar jet with advective `C ≫ 1` at the test latitude, advanced by `step_slsi` at
`N × dt_adv`. Gates:

- **(a)** bounded **and** mass-conserved to round-off;
- **(b)** matches a fine-dt explicit reference to a stated L2 tol over the run — an
  **accuracy** gate, not mere finiteness (a dissipative scheme staying bounded is *not* the
  headline);
- **(c)** `slice_remap` conserves mass exactly in isolation.

If SLICE + SLSI cannot hit the accuracy gate at large `C`, the combined approach is falsified
before the GPU port is built — mirroring how M2-core pulled its gravity-wave gate forward and
caught the scheme error early. On failure, the design re-enters brainstorming (candidate
fallbacks: reduced polar grid keeping explicit FCT; non-conservative SL + global mass fixer).

## 6. Validation & gates (end-to-end)

- **Conservative remap:** exact mass to round-off; monotonicity (no new extrema) on an
  adversarial near-floor IC; independent a-scaling (`a = 1` vs `a = 6.4e6`).
- **Trajectory solver:** measured convergence order of the departure point vs the analytic
  solid-body-rotation trajectory (≥ expected order on 2× refinement).
- **Per-kernel GPU↔CPU diff** at `atol 2e-5` on pre-division quantities (departure points,
  remap fluxes, SL-interpolated momentum, full `step_slsi` 1-step + N-step), each with
  a-scaling.
- **Headline accuracy gate:** fast-jet case, large-dt SLSI vs fine-dt reference, L2 over N
  steps → the honest realized factor = largest dt that stays within the L2 tol ÷ explicit dt.
  Reported as an accuracy factor, never a bare stability factor.
- **Dual-path byte-identity:** `fast_advection=False` ≡ M2-core; nested ≡ M1 (P0.5 hash).
- **Determinism:** byte-identical SHA1 of `(h, u, v, Δh-warmstart)` over a fixed multi-step run.
- **Long-run conservation:** `total_mass` rtol round-off (un-renormalized); potential
  enstrophy bounded.

## 7. GPU kernels (anticipated)

`sw_departure.comp` (iterated-midpoint trajectory), `sw_slice_remap.comp` (cascade PPM, two
passes), `sw_sl_momentum.comp` (tri-cubic departure interpolation). Reuse `cosCenter`,
`cosVface`, `sinAcc`, `wrapX` (branch form) from `sw_common.glsl`; reuse the M2-core Helmholtz
kernels verbatim. Checkpoint bumps to `version=3` (adds `fast_advection` + trajectory params).

## 8. Risks (carried into plan + adversarial review)

1. **SLICE cascade remap is research-grade** — the conservative 1-D PPM remap with monotone
   limiter and the cascade ordering are the hardest new numerics in all of M2. The T0-adv
   spike exists to falsify it early.
2. **Combined coupling vs splitting** — feeding SL departure values into the SI Helmholtz RHS
   (rather than operator-splitting) is the high-fidelity choice but the tightest integration;
   the RHS assembly must remain consistent with the M2-core operator's adjoint structure.
3. **Momentum SL is non-conservative** (acceptable — only mass must conserve exactly), but PV
   /enstrophy drift must stay bounded; gated.
4. **Trajectory polar wrap** — departure points near/over the pole need the longitude-π wrap;
   a correctness hazard with the `wrapX` branch form.

## 9. Out of scope (own milestones)

- **M2-AE** — azimuthal-equidistant prognostic chart (own brainstorm→spec→plan).
- **M3** — 2-layer baroclinic coupling (the next real render gate).

`shallow_water_ref.py` remains the gold-standard CPU ground truth for all of the above.
