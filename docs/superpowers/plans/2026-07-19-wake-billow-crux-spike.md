# Wake Billow Crux-Spike Implementation Plan (rev 2 — post 3-lens plan review)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Locate the binding constraint that prevents the hero-wake billow chain — among {relax-nudge, ambient strain, wake machinery, solver numerics, resolution} — via a validated CPU pseudo-spectral spike with pre-registered gates, BEFORE any production change.

**Architecture:** A throwaway numpy pseudo-spectral 2D barotropic vorticity solver on a doubly-periodic box at exactly production cell size, with a nested ladder of production-math ablations. Spec: `docs/superpowers/specs/2026-07-19-wake-billow-crux-spike-design.md` — its gates and decision tree are FROZEN; this plan implements them plus the plan-review deltas recorded in the spec's addendum.

**Tech Stack:** Python 3.13, numpy, scipy (ndimage/stats/special/signal), cv2 for composites. `gasgiant` imports are CPU-only — no GL anywhere.

## Global Constraints

- ALL files live in `SPIKE = C:\Users\blokn\AppData\Local\Temp\claude\C--Users-blokn-Documents-Github-gas-giant\6a4c44fe-478f-4964-af1c-aedd764f3bcb\scratchpad\spike_billow\`. **NOTHING committed to git during research.** No file under `src/gasgiant/**` is touched, ever.
- Commands run from the repo root: `uv run python <SPIKE>\<script>.py`.
- Radians + production time units everywhere; `config.py` is the SINGLE SOURCE for every constant, including all environment constants (frozen at Task 4 sign-off — any later change is a declared extension that VOIDS and reruns all completed Gate-1 cells).
- **Every forcing/damping rate is written and coded as a PER-STEP FRACTION** (e.g. `1/TAU_STEPS = 1/600`), never per-unit-time. (Review: one missed DT factor silently changes a rate 1500×.)
- Hard timebox 3 days. Gate 0a green by mid-day 1 or STOP → INCONCLUSIVE-INFRA. Scan cells frozen in Task 5 before any Gate-1 run. Extensions (fine-grid rescan, extra cells) only as declared extensions or user decisions — never "budget permitting" mid-run.
- Production-math rungs are TRANSCRIPTIONS with source file:line cited in a comment.
- Determinism: `default_rng(SeedSequence([777, id]))` per named stream; `SEEDS` frozen in Task 0. Gate runs: **both** seeds for **every** cell (pass AND fail must replicate — a falsification verdict may not rest on n=1).
- Visual judgments: reference-anchored (PIA07782 crop flipped WEST), by me AND an adversarial agent — at the Gate 0b mock (BEFORE sending it to the user), at Gate 1 passing panels, and at the Task 10 verdict panels.
- If the user rejects the Gate 0b mock at ANY point mid-spike: halt Tasks 6–9 immediately, jump to Task 10 with the "flow line moot" outcome.

---

### Task 0: `config.py` — nondim table + frozen environment constants

**Files:** Create `<SPIKE>\config.py`

**Interfaces:** module constants (below) + `warm_profile_window()` → `(y, u_amb, omega_amb)` + `transit_report()` printing/returning u(y_sheet), strip-exit and band transit times.

- [ ] **Step 1: Write config.py** (imports pre-corrected per review — `load_factory_preset` lives in `gasgiant.params.presets`; `coriolis_f0` on `P.solver`):

```python
"""Nondim table + FROZEN environment constants. SINGLE SOURCE. All rates per-step."""
import numpy as np
from gasgiant.params.presets import load_factory_preset
from gasgiant.sim.profiles import build_profiles
from gasgiant.sim.bands import generate_bands
from gasgiant.sim.solver import compute_dt

P = load_factory_preset("gas_giant_warm")

RC   = P.storms.hero_radius                 # 0.108
LAT0 = np.radians(P.storms.hero_latitude)   # -24 deg
DX   = np.pi / (P.sim.resolution // 2)      # pi/2048 (production dphi = dlam)
NX, NY = 1024, 512
LX, LY = NX * DX, NY * DX
LD   = P.solver.deformation_radius; INV_LD2 = 1.0 / (LD * LD)
BETA = P.solver.coriolis_f0 * np.cos(LAT0)  # ~2.74

_bands    = generate_bands(P.seed, P.bands)
_profiles = build_profiles(P.seed, _bands, P.bands, P.jets,
                           hero_lat_deg=P.storms.hero_latitude, hero_r_core=RC)
DT = compute_dt(P.sim.resolution, P.sim.dt_scale, _profiles.max_speed)

TAU_STEPS = P.solver.vort_relax_tau         # 600; per-step nudge fraction = 1/TAU_STEPS
OMEGA_CEILING = 60.0                        # omega_force.comp:30
HYPERVISC = P.solver.vort_hypervisc
PSI_DRAG  = P.solver.vort_psi_drag          # per-step fraction (omega_force.comp:225)
WAKE_TURB_AMP = 0.6 * P.storms.wake_turbulence * P.storms.hero_emergence  # :195
WAKE_FREQ = 0.9 / RC
EVOLUTION_RATE = P.turbulence.evolution_rate  # fbm time axis increment/step (solver.py:878)
DEV_STEPS = P.sim.dev_steps                 # 700 from the preset, not hard-coded

# ---- FROZEN environment constants (Task 4 sign-off locks these) -------------
Y_SHEET   = LY / 2.0
STRIP_X   = (0.5 * RC, 0.5 * RC + 4 * DX)   # Dirichlet inflow strip (hard-set)
SEED_AMP_FRAC = 0.02                        # fresh strip noise, fraction of A, k < pi/(4 DX)
TR_STEP_W = 0.5 * RC                        # tracer belt/zone step width
SPONGE_X  = 1.5 * RC                        # cosine-ramped, per-step fraction 1/20
SPONGE_RATE = 1.0 / 20.0
MEANHOLD_RATE = 1.0 / TAU_STEPS             # zonal-mean ambient hold; OFF when nudge flag on
BAND_X    = (3 * RC, 12 * RC)               # scoring band (x)
BAND_HW   = 1.2 * RC                        # scoring band half-width (y, about Y_SHEET)
FLANK_W   = 1.0 * RC                        # flank-guard region at y-boundaries
FLANK_MAX = 0.10                            # (deviation from spec's 'boundary third': recorded)
EDGE_BLEND = NY // 8                        # Tukey edge-match (deviation from spec 'mirrored': recorded)

SEEDS = {"gate1": 11, "gate1_replicate": 12, "seed_noise": 13, "fbm": 14,
         "gate15": 15, "gate2": 16, "collateral": 17}

def warm_profile_window():
    y = (np.arange(NY) + 0.5) * DX
    lat = LAT0 + (y - LY / 2.0)
    u = np.interp(lat, _profiles.lat[::-1], _profiles.u[::-1])
    edge = EDGE_BLEND
    ramp = 0.5 * (1 - np.cos(np.pi * np.arange(edge) / edge))
    mean_uv = 0.5 * (u[:edge].mean() + u[-edge:].mean())
    u[:edge]  = mean_uv + (u[:edge]  - mean_uv) * ramp
    u[-edge:] = mean_uv + (u[-edge:] - mean_uv) * ramp[::-1]
    return y, u, -np.gradient(u, DX)

def transit_report():
    """u at the sheet line and the advective clocks every gate is checked against."""
    y, u, _ = warm_profile_window()
    u_sheet = abs(np.interp(Y_SHEET, y, u))
    strip_to_band = (BAND_X[0] - STRIP_X[1]) / max(u_sheet, 1e-6) / DT   # steps
    band_fill     = (BAND_X[1] - BAND_X[0]) / max(u_sheet, 1e-6) / DT    # steps
    return u_sheet, strip_to_band, band_fill

if __name__ == "__main__":
    assert 5e-4 < DT < 1e-3, DT
    us, s2b, bf = transit_report()
    print(f"DT={DT:.3e} TAU_PHYS={TAU_STEPS*DT:.3f} BETA={BETA:.2f} "
          f"box={LX/RC:.1f}x{LY/RC:.1f} rc")
    print(f"u(sheet)={us:.3f}  strip->band={s2b:.0f} steps  band-fill={bf:.0f} steps")
    _, _, om = warm_profile_window()
    print(f"ambient max|du/dy|={np.abs(om).max():.1f} (expect ~22)")
```

- [ ] **Step 2: Run and verify.** `uv run python <SPIKE>\config.py`. Expected: DT ∈ [5e-4, 1e-3]; TAU_PHYS ≈ 0.4; BETA ≈ 2.74; ambient shear within ~2× of 22; **band-fill printed** — the deficit jet ITSELF carries u ≈ A·δ/2 at its flanks, but if u(sheet) is so small that band-fill ≫ run length, the run-length rule in Task 6 (fill + 2 transits) sizes the runs; if that exceeds ~8000 steps, STOP and re-derive the band geometry with the user (the "spatially-developing" premise fails at near-zero advection — review F2/M9). Any import/attr error = stop and amend the PLAN, no ad-hoc edits.

---

### Task 1: `solver.py` — spectral core (stability-reviewed)

**Files:** Create `<SPIKE>\solver.py`

**Interfaces:** `class Box(beta=BETA, screened=True)` with `w`, `tr`, `invert(w=None)`, `step_spectral(forcing_fn=None)`, `energy_total()` (KE + ψ²/(2L_d²) — the conserved quantity of the SCREENED system), `enstrophy()`, CFL monitor `cfl()`. `advector` attribute for Task 8 rungs.

- [ ] **Step 1: Write the core.** Same skeleton as rev 1 with the review's five corrections baked in:
  - **Tracer advected by RK4** with the frozen end-of-step velocity — forward Euler + spectral derivatives is unconditionally unstable (amplification √(1+(k·u·dt)²) at band edge k·u·dt ≈ 2.5 → ×2.7/step; RK2 also unstable; RK4 limit 2√2 = 2.83). Same RK4 helper as ω, velocity computed once.
  - **State truncation:** the post-step spectral filter is `* NU8_FAC * DEALIAS` for BOTH ω and tr (2/3-truncated state → quadratic aliasing lands only in the discarded band).
  - **`beta` and `screened` are Box instance attributes** (Gate 0a needs β=0 and unscreened — no module-global monkey-patching).
  - **CFL guard:** per-step monitor `cfl = max|u|·DT/DX`; if `(2/3)·π·cfl > 2.8`, sub-step internally (DT/2 ×2, sampling on production-step boundaries) and record it — a high-A blowup must not masquerade as physics.
  - Share `rfft2(w)` between inversion and derivatives (~24 transforms/step).
- [ ] **Step 2: Conservation check.** `__main__`: random band-limited ω (k<20, stream `seed_noise`), 500 steps, β=0, no forcing: `energy_total` and enstrophy drift each < 1% (with `screened=True`, KE alone exchanges ~24% with the ψ²/L_d² part — the total is the conserved quantity; review I3). Run and verify.

---

### Task 2: Gate 0a — known-answer validation (STOP GATE; tolerances FROZEN)

**Files:** Create `<SPIKE>\run_gate0a.py`

- [ ] **Step 1: Tanh-layer KH.** `Box(beta=0, screened=False)` (ψ̂[0,0]=0 guard). Base ω̄ = −(U0/δ)·sech²((y−LY/2)/δ), U0=1, δ=8·DX. Seed the box harmonic **m = 9** (kδ ≈ 0.442 — the exact 0.44/δ is not a box harmonic and its seeded cosine would be discontinuous at wrap; review M5), amplitude 1e-4. FROZEN pass bars: fitted growth rate within 20% of σ = 0.1897·U0/δ (fit window: mode amplitude between 10⁻³ and 10⁻¹ of saturation); rollup spacing at FIRST saturation (before pairing doubles it) within 30% of **λ = 2π/k_m ≈ 14.3δ** (the spec's "λ≈7δ" is in vorticity-thickness units δ_ω = 2δ — review B3: 7.1δ hard-coded would fail correct physics). Record the ~2.5% periodic return-shear from the layer's net vorticity (review M6).
- [ ] **Step 2: Screening check.** Gaussian blob (3·DX), azimuthal |u| vs C·K₁(r/L_d), **fit r ∈ [0.05, 0.2] rad** (the rev-1 range [0.1,0.5] exceeds LY/2 and is 50%+ contaminated by periodic images — review B4), RMS < 10% FROZEN.
- [ ] **Step 3: Run.** Both PASS or — after ≤ half a day of debugging — STOP → INCONCLUSIVE-INFRA, report.

---

### Task 3: Gate 0b — visual mock (parallel)

**Files:** Create `<SPIKE>\mock_billows.py`

- [ ] **Step 1:** As rev 1 (interleaved NEUTRAL ropes at 3.0 rc along-spacing / 1.2 rc transverse, flow-warped, composited into the wake box of `..\wakeA\baseline\color.png`). FIRST verify the inputs exist (`..\wakeA\baseline\color.png`, `..\wake_ref.png`, `..\compose_proto.py`); if the baseline render is missing, re-render via `..\wakeA_render.py` before compositing.
- [ ] **Step 2: Adversarial check BEFORE sending.** One agent reviews the mock against PIA07782 (flipped WEST): are the scales/polarity faithful to the measurement? Fix if not — a bad mock poisons the acceptance answer.
- [ ] **Step 3: Send to the user** as the acceptance question. Non-blocking for Tasks 4–8; on acceptance the mock is the FROZEN acceptance image (recorded in the verdict); on rejection → halt rule in Global Constraints.

---

### Task 4: `environment.py` — feeding (Dirichlet inflow model)

**Files:** Create `<SPIKE>\environment.py`

**Interfaces:** `init_state(box, cfg)`, `make_forcing(cfg)` → per-step closure, `flank_guard(box)`.

- [ ] **Step 1: Write it.** Review F3 replaced the rev-1 zone-nudge feeding with a **hard-set (Dirichlet) inflow strip** so the ONLY relaxation term in the measurement domain is Task 7's production nudge:
  - **Strip** (x ∈ STRIP_X, 4 cells): every step, `w` and `tr` are OVERWRITTEN to ambient + sheet-target + fresh strip noise (amplitude SEED_AMP_FRAC·A, k < π/(4·DX), stream `seed_noise` advanced per step — continuous seeding also matches production's per-step fbm environment; review F4).
  - Sheet target (primary): `w_sheet = A·s·exp(−s²)`, `s = (y−Y_SHEET)/δ` (E2's zero-mean deficit-jet vorticity). Variant form: single-signed `A·sech²(s)` (tanh jump).
  - Tracer: `tr_amb = tanh((y−Y_SHEET)/TR_STEP_W)`.
  - **Ambient hold:** zonal-mean-only nudge, per-step factor `MEANHOLD_RATE` (= 1/600); **automatically DISABLED whenever the Task 7 `nudge` flag is on** (the full nudge holds the profile; both on would double-damp the mean — review F5/M8). `make_forcing` logs the effective per-step mean and eddy damping rates of every configuration.
  - **Sponge:** last SPONGE_X before x-wrap, cosine-ramped per-step factor SPONGE_RATE on `(w − ambient)` and `(tr − tr_amb)`.
  - `flank_guard` per config.py constants (run VOID above FLANK_MAX during the scoring window).
- [ ] **Step 2: Smoke run + constant freeze.** A=45, δ=0.30·RC, run `max(3·DEV_STEPS, band_fill + 2·band_transit)` steps (from `transit_report`). Verify: sheet crosses the band, sponge kills structure before wrap, flank_guard < 0.10, CFL guard silent; print u(y_sheet) and the transit numbers. **Sign-off freezes every environment constant** (Global Constraints rule).

---

### Task 5: `metrics.py` — frozen gate criteria (rev 2: excludes the status quo)

**Files:** Create `<SPIKE>\metrics.py`

- [ ] **Step 1: Implement, all FROZEN:**
  - Windows: band = BAND_X × |y−Y_SHEET| < BAND_HW. `hp(f)` = f − gaussian_filter(f, σ=0.5·RC/DX, **mode='wrap'**).
  - **Threshold conditioning (review I4):** `rms0` = rms of hp(tr) over the band at t=0 (the unfolded-interface baseline — the rev-1 "mirror band" is constant-tr ≈ 0 and degenerates to roundoff). Billow segmentation: `|hp(tr)| > 1.4·rms0`.
  - **Billow** = labeled component (8-conn) with area ∈ [(0.3·RC/DX)², **(1.2·RC/DX)²**] and second-moment aspect < 4. The MAXIMUM size bound is the anti-status-quo criterion (review F1: without it, today's 3–5 large rolls pass every test).
  - **Wavelength:** per-row x-FFT of hp(tr), POWER averaged over band rows (band-averaging first cancels transverse-alternating structure — review F12), dominant peak λ*. FROZEN two-interval bracket: λ* ∈ [0.9, 1.5]·RC **or** [2.4, 3.6]·RC, matched hypothesis reported per cell (the rev-1 continuous bracket silently passed 2.0 rc, matching neither — review 3).
  - **Count coupled to wavelength (review F1):** required count `N* = max(4, floor(band_length/λ*) − 1)` when λ* ≤ 1.5·RC, and `N* = floor(band_length/λ*) − 1` (i.e. ≥ 2) when λ* is in the 3-rc window — a reference-true 3.0 rc chain yields ~3 billows in a 9 rc band and must not fail for geometry.
  - Polarity |skew(hp(tr))| ≤ 0.2; coherence (structure tensor, grads σ=0.25·RC/DX, per-pixel c band-averaged — the same convention as the reference measurement scripts; check `..\wake_measure` and pin it) ∈ [0.35, 0.55]; **standalone band-RMS criterion** rms(hp(tr))/rms0 ≥ 1.4 (review 2).
  - ω-core secondary diagnostic: σ0 from the STRIP-noise band, not the sheet-contaminated band (review M3); `sign_template` = sheet-target sign per y-side.
  - **Clocks (review F2/I5/F10):** metrics sampled every 50 steps FROM STEP 0. Formation = first sample meeting the count criterion; FROZEN formation bar = `strip_to_band + max(DEV_STEPS, 2/σ_expected)` steps, recorded per cell (the raw dev-700 transplant ignores fetch; the verdict maps box fetch → production in-place growth explicitly). Stationarity window = the post-`band_fill` portion, ≥ 2000 steps (runs are cheap — extend, don't shrink); criteria: median billow count difference ≤ 1 between window halves AND Theil–Sen slope of count consistent with 0 AND λ* drift ≤ 20%.
  - GATE1 PASS = count(N*) at ≥ 80% of window samples (FROZEN interpretation of the spec's "at any instant of a statistically-steady window" — review 4) ∧ wavelength ∧ polarity ∧ coherence ∧ band-RMS ∧ stationarity ∧ formation ∧ flank_guard ∧ CFL-clean.
- [ ] **Step 2: Self-test (MUST run before any Gate-1 run):** (a) synthetic reference-scale rope chain → passes everything except stationarity (single frame); (b) white noise → fails; (c) **the REAL status quo**: extract the wake-band tracer proxy from `..\wakeA\baseline\color.png` (luminance hp in the wake box) and verify it **FAILS** — a metric never run against the actual failure field has no standing to declare it beaten (review F1); (d) a 3.0 rc-spaced 3-billow chain → PASSES via the second interval.

---

### Task 6: Gate 1 — physics scan

**Files:** Create `<SPIKE>\run_gate1.py`

- [ ] **Step 1: Frozen cells.** A ∈ {10, 20, 45, 90} × δ ∈ {0.15, 0.30, 0.50}·RC (12) — **each in BOTH sheet forms** (deficit-jet and tanh-jump: they are different instabilities and one variant cell cannot catch a form×(A,δ) interaction; review F7) — plus shear-off control at (45, 0.30), plus **weak-shear cells at 0.5× and 0.25× ambient eddy-shear** at the best (A,δ) (a wall that exists only at current bracket strength has a cheap bracket-retune remedy — new decision row, spec addendum; review F6). **Every cell runs BOTH seeds** (`gate1`, `gate1_replicate`); pass = pass on both; fail = fail on both; split = MARGINAL (review F8). Run length from `transit_report` (fill + ≥2 transits, ≥ 3·DEV_STEPS). Ceiling flag: `|ω̄_amb + w_sheet|max > 60` → CONDITIONAL.
- [ ] **Step 2: Run** (background; corrected estimate: ~24–39 transforms/step ≈ 3–4 min/run → ~28 runs ≈ 1.5–2 h; review I6/11).
- [ ] **Step 3: Refinement check** on the best cell at DX/2 with **DT/2 and 2× steps** (`compute_dt` scales 1/res — unchanged DT at half DX doubles band-edge CFL past the RK4 limit and the "flip" would be a blowup; review B5). Verdict flip → record "coarse verdict unreliable" + report to the user; a full fine-grid rescan ONLY as a user-approved extension. This step + rung-1 of Task 8 jointly satisfy the spec's Gate-2 refinement item (review 12).
- [ ] **Step 4: Adversarial visual check** of passing panels vs the reference (flipped WEST), me + one agent.
- [ ] **Step 5: Branch.** All-fail (no MARGINAL-FAIL cells) + shear-off passes → physics wall → Task 10. Shear-off fails too → INCONCLUSIVE-INFRA. Wall vanishes at weak shear → bracket-retune candidate row → Task 10 (user decision). Only CONDITIONAL (above-ceiling) passes → report to the user BEFORE Task 7 (the production clamp would mutilate the sheet; proceeding is a declared extension; review 6). All passes MARGINAL → G1-fail-with-note, report before proceeding (review 5). Otherwise → Task 7 with the clean passing cells.

---

### Task 7: Gate 1.5 — production machinery

**Files:** Modify `<SPIKE>\environment.py`; create `<SPIKE>\run_gate15.py`

- [ ] **Step 1: Terms (exact math, cited; ceiling clamped BETWEEN nudge and injections and after hypervisc when present — omega_force.comp:157, :261; review M1):**
  - `nudge`: `w += (w_amb − w)/TAU_STEPS` per step (omega_force.comp:140; target = ambient only — the sheet is NOT in production's q_target). With the Dirichlet strip this is the ONLY relaxation downstream — "nudge kills" is unambiguous (review F3a).
  - `resupply` (**new variant, review F3b**): distributed per-step additive sheet re-supply `w += w_sheet/TAU_STEPS` over the whole band — the production-E2 forcing style. Separates mechanism (i) eddy damping out-competing growth (fix = q-relax-release lever) from (ii) sheet starvation of an un-resupplied layer (fix = keep the forcing on).
  - `wake_inject`: `w += WAKE_TURB_AMP · wmask · fbm4((x,y)·WAKE_FREQ, t)` — 2-D lattice value noise + time axis (production noise is 3-D; 1-D x-only noise would make y-constant streaks — review I7), 4 octaves, lac 2, gain 0.5, stream `fbm`, time advanced `EVOLUTION_RATE` per step (solver.py:878).
  - `psi_drag`: `w += PSI_DRAG·(ψ − ψ.mean(axis=1, keepdims=True))` (omega_force.comp:223-226).
- [ ] **Step 2: Run** on each clean Gate-1-passing cell (streams `gate15`, both seeds): {nudge}, {nudge+resupply}, {wake_inject}, {psi_drag}, {all}. Kill = the frozen Gate-1 criteria fail (binary). **Report the measured A(x) amplitude profile along the band in every nudge-on arm** — a kill without it cannot be attributed (review F3c).
- [ ] **Step 3: Branch.** Nudge kills but nudge+resupply passes → mechanism (ii): production remedy = sustained forcing, cheap. Nudge+resupply also kills → mechanism (i): q-relax-release lever candidate. Non-nudge term kills alone → named machinery-wall outcome (specific term) → Task 10. Gate 2 runs on the **maximal machinery subset that still passes** (review 7).

---

### Task 8: Gate 2 — numerics ladder

**Files:** Create `<SPIKE>\ladder.py`, `<SPIKE>\run_gate2.py`

- [ ] **Step 1: Cumulative rungs** (ω AND tr advected by the rung's scheme; production DT; streams `gate2`):
  1. `rk4_frozen`: **RK4** with ONE velocity per step (an Eulerian RK1 rung is unconditionally unstable and corresponds to nothing production runs — review B2). Recorded deviations: production stepping is semi-Lagrangian; production advects with the PREVIOUS step's ψ (solver.py:934-988 one-step lag).
  2. `sl_cr`: RK2 backtrace (omega_advect.comp:46-71, Cartesian) + Catmull-Rom (:84-112).
  3. `maccormack_unclamped` (:137-142). 4. `maccormack` + 2×2 clamp — BOTH production lines: revert-to-fwd on overshoot AND the final clamp of the result into [lo,hi] (:144-148; review M1).
  5. `+fd_vel`: centered-difference ψ→u (velocity.comp:42-48) + bilinear velocity in the backtrace (solver.py:272-274 `linear=True`).
  6. `+hypervisc`: `w += HYPERVISC·(−lap5(lap5(w)))·DX⁴/64` with **physically scaled** 5-point Laplacians (1/DX² per application — review M7; omega_force.comp:235-248).
  7. `+sor`: warm-started 48-sweep red-black screened SOR, flat 5-point (center −4/DX²−INV_LD2, sor_omega 1.7; poisson_sor.comp:41-66; sphere-metric deviation recorded — review: not verdict-flipping).
- [ ] **Step 2: Harness-credibility check.** Rung 7 + all machinery at E2's config (A=5, δ=0.17·RC) must reproduce the production failure (no rollup). Must pass before any attribution.
- [ ] **Step 3: Run + attribution (review F9).** Record CONTINUOUS degradation per rung (billow count, band-RMS, measured growth rate), not just the binary gate. First failing rung = candidate; one-out revert at rung 7 must RESTORE the pass. **If it does not restore → run the full leave-one-out sweep at rung 7** (7 runs, SOR-dominated ≈ 1–1.5 h): multi-term = no single revert restores but reverting ≥2 does; distributed = visible in the continuous metrics.
- [ ] **Step 4: Branch per the spec tree** (single-rung / multi-term / none) → Task 10.

---

### Task 9: Collateral tripwire (demoted from gate — review F11)

**Files:** Create `<SPIKE>\run_collateral.py`

- [ ] **Step 1:** Hero-free warm-analog (ambient + global-injection analog from the warm JSON, stream `collateral`, no sheet): rung-7 control vs winning candidate, 2·DEV_STEPS. **Report per-octave spectral deltas, filament-band RMS delta, max|ω| delta as DESCRIPTIVE numbers; >±50% anywhere = flag.** The authoritative collateral test is production-side renders + adversarial visual review at bake time (any candidate ships as a defines-gated default-off variant regardless).

---

### Task 10: Verdict + report

- [ ] **Step 1:** `<SPIKE>\VERDICT.md`: per-gate numbers, decision-tree row, the named binding constraint, the fetch→production-window mapping (Task 5 clocks), A(x) profiles, panels (adversarially reviewed), Gate 0b answer + frozen acceptance image if accepted. NOT committed; commit of harness+verdict+reference-measurement together only on user say-so.
- [ ] **Step 2:** Update memory (`wake-fidelity-investigation.md`).
- [ ] **Step 3:** Report to the user: decision-tree row landed, evidence, concrete next-step options. No production work without the user's explicit pick.

## Execution notes

- Order: 0 → 1 → 2 (STOP) → {3 parallel} → 4 → 5 (freeze) → 6 → 7 → 8 → 9 → 10. Tasks 4–5 may be WRITTEN while Gate-0a/smoke runs execute — no Gate-1 run before the Task 5 freeze.
- Corrected budget: spectral run ≈ 3–4 min; full Gate-1 scan (≈28 runs incl. both forms/seeds/controls) ≈ 1.5–2 h; SOR rung ≈ 2× spectral; LOO escalation ≈ 1–1.5 h. All inside the 3-day box with slack.
- Any infrastructure grind exceeding its slot → STOP → INCONCLUSIVE-INFRA with what's green.

## Plan-review record (rev 1 → rev 2)

3-agent adversarial plan review 2026-07-19 (code/numerics, spec-fidelity/process, experiment-design) — all NEEDS-CHANGES; every BLOCKING and IMPORTANT finding folded in: unstable tracer/rung-1 integrators → RK4 (code B1/B2); Gate-0a wavelength 2× error and screening fit range (code B3/B4); refinement DT rule (code B5); import/attr corrections `gasgiant.params.presets` + `P.solver.coriolis_f0` (code I1, spec-fid 1); state dealiasing, screened-energy conservation test, rms0 conditioning, formation sampling, runtime ×6, 2-D fbm, CFL guard (code I2–I8); status-quo-must-fail metric + max-size bound + wavelength-coupled count + two-interval bracket (exp F1, spec-fid 2/3); transit-time clocks + fetch-based formation + ≥2000-step robust stationarity (exp F2/F10, code M9); Dirichlet strip + resupply variant + A(x) reporting (exp F3/F4); per-step rate convention + mean-hold subsumption (exp F5, code M8/M10); weak-shear cells + new decision row (exp F6); both-forms scan (exp F7); two-seed symmetric replication (exp F8); LOO escalation + continuous degradation (exp F9); collateral demoted to tripwire (exp F11); per-row spectral wavelength (exp F12); MARGINAL/CONDITIONAL/non-nudge-killer routing, Gate-0b halt + frozen acceptance image, adversarial-review coverage, env-constant freeze, seed streams, DEV_STEPS from preset (spec-fid 4–19).
