# Hero jet-environment authoring — design

**Status:** design approved (brainstorm 2026-07-19), 3-lens adversarial review incorporated.
Implementation deferred behind the chirality PR (see Sequencing).

**Topic:** give the artist deterministic control of the hero storm's local jet
environment — (1) a diagnostic **seat meter** and (2) a deterministic **jet
override** — replacing the additive `local_jet`/`local_jet2` bracket.

---

## 1. Problem & motivation

The hero storm (Great Red Spot analog; a southern-hemisphere anticyclone at
~−22°) needs a two-sided jet **bracket** — a westward jet on its equatorward
flank and an eastward jet on its poleward flank — to seat it in an anticyclonic
shear and open a **moat** (band deflection around the oval). The moat / flow
deflection is a repeatedly-documented MISSING feature (`docs/realism.md`: "band
edge runs straight THROUGH the storm (no deflection/moat)").

The current mechanism (additive `local_jet` + `local_jet2` gaussians in
`build_profiles`) **fights the seeded band jets**: `build_profiles` plants one
seeded jet per band edge (alternating sign, seed-jittered amplitude
`0.55·(1+0.5·U[−1,1])`). On warm the band edge at −19.41° carries a seeded
*eastward* jet (+0.30…+0.60 across seeds) that opposes the westward north bracket
jet, so the effective north jet swings **0.294** across seeds. The developed
drift after 700 steps stays robust (~0, absorbed by evolution + the emergence
anchor), but the artist has no deterministic control at the profile level.

## 2. Goals / non-goals

**Goals**
- Deterministic bracket **shear** at the hero, seed-independent, at any latitude.
- A GUI readout that tells the artist whether the natural jets already give a
  good bearing at the hero's chosen latitude ("do I need the override here?").
- Replace `local_jet`/`local_jet2` with one coherent `hero_bracket_*` block.
- Default-off byte-identical (at default values).

**Non-goals (explicit)**
- **No auto-snap / placement optimizer.** Bearing quality ≠ GRS look; the best
  natural bearing on warm is ~−40° (a temperate-lace band regime), and an
  optimizer would steer artists away from the iconic −22°. The meter is a
  *diagnostic at the artist's chosen latitude*, never a mover.
- **Not tuned on "rolling."** Rolling is a motion cue a still texture can't show;
  acceptance is judged on the **rendered static moat vs the reference**.
- No support for seeded/unpinned heroes (the override no-ops without a pinned
  hero — same gate as `local_jet_hero_offset`).

## 3. Design

Layering (respects the import contract
`params|palette → gl → core → sim → render → jobs → export → engine → app|cli`):
the metric and the override are pure functions in `sim/profiles.py`; the GUI
reads the metric via `engine/facade.py`; the app renders the meter.

### 3.1 Shared metric — natural bearing quality (moat-oriented proxy)

Pure function of the **bracket-off** profile, at a candidate latitude `L` with
storm half-extent `DLAT` (≈ `r_core` in degrees):

- `u_north = u(L + DLAT)` (equatorward rim) — want westward (`u < 0`)
- `u_south = u(L − DLAT)` (poleward rim) — want eastward (`u > 0`)
- `two_sided = min(−u_north, u_south)` — both signs present and strong
- moat-orientation weight: scale by jet **strength relative to storm radius**
  (a sign-only score can read "great seat" while opening no moat — the moat
  depends on how hard the shear bends flow around an `r_core`-sized oval).
- `center = |u(L)|` (drift penalty)
- `quality = weight · two_sided − 0.5 · center`

The sign convention respects the hero's authored spin (default anticyclone;
flips for a cyclonic hero). **Naming discipline:** the equatorward rim of a
southern hero (−22°) is *less negative* (≈ −19°) — name variables by role
(`equatorward_rim` / `poleward_rim`), not by absolute compass, to avoid the
codebase's sign-landmine class.

**Honesty caveat (built into the readout):** the metric is measured on the
INITIAL profile (in vorticity mode, the omega nudge *target*), but the storm's
look emerges after ~700 steps, and the developed velocity-zero sits ~1.8°
poleward of the profile prediction (plus amplitude erosion from hyperviscosity).
The GUI labels the reading **"pre-development proxy"**; the value is a coarse
green/amber/red band, not a false-precision decimal. (Optional refinement: a
measured constant developed-shift correction, validated against a few developed
renders — deferred to calibration.)

### 3.2 Tool 1 — seat meter (diagnostic, no snap)

- `engine/facade.py` exposes `seat_quality(lat)` / a `seat_scan()` over latitude,
  computed from the **bracket-off** profile (so it always reports the *natural*
  bearing, even when the override is on).
- The app shows a live green/amber/red readout at the current `hero_latitude`,
  with a one-line suggestion ("natural bearing poor here → enable hero_bracket").
- **No button that moves the storm.** (Auto-snap rejected in review: optimizes
  the wrong objective, is circular via hero-coupled offsets, and fires an
  expensive RESTART re-develop.)

### 3.3 Tool 2 — jet override (carve-and-impose)

In `build_profiles`, replace the seeded jets in a feathered hero-centered window
with an authored bracket:

```
u_final = u_base·(1 − w(lat)) + (pedestal + bracket(lat))·w(lat)
```

- `w(lat)`: **C1 smoothstep** — 1 within `window`° of the hero, smoothstep to 0
  by `window + feather`°. Smoothstep (not a linear feather) is REQUIRED: its
  zero derivative at both ends keeps `du/dφ` continuous across the window, so
  `omega_jet` has no delta-function vorticity spike. (Verified in review.)
- `pedestal`: a **flat** level = `u_base(hero)` — NOT a sloped ramp. A ramp
  reintroduces a seed-dependent background shear that shifts the shear-zero off
  the hero center; a flat pedestal keeps the authored bracket's zero-crossing at
  the hero and minimizes residual seed-dependence.
- `bracket(lat) = strength · (north·gauss(lat; hero+north_offset, north_width)
  + south·gauss(lat; hero+south_offset, south_width))`.

**Ordering (verified in review):** apply the override **after** `u *= strength`
(line ~136), with `strength` baked into `bracket` and `pedestal`, so both
components scale once and a later `jets.strength` retune rescales them
consistently. Keep the override **before** the `omega_jet` computation
(line ~158) so the omega LUT reflects the carved `u`. `polar_fade` (line ~137):
inert for a −22° hero (`polar_fade ≈ 1`); documented LIMIT for a high-latitude
hero (the window would be unfaded) — fold `polar_fade` into the override if a
polar hero is ever wanted.

**Byte-identity (verified in review):** the `!= 0` structural guard must wrap the
**entire** override, pedestal blend included. With `north = south = 0` the
bracket is zero, but the pedestal-blended form is NOT equal to `u_base` inside
the window (it flattens the seeded jets), so guarding only the bracket term would
break byte-identity. The whole block is skipped when `north == 0 and south == 0`
(mirrors the existing `if jets.local_jet_speed != 0.0:` skip). This is a CPU/numpy
lever: byte-identity here means "default values skip the code path," NOT the GLSL
preprocessor-variant mechanism.

**Determinism claim (corrected):** "seed-independent bracket **shear**," not
"seed-independent profile." The prototype's 0.294→0.005 collapse measures the
bracket *increment* (differencing cancels the background by construction); the
real win over the additive approach is that the seeded −19.41° jet is now
*erased* inside the window instead of *opposing* the bracket.

### 3.4 Params — `jets.hero_bracket_*` (replaces `local_jet*`)

Flat prefixed fields (matching the `local_jet_*` precedent; GUI auto-panels them
under one "Hero Bracket" group), all **RESTART** tier (a VELOCITY rebuild must
not flip ambient shear under a stale storm rotation), **no `rand`** (geometry/
offset levers must not be seeded-randomized — deliberate omission, state it):

| param | default | meaning |
|---|---|---|
| `hero_bracket_north` | 0.0 | westward jet strength, equatorward flank (negative) |
| `hero_bracket_south` | 0.0 | eastward jet strength, poleward flank (positive) |
| `hero_bracket_north_offset` | +3.0 | deg equatorward of hero |
| `hero_bracket_south_offset` | −3.0 | deg poleward of hero |
| `hero_bracket_window` | 4.0 | full-strength half-width (deg) |
| `hero_bracket_feather` | 5.0 | smoothstep feather beyond full (deg) |
| `hero_bracket_north_width` / `_south_width` | 0.05 | jet gaussian widths (rad) |

Default `north = south = 0` → override skipped → byte-identical. Requires a
**pinned hero** (`hero_latitude` set AND `hero_count > 0`), threaded in as
`hero_lat_deg` at both `build_profiles` call sites (`facade._build` RESTART +
`update_params` VELOCITY rebuild — already wired; no stale-latitude teleport,
since `hero_latitude` is RESTART-tier). Calibrated warm bake target (from the
additive prototype): north −1.0 @ +3°, south +0.6 @ −3° (developed drift ~0).

## 4. Byte-identity & the deliberate warm re-bake

"Default-off byte-identical" holds **at defaults only** — enforced transitively
by the kinematic p05 render-hash (covers saturn_pale/jupiter_like/ice_giant,
none of which use these levers). There is **no pinned hash directly on
`build_profiles`**; the CPU no-op is proven by an `np.array_equal` capture
(precedent: `tests/unit/test_local_jet2.py`).

warm ships a *non-default* baked jet, and carve-and-impose ≠ additive, so
**warm's developed output WILL move** — a **deliberate re-bake** with a new p05
baseline hash + re-pin, stated up front. This is NOT a byte-identity violation to
tolerance-split away; it is an intended calibration change (warm is vorticity
mode, not byte-identity-gated; the `_dev0_omega` capture pins storm seeding, not
the jet profile, so it is not tripped).

## 5. Migration (atomic — one commit)

Removing the 8 `local_jet*`/`local_jet2*` fields under strict models
(`extra="forbid"`) hard-errors on load for **all three** vorticity presets
(warm, jupiter_vorticity, neptune all serialize `local_jet_speed`). Atomic set:

- Regenerate `presets/gas_giant_warm.json`, `jupiter_vorticity.json`,
  `neptune.json` (the JSONs must load under strict models post-rename).
- Update `scripts/build_warm_preset.py`, `build_neptune_preset.py`
  (`validate_assignment` errors otherwise); the build scripts' load==save
  diff-guard catches a mismatch. **Resolve jupiter_vorticity provenance** — it
  carries `local_jet_speed: 0.0` but `build_vorticity_presets.py` does not set
  it; find/own the source so regen is clean.
- Replace (don't just delete) `tests/unit/test_local_jet.py` and
  `test_local_jet2.py` with `test_hero_bracket.py` coverage.
- Regenerate `docs/sliders.md` (`render_slider_examples.py --no-render` +
  `--check`; CI drift gate is blocking).
- **Checkpoint incompatibility:** on-disk `.npz` that serialized the old params
  become unresumable (strict models, no graceful path). Acceptable on a solo-dev
  branch, but record it — bump `GENERATION_VERSION` or add a key-migration shim
  so the break is explicit, not a silent hard-error.

## 6. Testing

1. **CPU no-op:** `np.array_equal` of `build_profiles` output at default vs a
   saved baseline (pattern: `test_local_jet2.py`).
2. **Deterministic-across-seeds behavior:** bracket seats the anticyclonic shear
   at the hero across ≥3 seeds (replaces the `_ambient_sign` assertions).
3. **Seat-metric unit test:** monotone/correct against a known-good and
   known-bad seat (pure function, headless).
4. **Continuity guard:** `du/dφ` (hence `omega_jet`) continuous across the window
   — assert no spike at the feather edges (the smoothstep property).
5. **Migration diff-guard:** build-script load==save for all 3 presets; assert
   the JSONs load under strict models post-rename.
6. **Intentional re-bake:** new p05 baseline hash for warm, committed
   deliberately (NEVER a tolerance widen).
7. **Layering:** `uv run lint-imports` (metric in sim, exposed via facade; the
   app must not import `sim.profiles` directly — import-linter catches it).
8. **Facade-level GUI test:** the meter value + suggestion are testable via the
   facade method; pixel/interaction left untested (stated).

## 7. Open calibration items (PR-1/PR-2 calibration, not blockers)

- **Bald-stripe render test:** ≥4K, detail on, override ON vs OFF — compare
  band-pass texture energy of the ±(window..window+feather)° annulus against the
  two adjacent belts. Carving the seeded jet shear (which folds belt texture)
  could render the annulus smoother than its neighbors — a smoothness
  discontinuity at the feather edge is the failure signature. If it appears,
  re-inject belt shear in the annulus rather than nulling it.
- **Moat acceptance:** judge the rendered static moat/deflection around the oval
  vs the reference GRS moat (reference-anchored review, per standing rule).
- **Developed-shift:** decide correction-constant vs label-only for the meter.
- **Feather-seam across seeds:** verify the deterministic bracket meeting the
  seed-varying background at the feather edge hides the transition (no repeatable
  seam tell across seeds).

## 8. Sequencing & PR decomposition

1. **Chirality PR first.** Land the spin-sign fix as-is (reviewed, baked, green).
   **Strip the abandoned additive `local_jet2` lever** from the branch (commits
   `2b433b4` + `61ce038`, the branch tip — clean drop; keep the single `local_jet`
   that gives the chirality fix its anticyclonic jet). This keeps the merge-base
   diff auditable and lets the metric be calibrated on the settled
   post-chirality world (resolves the "calibrated on an in-flight sign fix"
   hazard).
2. **hero_bracket PR-1** (against merged master): `hero_bracket_*` params +
   carve-and-impose override + preset migration + deliberate warm re-bake. All
   the byte-identity / migration / re-bake risk lives here.
3. **hero_bracket PR-2:** seat metric façade method + GUI meter. Zero
   byte-identity risk; independently reviewable.

## 9. Review record

3-lens adversarial review 2026-07-19 (sim correctness, process/gates/migration,
visual/UX efficacy). All findings incorporated:

- **Sim:** `*= strength` ordering (apply post-strength, strength baked in);
  full-block `!= 0` guard (pedestal included) for byte-identity; flat pedestal
  not sloped ramp (shear-zero placement + determinism); C1 smoothstep required
  (no omega spike); keep before `omega_jet`; polar_fade LIMIT; determinism claim
  scoped to "bracket shear."
- **Process:** don't bundle onto the chirality branch → sequence + 2-PR split;
  "byte-identical at default only" honesty + deliberate warm re-bake; atomic
  migration of 3 JSON + 2 build scripts + 2 test files + sliders + checkpoint
  break; CPU no-op test pattern; deliberate `rand` omission.
- **Visual/UX:** drop the auto-snap (optimizes bearing not look; −40° trap;
  circular; RESTART cliff) → meter-only diagnostic; re-anchor payoff on the
  static moat, not rolling; metric weighted by moat-orientation not sign-only;
  profile-vs-developed honesty (pre-development label); bald-stripe annulus
  render test; validate sign conventions post-chirality (resolved by sequencing).
