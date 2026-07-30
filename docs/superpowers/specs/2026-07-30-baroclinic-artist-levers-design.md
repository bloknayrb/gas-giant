# Baroclinic artist levers — design

Date: 2026-07-30
Status: awaiting approval
Supersedes: `2026-07-29-baroclinic-growth-and-phase-design.md` (deleted — its central
proposal was falsified; the surviving findings are carried into §2 below)

## 1. Problem

With `solver.baroclinic.enabled` on, `gas_giant_warm` renders a mechanically regular
sawtooth comb along the band edges: ~14 near-identical crests, evenly spaced, all the
same height, with their tops and roots sharing horizontal lines pole-to-pole.

Root cause, measured: **the injected source is the initial seed, never a grown wave.**
Eddy interface variance decays monotonically from 15.32 to 0.0891 over 40,000 steps at
the shipped configuration. What reaches the Poisson RHS is therefore the seeding block
of `sim/shallow_water_ref.py`, and that block has exactly two properties the eye reads
as mechanical:

1. **One wavelength.** `cos_lam = np.cos(m_zonal * lam + phase)` plants a single zonal
   mode. There are no sidebands, so every crest is the same width and the same distance
   from its neighbours.
2. **One phase for the whole globe.** That expression is broadcast with
   `[None, :] * np.ones((H, 1))`, so every latitude row receives the identical scalar
   phase and all crests align vertically.

These are independent defects with independent fixes, and neither is a tuning error.

`resample_to_equirect` normalises the source to unit standard deviation, which is why
the amplitude collapse is invisible to every production consumer, and why
`test_production_config_is_stable_and_coherent` passes — it asserts only no-outcrop and
a dominant mode in [10, 20], both of which a decaying seed satisfies.

## 2. What was falsified (record)

Four separate attempts to make the mode actually grow. All are closed by measurement.
They are recorded here so they are not re-proposed.

**`gp2` 0.075 → 0.3** (the reverted PR #3 value). Falsified three ways: worst-seed
outcrop at 11,473 steps against the 11,300 driver steps `gas_giant_warm` actually
consumes (`warmup + ceil(dev_steps / update_every) * baro_steps_per_update`), leaving a
173-step margin, with `ember_dwarf` and `neptune` outcropping mid-run; emergent mode
m=7–8, the feature size PR #3 deliberately moved away from; and the accompanying
`SMOOTH_SIGMA` 1.26 → 2.54 turns out to be load-bearing gate-masking — at 1.26 the
source is dominated by m=44 with 62% of its power at m≥26 and `assert_coherent` raises.

**Raising `xi`.** Exhausted. The apparent ceiling is a base-state construction artifact,
not physics — `_balanced_sheared_base` sets `h1_prof = (H1_mean + H2_mean) - h2_prof`
and the h2 swing scales with xi, so the *upper* layer is negative before a step runs
(h1 min at build: 184.1 m at xi=3, −3921.2 at xi=4, −12131.9 at xi=6). Raising
`H1_mean` removes the artifact and lets xi=6 and xi=9 run clean — and every such run
still decays (`GROWTH=1`). So the failure is scale, not rate.

**Source-grid resolution.** The growth criterion is real: the mode only turns around
once L_D exceeds ~2.8 grid cells, which 192×96 and 288×144 do not reach and 384×192
does. But the thing that grows is **not the baroclinic mode**. Tracking the spectrum
along the 384×192 trajectory, the seeded m=14 decays continuously from 67% of the power
to 1.8% and never grows at any sample, while grid-scale power (m ≥ W/8) rises from 0.136
to 0.862 and the dominant mode becomes 91 — near Nyquist at W=384. At 512×256 the same
failure arrives 2.2× earlier and 1540× stronger, with a dominant mode of 119. The two
grids disagree completely. This is the C-grid computational mode, which the codebase
already documents as unfixable by velocity hyperviscosity.

**Polar zonal filter** (proposed to make the resolution path affordable). Dead. The
premise — that `dt` is throttled 31× by the polar zonal CFL of cells carrying no physics
— is wrong. Unfiltered timestep multipliers at 384×192 measure faithful to 1.25×,
drifting at 1.5×, and diverging 5.7× at 2.0×. `dt_safety = 0.30` is already at the
accuracy limit, so there is no headroom for any filter to reclaim.

**Consequence.** The physical mode does not grow at any resolution, any `xi`, or any
affordable `gp2`. This also explains the 2026-06-28 `jupiter_baroclinic` drop better than
the roadmap note does: the comb reads mechanical because the source is structurally a
decaying stamp, not because it was mistuned.

The existing `assert_coherent` gate (`M_GATE_MAX = 20`) would reject the grid-scale
output, so none of the above could have shipped silently. That gate is working.

## 3. Design

Seven levers, all defaulting to a no-op, plus one bug fix. Nothing here depends on
growth.

### 3.1 Two seeding levers

Both edit the seeding block only, both draw from their own named `subseed()` stream so
the broadband `noise` realisation stays bitwise fixed and each lever is an isolated axis.

| field | default | effect |
| --- | --- | --- |
| `phase_jitter` | 0.0 | per-latitude phase offset added to the seeded mode |
| `spectrum_width` | 0 | seeds `m_zonal ± K`, Gaussian-tapered, independent phases, renormalised to the same total variance |

Measured on the shipped 192×96 grid at `gp2 = 0.075`, warmup 8000, noise held fixed:

```
phase_jitter   0.0    0.5    1.0    2.0    3.0
R (alignment) 0.961  0.925  0.742  0.226  0.202
dominant m      14     14     14     14     14

spectrum_width  K=0    K=1    K=2    K=3    K=4
dominant share 0.915  0.834  0.442  0.302  0.228
crest CV       0.103  0.104  0.136  0.165  0.180
R (alignment)  0.962  0.968  0.963  0.960  0.961
dominant m       14     14     14     14     14
```

They are orthogonal: jitter moves alignment and leaves the spectrum alone; the packet
moves the spectrum and leaves alignment alone. Both preserve feature size. Confirmed at
native 4096 on `gas_giant_warm`: jitter alone individuates the crests but keeps one
rhythm; the packet alone varies the spacing but keeps the shared baseline; together the
band reads varied in both spacing and height. `(0, 0)` verified bitwise identical to
today across `h1, h2, u1, v1, u2, v2`.

### 3.2 Five sliders promoted from hardcoded constants

None of these is adjustable today at any level of the API.

| field | constant | today | note |
| --- | --- | --- | --- |
| `latitude` | `_PHI_TEST_DEG` | 45.0 | the effect can only ever appear at 20–70°N |
| `width` | `_BAND_HALFWIDTH_DEG` | 25.0 | band half-height |
| `eddy_scale` | `GP2` | 0.075 | feature size (L_D) |
| `zonal_count` | `M_ZONAL` | 14 | seeded features around the planet |
| `smooth` | `SMOOTH_SIGMA` | 1.26 | fine structure surviving into the source |

`eddy_scale` needs a validated upper bound below the outcrop cliff, and `smooth` a lower
bound that keeps `assert_coherent` satisfied; both bounds come from §2's measurements and
are enforced via `pfield(lo=, hi=)`, which map to pydantic `ge`/`le` on a strict model.

#### 3.2a Two construction defects, found by rendering

Neither was anticipated here. Both were found by driving the levers through the real
facade path rather than by measuring the seeding maths, and both made `latitude`
degrade the feature to off partway along its own declared range. They are the same
class of bug as the `xi` ceiling in §2 — an apparent physics limit that is really the
base-state builder clipping a layer.

The interface swing is `A = xi*H2_mean/tan(latitude)`, so it diverges as the band moves
equatorward: it nearly doubles between 45° and 28°.

1. **Upper layer.** `h1 = H1_mean - A*cumint`. At build, before a single step, 45°
   clears the floor by only 184 m of 12,500 and 35°/28°/20° clip outright. Fixed by
   deepening the upper layer instead of clipping it, only when it would otherwise clip.
   The realized total depth is then read back off the built state, because
   `H1_mean + H2_mean` is no longer it and `L_D` and the growth-rate diagnostics
   consume it.

2. **Lower layer, at the equator.** Deepening the upper layer does not help a band that
   straddles f = 0, where the source proxy `zeta = (gp2/f)*lap(h2e)` also diverges.
   Measured region at a 2000-step warmup: every `(latitude, width)` fails once `width`
   reaches `latitude`, and every `width <= latitude - 5` is clean across 10–75°. This is
   a CROSS-FIELD constraint, so no single-field `lo`/`hi` can express it. Resolved by
   clamping the effective half-width off the equator — the physically honest answer,
   since a geostrophic band cannot cross f = 0 at all — with the clamp reported through
   `validation_warnings` rather than applied silently. The rule lives in the params
   layer because params may not import `sim` and `validation_warnings` needs it.

One effective width feeds both the seeding envelope and the source mask; had they
disagreed, the mask would clip the storms it exists to pass.

3. **The coherence gate did not follow the band.** `dominant_zonal_m` samples a
   fixed row window — latitudes 53.4–15.9°, sized for the original hardcoded
   45 ± 25 band. Once `latitude` is a slider that window is simply the wrong
   place: a band at 75 ± 8 spans 67–83° and lies *entirely* outside it, so
   `assert_coherent` graded empty rows and returned m=1 on a band that was
   perfectly clean. This masqueraded as a physics failure and nearly cost a
   correct part of the range — the "bad corner" disappeared the moment the metric
   was fixed. `assert_coherent(..., in_band=True)` selects rows by amplitude;
   the default-band verdict is unchanged.

   The general lesson is sharper than the specific bug: **a metric calibrated for
   a fixed configuration silently becomes a measurement of the wrong thing when
   you make that configuration adjustable.** Every diagnostic that carries a
   hardcoded window needs re-examining in the same pass as the lever that moves
   what it looks at.

`latitude`'s equatorward bound of 20° comes from what survives once the metric is
trustworthy: the worst-width share of zonal power at the seeded m=14 runs 0.94 at
75°, 0.81 at 45°, 0.48 at 20°, collapsing only at 15° (clamped width 10 reads m=2
at 0.045). Measured through the FIXED window the same sweep condemns 20° (spurious
m=43) and 25° (0.24) — so the broken metric would have cost a third of the usable
range on top of the corner it falsely failed. The coherence gate cannot catch that failure — it rejects grid-scale
sources, and a washed-out band is large-scale — so the bound carries it, and the
corner test asserts both the gate AND that the dominant mode is still the seeded
one.

### 3.3 Bug fix: silent mid-run outcrop

`BaroclinicSourceDriver.advance` catches `PositivityViolation`, logs, latches
`self.outcropped` and returns. The facade's handler in `_update_baroclinic_source`
therefore never fires, `baroclinic_status` keeps reporting `active`, and the source
freezes on its last good state — reinstating the static stamp the driver exists to
replace. Surface it as `degraded` with a reason, matching the warmup path, which already
raises `BaroclinicWarmupError` and is toasted by `app/main.py`.

Latent at the shipped constants (they never outcrop) but live for any artist who raises
`eddy_scale`.

### 3.4 Wiring

`BaroclinicSourceDriver.__init__` hardcodes `bsrc.SRC_W`, `SRC_H`, `GP1`, `GP2`, `XI`,
and `current_source` hardcodes `SMOOTH_SIGMA`; `geostrophic_vorticity_source`'s
`lat_band` and `taper` parameters are dead defaults the driver never passes. All seven
levers thread through the driver constructor.

The facade cache key is `(w, h, bp.warmup_steps, self.params.seed)` — it omits every new
lever, so moving a slider would silently reuse a stale driver. All seven join the key.

The cache test must use the one-`Simulation` `update_params` pattern
(`tests/unit/test_m3_ship.py:162`). A fresh-`Simulation` test structurally cannot detect
a stale key, because `_baro_driver` is initialised in `__init__` and `release()` does not
clear it, so the cache only ever hits on the RESTART rebuild path.

### 3.5 Correct the `enabled` description

It currently reads "grown by a baroclinic instability model". Nothing grows. The
description is the artist's tooltip and the search haystack, and it must not assert a
mechanism that measurement has refuted.

## 4. Byte-identity strategy

All seven levers are CPU/numpy, so this is structural guarding, not GLSL variants.

`spectrum_width = 0` and `phase_jitter = 0.0` take the untouched expression through an
`if K > 0:` / `elif jitter != 0.0:` structure that preserves the existing statement's
exact shape — float non-associativity means an algebraically equivalent rewrite is not
sufficient. The five promoted constants are byte-identical by construction when left at
their current values, since they become defaults of the same numbers.

Binding gates: `tests/unit/test_m3_ship.py:67` and
`tests/unit/test_m3_external_source.py::test_off_path_is_a_noop`. The p05 render-hash
gate is **vacuous here** — no preset enables baroclinic — and must not be cited as
evidence.

## 5. Testing

Metrics. `R` computed on the argmax mode is not trustworthy: it reads 0.991 at
`SMOOTH_SIGMA = 1.26` versus 0.702 at 2.54 for the same state, because it reports the
phase of whichever mode wins, which can be the grid artifact. Tests use instead:

- cross-latitude phase concentration at a **fixed** seeded `m`, amplitude-weighted;
- power-weighted spectral width about the centroid;
- coefficient of variation of longitudinal crest spacing.

Per lever: one behaviour test asserting the intended metric moves and the orthogonal
metric does not, plus one no-op test asserting bitwise identity at the default. Plus a
driver test that a mid-run outcrop surfaces as `degraded`, and a facade test that each
lever participates in the cache key via `update_params`.

## 6. Blast radius

- Nine preset JSONs regenerate — `PlanetParams.to_json` uses `model_dump_json` with no
  `exclude_defaults`, so every preset carries the full `baroclinic` block. Build scripts
  carry a `load == save` diff-guard.
- `tests/unit/test_description_rubric.py` field counts.
- `tests/unit/test_facade_baroclinic_status.py:62` stub driver signature.
- `tests/unit/test_baroclinic_params.py:36-40` no-rand tuple.
- `docs/sliders.md` text regeneration plus GPU-rendered images; CI runs the generator's
  `--check` drift gate.
- Each new `description` must meet `test_description_rubric.py` and the terms pinned by
  `test_description_findability.py`.

## 7. Scope boundaries

Not in scope, with the evidence in §2: the `gp2` revert, raising `xi`, source-grid
resolution, and the polar filter.

**Framing.** This makes the output *look less mechanical*. It does not make it
physically correct — the source remains a decaying seed, and these levers make that seed
irregular rather than regular. The documentation must say so plainly rather than claim
naturalism.

Deferred, unverified: whether the rendered effect is invariant between a 1024 preview and
a 4096 export. The resolution-invariance work is nudge-dominated only and never covered
baroclinic injection into the Poisson RHS.

No preset bakes any of these levers under this design; every default is a no-op.

## 8. Open questions for review

1. Should `docs/roadmap.md` gain a falsification entry for §2, alongside the existing
   dead-end record? It closes a line that has now been attempted twice.
2. `spectrum_width` as an integer `K` (modes `m ± K`) versus a continuous spectral
   width. The integer form is what was measured.
3. Whether `eddy_scale` should ship at all, given that its safe range is bounded by an
   outcrop cliff whose margin depends on the *driver step budget*, which in turn depends
   on `warmup_steps`, `update_every`, `baro_steps_per_update` and the run length — a
   coupling no single slider bound can express. §3.2a resolved the same shape of problem
   for `latitude`/`width` by clamping rather than bounding; the same treatment may be
   right here, but it has not been measured.
4. Which values to bake, if any. Rendered at 4096 through the real facade path,
   `phase_jitter 1` with `spectrum_width 2` keeps the crisp discrete festoon character
   that is the one good thing about today's output, while breaking both the crest
   alignment and the wavelength monotony. `2` with `4` goes too far — the band edge
   softens and the plumes stop reading as discrete storms. Nothing is baked under this
   design; every default is a no-op.

## 8a. Measured usable range (band-aware gate, 8000-step warmup, seed 0)

Share of zonal power at the seeded m=14, over the rows carrying the band. Every
cell reports dominant m=14 unless noted.

```
lat \ width      8       25       40
  15          0.496   m=2 0.045   m=2 0.045     <- clamped to width 10; collapses
  20          0.763    0.483     0.483
  25          0.799    0.766     0.766
  30          0.891    0.774     0.774
  45          0.853    0.895     0.888
  60          0.681    0.654     0.497
  75          0.591    0.757     0.834
```

`latitude`'s bound of 20 is the last clean row. Measured through the FIXED window
the same sweep reads 20 as a spurious m=43 and 25 as 0.24, and reports m=1 with an
undefined share at 75/8 — it under-reads everywhere (45/25 reads 0.844 against the
true 0.895), so it was not merely wrong at the extremes.

## 9. What rendering caught that measuring did not

Worth recording as a working note, because it recurred twice in one pass. Every defect in
§3.2a was invisible to the source-grid measurements and to the unit tests, and appeared
immediately on the first render through the real `params -> facade -> driver` path. The
metrics answered "does the lever move the field it claims to" correctly and completely,
and had nothing to say about "does the lever work across the range it advertises".

Both were surfaced by §3.3 — the outcrop fix. Under the previous swallow-and-latch
behaviour the driver would have reported `active` with a frozen source, and a dead slider
would have shipped looking like a working one.
