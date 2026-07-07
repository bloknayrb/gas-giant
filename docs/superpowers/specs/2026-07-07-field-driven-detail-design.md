# Field-driven detail placement — design & implementation plan

*Date: 2026-07-07. Status: REV 3 (post round-2 adversarial verification).
Author: Claude (with bloknayrb).*

## Round-2 verification outcomes folded in (traceability)

Round-1 findings N1–N14 are all RESOLVED in REV 2/3 except where a round-2 item re-opens
them below. Round-2 (verification of REV 2) surfaced these; all folded into REV 3:

| ID | Finding (severity) | Resolution in REV 3 |
|----|--------------------|---------------------|
| M1 (arch New-1) **CRITICAL** | "Velocity-only cache invalidated on VELOCITY/RESTART only" is wrong — velocity re-bakes on EVERY sim step (dev run, `extend_run`, `_ADAPT_STEPS`). Cache goes stale during development. | Activity-dirty is tied to the **same `_tracers_changed`/tick signal** that already forces a re-derive, NOT `diff_tiers`. Rebuild activity whenever `_tracers_changed OR _post_dirty-with-velocity-change`. The ONLY cache-reuse case is a pure POST edit on a developed, **unstepped** sim (`_post_dirty` set, `_tracers_changed` clear) — the fast-edit-loop scenario. |
| M2 (arch New-2) **HIGH** | Scalar means `mean_S/mean_ω` not captured in `ExportSnapshot`; "store on synth" reintroduces export tearing at the scalar level (preview's `build()` overwrites them mid-export). | `build()` **returns** the means (never stores on the synth). `ExportSnapshot` gains `mean_strain`/`mean_vort` fields set in `capture()`. `_derive_tile` reads `snap.mean_*` and passes to `synthesize`. Means are snapshot-scoped exactly as the texture is. |
| M3 (vis NEW-1) **HIGH** | Raw strain = `shearN(lat) + eddy`; the global-mean self-disable gate fires at full strength on strong-jet **jupiter_like**, and the zonal-mean component can swamp the eddy signal on the vorticity targets → the crux becomes uninterpretable ("did strain-driving not help, or did the zonal mean drown the eddy?"). | **Build eddy-strain in v1.** The reduction emits per-latitude-row means; the driver uses `S_eddy = max(S − rowmean(S), 0)` (and the self-disable gate keys on the **eddy** mean, not total mean). Nearly free — the per-row reduction is a superset of the global reduction (O2r). |
| M4 (glsl Nnew1 + vis NEW-2) **MED-HIGH** | The fill floor `max(smoothstep(fill_n), fold)` sits OUTSIDE the partition → reintroduces double-stacking; and a too-high `FILL_LOD` low-pass spreads into "fill everywhere" = the stipple we're removing. | Fold fill into the **partition axis**: `place_axis = max(strain_n_eddy, FILL_W·fill_n)`; compute cell/lace/fold from `place_axis` so a filled interior moves cell→lace→fold coherently (sum stays 1). `FILL_LOD` conservatively LOW (2–3 mips); default fill = low-pass **strain**, not `|vort|` (which re-bands along jets). |
| M5 (glsl S1) **MED** | `field_vort_influence` in the variant predicate creates a wasted variant (does nothing at `drive=0`). | Predicate = `{field_drive}` ONLY; `EXPECTED_FIELD_DRIVE_LEVERS = {"field_drive"}`. `u_field_vort` stays in the tripwire's explicit uniform list (a known variant uniform, not a selector). |
| M6 (glsl S2) **MED** | Guard re-key (N12) missed the mottle `if (aw > 0.0)` early-out → lace silently skipped outside 30–63° even where strain wants it. | Re-key `if (aw > 0.0)` → `if (aw > 0.0 || lace*drive_eff > 0.0)`. |
| M7 (arch New-3) **MED** | `_release_sim` releasing `_activity_tex` without nulling → use-after-release on next RESTART (`_get_detail_tex` is NOT a true mirror — `_detail_tex` is never released). | Release AND `self._activity_tex = None` in `_release_sim`; guard the lazy getter for None. |
| M8 (arch New-4) **MED** | `capture()` builds activity unconditionally → 32–128 MB + two passes for a DISABLED feature; or None-deref. | Gate the `capture()` build on `field_drive_enabled(params.detail)`; `activity_eq=None`/means unset when off; `_derive_tile` passes activity only when `snap.activity_eq` present. |
| M9 (arch New-5) **LOW** | Fill term `textureLod` needs mip-complete activity in BOTH preview and snapshot textures or fill diverges preview≠export. | `build(out_tex)` contract: `out_tex` is mipmap-capable and `build()` owns `generate_mipmaps`. Both facade `_activity_tex` and snapshot `activity_eq` allocated mip-capable. |
| M10 (vis NEW-3) **MED** | Re-keyed guards + broad fill floor defeat the `belt>0.02` backtrace early-out → per-pixel backtrace cost returns broadly at 16K (round-1 O6 assumed one extra sim-res pass). | Keep the guard a genuine early-out: re-key to `max(belt, fold_place) > 0.02` AND keep the fill floor 0 in truly-quiet regions (low `FILL_LOD` + nonzero `FILL_LO`). **Re-verify 16K export cost** as an explicit acceptance step. |
| M11 (glsl part. + vis NEW-5) **LOW** | "A<B=C<D … sum ~1" invites a dead band/overlap if coded as two near values. | State as **3 cut points A < M < D** (middle shared literally). Partition sums to exactly 1; **no** `/(sum+ε)`. |
| M12 (vis R1-partial) **LOW** | Global gate can't blank a LOCALLY-quiet interior inside a busy preset → it gets cells, not bare. | Documented as intended (cells = quiet-zone filler, as today's `w_cell`); acceptable if cell amplitude stays low. Not a blocker. |
| M13 (vis NEW-4) **LOW** | 60–66° seam: mean masked at 60° but drive fades at 66° (`routeW`) → thin ring of inflated `strain_n` → extra fold at 60–66°. | Align the mean-mask edge with `ROUTE_LO` (66°) OR start the drive fade at 60°. |
| M14 (glsl Nnew3) **LOW** | Mean readback into a Python uniform = GPU→CPU→GPU stall. | Write the reduction result to a **1×1 texture** sampled in detail.comp (also snapshot-scopes the means cleanly alongside M2), OR return-and-pass. Pick the 1×1-texture route to avoid the stall and unify with M2. |
| M15 (doc nits) **LOW** | N13 parenthetical stale (`field_scale` is now a **detail.comp** uniform, sample-time); golden-hash "unit tier" assumes include-expansion without a GL context. | Corrected below; verify `#include` expansion is doable headless (it is textual in `gl/context.py`) before pinning the golden-hash as unit-tier — else mark it GPU-tier. |

**O2r (reduction shape) — CLOSED:** two-pass, workgroup-local tree reduction, no atomics.
Pass 1: each workgroup reduces its tile to partial sums (masked `|lat|<66°`), written to a
`partials[]` SSBO at its fixed workgroup index. Pass 2: one workgroup sums `partials[]` in
**fixed index order** and divides. Determinism from fixed grouping + fixed combine order.
Emit BOTH the global masked mean AND per-row means (the per-row reduction is the eddy-strain
input, M3). Divisor = the analytically-known in-band texel count per resolution (CPU
constant), so preview and export divide by an identical constant. Pin launch geometry as a
function of resolution only. Reject the serial reduction (TDR risk at 8192).

**O3r — CLOSED:** enforce `M` shared literally (3 cut points A<M<D), no explicit
normalization. Drive off **eddy** strain (M3). Calibrate A, M, D, `S_REF_ABS`, the eddy
`MEAN_LO/HI`, `SHEAR_HI`, `FILL_W`, `FILL_LOD` on renders — after **measuring eddy-mean per
preset** (not total mean, or ice_giant vs jupiter_like won't separate).

**O5r — CLOSED:** default fill = low-pass **strain** at a low LOD; `|vort|` fill NOT used
(re-bands along jets; its one genuine advantage — vortex-core interiors — is already the N7
`lace += field_vort·wn·(1−fold)` core-fill term).

---

*Below: the REV-2 body, still valid except where the M-table above amends it. Key amended
spots are flagged inline.*

## Problem

Render-side detail synthesis (`render/kernels/detail.comp`) reads as artificial. Root
cause (user-diagnosed 2026-07-07, confirmed in code): the **placement** of every texture
term is gated by a **latitude LUT**. `belt` and `shearN` are sampled from `u_profile_dyn`
at a single fixed row (`texture(u_profile_dyn, vec2(lat, 0.5))`) — pure functions of
latitude (`shearN` = `shear_norm = |du/dφ|`, `sim/profiles.py:112`); `mottle` has a
hard-coded 30–63° window. Texture is stamped in horizontal bands regardless of where the
flow has structure. Dialing amplitude down (the "lite/off" audit) only quiets a mis-placed
signal.

**Not a re-tread of falsified work.** The FALSIFIED work (`docs/roadmap.md`) was frozen-field
dye *advection* (manufacture folds by advecting a scalar; killed by the integrability wall).
This plan advects nothing, creates no dye, and keeps the existing backtrace-fold noise
character. It is the roadmap-endorsed "separable companion win: drive amplitude masks from
local 2-D sim fields (vorticity/strain) instead of the 1-D latitude LUT" (roadmap
L227-230). Confirmed by round-1 review.

## Goal / non-goals

**Goal:** drive detail *placement* from local flow, not latitude. Texture concentrates where
the baked velocity is deforming (jet edges, vortex rims, fold zones, the outbreak plume) and
fades in quiescent interiors — band structure *emerges* from the flow.

**Non-goals:** (a) changing noise *character* (parked, harder problem); (b) new sim pass or
sim coupling; (c) breaking preview==export or the POST fast edit loop; (d) moving any tracked
default output when off.

**Success criteria:** on the two VORTICITY presets (gas_giant_warm, jupiter_vorticity) at
2048–4096, vs the current latitude-gated look, field-driven (i) removes the "stamped
horizontal bands of stipple" read, (ii) puts texture on visible fold/shear structure,
(iii) keeps genuinely-quiet interiors clean, (iv) reads at least as good as `off` but with
more structure where the flow warrants. **User visual sign-off is the gate.**

## Decisions locked in brainstorming

1. Driving signal = flow structure: local strain `S = |∇velocity|` (not tracer gradient).
2. Keep the texture-flavor vocabulary, re-place it: high strain → folds, moderate → lace,
   quiet → cells. Keyed to strain level, not latitude.
3. Robust normalization is the primary look lever → **mean-relative with an absolute
   floor**, computed at sample time (see Round-1 fixes N1/N2).
4. Vorticity built into the same pass; its look-influence is an empirical knob (default 0),
   applied where strain is LOW (core-fill), decided by A/B.

## Round-1 review outcomes folded in (traceability)

| ID | Finding | Resolution |
|----|---------|-----------|
| N1 (arch#2, vis R1) | `field_scale` baked into cached map ⇒ cache/POST contradiction; mean-relative erases absolute quiescence | Store **raw** `S`,`ω`; divide + floor at **sample time**; `field_scale`, means become uniforms. Cache is velocity-only. |
| N2 (vis R1, R6) | "clean interiors" impossible; mean-relative spreads texture on laminar presets | **Absolute floor**: `strain_n = S / max(field_scale·mean_S, S_REF)`, and whole effect gated `drive_eff = drive·smoothstep(MEAN_LO,MEAN_HI,mean_S)` (physical units) ⇒ quiet fields self-disable. |
| N3 (vis R3, R7) | strain is edge/rim-concentrated ⇒ wireframe belts; interiors go bare | **Fill floor** from low-passed strain (mid-mip) + vorticity core-fill; strain SHARPENS on top of a filled base, does not replace the fill. |
| N4 (vis R4, glsl#2) | 3 smoothstep gates don't partition (dead band + double-stack); single `belt` mutation can't carry both fold & cell | **Contiguous partition** (cell/lace/fold cut points share edges, sum≤1); **per-site blends** with fresh locals; cell site is its own blend, not reuse of mutated `belt`. |
| N5 (vis R5, glsl#3, arch#4) | pole strain inflates the global mean (values, not area); 1/cos blowup; must scrub non-finite | Mean over **|lat|<60° mask only**, deterministic fixed-order reduction; 1/cos applied ONLY to λ-derivs, cos-floor raised; **finite-scrub** (`isnan/isinf→0`, clamp) before write. |
| N6 (arch#1) | shared synth-owned activity texture tears the export (snapshot bug) | Activity is **snapshot/caller-scoped**: `build(vel, out_tex, params)` writes a caller-supplied texture; export owns its own via `ExportSnapshot`. Never a synth singleton shared with preview. |
| N7 (arch#3) | second per-frame tile loop (`export_sequence_job`) not enumerated | Rebuild activity **per frame** from that frame's snapshot velocity, in BOTH exporter loops; add `activity` to `_derive_tile` signature. |
| N8 (arch#2/#6) | no invalidation wired; RESTART changes sim res; leak | Rebuild activity on **VELOCITY & RESTART**; `ActivitySynth.release()` called from `_release_sim`; size vs `vel_tex.size`. |
| N9 (glsl#1) | "byte-identical forced variant" test is wrong (cross-binary FP reschedule) | The `array_equal` test is the **routing** test (drive=0,vort=0 ⇒ non-FIELD_DRIVE program). Forced variant ⇒ `atol=1e-3`. |
| N10 (glsl#2) | base-path edits risk default-text drift | Base-path `w_streak`/`w_cell` wrapped `#ifdef FIELD_DRIVE/#else`; `#else` arm is a **verbatim** copy of today's line; a **preprocessed-source golden-hash** unit test catches drift. |
| N11 (glsl#4) | mip-mean ≠ arithmetic mean (non-po2); determinism | Commit to a **fixed-order tree reduction** compute pass over the masked band (NOT raw mipmap mean, NOT atomic-float). Mid-mips used only for the (look-only, non-critical) fill floor. |
| N12 (glsl#5, R6) | `field_vort_influence>0` at `drive=0` is a silent no-op / wasted variant; early-out guards skip off-band strain | Predicate requires `field_drive>0` for vort to matter; re-key `belt>0.02`/`belt<0.98` backtrace guards to the **blended** placement so off-band high-strain isn't skipped. |
| N13 (glsl#5/#6) | `field_scale` must be excluded from predicate; tripwire must cover `u_activity` | Predicate/metadata set = `{field_drive, field_vort_influence}` (pinned by test); tripwire covers `u_field_drive,u_field_vort,u_activity`, excludes `u_field_scale` (different program). |
| N14 (vis R2) | for banded flow raw strain ≈ the shearN LUT (circular); kinematic ≈ no new signal | Rollout targets **vorticity presets first**; kinematic adoption is a separate skeptical A/B. Optional **eddy-strain** (row-mean-subtracted) driver deferred as an empirical knob if circularity shows. |

## Architecture (revised)

### Activity pass — `render/activity.py` + kernels

Two kernels:
- **`activity.comp`** (sim-res): from the baked equirect velocity, central differences over
  ±1 sim texel (`u_vel_texel` uniform; **`texture()` with normalized coords**, never
  `texelFetch`, so the ±180° x-seam wraps — `vel_tex` is `repeat_x=True`). Compute
  `du/dφ, du/dλ, dv/dφ, dv/dλ`; apply `1/cos(φ)` to the **λ**-derivatives only, with a
  raised cos-floor (~0.3). `strain = sqrt((du_dλ' − dv_dφ)² + (du_dφ + dv_dλ')²)`,
  `vort = dv_dλ' − du_dφ`. **Finite-scrub and clamp** both to a bounded range before write.
  Output **RG32F, sim res**: `R = strain`, `G = vort` (RAW — no normalization here). Enable
  mipmaps for the low-pass fill sample (look-only). *(Deformation-tensor curvature terms
  omitted — acceptable for a look heuristic under the polar fade; documented approximation.)*
- **`activity_reduce.comp`** (fixed-order tree reduction): mean of `strain` (and `|vort|`)
  over the **|lat|<60° band only** → a tiny buffer/1×1 texture read into uniforms
  `u_mean_strain`, `u_mean_vort`. Deterministic (fixed reduction order); NOT atomic-float,
  NOT raw mipmap.

`ActivitySynth.build(vel_tex, out_activity_tex, params) -> None` writes into a
**caller-supplied** texture (mirrors `DetailSynth.synthesize(..., out_tex, ...)`); it also
returns/stores the two scalar means for the caller to pass through. Owns only its programs
+ the reduction scratch, never a shared output texture.

### detail.comp — new `FIELD_DRIVE` variant (independent of DETAIL_FX ⇒ 4-variant cache key)

When `FIELD_DRIVE` undefined ⇒ post-preprocess text is **exactly today's kernel**. When
defined, at **sample time** (REV 3 — eddy strain, fill folded into the partition axis, means
read from a 1×1 reduction texture per M14):
```glsl
float S     = texture(u_activity, eqUV(ll)).r;               // raw strain
float Sfill = textureLod(u_activity, eqUV(ll), FILL_LOD).r;  // low-pass strain (FILL_LOD 2-3)
float W     = texture(u_activity, eqUV(ll)).g;               // raw vorticity
// M3: drive off the EDDY (non-zonal) strain. rowmean(lat) sampled from the reduction's
// per-row output (1-D texture indexed by lat); mEddy = mean of S_eddy over |lat|<66 band.
float S_eddy    = max(S     - texture(u_rowmean, latV).r, 0.0);
float Sfill_eddy= max(Sfill - texture(u_rowmean, latV).r, 0.0);
float mE   = max(u_mean_eddy, 1e-6);                          // eddy mean (1x1 tex / uniform)
float strain_n = S_eddy     / max(u_field_scale * mE, S_REF_ABS);   // N2 absolute floor
float fill_n   = Sfill_eddy / max(u_field_scale * mE, S_REF_ABS);
float wn       = abs(W) / max(u_field_scale * u_mean_vort, W_REF_ABS);
float drive_eff = u_field_drive * smoothstep(MEAN_LO, MEAN_HI, mE)   // self-disable on EDDY mean
                                * (1.0 - routeW);                     // N5/O1/M13 pole fade
// M4: fold the fill into the partition AXIS (not a max outside it), so a filled interior
// moves cell->lace->fold coherently and the gates still sum to exactly 1.
float place = max(strain_n, FILL_W * fill_n);
// M11 contiguous partition — 3 cut points A < M < D, middle shared LITERALLY, sums to 1:
float cell = 1.0 - smoothstep(A, M, place);
float lace = smoothstep(A, M, place) * (1.0 - smoothstep(M, D, place));
float fold = smoothstep(M, D, place);
// N7/O5r vorticity core-fill (only where strain is low; NOT used as the fill signal):
lace += u_field_vort * wn * (1.0 - fold);
// transfer place -> a [0,1] shear-like weight (NOT a raw clamp) for w_streak:
float shear_drv = smoothstep(0.0, SHEAR_HI, place);
float fold_place = fold;   // belt placement = fold share of the (fill-aware) partition
```
Then blend each existing gate by `drive_eff`, at each site with fresh locals:
- FX belt sites (`belt_texture`, `belt_texture_fine`, striation belt-gate, w_streak
  belt-floor): `belt_place = mix(belt, fold_place, drive_eff)`.
- `zone_texture` / cell sites: `zone_place = mix(1.0 - belt, cell, drive_eff)`.
- mottle window: `aw' = mix(aw, lace, drive_eff)` (removes the hard 30–63° window at full
  drive).
- **Re-key early-out guards** so off-band high-strain isn't skipped (N12), while keeping them
  genuine early-outs to bound 16K cost (M10): `belt > 0.02` → `max(belt, fold_place·drive_eff)
  > 0.02`; `belt < 0.98` similarly; **and the mottle `if (aw > 0.0)` guard (M6)** →
  `if (aw > 0.0 || lace*drive_eff > 0.0)`.

**Base-path terms** (`w_streak`:260, `w_cell`:262 — present in BOTH fx and non-fx programs)
wrapped:
```glsl
#ifdef FIELD_DRIVE
    float sh = mix(shearN, shear_drv, drive_eff);
    w_streak = clamp(0.2 + 0.8*(sh + speedN),0.,1.) * (0.4+0.6*tr.b) * (1.0+1.4*hero);
    w_cell   = u_cell_amount * mix(1.0-belt, cell, drive_eff) * (1.0-speedN) * (1.0-sh) * (1.0-0.6*routeW);
#else
    // VERBATIM copies of detail.comp:260-263 (byte-for-byte; a stray space breaks identity)
#endif
```

### Params (`DetailParams`, POST tier, default no-op)
- `field_drive: float = 0.0` (lo 0, hi 1) — blend latitude→strain placement.
- `field_scale: float = 1.0` (lo 0.25, hi 4.0) — the `k` in `S/(k·mean)`; sample-time
  uniform (cheap POST). NOT in the variant predicate.
- `field_vort_influence: float = 0.0` (lo 0, hi 1) — empirical core-fill knob; only bites
  when `field_drive>0`.

Metadata flag `field_drive=True` on `field_drive` **only** (M5) — it alone selects the
FIELD_DRIVE variant via `field_drive_enabled`. `field_vort_influence` does nothing at
`drive=0`, so it is NOT a variant selector; its `u_field_vort` uniform stays in the
`_assert_field_drive_uniforms` explicit list (a known variant uniform). Pinned by
`EXPECTED_FIELD_DRIVE_LEVERS = {"field_drive"}`. `field_scale` is a plain detail.comp
tunable (sample-time, M15), excluded from both the predicate and the tripwire lever-set.

## Data flow
```
sim step → baked vel_tex (+ patch vels)      [RESTART sets sim res; VELOCITY re-bakes vel]
     │  field_drive_enabled(p.detail)
     ▼
ActivitySynth.build(vel_tex, activity_tex, p) → RAW strain/vort (sim res) + mean_S, mean_ω
     │   preview: facade-owned activity_tex, rebuilt on VELOCITY/RESTART
     │   export:  activity_tex lives in ExportSnapshot, rebuilt per frame (both loops)
     ▼
DetailSynth.synthesize(..., activity_tex, mean_S, mean_ω, params)  (FIELD_DRIVE variant)
     ▼
MapDeriver.derive(... detail_tex ...) → color/height/emission
```

## Implementation plan (file-by-file)
1. `params/model.py` — 3 pfields (tier POST, ui="Detail", adv; `field_drive`/
   `field_vort_influence` flagged `field_drive=True`; `field_scale` NOT flagged).
2. `render/kernels/activity.comp` — gradient tensor → raw strain/vort, 1/cos on λ-derivs,
   finite-scrub+clamp, RG32F out.
3. `render/kernels/activity_reduce.comp` — masked (|lat|<60°) fixed-order mean reduction.
4. `render/activity.py` — `ActivitySynth`: programs, reduction scratch, `build(vel, out,
   params)` (caller-supplied out), returns means; `release()`.
5. `render/kernels/detail.comp` — `#ifdef FIELD_DRIVE` block (uniforms, sample-time
   normalize+floor+partition+fill+vort), per-site `mix` blends, re-keyed guards, base-path
   `#ifdef/#else` with verbatim `#else`. ZERO change to default text.
6. `render/detail.py` — cache key `(fx, field_drive)`; `field_drive_enabled`;
   `_assert_field_drive_uniforms` (covers `u_field_drive,u_field_vort,u_activity`); bind
   `u_activity` at **location 7** in the FIELD_DRIVE branch ONLY; set uniforms incl.
   `u_field_scale,u_mean_strain,u_mean_vort`; accept `activity_tex` + means args.
7. `engine/facade.py` — facade-owned `ActivitySynth` + `_activity_tex` (sized to
   `vel_tex.size`); build in `_derive` when enabled; **invalidate/rebuild on VELOCITY &
   RESTART**; release in `_release_sim`.
8. `engine/snapshot.py` — add `activity_eq` to `ExportSnapshot`, built in `capture()` from
   the cloned `vel_eq`, released in `release()`.
9. `export/exporter.py` — build activity per frame from the snapshot velocity in BOTH
   `export_job` and `export_sequence_job`; add `activity` to `_derive_tile`.
10. Tests (below). 11. Docs: `sliders.md` regen, `architecture.md` detail note, roadmap
    entry. 12. Preset rollout — SEPARATE, gated on sign-off; vorticity presets first.

## Testing (revised per N9/N10/arch#5)
- **Routing byte-identity** (`np.testing.assert_array_equal`): `field_drive=0 AND
  field_vort_influence=0` ⇒ non-FIELD_DRIVE program selected ⇒ identical to today; p05 hash
  unchanged. *This is the byte-identity guarantee.*
- **Forced-variant no-op** (`atol=1e-3`, NOT array_equal): `field_drive=1e-6` ⇒ FIELD_DRIVE
  binary ≈ default (cross-binary FP reschedule tolerated). Run in BOTH fx=False and fx=True
  arms (base-path re-gate lands in both).
- **Preprocessed-source golden hash** (unit tier): the non-FIELD_DRIVE preprocessed
  detail.comp text hash is unchanged (catches `#else`-arm drift).
- **Activity finiteness**: on a real pole-inclusive sim-res render, `np.all(np.isfinite(activity))`.
- **Reduction** (GPU tier): masked mean on a known field matches expected (model the
  fixed-order reduction; assert `strain_n≈1` at an average-activity texel).
- **Behavior**: `field_drive>0` differs from 0; a synthetic high-shear strip yields higher
  `strain_n` and more fold there than a quiescent strip; an off-band high-strain patch DOES
  get texture (guards re-keyed — N12).
- **Seam** (tiled vs full, `validate_arrays`) at `field_drive>0`.
- **Dispatch cross-ref** (mirror `test_detail_fx_metadata` L89-128): the FIELD_DRIVE
  `synthesize` block uploads every predicate lever's uniform (catches a forgotten `_set`).
- **Metadata pin**: `EXPECTED_FIELD_DRIVE_LEVERS = {field_drive}` (M5 — predicate is
  `field_drive` only; excludes both `field_vort_influence` and `field_scale`).
- **VELOCITY invalidation**: a jet edit re-bakes velocity ⇒ activity rebuilds (not stale).
- **Determinism note**: activity is deterministic on the baked velocity. Kinematic ⇒
  byte-exact; vorticity ⇒ within documented floors (SOR LSB noise), NEVER byte-exact.
- **Empirical crux** (not unit): warm & jupiter_vorticity at `field_drive∈{0,0.5,1.0} ×
  field_vort_influence∈{0,on}`; side-by-side judge; go/no-go on vort + shipped drive value.

## Open questions — ALL CLOSED in round-2 (see M-table at top)
- **O2r — reduction shape:** CLOSED → two-pass workgroup-local tree reduction, partials
  SSBO summed in fixed index order, analytic in-band divisor; emits global + per-row means.
- **O3r — cut points / partition shape:** CLOSED → 3 cut points A<M<D, M shared literally,
  NO explicit `/(sum+ε)`; calibrate on renders after measuring **eddy**-mean per preset.
- **O4r — eddy-strain (N14):** CLOSED → build the eddy (row-mean-subtracted) driver in v1
  (M3); the per-row reduction that O2r already emits makes it near-free.
- **O5r — fill signal:** CLOSED → default fill = low-pass **strain** at a low LOD; `|vort|`
  is NOT the fill signal (re-bands along jets); vortex-core interiors are the N7 core-fill
  term instead.

## Rollout
Ship default-off (byte-identical), full test matrix green, pre-merge review. THEN a separate
visual pass turns `field_drive` up on **gas_giant_warm / jupiter_vorticity first**, empirical
vort decision, user sign-off on montages, deliberate p05 re-baseline, JSON regen. jupiter_like
only if a skeptical A/B shows a win (N14); saturn_pale / ice_giant likely stay latitude-gated.
