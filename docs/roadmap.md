# Roadmap

Forward-looking work items. Append-only-ish; move items to "Done" with a date
rather than deleting, so the reasoning survives.

## FALSIFIED (2026-06-23): Tier-3 emergent color regresses the look — V0 is the ship

The "Tier 3 — Emergent color via a designed curl field" plan below was executed
through its front-loaded **Phase-1 prototype** (throwaway, ungated kernel edits,
since reverted). The cheap render-and-judge gate did its job: it killed the
approach before any productionization.

- **Part B (passive non-banded color)** washes the bands to a uniform orange field
  in BOTH kinematic and vorticity modes — a divergence-free flow can't sustain the
  per-band DC contrast from a uniform-mean source (only storm cores survive). This
  is the plan's own #1 risk, confirmed.
- **The kinematic base** renders as laminar horizontal stripes; Part A's zone eddy
  adds only marginal mottle. It is strictly *worse* than what's shipped.
- **The shipped vorticity `gas_giant_warm` (V0)** already produces flowing swirls,
  billows, KH waves, and coherent vortices — it beats every Tier-3 variant.
- **Mode fact:** `psi.comp` is the streamfunction only in KINEMATIC mode; in
  VORTICITY mode it only feeds a poleward/apron feather blend (`solver.py:~722`), so
  the plan's "KEY" Part-A eddy is INERT on the vorticity preset regardless.

Decision (user, 2026-06-23): accept V0, close Tier-3. Real zone *brightness* would
require a true divergent upwelling field (a deep solver change), not justified —
the frost/blur complaints were already resolved by the palette + replenish fixes.
The "Tier 3" sections below are retained as the (now-falsified) reasoning trail.

## RESOLVED (2026-06-22): the "frosted glass" was the PALETTE, not the architecture

The frosted-glass look was chased for a whole session through the simulation and
color-seeding architecture (every "root cause" below — co-seeding, relaxation
pinning, laminar zones). **All of those theories were wrong.** A prototype
isolation proved it: dumping the raw T0 color-index field showed it is already
*gorgeous* — rich, high-contrast, swirling gas-giant structure at all latitudes
(`_diag/iso_T0_stretch.png`). The frost was introduced **downstream, at the
palette lookup** (`derive.comp:119`): the baked palette LUT was a near-flat,
low-chroma pale ramp across the T0 axis, so a T0 swing of 0.29→0.89 (huge field
contrast) collapsed to nearly one pale color = frosted glass. The simulation was
fine; the palette was throwing its structure away.

Why every prior fix failed: palette **de-blue** changed *hue* but kept the low
*value/chroma* contrast; turbulence/eddy/passive-color all enriched a T0 field
that was already being flattened at lookup. The missing ingredient was **value +
chroma dynamic range across the T0 axis**, nothing in the sim.

**Fix (shipped):** a new `gas_giant_warm` factory preset with a high-contrast
warm palette (real dark-brown→cream value range), applied **uniformly across
latitude rows** so band color follows T0 structure rather than a per-latitude
stamp (this also de-imposes the bands — the user's original ask). Appearance-only
(POST tier): **no kernel changes, zero byte-identity risk, no hash-pin churn.**
Built by `scripts/build_warm_preset.py`; frost-fix property pinned by
`test_gas_giant_warm_palette_has_value_contrast`.

Prototype detour (kept for the record, all reverted): a latitude-uniform eddy
term in `psi.comp` (Part A) DID add zone strain and is a valid future enhancement,
but is marginal once the palette carries contrast — dropped to keep the fix
appearance-only. Passive non-banded color (Part B) washed the bands out (the
divergence-free flow can't sustain band color contrast from a zero-DC source) —
falsified, matching the pre-build adversarial review.

The sections below are the SUPERSEDED investigation. They remain as the reasoning
trail but their "root cause" conclusions are wrong — the palette was the cause.

## Emergent banding (kinematic → dynamic)  *(SUPERSEDED — see RESOLVED above)*

**Finding (2026-06-22).** The bands read as "imposed upon the mapping" rather than
arising naturally — and architecturally they are. Both the **bands** (color) and the
**jets** (advecting flow) are prescribed latitude profiles:

- `bands.template` (or `generate_bands`) defines explicit edge latitudes + per-band
  color/height values — a painted latitude profile.
- The solver **continuously relaxes** the tracer back toward that profile every step
  (`turbulence.relax_tau`, default 350) and **replenishes** it
  (`turbulence.replenish_rate`, default 0.015; plus `belt_replenish`). So the band
  identity is pinned to the template; it cannot mix away.
- The jets are likewise a prescribed zonal-velocity latitude profile (`build_profiles`
  / `select_lanes`), not emergent. Turbulence + vortices + warp only *decorate and
  shear* the imposed structure.

The engine is **kinematic** (prescribe the mean structure, advect detail through it),
not **dynamic** (let the flow self-organize bands/jets). This is a deliberate "fake
it" shortcut — emergent jets are hard to get looking good — but it's the root of the
imposed feel.

Evidence: with the relaxation/replenish leash OFF (`relax_tau→∞`, `replenish_rate=0`)
the bands deform and mix instead of staying razor-striped — markedly less painted.
(See the `_diag/leash_on.png` vs `leash_off.png` comparison from the 2026-06-22
session.) That localizes a large fraction of the "imposed" quality to the relaxation
pinning, not to the bands existing.

### Tier 1 — Loosen the leash  *(IN PROGRESS, 2026-06-22)*
Weaken `turbulence.relax_tau` / `replenish_rate` so bands deform and mix. Still
kinematic underneath (jets prescribed), but reads far less imposed. Risk: fully
off + no replenish slowly washes fine detail out over a long develop (nothing
sustains it) → target a *tuned weak leash*, not zero. Cheap (param tuning + a preset
value). Deliverable: a weak-leash setting (and likely a new/updated preset) that
keeps structure alive while letting boundaries undulate and mix.

### Tier 2 — Soften the jets  *(roadmap)*
The jets are also a prescribed latitude profile. Let the banding lean more on the
flow and less on both templates (weaken/partially-randomize the jet profile, or drive
it from the evolving vorticity rather than a fixed target). Medium effort, heavy
tuning, easy to collapse to mush. Do only if Tier 1's weak-leash look still feels
imposed.

### Root cause sharpened (2026-06-22): color is co-seeded with the flow banding
The "frosted zones" / "imposed" feel survives every downstream knob (palette de-blue,
turbulence amplitude, dropped template, cranked cellular/mottle render-texture) because
of a deeper coupling the user identified: the **color tracer T0 is seeded from the same
band latitude profile as the flow** (`init.comp:47-53`, `t0 = u_profile_stamp.r + noise`)
AND **re-pinned to that band stamp every step** (`advect.comp:210`,
`result.x += (s0 + ... - result.x)*relax_k`). So bright-zone color is permanently
co-located with the calm (low-shear) zone flow → bright + smooth = frosted, structurally.
The bright/dark band color contrast exists ONLY because it's stamped; a passive tracer
with the stamp removed homogenizes (no bands) rather than forming them — which is why this
pulls toward Tier 3 (bands need a dynamical source to emerge). Bounded half-measure to try
first: phase-OFFSET the color band stamp from the flow band profile so bright color overlaps
high-shear flow and gets sheared/textured (breaks the frosted correlation while keeping
imposed zone/belt contrast). Core-kernel change (init.comp/advect.comp) → breaks the
kernel hash-pin (`test_kinematic_kernels_pinned.py`) and changes byte-identity for ALL
presets; scope accordingly.

### Tier 3 — Emergent color via a designed curl field  *(user's recipe, 2026-06-22)*
The validated direction (after the decorrelation experiment confirmed co-seeding is the
frosted-glass root): **the color must be MIXED into structure by the flow, never stamped.**
Concrete recipe the user specified:
1. **Initial color map = NOT strict latitude bands** — seed it non-banded (noise / broad
   blobs / contrast field), decorrelated from the flow profile.
2. **Faintly re-introduce color CONTRAST each step** — refresh a non-banded contrast source
   (like the detail-tracer replenish, NOT the band stamp), so the tracer doesn't homogenize
   to uniform; the flow then stirs that contrast into structure. (My prototype relaxed toward
   a *partial band stamp* — wrong; that re-imposes strict bands and re-frosts. Must refresh
   non-banded contrast instead.)
3. **The KEY: design the CURL FIELD so its advected output mimics gas-giant structures** —
   bands, textured zones (no laminar smooth patches), storms, festoons all emerge from the
   velocity field stirring the passive color. The current kinematic curl field has LAMINAR
   quiescent zones (low shear between jets) → smooth color patches → frosted; a designed
   field needs structure (eddies/strain) at ALL latitudes, including zone interiors.
Experiments tried & rejected (all still frosted, recorded so we don't repeat): palette
de-blue, turbulence-amplitude, drop template, crank cellular/mottle render-texture,
decorrelate seed + relax-to-uniform (washes out), decorrelate seed + relax-to-partial-band
(re-frosts). The throwaway prototypes live in this session's `_diag/` + `scripts/`.

### Tier 3 (general) — Truly emergent banding  *(research-grade)*
Make bands *arise*: force turbulence on a beta-plane and let zonal jets self-organize
via the Rhines mechanism, then a passive composition tracer bands naturally with **no
template at all**. Physically honest; the vorticity solver is a plausible starting
point (it already carries the latitudinal Coriolis gradient `f = f0·sin(lat)` that
drives jet formation — see `omega_force.comp` q_target). Big rebuild, real risk it
needs heavy tuning to look right, and no guarantee it beats the faked version
aesthetically. Gate the decision to start this on whether Tier 1 (and maybe 2) leave
the imposed feel unresolved.

> Context: this line of work came out of the M3 baroclinic aesthetic exploration. The
> M3 baroclinic coupling engine + its review hardening (PR #3) is sound and
> independent of this; it can ship/sit/close on its own merits regardless of the
> banding-architecture direction. The "big orange plume" that kicked this off was the
> hero (GRS-analog) vortex at `hero_radius=0.15`, not baroclinic — diagnosed and set
> to 0.05 in the `jupiter_baroclinic` preset (that preset has since been DROPPED —
> see "Recorded dead ends" below; the coupling engine itself remains).

## Recorded dead ends (do not re-attempt)

Falsified or dropped directions that are under-recorded elsewhere. Check this list
(and the FALSIFIED sections above, plus `docs/superpowers/specs/*verdict*.md`)
before proposing work in these areas.

- **Fibrous cirrus by STAMPING fine strands into the tracer — FALSIFIED (2026-07-08).**
  Neptune's bright methane clouds are multi-stranded "combed fiber" cirrus, not smooth
  lobes. Attempt: modulate the elongated bright cloud stamps (accent/companion, via
  `accent_aspect`/`companion_aspect`) with a ridged anisotropic pattern to bake in thin
  east-west strands. Result: the strands do not survive to the render — over the dev run
  each feature's own rotation winds the stamped tracer into a hook and hyperviscosity +
  advective diffusion smear the fine detail into a smooth wash (verified by zoomed render
  inspection + a ~14-agent adversarial visual panel over 6 rounds). The `accent_aspect`/
  `companion_aspect` elongation lever itself SHIPPED (round dots -> soft wind-stretched
  streaks, a clear improvement baked into the `neptune` preset; byte-identical when off).
  **The genuinely viable path** for true fibrous cirrus is a RENDER-TIME synthesis pass
  (post-advection, so nothing smears it) — **IMPLEMENTED 2026-07-08 (issue #36)** as
  `detail.cirrus_fibers`: a CPU-side drifted bright-cloud list (`bright_cloud_centers`,
  the `hero_centers` snapshot mechanism) masks a DETAIL_FX fiber block in `detail.comp`
  (dilated analytic ellipse ∩ T0-excess over the `profile_stamp` baseline; flow-oriented
  thresholded anisotropic fbm, carve-dominant). `detail.streak_mute` kills the ungated
  base filament streak so laminar planets can run the detail pass fiber-only. NOT an
  advected mask channel — advection would smear/hook it exactly like the stamp.
- **`jupiter_baroclinic` factory preset — DROPPED (2026-06-28).** Its baroclinic
  festoon comb is intrinsically regular and reads *mechanical*; a natural-looking
  baroclinic preset converges to an injection-driven Jupiter clone, so the preset
  added nothing. The baroclinic coupling **engine feature is kept** (opt-in,
  default-off, byte-identical when off) — only the factory preset is gone. See the
  top-of-file note in `scripts/build_vorticity_presets.py`.
- **Jupiter polar-cyclone discreteness — DEFERRED (blocked).** Rendering discrete
  polar cyclone rings (Juno-style popcorn/ring structure) is blocked on
  vortex-merger physics — without it, injected polar vortices merge into a smooth
  cap instead of holding a discrete configuration. Research-grade; not scheduled.
  See `docs/realism.md` (polar rows of the source-fidelity audit).
- **Render-side "detail dye" advected through the FROZEN baked velocity —
  FALSIFIED by analysis (2026-07-06).** Proposal was to fix the render detail's
  artificial look (latitude-locked masks + fBm-reads-as-noise) by seeding a scalar
  dye and advecting it M steps through the final baked velocity (± curl-noise
  enrichment), making the folded dye the primary visible detail. Three adversarial
  reviewers converged on a kill: a **frozen 2-D velocity field is integrable** (ψ is
  its Hamiltonian, zero Lyapunov exponent), so a passive dye advected through it —
  2 steps or 40 — **cannot chaotically fold**; it only reveals folds already frozen
  in and stretches into streamline stripes / spiral wind-up. Adding frozen
  curl-noise keeps the field steady → iterated domain-warped fBm = the "superimposed
  distorted noise" the look complaint is about, relocated into the velocity. This is
  the **F17 wall** (raising sim resolution moved folded-filament structure 0.9%) — a
  *dynamics* wall, not a resolution wall. `detail.comp` is ALREADY frozen-field
  backtrace-fold, and its own comments record that longer folds shred into grain
  (L460-462); `belt_texture_fine` is already a deliberately-short 2-hop version.
  Architecture also killed it independently: a fixed 4K dye is required for
  preview==export but detail is a POST-tier re-derive (20-40-step advection per
  slider drag breaks the edit loop), and a 4K dye upsampled to 16K is *softer* than
  today's `f(lon,lat)` analytic pass, which already resolves at full output res. And
  it re-enters the **emergent-color replenishment dilemma** (uniform replenish
  homogenizes / washed-out; banded replenish re-imports the very latitude masks it
  set out to retire). Do not re-propose any frozen-field render-time advection of a
  detail field.

- **Uniform detail coverage (`detail.spread`) — SHIPPED default-off (2026-07-07).**
  Fixes both the original "detail-starved zones + stamped latitude bands"
  complaint AND its would-be cure: applies the flow-folded detail-FX texture at
  EVEN density across latitude (single POST lever; 0 = band-gated byte-identical,
  >0 = uniform coverage at that level, pole-faded). Still flow-folded (the
  backtrace sites), so even ≠ flat. Opt-in `SPREAD` variant, p05 gate green.
  - **Superseded design — strain-driven PLACEMENT (built then dropped in
    calibration, 2026-07-07).** First attempt drove *where* detail lands from a
    local 2-D sim field (eddy strain `|∇v|` + vorticity via an activity pass +
    CPU row-mean reduction), concentrating texture on jet edges / vortex rims /
    fold zones. FALSIFIED by visual calibration: strain-*selective* density read
    PATCHY ("details exist in certain places but not others, doesn't read well —
    want it everywhere balanced or nowhere"). Even coverage won; the strain
    engine (activity.comp/activity.py + snapshot/export plumbing) was deleted.
    LESSON: for a gas giant, EVEN texture density reads better than selectivity —
    by latitude OR by flow. Design record (now superseded):
    `docs/superpowers/specs/2026-07-07-field-driven-detail-design.md`.
  - Neither this nor the strain attempt changes noise *character*; the character
    redesign (decouple tracer-res, advect a high-res passive tracer through the
    EVOLVING field) remains parked (W13/W14).
## Research direction (SPIKED — CONDITIONAL GO, 2026-07-08): detail CHARACTER = sim-advected high-res tracer

> **Status update (2026-07-08, T18 crux spike).** No longer unstarted: the
> pre-registered crux gate below was run (`scripts/spike_detail_character.py`,
> measurement-only, ships nothing to the render path) and returned
> **CONDITIONAL GO** — see `docs/superpowers/specs/2026-07-08-detail-character-spike-verdict.md`.
> Advecting a high-res passive tracer through the evolving vorticity field
> **does** manufacture oriented filamentary structure: a strain-attributable,
> orientation-controlled **×3.7** separation from isotropic noise (advected
> 0.314 vs control 0.085), with rot90 collapsing it (0.314→0.265) — the
> mechanism is real, NOT an emergent-color / q-target dead end. The absolute
> **0.384** GO number is INCONCLUSIVE only because software GL cannot run the
> native fidelity the bar was calibrated at: the proxy ran the dynamics grid
> 16× coarse (256 vs 4096) and coherence rose monotonically with both
> resolution and development, so 0.31 is a **lower bound**. **Remaining gate**
> before the multi-session subsystem build: ONE native-GPU run of the same
> spike at dynamics 1024–2048 / 700 steps / tracer 4× confirming advected
> coher ≥ 0.384 (re-run the harness unchanged). The design premise below stands.

The render detail's fBm reads as noise because fluid folded-filament morphology is a
**dynamics** property, and no frozen-field render trick produces it (see the
FALSIFIED entry above; F17). The genuinely viable path, identified 2026-07-06 and
**parked behind the W13/W14 remediation tail**: the sim's tracers (T0–T3) are
advected through the **evolving** velocity every dev step — time-dependent advection,
which *does* chaotically fold (this is why the vorticity solver manufactures folds:
orientation coherence kinematic 0.14 → vorticity 0.384 → reference 0.62). The
untested lever is to **decouple tracer resolution from dynamics resolution**: keep the
velocity/vorticity solve at its current moderate grid (F17 says finer dynamics doesn't
help), but advect a **high-resolution passive detail tracer** through that upsampled
evolving field, so the 1024-grid strain folds a ~4K scalar into real 4K-scale
filaments. The render pass samples that tracer as its primary structure; a thin
high-frequency `f(ll)` grain layer *oriented/gated by the tracer's local gradient*
restores crispness at 16K (the advected tracer alone is fluid-soft above its own res).
Procedural noise retreats to seeding/forcing — the honest form of the project's
"sim-advected procedural" thesis. **Crux gate before any subsystem build** (project
discipline, cf. m2-adv): carry ONE extra high-res tracer through the existing solver
on a `gas_giant_warm` run and measure whether its folded structure crosses the F17
orientation-coherence bar toward 0.384/0.62 (current render noise = the ~0.14
control). Go/no-go on that number before committing the multi-session build. Separable
companion win (independent of this, low-risk, stateless): fix the detail *placement*
complaint by driving the amplitude masks from local 2-D sim fields (vorticity/strain/
tracer-gradient) instead of the 1-D latitude LUT — the `intermittency` term already
advects a 2-D mask, so the machinery exists.
