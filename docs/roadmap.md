# Roadmap

Forward-looking work items. Append-only-ish; move items to "Done" with a date
rather than deleting, so the reasoning survives.

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
> to 0.05 in the `jupiter_baroclinic` preset.
