# Hero emergence — the GRS-realism pack (de-stamped hero storms)

**Date:** 2026-07-09 (rewritten same day after reference-anchoring; the original
relaxation-falloff-only design is superseded — see History below)
**Status:** implemented on `feat/hero-emergence`
**Lever:** `storms.hero_emergence` ∈ [0,1], default 0.0 (byte-identical off)

## Problem

Hero storms read as *stamped onto / imposed upon* the simulation. The user
confirmed all four tells (airbrushed interior, too-perfect boundary, no material
exchange, singular set-piece) and, after a first iteration, rejected
texture-level fixes: the storm was the wrong *kind* of storm and needed to be
anchored against the real GRS.

## Reference facts (Juno/Voyager; Wikipedia "Great Red Spot", JPL Juno releases,
## EBSCO research starter, JPL PIA01093 caption, Cassini PIA07782)

1. **The interior is stagnant** — "currents inside it seem stagnant, with little
   inflow or outflow"; "in the inner half of the Spot, reflectivity was lower,
   and the motion of the cloud deck was small and random." The ~430 km/h winds
   live in a **peripheral ring**.
2. **Cloud tops sit ~8 km above the surroundings** — the spot is a high, bright,
   *filled* salmon/brick-red mass; its red is a near-flat plateau with a fairly
   crisp edge, not a Gaussian stain.
3. It sits in the bright **Red Spot Hollow**; the turbulent wake is WNW.
4. It is famously **position-stable** for centuries.
5. Aspect is ~1.3:1 today and shrinking toward circular.
6. **Regional structure (the wide view):** the spot is a closed eddy trapped
   between opposing jets that "deflects the prevailing winds about its
   perimeter" — small clouds stream AROUND the oval in thin lines (equatorward
   edge → western cusp → southern edge), with entrainment only at the SE corner.
   The deflected westward jet indents the poleward side of the adjacent dark
   belt (the hollow) and collides with the higher-latitude eastward jet to make
   the turbulent rifting NW of the spot. **Beyond ~1.5–2 spot radii the bands
   are parallel again** — the neighborhood is NOT wound into a pinwheel.

## Why ours read wrong (root causes, found by measurement)

- **Velocity profile inverted:** solid-body rotation spins the whole interior,
  winding bright collar material into a pinwheel around a dark center (the old
  "whirlpool" failure dressed in modern texture). The real profile is a
  stagnant core + fast annular ring.
- **Fill profile wrong:** the tint/brightness stamp was `exp(-q²)` — a soft
  stain, not a filled plateau.
- **Stamp/core divergence (the invisible-tweaks bug):** the prognostic vorticity
  core wanders ~0.2 rad from the kinematic registry position over a dev run, so
  the relaxation painted red *where the storm wasn't*, smearing it into a muddy
  arc. Measured directly: developed T3 at the registry position was ~0.01 while
  the red sat at q≈−2. This is why every stamp-shape tweak looked identical.

## The mechanism (all paths gated by `hero_emergence`)

All radii are in q = distance / `hero_radius`: after the footprint compaction
(see History) **the visible plateau edge sits AT q≈1.0**, so the storm renders
at its authored size and q reads directly as "spot radii".

1. **Annular-ring vorticity** (`vortex_omega.glsl`): morph the solid-body disk
   toward a ring (`smoothstep(0.29,0.55,q) − smoothstep(0.78,1.04,q)`,
   amplitude −6.0, scaled so the peripheral wind speed matches the calibrated
   disk's). By Stokes' theorem the enclosed-vorticity-free interior has v≈0:
   the stagnant core *holds* its fill; the ring's shear folds the tracer at
   the boundary — genuine emergent raggedness exactly where the real storm has
   it, no injected noise. (An earlier interior-eddy injection was removed: the
   real interior is calm.)
2. **Partial shield skirt** (same block): the single-signed ring carries NET
   circulation, so its velocity decays like 1/r and winds the neighborhood
   into a pinwheel many spot-radii wide (the kinematic Gaussian never had
   this: a Laplacian-of-Gaussian is self-shielded — the solid-core lever lost
   that property). A gentle opposite-signed annulus
   (`+0.7·(smoothstep(1.05,1.4,q) − smoothstep(2.0,2.6,q))`) cancels ~70% of
   it. Deliberately PARTIAL and WIDE/WEAK: a full concentrated shield rolls up
   into its own companion cyclone and the near-dipole self-propels off the
   anchor (observed). Enclosed circulation at the rim — the peripheral wind —
   is untouched. Physically: the counter-flowing jets deflected around the
   Hollow (fact 6).
3. **Plateau fill** (`vortex_stamp.glsl`): morph tint/brightness/dome fill from
   the Gaussian core toward `1 − smoothstep(0.62,1.0,q)` with an fbm-frayed
   edge — a filled red oval, edge at the authored radius. Radial identity:
   inner half slightly darker (T0), outer annulus graded pale salmon (steep T3
   drop — the storm_tints LUT is non-monotonic and weight-coupled), faint
   azimuth-wobbled wrapped lanes. Wired into BOTH the init kernel and the
   advect relaxation target (an unset uniform on init was the first "nothing
   changed" bug — with relax_tau 2000 over 700 steps the target alone
   converges only ~19%). Thin dark ring at 1.05 and THIN pale collar at ~1.30
   hug the edge (widths sharpen with the lever), both softened ×(1−0.5e) — the
   real Hollow is subtle, not a glowing basin.
4. **Hero anchor** (`omega_force.comp` SUBPASS 0 + `heroAnchorWindow` in
   `vortex_omega.glsl`): boost the existing q-nudge ×60·emergence inside the
   hero window (fade 1.6→2.8 q) so the prognostic core stays glued to the
   registry position the stamp paints. Rate-clamped at 0.5/step. 60× holds the
   standing offset within ~half a core radius across 512–2048 res (the offset
   scales with per-step drift error, which doubles at half resolution). The
   window is deliberately WIDER than the visible anatomy: it is a capture
   basin (it must exceed the ~0.2 rad free-drift excursion, and it covers the
   shield skirt), not a visual feature — shrinking it to the compacted
   anatomy reproduced the 0.04-T3 wander failure.
5. **Rim-band relaxation falloff** (`heroRelaxWeight` in `vortex_stamp.glsl`,
   applied in `advect.comp` pass 2): fade tracer relaxation in the ring band
   only (`exp(−(q−1.0)²·3.8)`, eroded per-azimuth by a folded fbm), so
   advection owns the boundary and its exchange with the jets. The interior
   keeps FULL relaxation — it must hold the red fill (the original design's
   interior falloff was wrong and is removed).
6. **Neighborhood band-flush** (same `heroRelaxWeight`): BOOST relaxation
   ×(1+5e) in an outer annulus (q 1.55→2.1 up, 2.7→3.4 down). Over a dev run
   the ring's residual circulation still winds the neighborhood into
   concentric arcs and nothing erases them (relax_tau ~2000); on the real
   planet the jets sweep wound material downstream and the bands re-assert
   (fact 6). The hero stamp is ~0 out there, so the boosted target IS the band
   stamp. Paired with the partial shield: the shield slows the winding enough
   for the flush to win. Locality bound is q<3.6.
7. **Render-side detail awareness + quiet storm** (`detail.comp`, HERO_EMERGENCE
   variant; `render/detail.py` program cache keyed (fx, spread, emergence)):
   the legacy hero-keyed detail treatments were built for the old whirlpool
   look and actively fought the pack — the interior log-spiral lanes wind a
   whirlpool inside the now-stagnant core (suppressed ×(1−0.85e)), and the
   noise-stack winding/amplitude boosts (`×(1+1.2·hero)`, `×(1+1.4·hero)`)
   made the interior LOUD where the real GRS is quiet (Juno close-ups are
   smooth tonal fields with sparse ~3%-contrast wisps) — they fade with the
   lever, and the band-grain calm deepens to a 0.9·hero floor. Collar
   streamlines tighten onto the thin collar (outer window edge mixes
   1.85→1.52).

Modes: (1), (2) and (4) act in vorticity mode only; the rest act in both.

## Safety ("don't mess up the rest")

- **Off = byte-identical by construction**: every path compiles only in the
  `HERO_EMERGENCE` preprocessor variant, selected by `solver._domain_defines`
  when `hero_emergence > 0` AND a hero exists (`hero_count` or a cast-list
  hero) — the default program text is the pre-feature kernel (the `#else`
  branches carry the pre-feature lines verbatim). p05 9/9 green throughout.
- **Hero-local**: every window returns exactly 1.0 (or contributes 0) beyond
  q<3.6 of a hero; a hero-free config never compiles the variant at all
  (tested), so it never pays the per-pixel vortex scan.
- **Global chaos caveat (measured)**: in vorticity mode ANY real change to the
  flow re-rolls fine detail globally (butterfly effect). Control experiment:
  identical-params rerun noise floor = 0.106 far-field / 0.518 whole-field max
  pixel diff; emergence-on vs off = 0.351 / 0.475 — *within the run-to-run
  envelope the preset already has*. Full-disk A/B confirms all large-scale
  structure (bands, jets, ovals, poles) is preserved.
- Kinematic source-hash pins updated deliberately (advect.comp,
  vortex_stamp.glsl); vorticity kernels are not pinned (documented ~1e-3 SOR
  noise floor).

## Tests

`tests/gpu/test_hero_emergence.py` (byte-exact asserts are kinematic-mode only,
asserted in the param helper — vorticity output is never byte-compared):
- emergence=0 determinism canary (identical params, two full builds; the real
  off == pre-feature guarantee is the source-hash pins + p05, not runtime)
- emergence>0 with NO hero: default program selected (predicate pin) =>
  byte-identical
- forced-variant no-op, hero-local edition: variant compiled (hero present),
  effect near the hero + far-field byte-identity
- **anchor test**: vorticity mode, developed, mean T3 over the interior at the
  registry position > 0.3 (regression guard for the stamp/core divergence)

## Presets (fleet rollout, 2026-07-10)

Validated off-vs-on across every hero-bearing factory preset:

- `gas_giant_warm` — bakes 0.8 (build_warm_preset.py). The calibration preset.
- `jupiter_vorticity` — bakes 0.8 (build_vorticity_presets.py). Largest win in
  the fleet: its big (r 0.16) hero suffered the core/stamp divergence worst and
  smeared into a half-dissolved arc; now a filled red oval in a hollow.
- `neptune` — bakes 0.8 (build_neptune_preset.py). Sign-agnostic proof: the
  plateau fill makes the NEGATIVE-brightness GDS a coherent filled dark oval
  (Voyager's GDS was exactly that); the 3 bright companions ride the flush
  target (they are stamps) and survive at the oval's edge.
- `jupiter_like`, `ice_giant` (kinematic) — validated no-breakage (ring/anchor
  are vorticity-only no-ops; plateau/fade/flush act mildly on the Gaussian
  hero). NOT baked: jupiter_like is a p05 hash-gate config (baking would force
  a baseline advance for a marginal win), ice_giant's hero is a faint wisp.
- `saturn_pale` — hero_count 0; lever is a tested no-op.

## History

The first design cut only the relaxation falloff (including over the interior).
Visual result was near-invisible: at solid_core=1.0 the rigid interior has no
shear to reveal, and the stamp/core divergence meant stamp-side changes never
landed on the visible storm. A second cut added interior eddy injection — wrong
per the reference (the real interior is calm) and removed. The
reference-anchored redesign above replaced both.

**Quiet-storm pass (2026-07-10).** User: "we don't feel close." A perceptual
decomposition (color transfer / texture emphasis / basin dimming mockups)
identified the gap as LOUDNESS, not shape or color: the real GRS close-up is
quiet, ours was heavily textured — and the legacy detail layer was actively
boosting churn inside the hero mask. Mechanism path 7 is the fix; the sim-side
collar/ring brightness and mottle/tint_var amplitudes were softened in the same
pass.

**Footprint compaction + partial shield (2026-07-10).** User: "the effect the
hero storm has seems much bigger than its actual size." Two causes. (a) The
emergence anatomy had inflated the oval itself to ~1.55× the authored
`hero_radius` (plateau edge 1.55, collar 2.0, flush to 5.2, detail mask
×1.84): everything was rescaled ×~0.65 so the plateau edge sits AT q=1.0,
with interior wisp/lane frequencies scaled up to preserve the approved look.
(b) The deeper cause: the single-signed ring/disk vorticity carries net
circulation, whose 1/r far field wound the neighborhood into a pinwheel ~5
oval-diameters wide regardless of (a). A FULL concentrated shield
(+2.0 over q 1.04–2.0, exact cancellation) was falsified immediately: it
rolled up into a companion cyclone and the near-dipole self-propelled off the
anchor. The shipped fix is the partial (~70%), wide, weak skirt (path 2)
paired with a stronger ×6 band-flush (path 6). Also falsified in passing:
shrinking the anchor capture basin to the compacted anatomy (path 4's note).
