# W14 — Multi-seed calibration finale: per-row dispositions

*Addendum to `2026-07-02-comprehensive-review.md` §4 (coverage matrix). Records the
final disposition of the 19 `candidate_poor` partials + F28 + A05, after the Top-10
lever work (W3 chirality, W5 per-storm color, W6 band/epoch, W12 sequence) landed.*

**Protocol:** each row gets a multi-seed render pass (seeds 4201/7/23, 2048²) and an
adversarial visual read vs the reference; disposition is one of **RESOLVED** (the lever
work discharged it), **TUNE** (calibration change applied this wave), **ACCEPT** (partial
is the honest ceiling; no cheap improvement), or **DEFER** (fenced / falsified / research-grade —
recorded, not attempted). Falsified approaches are never re-attempted (emergent color,
Rhines jets, SL advection, polar-discreteness merger physics, pre-merge co-orbiting).

Started 2026-07-05 (in-session, user-driven; the first background-agent attempt died on
an account logout 2026-07-03 with no committed work).

## Dispositions

### Forced DEFER (falsified or fenced — record only, no work)
| Row | Phenomenon | Disposition | Rationale |
|-----|-----------|-------------|-----------|
| F03 | Alternating zonal jets / superrotation | DEFER | Emergent Rhines jets are research-grade, explicitly gated off (roadmap). Prescribed jets already ref-matched; the "partial" is the emergent-jet ceiling, not a lever gap. |
| F10 | Merge debris collar | DEFER | Pre-merge co-orbiting vortices are a recorded intentional deferral (animation release); the merger gate's purity blocks it. Debris collar itself is present. |
| F26 | Iso-luminance hue drift | DEFER | Emergent color FALSIFIED 2026-06-23 (passive tracer washes bands in both modes). Do not re-attempt. |
| F28 | Jupiter polar cyclone clusters | DEFER | Discreteness blocked on vortex-merger physics (research-grade). Lace + teal cap already ref-matched at map scale vs PIA21641. |

### Confirm intervening-wave fix (batch 1: seeds 4201/7/23)
| Row | Phenomenon | Disposition | Evidence |
|-----|-----------|-------------|----------|
| F12 | Festoons (blue-gray streamers) | RESOLVED | W3 NEB-south fix: both gas_giant_warm and jupiter_vorticity root festoons on the belt's south edge, curling into the cream EZ, across all three seeds. |
| F09 | Vortex street / string of pearls | RESOLVED | gas_giant_warm shows a clean row of discrete round white ovals along the southern belt edge. |
| F07 | White ovals | ACCEPT (+demo) | Small `KIND_OVAL` below the solid-core gate stays Gaussian and shears to eddies in vorticity render (recorded limit). Expressible path = W5 accent ovals (`accent_radius` above the gate); demo pending. |
| F34 | Vortex rims / collars | ACCEPT | Hero collar convincing (bright, streamline-wrapped); small-oval rings faint but present — tracks the F07 limit. |
| F21 | GRS internal spiral lanes | DEFER | Collar streamlines read; internal spiral is weak, BUT spiral-amplification was already rejected this program (reads mechanical — same failure class as over-driven festoons). Accept the collar-only ceiling. |
| F06 | Turbulent wake of hero | ACCEPT | Kinematic flank fixed in W3. In vorticity mode (gas_giant_warm, seed default @2048) the hero sits embedded in the turbulent belt — folded filaments shear and wrap around it, an *emergent* turbulent wake rather than a bare vortex. There is no clean due-east trailing plume, and a *prescribed* directional trail is exactly the mechanical-looking lever class this program has repeatedly rejected (over-driven festoons, GRS spiral-amplification). The emergent folded-filament churn is the honest ceiling; no cheap lever improves it. |

### Scenario demos (batch 2, seeds as noted @2048)
| Row | Phenomenon | Disposition | Evidence |
|-----|-----------|-------------|----------|
| A06 | Neptune GDS + companions | RESOLVED (+demo) | The reversed-LUT dark hero (`hero_brightness -0.3`/`hero_tint -0.9`) over the coherent `hero_solid_core` oval, with `hero_companions 2`, expresses Neptune's Great Dark Spot + bright companion cloud. **First cut read as a "blue Jupiter"** — and the tell was NOT the palette but the retained Jupiter *structure*. The Neptune-authentic recipe (seed 4201, on the gas_giant_warm vorticity engine) is a whole-preset structural retune: `solver.vort_inject 1.8→0.15` + `vort_hypervisc →1.0` (the shear-masked eddy injection was folding the belts into Jupiter churn; dropping it gives smooth broad Neptune zones), `waves.festoon_strength 2.6→0`, `bands.count 22→7` / `value_contrast 1.7→0.55`, remove the planet-girdling field (`pearls_count`/`oval_density`/`small_density`/`barge_density →0`), calm `turbulence.intensity →0.8`, drop belt-texture detail, and a deep saturated methane-blue palette. Demo: `out/w14b2/a06_neptune_v3`. |
| A04 | Saturn GWS epoch | ACCEPT | Outbreak-pinned bright eruption (`outbreak_latitude`/`outbreak_phase`/`outbreak_strength`, W6) on saturn_pale (seed 9120) gives a legible, localized Great-White-Spot *onset* plume (`out/w14b2/a04_clean`). The planet-girdling streak the real 2010 GWS became is a *time-evolution* outcome (the eruption sheared around the planet over weeks); a static export can pin the eruption but not the wrapped streak, and forcing a full-longitude bright band would read as a painted stripe. Eruption expressible; girdling streak is the honest ceiling. |

*Demo lesson (A06):* expressing a **different planet** is a whole-preset structural retune — injection strength, band count/contrast, and the storm field — not a palette recolor. The GDS levers (reversed-LUT hero + solid-core oval + companions) were already sufficient; the surrounding preset was the work.

### Genuine calibration sweeps (batch 3, seeds 4201/7/23 @2048)

**Method:** rendered `jupiter_vorticity` (F08/F13/F20/F23/A10/A05), `saturn_pale`
(F14/F29), and `ice_giant` (A08) across seeds 4201/7/23; adversarial read of the color
map. A pre-sweep lever audit mapped each row to its controlling pfield(s) first, so the
question at each row was "does the existing lever already express it?" not "does a lever
exist?". **Outcome: no preset value was changed.** Every dedicated lever is already set in
the shipping presets — the expressible rows already read; the rest are honest ceilings
(no dedicated lever, or a projection/contrast limit). The thermal rows (F13 emission
component, A05) were judged on the visible channel + the `derive.comp` emission mechanism
and params; the `emission.exr` channel was not visually inspected in-session (no EXR
viewer in the env) — noted for honesty, it does not change either disposition.

| Row | Phenomenon | Disposition | Evidence |
|-----|-----------|-------------|----------|
| F08 | Brown barges | RESOLVED | `storms.barge_density 2.989` on jupiter_vorticity renders discrete dark elongated cyclonic features in the belts (dedicated `VKIND_BARGE` stamp, `dome=-1` cloud-top dip). They sit low-contrast against the rust belt — as on the real NEB. *Limit:* barge aspect/contrast is not separately tunable (only heroes have `hero_aspect`); shared `stamp_contrast` only. Present and legible across all three seeds. |
| F13 | 5-µm hotspots | RESOLVED | The visible cloud-clearing component reads strongly: `waves.festoon_strength 1.6` + `hotspot_depth 1.0` give the sawtooth blue-gray festoon scallops dipping into the equatorial-belt edge across all seeds. The 5-µm thermal channel is active by construction (`emission.thermal_strength 0.35`, `thermal_hdr 16`), glowing where the deck is depressed (`derive.comp`). |
| F14 | Saturn ribbon wave | ACCEPT | `waves.ribbon_strength` is active on saturn_pale (1.0) but does **not** read against Saturn's intrinsically low band contrast. A test bump to 2.6 (+1.5× band contrast) only crisped the bands — no legible standing meander emerged, and pushing further enters the painted-sine failure class this program repeatedly rejects (over-driven festoons, GRS spiral-amp). Compounded by the shader fixing the ribbon latitude (no param to pin the real ~47 °N). No cheap legible improvement — honest ceiling. |
| F20 | Intermittent turbulence | RESOLVED | `detail.intermittency 0.65` + `vort_inject_mask "shear"` give exactly the target mosaic: violent folded-filament patches abutting calmer laminar belt runs, not uniform churn. Reads across all seeds. |
| F23 | Temperate lace mottle | RESOLVED | `detail.mottle 1.1` (latitude-windowed to 35–60°) renders the granular lace/dot texture at mid-latitudes where banding gives way; `bands.contrast_envelope 0.6` collapses banding poleward into it. Present. |
| F29 | Saturn polar hexagon | ACCEPT | Present in the data — saturn_pale uses `poles.north.style=polygon_jet`, `polygon_sides=6`, which stamps the hexagonal polar jet. But the equirect color-map export smears the pole across the top row and Saturn's low contrast leaves it non-legible in the exported map. Expressible; validating the hexagon needs a polar-projection view, which the export/`validate` path does not produce. Not a lever gap. |
| A08 | Uranus polar hood | DEFER | ice_giant does not express a bright polar hood (`appearance.polar_tint_strength 0.0`; poles just fade). The audit confirms **no dedicated bright-polar-hood lever** — the only polar-cap lever (`polar_tint_*`) is engineered for a dark blue-gray Juno cap (applied where cloud tops are LOW), the opposite of a bright hood. A bright hood is only approximable via high-latitude `appearance.palette_rows` + `bands.template`; a first-class bright-cap lever is feature-grade, not a calibration. Fenced as a feature, not attempted. |
| A10 | Cyclone/anticyclone asymmetry | RESOLVED | The asymmetry reads correctly on jupiter_vorticity: anticyclones (`oval_density 3.0` + accents/heroes) are the bright, round, discrete features; cyclones (`barge_density`) are the darker, dipped, lower-contrast ones — the brighter-anticyclone half is baked into the shader stamp sign (`derive.comp dome=+1` vs barge `-1`) and anticyclone-dry lightning. No single "asymmetry" scalar, but the two-population design expresses it by construction. |
| A05 | Belt/zone emission contrast | ACCEPT | The belt-vs-zone thermal split is not a param — `derive.comp` computes thermal glow from the tracer's cloud-top depression vs the band stamp, so belts glimmer and zones stay dark as an **emergent** function of `bands.template` heights. `emission.thermal_strength 0.35` / `thermal_hdr 16` set the global amplitude/range. Calibrating the split means moving band heights, not a dedicated knob; a first-class belt/zone emission-ratio would be DEFER. Approximated by general params — honest ceiling. |

**Batch-3 tally:** 5 RESOLVED (F08, F13, F20, F23, A10) · 3 ACCEPT (F14, F29, A05) · 1 DEFER
(A08). Zero preset changes — every expressible row was already dialled in the shipping
presets; the ACCEPT/DEFER rows are projection limits (F29), intrinsic-contrast limits
(F14), or genuine no-dedicated-lever ceilings (A05, A08). This closes the 21-row
`candidate_poor`/F28/A05 disposition list opened at the top of this addendum.

*Batch-3 lesson:* the lever audit before rendering was the efficiency win — knowing each
row's controlling pfield up front turned nine "does the engine do this?" open questions
into nine "is the shipped preset value enough?" checks, and the answer was yes for every
row that has a dedicated lever. The three that landed ACCEPT/DEFER are the three the audit
had already flagged as having *no* dedicated lever (A08, A10-as-single-scalar, A05) plus
two display-medium limits (F14 contrast, F29 equirect polar-smear) — none is an engine gap.
