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

### Pending — genuine calibration sweeps (batch 3)
F08 brown barges · F13 5-µm hotspots · F14 Saturn ribbon wave · F20 intermittent turbulence ·
F23 temperate lace mottle · F29 Saturn polar hexagon · A08 Uranus polar hood · A10 cyclone–anticyclone asymmetry · A05 belt–zone emission contrast.

*(dispositions filled in as each batch completes)*
