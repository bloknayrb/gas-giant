"""Build the gas_giant_warm factory preset (the frost-fix deliverable).

Pure appearance + preset-level sim params; NO kernel changes. The frost fix is
the high-contrast warm palette (real dark->bright value range across the T0
axis), applied uniformly across latitude rows so band color is T0-structure-
driven rather than latitude-stamped. Flow config is the loosened-leash warm
setup from the art pass.

Run: ./.venv/Scripts/python.exe scripts/build_warm_preset.py
"""
from __future__ import annotations

from pathlib import Path

from gasgiant.params.model import GradientStop, PaletteRow
from gasgiant.params.presets import load_factory_preset, save_preset

OUT = Path("src/gasgiant/presets/gas_giant_warm.json")

# High-contrast RUST palette across the T0 axis (the frost fix). Real dark->bright
# value range is what defeats the frosted-glass look; the rust hue is the chosen
# identity (redder, saturated belts; warm-cream cloud tops).
CONTRAST_STOPS = [
    (0.00, (0.14, 0.06, 0.04)),  # near-black brown (deep belt gaps)
    (0.22, (0.46, 0.17, 0.10)),  # saturated rust
    (0.45, (0.68, 0.36, 0.20)),  # rust-tan
    (0.68, (0.86, 0.62, 0.40)),  # warm sand
    (0.85, (0.94, 0.82, 0.62)),  # cream
    (1.00, (0.99, 0.94, 0.83)),  # bright cloud top
]


def main():
    p = load_factory_preset("jupiter_vorticity")

    # Loosened-leash warm flow (the de-imposed, frost-free develop). The high
    # replenish_rate feeds fresh detail faster than the zonal jets smear it away,
    # so the quiescent zone bands (which the jets otherwise starve of detail to
    # ~half the belts') fill with fine filaments instead of reading smooth.
    p.turbulence = p.turbulence.model_copy(update={
        "relax_tau": 2000.0,
        "belt_boost": 1.0,
        "shear_coupling": 0.7,
        "intensity": 1.3,
        "belt_replenish": 0.0,
        "replenish_rate": 0.35,
    })
    # Flow-negotiated hero edge (the de-stamped GRS): fade the hero's relaxation
    # forcing through the rim/collar/near-interior so the storm's own vortex
    # velocity folds ambient tracer into a ragged, filament-shedding boundary
    # woven into the band, instead of a clean analytic template re-imposed every
    # step. The deep-core anchor keeps full relaxation, so the oval stays
    # coherent (solid_core=1.0 preserved) and everything beyond the storm is
    # byte-identical. 0.8 = strong effect, still a distinct storm (calibrated by
    # emergence x solid_core sweep, 2026-07-09).
    # GRS BAKE (2026-07-19, plans hero-bracket-size-relative + hero-jet-
    # environment): the hero is seated by the size-relative carve-and-impose jet
    # BRACKET (jets.hero_bracket_*, set below), which SUPERSEDES the earlier
    # additive local_jet one-sided bearing. The bracket erases the seeded band
    # jets in a feathered hero-core-radius window and imposes an authored
    # two-sided anticyclonic shear (equatorward westward + poleward eastward), so
    # the storm co-rotates with an artist-authored, seed-independent environment
    # instead of a local jet hand-tuned to warm's seeded field. local_jet is
    # REMOVED (the bracket carves the same window; the developed look is unchanged
    # with it off, verified 2026-07-19). All below is a USER visual checkpoint,
    # reference-anchored vs PIA07782 (detail-off calibration + detail-on look):
    # - hero_latitude -22.0 -> -24.0: the storm sits ON the reddish band boundary
    #   in a pale lane (the Red Spot Hollow straddle), the bracket bowing the
    #   surrounding bands around it. -24 was walked in from -22 to sit on the
    #   boundary; the bracket (not the seeded field) now supplies the bearing, so
    #   the old chirality latitude/local_jet grid search no longer constrains it.
    # - hero_radius 0.062 -> 0.108 (~6.2 deg lat semi-axis; the real GRS dominates
    #   its band): the bigger storm reads in the right proportion to its band vs
    #   the reference. The size-relative bracket geometry tracks this radius
    #   automatically (offset/window/feather/widths are in core-radius units), so
    #   growing the storm keeps the bracket straddling it with no manual re-tune.
    # - rim_contrast 2.0 -> 1.3: the doubled ring/collar amplitude was a main
    #   driver of the etched-rings look; the real Hollow is only slightly
    #   brighter than the bands.
    # - wake_turbulence 3.2: now LIVE in vorticity mode (omega_force wake-wedge
    #   injection); 1.593 was calibrated when the lever was feather-only inert.
    # - hero_wake_detail 1.0, hero_mottle 0.9: full fray on the dimmed wake
    #   stamp; granular (still muted) interior.
    # - hero_emergence 0.8 -> 0.9: the retuned pack endpoints land at 90%.
    # Round B (de-bullseye + populate the neighborhood, same plan):
    # Aspect (2026-07-19 GRS bake, was 2.9): user picked hero_aspect 2.0 -- a
    # plump GRS oval (develops ~1.8, close to the reference's pale envelope) over
    # the more-elongated lozenge that authored 2.9 develops to (~2.4). The earlier
    # 2.9 bake targeted the reference RED CORE's ~2.9 elongation; at the
    # -24/r0.108 checkpoint the plumper whole-storm read was preferred on a
    # 2.0-vs-2.9 side-by-side. storms.hero_flow_aspect stays at its 1.0 default.
    # - hero_aspect 2.9 -> 2.0 (user checkpoint, 2026-07-19).
    # - hero_tint_var 0.55 -> 0.35: the interior now carries a deterministic
    #   T3 spiral lane; the isotropic tint fbm was the same magnitude as the
    #   lane signal (S/N ~ 1) and must come down for the banding to read.
    # - hero_rim_warp 0.65 -> 1.0: geometry-level radius lobes on the stamped
    #   rings (sub-sigma at 0.65 against the k=34 annulus = "amplitude-level").
    # - hero_companions 2: bright clouds hugging the wake-free flank
    #   (reference: white clouds packed against the GRS's leading side).
    # - accent_count 1 at -29 deg: an Oval-BA analog shearing past south of
    #   the hero; longitude auto-roots 0.3-0.55 rad downstream (hero-relative
    #   placement rule). oval_solid_core 1.0 keeps it (and the larger white
    #   ovals) coherent through the dev run (the F07 pairing).
    # Hero jet environment: the size-relative bracket seats the -24 hero. Geometry
    # is authored in hero-core-radius units at the model defaults (offset +/-1.0,
    # window 1.0, feather 1.4, widths 0.8) so it tracks hero_radius; only the
    # strengths are authored (N equatorward westward -3.0, S poleward eastward
    # +3.0 = anticyclonic seating; effective peak carries jets.strength). Set
    # explicitly (not relying on model defaults) so a future default drift cannot
    # silently move the bake. local_jet removed -- superseded by the bracket.
    p.jets = p.jets.model_copy(update={
        "local_jet_speed": 0.0,
        "hero_bracket_north": -3.0,
        "hero_bracket_south": 3.0,
        "hero_bracket_north_offset": 1.0,
        "hero_bracket_south_offset": -1.0,
        "hero_bracket_window": 1.0,
        "hero_bracket_feather": 1.4,
        "hero_bracket_north_width": 0.8,
        "hero_bracket_south_width": 0.8,
    })
    p.storms = p.storms.model_copy(update={
        "hero_radius": 0.108,
        "hero_latitude": -24.0,
        "hero_aspect": 2.0,
        "hero_emergence": 0.9,
        "rim_contrast": 1.3,
        "wake_turbulence": 3.2,
        "hero_wake_detail": 1.0,
        "hero_mottle": 0.9,
        "hero_tint_var": 0.35,
        "hero_rim_warp": 1.0,
        "hero_companions": 2,
        "companion_brightness": 0.55,   # reference flank clouds are among the
                                        # brightest pixels; 0.32 read as a
                                        # faint smudge on the pale moat
        "accent_count": 1,
        "accent_latitude": -29.0,
        "accent_radius": 0.05,
        "accent_tint": 0.65,            # round-2 review: a salmon canvas has
        "accent_brightness": -0.18,     # NO chroma headroom — "redder" is
                                        # invisible (R-B delta ~0-11). The
                                        # accent must read via VALUE, like the
                                        # reference's brown ovals and the
                                        # Neptune GDS (negative brightness =
                                        # dark storm, the W5 lever's dark path)
        "oval_solid_core": 1.0,
        "oval_density": 3.5,            # bounds raised 3.0 -> 4.0: the SW/S
        "small_density": 3.5,           # neighborhood counted ~0.2x the
                                        # reference's incident density
    })

    # Hero-adjacent festoon train (FESTOON2): streamers rooting on the belt
    # edge the hero straddles (-19.4 deg), weaving through the wake lane —
    # the "neighborhood busy with unrelated weather" reference cue.
    # Wavenumber deliberately != the primary's 20 (twin wavenumbers read as
    # a mechanical comb). Originally planned subordinate to the primary
    # train; raised to parity in round B —
    # amplitude 1.6 (match the primary train) + wavenumber 14: at 1.0/k11 the
    # per-plume jitter left whole 30-deg windows plumeless and nothing read at
    # the root latitude (round-B adversarial review).
    p.waves = p.waves.model_copy(update={
        "festoon_hero_strength": 1.6,
        "festoon_hero_wavenumber": 14,
    })

    # Uniform detail coverage: apply the flow-folded detail-FX texture at EVEN
    # density across latitude instead of leaving zones detail-starved. 0.36 was
    # the calibrated value (user sign-off 2026-07-07) — even, fluid, not patchy.
    # hero_collar_wrap 0.5 -> 0.85: the thin bright annulus should read as
    # STREAKED along-flow (wound collar lanes), not as a smooth painted ring.
    p.detail = p.detail.model_copy(update={"spread": 0.36, "hero_collar_wrap": 0.85})

    # The frost fix: high-contrast warm palette, uniform across latitude rows
    # (color follows T0 structure, not a per-latitude stamp). chroma_scale 1.0 —
    # the palette already carries the hue, no Oklab re-cast.
    stops = [GradientStop(pos=pp, color=c) for pp, c in CONTRAST_STOPS]
    rows = [PaletteRow(latitude=r.latitude, stops=stops) for r in p.appearance.palette_rows]
    p.appearance = p.appearance.model_copy(update={
        "palette_rows": rows,
        "chroma_scale": 1.0,
        # Two-material detail tint (S2-calibrated 2026-07-16, user sign-off):
        # 0.6 delivers the bright-cool/dark-warm material read while leaving
        # headroom for detail levers that stack stronger excursions through
        # the same vehicle (a planned wake-braid lever). Judge recommended
        # 0.6 over 1.0 for exactly that stacking reason.
        "detail_chroma": 0.6,
    })

    p.name = "gas_giant_warm"
    save_preset(p, OUT, name="gas_giant_warm")
    print("wrote", OUT, flush=True)


if __name__ == "__main__":
    main()
