"""Build the `ember_dwarf` factory preset — a cloudy L/T-transition brown dwarf.

Not a planet: a failed star. The preset library was six solar-system bodies, and
every one of them is lit by reflected sunlight off a bright cloud deck. This is
the opposite regime, and it is the one the engine's turbulence work actually
flatters most: a DARK deck with BRIGHT tears in it.

The physics being expressed (all of it real, none of it invented for looks):

  * **Color.** Brown dwarfs are not brown. Broad Na I D and K I absorption eats
    the middle of the visible spectrum, leaving red plus blue/violet, so a T
    dwarf reads dimly MAGENTA to the eye; L dwarfs are red. This preset sits at
    the L/T transition and takes a dark magenta-plum cloud deck as its canvas.
  * **The tears.** The L/T transition is *defined* by the silicate/iron cloud
    deck breaking up into patches. Through the holes you see down to hotter gas,
    which is why these objects vary in brightness as they rotate. So the bright
    end of every palette row is ember orange into gold — not white cloud tops.
    Bright = a hole, not a cloud. That inverts the usual gas-giant reading and
    is the whole idea of the preset.
  * **Fast rotation.** Periods of 2-5 hours (vs Jupiter's 10) mean a stronger
    Coriolis constraint and a smaller deformation radius: more bands, tighter
    eddies, a vigorous field. Hence coriolis_f0 up, L_d down, count up.

Structural retune from gas_giant_warm (the proven vorticity engine), following
the neptune lesson that a different object is a whole-preset job, not a recolor:

  * seeded bands instead of Jupiter's baked band template (which also unlocks
    value_contrast -- the template path uses its values VERBATIM and ignores it);
  * the storm field re-cast as cloud clearings: a bright gold hero (a giant
    hole, the inverse of the GRS's dark-cored red oval) with no red collar;
  * emission actually populated, because for once it is not decorative: these
    objects genuinely glow at depth. Blender users get the ember through the gaps.

Run: uv run python scripts/build_ember_dwarf_preset.py
"""
from __future__ import annotations

from pathlib import Path

from gasgiant.params.model import GradientStop, InjectMask, PaletteRow
from gasgiant.params.presets import load_factory_preset, load_preset, save_preset

PRESETS_DIR = Path("src/gasgiant/presets")

# Four latitude PROFILES (equator/midlat/subpolar/polar), mirrored into seven
# palette rows at build time -- see the comment on `rows` for why the mirroring is
# mandatory. Each runs near-black violet -> dark plum -> magenta -> ember -> gold,
# and what changes with latitude is how far up that ramp the deck lets you see:
# the equator reaches gold, the poles are clamped well short of it. The bright end stays CHROMATIC (gold, not white) -- the
# "frosted glass" failure that dogged the warm preset was a desaturated bright end,
# not a value-range problem.
#
# The stop POSITIONS are measured, not guessed, and they are COUPLED to the field
# -- if the band/turbulence tuning below ever moves, RE-MEASURE before trusting
# them. scripts/probe_lut_usage.py swaps in a grayscale ramp so rendered
# luminance IS the palette index. For the shipped tuning it reports a bell centred
# near 0.57 with ~10% of the disc above 0.80 and ~4% above 0.90.
#
# Both directions of getting this wrong were hit while building the preset:
#   * ember knee too EARLY (0.72) painted ~65% of the disc ember-or-brighter and
#     rendered a uniform lava red -- no dark deck, the whole premise lost;
#   * ember knee too LATE (0.93) sat past p99, so the tears vanished entirely and
#     the planet read as a flat purple ball with one gold spot.
# The knee is now ~0.84, just above the bulk, which spends the top ~10% of the
# distribution on fire and leaves the remaining 90% as dim magenta deck.
EQUATOR = [   # hottest: tears open widest and run gold
    (0.00, (0.048, 0.020, 0.078)),
    (0.28, (0.105, 0.034, 0.130)),
    (0.52, (0.200, 0.055, 0.215)),
    (0.72, (0.360, 0.090, 0.290)),
    (0.84, (0.620, 0.170, 0.230)),
    (0.93, (0.900, 0.360, 0.130)),
    (1.00, (1.000, 0.830, 0.480)),
]
MIDLAT = [    # the banded deck proper: magenta dominant, ember only in gaps
    (0.00, (0.040, 0.018, 0.072)),
    (0.28, (0.092, 0.030, 0.125)),
    (0.53, (0.175, 0.050, 0.210)),
    (0.73, (0.310, 0.078, 0.285)),
    (0.86, (0.540, 0.145, 0.235)),
    (0.94, (0.830, 0.320, 0.140)),
    (1.00, (0.985, 0.780, 0.440)),
]
SUBPOLAR = [  # deck thickening: tears go dull rose and never reach gold
    (0.00, (0.034, 0.016, 0.068)),
    (0.30, (0.076, 0.028, 0.122)),
    (0.55, (0.140, 0.046, 0.205)),
    (0.76, (0.235, 0.072, 0.275)),
    (0.88, (0.360, 0.120, 0.265)),
    (1.00, (0.500, 0.215, 0.260)),
]
# The polar rows exist to CAP the caps. Poleward of ~50 deg the field is laminar
# (the shear injection mask plus polar_decay leave no eddies there), so whatever
# value the outermost seeded bands happen to hold gets painted flat across the
# whole cap -- and at sim resolution 4096 that produced blazing gold laminar
# streaks over the south pole even after the polar TINT had been strengthened
# enough to look fixed at 1024. Tint alone cannot fix it: it is a hue push, and
# the problem is LUMINANCE. Clamping the bright end of the polar rows is the only
# dynamics-independent fix -- there is simply no bright colour available up here,
# whatever the band lottery or the resolution does.
POLAR = [
    (0.00, (0.030, 0.015, 0.062)),
    (0.34, (0.062, 0.026, 0.110)),
    (0.62, (0.105, 0.042, 0.175)),
    (0.84, (0.165, 0.065, 0.225)),
    (1.00, (0.255, 0.115, 0.250)),
]

# Storm LUT: storms here are CLEARINGS, so the bright end is the hot gold of
# exposed depth and the dark end is a thick plum cloud lid.
STORM_TINTS = [
    (0.00, (0.09, 0.03, 0.11)),
    (0.55, (0.42, 0.11, 0.19)),
    (1.00, (1.00, 0.82, 0.48)),
]

# Longer than warm's 700: the higher injection + smaller L_d need depth to fold
# the deck into genuinely torn patches rather than smooth wavy bands.
SIM = {"dev_steps": 900}

# Fast rotator, vigorous field. coriolis_f0 3 -> 5.5 tightens the geostrophic
# constraint (more, narrower bands); L_d 0.18 -> 0.11 shrinks the eddies;
# injection up to tear the deck. psi_drag carries most of the scale control --
# it is the scale-SELECTIVE hypofriction that stops the inverse cascade piling
# every eddy into one gravest-mode blob, and raising it is what lets injection
# go this high without the field collapsing into a few fat swirls.
SOLVER = {
    "coriolis_f0": 5.5,
    "deformation_radius": 0.11,
    "vort_inject": 2.6,
    "vort_inject_scale": 3.2,
    "vort_inject_mask": InjectMask.SHEAR,
    "vort_hypervisc": 0.35,
    "vort_psi_drag": 0.45,
}

# Seeded bands (count-driven), NOT Jupiter's baked template.
#
# value_contrast is the trap on this path. warm ships 1.7, but warm ships a
# TEMPLATE, and the template path uses its values VERBATIM and ignores
# value_contrast entirely -- so warm's 1.7 is inert and inheriting it here was
# inheriting a number that does nothing in its source and everything here. At 1.9
# it saturated every band to an extreme (zones 0.94-1.0, belts 0.00-0.17, no
# midtone anywhere): the deck became a binary square wave, the magenta mid-ramp
# went unaddressed, and a 12.8-deg-wide zone pinned at value 1.000 rendered as a
# flat razor-edged gold ring at -53 where the poleward field is too calm to break
# it up. 1.15 keeps real zone/belt separation while leaving the values spread
# across the middle of the ramp, which is where the deck's color lives.
BANDS = {
    "template": None,
    "count": 24,
    "value_contrast": 1.15,
    "width_jitter": 0.32,
    "edge_softness": 0.012,
    "detail_amount": 0.18,
    "contrast_envelope": 0.36,
    "edge_diversity": 0.6,
    "variance_amount": 0.28,
}

JETS = {
    "strength": 0.95,
    "equatorial_speed": 2.1,
    "equatorial_width": 0.16,
    # No authored hero bracket: warm's is seated for the GRS at -24 and this
    # hero is a different object at a different latitude.
    "local_jet_speed": 0.0,
    "hero_bracket_north": 0.0,
    "hero_bracket_south": 0.0,
}

# The tears are made HERE, not by the palette. Holding value_contrast moderate
# leaves the band skeleton in the middle of the ramp, so what pushes a pixel up
# into ember/gold is a convective excursion -- which is the honest mechanism (real
# L/T cloud holes are torn open by convection) and distributes the tears across
# every belt instead of parking them in whichever band the seed happened to pin at
# 1.0. belt_boost concentrates the churn in the cyclonic belts and belt_replenish
# is the engine's "emergent filaments" term; together they supply the ~10% of the
# disc that clears 0.8 and the ~4% that clears 0.9 (measured, detail on).
TURBULENCE = {
    "intensity": 1.7,
    "shear_coupling": 0.85,
    # scale 6.5 (not the finer 9.0 first tried): finer noise made the tears read
    # as scattered speckles, and what sells a torn cloud deck is FEWER, LONGER
    # openings drawn out along the belts.
    "scale": 6.5,
    "replenish_rate": 0.5,
    "belt_boost": 2.6,
    "belt_replenish": 0.05,
    "belt_replenish_scale": 1.4,
    "kh_amplitude": 1.0,
}

# The hero is a giant cloud CLEARING, not a red spot: bright gold core
# (hero_tint 1.0 -> top of the storm LUT, positive brightness), and the
# Jupiter-only red-collar levers zeroed. Solid core + emergence keep it a
# coherent oval rather than a whirlpool.
STORMS = {
    "hero_tint": 1.0,
    "hero_brightness": 0.3,
    "hero_radius": 0.1,
    "hero_latitude": 18.0,
    "hero_aspect": 1.8,
    "hero_rim_tint": 0.0,
    "hero_rim_warp": 0.7,
    "hero_solid_core": 1.0,
    "hero_emergence": 0.9,
    "hero_mottle": 0.7,
    "hero_companions": 2,
    "companion_brightness": 0.5,
    # A busy, storm-rich disc: these are rapidly rotating convective objects.
    "oval_density": 4.0,
    "small_density": 4.0,
    "barge_density": 2.0,
    "pearls_count": 10,
    "outbreak_count": 3,
    "outbreak_strength": 1.3,
    "stamp_contrast": 2.6,
    "accent_latitude": None,
    "accent_tint": 1.0,
    "accent_brightness": 0.35,
}

WAVES = {
    "festoon_strength": 1.3,
    "festoon_hero_strength": 0.0,
    "ribbon_strength": 0.0,
}

DETAIL = {
    "intensity": 1.0,
    "spread": 0.5,
    "belt_texture": 1.3,
    "belt_texture_fine": 1.5,
    "mottle": 1.2,
    "cellular_amount": 1.0,
    "intermittency": 0.7,
    "polar_filaments": 0.9,
    "polar_stipple": 0.5,
    "hero_calm": 0.6,
    "hero_spiral": 0.4,
    "hero_collar_wrap": 0.0,
}

# Dark violet polar caps via polar_tint (NOT polar_canvas_value, whose floor is
# a dark blue-TEAL and would fight a magenta world).
APPEARANCE = {
    "contrast": 1.08,
    # chroma_scale is the Oklab (perceptual) saturation multiplier and is the
    # recommended lever; `saturation` is an sRGB luma mix, left at 1.0.
    "saturation": 1.0,
    "chroma_scale": 1.3,
    "haze_amount": 0.03,
    "haze_color": (0.42, 0.20, 0.26),
    "chroma_variance": 0.5,
    "hue_variance": 0.24,
    # Both of these were measured pushing the deck RED and destroying the magenta
    # the whole preset is built on, so both are off:
    #   detail_chroma shades DARK detail excursions toward "warm belt material"
    #     -- and on a dark deck nearly every pixel is a dark excursion, so warm's
    #     baked 0.55 dragged the entire planet to maroon (neptune zeroes it for
    #     the same reason, from the cool side);
    #   chroma_aging adds "reddish-brown chromophore" with tracer age, which is a
    #     Jupiter chromophore story and simply the wrong pigment here.
    "detail_chroma": 0.0,
    "chroma_aging": 0.0,
    # The polar caps get no eddy injection (the shear mask plus polar_decay leave
    # the caps laminar), so whichever value the outermost seeded band lands on
    # paints a FLAT cap -- a seed lottery that on this seed produced a bright pink
    # laminar smear over the whole southern cap, brighter than the deck and reading
    # as a lighting error. Decisive violet tinting from 50 deg, plus a firmer
    # poleward contrast collapse (bands.contrast_envelope), makes the caps read as
    # cold cloud regardless of which value the lottery hands them.
    "polar_tint_strength": 0.74,
    "polar_tint_start_lat": 50.0,
    "polar_tint_color": (0.20, 0.09, 0.28),
    "polar_canvas_value": 0.0,
}

# Emission is normally decoration here (invisible in the GUI preview, Blender
# only). On a brown dwarf it is the point: the object really does glow through
# its cloud gaps. Keyed high so only the tears blaze.
# Vigorous polar cyclone clusters suit a fast rotator, but at 1.8 they rendered as
# blazing gold blobs sitting in a cold cap (any bright storm samples the gold top of
# the storm LUT). 1.15 keeps the cap populated without lighting it up.
POLES = {"strength": 1.15, "field_density": 1.15}

EMISSION = {
    "thermal_strength": 0.9,
    "thermal_color": (1.0, 0.52, 0.16),
    "thermal_threshold": 0.48,
    "thermal_hdr": 6.0,
    "lightning_strength": 0.0,
    "aurora_strength": 0.0,
}


def _stops(spec):
    return [GradientStop(pos=p, color=list(c)) for p, c in spec]


def build() -> None:
    p = load_factory_preset("gas_giant_warm")
    p.sim = p.sim.model_copy(update=SIM)
    p.solver = p.solver.model_copy(update=SOLVER)
    p.bands = p.bands.model_copy(update=BANDS)
    p.jets = p.jets.model_copy(update=JETS)
    p.turbulence = p.turbulence.model_copy(update=TURBULENCE)
    p.storms = p.storms.model_copy(update=STORMS)
    p.waves = p.waves.model_copy(update=WAVES)
    p.detail = p.detail.model_copy(update=DETAIL)
    p.poles = p.poles.model_copy(update={
        "north": p.poles.north.model_copy(update=POLES),
        "south": p.poles.south.model_copy(update=POLES),
    })
    p.emission = p.emission.model_copy(update=EMISSION)
    # Rows MUST be mirrored across the equator. bake_rows blends on SIGNED
    # latitude (-90..+90) and CLAMPS outside the outermost anchor, so a
    # northern-only ladder (0/32/56/78) silently paints the entire southern
    # hemisphere with the row at 0.0 -- here the hottest, gold-capable EQUATOR
    # row, right over the south pole. That is what produced a bright orange
    # laminar south cap that survived every dynamics-side fix, and it was
    # initially misread as a band-value seed lottery. The `latitude` field is
    # signed, not an absolute value; gas_giant_warm ships 12 rows from -78.5 to
    # +66 for exactly this reason.
    rows = [
        PaletteRow(latitude=-78.0, stops=_stops(POLAR)),
        PaletteRow(latitude=-56.0, stops=_stops(SUBPOLAR)),
        PaletteRow(latitude=-32.0, stops=_stops(MIDLAT)),
        PaletteRow(latitude=0.0, stops=_stops(EQUATOR)),
        PaletteRow(latitude=32.0, stops=_stops(MIDLAT)),
        PaletteRow(latitude=56.0, stops=_stops(SUBPOLAR)),
        PaletteRow(latitude=78.0, stops=_stops(POLAR)),
    ]
    p.appearance = p.appearance.model_copy(
        update={**APPEARANCE, "palette_rows": rows, "storm_tints": _stops(STORM_TINTS)}
    )

    out = PRESETS_DIR / "ember_dwarf.json"
    save_preset(p, out, name="ember_dwarf")
    # save_preset does NOT re-validate; load_preset does. Prove in-bounds.
    r = load_preset(out)
    assert r.bands.template is None
    assert r.bands.count == 24
    # The trap this preset was bitten by: warm's value_contrast 1.7 is inert on
    # its template path and saturates the band table on the seeded path.
    assert r.bands.value_contrast == 1.15
    assert r.appearance.detail_chroma == 0.0   # warm bakes 0.55; drags deck red
    assert r.appearance.chroma_aging == 0.0    # warm bakes 0.35; wrong pigment
    assert r.solver.vort_inject_mask == "shear"  # global destroys the banding
    assert r.solver.coriolis_f0 == 5.5
    assert r.storms.hero_tint == 1.0
    assert r.emission.thermal_strength == 0.9
    assert len(r.appearance.palette_rows) == 7
    lats = [row.latitude for row in r.appearance.palette_rows]
    # BOTH hemispheres must be anchored. bake_rows clamps outside the outermost
    # anchor, so a northern-only ladder paints the whole south with the equator
    # row (see the comment on `rows`). This guard is the regression trap.
    assert min(lats) <= -70.0 and max(lats) >= 70.0, lats
    # Neither polar row may reach a bright value: this is what holds the laminar
    # caps dark independently of the band lottery or the sim resolution.
    for row in r.appearance.palette_rows:
        if abs(row.latitude) >= 70.0:
            assert max(row.stops[-1].color) < 0.30, row.latitude
    print(f"wrote + verified {out}", flush=True)


def main() -> None:
    build()


if __name__ == "__main__":
    main()
