"""Build the `neptune` factory preset — an ice-giant scenario graduated from the
W14 batch-2 A06 demo (docs/reviews/2026-07-05-w14-calibration-addendum.md).

Neptune is NOT a recolored Jupiter. The batch-2 lesson: the Great Dark Spot levers
(reversed-LUT dark hero + coherent solid-core oval + bright companions) were already
sufficient; expressing a *different planet* is a whole-preset STRUCTURAL retune. So this
starts from gas_giant_warm's proven vorticity engine and:

  * drops the shear-masked eddy injection to ZERO (vort_inject 1.8 -> 0) with strong
    hyperviscosity -- injection folds the belts into Jupiter churn, and even a trace (0.06)
    left a mechanical Kelvin-Helmholtz sawtooth along one jet edge that read as a "blue
    Jupiter" tell under adversarial review. Laminar => smooth broad Neptune zones;
  * removes the planet-girdling Jupiter storm field (oval/barge/pearl/small -> 0) and the
    festoons, leaving only the dark GDS hero + 3 bright companion clouds;
  * collapses to a few soft low-contrast bands (count 22 -> 7, value_contrast 1.7 -> 0.55)
    with SMOOTH boundaries (no festoons, no ribbon) -- Voyager 2 shows no sharp band edge;
  * calms turbulence + strips the heavy Jupiter belt-texture detail;
  * a deep luminous methane cobalt-azure palette (matched to the Voyager reference; a first
    brighter cut read as pale periwinkle, so this is deepened + pulled off cyan), bluish
    haze, no polar tint/canvas;
  * ONE bright blue-white accent cloud (Scooter class) ELONGATED into a wispy east-west
    cirrus streak (accent_aspect), and the 3 GDS companion clouds likewise elongated
    (companion_aspect) -- real Neptune bright clouds are sheared streaks, not round dots.

The dark GDS hero is the reversed-LUT lever from W5 per-storm color: hero_tint -0.9 /
hero_brightness -0.3 over hero_solid_core (a coherent oval, not a whirlpool), with the
Jupiter-only rim/wake levers (rim_tint/rim_warp/wake_detail) zeroed -- the GDS has no red
collar or turbulent trailing wake.

The cloud-elongation levers (accent_aspect / companion_aspect) are a small engine capability
added in the same PR: they route accent/companion stamps through the hero's generic
elliptical-q aspect path plus a collar-free soft-streak glow branch. Both DEFAULT to 1.0
(round) and short-circuit that path, so every existing preset is byte-identical and the p05
render-hash baseline is unaffected. (True multi-strand "combed fiber" cirrus is NOT reachable
by stamping -- the sim advects/diffuses fine tracer detail into a wash over the dev run; it
needs a render-time synthesis pass on a dedicated cloud mask, recorded in docs/roadmap.md.)
The model_copy(update=) calls bypass pydantic bounds, so the preset is load_preset'd back at
the end to PROVE it is in-bounds (save does not re-validate).

Run: uv run python scripts/build_neptune_preset.py
"""
from __future__ import annotations

from pathlib import Path

from gasgiant.params.model import GradientStop, PaletteRow
from gasgiant.params.presets import load_factory_preset, load_preset, save_preset

PRESETS_DIR = Path("src/gasgiant/presets")

# Deep luminous methane cobalt-azure, matched to the Voyager 2 full-disk reference. A first
# brighter cut read as pale periwinkle/cyan under adversarial review, so this is deepened
# (rich cobalt shadows) and pulled off cyan (lower green vs blue): deep cobalt gaps -> saturated
# royal -> vivid azure -> pale blue-white bright-cloud caps (top stays off pure white).
DEEP_BLUE = [
    (0.00, (0.04, 0.11, 0.44)),
    (0.42, (0.07, 0.22, 0.66)),
    (0.75, (0.16, 0.42, 0.80)),
    (1.00, (0.55, 0.76, 0.95)),
]

# Storm-tint LUT (indexed by the storm T3 tracer): a blue ramp so the dark GDS core and the
# bright companion clouds read blue, not Jupiter brown. dark navy core -> mid azure -> blue-white.
STORM_TINTS_NEPTUNE = [
    (0.00, (0.10, 0.16, 0.32)),
    (0.50, (0.40, 0.55, 0.70)),
    (1.00, (0.85, 0.92, 0.97)),
]

# Longer than warm's 700 -- the calm low-injection field needs more steps to settle into
# broad smooth zones (the demo was calibrated at this depth).
SIM = {"dev_steps": 1256}

# Kill the Jupiter belt-churn engine AND the residual shear billow: vort_inject 0 (fully
# laminar zones -- even 0.06 left a mechanical Kelvin-Helmholtz sawtooth along one jet edge
# that read as a "blue Jupiter" tell), with strong hyperviscosity. The solid-core hero is a
# separate vorticity patch, so the coherent dark oval survives the calm field.
SOLVER = {
    "vort_inject": 0.0,
    "vort_inject_scale": 1.2,
    "vort_hypervisc": 1.5,
    "vort_psi_drag": 0.5,
}

# A few soft, low-contrast bands (Neptune is nearly bandless) with an early poleward
# contrast collapse.
BANDS = {
    "count": 7,
    "value_contrast": 0.55,
    "detail_amount": 0.0,
    "contrast_envelope": 0.35,
}

# The dark Great Dark Spot + 2 bright companion clouds. Reversed-LUT hero (dark anticyclone),
# coherent solid oval, and the Jupiter-only GRS rim/wake levers zeroed (no red collar/wake).
STORMS_HERO = {
    "hero_tint": -0.9,        # reversed LUT -> dark spot
    "hero_brightness": -0.3,
    "hero_companions": 3,     # bright companion clouds (Voyager GDS hugs its edge w/ bright cirrus)
    "hero_radius": 0.13,
    "hero_strength": 1.7,
    "hero_mottle": 0.35,
    "hero_tint_var": 0.35,
    "hero_rim_tint": 0.0,     # no red collar (Jupiter GRS only)
    "hero_rim_warp": 0.0,
    "hero_wake_detail": 0.0,  # no turbulent trailing wake
    "companion_aspect": 3.5,  # elongate the companion clouds into wispy east-west streaks
                              # (real GDS companions are sheared cirrus, not round dots)
    "hero_emergence": 0.8,  # GRS-realism pack (2026-07-09) — sign-agnostic: the
                            # plateau fill makes the NEGATIVE-brightness GDS a filled
                            # coherent dark oval (Voyager's GDS was exactly that), the
                            # anchor keeps it on its stamp, and the band-flush stops
                            # the neighborhood from winding around it. Companions ride
                            # the flush target (they are stamps) and survive.
    # Inheritance pins (Round B, 2026-07-15): warm's GRS-interaction pass
    # moved ITS hero placement/rim/wake values; the neptune GDS scene was
    # calibrated against the pre-pass inheritance and must not silently move
    # on regen. Any neptune retune is a deliberate Round-D verify+touchup.
    "hero_latitude": -22.5,
    "rim_contrast": 2.0,
    "wake_turbulence": 1.593,
    "companion_brightness": 0.32,  # the pre-lever constant; warm bakes 0.55
    # Aspect-pass pins (2026-07-16): warm bakes hero_aspect 2.9 (the GRS red
    # core is more elongated than its envelope); the Voyager GDS (~2.0) is
    # legitimately rounder, and the signed-off GDS scene was calibrated at
    # 2.2 — it must not silently stretch on regen. hero_taper / hero_flow_
    # aspect are pinned at their inert defaults so a future warm bake of
    # either GRS lever cannot leak into the GDS.
    "hero_aspect": 2.2,
    "hero_taper": 0.0,
    "hero_flow_aspect": 1.0,
}

# Remove the planet-girdling Jupiter storm field -- only the hero + companions remain, plus
# ONE bright blue-white ACCENT cloud (Neptune's discrete bright methane patch, the "Scooter"
# class), ELONGATED east-west into a wispy streak. Single accent, not a pair: two accents
# share one latitude+appearance and read as obvious clone stamps under review. accent_tint
# 1.0 picks the storm_tint bright end (blue-white); accent_brightness kept semi-transparent
# (0.32, not maxed) so it reads as a soft cirrus streak, not an opaque puff.
STORMS_FIELD = {
    "oval_density": 0.0,
    "barge_density": 0.0,
    "pearls_count": 0,
    "small_density": 0.0,
    "outbreak_count": 1,
    "outbreak_strength": 1.0,
    "accent_count": 1,
    "accent_tint": 1.0,
    "accent_brightness": 0.32,
    "accent_radius": 0.06,
    "accent_aspect": 4.0,     # elongate into a wispy east-west cirrus streak
    # Explicit inheritance pins (neptune derives FROM gas_giant_warm, which
    # bakes the Round-B GRS-neighborhood recipe; these must NOT leak):
    # accent_latitude None keeps the Scooter on its seeded-zone placement
    # (warm pins -29 and the hero-relative longitude rule keys off a pinned
    # latitude); oval_solid_core 0 keeps the Scooter's calibrated soft-streak
    # rendering (its 0.06 radius is past the 0.035 solid-core gate, so warm's
    # 1.0 would byte-change the GDS scene).
    "accent_latitude": None,
    "oval_solid_core": 0.0,
}

TURBULENCE = {
    "intensity": 0.8,
    "shear_coupling": 0.4,
    "replenish_rate": 0.45,
}

# No festoons, and no ribbon either: real Neptune's band boundaries are SMOOTH soft gradients
# (Voyager 2 shows no sharp band edge anywhere). The demo's ribbon 2.4 read as a mechanical
# sawtooth scallop -- exactly the "reads mechanical" failure the W14 F14 disposition flagged.
WAVES = {
    "festoon_strength": 0.0,
    "ribbon_strength": 0.0,
    # Inheritance pin: warm bakes a hero-adjacent festoon train (Round B);
    # Neptune's band boundaries are SMOOTH (Voyager 2) — no train.
    "festoon_hero_strength": 0.0,
}

# Deep methane-blue look: bluish haze, no polar tint/canvas (the Juno dark cap is a Jupiter
# lever), muted display (no chroma/hue variance or aging -- Neptune is uniform blue).
APPEARANCE_SCALARS = {
    "haze_amount": 0.25,
    "haze_color": (0.55, 0.70, 0.85),
    "contrast": 0.95,
    "saturation": 1.3,
    "chroma_variance": 0.0,
    "hue_variance": 0.0,
    "chroma_aging": 0.0,
    "polar_tint_color": (0.42, 0.50, 0.58),
    "polar_tint_strength": 0.0,
    "polar_tint_start_lat": 55.0,
    "polar_canvas_value": 0.0,
}

# The detail pass runs ON Neptune now (intensity > 0), but ONLY for the cirrus fibers:
# every other detail term is zeroed. The previous intensity=0.0 bake left the inherited
# gas_giant_warm amounts (belt_texture 0.3, mottle 0.3, polar_stipple 0.8, hero_spiral
# 0.55, ...) latent in the JSON -- flipping intensity on would have woken ALL of them and
# re-textured the deliberately smooth zones/poles/GDS. streak_mute=1.0 kills the one term
# with no zero lever of its own (the base filament streak's speed/shear floor).
DETAIL = {
    "intensity": 0.85,
    "cellular_amount": 0.0,
    "striation_amount": 0.0,
    "hero_calm": 0.0,
    "zone_texture": 0.0,
    "belt_texture": 0.0,
    "belt_texture_fine": 0.0,
    "mottle": 0.0,
    "polar_filaments": 0.0,
    "polar_stipple": 0.0,
    "intermittency": 0.0,
    "hero_spiral": 0.0,
    "hero_collar_wrap": 0.0,
    "spread": 0.0,
    "streak_mute": 1.0,
    "cirrus_fibers": 1.8,
    "cirrus_fiber_freq": 3.0,
}


def build() -> None:
    p = load_factory_preset("gas_giant_warm")
    p.sim = p.sim.model_copy(update=SIM)
    p.solver = p.solver.model_copy(update=SOLVER)
    p.bands = p.bands.model_copy(update=BANDS)
    p.storms = p.storms.model_copy(update={**STORMS_HERO, **STORMS_FIELD})
    p.turbulence = p.turbulence.model_copy(update=TURBULENCE)
    p.waves = p.waves.model_copy(update=WAVES)
    p.detail = p.detail.model_copy(update=DETAIL)
    # Deep methane-blue palette collapsed to a SINGLE uniform row (Neptune is uniform blue --
    # no per-latitude band palette like Jupiter), plus the blue storm-tint ramp.
    deep_stops = [GradientStop(pos=pp, color=list(c)) for pp, c in DEEP_BLUE]
    storm_tints = [GradientStop(pos=pp, color=list(c)) for pp, c in STORM_TINTS_NEPTUNE]
    rows = [PaletteRow(latitude=0.0, stops=deep_stops)]
    p.appearance = p.appearance.model_copy(
        update={**APPEARANCE_SCALARS, "palette_rows": rows, "storm_tints": storm_tints}
    )

    out = PRESETS_DIR / "neptune.json"
    save_preset(p, out, name="neptune")
    # Prove it is in-bounds (save_preset does not re-validate; load_preset does).
    reloaded = load_preset(out)
    assert reloaded.solver.vort_inject == 0.0
    assert reloaded.storms.hero_tint == -0.9
    assert reloaded.storms.hero_companions == 3
    assert reloaded.storms.companion_aspect == 3.5
    # Round-B inheritance pins actually landed (warm bakes the opposite).
    assert reloaded.storms.accent_latitude is None
    assert reloaded.storms.oval_solid_core == 0.0
    assert reloaded.waves.festoon_hero_strength == 0.0
    # Aspect-pass pins (warm bakes hero_aspect 2.9; taper/flow_aspect are the
    # inert defaults today but the pin + assert survive a future warm bake).
    assert reloaded.storms.hero_aspect == 2.2
    assert reloaded.storms.hero_taper == 0.0
    assert reloaded.storms.hero_flow_aspect == 1.0
    assert reloaded.storms.oval_density == 0.0
    assert reloaded.storms.accent_count == 1
    assert reloaded.storms.accent_aspect == 4.0
    assert reloaded.waves.ribbon_strength == 0.0
    assert reloaded.bands.count == 7
    # The detail pass is fiber-only: intensity on, every other term dead.
    assert reloaded.detail.intensity == 0.85
    assert reloaded.detail.cirrus_fibers == 1.8
    assert reloaded.detail.cirrus_fiber_freq == 3.0
    assert reloaded.detail.streak_mute == 1.0
    for dead in ("cellular_amount", "belt_texture", "belt_texture_fine", "mottle",
                 "polar_stipple", "intermittency", "hero_spiral", "hero_collar_wrap",
                 "zone_texture", "striation_amount", "spread"):
        assert getattr(reloaded.detail, dead) == 0.0, dead
    assert len(reloaded.appearance.palette_rows) == 1
    assert list(reloaded.appearance.palette_rows[0].stops[0].color) == [0.04, 0.11, 0.44]
    print(f"wrote + verified {out}", flush=True)


def main() -> None:
    build()


if __name__ == "__main__":
    main()
