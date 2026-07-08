"""Build the `neptune` factory preset — an ice-giant scenario graduated from the
W14 batch-2 A06 demo (docs/reviews/2026-07-05-w14-calibration-addendum.md).

Neptune is NOT a recolored Jupiter. The batch-2 lesson: the Great Dark Spot levers
(reversed-LUT dark hero + coherent solid-core oval + bright companions) were already
sufficient; expressing a *different planet* is a whole-preset STRUCTURAL retune. So this
starts from gas_giant_warm's proven vorticity engine and:

  * drops the shear-masked eddy injection hard (vort_inject 1.8 -> 0.15) and raises
    hyperviscosity -- the injection was folding the belts into Jupiter churn; without it
    the belts are smooth broad Neptune zones;
  * removes the planet-girdling Jupiter storm field (oval/barge/pearl/small -> 0) and the
    festoons, leaving only the dark GDS hero + 2 bright companion clouds;
  * collapses to a few soft low-contrast bands (count 22 -> 7, value_contrast 1.7 -> 0.55)
    with a Saturn-style ribbon meander (2.4 @ wn 30) giving the band edges Neptune's gentle
    waviness;
  * calms turbulence + strips the heavy Jupiter belt-texture detail;
  * a deep saturated methane-blue palette, bluish haze, no polar tint/canvas.

The dark GDS hero is the reversed-LUT lever from W5 per-storm color: hero_tint -0.9 /
hero_brightness -0.3 over hero_solid_core (a coherent oval, not a whirlpool), with the
Jupiter-only rim/wake levers (rim_tint/rim_warp/wake_detail) zeroed -- the GDS has no red
collar or turbulent trailing wake.

NO kernel changes: every lever here already defaults to a no-op, so this only bakes preset
values. Adding a NEW factory JSON changes no existing preset's output, so the p05 render-
hash baseline is unaffected. The model_copy(update=) calls bypass pydantic bounds, so the
preset is load_preset'd back at the end to PROVE it is in-bounds (save does not re-validate).

Run: uv run python scripts/build_neptune_preset.py
"""
from __future__ import annotations

from pathlib import Path

from gasgiant.params.model import GradientStop, PaletteRow
from gasgiant.params.presets import load_factory_preset, load_preset, save_preset

PRESETS_DIR = Path("src/gasgiant/presets")

# Deep saturated methane blue: deep navy belt gaps -> azure -> light azure bright clouds.
# The top end stays off pure white so the bright companion clouds read as blue-white.
DEEP_BLUE = [
    (0.00, (0.06, 0.14, 0.36)),
    (0.40, (0.12, 0.28, 0.58)),
    (0.75, (0.28, 0.48, 0.74)),
    (1.00, (0.58, 0.74, 0.90)),
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

# Kill the Jupiter belt-churn engine: drop injection, raise hyperviscosity to erase residual
# small-scale folds, and lean on psi-drag for gravest-mode calm. The solid-core hero is a
# separate vorticity patch, so the coherent dark oval survives the calm field.
SOLVER = {
    "vort_inject": 0.15,
    "vort_inject_scale": 1.2,
    "vort_hypervisc": 1.0,
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
    "hero_companions": 2,     # bright companion clouds
    "hero_radius": 0.13,
    "hero_strength": 1.7,
    "hero_mottle": 0.35,
    "hero_tint_var": 0.35,
    "hero_rim_tint": 0.0,     # no red collar (Jupiter GRS only)
    "hero_rim_warp": 0.0,
    "hero_wake_detail": 0.0,  # no turbulent trailing wake
}

# Remove the planet-girdling Jupiter storm field -- only the hero + companions remain.
STORMS_FIELD = {
    "oval_density": 0.0,
    "barge_density": 0.0,
    "pearls_count": 0,
    "small_density": 0.0,
    "outbreak_count": 1,
    "outbreak_strength": 1.0,
}

TURBULENCE = {
    "intensity": 0.8,
    "shear_coupling": 0.4,
    "replenish_rate": 0.45,
}

# No festoons (a Jupiter tell); a Saturn-style ribbon meander gives the band edges Neptune's
# gentle waviness instead.
WAVES = {
    "festoon_strength": 0.0,
    "ribbon_strength": 2.4,
    "ribbon_wavenumber": 30,
}

# Deep methane-blue look: bluish haze, no polar tint/canvas (the Juno dark cap is a Jupiter
# lever), muted display (no chroma/hue variance or aging -- Neptune is uniform blue).
APPEARANCE_SCALARS = {
    "haze_amount": 0.25,
    "haze_color": (0.55, 0.70, 0.85),
    "contrast": 0.95,
    "saturation": 1.05,
    "chroma_variance": 0.0,
    "hue_variance": 0.0,
    "chroma_aging": 0.0,
    "polar_tint_color": (0.42, 0.50, 0.58),
    "polar_tint_strength": 0.0,
    "polar_tint_start_lat": 55.0,
    "polar_canvas_value": 0.0,
}

# Strip the heavy Jupiter belt-texture detail -- Neptune's zones are smooth and low-texture.
DETAIL = {
    "intensity": 0.0,
    "cellular_amount": 0.2,
    "striation_amount": 0.0,
    "hero_calm": 0.0,
    "zone_texture": 0.0,
    "belt_texture": 0.3,
    "belt_texture_fine": 0.3,
    "mottle": 0.3,
    "polar_filaments": 0.0,
    "spread": 0.0,
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
    assert reloaded.solver.vort_inject == 0.15
    assert reloaded.storms.hero_tint == -0.9
    assert reloaded.storms.hero_companions == 2
    assert reloaded.storms.oval_density == 0.0
    assert reloaded.bands.count == 7
    assert len(reloaded.appearance.palette_rows) == 1
    assert list(reloaded.appearance.palette_rows[0].stops[0].color) == [0.06, 0.14, 0.36]
    print(f"wrote + verified {out}", flush=True)


def main() -> None:
    build()


if __name__ == "__main__":
    main()
