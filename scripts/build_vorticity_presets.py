"""Modernize the live-vorticity preset (Stream A of the preset-modernization
milestone): jupiter_vorticity.

It predated the swirl fix + hero realism work: it ran the vorticity solver but
with vort_inject=0 (laminar), a Gaussian whirlpool hero, and no deformation
radius. This bakes gas_giant_warm's PROVEN live-physics block onto it while
PRESERVING its own identity:

  * bands / storm layout are kept; the palette is de-frosted (see jupiter_palette).
  * the solver gains shear-masked injection + scale-selective vort_psi_drag +
    finite deformation_radius (the storm-driven-swirl fix), the hero becomes a
    solid-body coherent oval, and the texture/replenish/jet levers are tuned to
    warm-quality so the bands read detailed and flowing instead of frosted/shredded.

(jupiter_baroclinic was a sibling here but was DROPPED -- a natural-looking
baroclinic preset is just an injection-Jupiter clone; the coupling's intrinsic
regular festoon-comb is the "mechanical" look the user rejected. The baroclinic
feature/engine remains in the code, only the factory preset is gone.)

NO kernel changes; every lever here already defaults to a no-op, so this only
changes the preset's baked values. The model_copy(update=) calls bypass pydantic
bounds, so the preset is load_preset'd back at the end to PROVE it is in-bounds
(the save path does not re-validate).

Run: ./.venv/Scripts/python.exe scripts/build_vorticity_presets.py
"""
from __future__ import annotations

from pathlib import Path

from gasgiant.params.model import GradientStop, InjectMask, PaletteRow
from gasgiant.params.presets import load_factory_preset, load_preset, save_preset

PRESETS_DIR = Path("src/gasgiant/presets")

# The frost fix (same lesson as gas_giant_warm). The frost is NOT just low value range
# -- it is low CHROMA at the BRIGHT end: pale neutral/white zones read as frosted glass
# no matter how dark the belts get (an early ramp that gave the belts deep value but kept
# the zones pale grey-white still read frosted next to warm's chromatic cream zones).
# So the bright zones must carry real WARMTH/chroma (cream, like real Jupiter zones), not
# white. Belts are a browner, less-saturated rust than gas_giant_warm so the presets stay
# distinct. Applied uniformly across rows so band colour follows T0 structure.
CONTRAST_STOPS_JUPITER = [
    (0.00, (0.10, 0.07, 0.05)),  # near-black warm brown (deep belt gaps)
    (0.22, (0.40, 0.26, 0.16)),  # dark brown belt
    (0.45, (0.62, 0.45, 0.29)),  # tan-brown
    (0.66, (0.79, 0.64, 0.44)),  # warm tan
    (0.84, (0.90, 0.80, 0.59)),  # cream (chromatic, NOT white)
    (1.00, (0.97, 0.92, 0.77)),  # warm cream cloud top
]


def jupiter_palette(p):
    """Replace p.appearance.palette_rows with the high-contrast Jupiter ramp (frost
    fix), keeping each row's latitude; chroma_scale 1.0 since the ramp carries the hue."""
    stops = [GradientStop(pos=pp, color=c) for pp, c in CONTRAST_STOPS_JUPITER]
    rows = [PaletteRow(latitude=r.latitude, stops=stops) for r in p.appearance.palette_rows]
    return p.appearance.model_copy(update={"palette_rows": rows, "chroma_scale": 1.0})

# The proven live-physics block from gas_giant_warm (passes swirl_gate): shear-
# masked broadband injection folded by the jets, scale-selective psi-drag bleeding
# the gravest-mode swirl, finite L_d screening the inverse cascade.
#
# vort_psi_drag is tuned PER PRESET (see PRESET_DELTA): warm's hot 0.5 over-flattens
# these at their own dev_steps, where L_d already screens the cascade. The gate
# psi-sweep put the texture-preserving knee (m6 >= 0.45 floor) at ~0.1 for
# jupiter_vorticity; jupiter_baroclinic's source pumps a real swirl (m1 2.57 at
# psi 0) so it carries a touch more drag.
SOLVER_LIVE = {
    "deformation_radius": 0.18,
    "vort_inject": 1.8,
    "vort_inject_scale": 2.5,
    "vort_inject_mask": InjectMask.SHEAR,
}

# Solid-body coherent-oval hero + bold stamp contrast (the whirlpool->oval fix).
# hero_strength bumped to 1.9 and hero_radius set per-preset (below) so the GRS reads
# as a bolder visual anchor (the visual review found it small/muddy at 1.7/0.15).
STORMS_HERO = {
    "hero_solid_core": 1.0,
    "hero_strength": 1.9,
    "rim_contrast": 2.0,
    "stamp_contrast": 2.4,
    "hero_mottle": 0.35,
    "hero_tint_var": 0.35,
    "hero_aspect": 2.2,
}

# Warm's gentler jet profile (vs the stock 1.0/1.6/0.12/0.5): lower strength + WIDER
# equatorial jet = less shear, so the bands stop reading shredded/crumpled (the visual
# review's top structural complaint) and the jet cores stay continuous. Same bands
# template, so band POSITIONS are unchanged -- only the flow is gentler.
JETS_WARM = {
    "strength": 0.733,
    "equatorial_speed": 1.693,
    "equatorial_width": 0.194,
    "polar_decay": 0.648,
}

# Raise fresh-detail replenishment + global turbulence so the zone bands stay textured.
# belt_boost / shear_coupling sit BETWEEN the stock 1.6/1.0 (over-grainy) and warm's
# 1.0/0.7: warm's calm values unmasked the regular prescribed festoon wave (the visual
# review read it as a mechanical sine pattern in vorticity mode), so a touch more belt
# chaos folds it back up while staying below the original grain.
TURBULENCE_RICH = {
    "intensity": 1.6,
    "replenish_rate": 0.45,
    "relax_tau": 2000.0,
    "belt_boost": 1.3,
    "shear_coupling": 0.9,
    "belt_replenish": 0.0,
}

# Warm's denser storm field (vs the stock oval 2.5 / barge 2.2 / pearls 12 / small 1.2):
# many small vortices read more Jupiter-like; warm's low merge_rate + high merge_debris
# keep the storms persistent rather than rapidly coalescing.
STORMS_FIELD = {
    "oval_density": 3.0,
    "barge_density": 2.989,
    "pearls_count": 14,
    "small_density": 3.0,
    "merge_rate": 0.219,
    "merge_debris": 2.0,
    "wake_turbulence": 1.593,
}

# Storm-scale folded belt structure + temperate mottle + the wound-lane hero collar,
# plus warm's higher cellular/striation for more believable fine cloud texture.
DETAIL_RICH = {
    "belt_texture": 1.9,
    "belt_texture_fine": 2.2,
    "mottle": 1.1,
    "hero_collar_wrap": 0.5,
    "cellular_amount": 0.9,
    "striation_amount": 1.0,
}

# Festoons + warm's deeper hotspots. festoon_strength is held DOWN at 1.6 (not warm's
# 2.6): in vorticity mode a strong prescribed festoon reads as a regular embossed sine
# wave (the review's "mechanical" complaint) -- keep it a subtle band-edge accent and
# let the injection-fold turbulence be the primary texture. Ribbon stays 0 (Jupiter).
WAVES_RICH = {
    "festoon_strength": 1.6,
    "festoon_wavenumber": 20,
    "hotspot_depth": 1.0,
}

# Per-preset deltas on top of the shared blocks above.
#   jupiter_vorticity = INJECTION-driven: full shear injection folds the bands into
#     filaments; L_d already screens the cascade at dev 700 so only a gentle
#     gravest-mode insurance psi-drag. Larger 0.18 hero.
# (jupiter_baroclinic was dropped: a natural-looking baroclinic preset is just an
#  injection-Jupiter clone -- the coupling's intrinsic regular festoon-comb is exactly
#  the "mechanical" look the user rejected. The baroclinic FEATURE/engine stays; only
#  the factory preset is gone. See tests/gpu/test_m3_ship.py for engine coverage.)
PRESET_DELTA = {
    "jupiter_vorticity": {
        "solver": {"vort_psi_drag": 0.06},
        "storms": {"hero_radius": 0.18},
    },
}


def modernize(name: str) -> None:
    p = load_factory_preset(name)
    delta = PRESET_DELTA[name]

    p.solver = p.solver.model_copy(update={**SOLVER_LIVE, **delta.get("solver", {})})
    p.jets = p.jets.model_copy(update=JETS_WARM)
    p.storms = p.storms.model_copy(
        update={**STORMS_HERO, **STORMS_FIELD, **delta.get("storms", {})})
    p.turbulence = p.turbulence.model_copy(update=TURBULENCE_RICH)
    p.detail = p.detail.model_copy(update=DETAIL_RICH)
    p.waves = p.waves.model_copy(update=WAVES_RICH)
    p.appearance = jupiter_palette(p)  # frost fix

    out = PRESETS_DIR / f"{name}.json"
    save_preset(p, out, name=name)
    # Prove it is in-bounds (save_preset does not re-validate; load_preset does).
    reloaded = load_preset(out)
    assert reloaded.solver.vort_psi_drag == delta["solver"]["vort_psi_drag"]
    assert reloaded.storms.hero_solid_core == 1.0
    print(f"wrote + verified {out}", flush=True)


def main() -> None:
    modernize("jupiter_vorticity")


if __name__ == "__main__":
    main()
