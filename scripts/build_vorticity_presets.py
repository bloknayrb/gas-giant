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

import json
from pathlib import Path

from gasgiant.params.model import GradientStop, InjectMask, PaletteRow
from gasgiant.params.presets import load_factory_preset, load_preset, save_preset

PRESETS_DIR = Path("src/gasgiant/presets")

# -- SOURCE-FIDELITY PASS (2026-06-29, branch feat/jupiter-source-fidelity) -------------
# User target = FULL Cassini match: the prior warm "frost-fix" ramp (kept below as the
# rollback fallback) is deliberately far warmer / more saturated / higher-contrast than
# the Cassini true-color reference (refs/PIA07782.jpg). compare_reference.py proved our
# chroma was a SURPLUS, not a deficit; the real gaps were RGB level, contrast, and
# within-band hue spread. So the band palette is now fitted PER-LATITUDE directly from the
# reference (muted, cooler poles, pale zones, reddish-brown tropical belts), and the
# display contrast is pulled down to the reference's gentle envelope.
#
# Flip this to False to restore the warm CONTRAST_STOPS_JUPITER ramp (one-line rollback).
USE_PER_LATITUDE_PALETTE = True

# Reproducible build input: per-latitude rows fit from the Cassini map. Regenerate with
#   uv run python scripts/calibrate_palette.py --reference refs/PIA07782.jpg \
#     --fit-mode median --stops 5 --min-l-span 0.58 \
#     --anchors -78.5 -57.5 -45.7 -40.1 -28.1 -13.4 -0.7 12 20.1 23.5 33.3 66 \
#     --out scripts/jupiter_calib_rows.json   (then keep only the palette_rows key)
# median (not chroma-restore) faithfully reproduces the reference's muted chroma; the
# --min-l-span floor only lifts each row's value spread clear of the >0.5 anti-frost guard
# (every fitted row lands at >=0.647 Rec.709 luma spread).
_CALIB_ROWS = Path("scripts/jupiter_calib_rows.json")

# Display contrast pulled toward the reference's gentle envelope (was 1.05). 0.8 cut the
# compare_reference `contrast` distance 0.14 -> 0.08 while keeping band definition; 0.7
# washed the belts into the zones (visual reject).
SOURCE_CONTRAST = 0.8

# Phase C: a touch more longitudinal hue drift toward the reference's within-band hue
# variety (was 0.18; the hue_spread metric is only mildly responsive so this stays modest).
SOURCE_HUE_VARIANCE = 0.30

# Chromophore aging (dynamics-driven color): reddish-brown chromophore concentrates in
# the dark downwelling belts (and the GRS) and tracks the freshness tracer T2 within them,
# while bright zones stay pale and the poles keep their blue-gray. VIVID amplitude chosen
# by the user (rust belts, ~+33% tropical saturation) over the muted-only calibration.
SOURCE_CHROMA_AGING = 0.35

# Rollback fallback (the prior warm "frost-fix" ramp). The frost is NOT just low value
# range -- it is low CHROMA at the BRIGHT end: pale neutral/white zones read as frosted
# glass no matter how dark the belts get. Bright zones carry real WARMTH; belts a browner
# rust than gas_giant_warm. Broadcast uniformly across rows so band colour follows T0.
CONTRAST_STOPS_JUPITER = [
    (0.00, (0.10, 0.07, 0.05)),  # near-black warm brown (deep belt gaps)
    (0.22, (0.40, 0.26, 0.16)),  # dark brown belt
    (0.45, (0.62, 0.45, 0.29)),  # tan-brown
    (0.66, (0.79, 0.64, 0.44)),  # warm tan
    (0.84, (0.90, 0.80, 0.59)),  # cream (chromatic, NOT white)
    (1.00, (0.97, 0.92, 0.77)),  # warm cream cloud top
]

# The storm-tint LUT (indexed by the storm's T3 tracer): the hero's warm salmon core
# sits near the warm end (~0.95). Set EXPLICITLY here so the preset is reproducible from
# the override dicts -- jupiter_palette() previously left storm_tints to implicit carry
# from the prior JSON, which a regen from a pristine base would silently drop.
STORM_TINTS_JUPITER = [
    (0.00, (0.42, 0.50, 0.62)),  # cool blue-grey (deep cyclone / barge core)
    (0.50, (0.70, 0.58, 0.44)),  # neutral tan
    (0.72, (0.46, 0.30, 0.20)),  # brown notch, lifted so the hero anchor reads fuller
    (1.00, (0.79, 0.45, 0.30)),  # Phase C: muted brick-orange (was salmon-red .86/.42/.24)
                                 # to match the 2000-epoch reference GRS -- a defined oval,
                                 # not a vivid red spot, but still distinct from its surrounds
]


def _per_latitude_rows() -> list[PaletteRow]:
    """The Cassini-fitted per-latitude band rows (source-fidelity pass)."""
    doc = json.loads(_CALIB_ROWS.read_text())
    return [
        PaletteRow(
            latitude=r["latitude"],
            stops=[GradientStop(pos=s["pos"], color=s["color"]) for s in r["stops"]],
        )
        for r in doc["palette_rows"]
    ]


def jupiter_palette(p):
    """Set the Jupiter band palette + storm-tint LUT and the source-match display contrast.

    With USE_PER_LATITUDE_PALETTE (the source-fidelity default) the rows come straight from
    the Cassini fit; otherwise the warm CONTRAST_STOPS_JUPITER ramp is broadcast to every
    row (rollback). storm_tints + contrast are set explicitly so the preset is fully
    reproducible from these inputs (no implicit carry). Shared by jupiter_vorticity and
    jupiter_like (build_legacy_presets imports this), so both Jupiters track the reference."""
    if USE_PER_LATITUDE_PALETTE:
        rows = _per_latitude_rows()
    else:
        stops = [GradientStop(pos=pp, color=c) for pp, c in CONTRAST_STOPS_JUPITER]
        rows = [PaletteRow(latitude=r.latitude, stops=stops) for r in p.appearance.palette_rows]
    storm_tints = [GradientStop(pos=pp, color=c) for pp, c in STORM_TINTS_JUPITER]
    return p.appearance.model_copy(
        update={
            "palette_rows": rows,
            "chroma_scale": 1.0,
            "storm_tints": storm_tints,
            "contrast": SOURCE_CONTRAST,
            "hue_variance": SOURCE_HUE_VARIANCE,
            "chroma_aging": SOURCE_CHROMA_AGING,
            # Deepen the polar cap so the FFR lace + cyclones pop (gap #1, reviewed
            # PASS at 0.85). Shared by both Jupiters via this palette.
            "polar_canvas_value": 0.85,
        }
    )

# The live-physics block (gas_giant_warm's proven recipe, passes swirl_gate): shear-
# masked broadband injection folded by the jets, finite L_d screening the inverse
# cascade, and scale-selective psi-drag bleeding the gravest-mode swirl. vort_psi_drag
# is a GENTLE 0.06 here -- warm's hot 0.5 over-flattens at this preset's dev_steps,
# where L_d already screens the cascade, so the gate psi-sweep knee (m6 >= 0.45 floor)
# sits low; it is just gravest-mode insurance.
SOLVER_LIVE = {
    "deformation_radius": 0.18,
    "vort_inject": 1.8,
    "vort_inject_scale": 2.5,
    "vort_inject_mask": InjectMask.SHEAR,
    "vort_psi_drag": 0.06,
}

# Solid-body coherent-oval hero. Phase C (source-fidelity) shrinks + mutes it toward the
# faded 2000-epoch reference GRS: radius 0.18 -> 0.14 (the reference spot is small) and
# less interior salmon. hero_solid_core stays 1.0 (coherent oval, not a whirlpool).
STORMS_HERO = {
    "hero_solid_core": 1.0,
    "hero_strength": 1.9,
    "hero_radius": 0.16,    # Phase C: slightly smaller (was 0.18) but the reference GRS
                            # is a LARGE defined oval -- 0.14 over-shrank it to a smudge
    "hero_latitude": -22.5,  # pinned to the real GRS latitude (~22 S)
    "rim_contrast": 2.0,
    "stamp_contrast": 2.4,
    "hero_mottle": 0.70,    # strong interior churn: kill the airbrushed-blob core
                            # (flow-folded fbm => follows the vortex, not band-grain)
    "hero_tint_var": 0.40,  # Phase C: a touch less salmon (was 0.45) but keep interior
                            # definition -- 0.30 washed the oval out (m5 hero dropped)
    "hero_aspect": 2.2,
    "hero_rim_warp": 0.65,  # lumpy-oval boundary (break the perfect-ring look)
    "hero_rim_tint": 0.85,  # deeper, azimuthally-broken dark-red collar (Red Spot Hollow
                            # moat, not a uniform ring) => discrete vortex
    "hero_wake_detail": 0.80,  # fray the downstream wake into folded filaments instead
                               # of a smooth wedge (reads as turbulence, not a blob)
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
    "intensity": 1.2,  # Phase C: softer than 1.6 toward the reference's smoother belts
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
    "wake_turbulence": 1.593,  # left at the modernized value: bumping it read as a
                               # brighter smooth blob, not more discrete filaments (visual review)
    # Convective white-plume outbreaks (Great-White-Spot / SEB-revival): RE-ENABLED
    # after the rebuild (events.py belt placement + vortex_stamp.glsl ring-no-dome
    # stamp). The old version was disabled because it emerged as a SECOND hero-sized
    # storm; the reshaped plume train reads as belt convective churn, not a rival GRS
    # (adversarial review confirmed). 2 eruption sites, gentle strength.
    "outbreak_count": 2,
    "outbreak_strength": 1.1,
}

# Storm-scale folded belt structure + temperate mottle + the wound-lane hero collar,
# plus warm's higher cellular/striation for more believable fine cloud texture.
DETAIL_RICH = {
    "belt_texture": 1.2,        # Phase C: softer folded belt structure (was 1.9) -- the
    "belt_texture_fine": 1.4,   # reference belts are smoother; halved texture_energy gap
    "zone_texture": 1.0,        # fill the detail-starved zones (the smooth lanes
                               # between belts read as reduced-detail bands)
    "mottle": 1.1,
    "hero_calm": 0.75,          # calm the straight band-grain crossing the GRS so the
                                # vortex-aligned spiral lanes + mottle carry the interior
    "hero_spiral": 0.55,        # interior wound lanes (Juno close-up)
    "hero_collar_wrap": 0.5,    # wound-lane filaments wrapping the collar (GRS hollow)
    "intermittency": 0.65,      # longitudinal patchiness: violent folds abutting calm runs
    "cellular_amount": 0.9,
    "striation_amount": 1.0,
    "polar_filaments": 1.3,     # Juno folded-filamentary cap lace (sparse flow-folded
                                # ridge wisps poleward of 66 deg; DC-neutral by a clamped
                                # bright side so it is drift-stable across dev_steps)
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

# jupiter_vorticity is INJECTION-driven: full shear injection folds the bands into
# filaments; L_d screens the cascade so only a gentle gravest-mode insurance psi-drag.
# (jupiter_baroclinic was dropped: a natural-looking baroclinic preset is just an
#  injection-Jupiter clone -- the coupling's intrinsic regular festoon-comb is exactly
#  the "mechanical" look the user rejected. The baroclinic FEATURE/engine stays; only
#  the factory preset is gone. See tests/gpu/test_m3_ship.py for engine coverage.)
def modernize() -> None:
    p = load_factory_preset("jupiter_vorticity")
    p.solver = p.solver.model_copy(update=SOLVER_LIVE)
    p.jets = p.jets.model_copy(update=JETS_WARM)
    p.storms = p.storms.model_copy(update={**STORMS_HERO, **STORMS_FIELD})
    p.turbulence = p.turbulence.model_copy(update=TURBULENCE_RICH)
    p.detail = p.detail.model_copy(update=DETAIL_RICH)
    p.waves = p.waves.model_copy(update=WAVES_RICH)
    p.appearance = jupiter_palette(p)  # frost fix

    out = PRESETS_DIR / "jupiter_vorticity.json"
    save_preset(p, out, name="jupiter_vorticity")
    # Prove it is in-bounds (save_preset does not re-validate; load_preset does).
    reloaded = load_preset(out)
    assert reloaded.solver.vort_psi_drag == 0.06
    assert reloaded.storms.hero_solid_core == 1.0
    assert reloaded.storms.hero_rim_tint == 0.85
    assert reloaded.storms.hero_wake_detail == 0.80
    print(f"wrote + verified {out}", flush=True)


def main() -> None:
    modernize()


if __name__ == "__main__":
    main()
