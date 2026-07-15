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
    # GRS interaction pass (2026-07-15, plan ancient-snuggling-meadow): the
    # hero must READ as embedded in the flow, not sitting on it.
    # - hero_latitude -21.0 FORKS from jupiter_vorticity's -22.5 (deliberate):
    #   warm's belt spans -7.31..-19.41 deg, so at -21.0 the oval's north rim
    #   PROTRUDES into the belt (the Red Spot Hollow straddle) and the wake
    #   lane (equatorward-biased) folds dark belt material instead of
    #   invisible zone-on-zone cream.
    # - hero_radius 0.062 (~3.6 deg lat semi-axis; the real GRS is ~4-4.5):
    #   absolute presence; the fill RATIO is fixed kernel-side (q-normalized).
    # - rim_contrast 2.0 -> 1.3: the doubled ring/collar amplitude was a main
    #   driver of the etched-rings look; the real Hollow is only slightly
    #   brighter than the bands.
    # - wake_turbulence 3.2: now LIVE in vorticity mode (omega_force wake-wedge
    #   injection); 1.593 was calibrated when the lever was feather-only inert.
    # - hero_wake_detail 1.0, hero_mottle 0.9, hero_tint_var 0.55: full fray on
    #   the dimmed wake stamp; granular (still muted) interior.
    # - hero_emergence 0.8 -> 0.9: the retuned pack endpoints land at 90%.
    p.storms = p.storms.model_copy(update={
        "hero_radius": 0.062,
        "hero_latitude": -21.0,
        "hero_emergence": 0.9,
        "rim_contrast": 1.3,
        "wake_turbulence": 3.2,
        "hero_wake_detail": 1.0,
        "hero_mottle": 0.9,
        "hero_tint_var": 0.55,
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
    })

    p.name = "gas_giant_warm"
    save_preset(p, OUT, name="gas_giant_warm")
    print("wrote", OUT, flush=True)


if __name__ == "__main__":
    main()
