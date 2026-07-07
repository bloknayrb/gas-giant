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
    p.storms = p.storms.model_copy(update={"hero_radius": 0.05})

    # Uniform detail coverage: apply the flow-folded detail-FX texture at EVEN
    # density across latitude instead of leaving zones detail-starved. 0.36 was
    # the calibrated value (user sign-off 2026-07-07) — even, fluid, not patchy.
    p.detail = p.detail.model_copy(update={"spread": 0.36})

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
