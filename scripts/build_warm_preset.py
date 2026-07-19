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
    # Vortex chirality fix (2026-07-17, plan chirality_plan.md): storms now
    # co-rotate with the local ambient shear instead of counter-rotating, so
    # the hero at a cyclonic latitude would develop as a CYCLONE (wrong
    # class). No latitude on warm's authored ambient field gives a wide
    # anticyclonic window on its own, so a local zonal jet
    # (jets.local_jet_speed/-latitude/-width) is baked in alongside a
    # hero_latitude retune to CARVE one. C0 grid search (scratch,
    # (triplet x latitude) x seeded-registry measurement, seed 4201) tested
    # {primary -0.9/-20.0/0.05, fallback -0.7/-19.5/0.045} x
    # {-21.0, -21.5, -22.0}: primary @ -22.0 wins every axis simultaneously
    # -- strongest co-rotating ambient shear of the grid (zeta +7.03, vs.
    # +3.8..+5.2 elsewhere), a registry-measured WEST wake that is robust
    # across the full +/-20% r-core jitter band (flips only at the
    # mathematical +20% extreme), and the slowest hero drift of the primary
    # family (-10.89 deg over the dev-700 run, vs -14.95 deg @ -21.0).
    # hero_latitude -21.0 -> -22.0 for this reason (still inside the belt
    # window, now anticyclonic post-flip). The local jet's amplitude here is
    # PRE-strength/pre-polar_fade (equatorial_speed precedent): effective
    # peak = local_jet_speed * jets.strength.
    # - hero_radius 0.062 (~3.6 deg lat semi-axis; the real GRS is ~4-4.5):
    #   absolute presence; the fill RATIO is fixed kernel-side (q-normalized).
    # - rim_contrast 2.0 -> 1.3: the doubled ring/collar amplitude was a main
    #   driver of the etched-rings look; the real Hollow is only slightly
    #   brighter than the bands.
    # - wake_turbulence 3.2: now LIVE in vorticity mode (omega_force wake-wedge
    #   injection); 1.593 was calibrated when the lever was feather-only inert.
    # - hero_wake_detail 1.0, hero_mottle 0.9: full fray on the dimmed wake
    #   stamp; granular (still muted) interior.
    # - hero_emergence 0.8 -> 0.9: the retuned pack endpoints land at 90%.
    # Round B (de-bullseye + populate the neighborhood, same plan):
    # Aspect pass (2026-07-16, same plan): the reference GRS's RED CORE is
    # MORE elongated (~2.9) than its pale envelope (~1.8); ours rendered the
    # core at ~2.0. The deficit is anatomy-side (psi rounds any authored
    # geometry, but every erasure window is aspect-normalized, so raising the
    # AUTHORED aspect moves the flow and the anatomy windows together — no
    # metric mismatch, no dye dilution). storms.hero_flow_aspect stays at its
    # 1.0 default: K>1 widens the ENVELOPE (the wrong component — it inverts
    # the reference's core>envelope hierarchy) and pays the dilution.
    # - hero_aspect 2.2 -> 2.9: measured dye-core aspect 3.04 vs reference
    #   2.9, core saturation unchanged from the 2.2 bake; 2.6 was a dead zone
    #   (no core gain — the response is nonlinear, 2.9 punches through).
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
    p.jets = p.jets.model_copy(update={
        "local_jet_speed": -0.9,
        "local_jet_latitude": -20.0,
        "local_jet_width": 0.05,
    })
    p.storms = p.storms.model_copy(update={
        "hero_radius": 0.062,
        "hero_latitude": -22.0,
        "hero_aspect": 2.9,
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
