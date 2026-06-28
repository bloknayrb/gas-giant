"""Appearance-only polish for the three legacy (kinematic) presets (Stream B of
the preset-modernization milestone): jupiter_like, saturn_pale, ice_giant.

These presets have NO solver section -> they run the KINEMATIC path, not the
vorticity solver. The marquee physics levers (vort_psi_drag, hero_solid_core,
deformation_radius) are no-ops here -- hero_solid_core is consumed only in
vortex_omega.glsl, which never runs in kinematic mode -- so this touches ONLY
render-tier (POST) levers, which DO apply to the kinematic tracer/detail/palette:

  * jupiter_like (the startup default, a busy Jupiter): a real polish. Raise the
    folded belt texture + temperate mottle, apply the zone-detail fix
    (replenish_rate 0.015 -> 0.35, so the fast jets stop smearing the quiescent
    zones to "blurry bands"), and bump the stamp/rim contrast so the hero + ovals
    read boldly. Keeps its blue palette (already value-contrasted, not frosted).
  * saturn_pale / ice_giant: a WHISPER only. These are pale/cold and SMOOTH by
    design (low value_contrast, low detail intensity) -- that IS their identity,
    so warm's heavy texture would betray them. Add just enough subtle belt
    structure to read as cloud rather than flat paint; palette untouched.

NO kernel changes; load_preset round-trips each at the end to prove it is in-bounds.

Run: ./.venv/Scripts/python.exe scripts/build_legacy_presets.py
"""
from __future__ import annotations

from pathlib import Path

# Reuse the vorticity bake's high-contrast Jupiter ramp + applier (the frost fix) so
# jupiter_like gets the SAME de-frosted palette as the live Jupiters. saturn_pale and
# ice_giant are pale/cold BY DESIGN and are intentionally NOT de-frosted.
from build_vorticity_presets import jupiter_palette  # noqa: E402

from gasgiant.params.presets import load_factory_preset, load_preset, save_preset

PRESETS_DIR = Path("src/gasgiant/presets")

# Per-preset render-tier deltas, scaled to each preset's identity.  Each key maps
# to a params sub-model; missing sub-models keep their defaults.
PRESET_DELTA = {
    # The default preset: a genuine richness pass (it is MEANT to be detailed). Render-
    # tier warm lessons -- more folded/cellular/striated cloud texture, the zone-detail
    # replenish fix, a denser Jupiter storm field, bolder stamp. Flow is left alone (it
    # already reads clean -- the visual review liked jupiter_like's coherent bands).
    "jupiter_like": {
        "detail": {"belt_texture": 1.9, "belt_texture_fine": 2.2, "mottle": 1.1,
                   "cellular_amount": 0.9, "striation_amount": 1.0},
        "turbulence": {"replenish_rate": 0.35},          # zone-detail starvation fix
        "storms": {"stamp_contrast": 2.0, "rim_contrast": 2.0, "oval_density": 3.0,
                   "barge_density": 2.989, "pearls_count": 14, "small_density": 3.0},
        "waves": {"festoon_strength": 2.6},
    },
    # Pale, smooth Saturn: a whisper of belt structure; keep it pale.
    "saturn_pale": {
        "detail": {"intensity": 0.3, "belt_texture": 0.5, "mottle": 0.35},
    },
    # Cold, smooth ice giant: a whisper; keep it bland and blue.
    "ice_giant": {
        "detail": {"intensity": 0.5, "belt_texture": 0.4, "mottle": 0.3},
        "turbulence": {"intensity": 0.9},
        "storms": {"stamp_contrast": 1.6},
    },
}

_SUBMODELS = ("detail", "turbulence", "storms", "waves", "appearance")


def polish(name: str) -> None:
    p = load_factory_preset(name)
    for section, update in PRESET_DELTA[name].items():
        assert section in _SUBMODELS, section
        setattr(p, section, getattr(p, section).model_copy(update=update))

    if name == "jupiter_like":
        p.appearance = jupiter_palette(p)  # frost fix (same ramp as the live Jupiters)

    out = PRESETS_DIR / f"{name}.json"
    save_preset(p, out, name=name)
    load_preset(out)  # prove in-bounds (save does not re-validate)
    print(f"wrote + verified {out}", flush=True)


def main() -> None:
    for name in PRESET_DELTA:
        polish(name)


if __name__ == "__main__":
    main()
