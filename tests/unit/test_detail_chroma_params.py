"""appearance.detail_chroma param-level invariants (no GL).

These live in the unit tier deliberately: the GPU behavior tests in
tests/gpu/test_detail_chroma.py are deselected from the always-blocking
no-GPU run by their module gpu mark, and the invariants here — invalidation
tier and the preset bakes — need to gate every PR.
"""
from __future__ import annotations

from gasgiant.engine.invalidation import diff_tiers
from gasgiant.params.model import PlanetParams, Tier
from gasgiant.params.presets import load_factory_preset


def test_detail_chroma_is_post_tier():
    a = PlanetParams(seed=1)
    b = a.model_copy(deep=True)
    b.appearance.detail_chroma = 0.5
    assert diff_tiers(a, b) == {Tier.POST}


def test_warm_bakes_the_calibrated_value():
    """gas_giant_warm ships detail_chroma 0.6 (S2 calibration + user sign-off
    2026-07-16; 0.6 over 1.0 to leave stacking headroom). Only the build
    script guarded this — a regen or hand-edit that dropped it was invisible
    to CI (the replenish_rate pin in test_presets.py exists for the same
    reason)."""
    assert load_factory_preset("gas_giant_warm").appearance.detail_chroma == 0.6


def test_neptune_pins_zero_against_warm_inheritance():
    """Neptune builds FROM warm via model_copy(update=), which preserves
    unlisted fields: without its APPEARANCE_SCALARS pin the warm bake
    silently propagates the warm-side push onto Neptune's cool cirrus. The
    build script asserts this at regen time; this pins the SHIPPED JSON so
    a bad regen can't merge green."""
    assert load_factory_preset("neptune").appearance.detail_chroma == 0.0
