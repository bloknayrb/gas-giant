"""Per-storm color levers (review Top-10 #5: A01 / A06 / B5-1 / B5-5 / B5-7).

Registry-level behavior of the W5 lever group:

  - storms.hero_tint / storms.hero_brightness replace the hardwired hero
    constants (0.9 / 0.05); negative brightness = dark storm (Neptune GDS).
  - storms.accent_count/latitude/tint/brightness/radius: explicitly colored
    KIND_OVAL stamps (the Oval BA unlock), seeded on their own "accent-ovals"
    substream AFTER _enforce_cap so the base population is untouched.
  - storms.hero_companions: bright KIND_PEARL-class stamps offset from each
    hero (Neptune GDS companion / Scooter class), own "hero-companions" stream.
  - storms.stamp_tint_contrast: splits tint amplitude from brightness
    amplitude for non-hero storms; None = follow stamp_contrast (byte-identical).

Every lever defaults to current behavior: the default registry must be
IDENTICAL (dataclass equality) to the pre-lever population.
"""

from __future__ import annotations

import numpy as np
import pytest

from gasgiant.params.model import PlanetParams
from gasgiant.sim.bands import generate_bands
from gasgiant.sim.profiles import build_profiles
from gasgiant.sim.vortices import (
    KIND_HERO,
    KIND_OVAL,
    KIND_PEARL,
    KIND_POLAR,
    MAX_VORTICES,
    generate_vortices,
)


def _registry(seed: int = 5, dt: float | None = None, dev_steps: int = 0, **storms):
    p = PlanetParams(seed=seed)
    for key, value in storms.items():
        setattr(p.storms, key, value)
    bands = generate_bands(seed, p.bands)
    profiles = build_profiles(seed, bands, p.bands, p.jets)
    return generate_vortices(seed, bands, profiles, p.storms, p.poles,
                             dt=dt, dev_steps=dev_steps)


# ---------------------------------------------------------------- hero color

def test_hero_defaults_match_v1_constants():
    """Defaults reproduce the previously hardwired tint=0.9 / brightness=0.05."""
    heroes = _registry().heroes()
    assert heroes
    assert all(v.tint == 0.9 and v.brightness == 0.05 for v in heroes)


def test_hero_tint_and_brightness_flow_into_registry():
    heroes = _registry(hero_tint=0.3, hero_brightness=-0.4).heroes()
    assert heroes
    assert all(v.tint == 0.3 and v.brightness == -0.4 for v in heroes)


def test_hero_color_untouched_by_stamp_contrast():
    """stamp_contrast keeps its KIND_HERO exclusion even for custom hero color."""
    heroes = _registry(hero_tint=0.5, hero_brightness=-0.3,
                       stamp_contrast=2.0).heroes()
    assert all(v.tint == 0.5 and v.brightness == -0.3 for v in heroes)


def test_hero_color_forced_default_is_identical():
    """Forced-variant no-op: explicitly setting the defaults changes nothing."""
    assert _registry() == _registry(hero_tint=0.9, hero_brightness=0.05)


# ---------------------------------------------------------------- accent ovals

def test_accent_count_zero_is_identical_even_with_other_accent_params():
    """Forced-variant no-op: accent appearance params are inert at count=0."""
    base = _registry()
    off = _registry(accent_count=0, accent_latitude=-33.0, accent_tint=0.9,
                    accent_brightness=0.3, accent_radius=0.06)
    assert base == off


def test_accent_adds_exactly_count_ovals_with_explicit_color():
    base = _registry()
    reg = _registry(accent_count=2, accent_latitude=-33.0, accent_tint=0.85,
                    accent_brightness=0.3, accent_radius=0.06)
    added = reg.vortices[len(base.vortices):]
    assert len(added) == 2
    for v in added:
        assert v.kind == KIND_OVAL
        assert v.tint == 0.85
        assert v.brightness == 0.3
        assert v.r_core == 0.06
        assert v.lat == pytest.approx(np.deg2rad(-33.0))
    # Pair at offset longitudes.
    dlon = abs((added[0].lon - added[1].lon + np.pi) % (2 * np.pi) - np.pi)
    assert dlon > 0.3


def test_accent_base_population_is_untouched():
    """Accents seed AFTER _enforce_cap on their own stream: every pre-existing
    vortex is bit-identical to the accent-free registry."""
    base = _registry()
    reg = _registry(accent_count=1, accent_latitude=-33.0)
    assert reg.vortices[:len(base.vortices)] == base.vortices


def test_accent_auto_placement_lands_on_a_band():
    """accent_latitude=None auto-places deterministically at a zone latitude."""
    a = _registry(accent_count=1)
    b = _registry(accent_count=1)
    base_n = len(_registry().vortices)
    added_a, added_b = a.vortices[base_n:], b.vortices[base_n:]
    assert len(added_a) == 1
    assert added_a == added_b  # deterministic
    assert abs(added_a[0].lat) < np.deg2rad(68.0)


def test_accent_bypasses_stamp_contrast():
    """Accent color is EXPLICIT (seeded after the contrast pass): stamp_contrast
    must not scale it."""
    reg = _registry(accent_count=1, accent_tint=0.5, accent_brightness=0.2,
                    stamp_contrast=2.0)
    accents = [v for v in reg.vortices if v.tint == 0.5 and v.brightness == 0.2]
    assert len(accents) == 1


def test_accent_latitude_validator_rejects_out_of_range():
    from gasgiant.params.model import StormsParams
    with pytest.raises(ValueError, match="accent_latitude"):
        StormsParams(accent_latitude=60.0, accent_radius=0.05)


def test_accent_latitude_validator_radius_coupled():
    """Cap tightens with accent_radius (same exchange-band rule as the hero)."""
    from gasgiant.params.model import StormsParams
    # 63 - 206.3*0.12 = 38.2: 45 deg must be rejected at the max radius...
    with pytest.raises(ValueError, match="accent_latitude"):
        StormsParams(accent_latitude=45.0, accent_radius=0.12)
    # ...but accepted at a small one.
    StormsParams(accent_latitude=45.0, accent_radius=0.03)


def test_accent_respects_population_cap():
    reg = _registry(small_density=3.0, oval_density=3.0, barge_density=3.0,
                    accent_count=2, accent_latitude=-33.0)
    assert len(reg.vortices) <= MAX_VORTICES


# ---------------------------------------------------------------- companions

def test_hero_companions_zero_is_identical():
    assert _registry() == _registry(hero_companions=0)


def test_hero_companions_adds_bright_pearls_near_each_hero():
    base = _registry()
    reg = _registry(hero_companions=2)
    heroes = reg.heroes()
    assert heroes
    added = reg.vortices[len(base.vortices):]
    assert len(added) == 2 * len(heroes)
    for v in added:
        assert v.kind == KIND_PEARL
        assert v.brightness > 0.2  # bright companion clouds
        # Within a few core radii of some hero (great-circle).
        near = min(
            np.arccos(np.clip(
                np.sin(v.lat) * np.sin(h.lat)
                + np.cos(v.lat) * np.cos(h.lat) * np.cos(v.lon - h.lon), -1, 1))
            / h.r_core
            for h in heroes
        )
        assert near < 5.0


def test_hero_companions_base_population_untouched():
    base = _registry()
    reg = _registry(hero_companions=1)
    assert reg.vortices[:len(base.vortices)] == base.vortices


def test_hero_companions_deterministic():
    assert _registry(hero_companions=2) == _registry(hero_companions=2)


# ---------------------------------------------------------------- B5-7 split

def test_stamp_tint_contrast_none_follows_stamp_contrast():
    """Forced-variant no-op: an explicit value equal to stamp_contrast is
    identical to the default None (follow) behavior."""
    follow = _registry(stamp_contrast=2.0)
    forced = _registry(stamp_contrast=2.0, stamp_tint_contrast=2.0)
    assert follow == forced


def test_stamp_tint_contrast_splits_tint_from_brightness():
    """Polar vortices are excluded like in the existing stamp_contrast test:
    they are seeded after the contrast pass (pre-existing semantics)."""
    base = _registry(stamp_contrast=1.0)
    split = _registry(stamp_contrast=2.0, stamp_tint_contrast=1.0)
    for vb, vs in zip(base.vortices, split.vortices, strict=True):
        if vb.kind == KIND_HERO:
            assert vs.brightness == vb.brightness and vs.tint == vb.tint
        elif vb.kind != KIND_POLAR:
            np.testing.assert_allclose(vs.brightness, vb.brightness * 2.0)
            np.testing.assert_allclose(vs.tint, vb.tint)


def test_stamp_tint_contrast_alone_scales_tint_only():
    base = _registry()
    tinted = _registry(stamp_tint_contrast=1.5)
    for vb, vt in zip(base.vortices, tinted.vortices, strict=True):
        if vb.kind == KIND_HERO:
            assert vt.tint == vb.tint
        elif vb.kind != KIND_POLAR:
            np.testing.assert_allclose(vt.tint, vb.tint * 1.5)
        assert vt.brightness == vb.brightness
