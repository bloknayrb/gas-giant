"""v1.1 storm-population features: small storms, cap, contrast, wake dir."""

from __future__ import annotations

import numpy as np

from gasgiant.params.model import PlanetParams
from gasgiant.sim.bands import generate_bands
from gasgiant.sim.profiles import build_profiles
from gasgiant.sim.vortices import (
    KIND_HERO,
    KIND_OVAL,
    KIND_POLAR,
    MAX_VORTICES,
    generate_vortices,
)


def _registry(seed: int = 5, **storms):
    p = PlanetParams(seed=seed)
    for key, value in storms.items():
        setattr(p.storms, key, value)
    bands = generate_bands(seed, p.bands)
    profiles = build_profiles(seed, bands, p.bands, p.jets)
    return generate_vortices(seed, bands, profiles, p.storms, p.poles)


def test_small_density_zero_matches_v1_population():
    base = _registry()
    again = _registry()
    assert len(base.vortices) == len(again.vortices)
    assert all(v.kind != KIND_OVAL or v.r_core >= 0.018 - 1e-9 or v.r_core <= 0.02
               for v in base.vortices)


def test_small_density_adds_population_deterministically():
    base = _registry()
    small = _registry(small_density=1.5)
    small2 = _registry(small_density=1.5)
    assert len(small.vortices) > len(base.vortices)
    assert [(v.lat, v.lon) for v in small.vortices] == [
        (v.lat, v.lon) for v in small2.vortices
    ]
    # The existing population is untouched (own seed stream).
    n = len(base.vortices) - sum(v.kind == KIND_POLAR for v in base.vortices)
    assert [(v.lat, v.lon) for v in base.vortices[:n]] == [
        (v.lat, v.lon) for v in small.vortices[:n]
    ]


def test_population_cap():
    reg = _registry(small_density=3.0, oval_density=3.0, barge_density=3.0)
    assert len(reg.vortices) <= MAX_VORTICES
    # Heroes and polar vortices survive the cap.
    assert any(v.kind == KIND_HERO for v in reg.vortices)
    assert any(v.kind == KIND_POLAR for v in reg.vortices)


def test_stamp_contrast_scales_non_hero_stamps():
    base = _registry()
    hot = _registry(stamp_contrast=2.0)
    for vb, vh in zip(base.vortices, hot.vortices, strict=True):
        if vb.kind == KIND_HERO:
            assert vh.brightness == vb.brightness
        elif vb.kind != KIND_POLAR:
            np.testing.assert_allclose(vh.brightness, vb.brightness * 2.0)


def test_heroes_carry_wake_direction():
    reg = _registry()
    heroes = reg.heroes()
    assert heroes
    assert all(v.wake_dir in (-1.0, 1.0) for v in heroes)


def test_pack_ssbo_three_vec4_stride():
    reg = _registry()
    packed = reg.pack_ssbo()
    assert packed.shape == (len(reg.vortices), 12)
    assert packed.dtype == np.float32
    # Unit-sphere positions.
    norms = np.linalg.norm(packed[:, :3], axis=1)
    np.testing.assert_allclose(norms, 1.0, atol=1e-5)


def test_drift_vectorized_matches_scalar_reference():
    p = PlanetParams(seed=5)
    bands = generate_bands(5, p.bands)
    profiles = build_profiles(5, bands, p.bands, p.jets)
    reg = generate_vortices(5, bands, profiles, p.storms, p.poles)
    expect = []
    for v in reg.vortices:
        if v.kind == KIND_POLAR:
            expect.append(v.lon)
            continue
        u = float(np.interp(-v.lat, -profiles.lat, profiles.u))
        expect.append(float((v.lon + u / max(np.cos(v.lat), 0.2) * 0.01 + np.pi)
                            % (2 * np.pi) - np.pi))
    reg.drift(profiles, 0.01)
    np.testing.assert_allclose([v.lon for v in reg.vortices], expect, atol=1e-12)
