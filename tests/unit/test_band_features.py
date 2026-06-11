"""v1.1 band features: determinism, hue jitter, faded sector, tiers."""

from __future__ import annotations

import hashlib

import numpy as np

from gasgiant.engine.invalidation import diff_tiers
from gasgiant.params.model import PlanetParams, Tier
from gasgiant.sim.bands import generate_bands
from gasgiant.sim.profiles import build_profiles

# Band/profile fingerprints from v1 (pre-A3). New seeded features draw from
# their own SeedSequence streams, so neutral defaults must reproduce v1
# layouts bit-for-bit. If this fails, a new feature leaked draws into an
# existing stream.
_V1_FINGERPRINTS = {
    1: "530f1a44b2b2dd81",
    42: "7899d7c3c8fb4b81",
    9120: "ffba1c3ec3127c0d",
}


def _fingerprint(seed: int) -> str:
    p = PlanetParams(seed=seed)
    b = generate_bands(seed, p.bands)
    prof = build_profiles(seed, b, p.bands, p.jets)
    blob = np.concatenate(
        [b.edges, b.values, b.heights, prof.u.astype(np.float32), prof.t0_stamp.astype(np.float32)]
    ).tobytes()
    return hashlib.sha1(blob).hexdigest()[:16]


def test_neutral_defaults_reproduce_v1_layouts():
    for seed, expected in _V1_FINGERPRINTS.items():
        assert _fingerprint(seed) == expected


def test_hue_jitter_changes_values_not_edges():
    p = PlanetParams(seed=7)
    base = generate_bands(7, p.bands)
    p.bands.hue_jitter = 0.1
    jittered = generate_bands(7, p.bands)
    np.testing.assert_array_equal(base.edges, jittered.edges)
    assert not np.array_equal(base.values, jittered.values)
    assert np.abs(base.values - jittered.values).max() <= 0.1 + 1e-6


def test_hue_jitter_is_seeded():
    p = PlanetParams(seed=7)
    p.bands.hue_jitter = 0.1
    a = generate_bands(7, p.bands)
    b = generate_bands(7, p.bands)
    np.testing.assert_array_equal(a.values, b.values)


def test_fade_sector_targets_a_belt():
    p = PlanetParams(seed=11)
    bands = generate_bands(11, p.bands)
    lat_lo, lat_hi, lon, halfwidth = bands.fade_sector
    assert lat_lo < lat_hi
    assert -np.pi <= lon <= np.pi
    assert np.deg2rad(35.0) < halfwidth < np.deg2rad(60.0)
    # The sector's band is a belt (below-median color index).
    center = 0.5 * (lat_lo + lat_hi)
    j = int(np.searchsorted(-bands.edges, -center)) - 1
    assert bands.values[j] < np.median(bands.values)


def test_fade_sector_flows_into_profiles():
    p = PlanetParams(seed=11)
    bands = generate_bands(11, p.bands)
    prof = build_profiles(11, bands, p.bands, p.jets)
    assert prof.fade_sector == bands.fade_sector


def test_new_band_fields_are_restart_tier():
    for field in ("hue_jitter", "variance_amount", "faded_sector", "contrast_envelope",
                  "edge_diversity", "width_tail"):
        old = PlanetParams(seed=1)
        new = PlanetParams(seed=1)
        setattr(new.bands, field, 0.1)
        assert diff_tiers(old, new) == {Tier.RESTART}, field


def test_lane_density_is_velocity_tier_and_striation_is_post():
    old = PlanetParams(seed=1)
    new = PlanetParams(seed=1)
    new.bands.lane_density = 0.5
    assert diff_tiers(old, new) == {Tier.VELOCITY}
    new2 = PlanetParams(seed=1)
    new2.detail.striation_amount = 0.5
    assert diff_tiers(old, new2) == {Tier.POST}


def test_select_lanes_density_and_determinism():
    from gasgiant.sim.profiles import select_lanes

    p = PlanetParams(seed=3)
    bands = generate_bands(3, p.bands)
    assert select_lanes(3, bands, 0.0) == []
    full = select_lanes(3, bands, 1.0)
    some = select_lanes(3, bands, 0.5)
    assert 0 < len(some) <= len(full)
    # Raising density only adds lanes; it never reshuffles existing ones.
    assert set(some).issubset(set(full))
    assert select_lanes(3, bands, 0.5) == some


def test_width_tail_changes_widths_not_band_count():
    p = PlanetParams(seed=9)
    base = generate_bands(9, p.bands)
    p.bands.width_tail = 0.8
    tailed = generate_bands(9, p.bands)
    assert len(base.edges) == len(tailed.edges)
    assert not np.array_equal(base.edges[1:-1], tailed.edges[1:-1])
    # Heavier tail: the width spread grows.
    bw = -np.diff(base.edges)
    tw = -np.diff(tailed.edges)
    assert (tw.max() / tw.min()) > (bw.max() / bw.min())


def test_edge_diversity_changes_stamp_transitions_only():
    p = PlanetParams(seed=9)
    bands = generate_bands(9, p.bands)
    base = build_profiles(9, bands, p.bands, p.jets)
    p.bands.edge_diversity = 1.0
    varied = build_profiles(9, bands, p.bands, p.jets)
    np.testing.assert_array_equal(base.u, varied.u)  # velocity untouched
    assert not np.array_equal(base.t0_stamp, varied.t0_stamp)
