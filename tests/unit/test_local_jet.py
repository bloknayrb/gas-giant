"""jets.local_jet_speed / local_jet_latitude / local_jet_width: the additive
local zonal jet lever (build_profiles, profiles.py). Default-off, structurally
guarded no-op; RESTART tier (the VELOCITY live-edit path rebuilds profiles
without regenerating storms, which would flip the ambient shear sign under
stale storm rotations)."""

from __future__ import annotations

import dataclasses

import numpy as np

from gasgiant.engine.invalidation import diff_tiers
from gasgiant.params.model import PlanetParams, Tier
from gasgiant.sim.bands import generate_bands
from gasgiant.sim.profiles import build_profiles


def test_local_jet_speed_zero_is_a_true_noop():
    """speed=0.0 must be byte-identical to the term being structurally absent,
    for EVERY LatProfiles field -- including when latitude/width are moved
    away from their defaults (only speed gates the term)."""
    seed = 42
    p = PlanetParams(seed=seed)
    bands = generate_bands(seed, p.bands)
    baseline = build_profiles(seed, bands, p.bands, p.jets)

    variant = PlanetParams(seed=seed)
    variant.jets.local_jet_speed = 0.0
    variant.jets.local_jet_latitude = 33.0  # off-default; must not matter at speed=0
    variant.jets.local_jet_width = 0.2      # off-default; must not matter at speed=0
    off = build_profiles(seed, bands, p.bands, variant.jets)

    for field in dataclasses.fields(baseline):
        a = getattr(baseline, field.name)
        b = getattr(off, field.name)
        if isinstance(a, np.ndarray):
            assert np.array_equal(a, b), field.name
        else:
            assert a == b, field.name


def test_local_jet_produces_a_localized_westward_extremum():
    seed = 42
    p = PlanetParams(seed=seed)
    bands = generate_bands(seed, p.bands)
    baseline = build_profiles(seed, bands, p.bands, p.jets)

    variant = PlanetParams(seed=seed)
    variant.jets.local_jet_speed = -0.9
    variant.jets.local_jet_latitude = -20.0
    variant.jets.local_jet_width = 0.05
    jetted = build_profiles(seed, bands, p.bands, variant.jets)

    delta = jetted.u - baseline.u
    lat_deg = np.rad2deg(baseline.lat)

    # Extremum sits right at the authored latitude and is westward (negative)
    # of the right pre-strength depth (default jets.strength == 1.0, and the
    # authored latitude is well outside the polar_fade ramp, so the effective
    # peak equals local_jet_speed almost exactly).
    imin = int(np.argmin(delta))
    assert abs(lat_deg[imin] - (-20.0)) < 0.5
    assert delta[imin] < 0.0
    assert abs(delta[imin] - (-0.9)) < 0.02

    # Localized: negligible outside +/- 4 widths (0.05 rad ~= 2.86 deg).
    far = np.abs(lat_deg - (-20.0)) > 4.0 * np.rad2deg(0.05)
    assert np.abs(delta[far]).max() < 1e-4


def test_local_jet_fields_are_restart_tier():
    for field in ("local_jet_speed", "local_jet_latitude", "local_jet_width"):
        old = PlanetParams(seed=1)
        new = PlanetParams(seed=1)
        setattr(new.jets, field, 0.1 if field != "local_jet_speed" else -0.5)
        assert diff_tiers(old, new) == {Tier.RESTART}, field
