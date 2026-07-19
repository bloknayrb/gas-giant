"""jets.hero_bracket_*: the carve-and-impose hero jet override (build_profiles).
Default-off, structurally guarded no-op; RESTART tier (a VELOCITY rebuild would
flip ambient shear under a stale storm rotation). Machinery only -- no factory
preset bakes these; the warm migration is a deferred visual checkpoint."""
from __future__ import annotations

import dataclasses

import numpy as np

from gasgiant.engine.invalidation import diff_tiers
from gasgiant.params.model import PlanetParams, Tier
from gasgiant.sim.bands import generate_bands
from gasgiant.sim.profiles import build_profiles


def test_hero_bracket_defaults_are_off():
    j = PlanetParams(seed=1).jets
    assert j.hero_bracket_north == 0.0
    assert j.hero_bracket_south == 0.0
    assert j.hero_bracket_north_offset == 3.0
    assert j.hero_bracket_south_offset == -3.0
    assert j.hero_bracket_window == 4.0
    assert j.hero_bracket_feather == 5.0
    assert j.hero_bracket_north_width == 0.05
    assert j.hero_bracket_south_width == 0.05


def test_hero_bracket_fields_are_restart_tier():
    for field, val in (
        ("hero_bracket_north", -1.0), ("hero_bracket_south", 0.6),
        ("hero_bracket_north_offset", 2.0), ("hero_bracket_south_offset", -2.0),
        ("hero_bracket_window", 5.0), ("hero_bracket_feather", 6.0),
        ("hero_bracket_north_width", 0.06), ("hero_bracket_south_width", 0.06),
    ):
        old = PlanetParams(seed=1)
        new = PlanetParams(seed=1)
        setattr(new.jets, field, val)
        assert diff_tiers(old, new) == {Tier.RESTART}, field


def test_hero_bracket_fields_have_no_rand():
    """pfield stores `rand` only when non-None, and pydantic v2 merges
    json_schema_extra into the property top-level (no nested key), so a stray
    rand would appear as a top-level "rand" on the property. Its absence is the
    no-seeded-randomize contract (geometry/offset levers must not be
    randomized)."""
    from gasgiant.params.model import JetsParams
    schema = JetsParams.model_json_schema()["properties"]
    for field in ("hero_bracket_north", "hero_bracket_south",
                  "hero_bracket_north_offset", "hero_bracket_south_offset",
                  "hero_bracket_window", "hero_bracket_feather",
                  "hero_bracket_north_width", "hero_bracket_south_width"):
        assert schema[field].get("rand") is None, field


# ------------------------------------------------------------- override (Task 2)

def _rich(seed, **jet_over):
    p = PlanetParams(seed=seed)
    for k, v in jet_over.items():
        setattr(p.jets, k, v)
    bands = generate_bands(seed, p.bands)
    return p, bands


def test_bracket_off_is_byte_identical_even_with_pinned_hero():
    """north==south==0 skips the whole override, so a pinned hero_lat_deg and
    off-default geometry must not perturb ANY LatProfiles field (structural
    guard, matching the local_jet no-op contract)."""
    seed = 42
    p, bands = _rich(seed)
    base = build_profiles(seed, bands, p.bands, p.jets)  # no hero_lat_deg

    variant = PlanetParams(seed=seed)
    variant.jets.hero_bracket_north = 0.0   # off
    variant.jets.hero_bracket_south = 0.0   # off
    variant.jets.hero_bracket_window = 9.0  # off-default; must not matter
    variant.jets.hero_bracket_feather = 2.0
    off = build_profiles(seed, bands, p.bands, variant.jets, hero_lat_deg=-22.0)

    for field in dataclasses.fields(base):
        a, b = getattr(base, field.name), getattr(off, field.name)
        if isinstance(a, np.ndarray):
            assert np.array_equal(a, b), field.name
        else:
            assert a == b, field.name


def test_bracket_noops_without_pinned_hero():
    """A nonzero bracket with hero_lat_deg=None (no pinned hero) is skipped."""
    seed = 7
    p, bands = _rich(seed)
    base = build_profiles(seed, bands, p.bands, p.jets)
    v = PlanetParams(seed=seed)
    v.jets.hero_bracket_north = -1.0
    v.jets.hero_bracket_south = 0.6
    got = build_profiles(seed, bands, p.bands, v.jets, hero_lat_deg=None)
    assert np.array_equal(base.u, got.u)


def test_bracket_seats_two_sided_shear_deterministically_across_seeds():
    """The bracket erases the seeded jets inside the window and imposes a flat
    pedestal + authored gaussians, so the ON profile's TWO-SIDED shear
    u(equatorward_rim) - u(poleward_rim) == bracket(-19) - bracket(-25): the
    seed-dependent pedestal cancels EXACTLY, leaving a seed-independent shear.
    (NOTE: the per-rim on-minus-off INCREMENT is NOT seed-independent -- it
    carries u_base(hero) - u_base(rim), the natural background shear, swing ~0.4.
    Assert the shear, not the increment; and use the pedestal-independent
    ordering equatorward-more-westward-than-poleward for the sign, since the
    absolute u_eq<0 / u_pol>0 ride on the seed-dependent pedestal.)"""
    def u_at(prof, ld):
        return float(np.interp(np.deg2rad(ld), prof.lat[::-1], prof.u[::-1]))
    shears = []
    for seed in (4201, 1234, 555):
        p, bands = _rich(seed)
        v = PlanetParams(seed=seed)
        v.jets.hero_bracket_north = -1.0
        v.jets.hero_bracket_south = 0.6
        on = build_profiles(seed, bands, p.bands, v.jets, hero_lat_deg=-22.0)
        u_eq = u_at(on, -19.0)   # equatorward rim (role, not compass)
        u_pol = u_at(on, -25.0)  # poleward rim
        assert u_eq < u_pol, (
            f"seed {seed}: bracket did not seat anticyclonic shear "
            f"(equatorward {u_eq} not more westward than poleward {u_pol})")
        shears.append(u_eq - u_pol)
    swing = max(shears) - min(shears)
    assert swing < 1e-6, f"bracket two-sided shear not seed-independent: swing {swing}"


def test_bracket_window_has_no_vorticity_spike():
    """The C1 smoothstep window keeps du/dphi continuous, so omega_jet has no
    isolated spike at the feather edge. Assert the largest single-sample jump in
    du/dphi anywhere in the hero neighborhood is within a small multiple of the
    median jump there (a linear feather would produce a delta-function spike)."""
    seed = 99
    p, bands = _rich(seed)
    v = PlanetParams(seed=seed)
    v.jets.hero_bracket_north = -1.0
    v.jets.hero_bracket_south = 0.6
    on = build_profiles(seed, bands, p.bands, v.jets, hero_lat_deg=-22.0)
    lat_deg = np.rad2deg(on.lat)
    near = np.abs(lat_deg - (-22.0)) < 12.0  # window+feather+margin
    du = np.gradient(on.u, on.lat)
    d2 = np.abs(np.diff(du[near]))
    # A C1 smoothstep gives max/median curvature-jump ~4x; a broken C0 LINEAR
    # feather (w' jumps to 0 at the edges -> a du step) gives ~13x. 8x sits
    # between, so this PASSES the smoothstep and FAILS a linear regression.
    # (Reviewer-measured on the 2048-pt grid; re-confirm if the grid changes.)
    ratio = d2.max() / np.median(d2[d2 > 0])
    assert ratio < 8.0, f"vorticity spike at window edge: curvature ratio {ratio}"
