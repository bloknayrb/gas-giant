"""jets.hero_bracket_*: the carve-and-impose hero jet override (build_profiles).
Default-off, structurally guarded no-op; RESTART tier (a VELOCITY rebuild would
flip ambient shear under a stale storm rotation). Machinery only -- no factory
preset bakes these; the warm migration is a deferred visual checkpoint."""
from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from gasgiant.engine.invalidation import diff_tiers
from gasgiant.params.model import PlanetParams, Tier
from gasgiant.sim.bands import generate_bands
from gasgiant.sim.profiles import build_profiles


def test_hero_bracket_defaults_are_off():
    j = PlanetParams(seed=1).jets
    assert j.hero_bracket_north == 0.0
    assert j.hero_bracket_south == 0.0
    assert j.hero_bracket_north_offset == 1.0     # x hero core radius (jet at storm edge)
    assert j.hero_bracket_south_offset == -1.0
    assert j.hero_bracket_window == 1.0
    assert j.hero_bracket_feather == 1.4
    assert j.hero_bracket_north_width == 0.8
    assert j.hero_bracket_south_width == 0.8


def test_hero_bracket_fields_are_restart_tier():
    for field, val in (
        ("hero_bracket_north", -1.0), ("hero_bracket_south", 0.6),
        ("hero_bracket_north_offset", 2.0), ("hero_bracket_south_offset", -2.0),
        ("hero_bracket_window", 2.0), ("hero_bracket_feather", 2.0),
        ("hero_bracket_north_width", 0.15), ("hero_bracket_south_width", 0.15),
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
    variant.jets.hero_bracket_window = 3.5  # off-default, in-range; must not matter
    variant.jets.hero_bracket_feather = 2.0
    off = build_profiles(seed, bands, p.bands, variant.jets,
                         hero_lat_deg=-22.0, hero_r_core=0.11)  # off-default radius, unread

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
    HERO_R = 0.062  # warm hero_radius (radians); the bracket geometry scales by this

    def u_at(prof, ld):
        return float(np.interp(np.deg2rad(ld), prof.lat[::-1], prof.u[::-1]))
    shears = []
    for seed in (4201, 1234, 555):
        p, bands = _rich(seed)
        v = PlanetParams(seed=seed)
        v.jets.hero_bracket_north = -1.0
        v.jets.hero_bracket_south = 0.6
        on = build_profiles(seed, bands, p.bands, v.jets,
                            hero_lat_deg=-22.0, hero_r_core=HERO_R)
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
    on = build_profiles(seed, bands, p.bands, v.jets,
                        hero_lat_deg=-22.0, hero_r_core=0.062)
    lat_deg = np.rad2deg(on.lat)
    near = np.abs(lat_deg - (-22.0)) < 12.0  # window+feather+margin (core radii * deg/rc)
    du = np.gradient(on.u, on.lat)
    d2 = np.abs(np.diff(du[near]))
    # A C1 smoothstep gives max/median curvature-jump ~4x; a broken C0 LINEAR
    # feather (w' jumps to 0 at the edges -> a du step) gives ~13x. 8x sits
    # between, so this PASSES the smoothstep and FAILS a linear regression.
    # (Reviewer-measured on the 2048-pt grid; re-confirm if the grid changes.)
    ratio = d2.max() / np.median(d2[d2 > 0])
    assert ratio < 8.0, f"vorticity spike at window edge: curvature ratio {ratio}"


# --- size-relative geometry: every one of the six geometry quantities (both
# offsets, window, feather, both widths) must scale with hero_r_core. The tests
# below isolate each so a regression leaving ONE of them in absolute units fails
# (the whole point of the change). We work from the ON-minus-OFF difference (NOT
# argmin(u)): the difference isolates the imposed bracket, so a strong seed-
# dependent natural jet cannot confound the measurement.

HERO_SCALE = -22.0
_DEG = float(np.rad2deg(1.0))


def _on_off_diff(seed, r_core, **jet_over):
    """(lat_deg, on.u - off.u) for a bracket-ON build at the hero, scaled by
    r_core. off is the bracket-off baseline; the diff is the imposed bracket."""
    p, bands = _rich(seed)
    off = build_profiles(seed, bands, p.bands, p.jets)
    v = PlanetParams(seed=seed)
    for k, val in jet_over.items():
        setattr(v.jets, k, val)
    on = build_profiles(seed, bands, p.bands, v.jets,
                        hero_lat_deg=HERO_SCALE, hero_r_core=r_core)
    return np.rad2deg(on.lat), on.u - off.u


@pytest.mark.parametrize("flank", ["north", "south"])
def test_bracket_jet_offset_scales_with_hero_radius(flank):
    """The imposed jet's distance from the hero == |offset| * r_core (jet center
    latitude = hero + offset*r_core). Asserted against the ANALYTIC value at two
    radii for BOTH flanks (drive that flank at max magnitude so its dip dominates
    the r-invariant carve of the seeded jets; the argmin lands on the gaussian
    center). Covers north_offset AND south_offset: a south_offset left in absolute
    units would push the poleward jet ~57 deg out (blend weight ~0), silently
    dropping half the bracket while the two-sided-shear test still passed on the
    north jet alone. (Asserting each measurement against its analytic value, not a
    2x ratio, avoids compounding the two measurements' small background pulls.)"""
    if flank == "north":
        over = {"hero_bracket_north": -3.0, "hero_bracket_south": 0.0,
                "hero_bracket_north_offset": 1.0}
        abs_offset = 1.0
    else:  # drive the poleward jet westward (negative) so argmin locates its dip
        over = {"hero_bracket_north": 0.0, "hero_bracket_south": -3.0,
                "hero_bracket_south_offset": -1.0}
        abs_offset = 1.0

    for seed in (3, 17, 42):                       # robust across seeds, not seed-lucky
        for r_core in (0.05, 0.10):
            lat_deg, diff = _on_off_diff(seed, r_core, **over)
            near = np.abs(lat_deg - HERO_SCALE) < 30.0
            idx = np.argmin(np.where(near, diff, np.inf))   # the imposed jet's center
            measured = abs(float(lat_deg[idx]) - HERO_SCALE)
            expected = abs_offset * r_core * _DEG
            assert measured == pytest.approx(expected, rel=0.1), \
                (flank, seed, r_core, measured, expected)


def test_bracket_carve_extent_scales_with_hero_radius():
    """The carve region is (window + feather) core radii wide, so the OUTERMOST
    modified latitude == (window+feather) * r_core. Beyond `outer` the blend
    weight w is exactly 0, so |on.u - off.u| is crisply zero there. Asserted
    against the analytic edge at two radii. Covers window AND feather (either left
    in absolute units breaks the match)."""
    for seed in (3, 17, 42):
        for r_core in (0.05, 0.10):
            lat_deg, diff = _on_off_diff(seed, r_core,
                                         hero_bracket_north=-1.0, hero_bracket_south=0.6)
            moved = np.abs(lat_deg - HERO_SCALE)[np.abs(diff) > 1e-9]
            measured = float(moved.max())
            expected = (1.0 + 1.4) * r_core * _DEG   # (window + feather) defaults
            assert measured == pytest.approx(expected, rel=0.02), \
                (seed, r_core, measured, expected)


@pytest.mark.parametrize("flank", ["north", "south"])
def test_bracket_jet_width_scales_with_hero_radius(flank):
    """The imposed jet's gaussian half-width is (width * r_core), so doubling
    hero_r_core doubles the jet's angular width. Measured on the DIFFERENCE of two
    same-r builds -- one at width 0.8, one at width 0.1 -- so the pedestal, seeded
    profile, and blend weight (all identical between the two builds) cancel
    EXACTLY, leaving a pure difference of gaussians with zero seed-dependent
    background; at the wide jet's half-depth radius the narrow spike has vanished,
    so the FWHM is the clean wide gaussian's. Covers BOTH widths: a width left
    absolute would make the jet a fixed-degree lump that does not track storm
    size. (The self-similar difference scales the FWHM by exactly 2x.)"""
    strength_field = f"hero_bracket_{flank}"
    width_field = f"hero_bracket_{flank}_width"
    offset_field = f"hero_bracket_{flank}_offset"

    def fwhm_deg(seed, r_core):
        p, bands = _rich(seed)

        def build(width):
            v = PlanetParams(seed=seed)
            setattr(v.jets, strength_field, -3.0)   # this flank only, westward dip
            setattr(v.jets, offset_field, 0.0)      # centered on the hero (w=1 core)
            setattr(v.jets, width_field, width)
            return build_profiles(seed, bands, p.bands, v.jets,
                                  hero_lat_deg=HERO_SCALE, hero_r_core=r_core)

        wide, narrow = build(0.8), build(0.1)
        lat_deg = np.rad2deg(wide.lat)
        near = np.abs(lat_deg - HERO_SCALE) < 25.0
        d = np.where(near, wide.u - narrow.u, 0.0)   # pure gaussian difference
        c = int(np.argmin(d))
        half = 0.5 * d[c]
        left = c
        while left > 0 and d[left] <= half:
            left -= 1
        xl = np.interp(half, [d[left], d[left + 1]], [lat_deg[left], lat_deg[left + 1]])
        right = c
        while right < len(d) - 1 and d[right] <= half:
            right += 1
        xr = np.interp(half, [d[right], d[right - 1]], [lat_deg[right], lat_deg[right - 1]])
        return abs(xl - xr)

    for seed in (3, 17, 42):
        w1 = fwhm_deg(seed, 0.05)
        w2 = fwhm_deg(seed, 0.10)
        assert w2 == pytest.approx(2.0 * w1, rel=0.05), (flank, seed, w1, w2)


def test_active_bracket_without_radius_raises():
    """Activating the bracket while leaving hero_r_core at its 0.0 default is a
    caller bug (forgotten kwarg); build_profiles raises rather than silently
    dividing by zero. The facade always passes the radius, so this never fires
    in production -- it protects test authors and future callers."""
    seed = 1
    p, bands = _rich(seed)
    v = PlanetParams(seed=seed)
    v.jets.hero_bracket_north = -1.0
    with pytest.raises(ValueError, match="hero_r_core"):
        build_profiles(seed, bands, p.bands, v.jets, hero_lat_deg=-22.0)  # no hero_r_core
