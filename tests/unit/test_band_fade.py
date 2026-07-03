"""bands.belt_fade + bands.faded_band_index (2026-07-02 review B5-2/B5-4, W6).

The whole-belt fade (SEB-fade epoch) applies at derive time AFTER validation,
ONLY to the T0-stamp/value path: BandLayout grows a ``stamp_values`` array the
stamp profiles read, while ``values`` stays PRE-fade for every identity/
dynamics consumer (is_belt, storm seeding, outbreak candidate selection and
its darkest-half ordering). ``faded_band_index`` makes the target band
user-addressable and overrides BOTH the fade target and the faded_sector
longitude window's widest-belt heuristic (B5-4's 0.01-degree tiebreak hazard).

Recorded LIMIT (mirrored in the field description): a faded belt keeps
belt-like churn/dynamics -- the fade is visual (T0), not dynamical. That is
the real SEB-fade phenomenology (revival outbreaks erupt IN the faded belt).
"""

from __future__ import annotations

import numpy as np
import pytest

from gasgiant.params.model import BandsParams, PlanetParams
from gasgiant.params.presets import load_factory_preset
from gasgiant.sim.bands import generate_bands
from gasgiant.sim.events import EventSchedule
from gasgiant.sim.profiles import build_profiles
from gasgiant.sim.vortices import _band_centers, generate_vortices


def _params(preset: str | None = "jupiter_vorticity", **bands) -> PlanetParams:
    p = load_factory_preset(preset) if preset else PlanetParams(seed=4201)
    for key, value in bands.items():
        setattr(p.bands, key, value)
    return p


def _layout(preset: str | None = "jupiter_vorticity", **bands):
    p = _params(preset, **bands)
    return p, generate_bands(p.seed, p.bands)


def _neighbor_ref(values: np.ndarray, j: int) -> float:
    neighbors = [values[k] for k in (j - 1, j + 1) if 0 <= k < len(values)]
    return float(np.mean(neighbors))


# ------------------------------------------------------------------ defaults

def test_default_layout_has_prefade_stamp_values():
    """belt_fade=0 -> stamp_values IS the values array (no drift possible)."""
    for preset in ("jupiter_vorticity", None):
        _, bands = _layout(preset)
        assert bands.stamp_values is not None
        np.testing.assert_array_equal(bands.stamp_values, bands.values)
        assert bands.fade_index is not None  # the heuristic pick is recorded


def test_forced_variant_noop_zero_fade_with_explicit_index():
    """Lever-author checklist: forcing the variant at its neutral value
    (belt_fade=0.0 plus faded_band_index pinned to the band the widest-belt
    heuristic already picks) is IDENTICAL to the default layout."""
    for preset in ("jupiter_vorticity", None):
        _, base = _layout(preset)
        _, forced = _layout(preset, belt_fade=0.0, faded_band_index=int(base.fade_index))
        np.testing.assert_array_equal(forced.edges, base.edges)
        np.testing.assert_array_equal(forced.values, base.values)
        np.testing.assert_array_equal(forced.stamp_values, base.stamp_values)
        np.testing.assert_array_equal(forced.heights, base.heights)
        np.testing.assert_array_equal(forced.is_belt, base.is_belt)
        assert forced.fade_sector == base.fade_sector


# ------------------------------------------------------------------ the fade

def test_belt_fade_full_reaches_neighbor_zone_level():
    """belt_fade=1.0 -> the faded belt's stamp value equals the mean of its
    neighboring bands' (zone) values; every other band and the pre-fade
    values array are untouched."""
    _, base = _layout()
    _, faded = _layout(belt_fade=1.0)
    j = int(faded.fade_index)
    assert bool(base.is_belt[j])
    np.testing.assert_array_equal(faded.values, base.values)  # pre-fade view
    expected = _neighbor_ref(base.values, j)
    assert faded.stamp_values[j] == pytest.approx(expected, abs=1e-6)
    mask = np.ones(len(base.values), dtype=bool)
    mask[j] = False
    np.testing.assert_array_equal(faded.stamp_values[mask], base.values[mask])


def test_belt_fade_is_proportional():
    _, base = _layout()
    _, half = _layout(belt_fade=0.5)
    j = int(half.fade_index)
    expected = base.values[j] + 0.5 * (_neighbor_ref(base.values, j) - base.values[j])
    assert half.stamp_values[j] == pytest.approx(expected, abs=1e-6)


def test_belt_fade_flows_into_t0_stamp_profile():
    """The stamp profiles read stamp_values: at belt_fade=1.0 the faded belt's
    T0 stamp interior sits at the neighbor-zone level, not the belt level."""
    p_base = _params()
    p_faded = _params(belt_fade=1.0)
    bands_base = generate_bands(p_base.seed, p_base.bands)
    bands_faded = generate_bands(p_faded.seed, p_faded.bands)
    prof_base = build_profiles(p_base.seed, bands_base, p_base.bands, p_base.jets)
    prof_faded = build_profiles(p_faded.seed, bands_faded, p_faded.bands, p_faded.jets)

    j = int(bands_faded.fade_index)
    lat_hi, lat_lo = float(bands_faded.edges[j]), float(bands_faded.edges[j + 1])
    pad = 0.25 * (lat_hi - lat_lo)  # stay clear of the edge smoothstep
    rows = (prof_faded.lat < lat_hi - pad) & (prof_faded.lat > lat_lo + pad)
    assert rows.any()
    expected = _neighbor_ref(bands_base.values, j)
    assert float(np.abs(prof_faded.t0_stamp[rows] - expected).max()) < 1e-4
    # The belt was actually dark before: the fade moved it.
    assert float(np.abs(prof_base.t0_stamp[rows] - expected).max()) > 0.05
    # T1 (heights) and the belt mask are NOT faded.
    np.testing.assert_array_equal(prof_faded.t1_stamp, prof_base.t1_stamp)
    np.testing.assert_array_equal(prof_faded.belt_mask, prof_base.belt_mask)


# --------------------------------------------------- identity stability (W3)

def test_identity_dynamics_and_candidates_stable_under_fade():
    """is_belt, storm-seeding band centers, the vortex registry, and the
    outbreak schedule (candidate set AND darkest-half ordering) are IDENTICAL
    with the fade on vs off -- the fade is visual-only by construction."""
    p_base = _params(preset="jupiter_vorticity")
    p_base.storms.outbreak_count = 2
    p_faded = p_base.model_copy(deep=True)
    p_faded.bands.belt_fade = 1.0

    bands_base = generate_bands(p_base.seed, p_base.bands)
    bands_faded = generate_bands(p_faded.seed, p_faded.bands)
    np.testing.assert_array_equal(bands_faded.is_belt, bands_base.is_belt)
    assert _band_centers(bands_faded, want_belt=True) == _band_centers(bands_base, want_belt=True)

    prof_base = build_profiles(p_base.seed, bands_base, p_base.bands, p_base.jets)
    prof_faded = build_profiles(p_faded.seed, bands_faded, p_faded.bands, p_faded.jets)
    reg_base = generate_vortices(p_base.seed, bands_base, prof_base, p_base.storms, p_base.poles)
    reg_faded = generate_vortices(p_faded.seed, bands_faded, prof_faded, p_faded.storms,
                                  p_faded.poles)
    assert reg_faded == reg_base

    assert EventSchedule.generate(p_faded, bands_faded) == \
        EventSchedule.generate(p_base, bands_base)


# ------------------------------------------------------------ B5-4 targeting

def test_faded_band_index_redirects_fade_and_sector():
    """An explicit index overrides BOTH the fade target and the faded_sector
    window (no more silent 0.01-degree widest-belt tiebreak)."""
    _, base = _layout()
    auto_j = int(base.fade_index)
    # Pick a DIFFERENT belt than the heuristic's.
    other = [j for j in range(len(base.values))
             if bool(base.is_belt[j]) and j != auto_j]
    assert other
    j = other[0]
    _, redir = _layout(belt_fade=1.0, faded_band_index=j)
    assert int(redir.fade_index) == j
    # The sector spans the chosen band's edges (lat_lo, lat_hi ascending).
    assert redir.fade_sector[0] == pytest.approx(float(base.edges[j + 1]))
    assert redir.fade_sector[1] == pytest.approx(float(base.edges[j]))
    # The fade lands on the chosen band, not the heuristic's.
    assert redir.stamp_values[j] == pytest.approx(_neighbor_ref(base.values, j), abs=1e-6)
    np.testing.assert_array_equal(
        np.delete(redir.stamp_values, j), np.delete(base.values, j))
    # The seeded longitude/halfwidth draws are untouched by the override.
    assert redir.fade_sector[2:] == base.fade_sector[2:]


def test_faded_band_index_may_target_a_zone():
    """The ochre-EZ recipe: pointing the index at a ZONE blends it toward its
    (belt) neighbors' mean -- down-palette into the warm tans."""
    _, base = _layout()
    zones = [j for j in range(len(base.values)) if not bool(base.is_belt[j])
             and 0 < j < len(base.values) - 1]
    j = zones[len(zones) // 2]
    _, faded = _layout(belt_fade=0.5, faded_band_index=j)
    expected = base.values[j] + 0.5 * (_neighbor_ref(base.values, j) - base.values[j])
    assert faded.stamp_values[j] == pytest.approx(expected, abs=1e-6)
    assert faded.stamp_values[j] < base.values[j]  # zones fade DARKER


# ------------------------------------------------------------------ validator

def test_faded_band_index_validator_rejects_out_of_range():
    with pytest.raises(ValueError, match="faded_band_index"):
        BandsParams(count=14, faded_band_index=14)
    BandsParams(count=14, faded_band_index=13)  # in range -> fine
    with pytest.raises(ValueError):
        BandsParams(count=14, faded_band_index=-1)


def test_faded_band_index_validator_uses_template_band_count():
    template = load_factory_preset("jupiter_vorticity").bands.template
    n = len(template.values)
    with pytest.raises(ValueError, match="faded_band_index"):
        BandsParams(template=template, faded_band_index=n)
    BandsParams(template=template, faded_band_index=n - 1)


def test_faded_band_index_revalidates_on_assignment():
    p = BandsParams(count=6)
    with pytest.raises(ValueError, match="faded_band_index"):
        p.faded_band_index = 6
