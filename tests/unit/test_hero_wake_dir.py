"""storms.hero_wake_dir + the flow-derived wake frame (_hero_wake_frame).

CPU-only: exercises vortex GENERATION (the wake frame is registry data packed
into the SSBO; the kernels just consume wake_dir/wake_lat_off).

Key invariants:
- default (auto, emergence off) is the legacy authored frame: wake_dir -1,
  lane 0.5 r equatorward — byte-identical registries for legacy presets;
- auto + emergence: wake_dir follows the sign of the strongest jet within
  [0.4 r, 2.5 r] equatorward of the hero;
- east/west force wake_dir regardless of the flow, with the lane unchanged
  (it keeps tracking the jet: that is where the material is);
- the override does not perturb the RNG stream (placement identical).

GRS bake (2026-07-19): warm's hero moved to -24 and the anticyclonic bearing
that carves its wake frame is now supplied by the size-relative carve-and-impose
BRACKET (jets.hero_bracket_*), which SUPERSEDED the earlier local_jet. The
bracket lives in build_profiles and needs a pinned hero (hero_lat_deg), so -- to
match the facade, where generate_vortices runs on the BRACKETED profile -- these
tests build the profile with the pin (see _hero). The bracket's equatorward jet
is westward (north strength -3.0), so the auto+emergence wake is still WEST
(wake_dir -1.0), the same class the chirality fix established, now authored via
the bracket instead of local_jet.
"""
from __future__ import annotations

import numpy as np

from gasgiant.params.model import WakeDir
from gasgiant.params.presets import load_factory_preset
from gasgiant.sim.bands import generate_bands
from gasgiant.sim.profiles import build_profiles
from gasgiant.sim.vortices import generate_vortices


def _hero(preset: str = "gas_giant_warm", **storms_update):
    p = load_factory_preset(preset)
    if storms_update:
        p.storms = p.storms.model_copy(update=storms_update)
    bands = generate_bands(p.seed, p.bands)
    # Match the facade: generate_vortices runs on the BRACKETED profile (the hero
    # jet environment is what seats the wake frame). Pin predicate mirrors
    # facade._hero_lat_deg / _hero_r_core.
    hero_lat = p.storms.hero_latitude if p.storms.hero_count > 0 else None
    prof = build_profiles(p.seed, bands, p.bands, p.jets,
                          hero_lat_deg=hero_lat, hero_r_core=p.storms.hero_radius)
    reg = generate_vortices(p.seed, bands, prof, p.storms, p.poles)
    return reg.heroes()[0], prof


def test_default_auto_with_emergence_follows_the_jet():
    """warm ships auto + emergence 0.9 at hero_latitude=-24.0; the bracket's
    equatorward westward jet (hero_bracket_north -3.0) seats the bearing, so the
    dynamic wake trails WEST and the lane must sit in real flow (|u| >= 0.05), not
    at the legacy stagnation offset. (Measured on the bracketed profile, matching
    the facade -- the un-bracketed profile no longer carries the bearing.)"""
    hero, prof = _hero()
    assert hero.wake_dir == -1.0
    lane = hero.lat + hero.wake_lat_off
    u_lane = float(np.interp(-lane, -prof.lat, prof.u))
    assert abs(u_lane) >= 0.05, "auto lane sits in dead flow"
    assert np.sign(u_lane) == np.sign(hero.wake_dir)


def test_legacy_frame_without_emergence():
    """Emergence off => the authored F06 frame verbatim (wake_dir -1, lane
    0.5 r equatorward), regardless of the local jet — legacy presets must
    keep byte-identical registries."""
    hero, _ = _hero(hero_emergence=0.0)
    assert hero.wake_dir == -1.0
    assert np.isclose(hero.wake_lat_off, 0.5 * hero.r_core)


def test_forced_directions_override_auto():
    east, _ = _hero(hero_wake_dir=WakeDir.EAST)
    west, _ = _hero(hero_wake_dir=WakeDir.WEST)
    auto, _ = _hero(hero_wake_dir=WakeDir.AUTO)
    assert east.wake_dir == 1.0
    assert west.wake_dir == -1.0
    # The override only flips the direction: placement and lane are the
    # auto frame's (no RNG-stream perturbation, lane keeps tracking the jet).
    for forced in (east, west):
        assert forced.lat == auto.lat and forced.lon == auto.lon
        assert forced.r_core == auto.r_core
        assert forced.wake_lat_off == auto.wake_lat_off
