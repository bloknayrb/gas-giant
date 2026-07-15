"""storms.hero_wake_dir + the flow-derived wake frame (_hero_wake_frame).

CPU-only: exercises vortex GENERATION (the wake frame is registry data packed
into the SSBO; the kernels just consume wake_dir/wake_lat_off).

Key invariants:
- default (auto, emergence off) is the legacy authored frame: wake_dir -1,
  lane 0.5 r equatorward — byte-identical registries for legacy presets;
- auto + emergence: wake_dir follows the sign of the strongest jet within
  [0.4 r, 2.5 r] equatorward of the hero (east on gas_giant_warm — the whole
  hero band flows east), and the lane sits AT that jet;
- east/west force wake_dir regardless of the flow, with the lane unchanged
  (it keeps tracking the jet: that is where the material is);
- the override does not perturb the RNG stream (placement identical).
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
    prof = build_profiles(p.seed, bands, p.bands, p.jets)
    reg = generate_vortices(p.seed, bands, prof, p.storms, p.poles)
    return reg.heroes()[0], prof


def test_default_auto_with_emergence_follows_the_jet():
    """warm ships auto + emergence 0.9; its hero band flows EAST everywhere,
    so the dynamic wake must trail east and the lane must sit in real flow
    (|u| >= 0.05), not at the legacy stagnation offset."""
    hero, prof = _hero()
    assert hero.wake_dir == 1.0
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
