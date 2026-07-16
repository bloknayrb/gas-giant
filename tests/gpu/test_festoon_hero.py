"""GPU tests for waves.festoon_hero_strength (the FESTOON2 variant).

A second festoon train rooted on the interior band edge nearest the hero —
plumes only, T3 only, stamped into the relaxation TARGET (wave_stamp.glsl), so
it survives the emergence pack's anchor nudge and band flush the same way the
hero itself does.

Variant discipline mirrors test_hero_emergence.py: the byte-exact asserts run
on the KINEMATIC path only (the vorticity SOR solve carries a documented ~1e-3
noise floor and is never byte-compared); the behavior assert on the warm
preset is tolerance-class.
"""
from __future__ import annotations

import numpy as np
import pytest

from gasgiant.engine import Simulation
from gasgiant.params.model import PlanetParams, SolverType

pytestmark = pytest.mark.gpu


def _kin_params(strength: float, hero_count: int = 1) -> PlanetParams:
    p = PlanetParams(seed=42)
    p.sim.resolution = 512
    p.sim.dev_steps = 60
    p.storms.hero_count = hero_count
    p.storms.hero_latitude = -22.5
    p.waves.festoon_hero_strength = strength
    assert p.solver.type == SolverType.KINEMATIC
    return p


def _developed(p: PlanetParams, gpu) -> np.ndarray:
    sim = Simulation(p, gpu)
    sim.run_to_completion(chunk=64)
    return sim.gpu.read_texture(sim.solver.equirect.tracers.cur)


def test_no_hero_predicate_is_byte_identical(gpu):
    """strength > 0 with NO hero: the facade passes hero_wave_lat=None, so
    _domain_defines never selects FESTOON2 — the default program runs and the
    output is byte-identical to off (predicate pin; a heroless config must
    never pay for a guaranteed no-op)."""
    off = _developed(_kin_params(0.0, hero_count=0), gpu)
    on = _developed(_kin_params(1.0, hero_count=0), gpu)
    np.testing.assert_array_equal(off, on)


def test_far_field_locality_with_variant_compiled(gpu):
    """With the variant COMPILED (hero + strength > 0 + a root edge), the far
    NORTH quarter is byte-identical: the train has no psi-side meander (the
    velocity field is untouched), the plume stamp is confined to its root
    band near the hero at -22.5 deg, and 60 steps of advection spread ~10 deg
    — nothing reaches +45 deg."""
    off = _developed(_kin_params(0.0), gpu)
    on = _developed(_kin_params(1.2), gpu)
    # Premise: the variant actually engaged (something changed somewhere).
    assert np.abs(on - off).max() > 1e-4, (
        "festoon_hero_strength changed nothing — did the -22.5 deg placement "
        "lose its in-reach band edge?"
    )
    h = off.shape[0]
    np.testing.assert_array_equal(on[: h // 4], off[: h // 4])


def test_plumes_stamp_blue_streamers_in_the_hero_band(gpu):
    """Behavior on the warm preset (vorticity, tolerance-class): turning the
    train on must push T3 DOWN (blue-gray streamers) in the plume band at
    longitudes away from the hero, and leave the far-north quarter within the
    documented vorticity noise floor."""
    from gasgiant.params.presets import load_factory_preset

    def run(strength: float):
        p = load_factory_preset("gas_giant_warm").model_copy(update={"seed": 7})
        p.sim.resolution = 512
        p.sim.dev_steps = 60
        assert p.solver.type == SolverType.VORTICITY
        p.storms.hero_longitude = 0.0
        p.waves = p.waves.model_copy(update={"festoon_hero_strength": strength})
        sim = Simulation(p, gpu)
        sim.run_to_completion(chunk=64)
        return sim, sim.gpu.read_texture(sim.solver.equirect.tracers.cur)

    sim_off, off = run(0.0)
    sim_on, on = run(1.0)
    assert sim_on.solver.hero_wave_lat is not None, (
        "warm's hero lost its in-reach band edge — FESTOON2 never engaged"
    )
    root = sim_on.solver.hero_wave_lat
    pc = root - np.sign(root) * 0.045          # plume center (dip convention)

    h, w = off.shape[:2]
    lat_axis = (0.5 - (np.arange(h) + 0.5) / h) * np.pi
    band = np.abs(lat_axis - pc) < 0.05
    cols = slice(w // 8, 3 * w // 8)           # quarter turn from the hero
    t3_drop = off[band, cols, 3].mean() - on[band, cols, 3].mean()
    assert t3_drop > 0.01, (
        f"plume band T3 dropped only {t3_drop:.4f} — the streamers are not "
        "stamping (or the flush is erasing a target-held feature, which the "
        "design forbids)"
    )
    far_north = slice(0, h // 4)
    assert np.abs(on[far_north] - off[far_north]).max() < 1e-2, (
        "festoon_hero_strength leaked past the vorticity noise floor into "
        "the far-north quarter"
    )
