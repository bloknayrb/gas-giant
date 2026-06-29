from __future__ import annotations

import numpy as np
import pytest

from gasgiant.engine import Simulation
from gasgiant.engine.checkpoint import load_checkpoint, save_checkpoint
from gasgiant.params.model import PlanetParams

pytestmark = pytest.mark.gpu


def test_checkpoint_round_trip(gpu, tmp_path):
    p = PlanetParams(seed=55)
    p.sim.resolution = 512
    p.sim.dev_steps = 60
    sim = Simulation(p, gpu)
    sim.solver.step(30)
    path = tmp_path / "state.npz"
    save_checkpoint(sim, path)

    restored = load_checkpoint(path, gpu)
    assert restored.steps_done == 30
    np.testing.assert_array_equal(
        restored.tracers.read_current(), sim.tracers.read_current()
    )

    # Continuing from the restored state matches continuing the original.
    sim.solver.step(10)
    restored.solver.step(10)
    np.testing.assert_allclose(
        restored.tracers.read_current(), sim.tracers.read_current(), atol=1e-6
    )


def _registry_fields(sim):
    return [
        (v.lat, v.lon, v.r_core, v.strength, v.kind, v.tint, v.brightness, v.wake_dir)
        for v in sim.solver.vortices.vortices
    ]


def test_mid_outbreak_checkpoint_restores_registry_exactly(gpu, tmp_path):
    """v1.1 replayed drift() only: a mid-outbreak save restored a registry
    missing the outbreak vortex, and the rebuilt schedule re-spawned it at its
    un-drifted longitude. The registry is now serialized."""
    from gasgiant.sim.events import KIND_OUTBREAK

    sim = None
    for seed in (3, 5, 7, 11, 13):
        p = PlanetParams(seed=seed)
        p.sim.resolution = 512
        p.sim.dev_steps = 100
        p.storms.outbreak_count = 1
        cand = Simulation(p, gpu)
        sched = cand.solver.events
        if sched and sched.outbreaks and sched.outbreaks[0].step <= 60:
            sim = cand
            break
    assert sim is not None, "no seed produced an early outbreak"

    sim.solver.step(70)  # outbreak alive: age in (10, 70], LIFETIME=160
    live_kinds = [v.kind for v in sim.solver.vortices.vortices]
    assert KIND_OUTBREAK in live_kinds  # premise: we are mid-outbreak

    path = tmp_path / "mid_outbreak.npz"
    save_checkpoint(sim, path)
    restored = load_checkpoint(path, gpu)

    assert _registry_fields(restored) == _registry_fields(sim)
    # The schedule's link survives: stepping does not re-spawn a duplicate.
    n_before = len(restored.solver.vortices.vortices)
    sim.solver.step(10)
    restored.solver.step(10)
    assert len(restored.solver.vortices.vortices) == len(sim.solver.vortices.vortices)
    assert n_before == len(restored.solver.vortices.vortices)
    np.testing.assert_array_equal(
        restored.tracers.read_current(), sim.tracers.read_current()
    )


def test_velocity_edit_checkpoint_restores_adaptation(gpu, tmp_path):
    """_extra_steps (the VELOCITY-edit adaptation window) must round-trip:
    a save mid-adaptation previously restored as 'developed' and froze."""
    p = PlanetParams(seed=9)
    p.sim.resolution = 512
    p.sim.dev_steps = 60
    sim = Simulation(p, gpu)
    sim.run_to_completion()

    edited = sim.params.model_copy(deep=True)
    edited.jets.strength = sim.params.jets.strength * 1.2
    sim.update_params(edited)
    assert not sim.is_developed  # adaptation window open
    target = sim.steps_target
    sim.solver.step(10)  # mid-adaptation

    path = tmp_path / "mid_adapt.npz"
    save_checkpoint(sim, path)
    restored = load_checkpoint(path, gpu)

    assert restored.steps_target == target
    assert not restored.is_developed
    assert _registry_fields(restored) == _registry_fields(sim)
    np.testing.assert_array_equal(
        restored.tracers.read_current(), sim.tracers.read_current()
    )


def test_merger_checkpoint_round_trips_cooldowns(gpu, tmp_path):
    """With mergers enabled, the serialized registry (incl. cooldown) must
    restore exactly and continue bit-identically."""
    p = PlanetParams(seed=42)
    p.sim.resolution = 512
    p.sim.dev_steps = 400
    p.storms.merge_rate = 0.7
    sim = Simulation(p, gpu)
    sim.solver.step(200)  # past the earliest seeded merge targets

    path = tmp_path / "mergers.npz"
    save_checkpoint(sim, path)
    restored = load_checkpoint(path, gpu)

    assert _registry_fields(restored) == _registry_fields(sim)
    assert [v.cooldown for v in restored.solver.vortices.vortices] == [
        v.cooldown for v in sim.solver.vortices.vortices
    ]
    sim.solver.step(30)
    restored.solver.step(30)
    assert _registry_fields(restored) == _registry_fields(sim)
    np.testing.assert_array_equal(
        restored.tracers.read_current(), sim.tracers.read_current()
    )


def test_checkpoint_from_older_generation_is_refused(gpu, tmp_path):
    """A checkpoint pairs saved tracers with a registry replayed from the
    seed; if the generation algorithms changed since, it must refuse loudly
    rather than load and render subtly wrong."""
    import json

    from gasgiant.engine.checkpoint import GENERATION_VERSION
    from gasgiant.params.presets import to_preset_doc

    p = PlanetParams(seed=5)
    p.sim.resolution = 512
    path = tmp_path / "old.npz"
    blank = np.zeros((1, 1, 4), dtype=np.float32)
    # A v1-era checkpoint: no generation_version key at all.
    np.savez_compressed(
        path, preset=json.dumps(to_preset_doc(p)), step_index=3,
        tracers_eq=blank, tracers_n=blank, tracers_s=blank,
    )
    with pytest.raises(ValueError, match="generation_version"):
        load_checkpoint(path, gpu)
    assert GENERATION_VERSION >= 2


def test_vorticity_checkpoint_round_trip(gpu, tmp_path):
    """Vorticity mode: checkpoint must round-trip ω + warm-start ψ for ALL THREE
    domains so that continuing from the restored sim matches continuing the
    original.  The advected absolute-vorticity field and the warm-start ψ are
    PROGNOSTIC state — without them a resumed run DIVERGES (order ~0.1-1) within a
    few steps.

    Two tiers of strictness:
      * Immediately after restore the state is just deserialized (no recompute), so
        tracers + ω must be BYTE-EXACT (assert_array_equal) — this is the real
        serialization-fidelity guard.
      * After continuing both sims the comparison runs back through the vorticity
        SOR Poisson solve, which is NOT byte-deterministic (~1e-3 LSB noise; the
        kinematic path IS exact — see the sibling kinematic test's atol=1e-6). So
        the continuation is checked within the SOR noise floor, which still catches
        missing-prognostic-state divergence (>> the floor) with ~100x margin."""
    from gasgiant.params.model import SolverType
    from gasgiant.sim.solver import DOMAIN_EQUIRECT, DOMAIN_NORTH, DOMAIN_SOUTH

    p = PlanetParams(seed=77)
    p.sim.resolution = 512
    p.sim.dev_steps = 100
    p.solver.type = SolverType.VORTICITY
    sim = Simulation(p, gpu)
    sim.solver.step(30)

    path = tmp_path / "vort_state.npz"
    save_checkpoint(sim, path)

    restored = load_checkpoint(path, gpu)
    assert restored.steps_done == 30

    # Tracers must be byte-identical immediately after restore.
    np.testing.assert_array_equal(
        restored.tracers.read_current(), sim.tracers.read_current()
    )

    # ω field must be byte-identical for ALL three domains immediately after restore.
    for kind in (DOMAIN_EQUIRECT, DOMAIN_NORTH, DOMAIN_SOUTH):
        np.testing.assert_array_equal(
            gpu.read_texture(restored.solver._omega_states[kind].cur),
            gpu.read_texture(sim.solver._omega_states[kind].cur),
        ), f"omega mismatch for domain {kind} immediately after restore"

    # Continuing both sims another 10 steps must stay matched within the vorticity
    # SOR noise floor: this proves the warm-start ψ and ω serialization are
    # sufficient (missing prognostic state would diverge by order ~0.1-1, >> floor).
    _VORT_SOR_ATOL = 1e-3  # documented vorticity SOR LSB noise; kinematic path is exact
    sim.solver.step(10)
    restored.solver.step(10)
    np.testing.assert_allclose(
        restored.tracers.read_current(), sim.tracers.read_current(), atol=_VORT_SOR_ATOL
    )
    for kind in (DOMAIN_EQUIRECT, DOMAIN_NORTH, DOMAIN_SOUTH):
        np.testing.assert_allclose(
            gpu.read_texture(restored.solver._omega_states[kind].cur),
            gpu.read_texture(sim.solver._omega_states[kind].cur),
            atol=_VORT_SOR_ATOL,
            err_msg=f"omega mismatch for domain {kind} after 10 more steps",
        )
