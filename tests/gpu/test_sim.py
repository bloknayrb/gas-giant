from __future__ import annotations

import numpy as np
import pytest

from gasgiant.engine import Simulation
from gasgiant.params.model import PlanetParams, Tier
from gasgiant.params.presets import load_factory_preset
from gasgiant.validate import validate_arrays

pytestmark = pytest.mark.gpu


def _small_params(seed: int = 1234, steps: int = 40) -> PlanetParams:
    p = PlanetParams(seed=seed)
    p.sim.resolution = 512
    p.sim.dev_steps = steps
    return p


@pytest.fixture(scope="module")
def small_maps(gpu):
    sim = Simulation(_small_params(), gpu)
    return sim.render_maps(width=512)


def test_pipeline_runs(small_maps):
    assert small_maps["color"].shape == (256, 512, 4)
    assert small_maps["height"].shape == (256, 512)


def test_output_finite_and_in_range(small_maps):
    for name in ("color", "height"):
        arr = small_maps[name]
        assert np.isfinite(arr).all(), f"{name} has non-finite values"
        assert arr.min() >= -1e-5 and arr.max() <= 1.0 + 1e-5, f"{name} out of [0,1]"


def test_seam_and_pole_invariants(small_maps):
    report = validate_arrays(
        {"color": small_maps["color"][..., :3], "height": small_maps["height"]}
    )
    assert report.ok, report.summary()


def test_same_seed_is_deterministic(gpu):
    a = Simulation(_small_params(seed=42, steps=20), gpu).render_maps(width=256)
    b = Simulation(_small_params(seed=42, steps=20), gpu).render_maps(width=256)
    np.testing.assert_array_equal(a["color"], b["color"])
    np.testing.assert_array_equal(a["height"], b["height"])


def test_different_seeds_differ(gpu):
    a = Simulation(_small_params(seed=1, steps=10), gpu).render_maps(width=256)
    b = Simulation(_small_params(seed=2, steps=10), gpu).render_maps(width=256)
    assert not np.array_equal(a["color"], b["color"])


def test_advection_changes_tracers(gpu):
    """The development run must actually move structure around."""
    sim = Simulation(_small_params(steps=0), gpu)
    before = sim.tracers.read_current()
    sim.solver.step(25)
    after = sim.tracers.read_current()
    diff = np.abs(after - before).mean()
    assert diff > 1e-4, f"tracers barely changed after 25 steps (mean diff {diff:.2e})"


def test_no_washout_and_no_blowup(gpu):
    """Tracer variance must survive a long run (relaxation + limiter at work)."""
    sim = Simulation(_small_params(steps=0), gpu)
    t0_initial = sim.tracers.read_current()[..., 0]
    sim.solver.step(300)
    t0_final = sim.tracers.read_current()[..., 0]
    assert np.isfinite(t0_final).all()
    # Variance must not collapse (washout) nor explode (instability).
    var0, var1 = t0_initial.std(), t0_final.std()
    assert var1 > 0.35 * var0, f"washout: std {var0:.4f} -> {var1:.4f}"
    assert var1 < 3.0 * var0, f"blow-up: std {var0:.4f} -> {var1:.4f}"
    # The limiter must keep values in the physical range.
    assert t0_final.min() >= -1e-4 and t0_final.max() <= 1.0 + 1e-4


def test_zonal_advection_preserves_pattern(gpu):
    """With jets only (no storms/turbulence/KH, weak relaxation), advection
    should transport the banded pattern without destroying it."""
    p = _small_params(steps=0)
    p.storms.hero_count = 0
    p.storms.oval_density = 0.0
    p.storms.barge_density = 0.0
    p.storms.pearls_count = 0
    p.turbulence.intensity = 0.0
    p.turbulence.kh_amplitude = 0.0
    p.turbulence.relax_tau = 2000.0
    p.turbulence.replenish_rate = 0.0
    sim = Simulation(p, gpu)
    before = sim.tracers.read_current()[..., 0]
    sim.solver.step(100)
    after = sim.tracers.read_current()[..., 0]
    assert np.isfinite(after).all()
    assert after.std() > 0.6 * before.std(), "pure zonal advection diffused the pattern too much"


def test_factory_presets_render(gpu):
    for name in ("jupiter_like", "saturn_pale"):
        params = load_factory_preset(name)
        params.sim.resolution = 512
        params.sim.dev_steps = 30
        maps = Simulation(params, gpu).render_maps(width=256)
        report = validate_arrays({"color": maps["color"][..., :3]})
        assert report.ok, f"{name}: {report.summary()}"


def test_haze_reduces_contrast(gpu):
    base = _small_params(seed=10, steps=15)
    hazy = base.model_copy(deep=True)
    hazy.appearance.haze_amount = 0.8
    a = Simulation(base, gpu).render_maps(width=256)["color"]
    b = Simulation(hazy, gpu).render_maps(width=256)["color"]
    assert b.std() < a.std()


def test_velocity_tier_update_does_not_restart(gpu):
    sim = Simulation(_small_params(steps=20), gpu)
    sim.run_to_completion()
    steps_before = sim.steps_done
    new = sim.params.model_copy(deep=True)
    new.jets.strength = 1.5
    tiers = sim.update_params(new)
    assert Tier.VELOCITY in tiers and Tier.RESTART not in tiers
    assert sim.steps_done == steps_before  # run not reset
    assert sim.steps_target > steps_before  # adaptation steps queued


def test_restart_tier_reinitializes(gpu):
    sim = Simulation(_small_params(steps=10), gpu)
    sim.run_to_completion()
    new = sim.params.model_copy(deep=True)
    new.seed = 999
    sim.update_params(new)
    assert sim.steps_done == 0
