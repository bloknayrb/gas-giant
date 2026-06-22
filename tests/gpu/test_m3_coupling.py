"""M3 coupling integration: the evolving baroclinic source develops a v1.6 run,
records coverage, changes the render, and stays coherent."""
from __future__ import annotations

import numpy as np
import pytest

from gasgiant.engine.baroclinic_coupling import run_coupled
from gasgiant.engine.facade import Simulation
from gasgiant.params.model import PlanetParams, SolverType
from gasgiant.params.presets import load_factory_preset
from gasgiant.sim.baroclinic_driver import BaroclinicSourceDriver
from gasgiant.sim.solver import DOMAIN_EQUIRECT

pytestmark = pytest.mark.gpu


def _params(steps: int) -> PlanetParams:
    p = load_factory_preset("jupiter_vorticity")
    p.sim.resolution = 512
    p.sim.dev_steps = steps
    p.solver.type = SolverType.VORTICITY
    return p


def _midband_mean_abs_omega(sim: Simulation, lat_cut_deg: float = 50.0) -> float:
    """Mean |omega| over the mid-latitude active band, excluding the poles.

    A whole-grid max/mean is useless here: the uncoupled baseline already pins ~5%
    of pixels at the +/-60 ceiling near the poles (the 1/cos^2 metric blow-up). The
    baroclinic source injects in the mid-latitudes (~10-80deg), so the discriminator
    is the mid-latitude mean, where the baseline is well-behaved (~2.6) and the
    pre-fix accumulation inflates it several-fold."""
    arr = np.abs(np.asarray(sim.gpu.read_texture(sim.solver._omega_states[DOMAIN_EQUIRECT].cur)))
    h = arr.shape[0]
    lat = 90.0 - (np.arange(h) + 0.5) / h * 180.0
    return float(arr[np.abs(lat) < lat_cut_deg, :].mean())


def test_coupled_run_develops_and_changes_render(gpu):
    # The RHS-injected source is zero-lag (read fresh each step, decoupled from
    # vort_relax_tau), but its render signature only reads clearly once the v1.6
    # jet itself has developed over ~tau steps; develop ~100 steps at gain=2 for a
    # clear render delta.
    base = Simulation(_params(steps=96), gpu)
    base.run_to_completion(chunk=64)
    base_rgb = np.clip(base.render_maps(512)["color"][..., :3], 0, 1)
    base._release_sim()

    sim = Simulation(_params(steps=96), gpu)
    w, h = sim.solver.equirect.size
    driver = BaroclinicSourceDriver(grid_w=w, grid_h=h, warmup_steps=3000, seed=0)
    stats = run_coupled(sim, driver, gain=2.0, update_every=4,
                        baro_steps_per_update=100)
    coupled_rgb = np.clip(sim.render_maps(512)["color"][..., :3], 0, 1)
    sim._release_sim()

    assert sim.is_developed
    assert stats.v16_steps >= 96
    assert stats.source_updates >= 4
    assert float(np.abs(coupled_rgb - base_rgb).mean()) > 1e-4


def test_coupled_vorticity_bounded_over_long_horizon(gpu):
    """REGRESSION (band-crossing distortion): the baroclinic source is injected into
    the Poisson RHS (omega_recover.comp), NOT the persistent vorticity state q, so it
    never accumulates into q -- the mid-latitude band stays at baseline magnitude even
    at high gain over a long run. The original M3 spike injected into q every step,
    accumulating to ~tau*gain*src and inflating the mid-band several-fold (the smear).

    Measured at 512px/400 steps: baseline mid-band mean ~2.62; the RHS-injection build
    holds it at ~baseline; the original q-injection inflates it >5x (test fails).

    NB this metric reads the q state, and RHS injection deliberately never touches q,
    so the bound is intentionally one-sided (upper only). The "source has effect"
    direction is covered on the render by test_coupled_run_develops_and_changes_render
    and by test_nonzero_gain_changes_output."""
    steps = 400
    base = Simulation(_params(steps=steps), gpu)
    base.run_to_completion(chunk=64)
    base_mean = _midband_mean_abs_omega(base)
    base._release_sim()

    sim = Simulation(_params(steps=steps), gpu)
    w, h = sim.solver.equirect.size
    driver = BaroclinicSourceDriver(grid_w=w, grid_h=h, warmup_steps=3000, seed=0)
    run_coupled(sim, driver, gain=2.0, update_every=32, baro_steps_per_update=150)
    coupled_mean = _midband_mean_abs_omega(sim)
    sim._release_sim()

    # RHS injection never touches q, so the mid-band stays at baseline; the 1.5x bound
    # passes with headroom while the old q-accumulation (>5x) fails.
    assert coupled_mean < 1.5 * base_mean, (coupled_mean, base_mean)


def test_coupled_enriches_texture_keeping_bands(gpu):
    """The RHS-injected baroclinic source enriches mid-latitude belt texture while
    leaving the banded structure intact -- it is NOT the old q-accumulation that
    piled vorticity into the active band and smeared it across bands. At 512px/96
    steps, gain=2: high-frequency energy stays near unity (well within the 0.5-2.0
    gate) and the latitude concentration stays near baseline (well within the 0.85x
    floor), i.e. bands are not smeared into uniform."""
    from gasgiant.render.m3_metrics import highfreq_energy, latitude_concentration

    base = Simulation(_params(steps=96), gpu)
    base.run_to_completion(chunk=64)
    base_rgb = np.clip(base.render_maps(512)["color"][..., :3], 0, 1)
    base._release_sim()

    sim = Simulation(_params(steps=96), gpu)
    w, h = sim.solver.equirect.size
    driver = BaroclinicSourceDriver(grid_w=w, grid_h=h, warmup_steps=6000, seed=0)
    run_coupled(sim, driver, gain=2.0, update_every=4, baro_steps_per_update=200)
    coupled_rgb = np.clip(sim.render_maps(512)["color"][..., :3], 0, 1)
    sim._release_sim()

    # Texture enriched, not smoothed away or exploded.
    ratio = highfreq_energy(coupled_rgb) / (highfreq_energy(base_rgb) + 1e-12)
    assert 0.5 <= ratio <= 2.0, ratio
    # Bands intact: latitude concentration does not collapse (would drop sharply if
    # the source smeared structure across bands like the old q-accumulation).
    base_conc = latitude_concentration(base_rgb)
    coupled_conc = latitude_concentration(coupled_rgb)
    assert coupled_conc >= 0.85 * base_conc, (coupled_conc, base_conc)
