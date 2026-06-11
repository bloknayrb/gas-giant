"""GPU: D3 — polar detail routes through the patch velocities."""

from __future__ import annotations

import numpy as np
import pytest

from gasgiant.engine import Simulation
from gasgiant.params.model import PlanetParams

pytestmark = pytest.mark.gpu


def _params(seed: int = 41) -> PlanetParams:
    p = PlanetParams(seed=seed)
    p.sim.resolution = 512
    p.sim.dev_steps = 30  # real velocities everywhere
    p.detail.intensity = 1.0
    return p


def test_polar_detail_no_longer_fades_to_neutral(gpu):
    """v1 faded the synthesized detail to a constant 0.5 poleward of ~66 deg;
    the routed backtrace must place real texture there."""
    sim = Simulation(_params(), gpu)
    sim.run_to_completion()
    detail_params = sim.params.detail
    out = sim.gpu.texture2d((512, 256), 1, "f4")
    try:
        from gasgiant.engine.snapshot import hero_centers
        from gasgiant.render.detail import PolarRoute
        from gasgiant.sim.solver import RHO_MAX

        s = sim.solver
        sim.detail_synth.synthesize(
            sim.params.seed, s.equirect.vel_tex, s.equirect.tracers.cur,
            sim.profile_dyn, out, detail_params,
            heroes=hero_centers(sim.vortices),
            polar=PolarRoute(s.north.vel_tex, s.south.vel_tex,
                             s.north.tracers.cur, s.south.tracers.cur, RHO_MAX),
        )
        routed = sim.gpu.read_texture(out)[..., 0]
        sim.detail_synth.synthesize(
            sim.params.seed, s.equirect.vel_tex, s.equirect.tracers.cur,
            sim.profile_dyn, out, detail_params,
            heroes=hero_centers(sim.vortices),
            polar=None,  # legacy fade
        )
        legacy = sim.gpu.read_texture(out)[..., 0]
    finally:
        out.release()

    h = routed.shape[0]
    cap = slice(0, int(h * (90 - 78) / 180))  # poleward of 78N
    assert legacy[cap].std() < 1e-4            # v1: faded flat
    assert routed[cap].std() > 0.02            # routed: real texture
    # Equatorward of the route band the two paths are identical.
    mid = slice(int(h * 0.25), int(h * 0.75))
    np.testing.assert_array_equal(routed[mid], legacy[mid])
