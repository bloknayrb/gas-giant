from __future__ import annotations

import numpy as np
import pytest

from gasgiant.engine import Simulation
from gasgiant.params.model import PlanetParams, PoleStyle

pytestmark = pytest.mark.gpu


def _params(seed: int = 7, steps: int = 0) -> PlanetParams:
    p = PlanetParams(seed=seed)
    p.sim.resolution = 512
    p.sim.dev_steps = steps
    return p


def test_patch_tracers_finite_after_run(gpu):
    sim = Simulation(_params(steps=0), gpu)
    sim.solver.step(120)
    for dom in sim.solver.domains:
        arr = sim.gpu.read_texture(dom.tracers.cur)
        assert np.isfinite(arr).all(), f"domain {dom.kind} has non-finite tracers"


def test_exchange_keeps_domains_consistent(gpu):
    """The per-step nesting exchange must keep equirect and patch solutions
    slaved in the exchange band (the anti-ghosting requirement)."""
    sim = Simulation(_params(steps=0), gpu)
    sim.solver.step(150)
    rms = sim.solver.exchange_band_rms()
    assert rms < 0.06, f"cross-domain T0 RMS {rms:.4f} in the exchange band"


def test_polar_cyclones_leave_a_signature(gpu):
    """Cyclone-cluster pole must differ from a calm pole in the polar region."""
    base = _params(seed=11, steps=60)
    base.poles.north.style = PoleStyle.CYCLONE_CLUSTER
    calm = base.model_copy(deep=True)
    calm.poles.north.style = PoleStyle.CALM

    a = Simulation(base, gpu)
    a.run_to_completion()
    b = Simulation(calm, gpu)
    b.run_to_completion()
    pa = a.gpu.read_texture(a.solver.north.tracers.cur)[..., 0]
    pb = b.gpu.read_texture(b.solver.north.tracers.cur)[..., 0]
    assert np.abs(pa - pb).mean() > 1e-3, "cyclone cluster left no trace at the pole"


def test_hexagon_jet_swirls_the_patch(gpu):
    p = _params(seed=12, steps=0)
    p.poles.north.style = PoleStyle.POLYGON_JET
    p.poles.north.polygon_sides = 6
    sim = Simulation(p, gpu)
    vel = sim.gpu.read_texture(sim.solver.north.vel_tex)
    sim.solver.step(1)
    vel = sim.gpu.read_texture(sim.solver.north.vel_tex)
    speed = np.hypot(vel[..., 0], vel[..., 1])
    assert speed.max() > 0.05, "polygon jet produced no polar flow"


def test_derive_composites_patches(gpu):
    """The derived map's polar rows must come from the patch sims (no flat
    fallback): a cyclone cluster must be visible in the derived color map."""
    base = _params(seed=13, steps=40)
    base.poles.north.style = PoleStyle.CYCLONE_CLUSTER
    calm = base.model_copy(deep=True)
    calm.poles.north.style = PoleStyle.CALM
    a = Simulation(base, gpu).render_maps(width=512)["color"]
    b = Simulation(calm, gpu).render_maps(width=512)["color"]
    polar_rows = a[:40] - b[:40]  # top rows ~ poleward of 76 deg
    assert np.abs(polar_rows).mean() > 1e-4
