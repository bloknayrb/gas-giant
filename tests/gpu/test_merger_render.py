"""GPU: mergers render correctly — single coalesced oval, visible debris
collar that fades, validators green on a merger-heavy frame."""

from __future__ import annotations

import numpy as np
import pytest

from gasgiant.engine import Simulation
from gasgiant.params.model import PlanetParams
from gasgiant.sim.vortices import (
    KIND_DEBRIS,
    KIND_OVAL,
    Vortex,
    zonal_rate,
)
from gasgiant.validate.seams import validate_arrays

pytestmark = pytest.mark.gpu


def _quiet_params(seed: int = 21) -> PlanetParams:
    """No storms of any kind: the test installs its own registry."""
    p = PlanetParams(seed=seed)
    p.sim.resolution = 512
    p.sim.dev_steps = 600
    p.storms.hero_count = 0
    p.storms.oval_density = 0.0
    p.storms.barge_density = 0.0
    p.storms.pearls_count = 0
    p.storms.merge_rate = 0.7
    return p


def _install_converging_pair(sim: Simulation) -> tuple[Vortex, Vortex]:
    """Engineer a same-sign pair just inside capture on the REAL profiles,
    placed on the converging side so the strict gate passes."""
    profiles = sim.profiles
    lat_a = 0.30
    lat_b = 0.34
    rates = zonal_rate(profiles, np.array([lat_a, lat_b]))
    drate = float(rates[1] - rates[0])
    assert drate != 0.0
    r = 0.035
    capture = 1.5 * sim.params.storms.merge_rate * (2 * r)
    dlon_in = 0.8 * float(
        np.sqrt(max(capture**2 - (lat_b - lat_a) ** 2, 0.0)) / np.cos(lat_a)
    )
    gap = -np.sign(drate) * dlon_in
    a = Vortex(lat_a, 0.0, r, 0.014, KIND_OVAL, tint=0.1, brightness=0.22)
    b = Vortex(lat_b, float(gap), r, 0.012, KIND_OVAL, tint=0.1, brightness=0.22)
    sim.solver.vortices.vortices = [a, b]
    return a, b


def _t0_region(sim: Simulation, lat: float, lon: float, half: float) -> np.ndarray:
    """T0 tracer patch around (lat, lon)."""
    t = sim.tracers.read_current()[..., 0]
    h, w = t.shape
    y = int(round((np.pi / 2 - lat) / np.pi * h))
    x = int(round((lon + np.pi) / (2 * np.pi) * w))
    py = max(int(half / np.pi * h), 2)
    px = max(int(half / (2 * np.pi) * w), 2)
    ys = slice(max(y - py, 0), min(y + py, h))
    xs = slice(max(x - px, 0), min(x + px, w))
    return t[ys, xs]


def test_merged_pair_renders_as_single_bounded_oval(gpu):
    p = _quiet_params()
    p.storms.merge_debris = 0.0  # the (deliberately bright) collar would
    sim = Simulation(p, gpu)     # dominate this stamp-stacking comparison
    _install_converging_pair(sim)
    sim.solver.step(120)  # merge at step ~1, then relax toward the new stamp

    live = [v for v in sim.solver.vortices.vortices if abs(v.strength) > 1e-6]
    assert len(live) == 1  # coalesced
    prod = live[0]

    # An identical UN-merged run stacks both stamps; the merged peak must not
    # exceed the superposed one (the anti-"ever-brightening" guarantee).
    p2 = _quiet_params()
    p2.storms.merge_rate = 0.0
    sim2 = Simulation(p2, gpu)
    _install_converging_pair(sim2)
    sim2.solver.step(120)

    merged_peak = _t0_region(sim, prod.lat, prod.lon, 0.25).max()
    stacked_peak = _t0_region(sim2, prod.lat, prod.lon, 0.25).max()
    assert merged_peak <= stacked_peak + 1e-3

    maps = sim.render_maps(512)
    for name, arr in maps.items():
        assert np.isfinite(arr).all(), name
    validate_arrays({"color": maps["color"][..., :3], "height": maps["height"]})


def test_debris_collar_visible_then_fades(gpu):
    """Paired runs differing ONLY in merge_debris: the collar's tracer
    imprint must be clearly present mid-ttl and gone (sim reconverged with
    the debris-free run) after expiry + relax decay."""
    with_debris = Simulation(_quiet_params(seed=33), gpu)
    p_no = _quiet_params(seed=33)
    p_no.storms.merge_debris = 0.0
    without = Simulation(p_no, gpu)
    _install_converging_pair(with_debris)
    _install_converging_pair(without)

    with_debris.solver.step(120)  # mid-ttl: collar stamped and relaxed in
    without.solver.step(120)
    assert any(v.kind == KIND_DEBRIS for v in with_debris.solver.vortices.vortices)
    t0 = with_debris.tracers.read_current()[..., 0]
    t0_ref = without.tracers.read_current()[..., 0]
    mid = float(np.abs(t0 - t0_ref).max())
    assert mid > 0.05  # the collar reads in the tracers

    with_debris.solver.step(760)  # ttl 250 long gone + relax decay
    without.solver.step(760)
    assert all(v.kind != KIND_DEBRIS for v in with_debris.solver.vortices.vortices)
    t0 = with_debris.tracers.read_current()[..., 0]
    t0_ref = without.tracers.read_current()[..., 0]
    assert float(np.abs(t0 - t0_ref).max()) < 0.02  # no permanent artifact
