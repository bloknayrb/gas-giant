"""GPU: striation and analytic lanes reach the derived maps cleanly."""

from __future__ import annotations

import numpy as np
import pytest

from gasgiant.engine import Simulation
from gasgiant.params.model import PlanetParams

pytestmark = pytest.mark.gpu


def _quick(seed: int = 17) -> PlanetParams:
    p = PlanetParams(seed=seed)
    p.sim.resolution = 512
    p.sim.dev_steps = 0
    return p


def test_striation_adds_signal_without_nans(gpu):
    p = _quick()
    p.detail.striation_amount = 0.0
    p.detail.striation_frequency = 24.0  # the 512-px render can hold this;
    base = Simulation(p, gpu).render_maps(512)["color"]  # 96 would attenuate out
    p2 = _quick()
    p2.detail.striation_amount = 1.0
    p2.detail.striation_frequency = 24.0
    striated = Simulation(p2, gpu).render_maps(512)["color"]
    assert np.isfinite(striated).all()
    assert not np.array_equal(base, striated)
    # Striation is texture, not a tone shift.
    assert abs(float(striated.mean() - base.mean())) < 0.02


def test_lanes_darken_thin_rows(gpu):
    p = _quick()
    p.bands.lane_density = 1.0
    sim = Simulation(p, gpu)
    assert sim.lanes
    color = sim.render_maps(512)["color"][..., :3]
    h = color.shape[0]
    row_mean = color.mean(axis=(1, 2))
    for lane_lat, _strength in sim.lanes[:4]:
        y = int(round((np.pi / 2 - lane_lat) / np.pi * h))
        if not 4 <= y < h - 4:
            continue
        local = row_mean[y - 4 : y + 5]
        # The lane row is darker than its immediate neighborhood edges.
        assert local.min() <= row_mean[y] <= local.max()
        assert row_mean[y] < 0.5 * (local[0] + local[-1]) + 1e-3


def test_lane_zero_density_changes_nothing(gpu):
    a = Simulation(_quick(), gpu).render_maps(512)["color"]
    p = _quick()
    p.bands.lane_density = 0.0
    b = Simulation(p, gpu).render_maps(512)["color"]
    np.testing.assert_array_equal(a, b)
