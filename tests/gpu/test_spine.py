from __future__ import annotations

import numpy as np
import pytest

from gasgiant.engine import Simulation
from gasgiant.params.model import PlanetParams
from gasgiant.params.presets import load_factory_preset
from gasgiant.validate import validate_arrays

pytestmark = pytest.mark.gpu


@pytest.fixture(scope="module")
def small_maps(gpu):
    params = PlanetParams(seed=1234)
    sim = Simulation(params, gpu)
    return sim.render_maps(width=512)


def test_kernel_compiles_and_runs(small_maps):
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
    params = PlanetParams(seed=42)
    a = Simulation(params, gpu).render_maps(width=256)
    b = Simulation(params, gpu).render_maps(width=256)
    np.testing.assert_array_equal(a["color"], b["color"])
    np.testing.assert_array_equal(a["height"], b["height"])


def test_different_seeds_differ(gpu):
    a = Simulation(PlanetParams(seed=1), gpu).render_maps(width=256)
    b = Simulation(PlanetParams(seed=2), gpu).render_maps(width=256)
    assert not np.array_equal(a["color"], b["color"])


def test_factory_presets_render(gpu):
    for name in ("jupiter_like", "saturn_pale"):
        params = load_factory_preset(name)
        maps = Simulation(params, gpu).render_maps(width=256)
        report = validate_arrays({"color": maps["color"][..., :3]})
        assert report.ok, f"{name}: {report.summary()}"


def test_haze_reduces_contrast(gpu):
    base = PlanetParams(seed=10)
    hazy = base.model_copy(deep=True)
    hazy.appearance.haze_amount = 0.8
    a = Simulation(base, gpu).render_maps(width=256)["color"]
    b = Simulation(hazy, gpu).render_maps(width=256)["color"]
    assert b.std() < a.std()
