"""Latitude-anchored palette rows reach the derived color map."""

from __future__ import annotations

import numpy as np
import pytest

from gasgiant.engine import Simulation
from gasgiant.params.model import GradientStop, PaletteRow, PlanetParams, SolverType

pytestmark = pytest.mark.gpu


def _flat_stops(color):
    return [GradientStop(pos=0.0, color=color), GradientStop(pos=1.0, color=color)]


def _quick_params(**appearance) -> PlanetParams:
    p = PlanetParams(seed=4)
    p.sim.resolution = 512
    p.sim.dev_steps = 0
    for key, value in appearance.items():
        setattr(p.appearance, key, value)
    return p


def test_two_rows_color_by_hemisphere(gpu):
    p = _quick_params(
        palette_rows=[
            PaletteRow(latitude=-50.0, stops=_flat_stops((0.8, 0.2, 0.1))),  # south red
            PaletteRow(latitude=50.0, stops=_flat_stops((0.1, 0.2, 0.8))),   # north blue
        ]
    )
    sim = Simulation(p, gpu)
    color = sim.render_maps(256)["color"]
    h = color.shape[0]
    north = color[: h // 4].reshape(-1, 4)
    south = color[-h // 4 :].reshape(-1, 4)
    assert north[:, 2].mean() > north[:, 0].mean()  # blue dominates red up north
    assert south[:, 0].mean() > south[:, 2].mean()  # red dominates blue down south


def test_single_row_equals_duplicated_rows(gpu):
    """The latitude axis is pure plumbing: one row and the same stops
    duplicated at +-60 deg bake to the same LUT and render identically."""
    stops = [
        GradientStop(pos=0.0, color=(0.36, 0.27, 0.21)),
        GradientStop(pos=1.0, color=(0.93, 0.89, 0.81)),
    ]
    single = _quick_params(palette_rows=[PaletteRow(latitude=0.0, stops=stops)])
    doubled = _quick_params(
        palette_rows=[
            PaletteRow(latitude=-60.0, stops=stops),
            PaletteRow(latitude=60.0, stops=stops),
        ]
    )
    color_a = Simulation(single, gpu).render_maps(256)["color"]
    color_b = Simulation(doubled, gpu).render_maps(256)["color"]
    np.testing.assert_array_equal(color_a, color_b)


def test_palette_refreshes_on_restart_tier_update(gpu):
    """Regression: update_params must refresh the palette LUT even when a
    RESTART/VELOCITY tier ALSO changed. A preset load edits many tiers at once;
    the mutually-exclusive branches used to refresh the palette only on the pure
    POST path, so loading the rust gas_giant_warm over a paler preset (which also
    flips solver.type = RESTART) kept the stale colors and toggling the solver
    never recolored."""
    red = _quick_params(
        palette_rows=[PaletteRow(latitude=0.0, stops=_flat_stops((0.85, 0.15, 0.10)))]
    )
    red.solver.type = SolverType.KINEMATIC
    sim = Simulation(red, gpu)
    c_red = sim.render_maps(256)["color"].reshape(-1, 4)
    assert c_red[:, 0].mean() > c_red[:, 2].mean()  # starts red

    # Flip the palette to blue AND the solver (RESTART) in a single update.
    blue = _quick_params(
        palette_rows=[PaletteRow(latitude=0.0, stops=_flat_stops((0.10, 0.15, 0.85)))]
    )
    blue.solver.type = SolverType.VORTICITY
    sim.update_params(blue)
    c_blue = sim.render_maps(256)["color"].reshape(-1, 4)
    assert c_blue[:, 2].mean() > c_blue[:, 0].mean()  # blue now -- palette refreshed
