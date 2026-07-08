"""T6 contact sheet, GPU tier: a tiny multi-seed sheet renders a valid grid PNG
through the real facade, and the seed loop reuses ONE Simulation."""

from __future__ import annotations

import numpy as np
import pytest

from gasgiant.engine import Simulation
from gasgiant.export.sheet import run_sheet
from gasgiant.export.writers import decode_image
from gasgiant.params.model import PlanetParams

pytestmark = pytest.mark.gpu


def _params() -> PlanetParams:
    p = PlanetParams(seed=1)
    p.sim.resolution = 256
    p.sim.dev_steps = 6  # tiny: this is a preview grid, not an export
    return p


def test_sheet_renders_valid_grid(gpu, tmp_path):
    out = tmp_path / "contact.png"
    seeds = [1, 2]
    run_sheet(Simulation, _params(), seeds, out, width=64, cols=2, gpu=gpu)
    assert out.is_file()
    grid = decode_image(out, color=True)
    # 2 cells of 64x32 in a 1-row, 2-col grid with 8px default padding.
    assert grid.shape == (1 * 32 + 2 * 8, 2 * 64 + 3 * 8, 3)
    assert np.isfinite(grid).all()
    assert grid.min() >= 0.0 and grid.max() <= 1.0


def test_sheet_reuses_single_simulation(gpu, tmp_path):
    """The no-leak property against the REAL facade: exactly one Simulation is
    constructed for N seeds; it is re-seeded in place and released once."""
    created: list[Simulation] = []
    released: list[Simulation] = []

    class _CountingSim(Simulation):
        def __init__(self, params, gpu=None):
            super().__init__(params, gpu)
            created.append(self)

        def release(self):
            released.append(self)
            super().release()

    seeds = [10, 11, 12]
    run_sheet(_CountingSim, _params(), seeds, tmp_path / "s.png", width=64, gpu=gpu)
    assert len(created) == 1  # ONE facade for all seeds
    assert len(released) == 1  # torn down exactly once
    assert created[0] is released[0]
