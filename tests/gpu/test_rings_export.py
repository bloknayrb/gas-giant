"""T16 Saturn rings export (GPU tier).

Rings are a SEPARATE exported map, so the color/height/emission path is
untouched. Two contracts:
  (a) default-off: no rings.enabled => no rings.exr, file-set unchanged;
  (b) rings on => rings.exr written + a `rings` manifest entry + physical
      ring_*_km, and the color/height output is byte-identical to the rings-off
      export (rings never touch render_maps).
"""

from __future__ import annotations

import numpy as np
import pytest

from gasgiant.engine import Simulation
from gasgiant.export.exporter import run_export
from gasgiant.params.model import PlanetParams, SolverType

pytestmark = pytest.mark.gpu


def _params() -> PlanetParams:
    p = PlanetParams(seed=77)
    p.solver.type = SolverType.KINEMATIC  # byte-exact color/height path
    p.sim.resolution = 512
    p.sim.dev_steps = 15
    p.export.width = 2048
    return p


def test_default_export_writes_no_rings_identical(gpu, tmp_path):
    """Default-off: no rings.enabled => no rings.exr; the file-set is exactly the
    default color+height+manifest."""
    from gasgiant.export.manifest import read_manifest

    sim = Simulation(_params(), gpu)
    out = tmp_path / "mapset"
    run_export(sim, out)
    assert not (out / "rings.exr").exists()
    assert "rings" not in read_manifest(out)["maps"]
    assert {p.name for p in out.iterdir()} == {"color.png", "height.exr", "mapset.json"}


def test_rings_export_round_trip_and_color_unchanged(gpu, tmp_path):
    """rings on => rings.exr + manifest entry + physical extent, and the
    color/height maps are byte-identical to the rings-off export."""
    from gasgiant.export.manifest import read_manifest
    from gasgiant.export.writers import read_exr_gray, read_exr_rgba, read_png16

    off = tmp_path / "off"
    run_export(Simulation(_params(), gpu), off)

    p = _params()
    p.rings.enabled = True
    on = tmp_path / "on"
    run_export(Simulation(p, gpu), on)

    assert (on / "rings.exr").is_file()
    entry = read_manifest(on)["maps"]["rings"]
    assert entry["format"] == "exr32f"
    assert entry["colorspace"] == "non-color"
    assert entry["channels"] == 4
    assert entry["convention"] == "radial_inner_to_outer_alpha_coverage"
    phys = read_manifest(on)["physical"]
    assert phys["ring_inner_km"] == p.physical.ring_inner_km
    assert phys["ring_outer_km"] == p.physical.ring_outer_km

    rings = read_exr_rgba(on / "rings.exr")
    assert rings.shape == (2048, 64, 4)
    assert np.isfinite(rings).all()
    assert rings.min() >= 0.0 and rings.max() <= 1.0

    # render_maps untouched: color + height byte-identical with rings on vs off.
    np.testing.assert_array_equal(read_png16(off / "color.png"), read_png16(on / "color.png"))
    np.testing.assert_array_equal(
        read_exr_gray(off / "height.exr"), read_exr_gray(on / "height.exr")
    )
