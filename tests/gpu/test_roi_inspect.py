"""T5 ROI inspector: the export-resolution tile the inspector renders must equal
the corresponding crop of a full export at the same dims/origin -- byte-for-byte
in kinematic mode (the same tile-apron contract the mapset export relies on).

The test name contains ``identical`` so CI's PR-blocking gpu-smoke job
(``pytest -m gpu -k "identical or noop or no_op"``) selects it. Uses the session
``gpu`` fixture, so it skips cleanly without a GL 4.3 context.

The snapshot-A retake texture-leak assertion lives in tests/unit/test_ab_roi.py
(fake-texture level): ``snapshot_preview_color`` rebinds ``ctx.screen``, which a
headless standalone context has no default framebuffer for, so it cannot run
here.
"""

from __future__ import annotations

import numpy as np
import pytest

from gasgiant.engine import Simulation
from gasgiant.export.exporter import TILE, derive_tile, roi_tile_origin
from gasgiant.params.model import PlanetParams, SolverType

pytestmark = pytest.mark.gpu


def _params() -> PlanetParams:
    p = PlanetParams(seed=77)
    p.solver.type = SolverType.KINEMATIC  # byte-exact path
    p.sim.resolution = 512
    p.sim.dev_steps = 20
    return p


def _derive_full(gpu, sim, snap, w: int, h: int) -> np.ndarray:
    """Full (w, h) color map derived from the snapshot with origin=(0,0)."""
    color = gpu.texture2d((w, h), 4, "f4")
    height = gpu.texture2d((w, h), 1, "f4")
    detail = gpu.texture2d((w, h), 1, "f4", linear=True)
    try:
        derive_tile(sim, snap, snap.params, 0, 0, w, h, color, height, detail, None)
        return gpu.read_texture(color).copy()
    finally:
        color.release()
        height.release()
        detail.release()


def _derive_tile_at(gpu, sim, snap, x0: int, y0: int, w: int, h: int) -> np.ndarray:
    """A single TILE-sized tile at (x0, y0), exactly as the ROI inspector does."""
    color = gpu.texture2d((TILE, TILE), 4, "f4")
    height = gpu.texture2d((TILE, TILE), 1, "f4")
    detail = gpu.texture2d((TILE, TILE), 1, "f4", linear=True)
    try:
        derive_tile(sim, snap, snap.params, x0, y0, w, h, color, height, detail, None)
        return gpu.read_texture(color).copy()
    finally:
        color.release()
        height.release()
        detail.release()


def test_roi_tile_identical_to_export_crop(gpu):
    """The inspector tile == the same crop of a full-map derive, byte-for-byte."""
    sim = Simulation(_params(), gpu)
    sim.run_to_completion()
    w, h = 2048, 1024

    snap = sim.create_snapshot()
    try:
        full = _derive_full(gpu, sim, snap, w, h)
        for cx, cy in ((0.5, 0.5), (0.75, 0.25), (1.0, 1.0)):
            x0, y0 = roi_tile_origin(cx, cy, w, h, TILE)
            tile = _derive_tile_at(gpu, sim, snap, x0, y0, w, h)
            tw = min(TILE, w - x0)
            th = min(TILE, h - y0)
            np.testing.assert_array_equal(
                tile[:th, :tw], full[y0 : y0 + th, x0 : x0 + tw],
                err_msg=f"ROI tile at ({x0},{y0}) != full-map crop",
            )
    finally:
        snap.release()
    gpu.make_current()
