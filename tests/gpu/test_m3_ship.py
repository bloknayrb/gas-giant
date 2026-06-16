"""Ship-path: driver lifecycle, off-path no-op, preview==export, seed determinism.

Render comparisons here use a small tolerance (GPU_NOISE_ATOL), NOT byte-equality.
In a clean process the vorticity render is byte-deterministic (verified directly),
but across a long pytest session sharing ONE session-scoped GPU context, many
build/release cycles introduce ~0.004 LSB float nondeterminism in the render. That
noise is orders of magnitude below the coupling signal (on-vs-off maxdiff ~0.69,
~45% of pixels), so a tolerance of 0.01 cleanly separates the two and still catches
a cadence/clamp bug (which diverges far above 0.01). The dedicated BYTE-EXACT no-op
gate for the external hook lives in tests/gpu/test_m3_external_source.py.
"""
from __future__ import annotations

import numpy as np
import pytest

GPU_NOISE_ATOL = 1e-2  # > measured ~0.004 session LSB noise, << ~0.69 coupling signal

from gasgiant.engine.facade import Simulation
from gasgiant.params.model import PlanetParams, SolverType
from gasgiant.params.presets import load_factory_preset

pytestmark = pytest.mark.gpu


def _baro_params(seed: int = 7, dev_steps: int = 40, warmup: int = 600,
                 enabled: bool = True) -> PlanetParams:
    p = load_factory_preset("jupiter_vorticity")
    p = p.model_copy(update={"seed": seed})
    p.sim.resolution = 512
    p.sim.dev_steps = dev_steps
    p.solver.type = SolverType.VORTICITY
    p.solver.baroclinic = p.solver.baroclinic.model_copy(update={
        "enabled": enabled, "gain": 0.5, "warmup_steps": warmup,
        "baro_steps_per_update": 60, "update_every": 16,
    })
    return p


def test_enabled_builds_driver(gpu):
    sim = Simulation(_baro_params(), gpu)
    try:
        assert sim._baro_driver is not None
    finally:
        sim._release_sim()


def test_disabled_has_no_driver(gpu):
    sim = Simulation(_baro_params(enabled=False), gpu)
    try:
        assert sim._baro_driver is None
    finally:
        sim._release_sim()


def _dev_render(p: PlanetParams, gpu) -> np.ndarray:
    sim = Simulation(p, gpu)
    try:
        return sim.render_maps(512)["color"].astype(np.float64)
    finally:
        sim._release_sim()


def test_off_path_matches_plain_vorticity(gpu):
    """baroclinic.enabled=False must develop the same render as plain vorticity
    (no-op within the GPU session noise floor; byte-exact gate is in
    test_m3_external_source.py)."""
    base = _dev_render(_baro_params(enabled=False), gpu)
    p = load_factory_preset("jupiter_vorticity").model_copy(update={"seed": 7})
    p.sim.resolution = 512
    p.sim.dev_steps = 40
    p.solver.type = SolverType.VORTICITY
    plain = _dev_render(p, gpu)
    maxdiff = np.abs(base - plain).max()
    assert maxdiff <= GPU_NOISE_ATOL, f"default-off must not change render (maxdiff={maxdiff})"


def test_preview_equals_export(gpu):
    """Developing via many small tick(2) calls must match one run_to_completion.
    The cadence clamp makes the source-update schedule chunk-size-independent, so
    the two agree to the GPU noise floor; a clamp bug would diverge far above it."""
    p = _baro_params(dev_steps=48)
    export = Simulation(p, gpu)
    try:
        export.run_to_completion(chunk=64)
        export_img = export.render_maps(512)["color"].astype(np.float64)
    finally:
        export._release_sim()

    preview = Simulation(_baro_params(dev_steps=48), gpu)
    try:
        while preview.tick(2):
            pass
        preview_img = preview.render_maps(512)["color"].astype(np.float64)
    finally:
        preview._release_sim()

    maxdiff = np.abs(preview_img - export_img).max()
    assert maxdiff <= GPU_NOISE_ATOL, f"preview chunking must equal export (maxdiff={maxdiff})"


def test_enabled_changes_render(gpu):
    """Coupling on must produce a LARGE, structured change vs off (well above noise)."""
    on = _dev_render(_baro_params(enabled=True), gpu)
    off = _dev_render(_baro_params(enabled=False), gpu)
    maxdiff = np.abs(on - off).max()
    assert maxdiff > 0.05, f"enabled coupling must materially change the render (maxdiff={maxdiff})"
