"""Ship-path: driver lifecycle, off byte-identity, preview==export, seed determinism."""
from __future__ import annotations

import pytest

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


def _dev_render_bytes(p: PlanetParams, gpu) -> bytes:
    sim = Simulation(p, gpu)
    try:
        return sim.render_maps(512)["color"].tobytes()
    finally:
        sim._release_sim()


def test_off_path_byte_identical(gpu):
    """baroclinic.enabled=False renders byte-identically to plain vorticity."""
    base = _dev_render_bytes(_baro_params(enabled=False), gpu)
    p = load_factory_preset("jupiter_vorticity").model_copy(update={"seed": 7})
    p.sim.resolution = 512
    p.sim.dev_steps = 40
    p.solver.type = SolverType.VORTICITY
    plain = _dev_render_bytes(p, gpu)
    assert base == plain, "default-off baroclinic must not change the render"


def test_preview_equals_export(gpu):
    """Developing via many small tick(2) calls must match one run_to_completion."""
    p = _baro_params(dev_steps=48)
    export = Simulation(p, gpu)
    try:
        export.run_to_completion(chunk=64)
        export_bytes = export.render_maps(512)["color"].tobytes()
    finally:
        export._release_sim()

    preview = Simulation(_baro_params(dev_steps=48), gpu)
    try:
        while preview.tick(2):
            pass
        preview_bytes = preview.render_maps(512)["color"].tobytes()
    finally:
        preview._release_sim()

    assert preview_bytes == export_bytes, "preview chunking must equal export"


def test_enabled_changes_render(gpu):
    on = _dev_render_bytes(_baro_params(enabled=True), gpu)
    off = _dev_render_bytes(_baro_params(enabled=False), gpu)
    assert on != off, "enabled coupling must change the render"
